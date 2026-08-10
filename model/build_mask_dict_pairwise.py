#!/usr/bin/env python3
"""
Pre-build PAIRWISE alignment masks (CPU only, no GPU).

Differences from build_mask_dict.py:
  The old version ran one three-way MSA (common columns of ref/pdb/predict) and
  output [mask1,mask2,mask3].
  This version splits the alignment into [two independent pairwise alignments],
  both anchored to the "filtered structure sequence cur_seq":
      ref   <-> cur_seq   (for label scattering)
      fasta <-> cur_seq   (for PLM embedding scattering)
  This way gaps in fasta do not hurt label coverage (and vice versa), and both
  sides have the same number of aligned residues, so they can be scattered
  directly via canvas[mask_struct] = src[mask_src].

Output format (new):
  mask_dict[ukey] = {
      'ref':   (mask_ref,   mask_struct_r),   # over ref_seq / cur_seq respectively
      'fasta': (mask_fasta, mask_struct_f),   # over fasta_seq / cur_seq respectively
  }
  Plus a meta key ('__meta__',): {'format':'pairwise', 'preprocess': <CA_BB...>}
  data_handler / main take the new path when they read a dict value; a list
  value (old format) takes the old path.

Phases are exactly the same as build_mask_dict.py:
  manifest  -> {split}_manifest.pkl + {split}_unique_triples.pkl (HDF5 scan + dedupe)
  msa       -> pairwise alignment shards (--array_id/--num_tasks)
  assemble  -> {out_dir}/mask_dict_{split}.pkl (new format)
  all       -> msa + assemble on one machine
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Meta key: use a tuple instead of a str, so it can never collide with a real ukey (str)
META_KEY = ('__meta__',)


def _bootstrap_import_path():
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)


def print_run_context():
    print("[build_mask_dict_pairwise] __file__ =", os.path.abspath(__file__))
    print("[build_mask_dict_pairwise] script dir (model/) =", _SCRIPT_DIR)
    print("[build_mask_dict_pairwise] cwd (getcwd) =", os.getcwd())
    print("[build_mask_dict_pairwise] PYTHONPATH =", os.environ.get("PYTHONPATH", "(unset)"))


def _load_module_from_file(module_name, filepath):
    if not os.path.isfile(filepath):
        raise ImportError(
            f"Cannot load {module_name}: file not visible in this runtime: {filepath}"
        )
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_bootstrap_import_path()
print_run_context()

import h5py
import numpy as np
import torch as pt
from Bio import Align
from tqdm import tqdm

_data_handler = _load_module_from_file(
    "data_handler", os.path.join(_SCRIPT_DIR, "data_handler.py")
)
Dataset = _data_handler.Dataset
load_seq = _data_handler.load_seq
reduce_to_dense = _data_handler.reduce_to_dense
load_cluster_ref_seq_map = _data_handler.load_cluster_ref_seq_map
build_cluster_to_ref_seq_map = _data_handler.build_cluster_to_ref_seq_map
resolve_ref_seq_via_cluster = _data_handler.resolve_ref_seq_via_cluster
resolve_project_path = _data_handler.resolve_project_path
preprocess = _data_handler.preprocess

_src_dataset = _load_module_from_file(
    "src.dataset", os.path.join(_SCRIPT_DIR, "src", "dataset.py")
)
load_sparse_mask = _src_dataset.load_sparse_mask

DEFAULT_CLUSTER_REF_SEQ_PATH = resolve_project_path(
    '../colabel/cluster_labels_summary_merged.pkl'
)
DEFAULT_TRAIN_M = resolve_project_path(
    '../processing/data/datasets/cluster_30_80_benchclean_train_m.pkl'
)
DEFAULT_VALID_M = resolve_project_path(
    '../processing/data/datasets/cluster_30_80_benchclean_val_m.pkl'
)
DEFAULT_CHAIN_MAPPING = resolve_project_path('../colabel/colabel_mask_all.pkl')
DEFAULT_REF_DB = resolve_project_path('./all_seq_dict.pkl')
DEFAULT_CONFIG = resolve_project_path('./config.py')


def intervals_to_mask(intervals, length):
    mask = np.zeros(length, dtype=bool)
    for s, e in intervals:
        mask[s:e] = True
    return mask


def _pairwise_masks(query_seq, struct_seq):
    """Global pairwise alignment query_seq <-> struct_seq (filtered structure sequence).

    Returns (mask_query, mask_struct): two bool arrays of length len(query_seq)/len(struct_seq),
    True on aligned (gap-free matched) columns. Both sides have the same number of True
    entries in one-to-one (monotonic) correspondence, so they can be used directly for
    canvas[mask_struct] = src[mask_query] scattering.
    """
    if query_seq == struct_seq:
        return (np.ones(len(query_seq), dtype=bool),
                np.ones(len(struct_seq), dtype=bool))
    if len(query_seq) == 0 or len(struct_seq) == 0:
        return (np.zeros(len(query_seq), dtype=bool),
                np.zeros(len(struct_seq), dtype=bool))
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aln = aligner.align(query_seq, struct_seq)[0]
    blocks_q, blocks_s = aln.aligned            # each is [(start,end), ...] matched blocks
    mq = intervals_to_mask(blocks_q.tolist(), len(query_seq))
    ms = intervals_to_mask(blocks_s.tolist(), len(struct_seq))
    # matched blocks guarantee equal lengths on both sides; truncate as a fallback for extreme cases
    if int(mq.sum()) != int(ms.sum()):
        k = min(int(mq.sum()), int(ms.sum()))
        qi = np.flatnonzero(mq)[:k]
        si = np.flatnonzero(ms)[:k]
        mq = np.zeros_like(mq); mq[qi] = True
        ms = np.zeros_like(ms); ms[si] = True
    return mq, ms


def setup_mask_dataset(config_data, sel_m):
    mask_config = dict(config_data)
    mask_config['colabel_path'] = None
    mask_config.pop('cluster_labels_path', None)
    mask_config.pop('label_noise_std', None)
    mask_config.pop('label_binarize_threshold', None)
    mask_config.pop('mask_dict_path', None)     # the mask-building stage itself does not need masks
    mask_config['mask_only'] = True
    dataset = Dataset(mask_config)
    with open(sel_m, 'rb') as f:
        m = pickle.load(f)
    dataset.update_mask(m)
    dataset.set_types(mask_config['l_types'], mask_config['r_types'])
    return dataset


def load_config(config_path):
    path = resolve_project_path(config_path)
    spec = importlib.util.spec_from_file_location('custom_config', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['custom_config'] = mod
    spec.loader.exec_module(mod)
    config_data = mod.config_data
    df = config_data.get('dataset_filepath')
    if df and not os.path.isabs(df):
        config_data['dataset_filepath'] = resolve_project_path(df)
    return config_data


def paths_for_split(split, work_dir):
    return {
        'manifest': os.path.join(work_dir, f'{split}_manifest.pkl'),
        'unique': os.path.join(work_dir, f'{split}_unique_triples.pkl'),
        'msa_shard': lambda task: os.path.join(work_dir, f'{split}_pw_shard_{task:04d}.pkl'),
        'mask_out': lambda out_dir: os.path.join(out_dir, f'mask_dict_{split}.pkl'),
    }


def _require_manifest_files(split, work_dir):
    p = paths_for_split(split, work_dir)
    missing = [label for label, path in (
        ('manifest', p['manifest']),
        ('unique_triples', p['unique']),
    ) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"[{split}] manifest artifacts missing under work_dir={work_dir!r}\n"
            f"  missing: {', '.join(missing)}\n"
            f"  Run: python3 build_mask_dict_pairwise.py --phase manifest --split {split} "
            f"--work_dir {work_dir!r}"
        )


def _ftrs_from_config(config_data):
    return [
        fn for fn, ff in zip(['qe', 'qr', 'qn'], config_data.get('features_flags', (True, False, False)))
        if ff
    ]


def _manifest_row(ukey, q_raw, M, mapping, cluster_ref_seq_map, cluster_to_ref_seq, ref_db):
    M_dense = reduce_to_dense(M)
    if q_raw.shape[1] == 59:
        q = reduce_to_dense(q_raw[:, 30:])
    else:
        q = reduce_to_dense(q_raw)
    ref_seq = resolve_ref_seq_via_cluster(
        ukey, mapping, cluster_ref_seq_map, cluster_to_ref_seq
    )
    chainname = ukey.split('/')[1].upper() + '_' + ukey.split('/')[2].split(':')[0]
    fasta_seq = ref_db[chainname]
    cur_seq = load_seq(M_dense, q)
    cache_key = (ref_seq, cur_seq, fasta_seq)
    return {
        'ukey': ukey,
        'cache_key': cache_key,
        'len_fasta': len(fasta_seq),
        'len_pdb': len(cur_seq),
        'len_ref': len(ref_seq),
    }


_MANIFEST_CTX = None


def _manifest_worker_init(ctx):
    global _MANIFEST_CTX
    _MANIFEST_CTX = dict(ctx)
    _MANIFEST_CTX['hf'] = h5py.File(ctx['dataset_filepath'], 'r')


def _manifest_load_qM(key):
    hgrp = _MANIFEST_CTX['hf']['data/structures/' + key]
    ern_shape = {
        'qe': hgrp.attrs['qe_shape'],
        'qn': hgrp.attrs['qn_shape'],
        'qr': hgrp.attrs['qr_shape'],
    }
    X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32))
    M = load_sparse_mask(hgrp, 'M')
    if _MANIFEST_CTX['preprocess']:
        # preprocess now returns a 5-tuple (..., residue_indices); only q,M are needed here
        out = preprocess(
            X, M, hgrp['qe'], hgrp['qr'], hgrp['qn'],
            _MANIFEST_CTX['preprocess'], _MANIFEST_CTX['ftrs'], ern_shape,
        )
        X, _ids_topk, M, q = out[0], out[1], out[2], out[3]
    else:
        q_l = [load_sparse_mask(hgrp, fn) for fn in _MANIFEST_CTX['ftrs']]
        q = pt.cat(q_l, dim=1)
    return q, M


def _manifest_collect_idx(idx):
    ukey = _MANIFEST_CTX['ukeys'][idx]
    q_raw, M = _manifest_load_qM(ukey)
    return _manifest_row(
        ukey, q_raw, M,
        _MANIFEST_CTX['mapping'],
        _MANIFEST_CTX['cluster_ref_seq_map'],
        _MANIFEST_CTX['cluster_to_ref_seq'],
        _MANIFEST_CTX['ref_db'],
    )


def run_manifest(split, config_data, sel_m, mapping, cluster_ref_seq_map,
                 cluster_to_ref_seq, ref_db, work_dir, collect_workers,
                 collect_backend='process', skip_if_exists=False):
    os.makedirs(work_dir, exist_ok=True)
    p = paths_for_split(split, work_dir)
    if skip_if_exists and os.path.isfile(p['manifest']) and os.path.isfile(p['unique']):
        print(f'[{split}] manifest: skip (already exists in {work_dir})')
        return

    t0 = time.time()
    dataset = setup_mask_dataset(config_data, sel_m)
    n = len(dataset)
    print(f'[{split}] manifest: {n} structures, collect_workers={collect_workers}, backend={collect_backend}')

    ctx = {
        'dataset_filepath': config_data['dataset_filepath'],
        'preprocess': config_data.get('preprocess'),
        'ftrs': _ftrs_from_config(config_data),
        'ukeys': dataset.ukeys,
        'mapping': mapping,
        'cluster_ref_seq_map': cluster_ref_seq_map,
        'cluster_to_ref_seq': cluster_to_ref_seq,
        'ref_db': ref_db,
    }
    rows = [None] * n
    if collect_workers <= 1:
        _manifest_worker_init(ctx)
        try:
            for i in tqdm(range(n), desc=f'{split}/collect'):
                rows[i] = _manifest_collect_idx(i)
        finally:
            if _MANIFEST_CTX and _MANIFEST_CTX.get('hf'):
                _MANIFEST_CTX['hf'].close()
    elif collect_backend == 'thread':
        _manifest_worker_init(ctx)
        try:
            with ThreadPoolExecutor(max_workers=collect_workers) as ex:
                futs = {ex.submit(_manifest_collect_idx, i): i for i in range(n)}
                for fut in tqdm(as_completed(futs), total=n, desc=f'{split}/collect'):
                    rows[futs[fut]] = fut.result()
        finally:
            if _MANIFEST_CTX and _MANIFEST_CTX.get('hf'):
                _MANIFEST_CTX['hf'].close()
    else:
        with ProcessPoolExecutor(
            max_workers=collect_workers,
            initializer=_manifest_worker_init,
            initargs=(ctx,),
        ) as ex:
            futs = {ex.submit(_manifest_collect_idx, i): i for i in range(n)}
            for fut in tqdm(as_completed(futs), total=n, desc=f'{split}/collect'):
                rows[futs[fut]] = fut.result()

    print(f'[{split}] manifest collect done in {time.time() - t0:.1f}s')

    unique = {}
    for r in rows:
        ck = r['cache_key']
        if ck not in unique:
            unique[ck] = {'ref_seq': ck[0], 'pdb_seq': ck[1], 'predict_seq': ck[2]}

    with open(p['manifest'], 'wb') as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(p['unique'], 'wb') as f:
        pickle.dump(list(unique.items()), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[{split}] manifest rows={len(rows)}, unique triples={len(unique)}')


def ensure_manifest_cache(
    split, config_data, sel_m, mapping, cluster_ref_seq_map, cluster_to_ref_seq,
    ref_db, work_dir, collect_workers, collect_backend='process',
    skip_if_exists=False, require_manifest=False,
):
    p = paths_for_split(split, work_dir)
    if os.path.isfile(p['manifest']) and os.path.isfile(p['unique']):
        return
    if require_manifest:
        _require_manifest_files(split, work_dir)
    os.makedirs(work_dir, exist_ok=True)
    lock_path = os.path.join(work_dir, f'.{split}_manifest.lock')
    with open(lock_path, 'w', encoding='utf-8') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if os.path.isfile(p['manifest']) and os.path.isfile(p['unique']):
            return
        print(f'[{split}] building index (HDF5 scan + dedupe) ...')
        run_manifest(
            split, config_data, sel_m, mapping, cluster_ref_seq_map,
            cluster_to_ref_seq, ref_db, work_dir, collect_workers,
            collect_backend=collect_backend, skip_if_exists=skip_if_exists,
        )


def _align_one(cache_key, seqs):
    """Run two pairwise alignments for one deduplicated (ref, cur, fasta) triple."""
    ref_seq, cur_seq, fasta_seq = cache_key
    m_ref, m_struct_r = _pairwise_masks(ref_seq, cur_seq)      # ref  <-> cur
    m_fasta, m_struct_f = _pairwise_masks(fasta_seq, cur_seq)  # fasta<-> cur
    return cache_key, {
        'ref': (m_ref, m_struct_r),
        'fasta': (m_fasta, m_struct_f),
    }


def run_msa(split, work_dir, workers, array_id, num_tasks, manifest_kw):
    ensure_manifest_cache(split, **manifest_kw)
    p = paths_for_split(split, work_dir)
    with open(p['unique'], 'rb') as f:
        items = pickle.load(f)

    if num_tasks > 1:
        items = [it for i, it in enumerate(items) if i % num_tasks == array_id]
        out_path = p['msa_shard'](array_id)
        print(f'[{split}] pairwise array {array_id}/{num_tasks}: {len(items)} unique triples')
    else:
        out_path = p['msa_shard'](0)
        print(f'[{split}] pairwise: {len(items)} unique triples, workers={workers}')

    if not items:
        with open(out_path, 'wb') as f:
            pickle.dump({}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    cache = {}
    if workers <= 1:
        for cache_key, seqs in tqdm(items, desc=f'{split}/pairwise'):
            ck, pos = _align_one(cache_key, seqs)
            cache[ck] = pos
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_align_one, ck, seqs) for ck, seqs in items]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f'{split}/pairwise'):
                ck, pos = fut.result()
                cache[ck] = pos

    with open(out_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  wrote {out_path} ({len(cache)} entries)')


def _load_shards(split, work_dir):
    prefix = f'{split}_pw_shard_'
    merged = {}
    for name in sorted(os.listdir(work_dir)):
        if not name.startswith(prefix) or not name.endswith('.pkl'):
            continue
        with open(os.path.join(work_dir, name), 'rb') as f:
            shard = pickle.load(f)
        merged.update(shard)
        print(f'  merged {name}: {len(shard)}')
    return merged


def run_assemble(split, work_dir, out_dir, preprocess_tag):
    _require_manifest_files(split, work_dir)
    os.makedirs(out_dir, exist_ok=True)
    p = paths_for_split(split, work_dir)
    with open(p['manifest'], 'rb') as f:
        rows = pickle.load(f)
    align_cache = _load_shards(split, work_dir)

    mask_dict = {META_KEY: {'format': 'pairwise', 'preprocess': preprocess_tag}}
    missing = 0
    for r in tqdm(rows, desc=f'{split}/assemble'):
        pos = align_cache.get(r['cache_key'])
        if pos is None:
            missing += 1
            continue
        mask_dict[r['ukey']] = {
            'ref': (pos['ref'][0], pos['ref'][1]),
            'fasta': (pos['fasta'][0], pos['fasta'][1]),
        }

    out_path = p['mask_out'](out_dir)
    with open(out_path, 'wb') as f:
        pickle.dump(mask_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_real = len(mask_dict) - 1  # exclude META_KEY
    print(f'[{split}] mask_dict(pairwise) size={n_real}, missing align={missing}')
    print(f'  wrote {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Pre-build PAIRWISE alignment masks (CPU).')
    parser.add_argument('--phase', choices=['manifest', 'msa', 'assemble', 'all'], required=True)
    parser.add_argument('--split', choices=['train', 'valid'], action='append', default=None)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--train_m', default=DEFAULT_TRAIN_M)
    parser.add_argument('--valid_m', default=DEFAULT_VALID_M)
    parser.add_argument('--chain_mapping', default=DEFAULT_CHAIN_MAPPING)
    parser.add_argument('--ref_db', default=DEFAULT_REF_DB)
    parser.add_argument('--cluster_ref_seq_path', default=DEFAULT_CLUSTER_REF_SEQ_PATH)
    parser.add_argument('--work_dir', default=resolve_project_path('./mask_build_cache_pw'))
    parser.add_argument('--out_dir', default='./esm_newsplit_90sampler')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument('--collect_workers', type=int, default=None)
    parser.add_argument('--collect_backend', choices=['process', 'thread', 'legacy'], default='process')
    parser.add_argument('--skip_manifest_if_exists', action='store_true')
    parser.add_argument('--require_manifest', action='store_true')
    parser.add_argument('--array_id', type=int, default=0)
    parser.add_argument('--num_tasks', type=int, default=1)
    args = parser.parse_args()

    splits = args.split or (['train', 'valid'] if args.phase == 'all' else [])
    if not splits:
        parser.error('--split required for phase manifest/msa/assemble')

    args.config = resolve_project_path(args.config)
    args.train_m = resolve_project_path(args.train_m)
    args.valid_m = resolve_project_path(args.valid_m)
    args.chain_mapping = resolve_project_path(args.chain_mapping)
    args.ref_db = resolve_project_path(args.ref_db)
    args.cluster_ref_seq_path = resolve_project_path(args.cluster_ref_seq_path)
    args.work_dir = resolve_project_path(args.work_dir)
    args.out_dir = resolve_project_path(args.out_dir)

    print("[build_mask_dict_pairwise] work_dir =", args.work_dir)
    print("[build_mask_dict_pairwise] out_dir =", args.out_dir)
    print("[build_mask_dict_pairwise] phase =", args.phase, "split =", splits)

    config_data = load_config(args.config)
    preprocess_tag = config_data.get('preprocess')
    with open(args.ref_db, 'rb') as f:
        ref_db = pickle.load(f)
    with open(args.chain_mapping, 'rb') as f:
        mapping = pickle.load(f)
    cluster_ref_seq_map = load_cluster_ref_seq_map(args.cluster_ref_seq_path)
    cluster_to_ref_seq = build_cluster_to_ref_seq_map(mapping, cluster_ref_seq_map)

    collect_workers = args.collect_workers
    if collect_workers is None:
        collect_workers = max(1, min(32, (os.cpu_count() or 8) - 1))

    sel_for = {'train': args.train_m, 'valid': args.valid_m}

    for split in splits:
        sel_m = sel_for[split]
        manifest_kw = dict(
            config_data=config_data, sel_m=sel_m, mapping=mapping,
            cluster_ref_seq_map=cluster_ref_seq_map, cluster_to_ref_seq=cluster_to_ref_seq,
            ref_db=ref_db, work_dir=args.work_dir, collect_workers=collect_workers,
            collect_backend=args.collect_backend, skip_if_exists=args.skip_manifest_if_exists,
            require_manifest=args.require_manifest,
        )
        if args.phase == 'manifest':
            ensure_manifest_cache(split, **manifest_kw)
        elif args.phase in ('msa', 'all'):
            run_msa(split, args.work_dir, args.workers, args.array_id, args.num_tasks, manifest_kw)
        if args.phase in ('assemble', 'all'):
            if args.num_tasks > 1 and args.phase == 'all':
                print('Skip assemble in array worker; run assemble after array completes.')
            else:
                run_assemble(split, args.work_dir, args.out_dir, preprocess_tag)


if __name__ == '__main__':
    main()
