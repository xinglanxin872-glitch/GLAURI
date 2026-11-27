#!/usr/bin/env python3
"""Pocket clustering over a chain clustering -> per-contact interaction table.

For every cluster in ``--cluster-csv``, for one interface class:

  ref_seq  = selected by --ref-policy.  ``cluster-name`` reuses the representative in
             `cluster_names`; ``longest`` explicitly scans usable cluster members and is
             required when the input is the legacy MMseqs table, whose representative is
             not necessarily the longest sequence.
  per ckey = receptor sequence read out of the H5 structure, left-most globally aligned to
             ref_seq (leftmost_align.py -- ties bind to the earliest reference residue so
             every contact in a cluster lands in the same reference columns), then the
             contact residues are projected to a 0/1 string of len(ref_seq)
  pocket_id= Tanimoto/Jaccard distance on those 0/1 strings + average-linkage hierarchical
             clustering cut at --threshold (default 0.8)

Output columns follow colabel/pocket_pocket_interactions_all.csv, plus `ref_label`
(the aligned 0/1 string) and `pocket_n_contact`.

    conda activate boltz
    python cluster_contact_labels.py --contact-class 0 --shard 0/8
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))

import h5py
import numpy as np
import pandas as pd

import contact_io
import pocket_cluster_tanimoto as pct
from leftmost_align import leftmost_align

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CLSNAME = {0: "protein", 1: "nn", 2: "ion", 3: "ligand"}

COLUMNS = ["cluster_name", "ref_id", "ref_seq", "pocket_id", "pocket_n_contact",
           "ckey", "receptor_key", "ligand_key", "seq_len", "n_contact_res", "n_aligned",
           "receptor_seq", "labels_original", "ref_label"]


def seq_from_all_seq_dict(ukey: str, all_seq_dict: dict) -> str | None:
    p = ukey.split("/")
    pdb, ch = p[1], p[2].split(":")[0]
    for k in (f"{pdb.upper()}_{ch}", f"{pdb.upper()}_{ch.lower()}", f"{pdb.upper()}_{ch.upper()}"):
        if k in all_seq_dict:
            return str(all_seq_dict[k])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact-class", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--cluster-csv", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--all-seq-dict", required=True)
    ap.add_argument("--ref-policy", choices=["cluster-name", "longest"],
                    default="longest",
                    help="reference sequence selection (use longest for legacy MMseqs)")
    threshold_group = ap.add_mutually_exclusive_group()
    threshold_group.add_argument("--threshold", type=float, help="Jaccard distance cutoff; default 0.8")
    threshold_group.add_argument("--similarity-threshold", type=float, help="Equivalent similarity cutoff; 0.2 means distance 0.8")
    ap.add_argument("--method", default="average", choices=["average", "single", "complete"])
    ap.add_argument("--min-num-res", type=int, default=48)
    ap.add_argument("--shard", default=None, help="i/N")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-seqs", action="store_true",
                    help="Drop ref_seq/receptor_seq columns to shrink the output.")
    args = ap.parse_args()
    args.threshold = (1.0 - args.similarity_threshold if args.similarity_threshold is not None
                      else 0.8 if args.threshold is None else args.threshold)
    if not 0 <= args.threshold <= 1:
        ap.error("distance and similarity thresholds must lie in [0, 1]")
    if args.shard:
        try:
            shard_i, shard_n = map(int, args.shard.split("/"))
            if not 0 <= shard_i < shard_n:
                raise ValueError
        except ValueError:
            ap.error("--shard must be i/N with 0 <= i < N")

    cls = args.contact_class
    name = CLSNAME[cls]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}" + (f"_shard{args.shard.replace('/', 'of')}" if args.shard else "")
    out_csv = out_dir / f"{tag}_pocket_interactions.csv.gz"
    out_stats = out_dir / f"{tag}_stats.json"

    t0 = time.perf_counter()
    dataset = contact_io.H5Contacts(args.h5, args.min_num_res)
    mapping = {k: i for i, k in enumerate(dataset.ukeys)}
    with open(args.all_seq_dict, "rb") as f:
        all_seq_dict = pickle.load(f)
    cdf = pd.read_csv(args.cluster_csv)
    cdf["seq_name"] = cdf.seq_name.apply(ast.literal_eval)
    print(f"[INIT] {time.perf_counter()-t0:.0f}s  {len(mapping)} chains, "
          f"{len(cdf)} clusters, class={cls} ({name})", flush=True)

    rows_idx = list(range(len(cdf)))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        rows_idx = rows_idx[i::n]
    if args.limit:
        rows_idx = rows_idx[:args.limit]
    print(f"[INIT] processing {len(rows_idx)} clusters", flush=True)

    st = Counter()
    warn_examples: list[str] = []
    n_rows = 0
    cols = [c for c in COLUMNS if not (args.no_seqs and c in ("ref_seq", "receptor_seq"))]

    t0 = time.perf_counter()
    with gzip.open(out_csv, "wt", newline="") as fh, h5py.File(args.h5, "r") as hf:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for n_done, ri in enumerate(rows_idx, 1):
            r = cdf.iloc[ri]
            cluster_name = r.cluster_names
            try:
                members = [u for u in r.seq_name if u in mapping]
                if len(members) < 1:
                    st["cluster_no_member_in_dataset"] += 1
                    continue

                if args.ref_policy == "longest":
                    # Reuse the original GLAURI rule exactly: longest usable member;
                    # ties retain the first member in the cluster CSV.
                    ref_id = contact_io.choose_ref_id(members, mapping, all_seq_dict)
                    ref_seq = seq_from_all_seq_dict(ref_id, all_seq_dict)
                    st["cluster_ref_longest"] += 1
                else:
                    ref_id = cluster_name
                    ref_seq = seq_from_all_seq_dict(ref_id, all_seq_dict)

                # A cluster-name representative can be unavailable after Dataset
                # filtering.  Keep the historical fallback, but make it deterministic
                # and consistent with the explicit longest policy.
                if ref_id not in mapping or not ref_seq:
                    ref_id = contact_io.choose_ref_id(members, mapping, all_seq_dict)
                    ref_seq = seq_from_all_seq_dict(ref_id, all_seq_dict)
                    st["cluster_ref_fallback"] += 1

                binding_idx = dataset.collect_contacts(members, cls, hf)
                if not binding_idx:
                    st["cluster_no_contact"] += 1
                    continue

                n_ref = len(ref_seq)
                aln_cache: dict[str, tuple] = {}
                seq_cache: dict[str, str] = {}
                ckeys, labels, meta = [], [], []
                for ckey, br in binding_idx.items():
                    chainid = "/".join(ckey.split("/")[:3])
                    if chainid not in seq_cache:
                        g = hf.get(f"data/structures/{chainid}")
                        if g is None:
                            st["ckey_missing_structure"] += 1
                            continue
                        cs = dataset.structure_sequence(g)
                        seq_cache[chainid] = cs
                        aln_cache[chainid] = leftmost_align(cs, ref_seq)
                    cs = seq_cache[chainid]
                    rp, cp = aln_cache[chainid]
                    res_ids = [i for i, c in br if c == cls]
                    if not res_ids:
                        st["ckey_no_res_for_class"] += 1
                        continue
                    chain_lab = np.zeros(len(cs), dtype=bool)
                    ok = [i for i in res_ids if 0 <= i < len(cs)]
                    if len(ok) != len(res_ids):
                        st["ckey_res_index_out_of_range"] += 1
                        if len(warn_examples) < 20:
                            warn_examples.append(
                                f"{ckey}: res idx max {max(res_ids)} >= len(seq) {len(cs)}")
                    if not ok:
                        continue
                    chain_lab[np.asarray(ok)] = True
                    vec = np.zeros(n_ref, dtype=bool)
                    if len(rp):
                        vec[rp] = chain_lab[cp]
                    if not vec.any():
                        st["ckey_label_empty_after_align"] += 1
                        continue
                    rk, lk = contact_io.parse_ckey(ckey)
                    ckeys.append(ckey); labels.append(vec)
                    meta.append((rk, lk, cs, sorted(ok), int(len(rp))))

                if not ckeys:
                    st["cluster_all_ckeys_dropped"] += 1
                    continue
                L = np.stack(labels)
                pid, _, _ = pct.cluster_labels(L, threshold=args.threshold,
                                               method=args.method)
                psize = np.bincount(pid)
                st["clusters_ok"] += 1
                st["pockets"] += int(pid.max()) + 1

                for k, (ckey, vec, (rk, lk, cs, res, nal), p) in enumerate(
                        zip(ckeys, L, meta, pid)):
                    row = {"cluster_name": cluster_name, "ref_id": ref_id,
                           "ref_seq": ref_seq, "pocket_id": int(p),
                           "pocket_n_contact": int(psize[p]), "ckey": ckey,
                           "receptor_key": rk, "ligand_key": lk, "seq_len": n_ref,
                           "n_contact_res": len(res), "n_aligned": nal,
                           "receptor_seq": cs,
                           "labels_original": json.dumps(
                               [[int(i), cs[i] if i < len(cs) else "X", 1] for i in res]),
                           "ref_label": "".join("1" if v else "0" for v in vec.tolist())}
                    w.writerow(row); n_rows += 1
            except Exception as exc:
                st["cluster_error"] += 1
                if len(warn_examples) < 20:
                    warn_examples.append(f"{cluster_name}: {type(exc).__name__}: {exc}")
            if n_done % 2000 == 0:
                el = time.perf_counter() - t0
                print(f"[{name}] {n_done}/{len(rows_idx)}  {el:.0f}s  "
                      f"rows={n_rows}  ok={st['clusters_ok']}  "
                      f"eta={el/n_done*(len(rows_idx)-n_done)/60:.0f}min", flush=True)

    stats = dict(st)
    stats.update({"n_rows": n_rows, "class": cls, "name": name,
                  "threshold": args.threshold, "clusters_processed": len(rows_idx),
                  "ref_policy": args.ref_policy,
                  "seconds": round(time.perf_counter() - t0, 1),
                  "warnings": warn_examples})
    stats["method"] = args.method
    stats["similarity_threshold"] = 1.0 - args.threshold
    out_stats.write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"[DONE] {name}: {n_rows} rows, {st['clusters_ok']} clusters, "
          f"{st['pockets']} pockets, {time.perf_counter()-t0:.0f}s -> {out_csv}", flush=True)
    for k, v in sorted(st.items()):
        print(f"   {k}: {v}", flush=True)
    if warn_examples:
        print("   WARN examples:", flush=True)
        for x in warn_examples[:10]:
            print(f"     {x}", flush=True)

    if st["cluster_error"]:
        raise SystemExit("Cluster errors occurred; inspect the stats file before merging outputs")


if __name__ == "__main__":
    main()
