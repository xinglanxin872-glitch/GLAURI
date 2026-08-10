#!/usr/bin/env python3
"""STEP 1 of 3 (colabel pocket pipeline) -- pocket clustering + complex-PDB cache.
    step 1: step1_batch_cluster_pocket_labels.py  (this file)
    step 2: step2_pocket_binary_labels.py
    step 3: step3_build_pocket_store.py

Per cluster: collect contact residues for one interface class (protein/nucleic
acid/ion/ligand), superpose cluster members onto a reference chain (USalign),
then spatially group contact residues into "pockets" (DBSCAN on superposed
coordinates). Also caches per-ckey receptor+ligand complex PDBs, which
``step2_pocket_binary_labels.py --complex-pdb-dir`` (step 2) consumes downstream to
compute the actual (3.5A-contact, geometry-only) binary labels.

No label scoring here (no jdock/rosetta/direct score computation) -- this
script only clusters pockets and writes the ckey membership + complex-PDB
cache; label values are computed independently by step2_pocket_binary_labels.py.
"""

from __future__ import annotations

import argparse
import ast
import pickle
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch as pt


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for path in (THIS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# NOTE: intentionally not adding REPO_ROOT/"model" to sys.path -- having that directory's
# *contents* importable as bare top-level names (config, data_handler, model...) collides
# with "model" also being importable as the repo-root package (model.data_handler,
# model.model, ...); Python caches whichever resolves first in sys.modules['model'] and
# poisons every later `from model.xxx import ...` in the process. Repo-root + qualified
# imports only, everywhere.
from model.data_handler import Dataset  # noqa: E402
from src.data_encoding import categ_to_resnames  # noqa: E402
from src.dataset import select_by_interface_types  # noqa: E402

import colabel_util  # noqa: E402


class StageTimer:
    """Wall-clock checkpoints for profiling; grep logs with [TIMING]."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._t0 = time.perf_counter()
        self._last = self._t0

    def checkpoint(self, stage: str) -> None:
        now = time.perf_counter()
        step_sec = now - self._last
        total_sec = now - self._t0
        print(
            f"[TIMING] {self.label} | {stage}: +{step_sec:.2f}s (total {total_sec:.2f}s)",
            flush=True,
        )
        self._last = now


DETAIL_COLUMNS = [
    "cluster_name",
    "ref_id",
    "ref_seq",
    "pocket_id",
    "ckey",
    "receptor_key",
    "ligand_key",
]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(name)).strip("_") or "cluster"


def parse_cluster_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    if args.cluster_name:
        names.extend(args.cluster_name)
    if args.clusters:
        with open(args.clusters, "r") as f:
            names.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    return list(dict.fromkeys(names))


def parse_usalign_matrix(txt_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros((3, 3), dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    pattern = re.compile(
        r"^\s*(\d)\s+([-0-9.Ee]+)\s+([-0-9.Ee]+)\s+([-0-9.Ee]+)\s+([-0-9.Ee]+)"
    )
    found = 0
    with open(txt_path, "r") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            i = int(m.group(1))
            t[i] = float(m.group(2))
            u[i, 0] = float(m.group(3))
            u[i, 1] = float(m.group(4))
            u[i, 2] = float(m.group(5))
            found += 1
            if found == 3:
                break
    if found != 3:
        raise RuntimeError(f"cannot parse a complete USalign transform from {txt_path}")
    return u, t


def parse_tm_score(stdout: str) -> float | None:
    scores = re.findall(r"TM-score=\s*([0-9.]+)", stdout)
    if not scores:
        return None
    vals = [float(x) for x in scores[:2]]
    return max(vals)


def parse_alignment_stats(stdout: str) -> tuple[int | None, float | None]:
    m = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([0-9.]+)", stdout)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def parse_chainid(chainid: str) -> tuple[str, str, str]:
    parts = chainid.split("/")
    if len(parts) != 3:
        raise ValueError(f"chain id must be bucket/sample/chain:model, got {chainid!r}")
    sample = parts[1]
    chain_part = parts[2]
    if ":" in chain_part:
        chain, model = chain_part.split(":", 1)
    else:
        chain, model = chain_part, "0"
    return sample, chain, model


def chainid_to_all_seq_key(chainid: str) -> str:
    sample, chain, _model = parse_chainid(chainid)
    return f"{sample.upper()}_{chain}"


def structure_path(pdb_dir: str | Path, sample_id: str, layout: str) -> str:
    pdb_dir = Path(pdb_dir)
    candidates: list[Path]
    if layout == "bio":
        candidates = [pdb_dir / f"{sample_id}_bio1.pdb.gz", pdb_dir / f"{sample_id}_bio1.pdb"]
    else:
        sample_dir = pdb_dir / sample_id
        candidates = [
            sample_dir / f"{sample_id}_delig-desolv.pdb",
            sample_dir / f"{sample_id}_clean.pdb",
            pdb_dir / f"{sample_id}_bio1.pdb.gz",
            pdb_dir / f"{sample_id}_bio1.pdb",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"cannot find structure for {sample_id}; tried {tried}")


def cached_subunit_pdb_path(cache_dir: str | Path, chainid: str) -> Path:
    safe_key = chainid.replace("/", "__")
    return Path(cache_dir) / "pdb" / f"{safe_key}.pdb"


def write_cluster_subunit_pdbs(h5_path: str, cache_dir: str | Path, chainids: Iterable[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    unique_chainids = list(dict.fromkeys(chainids))
    with h5py.File(h5_path, "r") as hf:
        for chainid in unique_chainids:
            out = cached_subunit_pdb_path(cache_dir, chainid)
            if not out.is_file():
                colabel_util.write_structure_key_pdb(hf, chainid, out)
            paths[chainid] = out
    return paths


def run_usalign_one(
    usalign_bin: str,
    chainid: str,
    ref_id: str,
    pdb_dir: str,
    pdb_layout: str,
    pdb_cache_dir: str | None = None,
) -> dict[str, Any]:
    sample1, chain1, model1 = parse_chainid(chainid)
    sample2, chain2, model2 = parse_chainid(ref_id)
    if chainid == ref_id:
        return {
            "pdb1": chainid,
            "pdb2": ref_id,
            "u": np.eye(3, dtype=np.float64),
            "t": np.zeros(3, dtype=np.float64),
            "tm_score": 1.0,
            "aligned_stats": None,
            "RMSD": 0.0,
        }

    use_cached_subunits = False
    if pdb_cache_dir:
        cached1 = cached_subunit_pdb_path(pdb_cache_dir, chainid)
        cached2 = cached_subunit_pdb_path(pdb_cache_dir, ref_id)
        if cached1.is_file() and cached2.is_file():
            file1 = str(cached1)
            file2 = str(cached2)
            use_cached_subunits = True
        else:
            file1 = structure_path(pdb_dir, sample1, pdb_layout)
            file2 = structure_path(pdb_dir, sample2, pdb_layout)
    else:
        file1 = structure_path(pdb_dir, sample1, pdb_layout)
        file2 = structure_path(pdb_dir, sample2, pdb_layout)

    matrix_name = f"matrix_{safe_name(chainid)}.txt"
    with tempfile.TemporaryDirectory(prefix="usalign_") as tmpdir:
        if use_cached_subunits:
            cmd = [usalign_bin, file1, file2, "-m", matrix_name]
        else:
            cmd = [
                usalign_bin,
                "-chain1",
                chain1,
                "-model1",
                str(int(model1) + 1),
                file1,
                "-chain2",
                chain2,
                "-model2",
                str(int(model2) + 1),
                file2,
                "-m",
                matrix_name,
            ]
        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        u, t = parse_usalign_matrix(Path(tmpdir) / matrix_name)
    aligned_len, rmsd = parse_alignment_stats(proc.stdout)
    return {
        "pdb1": chainid,
        "pdb2": ref_id,
        "u": u,
        "t": t,
        "tm_score": parse_tm_score(proc.stdout),
        "aligned_stats": aligned_len,
        "RMSD": rmsd,
    }


def load_interface_labels(hgrp: h5py.Group, t0: pt.Tensor, t1_l: list[pt.Tensor]) -> pt.Tensor:
    shape = tuple(hgrp.attrs["Y_shape"])
    ids = pt.from_numpy(np.asarray(hgrp["Y"], dtype=np.int64))
    y_ctc_r = pt.any((ids[:, 2].view(-1, 1) == t0), dim=1).view(-1, 1)
    y_ctc_l = pt.stack([pt.any((ids[:, 3].view(-1, 1) == t1), dim=1) for t1 in t1_l], dim=1)
    y_ctc = y_ctc_r & y_ctc_l
    y = pt.zeros((shape[0], len(t1_l)), dtype=pt.bool)
    row_ids = ids[:, 0]
    pair_idx, col_idx = pt.where(y_ctc)
    y[row_ids[pair_idx], col_idx] = True
    return y


def get_ref_seq_from_all_seq_dict(chainid: str, all_seq_dict: dict[str, str]) -> str:
    all_seq_key = chainid_to_all_seq_key(chainid)
    try:
        return str(all_seq_dict[all_seq_key])
    except KeyError as exc:
        raise KeyError(f"missing {all_seq_key!r} for chainid={chainid!r} in all_seq_dict") from exc


def apply_usalign_transform(coords: pt.Tensor | np.ndarray, u: np.ndarray, t: np.ndarray) -> pt.Tensor:
    if not isinstance(coords, pt.Tensor):
        coords = pt.as_tensor(coords, dtype=pt.float32)
    u_t = pt.as_tensor(u, dtype=coords.dtype, device=coords.device)
    t_t = pt.as_tensor(t, dtype=coords.dtype, device=coords.device)
    return coords @ u_t.T + t_t


def atom_type_indices(atom_mode: str) -> list[int]:
    if atom_mode == "CA":
        return [0]
    if atom_mode == "CB":
        return [4]
    if atom_mode == "SC":
        return list(range(4, 36))
    raise ValueError(f"unsupported atom mode: {atom_mode}")


def dbscan_group_aware_fast(X: np.ndarray, eps: float, min_samples: int = 1) -> np.ndarray:
    if min_samples != 1:
        raise ValueError("dbscan_group_aware_fast currently supports min_samples=1 only")
    from scipy.spatial import cKDTree

    X = np.asarray(X, dtype=np.float64)
    coords = X[:, :3]
    groups = X[:, 3].astype(np.int64)
    unique_groups, inv = np.unique(groups, return_inverse=True)
    parent = np.arange(len(unique_groups), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    tree = cKDTree(coords)
    for i, coord in enumerate(coords):
        gi = inv[i]
        for j in tree.query_ball_point(coord, r=eps, p=2.0):
            gj = inv[j]
            if gi != gj:
                union(int(gi), int(gj))

    roots = np.array([find(i) for i in range(len(unique_groups))], dtype=np.int64)
    _, labels_by_group = np.unique(roots, return_inverse=True)
    return labels_by_group[inv].astype(np.int64)


def build_dataset(h5_path: str, min_num_res: int) -> Dataset:
    config = {
        "dataset_filepath": h5_path,
        "max_ba": 1,
        "max_size": 50000,
        "min_num_res": min_num_res,
        "l_types": categ_to_resnames["protein"],
        "r_types": [
            categ_to_resnames["protein"],
            categ_to_resnames["dna"] + categ_to_resnames["rna"],
            categ_to_resnames["ion"],
            categ_to_resnames["ligand"],
        ],
        "features_flags": (False, False, True),
        "extract_backbone": False,
    }
    dataset = Dataset(config)
    mask = dataset.sizes[:, 1] >= min_num_res
    mask &= select_by_interface_types(dataset, config["l_types"])
    dataset.update_mask(mask)
    dataset.set_types(config["l_types"], config["r_types"])
    return dataset


def collect_contacts(
    dataset: Dataset,
    id_list: Iterable[str],
    mapping: dict[str, int],
    *,
    contact_class: int,
) -> dict[str, list[tuple[int, int]]]:
    binding_idx: dict[str, list[tuple[int, int]]] = {}
    with h5py.File(dataset.dataset_filepath, "r") as hf:
        for chainid in id_list:
            if chainid not in mapping:
                print(f"[WARN] skip {chainid}: not in filtered dataset", file=sys.stderr)
                continue
            key = dataset.ukeys[mapping[chainid]]
            for ckey in dataset.ckeys_map.get(key, []):
                y = load_interface_labels(hf[f"data/contacts/{ckey}"], dataset.t0, dataset.t1_l)
                if contact_class >= y.shape[1]:
                    continue
                if y[:, contact_class].any():
                    row_idx = pt.where(y[:, contact_class])[0]
                    binding_idx[ckey] = [(int(i), int(contact_class)) for i in row_idx.tolist()]
    return binding_idx


def choose_ref_id(id_list: Iterable[str], mapping: dict[str, int], all_seq_dict: dict[str, str]) -> str:
    best_id = None
    best_len = -1
    for chainid in id_list:
        if chainid not in mapping:
            continue
        try:
            seq_len = len(get_ref_seq_from_all_seq_dict(chainid, all_seq_dict))
        except KeyError:
            continue
        if seq_len > best_len:
            best_id = chainid
            best_len = seq_len
    if best_id is None:
        raise RuntimeError("no cluster member is available in filtered dataset with all_seq_dict sequence")
    return best_id


def run_alignments(
    args: argparse.Namespace,
    id_list: list[str],
    ref_id: str,
    *,
    timer: StageTimer | None = None,
) -> pd.DataFrame:
    rows = []
    pdb_cache_dir = str(args.cache_dir) if args.cache_dir else None
    t_align = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.align_workers) as executor:
        futures = {
            executor.submit(
                run_usalign_one,
                args.usalign_bin,
                chainid,
                ref_id,
                args.pdb_dir,
                args.pdb_layout,
                pdb_cache_dir,
            ): chainid
            for chainid in id_list
        }
        for fut in as_completed(futures):
            chainid = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"[WARN] USalign failed for {chainid}: {exc}", file=sys.stderr)
    if not rows:
        raise RuntimeError("all USalign jobs failed")
    align_sec = time.perf_counter() - t_align
    msg = (
        f"USalign done: {len(rows)}/{len(id_list)} ok, "
        f"workers={args.align_workers}, {align_sec:.2f}s"
    )
    if timer is not None:
        timer.checkpoint(msg)
    else:
        print(f"[TIMING] {msg}", flush=True)
    return pd.DataFrame(rows).set_index("pdb1")


def load_chain_arrays(dataset: Dataset, mapping: dict[str, int], chainid: str) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    X, _ids_topk, q, M, _y = dataset[mapping[chainid]]
    return X, q, M


def build_pocket_ckey(
    dataset: Dataset,
    mapping: dict[str, int],
    binding_idx: dict[str, list[tuple[int, int]]],
    df_align: pd.DataFrame,
    *,
    atom_mode: str,
    pocket_eps: float,
    contact_class: int,
) -> dict[int, list[str]]:
    selected_atom_types = pt.tensor(atom_type_indices(atom_mode))
    global_atms: list[pt.Tensor] = []
    group_ckey: dict[int, str] = {}
    arrays_cache: dict[str, tuple[pt.Tensor, pt.Tensor, pt.Tensor]] = {}
    cur_group = 0

    for ckey, binding_res in binding_idx.items():
        chainid = "/".join(ckey.split("/")[:3])
        if chainid not in mapping or chainid not in df_align.index:
            cur_group += 1
            continue
        if chainid not in arrays_cache:
            arrays_cache[chainid] = load_chain_arrays(dataset, mapping, chainid)
        X, q, M = arrays_cache[chainid]
        atm_res = pt.stack(pt.where(M), dim=1)
        atm_type = pt.stack(pt.where(q), dim=1)

        x4d_parts = []
        for res_id, classnum in binding_res:
            if classnum != contact_class:
                continue
            mask = (atm_res[:, 1] == res_id) & pt.isin(atm_type[:, 1], selected_atom_types)
            coords = X[mask]
            if coords.numel() == 0:
                continue
            group_col = pt.full((coords.size(0), 1), cur_group, dtype=coords.dtype, device=coords.device)
            x4d_parts.append(pt.cat([coords, group_col], dim=1))
            group_ckey[cur_group] = ckey

        if x4d_parts:
            x4d = pt.cat(x4d_parts, dim=0)
            u = df_align.loc[chainid, "u"]
            t = df_align.loc[chainid, "t"]
            coords_transformed = apply_usalign_transform(x4d[:, :3], u, t)
            global_atms.append(pt.cat([coords_transformed, x4d[:, 3:4]], dim=1))
        cur_group += 1

    if not global_atms:
        return {}

    X_all = pt.cat(global_atms, dim=0).cpu().numpy()
    pocket_labels = dbscan_group_aware_fast(X_all, eps=pocket_eps, min_samples=1)
    group_ids = X_all[:, 3].astype(int)
    pocket_ckey: dict[int, list[str]] = defaultdict(list)
    for group in np.unique(group_ids):
        labels = pocket_labels[group_ids == group]
        vals, cnts = np.unique(labels, return_counts=True)
        pocket_id = int(vals[cnts.argmax()])
        ckey = group_ckey.get(int(group))
        if ckey is not None:
            pocket_ckey[pocket_id].append(ckey)
    return dict(sorted(pocket_ckey.items(), key=lambda item: item[0]))


def cache_pocket_complex_pdbs(
    h5_path: str,
    cache_dir: str | Path,
    pocket_ckey: dict[int, list[str]],
) -> dict[str, Path]:
    """Write one receptor+ligand complex PDB per unique ckey.

    Output path/naming (``<cache_dir>/complex_pdb/<ckey with / -> __, : -> _>.pdb``)
    matches the old scoring-step cache from ``H5RosettaContactScorer``/
    ``H5JdockContactScorer``, so ``step2_pocket_binary_labels.py --complex-pdb-dir`` keeps
    working unchanged against this cache. Extracted out of the (now-removed) scoring
    step -- this is the only piece of that step still needed downstream.
    """
    complex_dir = Path(cache_dir) / "complex_pdb"
    complex_dir.mkdir(parents=True, exist_ok=True)
    all_ckeys = list(dict.fromkeys(ckey for ckeys in pocket_ckey.values() for ckey in ckeys))
    written: dict[str, Path] = {}
    with h5py.File(h5_path, "r") as hf:
        for ckey in all_ckeys:
            out_pdb = complex_dir / f"{ckey.replace('/', '__').replace(':', '_')}.pdb"
            if out_pdb.is_file():
                written[ckey] = out_pdb
                continue
            receptor_key, ligand_key = colabel_util.parse_ckey(ckey)
            try:
                path, _rec_chain, _lig_chain = colabel_util.write_ckey_complex_pdb(
                    hf, receptor_key, ligand_key, out_pdb
                )
                written[ckey] = path
            except KeyError as exc:
                print(f"[WARN] complex pdb skip {ckey}: {exc}", file=sys.stderr)
    return written


def pocket_ckey_to_detail_df(
    cluster_name: str,
    ref_id: str,
    ref_seq: str,
    pocket_ckey: dict[int, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pocket_id, ckeys in pocket_ckey.items():
        for ckey in ckeys:
            receptor_key, ligand_key = colabel_util.parse_ckey(ckey)
            rows.append(
                {
                    "cluster_name": cluster_name,
                    "ref_id": ref_id,
                    "ref_seq": ref_seq,
                    "pocket_id": pocket_id,
                    "ckey": ckey,
                    "receptor_key": receptor_key,
                    "ligand_key": ligand_key,
                }
            )
    return pd.DataFrame(rows, columns=DETAIL_COLUMNS)


def detail_path_for_cluster(out_dir: str | Path, cluster_name: str) -> Path:
    return Path(out_dir) / "details" / f"{safe_name(cluster_name)}_pocket_interactions.csv"


def summary_row_from_detail_csv(cluster_name: str, detail_path: Path) -> dict[str, Any] | None:
    try:
        detail_df = pd.read_csv(detail_path)
    except pd.errors.EmptyDataError:
        return None
    if detail_df.empty:
        return None
    missing = [col for col in ("cluster_name", "ref_id", "ref_seq", "pocket_id", "ckey") if col not in detail_df.columns]
    if missing:
        print(f"[WARN] cannot resume {cluster_name}: missing detail columns {missing}", file=sys.stderr)
        return None
    return {
        "cluster_name": cluster_name,
        "ref_id": str(detail_df["ref_id"].iloc[0]),
        "ref_seq": str(detail_df["ref_seq"].iloc[0]),
        "detail_csv": str(detail_path),
        "n_pockets": int(detail_df["pocket_id"].nunique()),
        "n_ckeys": int(len(detail_df)),
    }


def write_summary_files(summary_rows: list[dict[str, Any]], out_dir: Path, *, partial: bool = False) -> None:
    if not summary_rows:
        return
    suffix = ".partial" if partial else ""
    pkl_path = out_dir / f"cluster_labels_summary{suffix}.pkl"
    csv_path = out_dir / f"cluster_labels_summary{suffix}.csv"
    with open(pkl_path, "wb") as f:
        pickle.dump(summary_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)


def process_cluster(
    cluster_name: str,
    cluster_df: pd.DataFrame,
    dataset: Dataset,
    mapping: dict[str, int],
    all_seq_dict: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    timer = StageTimer(cluster_name)
    timer.checkpoint("start")

    id_list = list(cluster_df.loc[cluster_name, "seq_name"])
    ref_id = choose_ref_id(id_list, mapping, all_seq_dict)
    ref_seq = get_ref_seq_from_all_seq_dict(ref_id, all_seq_dict)
    print(f"[INFO] {cluster_name}: {len(id_list)} members, ref={ref_id}, ref_len={len(ref_seq)}")
    timer.checkpoint("choose_ref")

    binding_idx = collect_contacts(
        dataset,
        id_list,
        mapping,
        contact_class=args.contact_class,
    )
    print(f"[INFO] {cluster_name}: {len(binding_idx)} contacts (class={args.contact_class})")
    timer.checkpoint(f"collect_contacts ({len(binding_idx)} ckeys)")

    align_ids = [x for x in id_list if x in mapping]
    if args.cache_dir:
        written = write_cluster_subunit_pdbs(args.h5, args.cache_dir, align_ids)
        print(f"[INFO] {cluster_name}: prepared {len(written)} cached subunit PDBs for USalign")
        timer.checkpoint(f"write_subunit_pdbs ({len(written)} files)")
    df_align = run_alignments(args, align_ids, ref_id, timer=timer)
    pocket_ckey = build_pocket_ckey(
        dataset,
        mapping,
        binding_idx,
        df_align,
        atom_mode=args.atom_mode,
        pocket_eps=args.pocket_eps,
        contact_class=args.contact_class,
    )
    n_ckeys = sum(len(v) for v in pocket_ckey.values())
    print(f"[INFO] {cluster_name}: {len(pocket_ckey)} pockets")
    timer.checkpoint(f"build_pocket_ckey ({len(pocket_ckey)} pockets, {n_ckeys} ckeys)")

    if args.cache_dir:
        written_complex = cache_pocket_complex_pdbs(args.h5, args.cache_dir, pocket_ckey)
        print(f"[INFO] {cluster_name}: cached {len(written_complex)}/{n_ckeys} complex PDBs")
        timer.checkpoint(f"cache_complex_pdbs ({len(written_complex)} files)")
    else:
        print(
            f"[WARN] {cluster_name}: --cache-dir not set, skipping complex-PDB cache "
            "(step2_pocket_binary_labels.py --complex-pdb-dir mode needs this; use --h5 mode "
            "there instead if you don't set --cache-dir here)",
            file=sys.stderr,
        )

    detail_df = pocket_ckey_to_detail_df(cluster_name, ref_id, ref_seq, pocket_ckey)

    details_dir = Path(args.out_dir) / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    detail_path = details_dir / f"{safe_name(cluster_name)}_pocket_interactions.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"[INFO] {cluster_name}: wrote {detail_path}")
    timer.checkpoint(f"write_detail_csv ({len(detail_df)} rows)")

    return {
        "cluster_name": cluster_name,
        "ref_id": ref_id,
        "ref_seq": ref_seq,
        "detail_csv": str(detail_path),
        "n_pockets": len(pocket_ckey),
        "n_ckeys": len(detail_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--cluster-summary", required=True)
    parser.add_argument("--clusters", help="Text file with one cluster name per line")
    parser.add_argument("--cluster-name", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--usalign-bin", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--pdb-layout", choices=["bio", "openbind"], default="bio")
    parser.add_argument(
        "--contact-class",
        type=int,
        required=True,
        help="Contact class column from load_interface_labels: 0=protein, 1=nucleic acid (dna+rna), 2=ion, 3=ligand.",
    )
    parser.add_argument(
        "--cache-dir",
        help=(
            "Root for cached subunit PDBs (USalign input) and per-ckey complex PDBs "
            "(consumed by step2_pocket_binary_labels.py --complex-pdb-dir mode). Strongly "
            "recommended -- without it neither cache is written and USalign/downstream "
            "pocket_binary need other arrangements."
        ),
    )
    parser.add_argument("--align-workers", type=int, default=8)
    parser.add_argument("--atom-mode", choices=["CA", "CB", "SC"], default="SC")
    parser.add_argument("--pocket-eps", type=float, default=0.5)
    parser.add_argument("--min-num-res", type=int, default=48)
    parser.add_argument(
        "--all-seq-dict",
        default=str(THIS_DIR.parent / "colabel" / "all_seq_dict.pkl"),
        help="PKL mapping SAMPLE_CHAIN to sequence; used to choose ref_id and load ref_seq.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing per-cluster detail CSVs and partial summary files in --out-dir.",
    )
    args = parser.parse_args()
    # USalign runs with cwd=<tempdir> (see run_usalign_one); a relative --cache-dir/--pdb-dir
    # would silently resolve to nothing there (or feed USalign a garbage/missing path -- this
    # has been observed to SIGSEGV it rather than error cleanly). Resolve to absolute once here
    # so every downstream consumer (subunit cache, USalign subprocess, complex-PDB cache) agrees.
    if args.cache_dir:
        args.cache_dir = str(Path(args.cache_dir).resolve())
    args.pdb_dir = str(Path(args.pdb_dir).resolve())
    return args


def main() -> None:
    job_timer = StageTimer("job")
    args = parse_args()
    cluster_names = parse_cluster_names(args)
    if not cluster_names:
        raise SystemExit("Provide --clusters or at least one --cluster-name")
    job_timer.checkpoint(f"parse_args ({len(cluster_names)} clusters)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    cluster_df = pd.read_csv(args.cluster_summary, index_col="cluster_names")
    cluster_df["seq_name"] = cluster_df["seq_name"].apply(ast.literal_eval)
    job_timer.checkpoint("load_cluster_summary")

    with open(args.all_seq_dict, "rb") as f:
        all_seq_dict = pickle.load(f)
    job_timer.checkpoint(f"load_all_seq_dict ({len(all_seq_dict)} entries)")

    dataset = build_dataset(args.h5, args.min_num_res)
    mapping = {key: idx for idx, key in enumerate(dataset.ukeys)}
    job_timer.checkpoint(f"build_dataset ({len(mapping)} ukeys)")

    summary_rows: list[dict[str, Any]] = []
    completed_clusters: set[str] = set()
    partial_pkl = out_dir / "cluster_labels_summary.partial.pkl"
    if args.resume_existing and partial_pkl.is_file():
        with open(partial_pkl, "rb") as f:
            loaded_rows = pickle.load(f)
        for row in loaded_rows:
            if "cluster_name" in row:
                summary_rows.append(row)
                completed_clusters.add(str(row["cluster_name"]))
        print(f"[INFO] resumed {len(completed_clusters)} clusters from {partial_pkl}", flush=True)

    for cluster_name in cluster_names:
        if cluster_name not in cluster_df.index:
            print(f"[WARN] skip missing cluster {cluster_name}", file=sys.stderr)
            continue
        if args.resume_existing and cluster_name in completed_clusters:
            print(f"[INFO] resume skip {cluster_name}: already in partial summary", flush=True)
            continue
        if args.resume_existing:
            existing_detail = detail_path_for_cluster(out_dir, cluster_name)
            if existing_detail.is_file():
                resumed_row = summary_row_from_detail_csv(cluster_name, existing_detail)
                if resumed_row is not None:
                    summary_rows.append(resumed_row)
                    completed_clusters.add(cluster_name)
                    print(f"[INFO] resume skip {cluster_name}: recovered from {existing_detail}", flush=True)
                    write_summary_files(summary_rows, out_dir, partial=True)
                    continue
                print(f"[INFO] resume rerun {cluster_name}: cannot recover from empty/incomplete {existing_detail}", flush=True)
        try:
            row = process_cluster(cluster_name, cluster_df, dataset, mapping, all_seq_dict, args)
            summary_rows.append(row)
            completed_clusters.add(cluster_name)
            write_summary_files(summary_rows, out_dir, partial=True)
        except Exception as exc:
            print(f"[ERROR] {cluster_name}: {exc}", file=sys.stderr)

    if not summary_rows:
        raise SystemExit("No clusters completed successfully")
    job_timer.checkpoint(f"all_clusters ({len(summary_rows)} ok)")

    pkl_path = out_dir / "cluster_labels_summary.pkl"
    csv_path = out_dir / "cluster_labels_summary.csv"
    write_summary_files(summary_rows, out_dir, partial=False)
    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {pkl_path}")
    job_timer.checkpoint("write_summary")


if __name__ == "__main__":
    main()
