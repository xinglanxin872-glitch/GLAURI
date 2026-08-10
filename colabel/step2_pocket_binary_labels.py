#!/usr/bin/env python3
"""STEP 2 of 3 (colabel pocket pipeline) -- per-pocket binary contact labels.
    step 1: step1_batch_cluster_pocket_labels.py
    step 2: step2_pocket_binary_labels.py  (this file)
    step 3: step3_build_pocket_store.py

For each protein cluster (one ``*_pocket_interactions.csv`` detail file produced
by ``step1_batch_cluster_pocket_labels.py``, step 1):

  * reuse the existing pocket assignment (``pocket_id`` column, 0-based);
  * for every complex (``ckey``) in a pocket, load the cached complex PDB
    (receptor = chain A, ligand = chain B) and mark receptor residues that have
    ANY heavy atom within ``--cutoff`` (default 3.5 A) of ANY ligand atom
    -> ``pdb_label`` (length = number of chain-A residues);
  * derive ``pdb_seq`` from chain A, globally align it to ``ref_seq`` (this
    file's own ``double_align``, a Bio.Align global-alignment helper) and map
    ``pdb_label`` -> ``ref_label`` (length = len(ref_seq));
  * OR (union) all complexes' ``ref_label`` within a pocket -> pocket label;
  * ``overlap`` = any two pockets share a 1 at the same position
    (False when the cluster has < 2 pockets).

Output: one row per pocket, columns
    cluster_id, pocket_index, pocket_sum, ref_id, ref_seq, pocket_label, overlap
``pocket_label`` is a compact "0/1" string of length len(ref_seq). Feeds into
``step3_build_pocket_store.py`` (step 3).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from Bio import Align

# CSV field with a very long ref_seq / label can exceed the default limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

OUT_COLUMNS = [
    "cluster_id", "pocket_index", "pocket_sum",
    "ref_id", "ref_seq", "pocket_label", "overlap",
]


def ckey_to_filename(ckey: str) -> str:
    """ckey 'bucket/pdb/recChain:model/ligChain:model[:idx]' -> complex pdb name."""
    return ckey.replace("/", "__").replace(":", "_") + ".pdb"


def receptor_chain_char(receptor_key: str) -> str:
    """First char of the receptor chain token, matching how the complex PDB was
    written (h5_jdock_contact_scores._chain_char_from_structure_key)."""
    chain_token = str(receptor_key).rsplit("/", 1)[-1].split(":", 1)[0]
    return chain_token[0] if chain_token else "A"


def double_align(pdb_seq: str, ref_seq: str):
    """Global align; return (ref_mask, pdb_mask) over aligned columns.

    Kept identical to batch_cluster_pocket_labels.double_align for consistency.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    alignment = aligner.align(ref_seq, pdb_seq)[0]
    ref_blocks, pdb_blocks = alignment.aligned
    ref_mask = np.zeros(len(ref_seq), dtype=bool)
    pdb_mask = np.zeros(len(pdb_seq), dtype=bool)
    for start, end in ref_blocks.tolist():
        ref_mask[start:end] = True
    for start, end in pdb_blocks.tolist():
        pdb_mask[start:end] = True
    return ref_mask, pdb_mask


def map_pdb_label_to_ref(pdb_label: np.ndarray, pdb_seq: str, ref_seq: str) -> np.ndarray:
    """Project a per-residue pdb_label onto ref_seq coordinates via alignment."""
    ref_label = np.zeros(len(ref_seq), dtype=np.int8)
    mask_ref, mask_pdb = double_align(pdb_seq, ref_seq)
    n = min(int(mask_ref.sum()), int(mask_pdb.sum()))
    if n:
        ref_pos = np.flatnonzero(mask_ref)[:n]
        pdb_pos = np.flatnonzero(mask_pdb)[:n]
        ref_label[ref_pos] = pdb_label[pdb_pos]
    return ref_label


def parse_complex(path: str, rec_chain: str):
    """Parse a cached complex PDB.

    Returns (rec_xyz[Ma,3], rec_resid[Ma], pdb_seq, lig_xyz[Mb,3], n_res).
    Receptor = the `rec_chain` chain (residues in file order); ligand = every
    other chain (the complex PDB always has exactly receptor + ligand chains,
    with distinct chain ids, per h5_jdock_contact_scores.write_ckey_complex_pdb).
    """
    rec_xyz: list[tuple[float, float, float]] = []
    rec_rid: list[int] = []
    resnames: list[str] = []
    res_index: dict[tuple[str, str], int] = {}
    lig_xyz: list[tuple[float, float, float]] = []

    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec != "ATOM  " and rec != "HETATM":
                continue
            chain = line[21]
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            if chain == rec_chain:
                rk = (line[22:26], line[26])  # (resSeq, iCode)
                idx = res_index.get(rk)
                if idx is None:
                    idx = len(resnames)
                    res_index[rk] = idx
                    resnames.append(line[17:20].strip())
                rec_xyz.append((x, y, z))
                rec_rid.append(idx)
            else:
                lig_xyz.append((x, y, z))

    pdb_seq = "".join(THREE_TO_ONE.get(r, "X") for r in resnames)
    return (
        np.asarray(rec_xyz, dtype=np.float64),
        np.asarray(rec_rid, dtype=np.int64),
        pdb_seq,
        np.asarray(lig_xyz, dtype=np.float64),
        len(resnames),
    )


def contact_residues(rec_xyz, rec_rid, lig_xyz, n_res, cutoff):
    """Residues (chain A) with any atom within `cutoff` of any ligand atom."""
    label = np.zeros(n_res, dtype=np.int8)
    if n_res == 0 or len(rec_xyz) == 0 or len(lig_xyz) == 0:
        return label
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(lig_xyz)
        dist, _ = tree.query(rec_xyz, distance_upper_bound=cutoff)
        hit = np.isfinite(dist) & (dist <= cutoff)
    except Exception:
        hit = np.zeros(len(rec_xyz), dtype=bool)
        cut2 = cutoff * cutoff
        for i in range(0, len(rec_xyz), 4096):
            chunk = rec_xyz[i:i + 4096]
            d2 = ((chunk[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1).min(1)
            hit[i:i + len(chunk)] = d2 <= cut2
    if hit.any():
        label[np.unique(rec_rid[hit])] = 1
    return label


# ---- worker globals (set via ProcessPoolExecutor initializer) ----
_INDEX: dict[str, str] = {}       # complex-pdb filename -> path (pdb mode)
_CUTOFF = 3.5
_H5PATH = None                    # if set, source atoms from the H5 (uniform mode)
_HF = None                        # per-worker h5py handle (lazy, not fork-safe)
_STD_RESNAMES = None
_RECCACHE: dict = {}              # per-worker receptor cache (receptors repeat)


def _init_worker(index: dict[str, str], cutoff: float, h5path) -> None:
    global _INDEX, _CUTOFF, _H5PATH, _HF, _RECCACHE
    _INDEX = index
    _CUTOFF = cutoff
    _H5PATH = h5path
    _HF = None
    _RECCACHE = {}


def _load_std_resnames() -> None:
    global _STD_RESNAMES
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "model"), os.path.join(here, "..", "model", "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from src.data_encoding import std_resnames
    except Exception:
        from data_encoding import std_resnames
    _STD_RESNAMES = [str(x).strip() for x in std_resnames]


def _get_hf():
    global _HF
    if _HF is None:
        import h5py
        if _STD_RESNAMES is None:
            _load_std_resnames()
        _HF = h5py.File(_H5PATH, "r")
    return _HF


def parse_ckey(ckey: str):
    """'B/PDB/rec:model/lig:model[:idx]' -> (receptor_key, ligand_key)."""
    parts = str(ckey).strip().split("/")
    return "/".join(parts[:3]), f"{parts[0]}/{parts[1]}/{parts[3]}"


def load_h5_receptor(key: str):
    """Return (X[N,3], res_of_atom[N], n_res, pdb_seq) for a receptor structure."""
    cached = _RECCACHE.get(key)
    if cached is not None:
        return cached
    g = _get_hf()[f"data/structures/{key}"]
    X = np.asarray(g["X"], dtype=np.float64)
    n = X.shape[0]
    M = np.asarray(g["M"], dtype=np.int64) if "M" in g else np.zeros((0, 2), np.int64)
    res_of_atom = np.zeros(n, dtype=np.int64)
    if M.size:
        res_of_atom[M[:, 0]] = M[:, 1]
    n_res = int(res_of_atom.max()) + 1 if n else 0
    cls_of_atom = np.full(n, -1, dtype=np.int64)
    if "qr" in g:
        qr = np.asarray(g["qr"], dtype=np.int64)
        if qr.size:
            cls_of_atom[qr[:, 0]] = qr[:, 1]
    seq_arr = np.full(n_res, "X", dtype="U1")
    if n_res:
        uniq, first = np.unique(res_of_atom, return_index=True)  # first atom per residue
        for u, ci in zip(uniq.tolist(), cls_of_atom[first].tolist()):
            rn = _STD_RESNAMES[ci] if 0 <= ci < len(_STD_RESNAMES) else "X"
            seq_arr[u] = THREE_TO_ONE.get(rn, "X")
    out = (X, res_of_atom, n_res, "".join(seq_arr.tolist()))
    _RECCACHE[key] = out
    return out


def load_h5_ligand_xyz(key: str) -> np.ndarray:
    hf = _get_hf()
    p = f"data/structures/{key}"
    if p not in hf:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(hf[p]["X"], dtype=np.float64)


def get_atoms(row):
    """Return (rec_xyz, rec_rid, pdb_seq, lig_xyz, n_res) or None if unavailable."""
    if _H5PATH:
        rk = row.get("receptor_key")
        lk = row.get("ligand_key")
        if not rk or not lk:
            rk, lk = parse_ckey(row["ckey"])
        try:
            X, res_of_atom, n_res, seq = load_h5_receptor(rk)
        except KeyError:
            return None
        lig = load_h5_ligand_xyz(lk)
        return X, res_of_atom, seq, lig, n_res
    path = _INDEX.get(ckey_to_filename(row["ckey"]))
    if path is None:
        return None
    rec_chain = receptor_chain_char(row.get("receptor_key") or row["ckey"])
    return parse_complex(path, rec_chain)


def process_cluster(detail_csv, cutoff):
    """Return (out_rows, stats) for one cluster detail CSV."""
    with open(detail_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], {"clusters": 0, "complexes": 0, "missing": 0, "empty": 0}

    cluster_id = rows[0]["cluster_name"]
    ref_id = rows[0]["ref_id"]
    ref_seq = rows[0]["ref_seq"]
    L = len(ref_seq)

    pockets: "OrderedDict[int, np.ndarray]" = OrderedDict()
    n_complex = n_missing = n_empty = 0

    for r in rows:
        pid = int(r["pocket_id"])
        acc = pockets.get(pid)
        if acc is None:
            acc = np.zeros(L, dtype=np.int8)
            pockets[pid] = acc
        n_complex += 1
        atoms = get_atoms(r)
        if atoms is None:
            n_missing += 1
            continue
        rec_xyz, rec_rid, pdb_seq, lig_xyz, n_res = atoms
        if n_res == 0:
            n_empty += 1
            continue
        pdb_label = contact_residues(rec_xyz, rec_rid, lig_xyz, n_res, cutoff)
        if not pdb_label.any():
            continue
        ref_label = map_pdb_label_to_ref(pdb_label, pdb_seq, ref_seq)
        np.maximum(acc, ref_label, out=acc)  # union / OR

    pocket_sum = len(pockets)
    labels = list(pockets.values())
    overlap = False
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if np.any(labels[i] & labels[j]):
                overlap = True
                break
        if overlap:
            break

    out_rows = []
    for pid in sorted(pockets):
        lab = pockets[pid]
        out_rows.append({
            "cluster_id": cluster_id,
            "pocket_index": pid,
            "pocket_sum": pocket_sum,
            "ref_id": ref_id,
            "ref_seq": ref_seq,
            "pocket_label": "".join("1" if v else "0" for v in lab.tolist()),
            "overlap": bool(overlap),
        })
    stats = {"clusters": 1, "complexes": n_complex, "missing": n_missing, "empty": n_empty}
    return out_rows, stats


def _work(detail_csv):
    try:
        return detail_csv, process_cluster(detail_csv, _CUTOFF), None
    except Exception as exc:  # keep the pool alive on a single bad cluster
        return detail_csv, ([], {}), f"{type(exc).__name__}: {exc}"


def build_complex_index(dirs) -> dict[str, str]:
    """Map complex pdb filename -> full path; earlier dirs win (per-job first)."""
    index: dict[str, str] = {}
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[WARN] complex dir not found: {d}", file=sys.stderr)
            continue
        with os.scandir(d) as it:
            for entry in it:
                if entry.name.endswith(".pdb") and entry.name not in index:
                    index[entry.name] = entry.path
    return index


def collect_detail_csvs(args) -> list[str]:
    files: list[str] = []
    if args.details:
        files.extend(args.details)
    for d in args.details_dir:
        files.extend(sorted(glob.glob(os.path.join(d, "*_pocket_interactions.csv"))))
    files = list(dict.fromkeys(files))
    if args.shard:
        i, n = args.shard
        files = files[i::n]
    if args.limit:
        files = files[: args.limit]
    return files


def parse_shard(value):
    if not value:
        return None
    i, n = value.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise argparse.ArgumentTypeError(f"--shard i/N needs 0<=i<N, got {value}")
    return (i, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--details-dir", action="append", default=[],
                   help="Directory of *_pocket_interactions.csv (repeatable).")
    p.add_argument("--details", nargs="*", default=[],
                   help="Explicit detail CSV files.")
    p.add_argument("--complex-pdb-dir", action="append", default=[],
                   help="Directory with cached complex PDBs (repeatable; earlier wins).")
    p.add_argument("--cache-root", default=None,
                   help="Root under which */complex_pdb dirs are globbed and added.")
    p.add_argument("--h5", default=None,
                   help="Source receptor+ligand atoms from this H5 (uniform, no complex-pdb needed).")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--cutoff", type=float, default=3.5, help="Contact distance in Angstrom.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", type=parse_shard, default=None,
                   help="Process only shard i of N detail files (e.g. 3/100), for SLURM arrays.")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N clusters (testing).")
    return p.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()

    detail_csvs = collect_detail_csvs(args)
    if not detail_csvs:
        raise SystemExit("No detail CSVs found (use --details-dir or --details)")

    index: dict[str, str] = {}
    if args.h5:
        if not os.path.isfile(args.h5):
            raise SystemExit(f"--h5 not found: {args.h5}")
        print(f"[INFO] {len(detail_csvs)} clusters, atom source = H5 {args.h5}, "
              f"cutoff={args.cutoff} A, workers={args.workers}", flush=True)
    else:
        complex_dirs = list(args.complex_pdb_dir)
        if args.cache_root:
            complex_dirs.extend(sorted(glob.glob(os.path.join(args.cache_root, "*", "complex_pdb"))))
        if not complex_dirs:
            raise SystemExit("Provide --h5, or --complex-pdb-dir/--cache-root")
        print(f"[INFO] {len(detail_csvs)} clusters, building complex index from {len(complex_dirs)} dir(s) ...",
              flush=True)
        index = build_complex_index(complex_dirs)
        print(f"[INFO] complex index: {len(index)} pdbs, cutoff={args.cutoff} A, workers={args.workers}",
              flush=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    tot = {"clusters": 0, "complexes": 0, "missing": 0, "empty": 0}
    n_pockets = 0
    n_err = 0

    with open(args.out_csv, "w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUT_COLUMNS)
        writer.writeheader()

        if args.workers <= 1:
            _init_worker(index, args.cutoff, args.h5)
            iterator = (_work(fp) for fp in detail_csvs)
            for done, (fp, (out_rows, stats), err) in enumerate(iterator, 1):
                n_err += _consume(writer, out_rows, stats, err, fp, tot)
                n_pockets += len(out_rows)
                if done % 200 == 0:
                    print(f"[INFO] {done}/{len(detail_csvs)} clusters", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_init_worker,
                                     initargs=(index, args.cutoff, args.h5)) as ex:
                futures = {ex.submit(_work, fp): fp for fp in detail_csvs}
                for done, fut in enumerate(as_completed(futures), 1):
                    fp, (out_rows, stats), err = fut.result()
                    n_err += _consume(writer, out_rows, stats, err, fp, tot)
                    n_pockets += len(out_rows)
                    if done % 200 == 0:
                        print(f"[INFO] {done}/{len(detail_csvs)} clusters", flush=True)

    dt = time.perf_counter() - t0
    print(f"[DONE] {tot['clusters']} clusters, {n_pockets} pocket rows, "
          f"{tot['complexes']} complexes, missing_pdb={tot['missing']}, "
          f"empty_receptor={tot['empty']}, errors={n_err}, {dt:.1f}s -> {args.out_csv}",
          flush=True)


def _consume(writer, out_rows, stats, err, fp, tot) -> int:
    if err is not None:
        print(f"[ERROR] {os.path.basename(fp)}: {err}", file=sys.stderr, flush=True)
        return 1
    for k in tot:
        tot[k] += stats.get(k, 0)
    for row in out_rows:
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    main()
