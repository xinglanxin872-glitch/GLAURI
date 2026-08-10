#!/usr/bin/env python3
"""Run the GLAURI interface model on a single protein chain and write four
B-factor-colored output PDBs, one per interface-type channel (protein/nn/ion/ligand).

Reuses the model-loading logic already shipped in benchmark/model_benchmark_eval.py
(same checkpoint format, same config-snapshot resolution) rather than re-implementing it,
and the structure/feature pipeline in src/utils_eval.py used by every other eval script
in this repo (case_study/*, benchmark/*_eval.py).

Output convention (matches case_study/nucleic_6du4's established colorpdb scheme):
  - the scored chain's ATOM residues get their B-factor set to the model's per-residue
    sigmoid probability for that channel, in [0, 1]
  - everything else (other chains, HETATM records, and -- rare, but possible -- residues
    of the scored chain the CA_BB preprocessing had to drop for missing backbone atoms)
    gets B-factor = -1.00, so PyMOL/ChimeraX can grey it out with `color grey80, b < 0`

Usage:
    conda activate boltz
    python3 infer_one.py structure.pdb
        # single polymer chain auto-detected; writes structure-{protein,nn,ion,ligand}.pdb
        # next to the input file
    python3 infer_one.py structure.pdb --chain A --out-prefix out/mystruct
        # writes out/mystruct-{protein,nn,ion,ligand}.pdb
    python3 infer_one.py structure.cif --chain B --model-dir model/GLAURI --device cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", help="Input single-chain structure file (.pdb or .cif)")
    ap.add_argument("--chain", default=None,
                     help="Chain ID to score (auto-detected if the file has exactly one polymer chain)")
    ap.add_argument("--model-dir", default=None,
                     help="Checkpoint dir (default: <repo>/model/GLAURI)")
    ap.add_argument("--out-prefix", default=None,
                     help="Output path prefix; writes <prefix>-{protein,nn,ion,ligand}.pdb "
                          "(default: <input path without extension>)")
    ap.add_argument("--device", default=None,
                     help="torch device, e.g. cuda:0 / cpu (default: cuda if available else cpu)")
    args = ap.parse_args()
    # Resolve every user-supplied path to absolute BEFORE importing benchmark/model_benchmark_eval
    # below -- that module chdir's the process to the repo root as an import-time side effect
    # (needed so its own relative dataset/config paths resolve), which would otherwise silently
    # break any relative path the caller passed on this script's command line.
    args.pdb = os.path.abspath(args.pdb)
    if args.model_dir:
        args.model_dir = os.path.abspath(args.model_dir)
    if args.out_prefix:
        args.out_prefix = os.path.abspath(args.out_prefix)
    return args


def detect_single_chain(pdb_path: str) -> str:
    """Return the sole polymer chain ID in a PDB, or raise if there's more than one."""
    chains = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch = line[21]
            if ch not in chains:
                chains.append(ch)
    if not chains:
        raise SystemExit(f"no ATOM records found in {pdb_path}")
    if len(chains) > 1:
        raise SystemExit(f"{pdb_path} has {len(chains)} polymer chains {chains} -- pass --chain to pick one")
    return chains[0]


def chain_residue_order(pdb_path: str, chain: str) -> list[tuple[int, str, str]]:
    """Ordered, first-occurrence-only (resnum, icode, resname) for one chain's ATOM records."""
    out, seen = [], set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM") or line[21] != chain:
                continue
            key = (int(line[22:26]), line[26])
            if key in seen:
                continue
            seen.add(key)
            out.append((key[0], key[1], line[17:20].strip()))
    return out


def align_model_seq_to_pdb(model_seq: str, pdb_residues: list[tuple[int, str, str]],
                            three_to_one: dict) -> list[tuple[int, str] | None]:
    """Map each position of `model_seq` (CA_BB output order, backbone-incomplete residues
    already dropped) back onto (resnum, icode) in `pdb_residues` (full file-order list,
    nothing dropped). CA_BB only ever *removes* residues, never reorders/duplicates them,
    so model_seq is guaranteed a subsequence of the pdb's own sequence -- a single forward
    two-pointer scan recovers the exact mapping unambiguously.
    """
    mapping: list[tuple[int, str]] = []
    j = 0
    for aa in model_seq:
        while j < len(pdb_residues) and three_to_one.get(pdb_residues[j][2], "X") != aa:
            j += 1
        if j >= len(pdb_residues):
            raise SystemExit(
                f"could not align model output (len={len(model_seq)}) back onto the PDB's own "
                f"chain sequence (len={len(pdb_residues)}) at model position {len(mapping)} "
                f"-- CA_BB preprocessing dropped/reordered more than expected, refusing to "
                f"silently write a possibly-misaligned B-factor column"
            )
        mapping.append((pdb_residues[j][0], pdb_residues[j][1]))
        j += 1
    return mapping


def write_channel_pdb(pdb_path: str, chain: str, scores_by_res: dict[tuple[int, str], float],
                       out_path: str) -> None:
    """Copy pdb_path, setting B-factor = scores_by_res[(resnum, icode)] for the scored
    chain's ATOM lines, -1.00 for everything else (other chains, HETATM, dropped residues)."""
    with open(pdb_path) as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            score = None
            if line.startswith("ATOM") and line[21] == chain:
                score = scores_by_res.get((int(line[22:26]), line[26]))
            b = score if score is not None else -1.0
            line = line[:60] + f"{b:6.2f}" + line[66:]
        out_lines.append(line)
    with open(out_path, "w") as f:
        f.writelines(out_lines)


def main() -> None:
    args = parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import torch as pt
    from Bio.Data.IUPACData import protein_letters_3to1
    from src.utils_eval import process_structure, process_seq_batch, inference
    from src.emb_cache import LazyPLMPack, get_cached_embedding

    three_to_one = {k.upper(): v for k, v in protein_letters_3to1.items()}

    spec = importlib.util.spec_from_file_location(
        "_model_benchmark_eval", os.path.join(repo_root, "benchmark", "model_benchmark_eval.py"))
    mbe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mbe)  # NOTE: chdir's the process to repo_root -- our paths are absolute already

    if args.device:
        mbe.DEVICE = pt.device(args.device)
    device = mbe.DEVICE

    model_dir = args.model_dir or os.path.join(repo_root, "model", "GLAURI")
    print(f"[infer_one] loading model from {model_dir} (device={device})")
    model, cd, cm, frame, plm_type, fusion, need_rsa = mbe.load_model(model_dir)

    chain = args.chain or detect_single_chain(args.pdb)
    print(f"[infer_one] scoring chain {chain!r} of {args.pdb}")

    structure = process_structure(args.pdb, chainid=f"{chain}:0")
    if not isinstance(structure, dict):
        raise SystemExit(f"failed to load chain {chain!r} from {args.pdb} "
                          f"(process_structure returned {structure!r} -- check the chain ID)")

    ff = cd["features_flags"]
    pre = cd.get("preprocess") or None
    mnn = cd.get("max_num_nn", 64)
    data_list, seq = process_seq_batch(structure, ff, pre, frame, mnn, with_rsa=need_rsa)
    print(f"[infer_one] {len(seq)} residues scored (of "
          f"{len(chain_residue_order(args.pdb, chain))} total ATOM residues in chain {chain!r})")

    plm_pack = LazyPLMPack(device, plm_type) if plm_type else None
    emb = get_cached_embedding(seq, device, plm_type, plm_pack) if plm_type else None

    pc = inference(model, device, data_list, frame, mask=None, plm_embedding=emb, plm_fusion=fusion)
    # pc columns: [1-max(p), protein, nn(DNA/RNA), ion, ligand] -- see model/main.py's
    # task_names = ("protein", "nucleic_acid", "ion", "ligand"), same r_types column order
    # used throughout this repo (config_data['r_types'], colabel's --contact-class, etc.)
    channels = {"protein": pc[:, 1], "nn": pc[:, 2], "ion": pc[:, 3], "ligand": pc[:, 4]}

    pdb_residues = chain_residue_order(args.pdb, chain)
    res_mapping = align_model_seq_to_pdb(seq, pdb_residues, three_to_one)

    out_prefix = args.out_prefix or os.path.splitext(args.pdb)[0]
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for name, col in channels.items():
        scores_by_res = {res_mapping[i]: float(col[i]) for i in range(len(res_mapping))}
        out_path = f"{out_prefix}-{name}.pdb"
        write_channel_pdb(args.pdb, chain, scores_by_res, out_path)
        vals = col.tolist()
        print(f"[infer_one] wrote {out_path}  (n={len(vals)}, "
              f"mean={sum(vals)/len(vals):.3f}, max={max(vals):.3f}, "
              f"n>=0.5={sum(1 for v in vals if v >= 0.5)})")


if __name__ == "__main__":
    main()
