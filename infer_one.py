#!/usr/bin/env python3
"""Run the GLAURI interface model on a single protein chain and write four
B-factor-colored output PDBs, one per interface-type channel (protein/nn/ion/ligand).

Uses src.checkpoint, src.embeddings and src.predict, shared with evaluation.

Output convention (per-residue probability output):
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
import os
import sys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", help="Input single-chain structure file (.pdb or .cif)")
    ap.add_argument("--chain", default=None,
                     help="Chain ID to score (auto-detected if the file has exactly one polymer chain)")
    ap.add_argument("--model-dir", default=None,
                     help="Checkpoint dir (default: <repo>/model/GLAURI)")
    ap.add_argument("--checkpoint", default=None,
                     help="Explicit checkpoint filename within model-dir, e.g. model_last.pt")
    ap.add_argument("--out-prefix", default=None,
                     help="Output path prefix; writes <prefix>-{protein,nn,ion,ligand}.pdb "
                          "(default: <input path without extension>)")
    ap.add_argument("--device", default=None,
                     help="torch device, e.g. cuda:0 / cpu (default: cuda if available else cpu)")
    ap.add_argument("--embedding-cache-dir", default=None, help="Versioned float32 cache directory (optional)")
    args = ap.parse_args()
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


def node_positions_to_residues(pdb_path, chain, positions, atom_name):
    """Map by exact named-atom coordinates; repeated sequence letters cannot shift scores."""
    import numpy as np
    from scipy.spatial import cKDTree
    coordinates=[];residues=[]
    for line in open(pdb_path):
        if line.startswith('ATOM') and line[21]==chain and line[12:16].strip()==atom_name:
            coordinates.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
            residues.append((int(line[22:26]),line[26]))
    if not coordinates:raise ValueError(f'No {atom_name} coordinates for chain {chain}')
    distances,indices=cKDTree(np.asarray(coordinates)).query(np.asarray(positions),k=1)
    if np.any(distances>0.002):raise ValueError('Cannot map model nodes onto named PDB atom coordinates')
    result=[residues[int(i)] for i in indices]
    if len(set(result))!=len(result):raise ValueError('Multiple model nodes map to one PDB residue')
    return result


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
    # The residue writer operates on PDB records; convert CIF/gzip inputs explicitly.
    if args.pdb.lower().endswith(('.cif','.mmcif','.gz')):
        import atexit, tempfile, gemmi
        original=args.pdb
        if args.out_prefix is None:
            base=original[:-3] if original.lower().endswith('.gz') else original
            args.out_prefix=os.path.splitext(base)[0]
        handle=tempfile.NamedTemporaryFile(suffix='.pdb',delete=False);handle.close()
        converted=gemmi.read_structure(original);converted.setup_entities()
        converted.write_minimal_pdb(handle.name)
        args.pdb=handle.name
        atexit.register(lambda: os.path.exists(handle.name) and os.unlink(handle.name))

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import torch as pt
    from src.checkpoint import load_checkpoint
    from src.embeddings import EmbeddingProvider
    from src.predict import predict_structure
    loaded = load_checkpoint(args.model_dir, args.checkpoint, args.device)
    model, cd, cm, frame, plm_type, fusion, need_rsa = loaded.legacy_tuple()
    device = next(model.parameters()).device
    model_dir = os.path.dirname(loaded.metadata['checkpoint_path'])
    provider = EmbeddingProvider(device, backend='math', directory=args.embedding_cache_dir)

    chain = args.chain or detect_single_chain(args.pdb)
    print(f"[infer_one] scoring chain {chain!r} of {args.pdb}")

    pc, seq, node_positions = predict_structure(loaded, args.pdb, chain, provider)
    print(f"[infer_one] {len(seq)} residues scored")
    # pc columns: [1-max(p), protein, nn(DNA/RNA), ion, ligand] -- see model/main.py's
    # task_names = ("protein", "nucleic_acid", "ion", "ligand"), same r_types column order
    # used throughout this repo (config_data['r_types'], colabel's --contact-class, etc.)
    channels = {"protein": pc[:, 1], "nn": pc[:, 2], "ion": pc[:, 3], "ligand": pc[:, 4]}

    anchor='CA' if cd.get('feature_protocol')=='ca_ncac_sidechain_v1' else 'N'
    res_mapping=node_positions_to_residues(args.pdb,chain,node_positions,anchor)

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
    import csv,json,hashlib
    with open(out_prefix+'-scores.csv','w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['node_index','chain','resnum','icode','protein','nn','ion','ligand'])
        for i,(resnum,icode) in enumerate(res_mapping):
            writer.writerow([i,chain,resnum,icode,*[format(float(channels[name][i]),'.17g') for name in ['protein','nn','ion','ligand']]])
    checkpoint_path=loaded.metadata['checkpoint_path']
    with open(out_prefix+'-inference.json','w') as stream:
        json.dump(dict(model_dir=model_dir,checkpoint_path=checkpoint_path,checkpoint_sha256=hashlib.sha256(open(checkpoint_path,'rb').read()).hexdigest(),
                       feature_protocol=cd.get('feature_protocol','legacy_nodes_angles_sidechain_v1_20261102'),
                       input_policy=loaded.metadata['input_policy'],embedding_protocol=provider.protocol,
                       nodes=len(res_mapping),chain=chain,scores_csv=os.path.basename(out_prefix+'-scores.csv')),stream,indent=2)


if __name__ == "__main__":
    main()
