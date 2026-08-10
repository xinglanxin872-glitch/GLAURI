#!/usr/bin/env python3
"""Slice one real cluster out of the full training H5 into a tiny H5 for smoke-testing
the step1/step2/step3 colabel pocket pipeline, without touching the 124GB source file.

Row-filters straight from the original metadata rather than reconstructing anything by
hand -- the one subtlety: ``metadata/ctypes`` is NOT row-aligned with
``metadata/keys``/``ckeys``/``sizes``. It's a separate (N,3) index table whose column 0 is
a row index *into* ``keys`` (consumed by ``src/dataset.py``'s ``select_by_interface_types``
to build the structure-selection mask). Subsetting it needs both a row filter AND a remap
of column 0 to the new 0-based subset indices -- copying it verbatim like the small
vocabulary tables (``std_elements``/``std_names``/``std_resnames``/``mids``, which *are*
safe to copy as-is) silently produces an empty structure-selection mask downstream.

Usage:
    conda activate boltz
    python3 extract_smoke_subset.py                       # default cluster EO/3EO0/B:0
    python3 extract_smoke_subset.py --cluster "XX/1XXX/A:0"
"""
from __future__ import annotations

import argparse
import ast
import os
import pickle

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

SRC_H5 = os.environ.get("SMOKE_SRC_H5", "/home/lanxin/data/Interface_prediction/bio_251210.h5")
SRC_CLUSTER_SUMMARY = os.environ.get(
    "SMOKE_SRC_CLUSTER_SUMMARY", "/home/lanxin/data/Interface_prediction/colabel/all_cluster_summary.csv"
)
SRC_ALL_SEQ_DICT = os.environ.get(
    "SMOKE_SRC_ALL_SEQ_DICT", "/home/lanxin/data/Interface_prediction/cluster/all_seq_dict.pkl"
)


def chainid_to_all_seq_key(chainid: str) -> str:
    parts = chainid.strip().split("/")
    sample, chain_part = parts[1], parts[2]
    chain = chain_part.split(":")[0]
    return f"{sample.upper()}_{chain}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", default="EO/3EO0/B:0", help="cluster_names value from all_cluster_summary.csv")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--force", action="store_true", help="re-extract even if the output h5 already exists")
    args = ap.parse_args()

    out_h5 = os.path.join(args.out_dir, "smoke_bio.h5")
    if os.path.isfile(out_h5) and not args.force:
        print(f"[SKIP] {out_h5} already exists (use --force to re-extract)")
        return

    df = pd.read_csv(SRC_CLUSTER_SUMMARY, index_col="cluster_names")
    df["seq_name"] = df["seq_name"].apply(ast.literal_eval)
    row = df.loc[args.cluster]
    target_chains = list(row["seq_name"])
    print("[INFO] cluster", args.cluster, "members:", target_chains)

    with h5py.File(SRC_H5, "r") as hf:
        meta = hf["metadata"]
        print("[INFO] loading metadata/keys + ckeys + sizes (full, small 1-D arrays)...")
        keys = np.array(meta["keys"]).astype(str)
        ckeys = np.array(meta["ckeys"]).astype(str)
        sizes = np.array(meta["sizes"])
        print("[INFO] loading metadata/ctypes (bigger but still metadata-only)...")
        ctypes = np.array(meta["ctypes"])
        std_elements = meta["std_elements"][()]
        std_names = meta["std_names"][()]
        std_resnames = meta["std_resnames"][()]
        mids = meta["mids"][()]

        sel_mask = np.isin(keys, target_chains)
        sel_idx = np.where(sel_mask)[0]
        print(f"[INFO] selected {len(sel_idx)} metadata rows for {len(target_chains)} target chains")
        if len(sel_idx) == 0:
            raise SystemExit(f"no metadata rows found for cluster {args.cluster!r} members -- wrong cluster name?")

        sub_keys = keys[sel_idx]
        sub_ckeys = ckeys[sel_idx]
        sub_sizes = sizes[sel_idx]

        # remap ctypes[:,0] (row index into keys) -> new subset row index
        old_to_new = {int(old): new for new, old in enumerate(sel_idx)}
        ct_mask = np.isin(ctypes[:, 0], sel_idx)
        sub_ctypes = ctypes[ct_mask].copy()
        sub_ctypes[:, 0] = np.array([old_to_new[int(v)] for v in sub_ctypes[:, 0]], dtype=sub_ctypes.dtype)
        print(f"[INFO] ctypes: {len(sub_ctypes)} rows kept (of {len(ctypes)})")

        # every structure key referenced: the receptor keys themselves + every ligand-partner
        # half of every selected ckey (parse_ckey convention: bucket/pdb/rec_chain/lig_chain)
        needed_structs = set(sub_keys.tolist())
        for ck in sub_ckeys:
            parts = ck.split("/")
            receptor_key = "/".join(parts[:3])
            ligand_key = f"{parts[0]}/{parts[1]}/{parts[3]}"
            needed_structs.add(receptor_key)
            needed_structs.add(ligand_key)
        print(f"[INFO] {len(needed_structs)} unique structure keys to copy")

        needed_contacts = set(sub_ckeys.tolist())
        print(f"[INFO] {len(needed_contacts)} unique contact (ckey) groups to copy")

        os.makedirs(args.out_dir, exist_ok=True)
        with h5py.File(out_h5, "w") as out:
            mg = out.create_group("metadata")
            mg.create_dataset("keys", data=sub_keys.astype("S"))
            mg.create_dataset("ckeys", data=sub_ckeys.astype("S"))
            mg.create_dataset("sizes", data=sub_sizes)
            mg.create_dataset("ctypes", data=sub_ctypes)
            mg.create_dataset("std_elements", data=std_elements)
            mg.create_dataset("std_names", data=std_names)
            mg.create_dataset("std_resnames", data=std_resnames)
            mg.create_dataset("mids", data=mids)

            out.create_group("data/structures")
            n_missing = 0
            for key in sorted(needed_structs):
                src_path = f"data/structures/{key}"
                if src_path not in hf:
                    n_missing += 1
                    continue
                hf.copy(src_path, out, name=f"data/structures/{key}")
            print(f"[INFO] copied structures, missing={n_missing}")

            out.create_group("data/contacts")
            n_missing_c = 0
            for ck in sorted(needed_contacts):
                src_path = f"data/contacts/{ck}"
                if src_path not in hf:
                    n_missing_c += 1
                    continue
                hf.copy(src_path, out, name=f"data/contacts/{ck}")
            print(f"[INFO] copied contacts, missing={n_missing_c}")

    print("[DONE] wrote", out_h5, os.path.getsize(out_h5) / 1e6, "MB")

    sub_df = df.loc[[args.cluster]].reset_index().rename(columns={"index": "cluster_names"})
    sub_df.to_csv(os.path.join(args.out_dir, "smoke_cluster_summary.csv"), index=False)
    print("[DONE] wrote smoke_cluster_summary.csv")

    with open(SRC_ALL_SEQ_DICT, "rb") as f:
        all_seq_dict = pickle.load(f)
    sub_seq_dict = {}
    for c in target_chains:
        k = chainid_to_all_seq_key(c)
        if k in all_seq_dict:
            sub_seq_dict[k] = all_seq_dict[k]
        else:
            print("[WARN] missing all_seq_dict entry for", c, "->", k)
    with open(os.path.join(args.out_dir, "smoke_all_seq_dict.pkl"), "wb") as f:
        pickle.dump(sub_seq_dict, f)
    print("[DONE] wrote smoke_all_seq_dict.pkl,", len(sub_seq_dict), "entries")


if __name__ == "__main__":
    main()
