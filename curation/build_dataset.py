#!/usr/bin/env python3
"""Build a NEW structure/contact H5 from biological assemblies or aligned structures.

python curation/build_dataset.py --pdb-glob '/data/assemblies/*.pdb.gz' --out /data/new.h5
The released H5 and its metadata order are authoritative for frozen split masks.
Building from a different input set does not recreate that historical ordering.
"""
import argparse
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import re
# import sys
import h5py
import numpy as np
import torch as pt
from glob import glob
from tqdm import tqdm
import os
from typing import Callable, Dict, List, Optional, Any

from src.structure import (
    clean_structure,
    tag_hetatm_chains,
    split_by_chain,
    filter_non_atomic_subunits,
    remove_duplicate_tagged_subunits,
)
from src.data_encoding import (
    config_encoding,
    encode_structure,
    encode_features,
    extract_topology,
    extract_all_contacts,

)
from src.dataset import StructuresDataset, save_data

from curation.settings import config_data
import pickle

pt.multiprocessing.set_sharing_strategy("file_system")
global saved_chains

# ---------------------------------------------------------------------------
# Switch data source: set ``config_dataset["source"]`` to ``bio_assembly`` or ``aligned_clean``,
# or adjust glob / sample_ids in the corresponding getter; no need to touch the main loop.
# ---------------------------------------------------------------------------

config_dataset = {
    "source": "bio_assembly",  # "bio_assembly" | "aligned_clean"
    "r_thr": 3.5,
    "max_num_atoms": 50000,
    "max_num_nn": 64,
    "molecule_ids": np.array(
        [
            "GLU",
            "LEU",
            "ALA",
            "ASP",
            "SER",
            "VAL",
            "GLY",
            "THR",
            "ARG",
            "PHE",
            "TYR",
            "ILE",
            "PRO",
            "ASN",
            "LYS",
            "GLN",
            "HIS",
            "TRP",
            "MET",
            "CYS",
            "A",
            "U",
            "G",
            "C",
            "DA",
            "DT",
            "DG",
            "DC",
            "MG",
            "ZN",
            "CA",
            "MN",
            "CD",
            "CU",
            "FE",
            "FE2",
            "NI",
            "SR",
            "CO",
            "HG",
            "SF4",
            "F3S",
            "FES",
            "ligand",
        ]
    ),
    # --- bio_assembly ---
    "pdb_glob": None,
    # --- aligned_clean ---
    "aligned_root": None,
    "aligned_glob": "*/*_clean.pdb",
    # If set, only process these sample_ids (folder names, e.g. A71EV2A-x0351a); None means all
    "sample_ids": None,
    # Alternatively, override the file list directly (highest priority)
    "pdb_filepaths": None,
    "dataset_filepath": None,
}


def get_bio_assembly_file_list_and_naming(
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
  Original bio assembly pipeline: glob ``*.pdb.gz``, H5 key is ``{pdb[1:3]}/{PDB}/{chain}``.

  Returns
  -------
  dict with keys:
    pdb_filepaths, parse_record_id, make_structure_key, source_name
    """
    cfg = config or config_dataset
    if cfg.get("pdb_filepaths"):
        paths = sorted(cfg["pdb_filepaths"])
    else:
        pattern = cfg.get("pdb_glob", "../bio_assemblies_search_1/*.pdb.gz")
        paths = sorted(glob(pattern))

    def parse_record_id(pdb_filepath: str) -> str:
        filename = os.path.basename(pdb_filepath)
        m = re.match(r"([A-Za-z0-9_+]+)\.pdb\.gz", filename)
        if not m:
            raise ValueError(f"cannot parse bio assembly filename: {filename}")
        return m.group(1).split("_")[0]

    def make_structure_key(record_id: str, chain_id: str) -> str:
        rid = record_id.upper()
        return f"{rid[1:3]}/{rid}/{chain_id}"

    return {
        "pdb_filepaths": paths,
        "parse_record_id": parse_record_id,
        "make_structure_key": make_structure_key,
        "source_name": "bio_assembly",
    }


def get_aligned_clean_file_list_and_naming(
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
  aligned_files pipeline: ``{root}/{sample_id}/{sample_id}_clean.pdb``.

  H5 key is ``{sample_id[1:3]}/{sample_id}/{chain}`` (sample_id keeps its case, e.g. A71EV2A-x0351a).
  Combined with Dataset.select_by_sid, the sid is roughly ``{sample_id}_{chain}``.

  Returns
  -------
  dict with keys:
    pdb_filepaths, parse_record_id, make_structure_key, source_name
    """
    cfg = config or config_dataset
    if cfg.get("pdb_filepaths"):
        paths = sorted(cfg["pdb_filepaths"])
    else:
        root = os.path.abspath(
            cfg["aligned_root"]
        )
        pattern = os.path.join(root, cfg.get("aligned_glob", "*/*_clean.pdb"))
        paths = sorted(glob(pattern))

    allow: Optional[set] = None
    if cfg.get("sample_ids"):
        allow = {str(s) for s in cfg["sample_ids"]}

    if allow is not None:
        paths = [
            p
            for p in paths
            if os.path.basename(os.path.dirname(os.path.abspath(p))) in allow
        ]

    def parse_record_id(pdb_filepath: str) -> str:
        # .../A71EV2A-x0351a/A71EV2A-x0351a_clean.pdb -> A71EV2A-x0351a
        parent = os.path.basename(os.path.dirname(os.path.abspath(pdb_filepath)))
        base = os.path.basename(pdb_filepath)
        expected = f"{parent}_clean.pdb"
        if base != expected and not base.endswith("_clean.pdb"):
            raise ValueError(
                f"aligned clean pdb name mismatch: {pdb_filepath} (expected *{{sample_id}}_clean.pdb)"
            )
        return parent

    def make_structure_key(record_id: str, chain_id: str) -> str:
        rid = str(record_id)
        bucket = rid[1:3] if len(rid) >= 3 else rid
        return f"{bucket}/{rid}/{chain_id}"

    return {
        "pdb_filepaths": paths,
        "parse_record_id": parse_record_id,
        "make_structure_key": make_structure_key,
        "source_name": "aligned_clean",
    }


SOURCE_GETTERS = {
    "bio_assembly": get_bio_assembly_file_list_and_naming,
    "aligned_clean": get_aligned_clean_file_list_and_naming,
}


def resolve_source_batch(config: Optional[dict] = None) -> Dict[str, Any]:
    cfg = config or config_dataset
    name = cfg.get("source", "bio_assembly")
    if name not in SOURCE_GETTERS:
        raise KeyError(f"unknown source {name!r}; choose from {list(SOURCE_GETTERS)}")
    return SOURCE_GETTERS[name](cfg)


def load_interface_labels(contact, t0, t1_l):
    shape = tuple(contact[1]["Y_shape"])
    ids = pt.from_numpy(np.array(contact[0]["Y"]).astype(np.int64))

    y_ctc_r = pt.any((ids[:, 2].view(-1, 1) == t0), dim=1).view(-1, 1)
    y_ctc_l = pt.stack([pt.any((ids[:, 3].view(-1, 1) == t1), dim=1) for t1 in t1_l], dim=1)
    y_ctc = y_ctc_r & y_ctc_l

    y = pt.zeros((shape[0], len(t1_l)), dtype=pt.bool)
    contact_rows, output_columns = pt.where(y_ctc)
    y[ids[contact_rows, 0], output_columns] = True

    return y


def contacts_types(s0, M0, s1, M1, ids, molecule_ids, ex_ids, device=pt.device("cpu")):
    unk_col = (molecule_ids == "ligand").argmax()
    c0 = pt.from_numpy(s0["resname"].reshape(-1, 1) == molecule_ids.reshape(1, -1)).to(device)
    c1 = pt.from_numpy(s1["resname"].reshape(-1, 1) == molecule_ids.reshape(1, -1)).to(device)
    mask_unmatched_c0 = pt.all(c0 == False, dim=1).to(device)
    mask_another_c0 = pt.from_numpy(np.isin(s0["resname"], ex_ids)).to(device)
    c0[mask_unmatched_c0 & (~mask_another_c0), unk_col] = True

    mask_unmatched_c1 = pt.all(c1 == False, dim=1).to(device)
    mask_another_c1 = pt.from_numpy(np.isin(s1["resname"], ex_ids)).to(device)
    c1[mask_unmatched_c1 & (~mask_another_c1), unk_col] = True

    H = c1[ids[:, 1]].unsqueeze(1) & c0[ids[:, 0]].unsqueeze(2)
    rids0 = pt.where(M0[ids[:, 0]])[1]
    rids1 = pt.where(M1[ids[:, 1]])[1]

    Y = pt.zeros((M0.shape[1], M1.shape[1], H.shape[1], H.shape[2]), device=device, dtype=pt.bool)
    Y[rids0, rids1] = H

    T = pt.any(pt.any(Y, dim=1), dim=0)
    return Y, T


def pack_structure_data(X, qe, qr, qn, M, ids_topk):
    return {
        "X": X.cpu().numpy().astype(np.float32),
        "ids_topk": ids_topk.cpu().numpy().astype(np.uint16),
        "qe": pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        "qr": pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        "qn": pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        "M": pt.stack(pt.where(M), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        "qe_shape": qe.shape,
        "qr_shape": qr.shape,
        "qn_shape": qn.shape,
        "M_shape": M.shape,
    }


def pack_contacts_data(Y, T):
    return {
        "Y": pt.stack(pt.where(Y), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        "Y_shape": Y.shape,
        "ctype": T.cpu().numpy(),
    }


def chainid(subunits_name):
    l = list(subunits_name.split("/"))
    if len(l[2].split(":")) == 2:
        a = l[1].split("_")[0]
        b = l[2].split(":")[0]
        return (True, a + "_" + b)
    else:
        return (False, a + "_" + b)


def pack_dataset_items(
    subunits,
    contacts,
    molecule_ids,
    ex_ids,
    max_num_nn,
    device=pt.device("cuda"),
    config_cluster=False,
):
    structures_data = {}
    contacts_data = {}
    M = {}
    for cid0 in contacts:
        s0 = subunits[cid0]
        qe0, qr0, qn0 = encode_features(s0)
        X0, M0 = encode_structure(s0, device=device)
        M[cid0] = M0
        ids0_topk = extract_topology(X0, max_num_nn)[0]
        structures_data[cid0] = pack_structure_data(X0, qe0, qr0, qn0, M0, ids0_topk)

    for cid0 in contacts:
        s0 = subunits[cid0]
        M0 = M[cid0]
        if cid0 not in contacts_data:
            contacts_data[cid0] = {}

        for cid1 in contacts[cid0]:
            if cid1 not in contacts_data:
                contacts_data[cid1] = {}

            if cid1 not in contacts_data[cid0]:
                s1 = subunits[cid1]
                X1, M1 = encode_structure(s1, device=device)

                if (M0.shape[1] * M1.shape[1] * (molecule_ids.shape[0] ** 2)) > 2e9:
                    ctc_ids = contacts[cid0][cid1]["ids"].cpu()
                    Y, T = contacts_types(
                        s0, M0.cpu(), s1, M1.cpu(), ctc_ids, molecule_ids, ex_ids, device=pt.device("cpu")
                    )
                else:
                    ctc_ids = contacts[cid0][cid1]["ids"].to(device)
                    Y, T = contacts_types(
                        s0, M0.to(device), s1, M1.to(device), ctc_ids, molecule_ids, ex_ids, device=device
                    )

                if pt.any(Y):
                    contacts_data[cid0][cid1] = pack_contacts_data(Y, T)
                    contacts_data[cid1][cid0] = pack_contacts_data(
                        Y.permute(1, 0, 3, 2), T.transpose(0, 1)
                    )
                pt.cuda.empty_cache()

    return structures_data, contacts_data


def store_dataset_items(
    hf,
    record_id: str,
    structures_data,
    contacts_data,
    t0,
    t1_l,
    make_structure_key: Callable[[str, str], str],
):
    metadata_l = []

    for cid0 in contacts_data:
        key = make_structure_key(record_id, cid0)
        hgrp = hf.create_group(f"data/structures/{key}")
        save_data(hgrp, attrs=structures_data[cid0][1], **structures_data[cid0][0])

        for cid1 in contacts_data[cid0]:
            ckey = f"{key}/{cid1}"
            hgrp = hf.create_group(f"data/contacts/{ckey}")
            save_data(hgrp, attrs=contacts_data[cid0][cid1][1], **contacts_data[cid0][cid1][0])

            metadata_l.append(
                {
                    "key": key,
                    "size": (np.max(structures_data[cid0][0]["M"], axis=0) + 1).astype(int),
                    "ckey": ckey,
                    "ctype": contacts_data[cid0][cid1][1]["ctype"],
                    "record_id": record_id,
                }
            )

    return metadata_l


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=tuple(SOURCE_GETTERS), default='bio_assembly')
    parser.add_argument('--pdb-glob', help='Quoted glob for biological-assembly *.pdb.gz inputs')
    parser.add_argument('--aligned-root', help='Input root when --source aligned_clean')
    parser.add_argument('--aligned-glob', default='*/*_clean.pdb')
    parser.add_argument('--out', required=True, help='New H5 output; existing files are rejected')
    parser.add_argument('--contact-distance', type=float, default=config_dataset['r_thr'])
    parser.add_argument('--max-atoms', type=int, default=config_dataset['max_num_atoms'])
    parser.add_argument('--max-nn', type=int, default=config_dataset['max_num_nn'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if pt.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.source == 'bio_assembly' and not args.pdb_glob:
        parser.error('--pdb-glob is required for biological assemblies')
    if args.source == 'aligned_clean' and not args.aligned_root:
        parser.error('--aligned-root is required for aligned structures')
    if args.workers < 0 or args.limit < 0 or args.contact_distance <= 0 or args.max_atoms <= 0 or args.max_nn <= 0:
        parser.error('Invalid workers, limit, distance or size setting')
    output = Path(args.out).expanduser().resolve()
    if output.exists(): parser.error(f'Refusing to overwrite existing H5: {output}')
    config_dataset.update(source=args.source, pdb_glob=args.pdb_glob,
        aligned_root=args.aligned_root, aligned_glob=args.aligned_glob,
        dataset_filepath=str(output), r_thr=args.contact_distance,
        max_num_atoms=args.max_atoms, max_num_nn=args.max_nn)
    batch = resolve_source_batch(config_dataset)
    pdb_filepaths = batch["pdb_filepaths"]
    if args.limit: pdb_filepaths = pdb_filepaths[:args.limit]
    if not pdb_filepaths: parser.error('No structures matched the input selection')
    parse_record_id = batch["parse_record_id"]
    make_structure_key = batch["make_structure_key"]

    print(f"source={batch['source_name']}  n_files={len(pdb_filepaths)}")
    if pdb_filepaths:
        print(f"  first: {pdb_filepaths[0]}")
        print(f"  example record_id: {parse_record_id(pdb_filepaths[0])}")
        print(f"  example key: {make_structure_key(parse_record_id(pdb_filepaths[0]), 'A')}")

    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=False)
    loader_options = dict(batch_size=None, shuffle=False, num_workers=args.workers, pin_memory=False)
    if args.workers: loader_options['prefetch_factor'] = 4
    dataloader = pt.utils.data.DataLoader(dataset, **loader_options)

    device = pt.device(args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config_dataset["dataset_filepath"], "x", libver="latest") as hf:
        print("folder_length", len(dataset))
        saved_chains = 0
        mids = config_dataset["molecule_ids"].astype(np.dtype("U"))
        print("mids", mids)
        l_types, r_types_l = config_data["l_types"], config_data["r_types"]
        crystal_l = config_data["ex_types"]
        t1_l = [pt.arange(mids.shape[0])]
        t0 = pt.arange(mids.shape[0])
        t0 = pt.from_numpy(np.where(np.isin(mids, l_types))[0])
        t1_l = [pt.from_numpy(np.where(np.isin(mids, r_types))[0]) for r_types in r_types_l]

        for key in config_encoding:
            hf[f"metadata/{key}"] = config_encoding[key].astype('S')

        hf["metadata/mids"] = config_dataset["molecule_ids"].astype('S')
        hf.attrs["source"] = batch["source_name"]
        hf.attrs["contact_distance_angstrom"] = config_dataset['r_thr']
        hf.attrs["max_num_atoms"] = config_dataset['max_num_atoms']
        hf.attrs["max_num_nn"] = config_dataset['max_num_nn']

        metadata_l = []
        pbar = tqdm(dataloader)
        with_contacts_num = 0
        contacts_chains = []
        structure_exceeded = 0
        failed = 0
        for structure, pdb_filepath in pbar:
            try:
                if structure is None:
                    print("structure is None", pdb_filepath)
                    continue

                record_id = parse_record_id(pdb_filepath)

                if structure["xyz"].shape[0] >= config_dataset["max_num_atoms"]:
                    structure_exceeded += 1
                    continue

                structure = clean_structure(structure)
                structure = tag_hetatm_chains(structure)
                subunits = split_by_chain(structure)
                subunits = filter_non_atomic_subunits(subunits)
                subunits = remove_duplicate_tagged_subunits(subunits)
                contacts = extract_all_contacts(subunits, config_dataset["r_thr"], device=device)

                structures_data, contacts_data = pack_dataset_items(
                    subunits,
                    contacts,
                    config_dataset["molecule_ids"],
                    crystal_l,
                    config_dataset["max_num_nn"],
                    device=device,
                )

                metadata = store_dataset_items(
                    hf,
                    record_id,
                    structures_data,
                    contacts_data,
                    t0,
                    t1_l,
                    make_structure_key=make_structure_key,
                )
                chains_keys = set(m["key"] for m in metadata)
                saved_chains += len(chains_keys)
                metadata_l.extend(metadata)

                if metadata_l:
                    pbar.set_description(f"{metadata_l[-1]['key']}: {metadata_l[-1]['size']}")
            except Exception as e:
                failed += 1
                print("error", pdb_filepath, e)
                continue

        hf["metadata/keys"] = np.array([m["key"] for m in metadata_l]).astype('S')
        hf["metadata/sizes"] = np.array([m["size"] for m in metadata_l])
        hf["metadata/ckeys"] = np.array([m["ckey"] for m in metadata_l]).astype('S')
        hf["metadata/ctypes"] = (np.stack(
            np.where(np.array([m["ctype"] for m in metadata_l])), axis=1
        ).astype(np.uint32) if metadata_l else np.empty((0, 3), dtype=np.uint32))
        print("saved_chains", saved_chains)
        print("structure_exceeded", structure_exceeded)
        print("failed", failed)

    if not metadata_l:
        raise SystemExit('No residue contacts were written; inspect the inputs and size/contact settings')
    if failed:
        raise SystemExit(f"Dataset build had {failed} failed input structures; inspect the log")
