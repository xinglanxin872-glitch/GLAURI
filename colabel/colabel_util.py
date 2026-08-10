"""
H5 structure -> 单链 / 复合物 PDB 写出（USalign 输入 + complex-PDB 缓存）。

历史上这个文件是"H5 -> prepare pdbqt -> jdock/rosetta 打分 -> 归一化残基向量"的完整
打分流水线（JdockPipelineConfig/RosettaPipelineConfig + 两个 Scorer 类）；打分部分随
step1_batch_cluster_pocket_labels.py 砍掉打分逻辑一起废弃了（下游标签改由
step2_pocket_binary_labels.py 自己重新算 3.5A 几何接触，不再需要对接/能量打分）。

现在只保留 step1_batch_cluster_pocket_labels.py（step 1）仍然依赖的那一小块：
  * write_structure_key_pdb -- 写单链 PDB，供 USalign 输入
  * write_ckey_complex_pdb  -- 写 receptor+ligand 复合物 PDB，缓存供
    step2_pocket_binary_labels.py（step 2）的 --complex-pdb-dir 模式消费
  * parse_ckey              -- ckey 字符串 -> (receptor_key, ligand_key)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_encoding import categ_to_resnames, std_elements, std_names, std_resnames  # noqa: E402
from src.structure_io import save_pdb  # noqa: E402

_PROTEIN_RES = set(categ_to_resnames["protein"])


def parse_ckey(ckey: str) -> Tuple[str, str]:
    """
    ``HL/3HLL/A:0/A:0:2`` -> receptor ``HL/3HLL/A:0``, ligand ``HL/3HLL/A:0:2``.
    """
    parts = str(ckey).strip().split("/")
    if len(parts) < 4:
        raise ValueError(f"invalid ckey (need bucket/pdb/chain/partner): {ckey!r}")
    receptor_key = "/".join(parts[:3])
    ligand_key = f"{parts[0]}/{parts[1]}/{parts[3]}"
    return receptor_key, ligand_key


def _decode_sparse_labels(
    sparse: np.ndarray, n_atoms: int, vocab: np.ndarray, default: str
) -> np.ndarray:
    out = np.full(n_atoms, default, dtype=vocab.dtype)
    if sparse is None or len(sparse) == 0:
        return out
    arr = np.asarray(sparse, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"expected sparse (N,2), got {arr.shape}")
    for atom_i, class_i in arr:
        if 0 <= atom_i < n_atoms and 0 <= class_i < len(vocab):
            out[atom_i] = vocab[class_i]
    return out


def h5_group_to_subunit(hgrp: h5py.Group, chain_id: str) -> Dict[str, Dict[str, np.ndarray]]:
    """H5 structure group -> save_pdb 所需的单亚基 dict。"""
    X = np.asarray(hgrp["X"], dtype=np.float32)
    n_atoms = X.shape[0]
    if n_atoms == 0:
        raise ValueError("empty structure in h5")

    m_sparse = np.asarray(hgrp["M"], dtype=np.int64)
    res_idx = np.zeros(n_atoms, dtype=np.int32)
    for atom_i, r_i in m_sparse:
        res_idx[atom_i] = int(r_i)

    elements = _decode_sparse_labels(
        np.asarray(hgrp["qe"]) if "qe" in hgrp else None,
        n_atoms,
        std_elements,
        "C",
    )
    resnames = _decode_sparse_labels(
        np.asarray(hgrp["qr"]) if "qr" in hgrp else None,
        n_atoms,
        std_resnames,
        "UNX",
    )
    names = _decode_sparse_labels(
        np.asarray(hgrp["qn"]) if "qn" in hgrp else None,
        n_atoms,
        std_names,
        "UNK",
    )

    het_flag = np.array(
        ["A" if rn in _PROTEIN_RES else "H" for rn in resnames], dtype="U1"
    )
    resid = res_idx.astype(np.int32) + 1

    return {
        chain_id: {
            "xyz": X,
            "name": names,
            "element": elements,
            "resname": resnames,
            "resid": resid,
            "het_flag": het_flag,
            # structure_io.save_pdb writes this value into both occupancy and B-factor.
            # Rosetta ignores atoms with occupancy 0, so keep exported H5 atoms visible.
            "bfactor": np.ones(n_atoms, dtype=np.float32),
        }
    }


def write_structure_key_pdb(
    hf: h5py.File, structure_key: str, out_pdb: Union[str, Path]
) -> Path:
    """从 ``data/structures/{key}`` 写出单链 PDB（H5 已清洁，无需 altloc collapse）。"""
    hpath = f"data/structures/{structure_key}"
    if hpath not in hf:
        raise KeyError(f"missing in h5: {hpath}")
    chain_id = structure_key.rsplit("/", 1)[-1]
    subunits = h5_group_to_subunit(hf[hpath], chain_id)
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    save_pdb(subunits, str(out_pdb))
    return out_pdb


def _chain_char_from_structure_key(structure_key: str, fallback: str = "A") -> str:
    chain_token = str(structure_key).rsplit("/", 1)[-1].split(":", 1)[0]
    if chain_token:
        return chain_token[0]
    return fallback


def _unused_chain_char(preferred: str, used: Iterable[str]) -> str:
    used_set = set(used)
    if preferred and preferred not in used_set:
        return preferred[0]
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
        if c not in used_set:
            return c
    raise ValueError("cannot assign a unique one-character PDB chain id")


def write_ckey_complex_pdb(
    hf: h5py.File,
    receptor_key: str,
    ligand_key: str,
    out_pdb: Union[str, Path],
) -> Tuple[Path, str, str]:
    """写出 receptor+ligand 复合物 PDB，并保证两条链的 chain id 可区分。"""
    rec_hpath = f"data/structures/{receptor_key}"
    lig_hpath = f"data/structures/{ligand_key}"
    if rec_hpath not in hf:
        raise KeyError(f"missing in h5: {rec_hpath}")
    if lig_hpath not in hf:
        raise KeyError(f"missing in h5: {lig_hpath}")

    rec_chain = _chain_char_from_structure_key(receptor_key, "A")
    lig_preferred = _chain_char_from_structure_key(ligand_key, "B")
    lig_chain = _unused_chain_char(lig_preferred, [rec_chain])

    subunits: Dict[str, Dict[str, np.ndarray]] = {}
    subunits.update(h5_group_to_subunit(hf[rec_hpath], rec_chain))
    subunits.update(h5_group_to_subunit(hf[lig_hpath], lig_chain))

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    save_pdb(subunits, str(out_pdb))
    return out_pdb, rec_chain, lig_chain
