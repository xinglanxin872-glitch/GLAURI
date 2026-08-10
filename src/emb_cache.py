#!/usr/bin/env python3
"""Shared PLM-embedding cache (h5, keyed by sha1(seq)) for the benchmark eval scripts.

Usage:
    from src.emb_cache import get_cached_embedding, LazyPLMPack
    emb = get_cached_embedding(seq, device, plm_type, plm_pack)

Cache layout: CACHE_DIR/emb_{plm}.h5, one dataset per sequence sha1,
float16 array + attrs["seq"] for verification. Single-process append only.
"""
import os, hashlib
import h5py
import torch as pt

CACHE_DIR = os.environ.get(
    "GLAURI_EMB_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark", "results", "_emb_cache"),
)


class LazyPLMPack:
    """Load a large PLM only when an embedding is genuinely absent from cache."""

    def __init__(self, device, plm_type):
        self.device = device
        self.plm_type = plm_type
        self._pack = None

    def get(self):
        if self._pack is None:
            from src.utils_eval import get_plm_model
            self._pack = get_plm_model(self.device, self.plm_type)
            if hasattr(self._pack[0], "eval"):
                self._pack[0].eval()
        return self._pack


def seq_key(seq):
    return hashlib.sha1(seq.encode()).hexdigest()

def cache_path(plm_type):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"emb_{str(plm_type).lower()}.h5")

def get_cached_embedding(seq, device, plm_type, plm_pack):
    """Return (L, D) float32 embedding for seq; read from cache, else compute + store."""
    from src.utils_eval import get_embedding
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(plm_type)
    key = seq_key(seq)
    with h5py.File(path, "a") as hf:
        if key in hf:
            return pt.from_numpy(hf[key][:]).float()
        if isinstance(plm_pack, LazyPLMPack):
            plm_pack = plm_pack.get()
        emb = get_embedding(seq, device, plm_type, plm_pack)
        arr = emb.detach().to(pt.float16).cpu().numpy()
        ds = hf.create_dataset(key, data=arr, compression="gzip")
        ds.attrs["seq"] = seq
        return emb.float() if hasattr(emb, "float") else emb

def iter_cached(hf, plm_type=None):
    """yield (key, seq) pairs of an existing cache file (diagnostics)."""
    with h5py.File(cache_path(plm_type or "ESM2"), "r") as f:
        for k in f:
            yield k, f[k].attrs.get("seq", "")
