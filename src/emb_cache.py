"""Compatibility API for the versioned float32 ESM provider."""
import os
from pathlib import Path
from src.embeddings import EmbeddingProvider

CACHE_DIR = os.environ.get('GLAURI_EMB_CACHE_DIR', str(Path(__file__).resolve().parents[1]/'benchmark/results/_emb_cache_v2'))

class LazyPLMPack:
    def __init__(self, device, plm_type):
        if plm_type != 'esm3':raise ValueError('This release uses ESM3')
        self.provider = EmbeddingProvider(device, backend='math', directory=CACHE_DIR)

    def get(self):
        return self.provider


def get_cached_embedding(seq, device, plm_type, plm_pack):
    if plm_type != 'esm3':raise ValueError('This release uses ESM3')
    if isinstance(plm_pack, LazyPLMPack):return plm_pack.provider.get(seq)
    if isinstance(plm_pack, EmbeddingProvider):return plm_pack.get(seq)
    raise TypeError('Use LazyPLMPack or EmbeddingProvider; raw model tuples have no cache protocol')
