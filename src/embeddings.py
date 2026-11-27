"""Frozen ESM3 provider with versioned float32 caches and isolated RNG state."""
from collections import OrderedDict
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import warnings

import h5py
import numpy as np
import torch

EXPECTED_ESM_WEIGHTS = '5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9'


@lru_cache(maxsize=1)
def esm_identity():
    spec = importlib.util.find_spec('esm3lib')
    source = Path(spec.origin).parent if spec else None
    sdk_sha = hashlib.sha256()
    if source:
        for p in sorted(source.rglob('*.py')):
            sdk_sha.update(str(p.relative_to(source)).encode())
            sdk_sha.update(p.read_bytes())
    hub = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser() / 'hub')))
    model_cache = hub / 'models--EvolutionaryScale--esm3-sm-open-v1'
    ref = model_cache/'refs/main'
    weights = ([model_cache/'snapshots'/ref.read_text().strip()/'data/weights/esm3_sm_open_v1.pth']
               if ref.is_file() else [])
    weights = [p for p in weights if p.is_file()]
    weight_sha = None
    if weights:
        from src.checkpoint import sha256
        weight_sha = sha256(weights[-1])
    return dict(model='esm3-open', sdk_source_sha256=sdk_sha.hexdigest() if source else None,
                weights_sha256=weight_sha)


class EmbeddingProvider:
    def __init__(self, device, cache_file=None, backend='auto', *, cache_files=(),
                 directory=None, allow_legacy_cache=False, memory_size=128,
                 identity=None, compute=None):
        if backend not in ('math', 'auto'):
            raise ValueError('Unknown ESM attention backend')
        self.device = torch.device(device)
        self.backend = backend
        self.client = None
        self.memory = OrderedDict()
        self.memory_size = memory_size
        self.allow_legacy_cache = allow_legacy_cache
        self.compute = compute
        self.identity = dict(identity if identity is not None else esm_identity())
        self.expected_weights = EXPECTED_ESM_WEIGHTS if identity is None else None
        if self.expected_weights and self.identity.get('weights_sha256') not in (None, self.expected_weights):
            raise ValueError('Cached ESM weights differ from the frozen paper sequence model')
        self.protocol = dict(format='glauri_esm_float32_v2', **self.identity,
                             attention_backend=backend, compute_dtype='float32' if self.device.type == 'cpu' else 'bfloat16',
                             storage_dtype='float32', output_dtype='float32')
        self.protocol_sha256 = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
        self.directory = Path(directory) if directory else None
        if self.directory:
            if self.identity.get('weights_sha256') is None:
                raise ValueError('Cache the identified ESM3 model weights before enabling persistent embedding caching')
            self.directory.mkdir(parents=True, exist_ok=True)
        self.handles = []
        for path in ([cache_file] if cache_file else []) + list(cache_files):
            handle = h5py.File(path, 'r')
            saved = handle.attrs.get('glauri_embedding_protocol')
            if saved is None or json.loads(saved) != self.protocol:
                if not allow_legacy_cache:
                    handle.close()
                    self.close()
                    raise ValueError(f'Unverified/legacy embedding cache: {path}; explicitly opt in for historical replay')
                if handle.attrs.get('sdpa_backend') != backend:
                    handle.close()
                    self.close()
                    raise ValueError(f'Legacy embedding cache backend differs: {path}')
                warnings.warn(f'Explicitly reading legacy embedding cache: {path}', RuntimeWarning)
            self.handles.append(handle)
        self.cache_file = cache_file
        self.cache = self.handles[0] if self.handles else None

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles = []

    def _validate(self, value, sequence):
        value = torch.as_tensor(value).detach().cpu().float().contiguous()
        if value.shape != (len(sequence), 1536) or not torch.isfinite(value).all():
            raise ValueError('Invalid ESM3 embedding')
        return value

    def get(self, sequence, key=None):
        digest = hashlib.sha1(sequence.encode()).hexdigest()
        if digest in self.memory:
            self.memory.move_to_end(digest)
            return self.memory[digest]
        path = self.directory / f'{self.protocol_sha256}_{digest}.pt' if self.directory else None
        value = None
        if path and path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['protocol'] != self.protocol or saved['sequence'] != sequence:
                raise ValueError(f'Embedding cache identity differs: {path}')
            value = saved['embedding']
        for handle in self.handles if value is None else ():
            for name in (digest, key):
                if not name or name not in handle:
                    continue
                entry = handle[name]
                seq = entry.attrs.get('seq')
                if isinstance(seq, bytes):
                    seq = seq.decode()
                if (seq is not None and seq != sequence) or (name != digest and seq != sequence):
                    raise ValueError('Embedding sequence/key mismatch')
                if entry.dtype != np.float32 and not self.allow_legacy_cache:
                    raise ValueError('Verified caches require float32 storage')
                value = torch.from_numpy(entry[:])
                break
            if value is not None:
                break
        if value is None:
            from src.utils_eval import get_plm_model, get_embedding
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
            numpy_state, python_state = np.random.get_state(), random.getstate()
            from torch.nn.attention import sdpa_kernel, SDPBackend
            context = sdpa_kernel(SDPBackend.MATH) if self.backend == 'math' else nullcontext()
            try:
                with torch.random.fork_rng(devices=devices), torch.no_grad(), context:
                    if self.compute:
                        value = self.compute(sequence)
                    else:
                        if self.client is None:
                            self.client = get_plm_model(self.device, 'esm3')
                            self.client[0].eval()
                            for parameter in self.client[0].parameters():
                                parameter.requires_grad_(False)
                            if self.identity.get('weights_sha256') is None:
                                esm_identity.cache_clear()
                                actual_identity = esm_identity()
                                if actual_identity.get('weights_sha256') != self.expected_weights:
                                    raise ValueError('Could not identify the frozen ESM sequence weights after loading')
                                self.identity = actual_identity
                                self.protocol.update(actual_identity)
                                self.protocol_sha256 = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
                        value = get_embedding(sequence, self.device, 'esm3', self.client)
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
        value = self._validate(value, sequence)
        if path and not path.exists():
            # Single-process writer; replace makes interrupted writes invisible.
            fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
            os.close(fd)
            try:
                torch.save(dict(protocol=self.protocol, sequence=sequence, embedding=value), temp)
                os.replace(temp, path)
            finally:
                Path(temp).unlink(missing_ok=True)
        self.memory[digest] = value
        while len(self.memory) > self.memory_size:
            self.memory.popitem(last=False)
        return value
