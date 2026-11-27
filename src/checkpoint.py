"""Load paired weights/configuration without importing evaluation scripts."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

RELEASE_DIR = Path(__file__).resolve().parents[1] / 'model' / 'GLAURI'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class LoadedCheckpoint:
    model: object
    config_data: dict
    config_model: dict
    metadata: dict

    def legacy_tuple(self):
        cd, cm = self.config_data, self.config_model
        return (self.model, cd, cm, cm.get('name', 'gcp'), cd.get('plm'),
                cd.get('plm_fusion', 'concat'), cd.get('rsa_filepath') is not None)


def _historical_config(folder):
    """Explicit old snapshots only; current defaults are never a fallback."""
    standalone = folder / 'config.py'
    candidates = sorted(p for p in folder.glob('*_config*.py') if not p.name.endswith('_base_config.py'))
    source = standalone if standalone.is_file() else (candidates[-1] if candidates else None)
    if source is None:
        raise FileNotFoundError('Checkpoint configuration snapshot missing')
    bases = sorted(folder.glob('*_base_config.py'))
    if source != standalone and 'from config import' in source.read_text() and not bases:
        raise FileNotFoundError('Historical checkpoint requires its original base_config')
    from importlib import import_module
    sys.modules.setdefault('src.data_encoding_redefine', import_module('src.data_encoding'))
    previous = sys.modules.get('config')
    try:
        base = bases[-1] if bases else source
        spec = importlib.util.spec_from_file_location('config', base)
        module = importlib.util.module_from_spec(spec)
        sys.modules['config'] = module
        spec.loader.exec_module(module)
        spec = importlib.util.spec_from_file_location('_glauri_historical_config', source)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        return deepcopy(cfg.config_data), deepcopy(cfg.config_model), source
    finally:
        if previous is None:
            sys.modules.pop('config', None)
        else:
            sys.modules['config'] = previous


def load_checkpoint(model_dir=None, checkpoint=None, device=None):
    from model.model import GCPNetModel
    folder = Path(model_dir or RELEASE_DIR).expanduser().resolve()
    path = Path(checkpoint or 'model_best_auc.pt')
    path = path if path.is_absolute() else folder / path
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {path}; last/legacy weights require an explicit filename')
    source = folder / 'effective_config.json'
    if source.is_file():
        effective = json.loads(source.read_text())
        cd, cm = deepcopy(effective['config_data']), deepcopy(effective['config_model'])
    else:
        cd, cm, source = _historical_config(folder)
    protocol = cd.get('feature_protocol', 'legacy_nodes_angles_sidechain_v1_20261102')
    if cm.get('feature_protocol', protocol) != protocol:
        raise ValueError('Model/data feature protocols disagree')
    if cm.get('name', 'gcp') != 'gcp':
        raise ValueError('Only GCP checkpoints are supported')
    cm['feature_protocol'] = protocol
    digest = sha256(path)
    release_path = folder / 'release_manifest.json'
    release = json.loads(release_path.read_text()) if release_path.is_file() else {}
    if path.name == release.get('checkpoint'):
        if release.get('effective_config_sha256') and sha256(source) != release['effective_config_sha256']:
            raise ValueError('Released effective configuration checksum differs')
        if digest != release['checkpoint_sha256'] or protocol != release['feature_protocol']:
            raise ValueError('Released checkpoint checksum/feature protocol differs')
        policy = cm.get('input_policy', {'geometry_mode': 'full', 'edge_mode': 'spatial'})
        if any(policy.get(k) != v for k, v in release.get('input_policy', {}).items()):
            raise ValueError('Released checkpoint input policy differs')
    best = folder / 'best_auc.json'
    if path.name == 'model_best_auc.pt' and best.is_file():
        expected = json.loads(best.read_text()).get('checkpoint_sha256')
        if expected and expected != digest:
            raise ValueError('best_auc.json does not match checkpoint bytes')
    selected_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = GCPNetModel(cm)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    model.eval().to(selected_device)
    return LoadedCheckpoint(model, cd, cm, dict(checkpoint_path=str(path), checkpoint_sha256=digest,
        config_path=str(source), config_sha256=sha256(source), feature_protocol=protocol,
        input_policy=cm.get('input_policy', {'geometry_mode': 'full', 'edge_mode': 'spatial', 'ablation_protocol': 'legacy'}),
        device=str(selected_device), release=release))
