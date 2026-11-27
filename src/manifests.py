"""Resolve explicitly declared manifest roots without depending on process CWD."""
from collections import Counter
import json
from pathlib import Path


PATH_FIELDS = {'path', 'rsa_path', 'masks', 'sequences', 'curated_representatives',
               'curated_replacement_manifest', 'curated_membership', 'previous_predict', 'predict'}


def read_evaluation_inputs(path, roots=()):
    path = Path(path).resolve()
    mapping = {'repo': str(Path(__file__).resolve().parents[1])}
    for value in roots:
        key, target = value.split('=', 1)
        mapping[key] = str(Path(target).expanduser().resolve())
    def walk(value, key=None):
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, key) for v in value]
        if isinstance(value, str) and key in PATH_FIELDS:
            # 'predict' is also used as a condition name; only path strings reach here.
            value = value.format(**mapping)
            p = Path(value).expanduser()
            return str(p if p.is_absolute() else path.parent/p)
        return value
    inputs = walk(json.loads(path.read_text()))
    records = inputs['records']
    pairs = [(r['dataset'], r['chain']) for r in records]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate dataset/chain evaluation records')
    counts = dict(Counter(r['dataset'] for r in records))
    if inputs.get('counts') and counts != inputs['counts']:
        raise ValueError('Evaluation cohort counts differ from the input manifest')
    return inputs
