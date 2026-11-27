"""Complete COACH420 test membership, shared by both eval drivers."""
import os
from pathlib import Path


def canonical_coach_id(value):
    value = str(value).strip()
    if len(value) < 5:
        raise ValueError(f'Invalid COACH chain ID: {value!r}')
    return value[:4].upper() + '_' + value[4:].lstrip('_').upper()


def read_coach_test_ids(path=None):
    path = Path(path or os.environ.get('COACH_TEST_LIST', Path(__file__).resolve().parents[1] / 'curation/coach_test_chains.txt'))
    ids = [canonical_coach_id(line) for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f'Expected unique test chains in {path}')
    from curation.build_final_splits import coach_ids
    if len(ids)!=420 or {s.lower() for s in ids}!=coach_ids():
        raise ValueError('Test membership must equal the complete COACH420 label table')
    return frozenset(ids)


def filter_coach_test_ids(candidates, *, require_all=False):
    allowed = read_coach_test_ids()
    selected = sorted({str(value) for value in candidates if canonical_coach_id(value) in allowed})
    covered = {canonical_coach_id(value) for value in selected}
    if require_all and covered != allowed:
        missing = sorted(allowed - covered)
        raise ValueError(f'Missing {len(missing)} COACH_TEST labels: {missing[:5]}')
    return selected
