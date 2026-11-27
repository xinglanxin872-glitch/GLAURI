"""Build benchmark-isolated H5 folds with the complete COACH420 held out."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from curation.split_components import chain_key, load_components


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coach_ids():
    with (REPO/'benchmark/labels/ligand/coach420_label.csv').open() as stream:
        ids = [row['PDB ID'].strip() for row in csv.DictReader(stream)]
    ids = [value[:4].lower() + '_' + value[4:].lstrip('_').lower() for value in ids]
    if len(ids) != 420 or len(set(ids)) != 420:
        raise ValueError('COACH420 requires exactly 420 unique chains')
    return set(ids)


def run(project, output, source_inputs=None, h5_path=None):
    project, output = Path(project), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty split output directory')
    inputs = Path(source_inputs) if source_inputs else None
    coarse = inputs/'coarse_clusters.txt' if inputs else project/'cluster/pdb_seqres_pro_clean_cluster_cluster_30_80_forSplit.txt'
    native_path = inputs/'native_mapping.pkl' if inputs else project/'model/20260917_mmseq_longest_soft_ml/support/colabel_mask_all.pkl'
    components, native, memberships, _ = load_components(coarse, native_path)
    protected = coach_ids()
    # Preserve an existing held-out chain rather than returning its three
    # biological-assembly structures to training when reconstructing source folds.
    additional = REPO/'curation/additional_test_chains.txt'
    protected.update(additional.read_text().lower().split())
    for name in ['MaSIF-53', 'DNA-129', 'RNA-117', 'Zn-59']:
        protected.update((REPO/f'curation/{name}_test_chains.txt').read_text().lower().split())
    allobench = inputs/'allobench_test_pockets.csv' if inputs else project/'colabel/allobench_pockets_testset.csv'
    with allobench.open() as stream:
        for row in csv.DictReader(stream):
            protected.update(value.lower() for value in row['pdb_chainname'].split(';') if value)
    held_out = {components.find(value) for value in protected}
    with h5py.File(h5_path or project/'bio_251210.h5', 'r') as hf:
        keys = np.asarray(hf['metadata/keys']).astype('U')
    sources = {}
    original = {}
    for role in ['train', 'valid', 'test']:
        path = inputs/f'base_masks/{role}_m.pkl' if inputs else project/f'processing/data/datasets/cluster_30_80_benchclean_{"val" if role == "valid" else role}_m.pkl'
        sources[role] = {'path': str(path.relative_to(inputs or project)), 'sha256': digest(path)}
        with path.open('rb') as stream:
            original[role] = np.asarray(pickle.load(stream), bool)
        if original[role].shape != keys.shape:
            raise ValueError('Original mask and H5 metadata differ')
    effective = np.logical_or.reduce(list(original.values()))
    roles_by_component = {}
    for role, mask in original.items():
        for key in set(keys[mask]):
            roles_by_component.setdefault(components.find(chain_key(key)), set()).add(role)
    assignment = {}
    for component, roles in roles_by_component.items():
        assignment[component] = ('test' if component in held_out or 'test' in roles
                                 else 'valid' if 'valid' in roles else 'train')
    role_keys = {role: set() for role in original}
    for key in set(keys[effective]):
        role_keys[assignment[components.find(chain_key(key))]].add(key)
    output.mkdir(parents=True, exist_ok=True)
    masks = {}
    chains = {}
    for role, selected in role_keys.items():
        mask = effective & np.isin(keys, list(selected))
        masks[role] = mask
        (output/f'{role}_m.pkl').write_bytes(pickle.dumps(mask, protocol=4))
        ordered = list(dict.fromkeys(keys[mask].tolist()))
        (output/f'{role}_h5_keys.txt').write_text('\n'.join(ordered)+'\n')
        chains[role] = {chain_key(key) for key in ordered}
        (output/f'{role}_chains.txt').write_text('\n'.join(sorted(chains[role]))+'\n')
    for left, right in [('train','valid'), ('train','test'), ('valid','test')]:
        if np.any(masks[left] & masks[right]) or chains[left] & chains[right]:
            raise ValueError('H5 fold overlap')
        if {components.find(c) for c in chains[left]} & {components.find(c) for c in chains[right]}:
            raise ValueError('Sequence/co-label components cross folds')
    ordered = (output/'train_h5_keys.txt').read_text().split()
    counts = Counter(str(native[key]['cluster']) for key in ordered)
    weights = np.asarray([1/counts[str(native[key]['cluster'])] for key in ordered], np.float64)
    (output/'train_weight.pkl').write_bytes(pickle.dumps(weights, protocol=4))
    (output/'coach_test_chains.txt').write_text('\n'.join(sorted(coach_ids()))+'\n')
    component_map = {chain: components.find(chain) for chain in sorted(set.union(*chains.values()) | protected)}
    (output/'component_membership.json.gz').write_bytes(gzip.compress(
        json.dumps(component_map, sort_keys=True, separators=(',', ':')).encode(), mtime=0))
    for role in ['train', 'valid']:
        if {components.find(c) for c in chains[role]} & held_out:
            raise ValueError('Held-out benchmark component entered '+role)
    summary = {
        'schema': 'glauri_dataset_coach420_v1', 'dataset_version': 'coach420_curated_v1',
        'feature_protocol': 'ca_ncac_sidechain_v1',
        'test_policy': 'Complete benchmark sequence/native-label connected components held out',
        'h5_structures': {role: len(values) for role, values in role_keys.items()},
        'h5_contact_rows': {role: int(mask.sum()) for role, mask in masks.items()},
        'unique_chains': {role: len(values) for role, values in chains.items()},
        'training_clusters': len(counts), 'coach_test_chains': 420,
        'coach_h5_coverage': {role: len(coach_ids() & values) for role, values in chains.items()},
        'coach_external_only': sorted(coach_ids() - set.union(*chains.values())),
        'component_overlap': {a+'/'+b: 0 for a,b in [('train','valid'),('train','test'),('valid','test')]},
        'canonical_split_sha256': {role: digest(output/f'{role}_m.pkl') for role in masks},
        'train_weight_sha256': digest(output/'train_weight.pkl'),
        'coach_test_membership_sha256': digest(output/'coach_test_chains.txt'),
        'component_membership_sha256': digest(output/'component_membership.json.gz'),
        'protected_test_chains': sorted(protected),
        'sources': {'coarse_clusters': {'path':str(coarse.relative_to(inputs or project)), 'sha256':digest(coarse)},
                    'additional_test_chains': {'path':'curation/additional_test_chains.txt','sha256':digest(additional)},
                    'native_mapping': {'path':str(native_path.relative_to(inputs or project)), 'sha256':digest(native_path)},
                    'allobench_test_pockets': {'path':str(allobench.relative_to(inputs or project)), 'sha256':digest(allobench)},
                    'base_masks': sources},
    }
    (output/'split_manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run(args.project_root.resolve(), args.output_dir.resolve())
