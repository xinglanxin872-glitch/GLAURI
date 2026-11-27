#!/usr/bin/env python3
"""Verify/export canonical benchmark-isolated folds, or reconstruct from source inputs."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import shutil
import sys
import h5py
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from curation.build_final_splits import coach_ids, digest
from curation.split_components import chain_key


def verify(h5_path, folder):
    manifest = json.loads((folder/'split_manifest.json').read_text())
    with h5py.File(h5_path, 'r') as hf:
        keys = np.asarray(hf['metadata/keys']).astype('U')
    masks, chains, component_sets = {}, {}, {}
    component_file = folder/'component_membership.json.gz'
    if digest(component_file) != manifest['component_membership_sha256']:
        raise ValueError('Component membership changed')
    components = json.loads(gzip.decompress(component_file.read_bytes()))
    for role in ['train', 'valid', 'test']:
        path = folder/f'{role}_m.pkl'
        if digest(path) != manifest['canonical_split_sha256'][role]:
            raise ValueError(role+' mask changed')
        masks[role] = np.asarray(pickle.loads(path.read_bytes()), bool)
        if masks[role].shape != keys.shape:
            raise ValueError('Metadata shape differs')
        ordered = list(dict.fromkeys(keys[masks[role]].tolist()))
        if len(ordered) != manifest['h5_structures'][role] or int(masks[role].sum()) != manifest['h5_contact_rows'][role]:
            raise ValueError(role+' count differs')
        if ordered != (folder/f'{role}_h5_keys.txt').read_text().split():
            raise ValueError(role+' H5 order differs')
        chains[role] = {chain_key(key) for key in ordered}
        if chains[role] != set((folder/f'{role}_chains.txt').read_text().split()):
            raise ValueError(role+' chain membership differs')
        component_sets[role] = {components[chain] for chain in chains[role]}
    for left, right in [('train','valid'), ('train','test'), ('valid','test')]:
        if np.any(masks[left] & masks[right]) or chains[left] & chains[right] or component_sets[left] & component_sets[right]:
            raise ValueError('Fold/component leakage')
    ids = (folder/'coach_test_chains.txt').read_text().split()
    if len(ids) != 420 or set(ids) != coach_ids():
        raise ValueError('COACH test membership must equal the full label table')
    if digest(folder/'coach_test_chains.txt') != manifest['coach_test_membership_sha256']:
        raise ValueError('COACH membership hash differs')
    held = {components[chain] for chain in manifest['protected_test_chains']}
    if held & (component_sets['train'] | component_sets['valid']):
        raise ValueError('A benchmark component entered training/validation')
    if digest(folder/'train_weight.pkl') != manifest['train_weight_sha256']:
        raise ValueError('Training sampler changed')
    print(json.dumps({'status':'PASS', 'dataset_version':manifest['dataset_version'],
                      'h5_structures':manifest['h5_structures'], 'coach_test_chains':len(ids),
                      'component_overlap':manifest['component_overlap']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--source-inputs', type=Path, help='Source inputs restored from the metadata supplement')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--h5', type=Path, default=REPO/'data/bio_251210.h5')
    args = parser.parse_args()
    if args.project_root or args.source_inputs:
        if args.output_dir is None:
            parser.error('--output-dir is required for reconstruction')
        from curation.build_final_splits import run
        run(args.project_root.resolve() if args.project_root else Path('.'), args.output_dir.resolve(), args.source_inputs, args.h5)
        verify(args.h5, args.output_dir)
        return
    source = REPO/'curation'
    verify(args.h5, source)
    if args.output_dir and not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.suffix in ['.txt','.pkl','.csv','.json','.gz'] and path.name != 'artifact_residue_names.json':
                shutil.copy2(path, args.output_dir/path.name)


if __name__ == '__main__':
    main()
