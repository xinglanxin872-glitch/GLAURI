#!/usr/bin/env python3
"""Select protein receptors from H5 metadata and export full chain sequences.

Preserves the pre-clustering stages of colabel_summary_merged.ipynb.
Default sequences come from all_seq_dict.pkl, not just contacting residues.
"""
import argparse
from pathlib import Path
import pickle

import h5py

from contact_io import H5Contacts, reference_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h5', required=True)
    p.add_argument('--all-seq-dict', help='Full-chain reference sequences; required for reference mode')
    p.add_argument('--sequence-source', choices=['reference', 'structure'], default='reference')
    p.add_argument('--min-num-res', type=int, default=48)
    p.add_argument('--max-atoms', type=int, default=8192, help='0 disables the original export size filter')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out-fasta', required=True)
    args = p.parse_args()
    if args.sequence_source == 'reference' and not args.all_seq_dict:
        p.error('--all-seq-dict is required for --sequence-source reference')
    data = H5Contacts(args.h5, args.min_num_res, args.max_atoms or None)
    sequences = {}
    if args.sequence_source == 'reference':
        with open(args.all_seq_dict, 'rb') as fh:
            sequences = pickle.load(fh)
    keys = data.ukeys[:args.limit] if args.limit else data.ukeys
    output = Path(args.out_fasta)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, 'r') as hf, output.open('w') as fh:
        for key in keys:
            seq = reference_sequence(key, sequences) if args.sequence_source == 'reference' else data.structure_sequence(hf[f'data/structures/{key}'])
            if not seq:
                raise ValueError(f'Empty sequence: {key}')
            fh.write(f'>{key}\n{seq}\n')
    output.with_suffix('.ids.txt').write_text('\n'.join(keys) + '\n')
    print(f'Exported {len(keys)} sequences -> {output}')


if __name__ == '__main__':
    main()
