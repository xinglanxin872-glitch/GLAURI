#!/usr/bin/env python3
"""MMseqs2 protein sequence clustering and compatible membership summary.

Defaults reproduce the original 95% identity / 90% coverage / cov-mode 0 setup.
Use --cluster-tsv to rebuild summaries from an existing MMseqs membership table.
"""
import argparse
from collections import OrderedDict
import csv
from pathlib import Path
import subprocess

from Bio import SeqIO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta', required=True)
    p.add_argument('--out-prefix', required=True)
    p.add_argument('--mmseqs', default='mmseqs')
    p.add_argument('--min-seq-id', type=float, default=0.95)
    p.add_argument('--coverage', type=float, default=0.9)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--tmp-dir')
    p.add_argument('--cluster-tsv', help='Reuse existing membership instead of running MMseqs')
    args = p.parse_args()
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv = Path(args.cluster_tsv or str(prefix) + '_cluster.tsv')
    if not args.cluster_tsv:
        subprocess.run([args.mmseqs, 'easy-cluster', args.fasta, str(prefix),
                        args.tmp_dir or str(prefix) + '_tmp', '--min-seq-id', str(args.min_seq_id),
                        '-c', str(args.coverage), '--cov-mode', '0', '--threads', str(args.threads)], check=True)
    lengths = {}
    for record in SeqIO.parse(args.fasta, 'fasta'):
        if record.id in lengths:
            raise ValueError(f'Duplicate FASTA ID: {record.id}')
        lengths[record.id] = len(record.seq)
    groups, seen = OrderedDict(), set()
    with tsv.open() as fh:
        for line in fh:
            representative, member = line.split()
            if member in seen:
                raise ValueError(f'Member occurs twice in MMseqs table: {member}')
            seen.add(member)
            groups.setdefault(representative, []).append(member)
    if seen != set(lengths):
        raise ValueError('FASTA and MMseqs membership must cover exactly the same IDs')
    summary = Path(str(prefix) + '_cluster_summary.csv')
    with summary.open('w', newline='') as fh, Path(str(prefix) + '_cluster.txt').open('w') as members_file:
        writer = csv.writer(fh)
        writer.writerow(['cluster_names', 'cluster_size', 'seq_name', 'seq_length', 'min_max'])
        for representative, members in groups.items():
            if representative not in members:
                raise ValueError(f'Representative is absent from its group: {representative}')
            sizes = [lengths[key] for key in members]
            writer.writerow([representative, len(members), repr(members), repr(sizes), repr((min(sizes), max(sizes)))])
            members_file.write(' '.join(members) + '\n')
    print(f'{len(seen)} chains in {len(groups)} sequence clusters -> {summary}')


if __name__ == '__main__':
    main()
