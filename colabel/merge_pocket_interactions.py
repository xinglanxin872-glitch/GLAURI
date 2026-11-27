#!/usr/bin/env python3
"""Merge complete per-class shards; preserve input shards and reject overlaps."""
import argparse
import csv
import gzip
from pathlib import Path
import re
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True)
    args = p.parse_args()
    directory = Path(args.dir)
    for name in ['protein', 'nn', 'ion', 'ligand']:
        shards = sorted(directory.glob(f'{name}_shard*of*_pocket_interactions.csv.gz'))
        if not shards:
            continue
        indices, totals = set(), set()
        for path in shards:
            match = re.fullmatch(rf'{name}_shard(\d+)of(\d+)_pocket_interactions.csv.gz', path.name)
            i, n = map(int, match.groups())
            indices.add(i)
            totals.add(n)
        if len(totals) != 1 or indices != set(range(next(iter(totals)))):
            raise ValueError(f'Incomplete or inconsistent shards for {name}')
        output = directory / f'{name}_pocket_interactions.csv.gz'
        temporary = output.with_suffix('.tmp')
        schema, owner, count = None, {}, 0
        try:
            with gzip.open(temporary, 'wt', newline='') as fh:
                writer = None
                for shard in shards:
                    with gzip.open(shard, 'rt', newline='') as source:
                        reader = csv.DictReader(source)
                        if schema is None:
                            schema = reader.fieldnames
                            writer = csv.DictWriter(fh, fieldnames=schema)
                            writer.writeheader()
                        if reader.fieldnames != schema:
                            raise ValueError(f'Shard schema differs: {shard}')
                        for row in reader:
                            cluster = row['cluster_name']
                            if cluster in owner and owner[cluster] != shard:
                                raise ValueError(f'Cluster appears in multiple shards: {cluster}')
                            owner[cluster] = shard
                            writer.writerow(row)
                            count += 1
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'{name}: {len(shards)} shards, {len(owner)} clusters, {count} rows -> {output}')


if __name__ == '__main__':
    main()
