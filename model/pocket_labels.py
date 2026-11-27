"""Read distinct ligand receptor-chain support from the frozen pocket interaction table."""
import csv,gzip,sys
from pathlib import Path
import numpy as np
csv.field_size_limit(min(sys.maxsize,2**31-1))

def load_ligand(path: Path):
    clusters = {}
    with gzip.open(path, 'rt', newline='') as handle:
        for row in csv.DictReader(handle):
            cluster = row['cluster_name']
            entry = clusters.get(cluster)
            if entry is None:
                entry = clusters[cluster] = dict(ref_id=row['ref_id'], ref_seq=row['ref_seq'],
                                                 chains=set(), pockets={})
            elif entry['ref_id'] != row['ref_id'] or entry['ref_seq'] != row['ref_seq']:
                raise ValueError(f'ligand interaction reference differs within {cluster}')
            chain = row['receptor_key']
            entry['chains'].add(chain)
            pid = int(row['pocket_id'])
            pocket = entry['pockets'].get(pid)
            if pocket is None:
                pocket = entry['pockets'][pid] = dict(
                    chains=set(), label=np.zeros(len(entry['ref_seq']), dtype=bool))
            raw = row['ref_label'].encode('ascii')
            label = np.frombuffer(raw, dtype=np.uint8)
            if len(label) != len(entry['ref_seq']) or not np.isin(label, [ord('0'), ord('1')]).all():
                raise ValueError(f'bad ref_label for {cluster} pocket {pid}')
            pocket['label'] |= label == ord('1')
            pocket['chains'].add(chain)
    return clusters
