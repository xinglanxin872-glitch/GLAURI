#!/usr/bin/env python3
"""Build reference sequences and binary non-ligand contact labels.

The fourth column is a zero placeholder, filled by model.prepare.
All four contact tables are required for references.
"""
import argparse,csv,gzip,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
csv.field_size_limit(min(sys.maxsize,2**31-1))
CLASS_FILES=[('protein',0),('nn',1),('ion',2),('ligand',3)]

def load_class(path: Path):
    """-> {cluster: {'ref_seq': str, 'ref_id': str,
                     'pockets': {pid: {'label': bool[L], 'chains': set}},
                     'chains': set}}"""
    out: dict[str, dict] = {}
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            c = r["cluster_name"]
            e = out.get(c)
            if e is None:
                e = out[c] = {"ref_seq": r["ref_seq"], "ref_id": r["ref_id"],
                              "pockets": {}, "chains": set()}
            elif e["ref_seq"] != r["ref_seq"]:
                raise ValueError(f"ref_seq length differs inside cluster {c}")
            pid = int(r["pocket_id"])
            p = e["pockets"].get(pid)
            if p is None:
                p = e["pockets"][pid] = {
                    "label": np.zeros(len(r["ref_seq"]), dtype=bool), "chains": set()}
            if len(r["ref_label"]) != len(e["ref_seq"]) or set(r["ref_label"]) - {"0", "1"}:
                raise ValueError(f"Invalid ref_label for {c}")
            lab = np.frombuffer(r["ref_label"].encode(), dtype=np.uint8) != ord("0")
            p["label"] |= lab
            p["chains"].add(r["receptor_key"])
            e["chains"].add(r["receptor_key"])
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ref-seq-csv',type=Path)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    tables={}
    sources={}
    for name,col in CLASS_FILES:
        path=args.in_dir/f'{name}_pocket_interactions.csv.gz'
        if not path.is_file():raise FileNotFoundError(path)
        tables[col]=load_class(path)
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    clusters=sorted(set().union(*(set(table) for table in tables.values())))
    rows=[]
    for cluster in clusters:
        entries=[table[cluster] for table in tables.values() if cluster in table]
        ref=entries[0]
        if any(e['ref_seq']!=ref['ref_seq'] or e['ref_id']!=ref['ref_id'] for e in entries):
            raise ValueError('Inconsistent reference: '+cluster)
        values=np.zeros((len(ref['ref_seq']),4),np.float32)
        for col in range(3):
            entry=tables[col].get(cluster)
            if entry:
                for pocket in entry['pockets'].values():
                    np.maximum(values[:,col],pocket['label'],out=values[:,col])
        rows.append(dict(cluster_name=cluster,label=values,ref_id=ref['ref_id'],ref_seq=ref['ref_seq']))
    if not rows:raise ValueError('No reference labels')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(pickle.dumps(rows,protocol=4))
    if args.ref_seq_csv:
        args.ref_seq_csv.parent.mkdir(parents=True,exist_ok=True)
        with args.ref_seq_csv.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['cluster_name','ref_seq'])
            writer.writerows((r['cluster_name'],r['ref_seq']) for r in rows)
    meta=dict(schema='glauri_reference_labels_v1',clusters=len(rows),sources=sources,
              labels='Binary non-ligand labels; ligand placeholder zero',
              sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    args.out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
