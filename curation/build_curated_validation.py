"""Curate native-positive sequence-cluster representatives, without model scores."""
import argparse
from collections import defaultdict, Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import h5py
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from model.experiment_config import resolve
from model.train import make_dataset
from model.label_profiles import store_path, file_sha256
from model.pocket_labels import load_ligand
from curation.split_components import chain_key

TASKS=['protein','nucleic','ion','ligand']


def build_truth(paths):
    """Common binary evaluation truth, separate from the fixed soft training labels."""
    rows=pickle.loads(Path(paths['base_label_store']).read_bytes())
    interactions=load_ligand(Path(paths['ligand_interactions']))
    for row in rows:
        values=np.asarray(row['label'],np.float32).copy();values[:,3]=0
        entry=interactions.pop(str(row['cluster_name']),None)
        if entry:
            if entry['ref_id']!=row['ref_id'] or entry['ref_seq']!=row['ref_seq']:
                raise ValueError('Evaluation reference mismatch')
            for pocket in entry['pockets'].values():
                if len(pocket['chains'])/len(entry['chains'])>=0.1:
                    values[:,3]=np.maximum(values[:,3],pocket['label'])
        if not np.isin(values,[0,1]).all():raise ValueError('Nonbinary evaluation truth')
        row['label']=values
    if interactions:raise ValueError('Unmatched evaluation clusters')
    return rows


def write_csv(path, rows, fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths');parser.add_argument('--coarse-clusters',type=Path,required=True)
    parser.add_argument('--existing-representatives',type=Path,help='Retain qualified protein/nucleic/ion representatives')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use an empty curated output directory')
    out.mkdir(parents=True,exist_ok=True)
    cfg=resolve('GLAURI',args.paths);paths=cfg['paths']
    ds=make_dataset(cfg,'valid',str(store_path(cfg)));index={k:i for i,k in enumerate(ds.ukeys)}
    sequences=pickle.loads(Path(paths['sequences']).read_bytes())
    seq_lower={str(k).lower():v for k,v in sequences.items()}
    memberships=defaultdict(set);centers={}
    for cid,line in enumerate(args.coarse_clusters.read_text().splitlines()):
        members=line.lower().split()
        if not members:continue
        centers[cid]=members[0]
        for chain in members:memberships[chain].add(cid)
    with h5py.File(paths['dataset'],'r') as hf:
        keys=np.asarray(hf['metadata/keys']).astype('U')
        ctypes=np.asarray(hf['metadata/ctypes']);mids=np.asarray(hf['metadata/mids']).astype('U')
    native_positive=np.zeros((len(keys),4),bool)
    for column,types in enumerate(cfg['config_data']['r_types']):
        mask=np.isin(mids[ctypes[:,1]],cfg['config_data']['l_types']) & np.isin(mids[ctypes[:,2]],types)
        native_positive[ctypes[mask,0],column]=True
    eligible={};excluded=[];rejected={}
    for row_idx in np.flatnonzero(np.asarray(pickle.loads(Path(paths['valid_split']).read_bytes()),bool)):
        key=keys[row_idx];chain=chain_key(key)
        reasons=[]
        if key not in index:reasons.append('not_in_effective_validation')
        if key not in ds.mask_dict:reasons.append('missing_ca_alignment')
        if chain not in seq_lower:reasons.append('missing_full_sequence')
        if len(memberships[chain])==0:reasons.append('missing_coarse_cluster')
        elif len(memberships[chain])!=1:reasons.append('ambiguous_coarse_cluster')
        if reasons:
            rejected[key]=reasons
            continue
        item=eligible.setdefault(key,dict(key=key,chain=chain,cluster30=next(iter(memberships[chain])),
                                        sequence_length=len(seq_lower[chain]),positive=np.zeros(4,bool)))
        item['positive'] |= native_positive[row_idx]
    groups=[defaultdict(list) for _ in TASKS]
    for item in eligible.values():
        for col in range(4):
            if item['positive'][col]:groups[col][item['cluster30']].append(item)
    for group in groups:
        for cid in group:group[cid].sort(key=lambda c:(-c['sequence_length'],c['chain'],c['key']))
    candidate_rows=[]
    for col,group in enumerate(groups):
        for cid,items in sorted(group.items()):
            candidate_rows.append(dict(interface=TASKS[col],cluster30=cid,cluster_center=centers[cid],
                                       n_positive_h5_keys=len(items),preferred_key=items[0]['key'],preferred_chain=items[0]['chain']))
    fixed=build_truth(paths)
    ds.cluster_label_store={str(row['cluster_name']):np.asarray(row['label'],np.float32) for row in fixed}
    existing=[]
    if args.existing_representatives:
        existing=list(csv.DictReader(args.existing_representatives.open()))
    cache={};selected=[];truth={};composition=[]
    def get(key):
        if key not in cache:cache[key]=ds[index[key]]
        return cache[key]
    for col,group in enumerate(groups):
        count=0;used=set();chosen=[]
        old=[r for r in existing if r['interface']==TASKS[col]] if col<3 else []
        for row in old:
            key=row['h5_key']
            if key not in eligible or not eligible[key]['positive'][col]:raise ValueError('Existing representative not qualified: '+key)
            item=eligible[key];cid=item['cluster30']
            if cid in used:raise ValueError('Repeated representative cluster')
            chosen.append(item);used.add(cid)
        for cid in np.random.default_rng(2025+col).permutation(sorted(group)).tolist():
            if len(chosen)>=100:break
            if cid in used:continue
            for item in group[cid]:
                try:
                    graph=get(item['key']);y=graph.y.numpy()[:,col]
                    if not (y>0).any():raise ValueError('Positive labels lost after projection')
                    chosen.append(item);used.add(cid);break
                except Exception as exc:
                    excluded.append(dict(interface=TASKS[col],cluster30=cid,h5_key=item['key'],reason=repr(exc)))
        for item in chosen:
            key=item['key'];graph=get(key);values=graph.y.numpy()
            if not np.isin(values,[0,1]).all() or not (values[:,col]>0).any():raise ValueError(key)
            mask=ds.mask_dict[key]
            record=dict(interface=TASKS[col],column=col,cluster30=item['cluster30'],cluster_center=centers[item['cluster30']],
                        pdb_chain=item['chain'],h5_key=key,sequence_length=item['sequence_length'],n_res=graph.num_nodes,
                        n_positive=int(values[:,col].sum()),positive_fraction=float(values[:,col].mean()),
                        high_id_cluster=mask['cluster_id'],ref_id=mask['cluster_id'])
            selected.append(record);truth[key]=values
        sub=[r for r in selected if r['interface']==TASKS[col]];lengths=[r['n_res'] for r in sub]
        composition.append(dict(interface=TASKS[col],candidate_positive_clusters=len(group),selected_clusters=len(sub),
                                n_residues=sum(lengths),positive_residues=sum(r['n_positive'] for r in sub),
                                positive_fraction=sum(r['n_positive'] for r in sub)/sum(lengths),
                                length_min=min(lengths),length_median=float(np.median(lengths)),length_max=max(lengths)))
        print(TASKS[col],composition[-1],flush=True)
    for task in TASKS:
        (out/f'{task}_valid_chains.txt').write_text('\n'.join(r['pdb_chain'] for r in selected if r['interface']==task)+'\n')
    write_csv(out/'representatives.csv',selected,list(selected[0]))
    write_csv(out/'composition.csv',composition,list(composition[0]))
    write_csv(out/'candidate_clusters.csv',candidate_rows,list(candidate_rows[0]))
    write_csv(out/'excluded_candidates.csv',excluded,['interface','cluster30','h5_key','reason'])
    (out/'labels.pkl').write_bytes(pickle.dumps(truth,protocol=4))
    manifest=dict(schema='glauri_curated_validation_v1',dataset_version='coach420_curated_v1',
                  seed_rule='2025 + task column; sorted cluster IDs permuted',
                  selection='Native-positive chains, one representative per coarse cluster, longest sequence first; no model scores',
                  retained_tasks=TASKS[:3] if existing else [],counts={r['interface']:r['selected_clusters'] for r in composition},
                  unique_h5_structures=len(truth),valid_split_sha256=file_sha256(paths['valid_split']),
                  coarse_clusters_sha256=file_sha256(args.coarse_clusters),feature_protocol='ca_ncac_sidechain_v1',
                  evaluation_labels='Binary non-ligand projected labels; ligand native maximum projected pockets with support >= 0.1; shared across all models',
                  label_store_sha256=file_sha256(out/'labels.pkl'),
                  representatives_sha256=file_sha256(out/'representatives.csv'),
                  train_label_definition=cfg['fixed']['ligand_labels'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    audit=dict(valid_unique_structures=len(index),usable_reference_structures=len(eligible),
               rejected_unique_structures=len(rejected),
               reasons=dict(Counter(reason for values in rejected.values() for reason in values)),
               rejected=[dict(h5_key=key,reasons=values) for key,values in sorted(rejected.items())])
    (out/'input_eligibility_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
