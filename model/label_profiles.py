"""Prepare cluster-level training targets from reference annotations."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile
import numpy as np
from model.experiment_config import fingerprint
from model.pocket_labels import load_ligand

def profile_parameters(resolved,validation=False):
    # H5 training and validation use this model's own profile. Public benchmark
    # ground truth is prepared independently and stays common across methods.
    return dict(resolved['fixed']['ligand_labels'])

def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def store_path(resolved,validation=False):
    parameters=profile_parameters(resolved,validation)
    # Data-source hashes in metadata catch stale inputs; config hashes distinguish profiles.
    key=fingerprint(parameters)[:16]
    return Path(resolved['paths']['label_store_directory'])/f'GLAURI_{key}.pkl'

def ensure_store(resolved,validation=False):
    params=profile_parameters(resolved,validation)
    path=store_path(resolved,validation); meta_path=path.with_suffix('.json')
    paths=resolved['paths']
    sources={k:{'path':paths[k],'sha256':file_sha256(paths[k])} for k in ['base_label_store','ligand_interactions']}
    expected={'parameters':params,'sources':sources,'non_ligand_columns':'unchanged binary union'}
    if path.exists():
        saved=json.loads(meta_path.read_text()) if meta_path.exists() else {}
        same_sources=all(saved.get('sources',{}).get(k,{}).get('sha256')==v['sha256'] for k,v in sources.items())
        if saved.get('parameters')!=params or saved.get('non_ligand_columns')!=expected['non_ligand_columns'] or not same_sources:
            raise ValueError(f'Stale/incompatible label store: {path}; choose a fresh label_store_directory')
        if saved.get('sha256')!=file_sha256(path):raise ValueError(f'Label-store checksum differs: {path}')
        return str(path)
    with open(paths['base_label_store'],'rb') as f:base=pickle.load(f)
    interactions=load_ligand(Path(paths['ligand_interactions']))
    positives=[0]*4; fractional=[0]*4
    for row in base:
        values=np.asarray(row['label'],dtype=np.float32).copy()
        if values.shape!=(len(row['ref_seq']),4) or not np.isin(values[:,:3],[0,1]).all():
            raise ValueError('Base store must contain three binary union columns and one ligand column')
        entry=interactions.pop(str(row['cluster_name']),None)
        ligand=np.zeros(len(row['ref_seq']),np.float32)
        if entry:
            if entry['ref_id']!=row['ref_id'] or entry['ref_seq']!=row['ref_seq']:
                raise ValueError('Pocket reference differs from base store')
            denominator=len(entry['chains'])
            for pocket in entry['pockets'].values():
                support=len(pocket['chains'])/denominator
                threshold=params['support_threshold'];floor=params['soft_floor']
                value=1. if support>=threshold else floor+(1-floor)*support/threshold
                np.maximum(ligand,pocket['label']*np.float32(value),out=ligand)
        values[:,3]=ligand;row['label']=values
        for col in range(4):
            positives[col]+=int((values[:,col]>0).sum())
            fractional[col]+=int(((values[:,col]>0)&(values[:,col]<1)).sum())
    if interactions:raise ValueError('Pocket interaction clusters absent from base labels')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:pickle.dump(base,f,protocol=4)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    expected.update(clusters=len(base),positive_counts=positives,fractional_counts=fractional,sha256=file_sha256(path))
    meta_path.write_text(json.dumps(expected,indent=2)+'\n')
    return str(path)
