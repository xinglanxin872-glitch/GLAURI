"""Rebuild CA-protocol pairwise masks from H5 atom identities; never relabel old masks."""
from functools import lru_cache
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import h5py
import numpy as np
from Bio import Align
from tqdm import tqdm
from model.experiment_config import resolve
from model.geometry import CA_PROTOCOL,select_backbone
from model.data_handler import atom_columns
from src.data_encoding import res_dict,aa_three_to_one

CTX={}

def pairwise_masks(query,structure):
    if query==structure:return np.ones(len(query),bool),np.ones(len(structure),bool)
    if not query or not structure:raise ValueError('Empty alignment sequence')
    aligner=Align.PairwiseAligner();aligner.mode='global'
    q,s=aligner.align(query,structure)[0].aligned
    qm=np.zeros(len(query),bool);sm=np.zeros(len(structure),bool)
    for a,b in q:qm[a:b]=True
    for a,b in s:sm[a:b]=True
    if qm.sum()!=sm.sum() or not qm.any():raise ValueError('Invalid pairwise alignment')
    return qm,sm

@lru_cache(maxsize=4096)
def align_triple(reference,sequence,fasta):
    return {'ref':pairwise_masks(reference,sequence),'fasta':pairwise_masks(fasta,sequence)}

def initialize(dataset):
    CTX['h5']=h5py.File(dataset,'r')

def collect_chunk(keys):
    output=[]
    for key in keys:
        group=CTX['h5']['data/structures/'+key]
        ar,at,rt=atom_columns(group)
        residue_ids,indices=select_backbone(ar,at,int(group.attrs['M_shape'][1]),CA_PROTOCOL)
        sequence=''.join(aa_three_to_one[res_dict[int(i)]] for i in rt[indices[:,1]])
        cluster=group['cluster'][()];cluster=cluster.decode() if isinstance(cluster,bytes) else str(cluster)
        reference=CTX['references'][cluster]
        pieces=key.split('/');fasta_key=pieces[1].upper()+'_'+pieces[2].split(':')[0]
        fasta=CTX['sequences'][fasta_key]
        masks=dict(align_triple(reference,sequence,fasta))
        masks['cluster_id']=cluster
        masks['node_sequence_sha256']=hashlib.sha256(sequence.encode()).hexdigest()
        masks['residue_indices']=residue_ids.astype(np.int32)
        output.append((key,masks))
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--split',choices=['train','valid','both'],default='both')
    parser.add_argument('--limit',type=int,default=0,help='diagnostic only; partial masks are marked and rejected by full training')
    args=parser.parse_args();cfg=resolve('GLAURI',args.paths);paths=cfg['paths']
    with open(paths['base_label_store'],'rb') as f:rows=pickle.load(f)
    CTX['references']={str(r['cluster_name']):str(r['ref_seq']) for r in rows};del rows
    with open(paths['sequences'],'rb') as f:CTX['sequences']=pickle.load(f)
    with h5py.File(paths['dataset'],'r') as hf:
        metadata_keys=np.asarray(hf['metadata/keys']).astype('U')
        names=np.asarray(hf['metadata/std_names']).astype('U')
        if list(names[:3])!=['CA','N','C']:raise ValueError('Unexpected H5 atom vocabulary')
    out=Path(paths['alignment_masks']);out.mkdir(parents=True,exist_ok=True)
    for split in (['train','valid'] if args.split=='both' else [args.split]):
        target=out/f'mask_dict_{split}.pkl'
        if target.exists():raise FileExistsError(f'Refusing to overwrite {target}')
        with open(paths[split+'_split'],'rb') as f:selection=np.asarray(pickle.load(f),bool)
        if selection.shape!=metadata_keys.shape:raise ValueError('H5 metadata / split shape differs')
        keys=list(dict.fromkeys(metadata_keys[selection].tolist()))
        expected=len(keys)
        if args.limit:keys=keys[:args.limit]
        meta={'format':'pairwise','preprocess':'CA_BB','feature_protocol':CA_PROTOCOL,
              'node_atom':'CA','backbone_order':['N','CA','C'],'expected_count':expected,
              'actual_count':len(keys),'complete':len(keys)==expected,'split':split,
              'split_sha256':hashlib.sha256(Path(paths[split+'_split']).read_bytes()).hexdigest(),
              'builder':'model.build_masks: atom identity selection from H5, pairwise realignment'}
        masks={('__meta__',):meta}
        chunks=[keys[i:i+256] for i in range(0,len(keys),256)]
        if args.workers<=1:
            initialize(paths['dataset'])
            for chunk in tqdm(chunks,desc=split):masks.update(collect_chunk(chunk))
            CTX['h5'].close();del CTX['h5']
        else:
            # Linux cluster fork preserves the immutable reference/sequence dictionaries.
            import multiprocessing
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize,initargs=(paths['dataset'],)) as pool:
                for chunk in tqdm(pool.map(collect_chunk,chunks,chunksize=1),total=len(chunks),desc=split):masks.update(chunk)
        if len(masks)-1!=len(keys):raise ValueError('Missing masks')
        temp=target.with_suffix('.tmp')
        with temp.open('wb') as f:pickle.dump(masks,f,protocol=4)
        os.replace(temp,target)
        target.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
