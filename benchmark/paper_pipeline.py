"""Produce the four paper experiments from newly trained CA-protocol checkpoints.

External baseline predictions are never executed by this module.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import h5py
from Bio import Align
from sklearn.metrics import roc_auc_score,f1_score,recall_score,accuracy_score,matthews_corrcoef
from model.experiment_config import read_json,MODEL,fingerprint
from model.data_handler import h5_graph,read_pickle
from model.model import GCPNetModel
from model.geometry import CA_PROTOCOL
from model.train import EmbeddingProvider
from model.label_profiles import file_sha256
from src.utils_eval import process_structure,process_seq_batch,inference
from src.metrics import icc_a1
from src.data_encoding import res_dict,aa_three_to_one

def pair_map(reference,sequence):
    if reference==sequence:return {i:i for i in range(len(reference))}
    aligner=Align.PairwiseAligner();aligner.mode='global'
    r,s=aligner.align(reference,sequence)[0].aligned
    return {ri:si for (ra,rb),(sa,sb) in zip(r,s) for ri,si in zip(range(ra,rb),range(sa,sb))}

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if y.shape!=p.shape or not np.isfinite(p).all():raise ValueError('Invalid score/label arrays')
    return dict(auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                f1=float(f1_score(y,p>=.5,zero_division=0)),tpr=float(recall_score(y,p>=.5,zero_division=0)),
                accuracy=float(accuracy_score(y,p>=.5)),mcc=float(matthews_corrcoef(y,p>=.5)),n_res=len(y),n_positive=int(y.sum()))

class Embeddings(EmbeddingProvider):
    """Paper API wrapper; historical caches require explicit permission."""
    def __init__(self,device,files,backend='math',allow_legacy_cache=False):
        super().__init__(device,backend=backend,cache_files=files,allow_legacy_cache=allow_legacy_cache)

class GraphBank:
    def __init__(self,root,embeddings):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.embeddings=embeddings;self.pickles={};self.h5={}
        if self.embeddings.identity.get('weights_sha256') is None:
            raise ValueError('Cache the identified ESM3 weights before creating paper feature caches')
        self.embeddings.directory=self.root/'sequence_embeddings';self.embeddings.directory.mkdir(exist_ok=True)

    def object(self,path):
        if path not in self.pickles:self.pickles[path]=read_pickle(path)
        return self.pickles[path]

    def get(self,source):
        signature=dict(source,protocol=CA_PROTOCOL,embedding_dtype='float32',embedding_attention_backend=self.embeddings.backend,embedding_protocol=self.embeddings.protocol)
        signature['stat']={k:[Path(source[k]).stat().st_size,Path(source[k]).stat().st_mtime_ns]
                           for k in ['path','rsa_path','masks'] if k in source}
        cache=self.root/(fingerprint(signature)+'.pt')
        if cache.exists():return torch.load(cache,map_location='cpu',weights_only=False)
        if source['kind']=='h5':
            if source['path'] not in self.h5:self.h5[source['path']]=h5py.File(source['path'],'r')
            if source['rsa_path'] not in self.h5:self.h5[source['rsa_path']]=h5py.File(source['rsa_path'],'r')
            key=source['key'];group=self.h5[source['path']]['data/structures/'+key]
            graph=h5_graph(group,self.h5[source['rsa_path']]['data/structures/'+key+'/rsa'][:],CA_PROTOCOL)
            sequence=''.join(aa_three_to_one[res_dict[int(i)]] for i in graph.x[:,:29].argmax(dim=1))
            pieces=key.split('/');fasta=self.object(source['sequences'])[pieces[1].upper()+'_'+pieces[2].split(':')[0]]
            masks=self.object(source['masks']);meta=masks.get(('__meta__',),{})
            if meta.get('feature_protocol')!=CA_PROTOCOL:raise ValueError('Curated validation requires rebuilt CA masks')
            source_mask,target_mask=map(lambda a:torch.as_tensor(a).bool(),masks[key]['fasta'])
            if len(target_mask)!=len(sequence) or len(source_mask)!=len(fasta):raise ValueError('Curated ESM mask mismatch')
            embedding=torch.zeros((len(sequence),1536))
            embedding[target_mask]=self.embeddings.get(fasta,key)[source_mask]
        else:
            if source['kind']=='structure_cache':
                struct=self.object(source['path'])[source['key']][source['condition']]
            elif source['kind']=='file':
                chain=source.get('chain');struct=process_structure(source['path'],chainid=(chain+':0') if chain else None)
            else:raise ValueError(source['kind'])
            if not isinstance(struct,dict) or not struct:raise ValueError(f'Cannot read {source}')
            graphs,sequence=process_seq_batch(struct,(False,True,False),'CA_BB','gcp',64,with_rsa=True,protocol=CA_PROTOCOL)
            graph=graphs[0];embedding=self.embeddings.get(sequence,source.get('key'))
        result={'graph':graph,'sequence':sequence,'embedding':embedding.float(),'signature':signature}
        torch.save(result,cache)
        return result

def score_record(row,model,bank,device):
    reference=row['reference_sequence'];objects={key:bank.get(row[key]) for key in ['bound','predict'] if key in row}
    maps={key:pair_map(reference,obj['sequence']) for key,obj in objects.items()}
    common=sorted(set.intersection(*(set(m) for m in maps.values())))
    if not common:raise ValueError('No shared residues: '+row['chain'])
    target=np.frombuffer(row['label'].encode(),np.uint8)-48
    output={'y':target[common],'ref_index':np.asarray(common),'reference_sequence':reference,'column':row['column']}
    shared=None
    if row['dataset'] in ['coach_test']:
        shared=bank.embeddings.get(reference,row['chain'])[common]
    for condition,obj in objects.items():
        selected=np.asarray([maps[condition][i] for i in common]);graph=obj['graph'].clone()
        embedding=obj['embedding'].float()
        if shared is not None:
            graph=graph.subgraph(torch.from_numpy(selected));embedding=shared
        scores=inference(model,device,[graph],'gcp',plm_embedding=embedding,plm_fusion='concat').numpy()[:,row['column']+1]
        output['p_'+condition]=(scores if shared is not None else scores[selected]).astype(np.float32)
    if 'pockets' in row:
        output['pockets']=[dict(p,id=p['id'],label=(np.frombuffer(p['label'].encode(),np.uint8)-48)[common]) for p in row['pockets']]
    return output

def summarize(dataset,model,records):
    values=list(records.values());y=np.concatenate([r['y'] for r in values]);rows=[];chains=[];pockets=[]
    paired=all('p_bound' in r and 'p_predict' in r for r in values)
    for cid,r in records.items():
        chains.append(dict(dataset=dataset,model=model,chain=cid,n_res=len(r['y']),icc=icc_a1(r['p_bound'],r['p_predict']) if paired else np.nan))
    finite=np.asarray([r['icc'] for r in chains]);finite=finite[np.isfinite(finite)]
    for condition in ['bound','predict']:
        if not all('p_'+condition in r for r in values):continue
        p=np.concatenate([r['p_'+condition] for r in values])
        rows.append(dict(dataset=dataset,model=model,structure=condition,n_chains=len(values),
                         icc_median=float(np.median(finite)) if len(finite) else np.nan,**metrics(y,p)))
    if dataset.startswith('allobench'):
        for cid,r in records.items():
            for pocket in r['pockets']:
                truth=np.asarray(pocket['label'],bool);exclude=np.zeros(len(truth),bool)
                for other in r['pockets']:
                    if other['id']!=pocket['id']:exclude|=np.asarray(other['label'],bool)&~truth
                for threshold in [.2,.3,.4,.5,.6,.7,.8,.85]:
                    predicted=(r['p_predict']>=threshold)&~exclude
                    intersection=int((truth&predicted).sum());union=int((truth|predicted).sum())
                    pockets.append(dict(dataset=dataset,model=model,target=cid,pocket=pocket['id'],pocket_type=pocket['type'],threshold=threshold,intersection=intersection,union=union,jaccard=intersection/union if union else np.nan))
    return rows,chains,pockets

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,type=Path)
    parser.add_argument('--asset-root',action='append',default=[],help='Manifest root mapping NAME=/path (repeatable)')
    parser.add_argument('--runs-root',type=Path,default=MODEL/'runs')
    parser.add_argument('--checkpoint-name',default='model_best_auc.pt',help='Checkpoint filename within each model run (e.g. model_last.pt)')
    parser.add_argument('--feature-cache',type=Path,help='Reuse a verified feature-cache directory; prediction outputs remain separate')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--groups',default='standard_test,weighting,ablation')
    parser.add_argument('--models',help='Optional comma-separated model IDs')
    parser.add_argument('--datasets',help='Optional comma-separated dataset keys for prioritized subsets')
    parser.add_argument('--stage',choices=['plan','features','infer','score','all'],default='all')
    parser.add_argument('--embedding-cache',action='append',default=[])
    parser.add_argument('--allow-legacy-embedding-cache',action='store_true',help='Explicit historical cache replay only; records this exception in the manifest')
    parser.add_argument('--device',default='cuda');parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--sdpa-backend',choices=['math','auto'],default='math')
    args=parser.parse_args()
    if args.models:
        from model.experiment_config import resolve
        args.models=','.join(dict.fromkeys(resolve(name)['id'] for name in args.models.split(',')))
    from src.manifests import read_evaluation_inputs
    inputs=read_evaluation_inputs(args.inputs,args.asset_root);manifest_sha=file_sha256(args.inputs)
    cohort_sha={dataset:fingerprint([{k:v for k,v in r.items() if k not in ('bound','predict')} for r in inputs['records'] if r['dataset']==dataset]) for dataset in {r['dataset'] for r in inputs['records']}}
    if Path(args.checkpoint_name).name!=args.checkpoint_name:raise ValueError('--checkpoint-name must be a filename within each model run')
    registry={r['id']:r for r in read_json(MODEL/'model_registry.json')}
    groups=read_json(Path(__file__).with_name('paper_experiments.json'));selected=args.groups.split(',')
    todo={};membership=[]
    columns={'MaSIF-53':0,'DNA-129':1,'RNA-117':1,'Zn-59':2,'ligand_valid':3,'coach_test':3,'protein_valid':0,'nucleic_valid':1,'ion_valid':2,'allobench_valid':3,'allobench_test':3}
    if args.datasets and set(args.datasets.split(','))-set(columns):raise ValueError('Unknown dataset filter')
    for group in selected:
        spec=groups[group]
        for name in spec['models']:
            if args.models and name not in args.models.split(','):continue
            from model.experiment_config import resolve
            snapshot=args.runs_root/name/'effective_config.json'
            active=(read_json(snapshot)['experiment']['active_tasks'] if snapshot.exists()
                    else resolve(name)['experiment']['active_tasks'])
            datasets=[d for d in spec['datasets'] if columns[d] in active]
            datasets += [d for d,names in spec.get('additional_datasets',{}).items() if name in names]
            if args.datasets:datasets=[d for d in datasets if d in args.datasets.split(',')]
            for dataset in datasets:
                todo.setdefault(name,set()).add(dataset);membership.append((group,name,dataset))
    args.output.mkdir(parents=True,exist_ok=True)
    if not todo:raise ValueError('No registered model/dataset selected')
    plan={'feature_protocol':CA_PROTOCOL,'input_manifest_sha256':manifest_sha,'cohort_sha256':cohort_sha,'groups':selected,'partial_limit':args.limit,
          'embedding_dtype':'float32','embedding_attention_backend':args.sdpa_backend,
          'allow_legacy_embedding_cache':args.allow_legacy_embedding_cache,'metric_protocol':'icc_a1_residual_df_v1',
          'model_filter':args.models,'dataset_filter':args.datasets,'checkpoint_name':args.checkpoint_name,
          'feature_cache_root':str((args.feature_cache or args.output/'features').resolve()),
          'models':{name:{'datasets':sorted(datasets),'checkpoint':str(args.runs_root/name/args.checkpoint_name),
                          'checkpoint_exists':(args.runs_root/name/args.checkpoint_name).is_file()} for name,datasets in todo.items()}}
    (args.output/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.stage=='plan':print(json.dumps(plan,indent=2));return
    bank=None
    if args.stage in ['features','infer','all']:
        bank=GraphBank(args.feature_cache or args.output/'features',Embeddings(torch.device(args.device),args.embedding_cache,args.sdpa_backend,args.allow_legacy_embedding_cache))
    datasets=set.union(*todo.values())
    if args.stage=='features':
        for dataset in sorted(datasets):
            rows=[r for r in inputs['records'] if r['dataset']==dataset]
            if args.limit:rows=rows[:args.limit]
            for row in rows:
                for condition in ['bound','predict']:
                    if condition in row:bank.get(row[condition])
            print('FEATURES',dataset,len(rows),flush=True)
        return
    summary=[];chain_rows=[];pocket_rows=[]
    for name,datasets in todo.items():
        folder=args.output/'predictions'/name;folder.mkdir(parents=True,exist_ok=True)
        run=args.runs_root/name;model=None;checkpoint_sha=None
        if args.stage in ['infer','all']:
            cfg=read_json(run/'effective_config.json')
            if cfg.get('status','').startswith('diagnostic') and not args.limit:
                raise ValueError('Diagnostic weights require an explicit partial --limit')
            if cfg['id']!=name or cfg['config_data']['feature_protocol']!=CA_PROTOCOL:raise ValueError('Wrong checkpoint/model feature protocol')
            checkpoint=run/args.checkpoint_name;checkpoint_sha=file_sha256(checkpoint)
            from src.checkpoint import load_checkpoint
            loaded=load_checkpoint(run,args.checkpoint_name,args.device)
            model=loaded.model
        for dataset in sorted(datasets):
            path=folder/(dataset+'.pkl')
            if args.stage in ['infer','all']:
                rows=[r for r in inputs['records'] if r['dataset']==dataset]
                if args.limit:rows=rows[:args.limit]
                if path.exists():
                    saved=read_pickle(path)
                    if saved.get('embedding_protocol')!=bank.embeddings.protocol:raise ValueError('Embedding protocol differs from saved predictions')
                    if saved['checkpoint_sha256']!=checkpoint_sha or saved.get('checkpoint_name','model_best_auc.pt')!=args.checkpoint_name or saved['input_manifest_sha256']!=manifest_sha or saved['partial_limit']!=args.limit or saved.get('embedding_attention_backend')!=args.sdpa_backend or saved.get('embedding_dtype')!='float32':
                        raise ValueError('Refusing to mix different checkpoint/input results: '+str(path))
                else:
                    records={}
                    for i,row in enumerate(rows,1):
                        records[row['chain']]=score_record(row,model,bank,torch.device(args.device))
                        if i==1 or i%25==0 or i==len(rows):
                            print('PROGRESS',name,dataset,i,'/',len(rows),flush=True)
                    saved=dict(model=name,paper_name=registry[name]['paper_name'],dataset=dataset,feature_protocol=CA_PROTOCOL,
                               embedding_attention_backend=args.sdpa_backend,embedding_dtype='float32',
                               embedding_protocol=bank.embeddings.protocol,input_policy=cfg['config_model'].get('input_policy',{'geometry_mode':'full','edge_mode':'spatial'}),cohort_sha256=cohort_sha[dataset],metric_protocol='icc_a1_residual_df_v1',
                               checkpoint_sha256=checkpoint_sha,checkpoint_name=args.checkpoint_name,input_manifest_sha256=manifest_sha,partial_limit=args.limit,records=records)
                    temp=path.with_suffix('.tmp')
                    with temp.open('wb') as f:pickle.dump(saved,f,protocol=4)
                    temp.replace(path)
            else:
                saved=read_pickle(path)
                if saved['checkpoint_sha256']!=file_sha256(run/args.checkpoint_name):raise ValueError('Saved predictions belong to a different checkpoint')
            if saved.get('input_manifest_sha256')!=manifest_sha or saved.get('feature_protocol')!=CA_PROTOCOL or saved.get('partial_limit')!=args.limit or saved.get('embedding_attention_backend')!=args.sdpa_backend or saved.get('embedding_dtype')!='float32' or saved.get('checkpoint_name','model_best_auc.pt')!=args.checkpoint_name:
                raise ValueError('Saved scores belong to a different input/protocol/diagnostic scope: '+str(path))
            checkpoint=run/args.checkpoint_name
            if checkpoint.exists() and saved['checkpoint_sha256']!=file_sha256(checkpoint):
                raise ValueError('Saved scores belong to a different current checkpoint: '+str(path))
            if saved.get('cohort_sha256')!=cohort_sha[dataset]:raise ValueError('Saved cohort labels/membership differ')
            expected_rows=[r for r in inputs['records'] if r['dataset']==dataset]
            if args.limit:expected_rows=expected_rows[:args.limit]
            if set(saved['records'])!={r['chain'] for r in expected_rows}:raise ValueError('Incomplete/extra prediction members')
            rows,chains,pockets=summarize(dataset,name,saved['records']);summary.extend(rows);chain_rows.extend(chains);pocket_rows.extend(pockets)
            print('SCORED',name,dataset,len(saved['records']),flush=True)
        if model is not None:del model;torch.cuda.empty_cache()
    metrics_frame=pd.DataFrame(summary);metrics_frame['paper_name']=metrics_frame.model.map({k:v['paper_name'] for k,v in registry.items()})
    metrics_frame.to_csv(args.output/'metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(chain_rows).to_csv(args.output/'icc_per_chain.csv',index=False,float_format='%.17g')
    pd.DataFrame(pocket_rows).to_csv(args.output/'pocket_jaccard.csv',index=False,float_format='%.17g')
    group_rows=[]
    for group,name,dataset in membership:
        subset=metrics_frame[(metrics_frame.model==name)&(metrics_frame.dataset==dataset)].copy();subset['experiment']=group;group_rows.append(subset)
    pd.concat(group_rows,ignore_index=True).to_csv(args.output/'experiment_metrics.csv',index=False,float_format='%.17g')
    status=('partial_diagnostic' if args.limit else 'complete'
            if set(selected)==set(groups) and not args.models and not args.datasets else 'subset_complete')
    (args.output/'result_manifest.json').write_text(json.dumps(dict(plan,status=status,icc='ICC(A,1), residual MSE denominator (n-1)*(k-1)',external_methods='not executed'),indent=2)+'\n')

if __name__=='__main__':main()
