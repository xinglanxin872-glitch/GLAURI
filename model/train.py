"""One config-driven trainer for GLAURI and every Section 4 model."""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import pprint
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from model.experiment_config import resolve,parameter_decisions,read_json,REPO
from model.data_handler import Dataset,collate_batch_data_key_pyg,read_pickle
from model.label_profiles import ensure_store,store_path,file_sha256
from model.model import GCPNetModel

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)

def dataset_config(cfg,stage,store):
    data=dict(cfg['config_data'])
    data.update(mask_dict_path=str(Path(cfg['paths']['alignment_masks'])/f'mask_dict_{stage}.pkl'),
                cluster_labels_path=store,
                native_only=cfg['experiment']['labels']['native_only'])
    return data

def make_dataset(cfg,stage,store):
    ds=Dataset(dataset_config(cfg,stage,store))
    ds.update_mask(read_pickle(cfg['paths'][stage+'_split']))
    if not set(ds.ukeys)<=set(ds.mask_dict):raise ValueError(f'Masks do not cover {stage} split')
    return ds

def make_train_loader(cfg,train):
    exp=cfg['experiment'];tr=exp['training'];paths=cfg['paths']
    sampler=None
    if exp['sampling']['cluster_balanced']:
        weights=torch.as_tensor(read_pickle(paths['train_weight'])).double().flatten()
        if len(weights)!=len(train) or not torch.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('Training weights differ from selected chain order/length')
        from collections import Counter
        clusters=[train.mask_dict[key]['cluster_id'] for key in train.ukeys]
        counts=Counter(clusters)
        expected=torch.tensor([1/counts[c] for c in clusters],dtype=torch.float64)
        if not torch.equal(weights,expected):raise ValueError('Training sampler weights are not inverse cluster sizes in current Dataset order')
        sampler=WeightedRandomSampler(weights,len(train),replacement=True,generator=torch.Generator().manual_seed(tr['seed']))
    common=dict(batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate_batch_data_key_pyg,
                worker_init_fn=seed_worker,pin_memory=str(tr['device']).startswith('cuda'))
    if tr['workers']>0:common['prefetch_factor']=2
    return DataLoader(train,sampler=sampler,shuffle=sampler is None,generator=torch.Generator().manual_seed(tr['seed']),**common)

def build_loaders(cfg,train_store,valid_store):
    exp=cfg['experiment'];tr=exp['training']
    train=make_dataset(cfg,'train',train_store);valid=make_dataset(cfg,'valid',valid_store)
    generator=torch.Generator().manual_seed(exp['validation']['seed'])
    indices=sorted(torch.randperm(len(valid),generator=generator)[:exp['validation']['size']].tolist())
    valid=Subset(valid,indices)
    options=dict(batch_size=tr['batch_size'],num_workers=tr['workers'],collate_fn=collate_batch_data_key_pyg,
                 worker_init_fn=seed_worker,pin_memory=str(tr['device']).startswith('cuda'))
    if tr['workers']>0:options['prefetch_factor']=2
    return make_train_loader(cfg,train),DataLoader(valid,shuffle=False,generator=torch.Generator().manual_seed(tr['seed']+1),**options)

def weighted_bce(logits,target,pos_ratios,step,exp,is_train):
    """The sole loss implementation; every variant supplies only config values."""
    if logits.shape!=target.shape or not torch.isfinite(logits).all():raise ValueError('Invalid logits/target')
    element=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
    active=exp['active_tasks'];factors=torch.zeros_like(pos_ratios)
    if is_train:
        with torch.no_grad():
            average=target.mean(dim=0)
            pos_ratios[active]+=(average[active]-pos_ratios[active])/(1+math.sqrt(step))
        positive=torch.as_tensor(exp['loss']['positive_factors'],device=logits.device)
        positive=positive*(1-pos_ratios)/(pos_ratios+1e-6)
        weights=target*positive[None,:]+(1-target)
        if exp['loss']['normalize_positive_weights']:
            weights=weights/(weights.mean(dim=0,keepdim=True)+1e-6)
        element=element*weights
        if exp['loss']['task_weighting']=='pos_ratio':
            density=pos_ratios[active]
            factors[active]=density/density.sum().clamp_min(1e-12)
        else:factors[active]=1/len(active)
    else:
        # Validation has no class/task reweighting and never updates pos_ratios.
        factors[active]=1.
    loss=(element.mean(dim=0)*factors).sum()
    return loss,{'task_factors':factors.detach().cpu().tolist(),
                 'positive_ratios':pos_ratios.detach().cpu().tolist()}

from src.embeddings import EmbeddingProvider

def attach_embeddings(batch,dataset,sequences,provider,zero_plm=False):
    if len(batch.key)!=1:raise ValueError('Paper trainer expects one chain per batch')
    key=batch.key[0]
    sequence,masks=dataset.embedding_context(key,sequences)
    source,target=map(lambda a:torch.as_tensor(a).bool(),masks)
    if len(source)!=len(sequence) or len(target)!=batch.num_nodes or source.sum()!=target.sum():
        raise ValueError(f'ESM3/CA alignment mismatch for {key}')
    canvas=torch.zeros((batch.num_nodes,1536),dtype=torch.float32)
    if not zero_plm:canvas[target]=provider.get(sequence,key)[source]
    if batch.x.shape[1]!=36:raise ValueError('Expected 36 unmodified structural scalars')
    batch.x=torch.cat([batch.x,canvas],dim=1)
    return batch

def validation(model,loader,sequences,provider,ratios,cfg):
    model.eval();exp=cfg['experiment'];active=exp['active_tasks'];device=torch.device(exp['training']['device'])
    dataset=loader.dataset.dataset if isinstance(loader.dataset,Subset) else loader.dataset
    aucs={i:[] for i in active};losses=[];used=0
    with torch.no_grad():
        for batch in loader:
            batch=attach_embeddings(batch,dataset,sequences,provider,exp['inputs']['zero_plm']).to(device)
            logits=model(batch)['node_embedding']
            loss,_=weighted_bce(logits,batch.y,ratios,1,exp,False);losses.append(float(loss));used+=1
            p=logits.sigmoid().cpu().numpy();y=(batch.y.cpu().numpy()>=cfg['fixed']['metrics']['label_threshold'])
            eligible={i:len(np.unique(y[:,i]))==2 and len(np.unique(p[:,i]))>=2 for i in active}
            if cfg['fixed']['metrics']['auc_eligibility']=='all_active_tasks' and not all(eligible.values()):continue
            for i in active:
                if eligible[i]:aucs[i].append(float(roc_auc_score(y[:,i],p[:,i])))
    per={str(i):float(np.mean(values)) if values else None for i,values in aucs.items()}
    mean=float(np.mean(list(per.values()))) if all(v is not None for v in per.values()) else None
    return {'loss':float(np.mean(losses)) if losses else None,'mean_auc':mean,'per_task_auc':per,
            'eligible_chains':{str(i):len(v) for i,v in aucs.items()},'evaluated_chains':used,
            'selection_source':'h5'}

def save_best(model,output,metrics,step,epoch,previous,cfg):
    auc=metrics.get('mean_auc')
    if auc is None or not math.isfinite(auc) or auc<previous:return previous,False
    destination=Path(output)/'model_best_auc.pt';temporary=destination.with_suffix('.tmp')
    torch.save(model.state_dict(),temporary);os.replace(temporary,destination)
    record=dict(metrics,global_step=step,epoch_index=epoch,id=cfg['id'],paper_name=cfg['paper_name'],
                training_fingerprint=cfg['training_fingerprint'],feature_protocol=cfg['config_data']['feature_protocol'],
                criterion=cfg['fixed']['metrics'],checkpoint_sha256=file_sha256(destination))
    temp=Path(output)/'best_auc.json.tmp';temp.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(temp,Path(output)/'best_auc.json')
    return auc,True

def check_inputs(cfg,require_profiles=True):
    paths=cfg['paths'];required=['dataset','rsa','sequences','base_label_store','ligand_interactions','train_split','valid_split']
    if cfg['experiment']['sampling']['cluster_balanced']:required.append('train_weight')
    for key in required:
        if not Path(paths[key]).is_file():raise FileNotFoundError(f'{key}: {paths[key]}')
    for stage in ['train','valid']:
        meta=Path(paths['alignment_masks'])/f'mask_dict_{stage}.json'
        info=read_json(meta)
        if info.get('feature_protocol')!=cfg['config_data']['feature_protocol'] or not info.get('complete'):
            raise ValueError(f'Incomplete or incompatible {stage} alignment masks')
        if info.get('split_sha256')!=file_sha256(paths[stage+'_split']):
            raise ValueError(f'{stage} masks were built for a different split; rebuild them')
    if require_profiles:
        for valid in [False,True]:
            if not cfg['experiment']['labels']['native_only']:
                if not store_path(cfg,valid).is_file():raise FileNotFoundError('Run model.prepare first: '+str(store_path(cfg,valid)))

def train(cfg):
    check_inputs(cfg)
    output=Path(cfg['paths']['output_root'])/cfg['id']
    if output.exists():raise FileExistsError(f'Fresh output directory required: {output}')
    train_store=ensure_store(cfg,False) if not cfg['experiment']['labels']['native_only'] else None
    valid_store=ensure_store(cfg,True) if not cfg['experiment']['labels']['native_only'] else None
    exp=cfg['experiment'];tr=exp['training'];seed_all(tr['seed'])
    train_loader,valid_loader=build_loaders(cfg,train_store,valid_store)
    output.mkdir(parents=True)
    cfg=dict(cfg,train_label_store=train_store,valid_label_store=valid_store)
    (output/'effective_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    # Standalone snapshot for inference; it imports no mutable training defaults.
    snapshot='config_data = '+pprint.pformat(cfg['config_data'])+'\nconfig_model = '+pprint.pformat(cfg['config_model'])+'\nconfig_runtime = '+pprint.pformat(tr)+'\n'
    (output/'config.py').write_text(snapshot)
    (output/'parameter_decisions.json').write_text(json.dumps(parameter_decisions(cfg),indent=2)+'\n')
    selected=[valid_loader.dataset.dataset.ukeys[i] for i in valid_loader.dataset.indices]
    (output/'validation_subset.txt').write_text('\n'.join(selected)+'\n')
    sources={str(f.relative_to(REPO)):file_sha256(f) for folder in ['model','src'] for f in (REPO/folder).rglob('*.py') if '__pycache__' not in f.parts}
    artifacts={k:file_sha256(cfg['paths'][k]) for k in ['train_split','valid_split','train_weight'] if k!='train_weight' or exp['sampling']['cluster_balanced']}
    (output/'run_manifest.json').write_text(json.dumps({'code_sha256':sources,'split_weight_sha256':artifacts,'train_label_store':train_store,'valid_label_store':valid_store,'training_fingerprint':cfg['training_fingerprint'],'feature_protocol':cfg['config_data']['feature_protocol']},indent=2)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(output/'tb'));device=torch.device(tr['device'])
    model=GCPNetModel(cfg['config_model']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=tr['learning_rate'])
    ratios=torch.full((4,),.5,device=device)
    sequences=read_pickle(cfg['paths']['sequences']);provider=EmbeddingProvider(device,cfg['paths']['embedding_cache'],backend=exp['embeddings']['attention_backend'],memory_size=exp['embeddings']['memory_size'])
    best=-math.inf;step=0;last_evaluation=-1
    def evaluate(epoch):
        nonlocal best,last_evaluation
        metrics=validation(model,valid_loader,sequences,provider,ratios,cfg)
        best,saved=save_best(model,output,metrics,step,epoch,best,cfg);last_evaluation=step
        record=dict(kind='validation',step=step,epoch=epoch,saved_best_auc=saved,**metrics)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if metrics['mean_auc'] is not None:writer.add_scalar('validation/mean_auc',metrics['mean_auc'],step)
        model.train()
    try:
        for epoch in range(tr['epochs']):
            model.train()
            for batch in train_loader:
                step+=1;optimizer.zero_grad()
                batch=attach_embeddings(batch,train_loader.dataset,sequences,provider,exp['inputs']['zero_plm']).to(device)
                logits=model(batch)['node_embedding']
                loss,trace=weighted_bce(logits,batch.y,ratios,step,exp,True)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tr['gradient_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                if step==1 or step%tr['log_every_steps']==0:
                    record=dict(kind='train',step=step,epoch=epoch,loss=float(loss.detach()),gradient_norm=float(norm),**trace)
                    print(json.dumps(record),flush=True)
                    with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    writer.add_scalar('train/loss',record['loss'],step)
                if step%tr['evaluate_every_steps']==0:evaluate(epoch)
            # Include the last weights of every epoch among selection candidates.
            if last_evaluation!=step:evaluate(epoch)
            torch.save(model.state_dict(),output/'model_last.pt')
    finally:writer.close()
    if not (output/'model_best_auc.pt').is_file():raise RuntimeError('No finite validation AUC; no best-AUC checkpoint produced')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='GLAURI',help='Paper model name or variant JSON file')
    parser.add_argument('--paths',help='Deployment path JSON; contains no training hyperparameters')
    parser.add_argument('--print-config',action='store_true')
    parser.add_argument('--check-inputs',action='store_true')
    args=parser.parse_args();cfg=resolve(args.config,args.paths)
    parameter_decisions(cfg)
    if args.print_config:print(json.dumps(cfg,indent=2));return
    if args.check_inputs:check_inputs(cfg);print('INPUT CHECKS PASSED: '+cfg['paper_name']);return
    train(cfg)

if __name__=='__main__':main()
