"""Verify config-to-training decisions, corrected geometry and every paper model."""
import argparse
import copy
import gc
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch_geometric.data import Batch
from model.experiment_config import resolve,parameter_decisions,MODEL,read_json
from model.train import (make_dataset,weighted_bce,EmbeddingProvider,attach_embeddings,
                         seed_all,save_best,dataset_config,make_train_loader)
from model.data_handler import read_pickle
from model.label_profiles import store_path,profile_parameters
from model.geometry import CA_PROTOCOL,LEGACY_PROTOCOL,residue_geometry
from model.model import GCPNetModel
from src.metrics import icc_a1

def core_checks():
    names=[r['id'] for r in read_json(MODEL/'model_registry.json')]
    configs=[resolve(name) for name in names]
    assert len(set(c['training_fingerprint'] for c in configs))==len(configs)
    for cfg in configs:
        parameter_decisions(cfg)
        assert profile_parameters(cfg,False)==profile_parameters(cfg,True)
        assert store_path(cfg,False)==store_path(cfg,True)
        assert dataset_config(cfg,'valid',None)['native_only']==cfg['experiment']['labels']['native_only']
    assert resolve('GLAURI')['fixed']['ligand_labels']['support_threshold']==.5
    assert resolve('zero_seq')['paper_name']=='w/o PLM'
    # Zero inactive task gradients; changing config alters actual loss factors.
    for cfg in configs:
        target=torch.tensor([[0.,1.,0.,1.],[1.,0.,0.,1.],[1.,1.,1.,0.]])
        logits=torch.zeros((3,4),requires_grad=True);ratios=torch.full((4,),.5)
        loss,trace=weighted_bce(logits,target,ratios,1,cfg['experiment'],True)
        loss.backward();active=cfg['experiment']['active_tasks'];inactive=list(set(range(4))-set(active))
        if inactive:assert torch.count_nonzero(logits.grad[:,inactive])==0
        assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad[:,active])>0
        if cfg['experiment']['loss']['task_weighting']=='uniform':assert np.allclose(np.asarray(trace['task_factors'])[active],1/len(active))
        before=ratios.clone();weighted_bce(logits.detach(),target,ratios,1,cfg['experiment'],False)
        assert torch.equal(before,ratios)
    # Known ICC residual-df example; includes negative and degenerate-case behavior.
    assert abs(icc_a1([.1,.4,.8,.3],[.2,.6,.7,.1])-.8484848484848485)<1e-12
    assert abs(icc_a1([0.,1.,2.],[0.,1.,2.])-1)<1e-12
    assert np.isnan(icc_a1([1.],[1.]))
    # A changed config value must never silently alias a valid parameter.
    with tempfile.TemporaryDirectory() as td:
        wrong=Path(td)/'bad.json';wrong.write_text(json.dumps(dict(id='bad',paper_name='bad',changes={'loss':{'task_weigthing':'uniform'}})))
        try:resolve(str(wrong))
        except ValueError:pass
        else:raise AssertionError('Unknown config key accepted')
        model=torch.nn.Linear(2,4);cfg=configs[0]
        good={'mean_auc':.8,'per_task_auc':{str(i):.8 for i in range(4)}}
        best,saved=save_best(model,td,good,1,0,-np.inf,cfg);assert saved
        data=(Path(td)/'model_best_auc.pt').read_bytes()
        for bad in [.7,None,float('nan')]:
            _,saved=save_best(model,td,dict(good,mean_auc=bad),2,0,best,cfg);assert not saved
            assert (Path(td)/'model_best_auc.pt').read_bytes()==data
        restored=torch.nn.Linear(2,4);restored.load_state_dict(torch.load(Path(td)/'model_best_auc.pt',weights_only=True),strict=True)
    return {'models':names,'config_decisions_per_model':len(parameter_decisions(configs[0])),
            'loss_and_inactive_gradients':'PASS','unknown_parameter_rejected':'PASS','icc_independent_example':'PASS','best_auc_guard_and_reload':'PASS'}

def real_checks(paths_file):
    import h5py
    cfg=resolve('GLAURI',paths_file);seed_all(42);torch.set_num_threads(4)
    sequences=read_pickle(cfg['paths']['sequences']);device=torch.device(cfg['experiment']['training']['device'])
    provider=EmbeddingProvider(device,cfg['paths']['embedding_cache'],backend=cfg['experiment']['embeddings']['attention_backend'])
    baseline=make_dataset(cfg,'train',str(store_path(cfg)))
    selected=None
    with h5py.File(cfg['paths']['dataset'],'r') as hf:
        for i,key in enumerate(baseline.ukeys[:2000]):
            if 400<=hf['data/structures/'+key+'/X'].shape[0]<=800:selected=i;break
    assert selected is not None
    graph=baseline[selected];key=baseline.ukeys[selected]
    # Verify graph positions against named CA coordinates, and against the N alternative.
    with h5py.File(cfg['paths']['dataset'],'r') as hf:
        group=hf['data/structures/'+key]
        from model.data_handler import atom_columns
        ar,at,rt=atom_columns(group)
        old=residue_geometry(group['X'][:],ar,at,int(group.attrs['M_shape'][1]),LEGACY_PROTOCOL)
        corrected=residue_geometry(group['X'][:],ar,at,int(group.attrs['M_shape'][1]),CA_PROTOCOL)
        assert torch.equal(graph.pos,corrected['positions']) and not torch.equal(graph.pos,old['positions'])
        assert torch.allclose(graph.bb_coords[:,1],graph.pos)
    blank=Batch.from_data_list([graph.clone()])
    rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    embedded=attach_embeddings(blank,baseline,sequences,provider)
    assert torch.equal(rng_cpu,torch.get_rng_state())
    if rng_gpu is not None:assert torch.equal(rng_gpu,torch.cuda.get_rng_state(device))
    embedding=embedded.x[:,36:].clone()
    assert embedding.abs().sum()>0
    valid_ds=make_dataset(cfg,'valid',str(store_path(cfg,True)))
    with h5py.File(cfg['paths']['dataset'],'r') as hf:
        valid_index=next(i for i,key in enumerate(valid_ds.ukeys[:2000]) if 400<=hf['data/structures/'+key+'/X'].shape[0]<=800)
    valid_graph=valid_ds[valid_index]
    valid_embedded=attach_embeddings(Batch.from_data_list([valid_graph.clone()]),valid_ds,sequences,provider)
    valid_features=valid_embedded.x[:,36:].clone()
    rows=[]
    for item in read_json(MODEL/'model_registry.json'):
        current=resolve(item['id'],paths_file);exp=current['experiment'];seed_all(exp['training']['seed'])
        ds=make_dataset(current,'train',str(store_path(current)) if not exp['labels']['native_only'] else None)
        assert ds.native_only==exp['labels']['native_only']
        assert ds.ukeys[selected]==key
        sample=ds[selected];batch=Batch.from_data_list([sample])
        sequence_features=torch.zeros_like(embedding) if exp['inputs']['zero_plm'] else embedding
        batch.x=torch.cat([batch.x,sequence_features],dim=1);batch=batch.to(device)
        model=GCPNetModel(current['config_model']).to(device)
        captured={}
        def capture(module,args):
            data=args[0];captured.update({key:getattr(data,key).detach().clone() for key in ('h','chi','e','xi','pos','sc','bb_coords','f_ij','edge_index')})
        def capture_esm(module,args):captured['raw_esm']=args[0].detach().clone()
        def capture_edge(module,args):captured['rbf_edges']=args[0].scalar.detach().clone()
        handles=[model.gcp_embedding.register_forward_pre_hook(capture),model.esm_proj.register_forward_pre_hook(capture_esm),model.gcp_embedding.edge_normalization.register_forward_pre_hook(capture_edge)]
        model.train();optimizer=torch.optim.Adam(model.parameters(),lr=exp['training']['learning_rate'])
        ratios=torch.full((4,),.5,device=device);logits=model(batch)['node_embedding']
        loss,trace=weighted_bce(logits,batch.y,ratios,1,exp,True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),exp['training']['gradient_clip'])
        assert torch.isfinite(loss) and torch.isfinite(grad) and grad>0
        inactive=list(set(range(4))-set(exp['active_tasks']))
        if inactive:assert torch.count_nonzero(model.dm[-1].weight.grad[inactive])==0
        assert bool(torch.count_nonzero(captured['raw_esm'])==0)==exp['inputs']['zero_plm']
        if exp['inputs']['geometry_mode']=='zero':
            assert torch.equal(captured['h'][:,:29],sample.x[:,:29].to(device))
            assert captured['h'][:,29:35].count_nonzero()==0
            for feature_name in ('chi','e','xi','pos','sc','bb_coords','f_ij','rbf_edges'):assert captured[feature_name].count_nonzero()==0
            assert model.augment_eps==0 and model.edge_mode=='sequence'
        if exp['inputs']['zero_rsa']:assert torch.count_nonzero(captured['h'][:,35])==0
        assert bool(model.zero_seq_input)==exp['inputs']['zero_plm']
        assert bool(model.zero_struct_input)==exp['inputs']['zero_structure']
        assert bool(model.zero_rsa_input)==exp['inputs']['zero_rsa']
        optimizer.step()
        # H5 validation follows each model's own label profile and native-only flag.
        assert dataset_config(current,'valid',str(store_path(current,True)))['native_only']==exp['labels']['native_only']
        current_valid=make_dataset(current,'valid',None if exp['labels']['native_only'] else str(store_path(current,True)))
        current_valid_graph=current_valid[valid_index]
        model.eval()
        vb=Batch.from_data_list([current_valid_graph])
        vb.x=torch.cat([vb.x,torch.zeros_like(valid_features) if exp['inputs']['zero_plm'] else valid_features],dim=1)
        before=ratios.clone()
        with torch.no_grad():
            vb=vb.to(device);pred=model(vb)['node_embedding']
            valid_loss,_=weighted_bce(pred,vb.y,ratios,1,exp,False)
        assert torch.equal(before,ratios) and torch.isfinite(valid_loss)
        # Inspect the real loader's sampler without spawning worker processes.
        loader=make_train_loader(current,ds);sampler=loader.sampler
        expected='WeightedRandomSampler' if exp['sampling']['cluster_balanced'] else 'RandomSampler'
        assert type(sampler).__name__==expected
        rows.append(dict(id=current['id'],paper_name=current['paper_name'],loss=float(loss.detach()),gradient_norm=float(grad),
                         active_tasks=exp['active_tasks'],native_only=ds.native_only,sampler=expected,
                         train_store=None if ds.native_only else str(store_path(current)),task_factors=trace['task_factors'],
                         structural_zero=exp['inputs']['zero_structure'],rsa_zero=exp['inputs']['zero_rsa'],plm_zero=exp['inputs']['zero_plm'],validation_loss=float(valid_loss)))
        for handle in handles:handle.remove()
        for handle in [ds._h5,ds._rsa_hf]:
            if handle is not None:handle.close()
        for handle in [current_valid._h5,current_valid._rsa_hf]:
            if handle is not None:handle.close()
        del ds,current_valid,current_valid_graph,batch,sample,model,optimizer,loss,logits,sampler,loader,vb,pred
        gc.collect();torch.cuda.empty_cache()
        print('[PASS] real training '+current['paper_name'],flush=True)
    return {'status':'PASS','geometry':'CA positions and N-CA-C backbone verified from atom identities',
            'chain':key,'nodes':graph.num_nodes,'nonzero_sidechain':int((graph.sc.norm(dim=1)>.5).sum()),'variants':rows}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--real',action='store_true');parser.add_argument('--paths');parser.add_argument('--output',required=True)
    args=parser.parse_args();result={'status':'PASS','core':core_checks()}
    if args.real:result['real']=real_checks(args.paths)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print('ALL TRAINING CHECKS PASSED',flush=True)

if __name__=='__main__':main()
