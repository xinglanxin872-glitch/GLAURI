"""Paper training data: complete CA residues, strict pairwise masks and soft BCE targets."""
import os
import pickle
import hashlib
from pathlib import Path
import h5py
import numpy as np
import torch as pt
from torch_geometric.data import Data, Batch
from src.data_encoding import aa_three_to_one, res_dict
from model.geometry import CA_PROTOCOL, residue_geometry

def reduce_to_dense(M):    # 保证规范
    # M: [N, R] dense one-hot
    resid = M.argmax(dim=1)                      # [N]
    idx = pt.arange(M.shape[0], device=M.device)
    return pt.stack([idx, resid], dim=1)

def load_seq(M,qr):
    residue_id = np.array(M)[:,1]
    residue_type = np.array(qr)[:,1]

    _, idx = np.unique(residue_id, return_index=True)

    sequence = residue_type[np.sort(idx)]

    align_b=[aa_three_to_one[res_dict[x]]for x in sequence ]
    return ('').join(align_b)

def load_interface_labels(hgrp, t0, t1_l):
    shape=tuple(hgrp.attrs['Y_shape'])
    ids=pt.from_numpy(np.asarray(hgrp['Y'],dtype=np.int64))
    receptor=(ids[:,2,None]==t0).any(dim=1)[:,None]
    partners=pt.stack([(ids[:,3,None]==types).any(dim=1) for types in t1_l],dim=1)
    y=pt.zeros((shape[0],len(t1_l)),dtype=pt.bool)
    row,col=pt.where(receptor&partners)
    y[ids[row,0],col]=True
    return y

def read_pickle(path):
    with open(path,'rb') as f:return pickle.load(f)

def resolve_project_path(path):
    """Legacy relative-path adapter retained for external callers; trainer uses absolute paths."""
    if path is None or not isinstance(path,str) or not path.strip():return path
    if os.path.isabs(path):return path
    repo=Path(__file__).resolve().parents[1]
    return str((repo/path[3:] if path.startswith('../') else repo/'model'/path).resolve())

def load_cluster_label_store(path):
    rows=read_pickle(path)
    if not isinstance(rows,list):raise ValueError('Expected label rows, not an implicit alternate format')
    store={str(r['cluster_name']):np.asarray(r['label'],np.float32) for r in rows}
    if len(store)!=len(rows):raise ValueError('Duplicate cluster labels')
    return store

def load_cluster_ref_seq_map(path):
    return {str(r['cluster_name']):str(r['ref_seq']) for r in read_pickle(path)}

def atom_columns(group):
    """H5 encodes one class assignment per atom as (atom_index, class_index)."""
    n_atoms=group['X'].shape[0]
    def column(name):
        sparse=np.asarray(group[name],dtype=np.int64)
        if sparse.shape!=(n_atoms,2) or not np.array_equal(sparse[:,0],np.arange(n_atoms)):
            raise ValueError(f'{group.name}/{name}: expected one sorted assignment per atom')
        return sparse[:,1]
    return column('M'),column('qn'),column('qr')

def h5_graph(group, rsa, protocol=CA_PROTOCOL):
    atom_residue,atom_type,residue_type=atom_columns(group)
    n_residues=int(group.attrs['M_shape'][1])
    geom=residue_geometry(np.asarray(group['X'],np.float32),atom_residue,atom_type,n_residues,protocol)
    selected=geom['node_indices']
    classes=int(group.attrs['qr_shape'][1])
    if classes!=29:raise ValueError(f'Expected 29 residue classes, got {classes}')
    onehot=pt.nn.functional.one_hot(pt.from_numpy(residue_type[selected]),classes).float()
    rsa=np.asarray(rsa,np.float32)
    if rsa.shape!=(n_residues,):raise ValueError(f'RSA shape {rsa.shape} differs from residue count {n_residues}')
    scalars=pt.cat([onehot,geom['angles'],pt.from_numpy(rsa[geom['residues']])[:,None]],dim=1)
    if scalars.shape[1]!=36 or not pt.isfinite(scalars).all():raise ValueError('Invalid structural scalars')
    nodes=len(selected)
    graph=Data(pos=geom['positions'],x=scalars,sc=geom['sidechain'],bb_coords=geom['backbone'],
               M=pt.arange(nodes)[:,None],residue_indices=pt.from_numpy(geom['residues']))
    graph.feature_protocol=protocol
    return graph

class Dataset(pt.utils.data.Dataset):
    """No silent label fallback, no global config, no training-time branch outside config."""
    def __init__(self,config):
        self.config=dict(config)
        self.dataset_filepath=config['dataset_filepath']
        self.rsa_filepath=config['rsa_filepath']
        self.protocol=config.get('feature_protocol',CA_PROTOCOL)
        if self.protocol!=CA_PROTOCOL:raise ValueError('New paper training requires corrected CA geometry')
        self.native_only=bool(config.get('native_only',False))
        self.mask_only=bool(config.get('mask_only',False))
        self._rsa_hf=None
        self._h5=None
        with h5py.File(self.dataset_filepath,'r') as hf:
            self.keys=np.asarray(hf['metadata/keys']).astype('U')
            self.ckeys=np.asarray(hf['metadata/ckeys']).astype('U')
            self.mids=np.asarray(hf['metadata/mids']).astype('U')
        self.m=np.ones(len(self.keys),bool)
        self.update_mask(self.m)
        self.set_types(config['l_types'],config['r_types'])
        self.mask_dict={}
        self.cluster_label_store=None
        if not self.mask_only:
            self.mask_dict=read_pickle(config['mask_dict_path'])
            meta=self.mask_dict.get(('__meta__',),{})
            if meta.get('format')!='pairwise' or meta.get('feature_protocol')!=self.protocol:
                raise ValueError('Mask protocol differs; rebuild CA masks before training')
            if not meta.get('complete'):
                raise ValueError('Partial/diagnostic masks cannot be used for full training')
            if not self.native_only:
                self.cluster_label_store=load_cluster_label_store(config['cluster_labels_path'])

    def update_mask(self,mask):
        mask=np.asarray(mask,bool)
        if mask.shape!=self.m.shape:raise ValueError('Split mask length differs from H5 metadata')
        self.m &= mask
        self.ckeys_map={}
        for key,contact in zip(self.keys[self.m],self.ckeys[self.m]):
            self.ckeys_map.setdefault(str(key),[]).append(str(contact))
        self.ukeys=list(self.ckeys_map)

    def set_types(self,l_types,r_types):
        self.t0=pt.from_numpy(np.flatnonzero(np.isin(self.mids,l_types)))
        self.t1_l=[pt.from_numpy(np.flatnonzero(np.isin(self.mids,types))) for types in r_types]

    def __len__(self):return len(self.ukeys)

    def embedding_context(self,key,sequences):
        parts=key.split('/')
        sequence=sequences[parts[1].upper()+'_'+parts[2].split(':')[0]]
        return sequence,self.mask_dict[key]['fasta']

    def _resolve_cluster(self,group,key):
        value=group['cluster'][()]
        return value.decode() if isinstance(value,bytes) else str(value)

    def __getstate__(self):
        state=self.__dict__.copy();state['_rsa_hf']=None;state['_h5']=None;return state

    def __getitem__(self,index):
        if self._h5 is None:self._h5=h5py.File(self.dataset_filepath,'r')
        if self._rsa_hf is None:self._rsa_hf=h5py.File(self.rsa_filepath,'r')
        key=self.ukeys[index]
        group=self._h5['data/structures/'+key]
        graph=h5_graph(group,self._rsa_hf['data/structures/'+key+'/rsa'][:],self.protocol)
        graph.key=key
        if self.mask_only:return graph
        masks=self.mask_dict[key]
        if masks.get('node_sequence_sha256') is None:raise ValueError('Mask lacks verified node sequence digest')
        node_types=graph.x[:,:29].argmax(dim=1).tolist()
        node_sequence=''.join(aa_three_to_one[res_dict[int(i)]] for i in node_types)
        if hashlib.sha256(node_sequence.encode()).hexdigest()!=masks['node_sequence_sha256']:
            raise ValueError(f'Node sequence/mask mismatch for {key}')
        if not np.array_equal(graph.residue_indices.numpy(),masks['residue_indices']):
            raise ValueError(f'Residue selection/mask mismatch for {key}')
        native=None
        for contact in self.ckeys_map[key]:
            labels=load_interface_labels(self._h5['data/contacts/'+contact],self.t0,self.t1_l)
            native=labels if native is None else native|labels
        if native is None:raise ValueError(f'No native contact records for {key}')
        native=native[graph.residue_indices].float()
        if self.native_only:
            graph.y=native
        else:
            cluster=self._resolve_cluster(group,key)
            if cluster not in self.cluster_label_store:raise KeyError(f'No projected labels for {key}: {cluster}')
            labels=pt.as_tensor(self.cluster_label_store[cluster]).float()
            source,target=map(lambda x:pt.as_tensor(x).bool(),masks['ref'])
            if len(source)!=len(labels) or len(target)!=graph.num_nodes or source.sum()!=target.sum():
                raise ValueError(f'Label/reference scatter differs for {key}')
            projected=pt.zeros_like(native);projected[target]=labels[source]
            graph.y=pt.maximum(native,projected)
        if graph.y.shape!=(graph.num_nodes,4) or not pt.isfinite(graph.y).all():raise ValueError('Invalid targets')
        if not ((graph.y>=0)&(graph.y<=1)).all():raise ValueError('Targets outside [0,1]')
        return graph

def collate_batch_data_key_pyg(graphs):
    return Batch.from_data_list(graphs)
