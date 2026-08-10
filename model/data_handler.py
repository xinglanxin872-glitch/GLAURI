import os
import random
import h5py
import numpy as np
import pandas as pd
import torch as pt
import pickle
from torch_geometric.data import Data, Batch
from src.dataset import load_sparse_mask, collate_batch_features
from src.data_encoding import extract_topology,aa_three_to_one,res_dict
from graphein.protein.tensor.angles import dihedrals
from model.config import config_data,config_runtime


def reduce_to_dense(M):    # enforce canonical form
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

def backbone_coords_by_res_mask(X, M, atom_type, backbone_idx=[0,1,2]):
    X = np.asarray(X)
    M = np.asarray(M)
    atom_type = np.asarray(atom_type)
    N_atom, N_res = M.shape
    coords_list = []
    valid_res=0
    mask_index=[]
    for j in range(N_res):
        # find the atom indices belonging to residue j
        atom_idx_in_res = np.where(M[:,j] > 0)[0]
        # keep only backbone atoms
        if 1 in atom_type[atom_idx_in_res]:
            valid_res+=1
            atom_idx_in_res = [i for i in atom_idx_in_res if atom_type[i] in backbone_idx]
            # sort by atom_type (ensures N->CA->C)
            atom_idx_in_res.sort(key=lambda i: backbone_idx.index(atom_type[i]))
            coords_arr = X[atom_idx_in_res]
            if len(coords_arr)<3:
                mask_index.append(valid_res-1)
            else:
                coords_list.append(coords_arr)
    mask=np.ones(valid_res,dtype=bool)
    mask[mask_index]=0

    return np.stack(coords_list),mask


def preprocess(X,M,qe,qr,qn,preprocess_type,ftrs,ern_shape,rsa=None):

    # rsa: optional per-residue RSA feature (N_res,) float (original residue-column space, aligned with M.shape[1]).
    # Supported only under CA_BB: gather the RSA of kept nodes via residue_indices and cat it as the last node-scalar column.
    if rsa is not None and preprocess_type != "CA_BB":
        raise NotImplementedError("rsa 特征目前只支持 preprocess='CA_BB'")

    types = qn[:, 1]

    # Original residue-column indices kept by CA_BB (same order as the filtered nodes); None for other preprocess types
    residue_indices = None
    # CA_BB side-chain orientation unit vectors (N_ca,3); None for other preprocess types
    sc_feat = None

    # build a boolean mask: True marks the rows to keep
    if preprocess_type=="extract_backbone":
        sel_types=[0, 1, 2, 3, 4]
        mask =np.isin(types, sel_types)
        sel = pt.nonzero(pt.from_numpy(mask), as_tuple=False).squeeze(-1)  # shape: [N_selected]
        X,M,qn,qr,qe=X[sel],M[sel],qn[sel],qr[sel],qe[sel]
    # get the indices that match
    elif preprocess_type=="CA_only":
        sel_types=[0]

        mask =np.isin(types, sel_types)
        sel = pt.nonzero(pt.from_numpy(mask), as_tuple=False).squeeze(-1)  # shape: [N_selected]        
        res_idx = M.argmax(dim=1)
        selected = []
        seen_res = set()

        for i in sel.tolist():
            r = int(res_idx[i])
            if r not in seen_res:
                selected.append(i)
                seen_res.add(r)
        sel = pt.tensor(selected, device=X.device)
        X,qn,qr,qe,M=X[sel],qn[sel],qr[sel],qe[sel],M[sel]
        N = M.shape[0]
        M = pt.eye(N, device=M.device, dtype=M.dtype)
    elif preprocess_type=="CA_BB":
        N_res=M.shape[1]
        coords_list = []
        ca_atom_indices = []
        residue_indices = []
        sc_list = []   # side-chain orientation unit vectors, in kept-node order

        for j in range(N_res):

            atom_idx = np.where(M[:, j] > 0)[0]

            idx_N  = [i for i in atom_idx if types[i] == 0]
            idx_CA = [i for i in atom_idx if types[i] == 1]
            idx_C  = [i for i in atom_idx if types[i] == 2]

            if len(idx_N)==1 and len(idx_CA)==1 and len(idx_C)==1:

                coords = np.stack([
                    X[idx_N[0]],
                    X[idx_CA[0]],
                    X[idx_C[0]],
                ])

                coords_list.append(coords)
                ca_atom_indices.append(idx_CA[0])
                residue_indices.append(j)

                # Side-chain orientation: anchored at the kept node (idx_CA[0], i.e. the node this residue
                # collapses into), a unit vector pointing to the centroid of the side-chain heavy atoms
                # (std_names indices 4..35, i.e. CB..CH2). GLY / missing side chain -> zero vector (GCP-safe).
                node_pos = X[idx_CA[0]]
                sc_atoms = [int(i) for i in atom_idx if 4 <= int(types[i]) <= 35]
                if len(sc_atoms) > 0:
                    v = X[sc_atoms].mean(dim=0) - node_pos
                    nrm = pt.linalg.norm(v)
                    v = v / nrm if float(nrm) > 1e-6 else pt.zeros(3, dtype=X.dtype)
                else:
                    v = pt.zeros(3, dtype=X.dtype)
                sc_list.append(v)

        angles=dihedrals(pt.from_numpy(np.stack(coords_list)), sparse=True)
        sc_feat = pt.stack(sc_list).to(X.dtype)   # (N_ca, 3), node order
        
        mask=np.zeros(M.shape[0],dtype=bool)
        mask[ca_atom_indices]=1
        X,qn,qr,qe,M=X[mask],qn[mask],qr[mask],qe[mask],M[mask]
        N = X.shape[0]
        M = pt.eye(N, device=M.device, dtype=M.dtype)
        assert M.shape[0] == M.shape[1], f"M.shape = {M.shape}"
        # coords_array = np.stack(coords_list)              # (N_valid, 3, 3)
        # ca_atom_indices = np.array(ca_atom_indices)      # (N_valid,)
        # residue_indices = np.array(residue_indices)
        # #first turn the coords into an N_res-row form coords_before based on M and qn
        # coords_masked,mask=backbone_coords_by_res_mask(X, M, types, backbone_idx=[0,1,2])

        # #get the sequence from M and q
        # residue_id = np.array(M)[:,1]
        # residue_type = np.array(qr)[:,1]

        # _, idx = np.unique(residue_id, return_index=True)
        # # extract residue ids and types
        # residue_id_unique = residue_id[idx]
        # residue_type_unique = residue_type[idx]

        # # merge into (N_RES, 2)
        # qr_before = np.stack([residue_id_unique, residue_type_unique], axis=1)

        # #use dihedrals(coords, sparse=True) to get the masked angles and mask,
        # angles=dihedrals(pt.from_numpy(coords_masked), sparse=True)

        # sel_types=[0]
        # mask2 =np.isin(types, sel_types)
        # sel = pt.nonzero(pt.from_numpy(mask2), as_tuple=False).squeeze(-1)
        # # res_idx = M.argmax(dim=1)
        # # selected = []
        # # seen_res = set()
        # # for i in sel.tolist():
        # #     r = int(res_idx[i])
        # #     if r not in seen_res:
        # #         selected.append(i)
        # #         seen_res.add(r)

        # X,qn,qr,qe,M=X[sel],qn[sel],qr[sel],qe[sel],M[sel]


        # N = X.shape[0]
        # M = pt.eye(N, device=M.device, dtype=M.dtype)
        

        # X,qn,qr,qe,M=X[mask],qn[mask],qr[mask],qe[mask],M[mask]
        
        # mask coords_before
        #concat q and angles
        #build the diagonal M
    # elif representation_type == "BB":
    #         return ca_to_bb_repr(x)
    # elif representation_type == "FA":
    #     return ca_to_fa_repr(x)
    # elif representation_type == "BB_SC":
    #     return ca_to_bb_sc_repr(x)
    # elif representation_type == "CA_SC":
    #     return ca_to_ca_sc_repr(x)
    else:
        raise NotImplementedError("preprocess type not defined")
    

    ids_topk = extract_topology(X, config_data['max_num_nn'])[0]
    q={'qn':qn,'qe':qe,'qr':qr}
    q_l = []
    for fn in ftrs:
        q_l.append(load_sparse_mask_with_qbb(q, fn,ern_shape))
    q = pt.cat(q_l, dim=1)
    if preprocess_type=="CA_BB":
        #print(q.shape,angles.shape)
        q=pt.cat([q,angles],dim=1)
        if rsa is not None:
            # rsa lives in the original residue-column space (N_res,); residue_indices records the original column of each kept node
            rsa = np.asarray(rsa, dtype=np.float32)
            if len(residue_indices) == 0 or int(np.max(residue_indices)) >= rsa.shape[0]:
                raise ValueError(f"rsa 长度 {rsa.shape[0]} 与残基空间不匹配 (max residue_index={int(np.max(residue_indices)) if len(residue_indices) else 'NA'})")
            rsa_ca = pt.from_numpy(rsa[np.asarray(residue_indices, dtype=np.int64)]).unsqueeze(1)
            q = pt.cat([q, rsa_ca], dim=1)
    assert X.shape[0]==q.shape[0]==M.shape[0]
    return X,ids_topk,M,q,residue_indices,sc_feat

def extract_backbone(X,M,qe,qr,qn):
    
    types = qn[:, 1]

    # build a boolean mask: True marks the rows to keep
    mask =np.isin(types, [0])

    # get the indices that match
    sel = pt.nonzero(pt.from_numpy(mask), as_tuple=False).squeeze(-1)  # shape: [N_selected]

    X_ex,M_ex,qn_ex,qr_ex,qe_ex=X[sel],M[sel],qn[sel],qr[sel],qe[sel]
    ids_topk_ex = extract_topology(X_ex, config_dataset['max_num_nn'])[0]
    q_ex={'qn':qn_ex,'qe':qe_ex,'qr':qr_ex}

    return X_ex,ids_topk_ex,M_ex,q_ex

def positionToMask(pos_list,length):
    mask=np.zeros(length)
    for s,e in pos_list:
        mask[s:e]=1
    return mask.astype(bool)
        

def load_sparse_mask_with_qbb(q, fn,ern_shape):

    shape=tuple((q[fn].shape[0],ern_shape[fn][1]))
    
    M = pt.zeros(shape, dtype=pt.float)
    ids = pt.from_numpy(np.array(q[fn]).astype(np.int64))
    M.scatter_(1, ids[:,1:], 1.0)

    return M


DEFAULT_CLUSTER_LABELS_PATH = '../colabel/cluster_labels_summary_merged.pkl'

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODEL_DIR)


def resolve_project_path(path):
    """Resolve paths like ../colabel/... relative to project root, not process cwd."""
    if path is None or not isinstance(path, str) or not path.strip():
        return path
    if os.path.isabs(path):
        return path
    if path.startswith('../'):
        return os.path.normpath(os.path.join(_PROJECT_ROOT, path[3:]))
    return os.path.normpath(os.path.join(_MODEL_DIR, path))


def load_cluster_label_store(path):
    """cluster_name -> (n_res, 4) label matrix from merged PKL."""
    path = resolve_project_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"cluster labels file not found: {path}")
    with open(path, 'rb') as f:
        rows = pickle.load(f)
    store = {}
    for row in rows:
        if not isinstance(row, dict) or 'cluster_name' not in row or 'label' not in row:
            raise ValueError(f"Expected list[dict] with cluster_name/label in {path}")
        store[str(row['cluster_name'])] = np.asarray(row['label'], dtype=np.float32)
    print(f"[Dataset] loaded {len(store)} cluster labels from {path}")
    return store


def load_cluster_ref_seq_map(path):
    """cluster_name -> ref_seq from cluster label summary CSV or merged PKL."""
    path = resolve_project_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"cluster ref_seq file not found: {path}")
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            rows = pickle.load(f)
        ref_map = {}
        for row in rows:
            if not isinstance(row, dict) or 'cluster_name' not in row:
                raise ValueError(
                    f"Expected list[dict] with cluster_name/ref_seq in {path}, "
                    f"got {type(row)!r}"
                )
            ref_map[str(row['cluster_name'])] = (
                str(row['ref_seq']) if row.get('ref_seq') is not None else ''
            )
        print(f"[Dataset] loaded {len(ref_map)} cluster ref_seq entries from {path}")
        return ref_map

    df = pd.read_csv(path, usecols=['cluster_name', 'ref_seq'])
    ref_map = {
        str(row['cluster_name']): str(row['ref_seq']) if pd.notna(row['ref_seq']) else ''
        for _, row in df.iterrows()
    }
    print(f"[Dataset] loaded {len(ref_map)} cluster ref_seq entries from {path}")
    return ref_map


def build_cluster_to_ref_seq_map(mapping, cluster_ref_seq_map):
    """Map mapping[ukey]['cluster'] -> ref_seq using cluster representative ukeys in ref map."""
    cluster_to_ref = {}
    if not mapping:
        return cluster_to_ref
    for ukey, ref_seq in cluster_ref_seq_map.items():
        if ukey not in mapping:
            continue
        cluster = mapping[ukey].get('cluster')
        if cluster is not None:
            cluster_to_ref[str(cluster)] = ref_seq
    return cluster_to_ref


def resolve_ref_seq_via_cluster(ukey, mapping, cluster_ref_seq_map, cluster_to_ref_seq):
    """ukey -> cluster (via chain mapping) -> ref_seq from merged cluster labels."""
    if isinstance(ukey, (list, tuple)):
        ukey = ukey[0]
    ukey = str(ukey)
    if ukey in cluster_ref_seq_map:
        return cluster_ref_seq_map[ukey]
    if mapping and ukey in mapping:
        cluster = mapping[ukey].get('cluster')
        if cluster is not None:
            ref_seq = cluster_to_ref_seq.get(str(cluster))
            if ref_seq is not None:
                return ref_seq
    raise KeyError(f"ref_seq not found for ukey={ukey!r} via cluster mapping")


def resolve_ref_seq(ukey, mapping, cluster_ref_seq_map, ref_db):
    """Resolve reference sequence for mask alignment (CSV first, then ref_db fallback)."""
    if isinstance(ukey, (list, tuple)):
        ukey = ukey[0]
    if ukey in cluster_ref_seq_map:
        return cluster_ref_seq_map[ukey]
    if mapping and ukey in mapping:
        cluster = mapping[ukey].get('cluster', ukey)
        if cluster in cluster_ref_seq_map:
            return cluster_ref_seq_map[cluster]
        return ref_db[mapping[ukey]['ref']]
    raise KeyError(f"ref_seq not found for ukey={ukey!r}")


def add_cluster_based_noise(labels, cluster_ids, noise_std=0.02):
    """Add independent (per-element) noise to labels within the same cluster."""
    labels = np.asarray(labels, dtype=np.float32)
    noise = np.random.normal(0, noise_std, size=labels.shape)
    mask_one = (labels == 1.0)
    noise[mask_one] = noise[mask_one] * 0.3
    labels_noisy = labels + noise
    return np.clip(labels_noisy, 0, 1)


def postprocess_cluster_labels(y_np, label_noise_std=0.0, binarize_threshold=None):
    y_np = np.asarray(y_np, dtype=np.float32)
    if label_noise_std > 0:
        y_np = add_cluster_based_noise(y_np, None, noise_std=label_noise_std)
    if binarize_threshold is not None:
        y_np = (y_np >= binarize_threshold).astype(np.float32)
    return y_np


def load_interface_labels(hgrp, t0, t1_l):
    # load stored data
    shape = tuple(hgrp.attrs['Y_shape'])
    ids = pt.from_numpy(np.array(hgrp['Y']).astype(np.int64))

    # matching contacts type for receptor and ligand
    y_ctc_r = pt.any((ids[:,2].view(-1,1) == t0), dim=1).view(-1,1)
    y_ctc_l = pt.stack([pt.any((ids[:,3].view(-1,1) == t1), dim=1) for t1 in t1_l], dim=1)
    y_ctc = (y_ctc_r & y_ctc_l)

    # save contacts of right type
    y = pt.zeros((shape[0], len(t1_l)), dtype=pt.bool)
    #[N_RES,5]
    try:
        row_ids = ids[:, 0]                     # [11]
        pair_idx, col_idx = pt.where(y_ctc)                          # [N_true]
        y[row_ids[pair_idx], col_idx] = True
    except Exception as e:
        print(e,"error_ckey",hgrp.name,
              "y_ctc_r",y_ctc_r.shape,
            "y_ctc_l" ,y_ctc_l.shape,
            "y_ctc", y_ctc.shape,
            "ids",ids.shape,
            "y",y.shape,"shape",shape)
        
    return y


def collate_batch_data_key_pyg(batch_data):

    data_list = []
    # extract sizes
    sizes = pt.tensor([data[3].shape for data in batch_data])


    y=[]
    keys=[]
    for idx,data in enumerate(batch_data):

        # 9-tuple (single-pocket sampling: +pocket_mask +y_union) / 7-tuple (sc_feat at the end) / legacy 6-tuple
        pm_i = yu_i = None
        if len(data) >= 9:
            X_i, ids_i, q_i, M_i, y_i, key_i, sc_i, pm_i, yu_i = data[:9]
        elif len(data) >= 7:
            X_i, ids_i, q_i, M_i, y_i, key_i, sc_i = data[:7]
        else:
            X_i, ids_i, q_i, M_i, y_i, key_i = data[:6]
            sc_i = None
        src = []
        dst = []
        y.append(y_i)
        keys.append(key_i)

        data_kwargs = dict(
            pos=X_i,                       # (Ni, 3)
            x=q_i,           # (Ni, 1)
            ids_topk=ids_i,
            batch=idx,
            M=M_i,
        )
        if sc_i is not None:
            data_kwargs["sc"] = sc_i       # (Ni, 3) side-chain orientation unit vectors, as a node attribute
        # Single-pocket sampling: pm/yu ride as pyg node attributes, concatenated along dim0 with the graph (same mechanism as sc).
        # They are node attributes rather than an extra return value so the collate return signature stays unchanged — eval_step
        # unpacks via data_list[-1]/[-2] and main_regression reuses this collate; changing the signature would break both.
        if pm_i is not None:
            data_kwargs["pm"] = pm_i       # (Ni, 4) loss mask: 1=active, 0=ignore
        if yu_i is not None:
            data_kwargs["yu"] = yu_i       # (Ni, 4) union labels, used for metrics only
        pyg_data = Data(**data_kwargs)
        data_list.append(pyg_data)


    y = pt.cat(y,dim=0)
    batch = Batch.from_data_list(data_list)

    return batch,y,keys




class Dataset(pt.utils.data.Dataset):
    def __init__(self, config):
        
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = config.get("dataset_filepath")

        self.preprocess = config.get("preprocess", None)
        self.nn = config.get("max_num_nn", None)
        # selected features
        self.ftrs = [fn for fn, ff in zip(['qe','qr','qn'], config.get("features_flags",(True,False,False))) if ff]

        # Optional per-residue RSA (relative solvent accessibility) feature: a sidecar h5
        # generated by processing/add_sasa_to_h5.py. When set, __getitem__ reads
        # data/structures/<key>/rsa and cats it as the last node-scalar column in the CA_BB
        # preprocess; config_model's h_input_dim must be bumped by +1 (e.g. 35+1536 -> 36+1536).
        self.rsa_filepath = resolve_project_path(config.get("rsa_filepath", None))
        self._rsa_hf = None          # lazily opened persistent handle (each DataLoader worker opens its own)
        if self.rsa_filepath is not None and not os.path.isfile(self.rsa_filepath):
            raise FileNotFoundError(
                f"rsa_filepath 不存在: {self.rsa_filepath!r}; "
                f"先跑 processing/add_sasa_to_h5.py 生成 sidecar"
            )
        if self.rsa_filepath is not None and self.preprocess != "CA_BB":
            raise ValueError("rsa_filepath 需要 preprocess='CA_BB'")

        self.colabel_path=config.get("colabel_path", None)
        self.return_key = bool(config.get("return_key", False))
        self.mask_only = config.get("mask_only", False)
        self.plm=config.get("plm", None)
        self.cluster_labels_path = config.get("cluster_labels_path", None)
        self.label_noise_std = float(config.get("label_noise_std", 0.0))
        self.label_binarize_threshold = config.get("label_binarize_threshold", None)
        if self.label_binarize_threshold is not None:
            self.label_binarize_threshold = float(self.label_binarize_threshold)
        self.cluster_label_store = None

        # ---- Single-pocket label sampling + ignoring unsampled positives (pocket_mask branch) ----
        # Toggle: config['pocket_sample'] (off by default). When on, each __getitem__ independently samples,
        # for each of the 4 task heads, 1 candidate from "this cluster's colabel pockets + native (this structure's
        # own contacts)" as this round's positives; other true positives are ignored (mask=0, excluded from loss);
        # residues in no pocket and not in native stay negatives (mask=1, y=0) — the negative set matches the
        # union baseline position by position. native wins with probability pocket_native_prob (default 0.2), on par
        # with the pockets. When off, the path is identical to before (pocket_mask=None), comparable against baseline.
        self.pocket_sample = bool(config.get("pocket_sample", False))
        self.native_prob = float(config.get("pocket_native_prob", 0.2))
        self.pocket_store = None
        if self.pocket_sample:
            ps_path = resolve_project_path(config.get("pocket_store_path", None))
            if not ps_path or not os.path.isfile(ps_path):
                raise FileNotFoundError(
                    f"pocket_sample=True 但 pocket_store 不存在: {ps_path!r}; "
                    f"先跑 colabel/step3_build_pocket_store.py 生成 pocket_store.pkl"
                )
            with open(ps_path, 'rb') as pf:
                self.pocket_store = pickle.load(pf)
            meta = self.pocket_store.pop(("__meta__",), None)
            print(f"[Dataset] pocket_sample=ON  store={ps_path}  "
                  f"clusters={len(self.pocket_store)}  native_prob={self.native_prob}  "
                  f"meta={meta.get('format') if meta else None}")

        # New pairwise alignment mask: {ukey: {'ref':(m_ref,m_struct_r),'fasta':(m_fasta,m_struct_f)}}
        # If provided in pairwise format and preprocess==CA_BB, __getitem__ aligns/scatters the labels
        # onto the filtered structure nodes (N_ca) right at the data layer, so no mask is needed during training. Old format / absent: keep the old behavior.
        self.mask_dict = None
        self.mask_new_format = False
        _mask_dict_path = config.get("mask_dict_path", None)
        if _mask_dict_path and os.path.isfile(_mask_dict_path):
            with open(_mask_dict_path, "rb") as _mf:
                self.mask_dict = pickle.load(_mf)
            _meta = self.mask_dict.get(("__meta__",)) if isinstance(self.mask_dict, dict) else None
            if _meta is not None and _meta.get("format") == "pairwise":
                self.mask_new_format = (self.preprocess == "CA_BB")
                if _meta.get("preprocess") not in (None, self.preprocess):
                    print(f"[Dataset] WARN mask_dict preprocess={_meta.get('preprocess')} "
                          f"!= current {self.preprocess}; 对齐可能错位, 建议重建 mask")
            print(f"[Dataset] loaded mask_dict from {_mask_dict_path} "
                  f"(new_format={self.mask_new_format})")
            if not self.mask_new_format:
                # old format / non-CA_BB: unused at the data layer (eval_step loads it separately); free the memory
                self.mask_dict = None

        # preload data
        with h5py.File(self.dataset_filepath,'r') as hf:
            # load keys, sizes and types
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype('U'))
            self.sizes = np.array(hf["metadata/sizes"])
            self.ckeys = np.array(hf["metadata/ckeys"]).astype(np.dtype('U'))
            self.ctypes = np.array(hf["metadata/ctypes"])
            # load parameters to reconstruct data
            self.std_elements = np.array(hf["metadata/std_elements"]).astype(np.dtype('U'))
            self.std_resnames = np.array(hf["metadata/std_resnames"]).astype(np.dtype('U'))
            self.std_names = np.array(hf["metadata/std_names"]).astype(np.dtype('U'))
            self.mids = np.array(hf["metadata/mids"]).astype(np.dtype('U'))
        

        # set default selection mask
        self.m = np.ones(len(self.keys), dtype=bool)

        # prepare ckeys mapping
        self.__update_selection()

        # set default runtime selected interface types
        self.t0 = pt.arange(self.mids.shape[0])
        self.t1_l = [pt.arange(self.mids.shape[0])]

        if self.cluster_labels_path:
            self.cluster_label_store = load_cluster_label_store(self.cluster_labels_path)
            if self.colabel_path and self.colabel_path.get('chain_mapping'):
                with open(self.colabel_path['chain_mapping'], 'rb') as m:
                    self.chain_mapping = pickle.load(m)
                self.colabel = None
            else:
                self.colabel, self.chain_mapping = None, None
        elif self.colabel_path:
            print(self.colabel_path)
            with open(self.colabel_path['colabel'],'rb')as c,open(self.colabel_path['chain_mapping'],'rb')as m:
                self.colabel,self.chain_mapping=pickle.load(c),pickle.load(m)

    def __getstate__(self):
        # h5py handles cannot be pickled: if the main process has one open, drop it before
        # forking/spawning DataLoader workers; each worker lazily reopens it on first __getitem__.
        state = self.__dict__.copy()
        state['_rsa_hf'] = None
        return state

    def __update_selection(self):
        # ckeys mapping with keys
        self.ckeys_map = {}
        for key, ckey in zip(self.keys[self.m], self.ckeys[self.m]):
            if key in self.ckeys_map:
                self.ckeys_map[key].append(ckey)
            else:
                self.ckeys_map[key] = [ckey]

        # keep unique keys
        self.ukeys = list(self.ckeys_map)

    def update_mask(self, m):
        # update mask
        self.m &= m

        # update ckeys mapping
        self.__update_selection()

    def set_types(self, l_types, r_types_l):
        self.t0 = pt.from_numpy(np.where(np.isin(self.mids, l_types))[0])
        self.t1_l = [pt.from_numpy(np.where(np.isin(self.mids, r_types))[0]) for r_types in r_types_l]

    def _combine_labels(self, native, scattered):
        """Union of the structure's own labels (native) and the scattered/aligned colabel.
        Base class (classification): logical OR -> strictly {0,1}. The regression subclass switches to max to keep continuous values.
        native/scattered: (N_ca, C) float; scattered is 0 where unaligned."""
        return ((native > 0.5) | (scattered > 0.5)).float()

    def _get_colabel_full(self, cluster):
        """cluster -> (len_ref, C) projected labels (on ref_seq); None if missing.
        The cluster_label_store branch applies noise/binarize post-processing; the colabel branch returns raw values."""
        if self.cluster_label_store is not None:
            y_np = self.cluster_label_store.get(cluster)
            if y_np is None:
                return None
            return postprocess_cluster_labels(
                y_np,
                label_noise_std=self.label_noise_std,
                binarize_threshold=self.label_binarize_threshold,
            )
        colabel = getattr(self, 'colabel', None)
        if colabel is not None:
            try:
                return colabel[cluster]
            except (KeyError, IndexError):
                return None
        return None

    def _native_ca(self, hf, ckeys, residue_indices, n_ca):
        """This structure's own contact labels (native), mapped into the filtered structure-node space -> (N_ca, C) float.
        Union over multiple ckeys. Returns None if reliable alignment is impossible."""
        if residue_indices is None:
            return None
        ri = pt.as_tensor(list(residue_indices), dtype=pt.long)
        if ri.numel() != n_ca:
            return None
        native_full = None
        for ckey in ckeys:
            yl = load_interface_labels(hf['data/contacts/' + ckey], self.t0, self.t1_l)
            native_full = yl.clone() if native_full is None else (native_full | yl)
        if native_full is None:
            return None
        if int(ri.max()) >= native_full.shape[0]:
            # residue_indices exceed the native length -> inconsistent spaces, fall back
            return None
        return native_full[ri].float()                            # (N_ca, C)

    def _ref_scatter_masks(self, ukey, n_ca, ref_len):
        """Take the ref components (m_ref, m_struct_r) of the pairwise mask and check compatibility with ref_len/n_ca.
        colabel and all pocket labels live in the same ref coordinate system (lengths verified identical), so they share this scatter mask.
        Incompatible -> (None, None)."""
        md = self.mask_dict.get(ukey) if self.mask_dict is not None else None
        if md is None:
            return None, None
        m_ref = pt.as_tensor(np.asarray(md['ref'][0])).bool()
        m_struct_r = pt.as_tensor(np.asarray(md['ref'][1])).bool()
        if (m_ref.numel() != ref_len or m_struct_r.numel() != n_ca
                or int(m_ref.sum()) != int(m_struct_r.sum())):
            return None, None
        return m_ref, m_struct_r

    def _build_aligned_label(self, hf, ckeys, cluster, residue_indices, n_ca, ukey):
        """Build aligned labels in structure space (N_ca) = native (structure's own) ∪ colabel (cluster projection scattered).
        Returns an (N_ca, C) tensor; None if reliable alignment is impossible (the caller falls back to the old path)."""
        if self.mask_dict is None:
            return None
        native_ca = self._native_ca(hf, ckeys, residue_indices, n_ca)
        if native_ca is None:
            return None

        # scatter the colabel projected labels onto N_ca
        scattered = pt.zeros_like(native_ca)
        colabel_full = self._get_colabel_full(cluster)
        if colabel_full is not None:
            cf = pt.as_tensor(np.asarray(colabel_full)).float()
            if cf.ndim == 1:
                cf = cf.unsqueeze(1)
            m_ref, m_struct_r = self._ref_scatter_masks(ukey, n_ca, cf.shape[0])
            if m_ref is not None and cf.shape[1] == scattered.shape[1]:
                scattered[m_struct_r] = cf[m_ref]
            else:
                print(f"[Dataset] WARN skip colabel scatter {ukey}: "
                      f"cf={tuple(cf.shape)} n_ca={n_ca}")
        return self._combine_labels(native_ca, scattered)

    def _build_pocket_label(self, hf, ckeys, cluster, residue_indices, n_ca, ukey):
        """Single-pocket label sampling + ignoring unsampled positives. Returns (y, pocket_mask, y_union), all (N_ca, C) float.

        y_union     : native ∪ colabel = the baseline union labels (metrics only, not in loss)
        y           : the single pocket selected this round (each of the 4 task heads samples independently)
        pocket_mask : 1=included in loss, 0=ignore. Exactly the unselected positives of y_union are ignored,
                      so the negative set (y=0 and mask=1) matches the union baseline position by position — no false negatives.

        Each task head's candidate pool = this cluster's colabel pockets for that task + native (this structure's own contacts);
        native is selected with probability self.native_prob, otherwise one pocket is chosen uniformly at random.
        Empty candidate pool and empty union -> the task is all negatives (empty label).
        If reliable alignment is impossible -> (None, None, None); the caller falls back to union.
        """
        y_union = self._build_aligned_label(hf, ckeys, cluster, residue_indices, n_ca, ukey)
        if y_union is None:
            return None, None, None
        native_ca = self._native_ca(hf, ckeys, residue_indices, n_ca)
        if native_ca is None:
            return None, None, None

        C = y_union.shape[1]

        # scatter this cluster's per-task pockets (sparse indices in ref coordinates) into structure space (N_ca,)
        pockets_ca = {c: [] for c in range(C)}
        entry = self.pocket_store.get(cluster) if self.pocket_store is not None else None
        if entry is not None:
            ref_len = int(entry['ref_len'])
            m_ref, m_struct_r = self._ref_scatter_masks(ukey, n_ca, ref_len)
            if m_ref is not None:
                for c, plist in entry['pockets'].items():
                    if c >= C:
                        continue
                    for idx in plist:
                        p_ref = pt.zeros(ref_len, dtype=pt.bool)
                        p_ref[pt.as_tensor(np.asarray(idx), dtype=pt.long)] = True
                        p_ca = pt.zeros(n_ca, dtype=pt.bool)
                        p_ca[m_struct_r] = p_ref[m_ref]
                        if bool(p_ca.any()):   # pockets that come up empty after scattering (all fell into alignment gaps) -> drop
                            pockets_ca[c].append(p_ca)

        y = pt.zeros_like(y_union)
        pocket_mask = pt.ones_like(y_union)

        for c in range(C):
            u = y_union[:, c].bool()           # baseline positive set (native ∪ colabel)
            cands = pockets_ca[c]
            nat = native_ca[:, c].bool()
            has_nat = bool(nat.any())

            if cands and (not has_nat or random.random() >= self.native_prob):
                sel = cands[random.randrange(len(cands))]     # pick one colabel pocket uniformly
            elif has_nat:
                sel = nat                      # native: a candidate on par with the pockets (probability native_prob)
            elif bool(u.any()):
                sel = u                        # fallback: colabel has positives but no pocket rows -> treat the whole set as one pocket
            else:
                continue                       # truly empty label for this task: y=0, mask=1 -> all negatives

            y[:, c] = sel.float()
            pocket_mask[:, c] = (~u).float()   # baseline positives are ignored by default
            pocket_mask[sel, c] = 1.0          # the selected one re-enters the loss (applied last, so overlap is handled correctly)

        return y, pocket_mask, y_union

    def get_largest(self):
        i = np.argmax(self.sizes[:,0] * self.m.astype(int))
        k = np.where(np.isin(self.ukeys, self.keys[i]))[0][0]
        return self[k]

    def __len__(self):
        return len(self.ukeys)

    def __getitem__(self, k):
        # get corresponding interface keys
        #try:
            key = self.ukeys[k]
            ckeys = self.ckeys_map[key]
            

            # load data
            with h5py.File(self.dataset_filepath, 'r') as hf:
                # hdf5 group
                hgrp = hf['data/structures/'+key]
                ern_shape={"qe":hgrp.attrs["qe_shape"],"qn":hgrp.attrs["qn_shape"],"qr":hgrp.attrs["qr_shape"]}
                # topology
                X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32))
                M = load_sparse_mask(hgrp, 'M')
                residue_indices = None
                sc_feat = None
                rsa = None
                if self.rsa_filepath is not None:
                    if self._rsa_hf is None:
                        self._rsa_hf = h5py.File(self.rsa_filepath, 'r')
                    try:
                        rsa = np.array(self._rsa_hf['data/structures/' + key + '/rsa'])
                    except KeyError:
                        raise KeyError(
                            f"RSA sidecar 缺少 {key}: 请补跑 processing/add_sasa_to_h5.py"
                        )
                if self.preprocess:
                    X,ids_topk,M,q,residue_indices,sc_feat = preprocess(X,M,hgrp['qe'],hgrp['qr'],hgrp['qn'],self.preprocess,self.ftrs,ern_shape,rsa=rsa)
                else:
                    q_l = []
                    for fn in self.ftrs:
                        q_l.append(load_sparse_mask(hgrp, fn))
                    q = pt.cat(q_l, dim=1)
                    ids_topk = pt.from_numpy(np.array(hgrp['ids_topk']).astype(np.int64))

                if self.mask_only:
                    return X, ids_topk, q, M, self.ukeys[k]

                # New pairwise flow: align/scatter the labels onto the filtered structure nodes (N_ca)
                # at the data layer — native ∪ colabel; the returned y corresponds 1:1 to graph nodes, so no mask is needed during training.
                if self.mask_new_format:
                    cluster = hgrp['cluster'][()]
                    if isinstance(cluster, bytes):
                        cluster = cluster.decode('utf-8')
                    n_ca = M.shape[0]  # after CA_BB, M=eye(N_ca)

                    if self.pocket_sample:
                        # Single-pocket sampling: y=the pocket selected this round, pm=loss mask (0=ignore),
                        # yu=union labels (fed to metrics only, not in loss). On failure, fall back to union.
                        y, pm, yu = self._build_pocket_label(
                            hf, ckeys, cluster, residue_indices, n_ca, self.ukeys[k]
                        )
                        if y is not None:
                            return (X, ids_topk, q, M, y.float(), self.ukeys[k], sc_feat,
                                    pm.float(), yu.float())

                    y = self._build_aligned_label(
                        hf, ckeys, cluster, residue_indices, n_ca, self.ukeys[k]
                    )
                    if y is not None:
                        return X, ids_topk, q, M, y.float(), self.ukeys[k], sc_feat
                    # alignment failed -> fall back to the old branch below

                if self.cluster_label_store is not None:
                    cluster = hgrp['cluster'][()]
                    if isinstance(cluster, bytes):
                        cluster = cluster.decode('utf-8')
                    y_np = self.cluster_label_store.get(cluster)
                    if y_np is None:
                        y_np = np.zeros((M.shape[1], 4), dtype=np.float32)
                    y_np = postprocess_cluster_labels(
                        y_np,
                        label_noise_std=self.label_noise_std,
                        binarize_threshold=self.label_binarize_threshold,
                    )
                    y = pt.from_numpy(y_np).float()
                    return X, ids_topk, q, M, y, self.ukeys[k], sc_feat

                if self.colabel_path:
                    colabel=self.colabel
                    chain_mapping=self.chain_mapping
                    cluster = hgrp['cluster'][()]
                    if isinstance(cluster, bytes):
                        cluster = cluster.decode('utf-8')
                    colabel_chain=colabel[cluster]

                    y=pt.from_numpy(colabel_chain).float()

                    return X, ids_topk, q, M, y.float(),self.ukeys[k], sc_feat


                else:
                    # interface labels
                    y = pt.zeros((M.shape[1], len(self.t1_l)), dtype=pt.bool)
                    for ckey in ckeys:
                        y |= load_interface_labels(hf['data/contacts/'+ckey], self.t0, self.t1_l)                    
                    
            if self.return_key:
                return X, ids_topk, q, M, y.float(), self.ukeys[k], sc_feat
            return X, ids_topk, q, M, y.float()

        # except Exception as e:
        #     print(f"[WARNING] Skipping {self.ukeys[k]} due to error: {e}")
        #     # 👉 take the next sample
        #     return self.__getitem__((k + 1) % len(self))
if __name__ == '__main__':
    dataset = Dataset(**config_data)
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=config_runtime['batch_size'], shuffle=True, num_workers=8, collate_fn=collate_batch_data_key_pyg, pin_memory=True, prefetch_factor=2)
