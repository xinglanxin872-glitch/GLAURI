"""Structure encoding, ESM3 embeddings and inference for the current GLAURI model."""
import os
import numpy as np
import torch as pt
import h5py
INFERENCE_FEATURE_PROTOCOL = 'legacy_nodes_angles_sidechain_v1_20261102'
from torch_geometric.data import Data, Batch
from graphein.protein.tensor.angles import dihedrals
from src.structure_io import read_cif, read_pdb
from src.data_encoding import encode_structure, encode_features, extract_topology, std_resnames, std_elements
from src.structure import clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits, concatenate_chains
from model.data_handler import load_seq
from model.geometry import residue_geometry, CA_PROTOCOL, LEGACY_PROTOCOL
from curation.add_sasa_to_h5 import shrake_rupley_per_atom, MAX_ASA, VDW_RADII, FALLBACK_MAX_ASA, FALLBACK_RADIUS
_ELEM_RADIUS_LUT = np.array([VDW_RADII.get(e, FALLBACK_RADIUS) for e in std_elements] + [FALLBACK_RADIUS], dtype=np.float64)
_RES_MAXASA_LUT = np.array([MAX_ASA.get(r, FALLBACK_MAX_ASA) for r in std_resnames] + [FALLBACK_MAX_ASA], dtype=np.float64)

def rsa_from_tensors(X, M, qe, qr):
    """Per-residue RSA (N_res,) from encoded structure tensors (pre-preprocess).

    X: (N_atom,3); M: (N_atom,N_res) bool; qe/qr: onehot (N_atom, n_class+1).
    Column order of M = pt.unique(resid) order, matching CA_BB residue_indices.
    """
    elem_idx = qe.argmax(1).cpu().numpy()
    resname_idx = qr.argmax(1).cpu().numpy()
    atom_sasa = shrake_rupley_per_atom(X.cpu().numpy().astype(np.float64), _ELEM_RADIUS_LUT[elem_idx])
    res_of_atom = M.cpu().numpy().argmax(1)
    n_res = M.shape[1]
    res_sasa = np.zeros(n_res, dtype=np.float64)
    np.add.at(res_sasa, res_of_atom, atom_sasa)
    _, first = np.unique(res_of_atom, return_index=True)
    rsa = res_sasa / _RES_MAXASA_LUT[resname_idx[first]]
    return np.clip(rsa, 0.0, 1.0).astype(np.float32)

def get_plm_model(device, plm_type='esm3'):
    if plm_type != 'esm3':
        raise ValueError('This release uses ESM3 sequence embeddings')
    from esm3lib.models.esm3 import ESM3
    client = ESM3.from_pretrained('esm3-open', device=pt.device('cpu'))
    if pt.device(device).type != 'cpu':
        client = client.to(pt.bfloat16)
    return [client.to(device)]

def open_plm_embedding_h5(path):
    """打开预计算 PLM embedding 的 HDF5（调大 chunk cache，适合大文件反复按 key 读）。"""
    if path is None or path == '':
        return None
    return h5py.File(path, 'r', libver='latest', rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=1000003, rdcc_w0=0.75)

def close_plm_embedding_handles(h5_train_f, h5_valid_f):
    """关闭 train/valid 的 embedding H5；若 valid 与 train 为同一文件对象则只关一次。"""
    if h5_valid_f is not None and h5_valid_f is not h5_train_f:
        try:
            h5_valid_f.close()
        except Exception:
            pass
    if h5_train_f is not None:
        try:
            h5_train_f.close()
        except Exception:
            pass

def get_embedding(seq, device, plm_type, plm_pack, ukey=None, h5_path=None, h5_file=None):
    """
    预计算 embedding：优先使用已打开的 ``h5_file``（训练循环内复用，避免每个 batch 反复 open 大文件）；
    否则在传入 ``h5_path`` 时临时打开文件。
    """
    if h5_file is not None:
        with pt.cuda.amp.autocast():
            arr = h5_file[ukey][:]
            plm_embedding = pt.from_numpy(arr).float()
        return plm_embedding
    if h5_path is not None:
        with pt.cuda.amp.autocast():
            with h5py.File(h5_path, 'r', libver='latest', rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=1000003, rdcc_w0=0.75) as hf1:
                arr = hf1[ukey][:]
                plm_embedding = pt.from_numpy(arr).float()
        return plm_embedding
    if plm_type != 'esm3':
        raise ValueError('This release uses ESM3 sequence embeddings')
    return _esm3lib_embed(seq, plm_pack)

def _esm3lib_embed(seq, plm_pack):
    from esm3lib.sdk.api import ESMProtein, LogitsConfig
    client = plm_pack[0]
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    return logits_output.embeddings.squeeze(0)[1:-1, :].cpu()

def prepocess_sparse(X, M, qe, qr, qn, preprocess_type, ftrs, max_nn, with_rsa=False, protocol=LEGACY_PROTOCOL):
    qn_idx = pt.stack(pt.where(qn), dim=1)
    types = qn_idx[:, 1]
    q = {}
    if preprocess_type == 'extract_backbone':
        sel_types = [0, 1, 2, 3, 4]
        mask = np.isin(types, sel_types)
        sel = pt.nonzero(pt.from_numpy(mask), as_tuple=False).squeeze(-1)
        X, M, qn, qr, qe = (X[sel], M[sel], qn[sel], qr[sel], qe[sel])
    elif preprocess_type == 'CA_only':
        sel_types = [0]
        mask = np.isin(types, sel_types)
        sel = pt.nonzero(pt.from_numpy(mask), as_tuple=False).squeeze(-1)
        res_idx = pt.where(M)[1]
        selected = []
        seen_res = set()
        for i in sel.tolist():
            r = int(res_idx[i])
            if r not in seen_res:
                selected.append(i)
                seen_res.add(r)
        sel = pt.tensor(selected, device=X.device)
        X, M, qn, qr, qe = (X[sel], M[sel], qn[sel], qr[sel], qe[sel])
        N = M.shape[0]
        M = pt.eye(N, device=M.device, dtype=M.dtype)
    elif preprocess_type == 'CA_BB':
        rsa_full = rsa_from_tensors(X, M, qe, qr) if with_rsa else None
        geom=residue_geometry(X,M.to(pt.int64).argmax(dim=1).cpu().numpy(),types,M.shape[1],protocol)
        q['angles']=geom['angles']
        q['bb_coords']=geom['backbone'].numpy()
        q['sc']=geom['sidechain']
        if rsa_full is not None:
            q['rsa']=pt.from_numpy(rsa_full[geom['residues']]).unsqueeze(1)
        selected=pt.from_numpy(geom['node_indices'])
        X,qn,qr,qe=X[selected],qn[selected],qr[selected],qe[selected]
        M=pt.eye(len(selected),dtype=M.dtype,device=M.device)
    else:
        raise NotImplementedError('preprocess type not defined')
    ids_topk = extract_topology(X, max_nn)[0]
    q['qn'] = qn
    q['qe'] = qe
    q['qr'] = qr
    return (X, ids_topk, M, q)

def process_structure(path, chainid=None):
    if os.path.exists(path) == False:
        print(f'{path} not found')
        return []
    if os.path.basename(path).split('.')[1] == 'cif':
        structure = read_cif(path)
    else:
        structure = read_pdb(path)
    structure = clean_structure(structure)
    structure = tag_hetatm_chains(structure)
    subunits = split_by_chain(structure)
    subunits = filter_non_atomic_subunits(subunits)
    subunits = remove_duplicate_tagged_subunits(subunits)
    if chainid:
        chain_key = chainid.split('/')[-1]
        if chain_key not in subunits:
            print(f'{chainid!r}: chain {chain_key!r} not found among subunits {list(subunits)}')
            return []
        structure = subunits[chain_key]
    else:
        structure = concatenate_chains(subunits)
    return structure

def process_seq_batch(structure, feature_flags, preprocess_type, model_frame, max_nn, with_rsa=False, protocol=LEGACY_PROTOCOL):
    ftrs = [fn for fn, ff in zip(['qe', 'qr', 'qn'], feature_flags) if ff]
    X, M = encode_structure(structure)
    q_l = []
    q = list(encode_features(structure))
    q_dict = {'qe': q[0], 'qr': q[1], 'qn': q[2]}
    if preprocess_type:
        X, ids_topk, M, q_dict = prepocess_sparse(X, M, q_dict['qe'], q_dict['qr'], q_dict['qn'], preprocess_type, ftrs, max_nn, with_rsa=with_rsa, protocol=protocol)
    else:
        ids_topk, _, _, _, _ = extract_topology(X, max_nn)
    q_l = []
    qr_idx = pt.stack(pt.where(q_dict['qr']), dim=1)
    for fn in ftrs:
        q_l.append(q_dict[fn])
    q = pt.cat(q_l, dim=1)
    if 'angles' in q_dict:
        q = pt.cat([q, q_dict['angles']], dim=1)
    if 'rsa' in q_dict:
        q = pt.cat([q, q_dict['rsa']], dim=1)
    dense_M = pt.stack(pt.where(M), dim=1)
    data_list = []
    pyg_data = Data(pos=X, x=q, ids_topk=ids_topk, batch=0, M=dense_M)
    if 'bb_coords' in q_dict:
        pyg_data.bb_coords = pt.from_numpy(q_dict['bb_coords'])
    if 'sc' in q_dict:
        pyg_data.sc = q_dict['sc']
    pyg_data.feature_protocol=protocol
    data_list.append(pyg_data)
    seq = load_seq(dense_M, qr_idx)
    return (data_list, seq)

def inference(model, device, data_list, model_frame, mask=None, plm_embedding=None, plm_fusion='concat'):
    batch = Batch.from_data_list(data_list)
    actual=getattr(batch,'feature_protocol',[LEGACY_PROTOCOL])
    actual=set(actual if isinstance(actual,list) else [actual])
    expected=getattr(model,'feature_protocol',LEGACY_PROTOCOL)
    if actual!={expected}:
        raise ValueError(f'Checkpoint/feature protocol mismatch: model={expected}, graph={actual}')
    if getattr(model, 'sc_vector', False) and getattr(batch, 'sc', None) is None:
        raise ValueError('Side-chain vectors are enabled but missing from the inference graph')
    if mask is not None:
        mask = pt.as_tensor(mask).bool()
        batch = batch.subgraph(mask)
    batch = batch.to(device, non_blocking=True)
    if plm_embedding is not None:
        batch.x = batch.x.to(device)
        plm_embedding = plm_embedding.to(device)
        batch.x = pt.cat([batch.x, plm_embedding], dim=1)
    with pt.inference_mode():
        z = model(batch)['node_embedding']
    p = pt.sigmoid(z)
    pc = pt.cat([1.0 - pt.max(p, axis=1)[0].unsqueeze(1), p], axis=1).cpu()
    return pc
