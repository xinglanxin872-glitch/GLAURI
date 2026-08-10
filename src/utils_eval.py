import os
import re

import numpy as np
import torch as pt
import h5py
import esm
from torch_geometric.data import Data, Batch
from graphein.protein.tensor.angles import dihedrals

from src.structure_io import read_cif, read_pdb
from src.dataset import collate_batch_features
from src.data_encoding import encode_structure, encode_features, extract_topology, std_resnames, std_elements
from src.structure import (
    clean_structure, tag_hetatm_chains, split_by_chain,
    filter_non_atomic_subunits, remove_duplicate_tagged_subunits, concatenate_chains,
)
from model.data_handler import load_seq
from curation.add_sasa_to_h5 import shrake_rupley_per_atom, MAX_ASA, VDW_RADII, FALLBACK_MAX_ASA, FALLBACK_RADIUS

# esmc/esm3 live in the EvolutionaryScale SDK, which also ships as top-level `esm`
# and would shadow fair-esm above. It is vendored as `esm3lib` instead and imported
# lazily in the branches below. See plm_env_port/CLUSTER_SETUP.md.

# ProtT5 weights dir; override via env if running off this machine (same pattern as
# src/emb_cache.py's GLAURI_EMB_CACHE_DIR).
PROT_T5_MODEL_DIR = os.environ.get("GLAURI_PROT_T5_DIR", "/home/lanxin/data/plm/prot_t5_xl")

# ESM2-family repr layer per size. Only the 650M checkpoint ("ESM2") is wired up below --
# 3B/15B variants existed here at one point (ESM2_3b/esm2_3b -- a casing fork with
# inconsistent dtype handling -- and esm2_15b) but grep against every shipped config
# (model/config.py, model/GLAURI/*_base_config.py, ablation/config_*.py) found zero
# references to any of them; removed as dead code. To bring one back: add its
# `esm.pretrained.esm2_t*` loader to get_plm_model's "ESM2" branch and its layer count
# here, both call sites are otherwise generic.
_ESM2_REPR_LAYER = {"ESM2": 33}

# --- per-residue RSA feature (cat36rsa models) -------------------------------
# Same Shrake-Rupley math as training side (curation/add_sasa_to_h5.py):
# heavy atoms, isolated chain, per-atom SASA summed per residue, normalized by
# Tien et al. 2013 max ASA, clipped to [0,1]. Class LUTs follow the onehot()
# layout (std classes + trailing unknown).
_ELEM_RADIUS_LUT = np.array([VDW_RADII.get(e, FALLBACK_RADIUS) for e in std_elements]
                            + [FALLBACK_RADIUS], dtype=np.float64)
_RES_MAXASA_LUT = np.array([MAX_ASA.get(r, FALLBACK_MAX_ASA) for r in std_resnames]
                           + [FALLBACK_MAX_ASA], dtype=np.float64)

def rsa_from_tensors(X, M, qe, qr):
    """Per-residue RSA (N_res,) from encoded structure tensors (pre-preprocess).

    X: (N_atom,3); M: (N_atom,N_res) bool; qe/qr: onehot (N_atom, n_class+1).
    Column order of M = pt.unique(resid) order, matching CA_BB residue_indices.
    """
    elem_idx = qe.argmax(1).cpu().numpy()
    resname_idx = qr.argmax(1).cpu().numpy()
    atom_sasa = shrake_rupley_per_atom(X.cpu().numpy().astype(np.float64),
                                       _ELEM_RADIUS_LUT[elem_idx])
    res_of_atom = M.cpu().numpy().argmax(1)
    n_res = M.shape[1]
    res_sasa = np.zeros(n_res, dtype=np.float64)
    np.add.at(res_sasa, res_of_atom, atom_sasa)
    _, first = np.unique(res_of_atom, return_index=True)
    rsa = res_sasa / _RES_MAXASA_LUT[resname_idx[first]]
    return np.clip(rsa, 0.0, 1.0).astype(np.float32)
def get_plm_model(device,plm_type):
    if plm_type=="ESM2":
        plm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        plm_pack=[plm_model.to(device),alphabet,batch_converter]
    elif plm_type=="protT5":
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained(PROT_T5_MODEL_DIR, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(PROT_T5_MODEL_DIR).to(device)
        plm_pack=[tokenizer,model]

    elif plm_type=="esmc":
        from esm3lib.models.esmc import ESMC
        client = ESMC.from_pretrained("esmc_300m").to(device)
        plm_pack=[client]
    elif plm_type=="esm3":
        os.environ["HF_HUB_OFFLINE"] = "1"  # 用本地 HF 缓存，不联网
        from esm3lib.models.esm3 import ESM3
        # 在 CPU 上构造+加载再搬到 GPU:from_pretrained 默认在 torch.device("cuda") 下
        # 随机初始化全部 1.4B 参数,sm_120 上每个 init kernel 都要 JIT,会卡 10+ 分钟。
        # 与默认 cuda 路径一致地转 bf16。
        client = ESM3.from_pretrained("esm3-open", device=pt.device("cpu"))
        if pt.device(device).type != "cpu":
            client = client.to(pt.bfloat16)
        client = client.to(device)
        plm_pack=[client]
    else:
        raise NotImplementedError(f"PLM type {plm_type!r} not supported "
                                   f"(supported: {sorted(_ESM2_REPR_LAYER)+['protT5','esmc','esm3']})")
    return plm_pack




def open_plm_embedding_h5(path):
    """打开预计算 PLM embedding 的 HDF5（调大 chunk cache，适合大文件反复按 key 读）。"""
    if path is None or path == "":
        return None
    return h5py.File(
        path,
        "r",
        libver="latest",
        rdcc_nbytes=256 * 1024 * 1024,
        rdcc_nslots=1_000_003,
        rdcc_w0=0.75,
    )


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


def get_embedding(
    seq,
    device,
    plm_type,
    plm_pack,
    ukey=None,
    h5_path=None,
    h5_file=None,
):
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
            with h5py.File(
                h5_path,
                "r",
                libver="latest",
                rdcc_nbytes=256 * 1024 * 1024,
                rdcc_nslots=1_000_003,
                rdcc_w0=0.75,
            ) as hf1:
                arr = hf1[ukey][:]
                plm_embedding = pt.from_numpy(arr).float()
        return plm_embedding
    if plm_type in _ESM2_REPR_LAYER:
        seq_rep = _esm2_family_embed(seq, plm_pack, device, _ESM2_REPR_LAYER[plm_type])
    elif plm_type=="protT5":
        data_accumulator = [("protein1", seq)]
        tokenizer,model=plm_pack
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence[1]))) for sequence in data_accumulator]
        ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
        input_ids = pt.tensor(ids['input_ids']).to(device)
        attention_mask =pt.tensor(ids['attention_mask']).to(device)
        batch_lens = attention_mask.sum(dim=1).cpu().tolist()
        with pt.no_grad():
            with pt.cuda.amp.autocast():
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        for i, tokens_len in enumerate(batch_lens):
            seq_rep = embedding_repr.last_hidden_state[i,  : tokens_len - 1].cpu()
    elif plm_type in ("esmc", "esm3"):
        # identical call shape for both -- esm3 tokenizes as <cls> + seq + <eos>,
        # esmc likewise; strip both so the rows line up with residues.
        seq_rep = _esm3lib_embed(seq, plm_pack)
    else:
        raise NotImplementedError(f"PLM type {plm_type!r} not supported "
                                   f"(supported: {sorted(_ESM2_REPR_LAYER)+['protT5','esmc','esm3']})")

    return seq_rep


def _esm2_family_embed(seq, plm_pack, device, repr_layer):
    model, alphabet, batch_converter = plm_pack
    data_accumulator = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data_accumulator)
    batch_tokens = batch_tokens.to(device)
    with pt.no_grad():
        with pt.cuda.amp.autocast():
            results = model(batch_tokens, repr_layers=[repr_layer])
    token_representations = results["representations"][repr_layer]
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    return token_representations[0, 1 : batch_lens[0] - 1].cpu()


def _esm3lib_embed(seq, plm_pack):
    from esm3lib.sdk.api import ESMProtein, LogitsConfig
    client = plm_pack[0]
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings.squeeze(0)[1:-1, :].cpu()

def prepocess_sparse(X,M,qe,qr,qn,preprocess_type,ftrs,max_nn,with_rsa=False):
    qn_idx = pt.stack(pt.where(qn), dim=1)

    types = qn_idx[:, 1]
    q={}
    # 生成一个布尔mask：True表示要保留的行
    if preprocess_type=="extract_backbone":
        sel_types=[0, 1, 2, 3, 4]
        mask =np.isin(types, sel_types)
        sel = pt.nonzero(pt.from_numpy(mask), as_tuple=False).squeeze(-1)  # shape: [N_selected]
        X,M,qn,qr,qe=X[sel],M[sel],qn[sel],qr[sel],qe[sel]
    # 获取符合条件的索引
    elif preprocess_type=="CA_only":
        sel_types=[0]

        mask =np.isin(types, sel_types)
        sel = pt.nonzero(pt.from_numpy(mask), as_tuple=False).squeeze(-1)  # shape: [N_selected]
        res_idx = pt.where(M)[1]
        selected = []
        seen_res = set()

        for i in sel.tolist():
            r = int(res_idx[i])
            if r not in seen_res:
                selected.append(i)
                seen_res.add(r)
        sel = pt.tensor(selected, device=X.device)
        X,M,qn,qr,qe=X[sel],M[sel],qn[sel],qr[sel],qe[sel]
        N = M.shape[0]
        M = pt.eye(N, device=M.device, dtype=M.dtype)
    elif preprocess_type=="CA_BB":
        N_res=M.shape[1]
        rsa_full = rsa_from_tensors(X, M, qe, qr) if with_rsa else None
        coords_list = []
        ca_atom_indices = []
        residue_indices = []

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

        angles=dihedrals(pt.from_numpy(np.stack(coords_list)), sparse=True)
        q['angles']=angles
        if rsa_full is not None:
            # 与训练侧一致 (model/data_handler.py preprocess)：rsa 为节点标量最后一列
            q['rsa'] = pt.from_numpy(
                rsa_full[np.asarray(residue_indices, dtype=np.int64)]).unsqueeze(1)
        mask=np.zeros(M.shape[0],dtype=bool)
        mask[ca_atom_indices]=1
        X,qn,qr,qe,M=X[mask],qn[mask],qr[mask],qe[mask],M[mask]
        N = X.shape[0]
        M = pt.eye(N, device=M.device, dtype=M.dtype)
        assert M.shape[0] == M.shape[1], f"M.shape = {M.shape}"


    else:
        raise NotImplementedError("preprocess type not defined")

    #assert M.shape[0] == M.shape[1], f"M.shape = {M.shape}"
    ids_topk = extract_topology(X, max_nn)[0]
    q['qn']=qn
    q['qe']=qe
    q['qr']=qr


    return X,ids_topk,M,q

def process_structure(path,chainid=None):
    if os.path.exists(path)==False:
            print(f'{path} not found')
            return []
    if os.path.basename(path).split('.')[1]=='cif':
    #process structure
        structure = read_cif(path)
        #print(os.path.basename(path))
    else:
        structure = read_pdb(path)
        #print(structure['chain_name'])

    structure = clean_structure(structure)
    # update molecules chains
    structure = tag_hetatm_chains(structure)
    # split structure
    subunits = split_by_chain(structure)
    # remove non atomic structures
    subunits = filter_non_atomic_subunits(subunits)
    # remove duplicated molecules and ions
    subunits = remove_duplicate_tagged_subunits(subunits)

    if chainid:
        chain_key = chainid.split("/")[-1]
        if chain_key not in subunits:
            # NOTE: used to be a bare `except:` that printed and fell through, silently
            # returning the pre-split whole-structure `structure` instead of the
            # requested chain -- callers had no way to tell they got the wrong chain.
            # Return [] instead, matching this function's own "not found" convention above.
            print(f"{chainid!r}: chain {chain_key!r} not found among subunits {list(subunits)}")
            return []
        structure = subunits[chain_key]
    else:
        structure = concatenate_chains(subunits)
    return structure


def process_seq_batch(structure,feature_flags,preprocess_type,model_frame,max_nn,with_rsa=False):
    ftrs=[fn for fn, ff in zip(['qe','qr','qn'],feature_flags) if ff]
    X, M = encode_structure(structure)
    q_l=[]
    q = list(encode_features(structure))#onehot
    q_dict={"qe":q[0],"qr":q[1],"qn":q[2]}
    if preprocess_type:
        X,ids_topk,M,q_dict = prepocess_sparse(X,M,q_dict["qe"],q_dict["qr"],q_dict["qn"],preprocess_type,ftrs,max_nn,with_rsa=with_rsa)
    else:
        ids_topk, _, _, _, _ = extract_topology(X, max_nn)
    q_l = []
    qr_idx = pt.stack(pt.where(q_dict["qr"]), dim=1)

    for fn in ftrs:
        q_l.append(q_dict[fn])
    q = pt.cat(q_l, dim=1)#onehot
    if 'angles'in q_dict:
        q=pt.cat([q,q_dict['angles']],dim=1)
    if 'rsa' in q_dict:
        q=pt.cat([q,q_dict['rsa']],dim=1)

    if model_frame=="PeSTo":
        X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])
        data_list=[X, ids_topk, q, M]
        dense_M = pt.stack(pt.where(M), dim=1)
    elif model_frame=="gcp":
        dense_M = pt.stack(pt.where(M), dim=1)
        data_list=[]
        pyg_data = Data(
            pos=X,                       # (Ni, 3)
            x=q,           # (Ni, 1)
            ids_topk=ids_topk,
            batch=0,
            M=dense_M
        )
        data_list.append(pyg_data)

    seq=load_seq(dense_M,qr_idx)
    return data_list,seq

def inference(model,device,data_list,model_frame,mask=None,plm_embedding=None,plm_fusion="concat"):


    if model_frame=="PeSTo":
        if mask is not None:
            M=data_list[-1]
            atom_mask_float = np.dot(M, mask)

            # 转换为布尔型掩码
            atom_mask = atom_mask_float > 0.5
            mask = pt.as_tensor(atom_mask).bool()
            # 重新生成一个新的列表覆盖原变量
            data_list = [d[mask] for d in data_list]
        X, ids_topk, q, M = data_list

        with pt.no_grad():
            #try:
            z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))
        #print("M",M.shape)
        M = pt.stack(pt.where(M), dim=1)

    elif model_frame=="gcp":

        batch = Batch.from_data_list(data_list)

        if mask is not None:
            mask=pt.as_tensor(mask).bool()
            batch=batch.subgraph(mask)
        batch=batch.to(device, non_blocking=True)
        if plm_embedding is not None:
            batch.x=batch.x.to(device)
            plm_embedding=plm_embedding.to(device)
            if plm_fusion=="series":
                batch.x=plm_embedding
            elif plm_fusion in ["self_attn","cross_attn","concat_late"]:
                    batch.plm_embedding=plm_embedding
            else:
                batch.x=pt.cat([batch.x,plm_embedding],dim=1)



        with pt.inference_mode():
            z = model(batch)["node_embedding"]


    p=pt.sigmoid(z)
    pc = pt.cat([1.0 - pt.max(p, axis=1)[0].unsqueeze(1), p], axis=1).cpu()

    return pc
