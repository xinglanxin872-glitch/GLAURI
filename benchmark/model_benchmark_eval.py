#!/usr/bin/env python3
"""Evaluate the supplied GLAURI best-AUC checkpoint on five benchmark tasks.

Tasks: MaSIF-53, DNA-129, RNA-117, Zn-59, coach_test. Ligand labels and raw
structures retain their source filenames, but membership is restricted to the
complete COACH420 list before caching, inference and scoring.

python benchmark/model_benchmark_eval.py --sets coach_test --stage all
"""
import os, sys, re, glob, pickle, tempfile, subprocess, types, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# This script imports the model/eval internals as a package rooted one level up.
REPO = os.environ.get("GLAURI_REPO_ROOT", os.path.dirname(HERE))
sys.path.insert(0, REPO)
_stub = types.ModuleType("Bio.Align.Applications"); _stub.ClustalOmegaCommandline = object
sys.modules["Bio.Align.Applications"] = _stub

from benchmark.coach_split import filter_coach_test_ids

import torch as pt, pandas as pd, h5py
from glob import glob as _glob
from tqdm import tqdm
import importlib.util
from src.utils_eval import process_structure, process_seq_batch, inference, get_embedding
from model.model import GCPNetModel
from sklearn import metrics

RESULTS = os.environ.get("BENCH_RESULTS_DIR", os.path.join(HERE, "results"))
LABELS = os.environ.get("BENCH_LABELS_DIR", os.path.join(HERE, "labels"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "data"))
DEVICE = pt.device(os.environ.get("EVAL_DEVICE", "cuda:0"))
bc_score_names = ['acc', 'ppv', 'npv', 'tpr', 'tnr', 'mcc', 'auc', 'std']

# Raw bound/predicted structures are not part of this repo (only already-scored results
# are); point these at wherever you reproduced them. See benchmark/EVAL_SCRIPTS.md.
STD_STRUCT_DIR = os.environ.get("STD_STRUCT_DIR", os.path.join(DATA_ROOT, "benchmark_structures"))
COACH_STRUCT_DIR = os.environ.get("COACH_STRUCT_DIR", os.path.join(DATA_ROOT, "benchmark_structures", "coach420"))

SETS = {
 "MaSIF-53": dict(itype="protein", col=1, cache=False,
                   label=os.path.join(LABELS, "protein", "masif-53-label.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "MaSIF_53_pdb")),
 "DNA-129":  dict(itype="DNA/RNA", col=2, cache=False,
                   label=os.path.join(LABELS, "NN", "DNA-129_Test.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "DNA-129_pdb")),
 "RNA-117":  dict(itype="DNA/RNA", col=2, cache=False,
                   label=os.path.join(LABELS, "NN", "RNA-117_Test.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "RNA-117_pdb")),
 "Zn-59":    dict(itype="ion", col=3, cache=False,
                   label=os.path.join(LABELS, "ion", "Zn-59_Test.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "metal_3d_pdb")),
 "coach_test": dict(itype="ligand", col=4, cache=True,
                   label=os.path.join(LABELS, "ligand", "coach420_label.csv")),
}

PRED = {}
for d in _glob(os.path.join(STD_STRUCT_DIR, "infer_output", "gpu_*", "*") + os.sep):
    key = os.path.basename(d.rstrip("/")); cif = os.path.join(d, f"{key}_model.cif")
    if os.path.isfile(cif): PRED[key] = cif

# ----------------------------- shared: label readers -----------------------------
def read_label(path):
    """Fixed-reference-sequence label file: FASTA triplets (>id / seq / 0-1 string)."""
    out, lines = {}, [l.rstrip("\n") for l in open(path)]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            out[lines[i][1:].strip()] = (lines[i+1].strip(),
                                         np.array([int(c) for c in lines[i+2].strip()], dtype=np.float32))
            i += 3
        else: i += 1
    return out

def coach420_labels_for_refseq(ref_seq, label_field):
    """coach420_label.csv 'Label' field is tokens like 'A_12 A_57' (chain_resnum, 1-based
    on predict_seq); decode into a 0/1 vector over ref_seq (== predict_seq here)."""
    n = len(ref_seq); out = np.zeros(n, dtype=np.int8)
    if label_field is None: return out
    if isinstance(label_field, float) and label_field != label_field: return out
    s = str(label_field).strip()
    if not s or s.lower() == "nan": return out
    for chunk in s.split(";"):
        for tok in chunk.split():
            t = tok.strip()
            if "_" not in t: continue
            _, num = t.rsplit("_", 1)
            if not num.isdigit(): continue
            k = int(num)
            if 1 <= k <= n: out[k - 1] = 1
    return out

def label_stem(idx):
    p, c = idx.rsplit("_", 1); return f"{p.lower()}{c.upper()}"

# ----------------------------- shared: alignment (pairwise, subprocess clustalo) -----------------------------
def _read_fa(p):
    name, seq, out = None, [], {}
    for ln in open(p):
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if name is not None: out[name] = "".join(seq)
            name, seq = ln[1:].split()[0], []
        else: seq.append(ln.strip())
    if name is not None: out[name] = "".join(seq)
    return out

def pairwise_map(a, b):
    """Free identity shortcut when a==b (coach420: ref_seq==predict_seq by construction)."""
    if a == b: return {i: i for i in range(len(a))}
    fd, fin = tempfile.mkstemp(suffix=".fa"); os.close(fd)
    fd, fout = tempfile.mkstemp(suffix=".fa"); os.close(fd)
    try:
        with open(fin, "w") as f: f.write(f">a\n{a}\n>b\n{b}\n")
        subprocess.run(["clustalo", "-i", fin, "-o", fout, "--force", "--auto"], check=True, capture_output=True)
        aln = _read_fa(fout)
    finally:
        for p in (fin, fout):
            try: os.remove(p)
            except OSError: pass
    A, B = aln["a"], aln["b"]; ai = bi = 0; m = {}
    for ca, cb in zip(A, B):
        if ca != "-" and cb != "-": m[ai] = bi
        if ca != "-": ai += 1
        if cb != "-": bi += 1
    return m

def align_anchor(ref, b, p):
    mb = pairwise_map(ref, b); mp = pairwise_map(ref, p)
    common = sorted(set(mb) & set(mp))
    return (np.array(common, int), np.array([mb[i] for i in common], int), np.array([mp[i] for i in common], int))

# ----------------------------- shared: scoring -----------------------------
def bc_scoring(y, p):
    q = (p >= .5).to(p.dtype)
    TP = pt.sum(q*y, 0); TN = pt.sum((1-q)*(1-y), 0); FP = pt.sum(q*(1-y), 0); FN = pt.sum((1-q)*y, 0)
    P = pt.sum(y, 0); N = pt.sum(1-y, 0)
    def sf(v): v = v.clone(); v[pt.isinf(v)] = float('nan'); return v
    acc = (TP+TN)/(TP+TN+FP+FN); ppv = TP/(TP+FP); ppv[~(P > 0)] = float('nan')
    npv = TN/(TN+FN); npv[~(N > 0)] = float('nan'); tpr = sf(TP/(TP+FN)); tnr = sf(TN/(TN+FP))
    mcc = sf(((TP*TN)-(FP*FN))/pt.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    auc = pt.full((y.shape[1],), float('nan')); m = (P > 0) & (N > 0)
    if bool(pt.any(m)):
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(y[:, m].cpu().numpy(), p[:, m].cpu().numpy(), average=None)
        auc[m] = pt.tensor(np.atleast_1d(a), dtype=pt.float32)
    return pt.stack([acc, ppv, npv, tpr, tnr, mcc, auc, pt.std(p, 0)])

def cal_mae(p, q): return pt.mean(pt.abs(q-p)).item()
def cal_w_mae(p, q):
    w = pt.abs(q+p-1); return (pt.sum(w*pt.abs(p-q))/pt.sum(w)).item() if w.sum() > 0 else 0.0
def cal_rmse(p, q): return pt.sqrt(pt.mean((p-q)**2)).item()
def cal_JS_div(p, q, eps=1e-12):
    p = p.double().clamp(eps, 1-eps); q = q.double().clamp(eps, 1-eps); m = .5*(p+q)
    return (.5*((p*pt.log(p/m)+(1-p)*pt.log((1-p)/(1-m)))+(q*pt.log(q/m)+(1-q)*pt.log((1-q)/(1-m))))).mean().item()
def cal_ICC2_1(p, q):
    from src.metrics import icc_a1
    return icc_a1(p, q)

# ----------------------------- shared: model loading -----------------------------
def load_model(model_dir=None, checkpoint=None, device=None):
    """Compatibility wrapper around the shared checkpoint loader."""
    from src.checkpoint import load_checkpoint
    return load_checkpoint(model_dir, checkpoint, device or DEVICE).legacy_tuple()

# ----------------------------- MaSIF-53/DNA-129/RNA-117/Zn-59: single-pass -----------------------------
def run_std_set(sname, s, model, cd, frame, plm_type, fusion, need_rsa, plm_pack, tag, outdir, limit=0):
    from src.emb_cache import get_cached_embedding
    ff = cd["features_flags"]; pre = cd.get("preprocess") or None; mnn = cd.get("max_num_nn", 64)

    def infer_full(struct):
        dl, seq = process_seq_batch(struct, ff, pre, frame, mnn, with_rsa=need_rsa, protocol=cd.get('feature_protocol', 'legacy_nodes_angles_sidechain_v1_20261102'))
        emb = get_cached_embedding(seq, DEVICE, plm_type, plm_pack) if plm_type else None
        pc = inference(model, DEVICE, dl, frame, mask=None, plm_embedding=emb, plm_fusion=fusion)
        return pc, seq

    label = read_label(s["label"]); col = s["col"]; ids = sorted(label)
    if limit: ids = ids[:limit]
    Y, PB, PP = [], [], []; aligned = {}; n_eval = n_skip = 0
    for cid in ids:
        bpath = os.path.join(s["bound"], f"{cid}.pdb")
        if not os.path.isfile(bpath): bpath = os.path.join(s["bound"], f"{cid.upper()}.pdb")  # MaSIF files are uppercase
        cif = PRED.get(cid.lower())
        if not os.path.isfile(bpath) or cif is None: n_skip += 1; continue
        _, ch = cid.rsplit("_", 1)
        try:
            bstruct = process_structure(bpath, chainid=f"{ch}:0"); pstruct = process_structure(cif)
            if not (isinstance(bstruct, dict) and isinstance(pstruct, dict)): n_skip += 1; continue
            pc_b, bseq = infer_full(bstruct); pc_p, pseq = infer_full(pstruct)
            rseq, ylab = label[cid]; ri, bi, pi = align_anchor(rseq, bseq, pseq)
        except Exception as e:
            print(f"  !! {sname}/{cid}: {repr(e)[:90]}"); n_skip += 1; continue
        if len(ri) == 0: n_skip += 1; continue
        y = pt.tensor(ylab[ri]).float(); pb = pc_b[bi, col].float(); pp = pc_p[pi, col].float()
        Y.append(y); PB.append(pb); PP.append(pp)
        aligned[cid] = dict(y=y.numpy(), p_bound=pb.numpy(), p_pred=pp.numpy()); n_eval += 1
    y = pt.cat(Y)[:, None]; pb = pt.cat(PB)[:, None]; pp = pt.cat(PP)[:, None]
    sb = bc_scoring(y, pb).squeeze(1).tolist(); sp = bc_scoring(y, pp).squeeze(1).tolist()
    row = dict(set=sname, method=tag, interface_type=s["itype"], n_chain=n_eval, skip=n_skip,
               n_res=int(y.numel()), pos_ratio=float(y.mean()),
               F1_b=metrics.f1_score(y.int().numpy().ravel(), pb.round().int().numpy().ravel()),
               F1_p=metrics.f1_score(y.int().numpy().ravel(), pp.round().int().numpy().ravel()),
               mae=cal_mae(pb, pp), rmse=cal_rmse(pb, pp), JS_div=cal_JS_div(pb, pp))
    for nm, v in zip(bc_score_names, sb): row[nm+"_b"] = v
    for nm, v in zip(bc_score_names, sp): row[nm+"_p"] = v
    pickle.dump({sname: {tag: aligned}}, open(os.path.join(outdir, f"{sname}_{tag}_aligned.pkl"), "wb"))
    print(f"[{sname}] eval={n_eval} skip={n_skip} | {s['itype']} bound auc={sb[6]:.3f} f1={row['F1_b']:.3f} | af3 auc={sp[6]:.3f} f1={row['F1_p']:.3f}")
    return row

# ----------------------------- coach420: 3-stage disk-cached pipeline -----------------------------
CLASS_NAMES = ["protein", "DNA/RNA", "ion", "ligand"]
LIG = 4  # ligand column in the 5-wide pc/yc ([bg,protein,DNA/RNA,ion,ligand])

def _coach420_paths():
    coach_pdb_dir = os.path.join(COACH_STRUCT_DIR, "coach420_pdb")          # primary bound (full 420)
    receptor_dir = os.path.join(COACH_STRUCT_DIR, "receptor_pdb_coach420")  # fallback for 48 chain-mismatches
    af3_dir = os.path.join(COACH_STRUCT_DIR, "coach24_af3")
    return coach_pdb_dir, receptor_dir, af3_dir

def coach_test_ids(label_csv_path):
    lab = pd.read_csv(label_csv_path)
    return filter_coach_test_ids({f"{s[:4].upper()}_{s[4:].upper()}" for s in lab["PDB ID"].astype(str)}, require_all=True)

def bound_pdb_path(idx):
    coach_pdb_dir, receptor_dir, _ = _coach420_paths()
    p, c = idx.rsplit("_", 1)
    primary = os.path.join(coach_pdb_dir, f"{p.upper()}_{c.upper()}.pdb")
    if os.path.isfile(primary):
        return primary
    return os.path.join(receptor_dir, f"{p.lower()}{c.upper()}.pdb")

def af3_cif_path(idx):
    _, _, af3_dir = _coach420_paths()
    p, c = idx.rsplit("_", 1); key = f"{p.lower()}_{c.lower()}"
    return os.path.join(af3_dir, key, f"{key}_model.cif")

def stage_struct(ids, struct_pkl):
    ids = filter_coach_test_ids(ids)
    cache = {}
    if os.path.exists(struct_pkl):
        cache = pickle.load(open(struct_pkl, "rb"))
    cache = {key: cache[key] for key in filter_coach_test_ids(cache)}
    n_new = n_fail = 0
    for chain in tqdm(ids, desc="struct"):
        cur = cache.get(chain, {})
        ok_b = isinstance(cur.get("bound"), dict) and len(cur.get("bound", {})) > 0
        ok_p = isinstance(cur.get("predict"), dict) and len(cur.get("predict", {})) > 0
        if ok_b and ok_p:
            continue
        _, ch = chain.rsplit("_", 1)
        try:
            b = process_structure(bound_pdb_path(chain), chainid=ch + ":0")
            p = process_structure(af3_cif_path(chain))
        except Exception as e:
            print(f"  !! struct fail {chain}: {repr(e)[:100]}"); n_fail += 1; continue
        if not (isinstance(b, dict) and isinstance(p, dict) and b and p):
            print(f"  !! struct empty {chain}: bound={type(b).__name__} predict={type(p).__name__}")
            n_fail += 1; continue
        cache[chain] = {"bound": b, "predict": p}; n_new += 1
    pickle.dump(cache, open(struct_pkl, "wb"))
    valid = sum(1 for v in cache.values() if v.get("bound") and v.get("predict"))
    print(f"[struct] new={n_new} fail={n_fail} valid_total={valid}/{len(ids)} -> {struct_pkl}")
    return cache

def stage_emb(cache, cd, frame, plm_type, emb_h5):
    from src.emb_cache import get_cached_embedding, LazyPLMPack
    ff = cd["features_flags"]; pre = cd.get("preprocess") or None; mnn = cd.get("max_num_nn", 64)
    plm_pack = LazyPLMPack(DEVICE, plm_type)
    n_new = n_skip = 0
    with h5py.File(emb_h5, "a") as hf:
        import json
        protocol=plm_pack.provider.protocol
        if len(hf) and hf.attrs.get("glauri_embedding_protocol")!=json.dumps(protocol,sort_keys=True):raise ValueError("Embedding H5 protocol differs; use a fresh output directory")
        hf.attrs["glauri_embedding_protocol"]=json.dumps(protocol,sort_keys=True)
        hf.attrs["sdpa_backend"]="math"
        for chain in tqdm(filter_coach_test_ids(cache), desc="emb"):
            if chain in hf: n_skip += 1; continue
            try:
                _, predict_seq = process_seq_batch(cache[chain]["predict"], ff, pre, frame, mnn, protocol=cd.get("feature_protocol", "legacy_nodes_angles_sidechain_v1_20261102"))
                emb = get_cached_embedding(predict_seq, DEVICE, plm_type, plm_pack)
                arr = emb.detach().float().cpu().numpy()
            except Exception as e:
                print(f"  !! emb fail {chain}: {repr(e)[:100]}"); continue
            hf.create_dataset(chain, data=arr, compression="gzip")
            hf[chain].attrs["seq"] = predict_seq
            n_new += 1
    del plm_pack
    pt.cuda.empty_cache()
    print(f"[emb] new={n_new} skip={n_skip} -> {emb_h5}")

def stage_infer(cache, cd, cm, frame, plm_type, fusion, need_rsa, model, label_csv_path, emb_h5):
    ff = cd["features_flags"]; pre = cd.get("preprocess") or None; mnn = cd.get("max_num_nn", 64)
    label_csv = pd.read_csv(label_csv_path)
    pp_i, pb_i, y_i = [], [], []; aligned = {}; n_eval = n_skip = 0
    with h5py.File(emb_h5, "r") as hf:
        import json
        from src.embeddings import EmbeddingProvider
        protocol=EmbeddingProvider(DEVICE,backend="math").protocol
        if hf.attrs.get("glauri_embedding_protocol")!=json.dumps(protocol,sort_keys=True):raise ValueError("Unverified embedding H5")
        for chain in tqdm(filter_coach_test_ids(cache), desc="infer"):
            try:
                dl_b, pdb_seq = process_seq_batch(cache[chain]["bound"], ff, pre, frame, mnn, with_rsa=need_rsa, protocol=cd.get('feature_protocol', 'legacy_nodes_angles_sidechain_v1_20261102'))
                dl_p, predict_seq = process_seq_batch(cache[chain]["predict"], ff, pre, frame, mnn, with_rsa=need_rsa, protocol=cd.get('feature_protocol', 'legacy_nodes_angles_sidechain_v1_20261102'))
                stem = label_stem(chain)
                mrow = label_csv[label_csv["PDB ID"].astype(str).str.lower() == stem.lower()]
                lab_str = "" if mrow.empty else mrow.iloc[0]["Label"]
                ref_seq = predict_seq
                ref_np = np.zeros((len(ref_seq), 4), dtype=np.float32)
                ref_np[:, 3] = coach420_labels_for_refseq(ref_seq, lab_str).astype(np.float32)
                ref_label = pt.from_numpy(ref_np)

                ri, bi, pi = align_anchor(ref_seq, pdb_seq, predict_seq)
                m3 = np.zeros(len(ref_seq), dtype=bool); m3[ri] = True
                m2 = np.zeros(len(pdb_seq), dtype=bool); m2[bi] = True
                m1 = np.zeros(len(predict_seq), dtype=bool); m1[pi] = True

                emb = get_embedding(ref_seq, DEVICE, plm_type, None, ukey=chain, h5_path=emb_h5)[m3] if plm_type else None
                pc_p = inference(model, DEVICE, dl_p, frame, mask=m1, plm_embedding=emb, plm_fusion=fusion)
                pc_b = inference(model, DEVICE, dl_b, frame, mask=m2, plm_embedding=emb, plm_fusion=fusion)
                y_m = ref_label[m3]
                yc = pt.cat([1.0 - pt.any(y_m > 0.5, axis=1).float().unsqueeze(1), y_m], dim=1).cpu()
            except Exception as e:
                print(f"  !! infer fail {chain}: {repr(e)[:120]}"); n_skip += 1; continue
            if not (pc_p.shape[0] == pc_b.shape[0] == yc.shape[0]) or yc.shape[0] == 0:
                print(f"  !! len mismatch {chain}: {pc_p.shape[0]}/{pc_b.shape[0]}/{yc.shape[0]}"); n_skip += 1; continue
            pp_i.append(pc_p); pb_i.append(pc_b); y_i.append(yc)
            aligned[chain] = dict(
                y=yc[:, LIG].numpy().astype(np.float32),
                p_bound=pc_b[:, LIG].numpy().astype(np.float32),
                p_pred=pc_p[:, LIG].numpy().astype(np.float32))
            n_eval += 1
    print(f"[infer] eval={n_eval} skip={n_skip}")
    return pp_i, pb_i, y_i, aligned

def _to(x): return x.item() if hasattr(x, "item") else float(x)

def write_coach_test_outputs(pp_i, pb_i, y_i, aligned, tag, outdir):
    p_b_all = pt.cat(pb_i, 0); p_p_all = pt.cat(pp_i, 0); y_all = pt.cat(y_i, 0)
    sb, sp = [], []
    for ci in range(4):
        col = ci + 1
        pb = p_b_all[:, col]; pp = p_p_all[:, col]; y = y_all[:, col]
        s_b = bc_scoring(y.unsqueeze(1), pb.unsqueeze(1)).squeeze().numpy()
        s_p = bc_scoring(y.unsqueeze(1), pp.unsqueeze(1)).squeeze().numpy()
        f1_b = metrics.f1_score(y.numpy().astype(int), (pb.numpy()>=.5))
        f1_p = metrics.f1_score(y.numpy().astype(int), (pp.numpy()>=.5))
        r = float(pt.mean(y))
        sb.append(np.concatenate([s_b, [f1_b, r]])); sp.append(np.concatenate([s_p, [f1_p, r]]))
    idx = bc_score_names + ["F1", "r"]
    pd.DataFrame(np.round(np.stack(sb).T, 4), index=idx, columns=CLASS_NAMES).to_csv(os.path.join(outdir, f"{tag}_bound.csv"))
    pd.DataFrame(np.round(np.stack(sp).T, 4), index=idx, columns=CLASS_NAMES).to_csv(os.path.join(outdir, f"{tag}_af3.csv"))
    rows = []
    for chain, a in aligned.items():
        y = pt.tensor(a["y"]).float(); pb = pt.tensor(a["p_bound"]).float()[:, None]; pp = pt.tensor(a["p_pred"]).float()[:, None]
        s_b = bc_scoring(y[:, None], pb).squeeze().numpy(); s_p = bc_scoring(y[:, None], pp).squeeze().numpy()
        row = dict(chainid=chain, interface_type="ligand", n_aligned=len(y), n_pos=int(y.sum()),
                   mae=_to(cal_mae(pb, pp)), w_mae=_to(cal_w_mae(pb, pp)), rmse=_to(cal_rmse(pb, pp)),
                   JS_div=_to(cal_JS_div(pb, pp)), ICC=_to(cal_ICC2_1(pb, pp)))
        for nm, v in zip(bc_score_names, s_b): row[nm + "_b"] = v
        for nm, v in zip(bc_score_names, s_p): row[nm + "_p"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{tag}_pVb_score.csv"), index=False)
    pickle.dump({"coach_test": {tag: aligned}}, open(os.path.join(outdir, f"{tag}_aligned.pkl"), "wb"))
    print("wrote:", os.path.join(outdir, f"{tag}_bound.csv"), f"{tag}_af3.csv", f"{tag}_pVb_score.csv", f"{tag}_aligned.pkl", sep="\n  ")
    print("\n[ligand] pooled: bound auc=%.3f f1=%.3f | af3 auc=%.3f f1=%.3f" % (
        np.stack(sb).T[6, 3], np.stack(sb).T[8, 3], np.stack(sp).T[6, 3], np.stack(sp).T[8, 3]))

def run_coach_test(s, model, cd, cm, frame, plm_type, fusion, need_rsa, tag, outdir, stage="all"):
    ids = coach_test_ids(s["label"])
    cache_dir = os.path.join(outdir, "_cache")  # struct+emb cache is model-independent, kept across runs
    os.makedirs(cache_dir, exist_ok=True)
    struct_pkl = os.path.join(cache_dir, "coach_test_structures.pkl")
    emb_h5 = os.path.join(cache_dir, f"coach_test_emb_{plm_type}.h5")
    cache = stage_struct(ids, struct_pkl)
    if stage in ("all", "emb"):
        stage_emb(cache, cd, frame, plm_type, emb_h5)
    if stage in ("all", "infer"):
        pp_i, pb_i, y_i, aligned = stage_infer(cache, cd, cm, frame, plm_type, fusion, need_rsa, model, s["label"], emb_h5)
        write_coach_test_outputs(pp_i, pb_i, y_i, aligned, tag, outdir)

# ----------------------------- main -----------------------------
def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(REPO, "model", "GLAURI"))
    ap.add_argument("--checkpoint", default=None, help="Checkpoint filename or absolute path; defaults to best-AUC weights")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device",default=str(DEVICE))
    ap.add_argument("--output",default=None,help="Explicit output directory; checkpoint identity is checked")
    ap.add_argument("--sets", default=None, help="comma-separated subset of SETS (default: all 5)")
    ap.add_argument("--limit", type=int, default=0, help="std sets only: cap #chains (debug)")
    ap.add_argument("--stage", choices=["all", "struct", "emb", "infer"], default="all",
                     help="COACH420 only: resume from a given stage")
    a = ap.parse_args()
    import json
    from src.checkpoint import load_checkpoint
    DEVICE=pt.device(a.device)
    loaded=load_checkpoint(a.model_dir,a.checkpoint,DEVICE)
    model,cd,cm,frame,plm_type,fusion,need_rsa=loaded.legacy_tuple()
    tag=a.tag or os.path.basename(os.path.abspath(a.model_dir))
    outdir=a.output or os.path.join(RESULTS,"_runs",f"{tag}_{loaded.metadata['checkpoint_sha256'][:12]}_benchmark_out")
    from src.embeddings import EmbeddingProvider
    protocol=dict(checkpoint_sha256=loaded.metadata['checkpoint_sha256'],config_sha256=loaded.metadata['config_sha256'],
        feature_protocol=loaded.metadata['feature_protocol'],input_policy=loaded.metadata['input_policy'],
        embeddings=EmbeddingProvider(DEVICE,backend='math').protocol,metric_protocol='icc_a1_residual_df_v1_threshold_ge05')
    manifest=os.path.join(outdir,'inference_protocol.json')
    if os.path.isfile(manifest):
        if json.load(open(manifest))!=protocol:raise ValueError('Output checkpoint/protocol differs')
    elif os.path.isdir(outdir) and os.listdir(outdir):raise ValueError('Existing output has no verified protocol; choose a new directory')
    os.makedirs(outdir,exist_ok=True)
    with open(manifest,'w') as stream:json.dump(protocol,stream,indent=2)
    from src.emb_cache import LazyPLMPack
    plm_pack = LazyPLMPack(DEVICE, plm_type) if plm_type else None
    print(f"model={tag} frame={frame} plm={plm_type} fusion={fusion}")

    sel = a.sets.split(",") if a.sets else list(SETS)
    summary = []
    for sname in sel:
        s = SETS[sname]
        if s["cache"]:
            run_coach_test(s, model, cd, cm, frame, plm_type, fusion, need_rsa, tag, outdir, stage=a.stage)
        else:
            row = run_std_set(sname, s, model, cd, frame, plm_type, fusion, need_rsa, plm_pack, tag, outdir, limit=a.limit)
            summary.append(row)
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(outdir, f"{tag}_benchmark_summary.csv"), index=False)
        print("wrote", os.path.join(outdir, f"{tag}_benchmark_summary.csv"))

if __name__ == "__main__":
    main()
