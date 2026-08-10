#!/usr/bin/env python3
"""Run OUR interface model (PeSTo or gcp frame) on all 5 benchmark tasks (protein,
DNA/RNA x2, ion, ligand), scored with the same pairwise alignment as bench_eval.py.

pc = [1-max, protein, DNA/RNA, ion, ligand]; per-set channel:
  MaSIF-53 -> col 1 (protein), DNA-129/RNA-117 -> col 2 (DNA/RNA), Zn-59 -> col 3 (ion),
  coach420 -> col 4 (ligand, and the only set scored on all 4 foreground columns).

Usage (boltz):
  python model_benchmark_eval.py --model-dir <dir> [--tag T] [--sets MaSIF-53,DNA-129,...]
  python model_benchmark_eval.py --model-dir <dir> --sets coach420 --stage struct|emb|infer|all

2026-08: merged coach420_ourmodel_eval.py into this script as a 5th SET entry
(cache=True), so all 5 tasks share one driver -- results/ is organized uniformly per
interface type now, the eval script covering them should be too. coach420 keeps its own
3-stage struct/emb/infer disk cache (SETS["coach420"]["cache"]=True): 420 chains with
expensive ESM embeddings are worth persisting across repeated runs against different
checkpoints; the other 4 sets are cheap enough to run in a single pass, unchanged.
coach420 also keeps its own output writer (4 foreground columns vs. the 1 column/set the
other 4 sets score) -- forcing both into one row format would obscure more than it'd share.
"""
import os, sys, re, glob, pickle, tempfile, subprocess, types, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# This script imports the model/eval internals as a package rooted one level up.
REPO = os.environ.get("GLAURI_REPO_ROOT", os.path.dirname(HERE))
sys.path.insert(0, REPO); os.chdir(REPO)
_stub = types.ModuleType("Bio.Align.Applications"); _stub.ClustalOmegaCommandline = object
sys.modules["Bio.Align.Applications"] = _stub

import torch as pt, pandas as pd, h5py
from glob import glob as _glob
from tqdm import tqdm
import importlib.util
from src.utils_eval import process_structure, process_seq_batch, inference, get_embedding
from model.model import GCPNetModel
from sklearn import metrics

RESULTS = os.environ.get("BENCH_RESULTS_DIR", os.path.join(HERE, "results"))
DEVICE = pt.device(os.environ.get("EVAL_DEVICE", "cuda:0"))
bc_score_names = ['acc', 'ppv', 'npv', 'tpr', 'tnr', 'mcc', 'auc', 'std']

# Raw bound/predicted structures are not part of this repo (only already-scored results
# are); point these at wherever you reproduced them. See docs/DATA.md.
STD_STRUCT_DIR = os.environ.get("STD_STRUCT_DIR", os.path.join(HERE, "results", "_raw_structures"))
COACH420_STRUCT_DIR = os.environ.get("COACH420_STRUCT_DIR", os.path.join(HERE, "results", "_raw_structures", "coach420"))

SETS = {
 "MaSIF-53": dict(itype="protein", col=1, cache=False,
                   label=os.path.join(RESULTS, "protein", "masif-53-label.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "MaSIF_53_pdb")),
 "DNA-129":  dict(itype="DNA/RNA", col=2, cache=False,
                   label=os.path.join(RESULTS, "NN", "DNA-129_Test.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "DNA-129_pdb")),
 "RNA-117":  dict(itype="DNA/RNA", col=2, cache=False,
                   label=os.path.join(RESULTS, "NN", "RNA-117_Test.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "RNA-117_pdb")),
 "Zn-59":    dict(itype="ion", col=3, cache=False,
                   label=os.path.join(RESULTS, "ion", "Zn-59_Test.txt"),
                   bound=os.path.join(STD_STRUCT_DIR, "metal_3d_pdb")),
 "coach420": dict(itype="ligand", col=4, cache=True,
                   label=os.path.join(RESULTS, "ligand", "coach420_label.csv")),
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
    q = pt.round(p)
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
    X = pt.stack([p.double().squeeze(), q.double().squeeze()], 1)
    if X.shape[0] < 2: return float('nan')
    n, k = X.shape; mr = X.mean(0); ms = X.mean(1); ma = X.mean()
    MSR = k*pt.var(ms, unbiased=True); MSC = n*pt.var(mr, unbiased=True)
    MSE = pt.var(X-ms[:, None]-mr[None, :]+ma, unbiased=True)
    return ((MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n)).item()

# ----------------------------- shared: model loading -----------------------------
def load_model(model_dir):
    """Load config (preferring the training-time effective snapshot) + checkpoint.
    Returns (model, cfg_data, cfg_model, frame, plm_type, plm_fusion, need_rsa)."""
    cfg_candidates = [p for p in _glob(os.path.join(model_dir, "*_config*.py"))
                       if not p.endswith("_base_config.py")]
    if not cfg_candidates:
        cfg_candidates = [os.path.join(REPO, "model", "config_baseline.py")]
    cfg_path = max(cfg_candidates)

    # cfg_path (the training-time override snapshot, or the config_baseline.py fallback)
    # does `from config import config_data, ...` at its top -- it needs an actual module
    # named "config" resolvable via sys.modules; nothing on sys.path provides a top-level
    # `config` module in this repo's flat layout (main.py only gets away with the bare
    # `import config` inside these files because it's launched with cwd=model/, putting
    # model/ implicitly on sys.path[0] -- this script runs from REPO root instead). Load
    # the matching base-config snapshot (same dir, *_base_config.py) under the name
    # "config" first so the override's import resolves via the sys.modules cache; fall
    # back to the live model/config.py for runs that predate snapshotting.
    base_candidates = _glob(os.path.join(model_dir, "*_base_config.py"))
    base_path = max(base_candidates) if base_candidates else os.path.join(REPO, "model", "config.py")
    # Old snapshots predate the src.data_encoding_redefine -> src.data_encoding rename
    # (pure rename, verified symbol-identical at the time) and still say the old name;
    # don't touch the snapshot file itself (it's the historical record of what actually
    # trained this checkpoint) -- alias it instead so `from src.data_encoding_redefine
    # import ...` resolves to the current module.
    if "src.data_encoding_redefine" not in sys.modules:
        import src.data_encoding as _de
        sys.modules["src.data_encoding_redefine"] = _de
    base_spec = importlib.util.spec_from_file_location("config", base_path)
    base_mod = importlib.util.module_from_spec(base_spec)
    sys.modules["config"] = base_mod
    base_spec.loader.exec_module(base_mod)

    spec = importlib.util.spec_from_file_location("custom_config", cfg_path)
    cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    cm, cd = cfg.config_model, cfg.config_data
    frame = cm.get("name", "gcp"); plm_type = cd.get("plm", None); fusion = cd.get("plm_fusion", "concat")
    need_rsa = cd.get("rsa_filepath") is not None  # cat36rsa: RSA is computed on the fly at inference time
    if frame != "gcp":
        raise ValueError(
            f"model/ 已重构为仅 gcp 通路(PeSTo 类已删除); 该 checkpoint config 声明 frame={frame!r}, 无法加载"
        )

    model = GCPNetModel(cm)
    # WSL2 workaround: synchronous H2D copies inside pt.load(map_location=cuda) occasionally
    # spin forever on this box (driver 591.86 / torch dev build). Warm the context first,
    # then load on CPU; Module.to(DEVICE) below has been reliable. "cuda_warmup_ok" doubles
    # as the pipeline watchdog marker.
    _w = pt.zeros(4, device=DEVICE); pt.cuda.synchronize(); del _w
    print("cuda_warmup_ok", flush=True)
    sd = pt.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    # concat_late esm_mlp: older ckpts have no Dropout module (2nd Linear at index .2),
    # current model.py inserts Dropout(0.0) at .2 (a no-op), shifting the Linear to .3.
    want = set(model.state_dict().keys())
    if "esm_mlp.3.weight" in want and "esm_mlp.2.weight" in sd and "esm_mlp.3.weight" not in sd:
        for k in ("weight", "bias"):
            sd[f"esm_mlp.3.{k}"] = sd.pop(f"esm_mlp.2.{k}")
    model.load_state_dict(sd); model = model.eval().to(DEVICE)
    print(f"config={cfg_path}\nframe={frame} plm={plm_type} fusion={fusion} need_rsa={need_rsa}")
    return model, cd, cm, frame, plm_type, fusion, need_rsa

# ----------------------------- MaSIF-53/DNA-129/RNA-117/Zn-59: single-pass -----------------------------
def run_std_set(sname, s, model, cd, frame, plm_type, fusion, need_rsa, plm_pack, tag, outdir, limit=0):
    from src.emb_cache import get_cached_embedding
    ff = cd["features_flags"]; pre = cd.get("preprocess") or None; mnn = cd.get("max_num_nn", 64)

    def infer_full(struct):
        dl, seq = process_seq_batch(struct, ff, pre, frame, mnn, with_rsa=need_rsa)
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
    coach_pdb_dir = os.path.join(COACH420_STRUCT_DIR, "coach420_pdb")          # primary bound (full 420)
    receptor_dir = os.path.join(COACH420_STRUCT_DIR, "receptor_pdb_coach420")  # fallback for 48 chain-mismatches
    af3_dir = os.path.join(COACH420_STRUCT_DIR, "coach24_af3")
    return coach_pdb_dir, receptor_dir, af3_dir

def coach420_ids(label_csv_path):
    lab = pd.read_csv(label_csv_path)
    return sorted({f"{s[:4].upper()}_{s[4:].upper()}" for s in lab["PDB ID"].astype(str)})

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
    cache = {}
    if os.path.exists(struct_pkl):
        cache = pickle.load(open(struct_pkl, "rb"))
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
        for chain in tqdm(cache, desc="emb"):
            if chain in hf: n_skip += 1; continue
            try:
                _, predict_seq = process_seq_batch(cache[chain]["predict"], ff, pre, frame, mnn)
                emb = get_cached_embedding(predict_seq, DEVICE, plm_type, plm_pack)
                arr = emb.detach().to(pt.float16).cpu().numpy()
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
        for chain in tqdm(cache, desc="infer"):
            try:
                dl_b, pdb_seq = process_seq_batch(cache[chain]["bound"], ff, pre, frame, mnn, with_rsa=need_rsa)
                dl_p, predict_seq = process_seq_batch(cache[chain]["predict"], ff, pre, frame, mnn, with_rsa=need_rsa)
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

def write_coach420_outputs(pp_i, pb_i, y_i, aligned, tag, outdir):
    p_b_all = pt.cat(pb_i, 0); p_p_all = pt.cat(pp_i, 0); y_all = pt.cat(y_i, 0)
    sb, sp = [], []
    for ci in range(4):
        col = ci + 1
        pb = p_b_all[:, col]; pp = p_p_all[:, col]; y = y_all[:, col]
        s_b = bc_scoring(y.unsqueeze(1), pb.unsqueeze(1)).squeeze().numpy()
        s_p = bc_scoring(y.unsqueeze(1), pp.unsqueeze(1)).squeeze().numpy()
        f1_b = metrics.f1_score(y.numpy().astype(int), pb.numpy().round())
        f1_p = metrics.f1_score(y.numpy().astype(int), pp.numpy().round())
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
    pickle.dump({"coach420": {tag: aligned}}, open(os.path.join(outdir, f"{tag}_aligned.pkl"), "wb"))
    print("wrote:", os.path.join(outdir, f"{tag}_bound.csv"), f"{tag}_af3.csv", f"{tag}_pVb_score.csv", f"{tag}_aligned.pkl", sep="\n  ")
    print("\n[ligand] pooled: bound auc=%.3f f1=%.3f | af3 auc=%.3f f1=%.3f" % (
        np.stack(sb).T[6, 3], np.stack(sb).T[8, 3], np.stack(sp).T[6, 3], np.stack(sp).T[8, 3]))

def run_coach420(s, model, cd, cm, frame, plm_type, fusion, need_rsa, tag, outdir, stage="all"):
    ids = coach420_ids(s["label"])
    cache_dir = os.path.join(outdir, "_cache")  # struct+emb cache is model-independent, kept across runs
    os.makedirs(cache_dir, exist_ok=True)
    struct_pkl = os.path.join(cache_dir, "coach420_structures.pkl")
    emb_h5 = os.path.join(cache_dir, "coach420_emb.h5" if plm_type in (None, "ESM2") else f"coach420_emb_{plm_type}.h5")
    cache = stage_struct(ids, struct_pkl)
    if stage in ("all", "emb"):
        stage_emb(cache, cd, frame, plm_type, emb_h5)
    if stage in ("all", "infer"):
        pp_i, pb_i, y_i, aligned = stage_infer(cache, cd, cm, frame, plm_type, fusion, need_rsa, model, s["label"], emb_h5)
        write_coach420_outputs(pp_i, pb_i, y_i, aligned, tag, outdir)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sets", default=None, help="comma-separated subset of SETS (default: all 5)")
    ap.add_argument("--limit", type=int, default=0, help="std sets only: cap #chains (debug)")
    ap.add_argument("--stage", choices=["all", "struct", "emb", "infer"], default="all",
                     help="coach420 only: resume from a given stage")
    a = ap.parse_args()
    tag = a.tag or os.path.basename(a.model_dir.rstrip("/"))
    outdir = os.path.join(RESULTS, "_runs", f"{tag}_benchmark_out")
    os.makedirs(outdir, exist_ok=True)

    model, cd, cm, frame, plm_type, fusion, need_rsa = load_model(a.model_dir)
    from src.emb_cache import LazyPLMPack
    plm_pack = LazyPLMPack(DEVICE, plm_type) if plm_type else None
    print(f"model={tag} frame={frame} plm={plm_type} fusion={fusion}")

    sel = a.sets.split(",") if a.sets else list(SETS)
    summary = []
    for sname in sel:
        s = SETS[sname]
        if s["cache"]:
            run_coach420(s, model, cd, cm, frame, plm_type, fusion, need_rsa, tag, outdir, stage=a.stage)
        else:
            row = run_std_set(sname, s, model, cd, frame, plm_type, fusion, need_rsa, plm_pack, tag, outdir, limit=a.limit)
            summary.append(row)
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(outdir, f"{tag}_benchmark_summary.csv"), index=False)
        print("wrote", os.path.join(outdir, f"{tag}_benchmark_summary.csv"))

if __name__ == "__main__":
    main()
