#!/usr/bin/env python3
"""Score precomputed external-baseline results on the five benchmark tasks.

The ligand task is COACH420, with the complete test membership applied
before scoring. Source labels and raw-tool files retain their original names.
GLAURI inference is provided by model_benchmark_eval.py.
"""
import os, sys, re, csv, glob, pickle, tempfile, subprocess, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.coach_split import filter_coach_test_ids

import numpy as np
import pandas as pd
import torch as pt

csv.field_size_limit(10**8)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(HERE), "data"))
ROOT = os.environ.get("BENCH_BASELINES_DIR", os.path.join(DATA_ROOT, "benchmark_baselines"))
LABELS = os.environ.get("BENCH_LABELS_DIR", os.path.join(HERE, "labels"))
OUTPUT = os.environ.get("BENCH_RESULTS_DIR", os.path.join(HERE, "results"))
# Raw baseline-tool outputs for coach420 (p2rank residues.csv, fpocket pocket dirs, GrASP
# colored pdbs, ProtGeoNet fold pkls) are NOT part of this repo -- results/ligand/{method}/
# only holds this script's own (already-scored) output. Point COACH_RAW_DIR at wherever
# you reproduced those raw runs; see benchmark/EVAL_SCRIPTS.md.
COACH_RAW_DIR = os.environ.get("COACH_RAW_DIR", os.path.join(ROOT, "ligand_raw"))
bc_score_names = ['acc', 'ppv', 'npv', 'tpr', 'tnr', 'mcc', 'auc', 'std']

# --------------------------------------------------------------------------
# scoring (copied faithfully from src.scoring / src.utils_eval)
# --------------------------------------------------------------------------
def bc_scoring(y, p):
    """y,p : (N,1) tensors. returns (8,1)."""
    q = (p >= .5).to(p.dtype)
    TP = pt.sum(q * y, dim=0); TN = pt.sum((1-q)*(1-y), dim=0)
    FP = pt.sum(q*(1-y), dim=0); FN = pt.sum((1-q)*y, dim=0)
    P = pt.sum(y, dim=0); N = pt.sum(1-y, dim=0)
    def safe(v):
        v = v.clone(); v[pt.isinf(v)] = float('nan'); return v
    acc = (TP+TN)/(TP+TN+FP+FN)
    ppv = TP/(TP+FP); ppv[~(P > 0)] = float('nan')
    npv = TN/(TN+FN); npv[~(N > 0)] = float('nan')
    tpr = safe(TP/(TP+FN)); tnr = safe(TN/(TN+FP))
    mcc = safe(((TP*TN)-(FP*FN))/pt.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    # auc
    auc = pt.full((y.shape[1],), float('nan'))
    m = (P > 0) & (N > 0)
    if bool(pt.any(m)):
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(y[:, m].cpu().numpy(), p[:, m].cpu().numpy(), average=None)
        auc[m] = pt.tensor(np.atleast_1d(a), dtype=pt.float32)
    std = pt.std(p, dim=0)
    return pt.stack([acc, ppv, npv, tpr, tnr, mcc, auc, std])

def cal_mae(p, q):    return pt.mean(pt.abs(q - p)).item()
def cal_w_mae(p, q):
    w = pt.abs(q + p - 1)
    return (pt.sum(w*pt.abs(p-q))/pt.sum(w)).item() if w.sum() > 0 else 0.0
def cal_rmse(p, q):   return pt.sqrt(pt.mean((p-q)**2)).item()
def cal_JS_div(p, q, eps=1e-12):
    p = p.double(); q = q.double()  # float32 rounds 1-eps back to 1.0 -> 0*log0 NaN
    p = pt.clamp(p, eps, 1-eps); q = pt.clamp(q, eps, 1-eps); m = .5*(p+q)
    kl_pm = p*pt.log(p/m)+(1-p)*pt.log((1-p)/(1-m))
    kl_qm = q*pt.log(q/m)+(1-q)*pt.log((1-q)/(1-m))
    return (.5*(kl_pm+kl_qm)).mean().item()
def cal_ICC2_1(p, q):
    from src.metrics import icc_a1
    return icc_a1(p, q)

# --------------------------------------------------------------------------
# alignment (pairwise, subprocess clustalo)
# --------------------------------------------------------------------------
def _read_fasta(path):
    name, seq, out = None, [], {}
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if name is not None: out[name] = "".join(seq)
            name, seq = ln[1:].strip().split()[0], []
        else:
            seq.append(ln.strip())
    if name is not None: out[name] = "".join(seq)
    return out

def pairwise_map(a_seq, b_seq):
    """Align a vs b; return dict {a_idx: b_idx} for columns where BOTH non-gap.
    Anchors on `a`; robust to repeats/multimers in `b`. Free identity shortcut when a==b."""
    if a_seq == b_seq:
        return {i: i for i in range(len(a_seq))}
    fd_i, fin = tempfile.mkstemp(suffix=".fa"); os.close(fd_i)
    fd_o, fout = tempfile.mkstemp(suffix=".fa"); os.close(fd_o)
    try:
        with open(fin, "w") as f:
            f.write(f">a\n{a_seq}\n>b\n{b_seq}\n")
        subprocess.run(["clustalo", "-i", fin, "-o", fout, "--force", "--auto"],
                       check=True, capture_output=True)
        aln = _read_fasta(fout)
    finally:
        for p in (fin, fout):
            try: os.remove(p)
            except OSError: pass
    A, B = aln["a"], aln["b"]
    ai = bi = 0; m = {}
    for ca, cb in zip(A, B):
        if ca != "-" and cb != "-": m[ai] = bi
        if ca != "-": ai += 1
        if cb != "-": bi += 1
    return m

def align_anchor(ref_seq, b_seq, p_seq):
    """Two independent pairwise alignments (ref-vs-b, ref-vs-p), intersected on ref indices.
    Returns (ri, bi, pi): same-length, same-order index arrays into ref/b/p respectively."""
    mb = pairwise_map(ref_seq, b_seq)
    mp = pairwise_map(ref_seq, p_seq)
    common = sorted(set(mb) & set(mp))
    ri = np.array(common, dtype=int)
    bi = np.array([mb[i] for i in common], dtype=int)
    pi = np.array([mp[i] for i in common], dtype=int)
    return ri, bi, pi

# --------------------------------------------------------------------------
# id normalisation + loaders for MaSIF-53 / DNA-129 / RNA-117 / Zn-59
# -> dict[norm_id] = (seq:str, scores:np.float array)
# --------------------------------------------------------------------------
def norm(x): return re.sub(r'_model$', '', str(x).strip()).upper()

def _read_label(path):
    """Fixed-reference-sequence label file: FASTA triplets (>id / seq / 0-1 string)."""
    out, lines = {}, [l.rstrip("\n") for l in open(path)]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            cid = norm(lines[i][1:])
            seq = lines[i+1].strip()
            lab = lines[i+2].strip()
            out[cid] = (seq, np.array([int(c) for c in lab], dtype=np.float32))
            i += 3
        else:
            i += 1
    return out

def load_masif_csv(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[norm(r["id"])] = (r["sequence"],
                              np.array(r["residue_scores"].split(), dtype=np.float32))
    return out

def load_spacecsv(path):     # GraphSite: id,results(space),seq
    out = {}
    for r in csv.DictReader(open(path)):
        out[norm(r["id"])] = (r["seq"], np.array(r["results"].split(), dtype=np.float32))
    return out

def load_commacsv(path):     # Metal3D: id,results(comma),seq
    out = {}
    for r in csv.DictReader(open(path)):
        out[norm(r["id"])] = (r["seq"], np.array(r["results"].split(","), dtype=np.float32))
    return out

def load_pesto_pkl(path):
    d = pickle.load(open(path, "rb")); out = {}
    for k, v in d.items():
        out[norm(k)] = (v["seq"], np.asarray(v["results"], dtype=np.float32).reshape(-1))
    return out

# --------------------------------------------------------------------------
# coach420 (ligand) specifics: dynamic per-chain label, raw baseline-tool loaders
# --------------------------------------------------------------------------
aa3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
       'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def label_stem(idx):
    p, c = idx.rsplit("_", 1); return f"{p.lower()}{c.upper()}"

def coach420_labels_for_refseq(ref_seq, label_field):
    """coach420_label.csv 'Label' field is tokens like 'A_12 A_57' (chain_resnum, 1-based
    on predict_seq); decode into a 0/1 vector over ref_seq (== predict_seq here)."""
    n = len(ref_seq); out = np.zeros(n, dtype=np.float32)
    if label_field is None or (isinstance(label_field, float) and label_field != label_field): return out
    s = str(label_field).strip()
    if not s or s.lower() == "nan": return out
    for chunk in s.split(";"):
        for tok in chunk.split():
            t = tok.strip()
            if "_" not in t: continue
            _, num = t.rsplit("_", 1)
            if num.isdigit():
                k = int(num)
                if 1 <= k <= n: out[k-1] = 1.0
    return out

from Bio.PDB import PDBParser
_pdbp = PDBParser(QUIET=True)
def seq_and_atommap(pdb):
    st = _pdbp.get_structure("s", pdb)
    idxres = {}; atom2res = {}
    for model in st:
        for chain in model:
            for res in chain:
                if res.id[0] != " ": continue
                rn = res.resname.strip()
                key = (chain.id, res.id[1])
                if key not in idxres:
                    if rn not in aa3: continue
                    idxres[key] = rn
                for atom in res: atom2res[atom.serial_number] = key
    sres = sorted(idxres.items(), key=lambda x: (x[0][0], x[0][1]))
    seqidx = {k: i for i, (k, _) in enumerate(sres)}
    seq = "".join(aa3[rn] for _, rn in sres)
    return seq, seqidx, atom2res

_COACH_IDS = None
def coach_test_ids():
    global _COACH_IDS
    if _COACH_IDS is None:
        lab = pd.read_csv(BENCH["coach_test"]["label"][1])
        _COACH_IDS = filter_coach_test_ids({f"{s[:4].upper()}_{s[4:].upper()}" for s in lab["PDB ID"].astype(str)}, require_all=True)
    return _COACH_IDS

def load_p2rank(side):
    d = {}
    rdir = os.path.join(COACH_RAW_DIR, "prank_res", side)
    for idx in coach_test_ids():
        p = os.path.join(rdir, f"{idx}.pdb_residues.csv")
        if not os.path.isfile(p): continue
        df = pd.read_csv(p); df.columns = df.columns.str.strip()
        df["rl"] = pd.to_numeric(df["residue_label"].astype(str).str.strip(), errors="coerce")
        df = df[df["rl"].notna()]
        df = df[df["residue_name"].str.strip().isin(aa3.keys())].copy()
        df = df.sort_values(["chain", "rl"])
        sc = df["score"].to_numpy(float)
        rng = sc.max()-sc.min()
        n = (sc-sc.min())/rng if rng > 0 else np.zeros_like(sc)
        seq = "".join(aa3[r.strip()] for r in df["residue_name"])
        d[idx] = (seq, n.astype(np.float32))
    return d

def load_fpocket(side):
    d = {}
    sdir = os.path.join(COACH_RAW_DIR, "staging", side)
    for idx in coach_test_ids():
        pdb = os.path.join(sdir, f"{idx}.pdb"); out = os.path.join(sdir, f"{idx}_out", "pockets")
        if not os.path.isfile(pdb) or not os.path.isdir(out): continue
        seq, seqidx, atom2res = seq_and_atommap(pdb)
        atoms = []
        for pf in sorted(glob.glob(os.path.join(out, "pocket*_atm.pdb"))):
            for line in open(pf):
                if line.startswith("ATOM"):
                    try: atoms.append(int(line[6:11]))
                    except ValueError: pass
        y = np.zeros(len(seqidx), dtype=np.float32)
        for a in atoms:
            k = atom2res.get(a)
            if k in seqidx: y[seqidx[k]] = 1.0
        d[idx] = (seq, y)
    return d

def load_grasp(side):
    """GrASP colored PDB: per-residue = max b-factor over atoms."""
    d = {}
    cdir = os.path.join(COACH_RAW_DIR, "grasp_res", f"{side}_colored" if side == "bound" else "predict_colored")
    for idx in coach_test_ids():
        p, c = idx.rsplit("_", 1)
        cands = [os.path.join(cdir, f"{idx}_probs.pdb"),
                 os.path.join(cdir, f"{p.lower()}_{c.lower()}_probs.pdb"),
                 os.path.join(cdir, f"{p.lower()}_{c.lower()}_model_probs.pdb")]
        path = next((x for x in cands if os.path.isfile(x)), None)
        if path is None: continue
        st = _pdbp.get_structure("g", path)
        idxres = {}; rmax = {}
        for model in st:
            for chain in model:
                for res in chain:
                    if res.id[0] != " ": continue
                    rn = res.resname.strip()
                    if rn not in aa3: continue
                    key = (chain.id, res.id[1]); idxres.setdefault(key, rn)
                    mx = max((float(a.bfactor) for a in res), default=0.0)
                    rmax[key] = max(rmax.get(key, 0.0), mx)
        sres = sorted(idxres.items(), key=lambda x: (x[0][0], x[0][1]))
        seq = "".join(aa3[rn] for _, rn in sres)
        y = np.array([rmax[k] for k, _ in sres], dtype=np.float32)
        d[idx] = (seq, y)
    return d

def load_protgeonet(side):
    """ProtGeoNet output pkl = {'df': DataFrame[fold,protein_id,seq,pred], 'meta':..}.
    Average per-residue `pred` over the 10 folds; protein_id matches coach_test_ids()."""
    fn = "coach420_420_ProtGeoNet_bound_output.pkl" if side == "bound" else "coach420_420_ProtGeoNet_af3_output.pkl"
    raw = pickle.load(open(os.path.join(COACH_RAW_DIR, "protgeonet", fn), "rb"))
    df = raw["df"] if isinstance(raw, dict) and "df" in raw else raw
    ids = coach_test_ids()
    nid = {re.sub(r'[^A-Za-z0-9]', '', i).upper(): i for i in ids}
    d = {}
    for pid, g in df.groupby("protein_id"):
        seq = g.iloc[0]["seq"]
        preds = np.stack([np.asarray(p, float) for p in g["pred"]])  # (folds, N)
        sc = preds.mean(0).astype(np.float32)
        key = nid.get(re.sub(r'[^A-Za-z0-9]', '', str(pid)).upper(), str(pid))
        d[key] = (seq, sc)
    return d

# --------------------------------------------------------------------------
# benchmark configuration
# --------------------------------------------------------------------------
def P(*a): return os.path.join(ROOT, *a)
def L(*parts): return os.path.join(LABELS, *parts)

BENCH = {
 "MaSIF-53": dict(itype="protein", label=("fasta", L("protein", "masif-53-label.txt")), methods={
     "MaSIF":   lambda: (load_masif_csv(P("protein", "masif", "scores_exp.csv")),
                         load_masif_csv(P("protein", "masif", "scores_pred.csv"))),
     "PeSTo":   lambda: (load_pesto_pkl(P("protein", "pesto", "masif53_bound_pesto_article.pkl")),
                         load_pesto_pkl(P("protein", "pesto", "masif53_pred_pesto_article.pkl"))),
 }),
 "DNA-129": dict(itype="DNA/RNA", label=("fasta", L("NN", "DNA-129_Test.txt")), methods={
     "GraphSite": lambda: (load_spacecsv(P("NN", "gesite", "DNA-129_from_pdb.csv")),
                           load_spacecsv(P("NN", "gesite", "DNA-129_from_seq.csv"))),
 }),
 "RNA-117": dict(itype="DNA/RNA", label=("fasta", L("NN", "RNA-117_Test.txt")), methods={
     "GraphSite": lambda: (load_spacecsv(P("NN", "gesite", "RNA-117_from_pdb.csv")),
                           load_spacecsv(P("NN", "gesite", "RNA-117_from_seq.csv"))),
 }),
 "Zn-59": dict(itype="ion", label=("fasta", L("ion", "Zn-59_Test.txt")), methods={
     "Metal3D":  lambda: (load_commacsv(P("ion", "metal3d_results", "labels_shell_3.5A.csv")),
                          load_commacsv(P("ion", "metal3d_results_predicted", "labels_shell_3.5A.csv"))),
 }),
 # dynamic_label: ref_seq = predict_seq (no fixed reference sequence file); labels are
 # decoded per-chain from the token field in coach420_label.csv. See coach420_labels_for_refseq.
 "coach_test": dict(itype="ligand", label=("coach420csv", L("ligand", "coach420_label.csv")), methods={
     "p2rank":     lambda: (load_p2rank("bound"), load_p2rank("predict")),
     "fpocket":    lambda: (load_fpocket("bound"), load_fpocket("predict")),
     "grasp":      lambda: (load_grasp("bound"), load_grasp("predict")),
     "protgeonet": lambda: (load_protgeonet("bound"), load_protgeonet("predict")),
 }),
}

# --------------------------------------------------------------------------
def run(benchmarks):
    per_chain, summary = [], []
    aligned = {}   # benchmark -> method -> chainid -> {y,p_bound,p_pred,ref_idx,interface_type}
    for bn in benchmarks:
        cfg = BENCH[bn]
        label_kind, label_path = cfg["label"]
        dynamic_label = label_kind == "coach420csv"
        label = None if dynamic_label else _read_label(label_path)
        label_csv = pd.read_csv(label_path) if dynamic_label else None
        for mname, loader in cfg["methods"].items():
            bd, pr = loader()
            ids = filter_coach_test_ids(set(bd) & set(pr)) if dynamic_label else sorted(set(label) & set(bd) & set(pr))
            if not dynamic_label:
                miss_b = sorted(set(label) - set(bd)); miss_p = sorted(set(label) - set(pr))
                print(f"[{bn}/{mname}] eval={len(ids)}  miss_bound={len(miss_b)} miss_pred={len(miss_p)}", flush=True)
                if miss_b: print("    miss_bound:", miss_b)
                if miss_p: print("    miss_pred :", miss_p)
            else:
                print(f"[{bn}/{mname}] eval={len(ids)} (of {len(coach_test_ids())} chains)", flush=True)
            Y, PB, PP = [], [], []
            for cid in ids:
                bseq, bsc = bd[cid]; pseq, psc = pr[cid]
                if bseq is None: bseq = pseq  # protgeonet may lack a bound seq -> assume same coords as predict
                if dynamic_label:
                    rseq = pseq
                    mrow = label_csv[label_csv["PDB ID"].astype(str).str.lower() == label_stem(cid).lower()]
                    lab_str = "" if mrow.empty else mrow.iloc[0]["Label"]
                    y01 = coach420_labels_for_refseq(rseq, lab_str)
                else:
                    rseq, y01 = label[cid]
                # guard length mismatches between a source's own seq and its score vector
                if len(bsc) != len(bseq) or len(psc) != len(pseq) or len(y01) != len(rseq):
                    print(f"    !! len mismatch {cid}: ref {len(y01)}/{len(rseq)} "
                          f"bnd {len(bsc)}/{len(bseq)} prd {len(psc)}/{len(pseq)} -> skip")
                    continue
                try:
                    ri, bi, pi = align_anchor(rseq, bseq, pseq)
                except Exception as e:
                    print(f"    !! align fail {cid}: {e} -> skip"); continue
                y = pt.tensor(y01[ri]).float()
                pb = pt.tensor(bsc[bi]).float(); pp = pt.tensor(psc[pi]).float()
                if not (len(y) == len(pb) == len(pp)) or len(y) == 0:
                    print(f"    !! aligned-len {cid}: {len(y)}/{len(pb)}/{len(pp)} -> skip"); continue
                sb = bc_scoring(y[:, None], pb[:, None]).squeeze(1).tolist()
                sp = bc_scoring(y[:, None], pp[:, None]).squeeze(1).tolist()
                row = dict(benchmark=bn, method=mname, chainid=cid,
                           interface_type=cfg["itype"], n_aligned=len(y),
                           n_pos=int(y.sum().item()),
                           mae=cal_mae(pb[:, None], pp[:, None]),
                           w_mae=cal_w_mae(pb[:, None], pp[:, None]),
                           rmse=cal_rmse(pb[:, None], pp[:, None]),
                           JS_div=cal_JS_div(pb[:, None], pp[:, None]),
                           ICC=cal_ICC2_1(pb[:, None], pp[:, None]))
                for nm, v in zip(bc_score_names, sb): row[nm+"_b"] = v
                for nm, v in zip(bc_score_names, sp): row[nm+"_p"] = v
                per_chain.append(row)
                Y.append(y); PB.append(pb); PP.append(pp)
                aligned.setdefault(bn, {}).setdefault(mname, {})[cid] = dict(
                    interface_type=cfg["itype"],
                    ref_idx=ri.astype(np.int32),
                    y=y01[ri].astype(np.float32),
                    p_bound=bsc[bi].astype(np.float32),
                    p_pred=psc[pi].astype(np.float32))
            if Y:  # pooled-over-residues summary (whole_eval style)
                y = pt.cat(Y)[:, None]; pb = pt.cat(PB)[:, None]; pp = pt.cat(PP)[:, None]
                sb = bc_scoring(y, pb).squeeze(1).tolist()
                sp = bc_scoring(y, pp).squeeze(1).tolist()
                from sklearn.metrics import f1_score
                srow = dict(benchmark=bn, method=mname, interface_type=cfg["itype"],
                            n_chain=len(Y), n_res=int(y.numel()), pos_ratio=float(y.mean()),
                            F1_b=f1_score(y.int().numpy().ravel(), pb.round().int().numpy().ravel()),
                            F1_p=f1_score(y.int().numpy().ravel(), pp.round().int().numpy().ravel()),
                            mae=cal_mae(pb, pp), w_mae=cal_w_mae(pb, pp), rmse=cal_rmse(pb, pp),
                            JS_div=cal_JS_div(pb, pp), ICC=cal_ICC2_1(pb, pp))
                for nm, v in zip(bc_score_names, sb): srow[nm+"_b"] = v
                for nm, v in zip(bc_score_names, sp): srow[nm+"_p"] = v
                summary.append(srow)
    return per_chain, summary, aligned

def write_csv(rows, path):
    if not rows:
        print("no rows ->", path); return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print("wrote", path, f"({len(rows)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCH))
    ap.add_argument("--outdir", default=OUTPUT)
    args = ap.parse_args()
    pc, sm, aligned = run(args.benchmarks)
    os.makedirs(args.outdir, exist_ok=True)
    write_csv(pc, os.path.join(args.outdir, "benchmark_per_chain.csv"))
    write_csv(sm, os.path.join(args.outdir, "benchmark_summary.csv"))
    pkl_path = os.path.join(args.outdir, "benchmark_aligned.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(aligned, f)
    nchain = sum(len(c) for b in aligned.values() for c in b.values())
    print("wrote", pkl_path, f"({nchain} chain-entries)")
