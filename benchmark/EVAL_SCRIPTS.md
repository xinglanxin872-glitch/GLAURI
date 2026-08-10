# Benchmark eval scripts

Two scripts cover all 5 tasks (protein, DNA-129, RNA-117, ion, ligand/coach420). Each has a
counterpart role:

| | scores external baselines | runs GLAURI itself |
|---|---|---|
| all 5 tasks | `bench_eval.py` | `model_benchmark_eval.py` |

Neither script runs a baseline tool (p2rank/fpocket/GrASP/ProtGeoNet/MaSIF/PeSTo/GraphSite/
Metal3D) — those are assumed already run elsewhere; these scripts only align + score their
outputs. `model_benchmark_eval.py` is the only one that does inference (our own checkpoint).

## `bench_eval.py`

Reads each baseline method's precomputed per-residue scores (or, for coach420, raw tool
output files), aligns to the ground-truth label with `align_anchor` (two independent
pairwise alignments, ref-vs-bound and ref-vs-predict, intersected on ref indices), scores
with `bc_scoring` (acc/ppv/npv/tpr/tnr/mcc/auc/std) + pair-consistency (mae/w_mae/rmse/
JS_div/ICC).

```bash
python bench_eval.py                              # all 5 benchmarks
python bench_eval.py --benchmarks MaSIF-53 Zn-59   # subset
```

Writes `benchmark_per_chain.csv`, `benchmark_summary.csv`, `benchmark_aligned.pkl` under
`results/` (override with `--outdir`).

coach420 is different from the other 4: there's no fixed reference-sequence label file —
`ref_seq` is each chain's own predict_seq, and labels are decoded per-chain from
`results/ligand/coach420_label.csv`'s token field (`BENCH["coach420"]["label"] =
("coach420csv", ...)`, handled by the `dynamic_label` branch in `run()`).

Baseline result sources (see `BENCH` dict for exact paths):

| task | methods | source |
|---|---|---|
| MaSIF-53 | MaSIF, PeSTo | `results/protein/{masif,pesto}/` |
| DNA-129, RNA-117 | GraphSite | `results/NN/gesite/` |
| Zn-59 | Metal3D | `results/ion/metal3d_results{,_predicted}/` |
| coach420 | p2rank, fpocket, grasp, protgeonet | raw tool outputs, **not shipped** — set `COACH420_RAW_DIR` (see below) |

GraphBind (DNA-129/RNA-117) isn't listed: it was never scored through this script (a
separate model run appends its rows straight into `benchmark_aligned.pkl`), so there's no
loader for it here — see `results/NN/graphbind/`.

## `model_benchmark_eval.py`

Loads a GLAURI checkpoint, runs inference, aligns + scores the same way.

```bash
python model_benchmark_eval.py --model-dir <dir> --tag <name>                  # all 5
python model_benchmark_eval.py --model-dir <dir> --sets coach420 --stage infer # resume
```

MaSIF-53/DNA-129/RNA-117/Zn-59 run single-pass (no disk cache). coach420 (420 chains,
expensive ESM embeddings, evaluated repeatedly against different checkpoints) keeps a
3-stage disk cache instead — `--stage struct|emb|infer|all`, resumable, cache lives under
`results/_runs/<tag>_benchmark_out/_cache/` and is checkpoint-independent (only the model
weights, not the cached structures/embeddings, change between runs).

Output: `results/_runs/<tag>_benchmark_out/` — `{set}_{tag}_aligned.pkl` + `{tag}_benchmark_summary.csv`
for the 4 std sets; `{tag}_{bound,af3,pVb_score,aligned}` for coach420 (4 foreground columns,
not the 1-column-per-set format the others use — ligand test set, but the model scores all
4 interface classes on it, worth keeping visible).

## Not shipped in this repo

Both scripts need external data this repo doesn't include (only already-computed *results*
are checked in, not raw structures or raw baseline-tool outputs):

- `model_benchmark_eval.py`: bound structure directories (`STD_STRUCT_DIR`) and AF3-predicted
  structures (`STD_STRUCT_DIR/infer_output/`) for the 4 std sets; coach420 bound/predicted
  structures (`COACH420_STRUCT_DIR`).
- `bench_eval.py`: coach420 baseline raw tool outputs (`COACH420_RAW_DIR` — p2rank
  `.pdb_residues.csv`, fpocket pocket dirs, GrASP colored pdbs, ProtGeoNet fold pkls).

All overridable via environment variables (see top of each script); see `docs/DATA.md` for
what to put there and how to reproduce it.

## Hotspot / regression

Out of scope for both scripts — neither has a hotspot task or a regression-model entry.
(If you're looking for those, they were intentionally dropped from this release.)
