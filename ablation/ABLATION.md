# Ablation suite

9 runs, each changing exactly one thing relative to the baseline. Same data split, sampler
weights, pairwise mask, validation subset (2048, seed 42), architecture, parameter count
(2,404,796), init and batch order — only the listed switch differs.

Baseline = the joint model reported elsewhere in this repo (`model/config_baseline.py`,
checkpoint tag `ourmodel-rsa`). No separate run needed for it here — it's the existing
baseline training run.

Feature ablations zero inputs **at the forward pass**, not by deleting columns, so parameter
count and initialization stay identical to baseline and no ablation can be explained by "one
fewer input dim -> fewer params."

## Experiments

| # | name | change | config | script |
|---|---|---|---|---|
| 0 | baseline | — | `config_baseline.py` | *(existing baseline run, not part of this bundle)* |
| 1 | zeroseq | `zero_seq_input=True` | `config_zeroseq.py` | `run_zeroseq.sh` |
| 2 | zerostruct | `zero_struct_input=True` | `config_zerostruct.py` | `run_zerostruct.sh` |
| 3 | zerorsa | `zero_rsa_input=True` | `config_zerorsa.py` | `run_zerorsa.sh` |
| 4 | zerostructrsa | struct + rsa zeroed | `config_zerostructrsa.py` | `run_zerostructrsa.sh` |
| 5 | nativeonly | `--train_native_only` (labels: native contacts only; valid still native ∪ colabel) | `config_baseline.py` | `run_nativeonly.sh` |
| 6 | single_protein | `--active_tasks 0` | `config_baseline.py` | `run_single_protein.sh` |
| 7 | single_nucleic | `--active_tasks 1` | `config_baseline.py` | `run_single_nucleic.sh` |
| 8 | single_ion | `--active_tasks 2` | `config_baseline.py` | `run_single_ion.sh` |
| 9 | single_ligand | `--active_tasks 3` | `config_baseline.py` | `run_single_ligand.sh` |

## What each feature ablation actually zeroes

Node scalar layout: `[0,35) one-hot+dihedrals | [35,36) RSA | [36,1572) ESM3`.

| | `[0,35)` | RSA `[35,36)` | ESM3 `[36,1572)` | edge scalars `e` | edge vectors `xi` |
|---|---|---|---|---|---|
| zeroseq | kept | kept | **zeroed** | kept | kept |
| zerostruct | **zeroed** | kept | kept | **zeroed** | **zeroed** |
| zerorsa | kept | **zeroed** | kept | kept | kept |
| zerostructrsa | **zeroed** | **zeroed** | kept | **zeroed** | **zeroed** |

**Read these names literally, not loosely:**
- `zeroseq` removes only the PLM embedding. The 29-dim residue-identity one-hot survives, so
  sequence identity is not fully removed — this measures the marginal contribution of the
  *PLM representation*, not of "having sequence information" per se.
- `zerostruct` removes explicit structural scalars/edge features. Coordinates, the 64-NN graph
  topology, local frames (`f_ij`), and node vector channels (backbone/sidechain orientation)
  are still present — GCP message passing needs them to run at all. This measures the marginal
  contribution of *explicit structural descriptors*, not "no structure."
- `zerorsa` is a clean single-column ablation.
- `zerostructrsa` = zerostruct + zerorsa: only the ESM3 segment survives among node scalars —
  the complement of `zeroseq` (which keeps only structure). Together they let you separate
  RSA's *conditional* contribution (with vs. without the other structural descriptors present)
  from its marginal one.

Don't report any of these as "no sequence" / "no structure" without the qualifier above.

## Joint vs. single-task comparison

Network is always 4-output; inactive columns forward normally but their loss is exactly zero,
so no gradient flows to them (verified with a synthetic-gradient test — see git history / the
original 2026-08 dev notes if you need the proof, not reproduced here). Compare per-task using
the matching output index, not the `model.pt` that was checkpointed on the mean:

```
joint model_0.pt col 0  vs  single_protein model_0.pt col 0
joint model_1.pt col 1  vs  single_nucleic model_1.pt col 1
joint model_2.pt col 2  vs  single_ion     model_2.pt col 2
joint model_3.pt col 3  vs  single_ligand  model_3.pt col 3
```
`model.pt` (joint) is selected on mean AUC across columns; single-task `model.pt` is selected
on its one active column — comparing those two directly biases in favor of single-task.

## Running

```bash
cd <dir containing this repo's model/ folder>
bash submit_ablation_suite.sh              # deploy + precheck + sbatch all 9
DRY_RUN=1 bash submit_ablation_suite.sh    # deploy + precheck only
DEPLOY_ONLY=1 bash submit_ablation_suite.sh # just lay files down
```

`model.py` is not part of this folder — the ablation switches live in `../model/model.py`
directly (confirmed identical, single source of truth). The submit script only backs up and
overwrites `config_baseline.py` in the target `model/` directory.

## Caveat

Single seed per run. Don't claim a feature "necessarily" helps or hurts off one run — that
needs several seeds with mean ± std.
