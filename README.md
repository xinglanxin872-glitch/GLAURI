# GLAURI

Residue-level prediction of protein, nucleic-acid, ion and ligand interfaces.
The supplied GLAURI inference weights use true-Cα geometry. Paired inference
configuration and checkpoint identity are in `model/GLAURI/`.

Explore co-label structure sets and their interface-specific binding modes with
the [chain-name lookup](colabel/BINDING_MODES.md), including a ≥4-residue filter.

## Install

Use Python 3.10. The recorded environment is in
[reference_versions.json](requirements/reference_versions.json); matching graph
extension versions are listed in [graph-reference.txt](requirements/graph-reference.txt).
Install compatible PyTorch and torch-scatter/torch-cluster wheels first, then
the runtime dependencies and this project from its root:

```bash
python -m pip install torch-geometric==2.7.0
python -m pip install -r requirements/runtime.txt
python -m pip install --no-deps -e .
```

The reference PyTorch build is a CUDA 12.8 nightly; retain its working wheels or
environment when reproducing exact paper numerics. Prepare the ESM3 namespace
from upstream SDK source version 3.2.3 (or its downloaded wheel):

```bash
python tools/install_esm3lib.py --source /path/to/upstream/esm --target vendor
# Alternatively: --wheel /path/to/esm-3.2.3-py3-none-any.whl --target vendor
export PYTHONPATH="$PWD/vendor:${PYTHONPATH:-}"
```

The adapter retains the upstream license and separates ESM3 from fair-esm's
`esm` namespace. This setup does not download pretrained weights.

Large H5 datasets and ESM3 weights are external assets. See
[assets/manifest.json](assets/manifest.json) and [dataset restoration](dataset_release/README.txt).
Download ESM3 weights using their upstream access/usage terms before offline runs.

## Binding Mode Explorer

Search a chain name to inspect its structure set, four interface types, binding
modes and **medoid representative chains**. An index-only contact map shows each
mode along the full reference FASTA. With H5 coordinates, also view the selected
pocket on a monomer cartoon or export a PyMOL script. The default filter retains
modes with at least four participating reference residues.

The browser uses a small CPU-only environment and needs no model weights or
training installation. From the repository root, with Python 3.10:

```bash
python -m pip install -r requirements/binding-browser.txt
# Extract the separately distributed index archive into colabel/outputs/ first.
python colabel/binding_modes.py serve \
  --db colabel/outputs/binding_modes/binding_modes.sqlite \
  --h5 /path/to/restored/bio_251210.h5 --port 8765
```

Keep the terminal running and open **http://127.0.0.1:8765**. For Remote SSH,
forward port **8765** in VS Code's **Ports** panel first. Use **Search** with
`6CBZ_B` or `2PYI_A`, then choose an interface and mode or click **Highlight pocket**.
The HTML requires this Python service; opening it directly on GitHub is insufficient.

**Data availability:** the SQLite index and manifest are separate from the code.
`GLAURI_binding_modes_v2.zip` has been prepared locally, but its public download
URL has not yet been added. The [processed H5 dataset](https://huggingface.co/datasets/sindax/GLAURI_processed/tree/main)
provides the native coordinates after restoration; it does not currently include
the browser index. Without H5, the index supports reference-sequence contact maps,
tables and JSON. Use `--index-only` to explicitly disable coordinate loading.
For cartoons, put the restored native file at `data/bio_251210.h5` or supply its
location with `--h5 /path/to/restored/bio_251210.h5`.
See [browser setup and binding-mode definitions](colabel/BINDING_MODES.md#browser-search).

## Predict one chain

```bash
python infer_one.py examples/apo_4x8m_A.pdb --chain A --out-prefix out/example
# Optional versioned float32 embedding cache:
python infer_one.py examples/apo_4x8m_A.pdb --chain A --out-prefix out/cached --embedding-cache-dir out/esm_cache
```

Default weights: `model/GLAURI/model_best_auc.pt`. Outputs are four probability-colored
PDB files, a full-precision scores CSV and an inference manifest. B-factors encode
probabilities in [0,1]; unscored atoms use -1. PDB/mmCIF input is supported.
Use `--device cpu` for CPU execution. A different checkpoint requires an explicit
`--model-dir` / `--checkpoint`; current training defaults never replace its snapshot.

For a resumable structure cohort, `python tools/infer_cohort.py --cohort
/path/to/cohort.json --output /path/to/new_predictions` uses the same predictor
and writes four-column NPZ probabilities. The script documents its small JSON
schema; this is inference only, with no GO analysis or plotting.

## Train and prepare data

GLAURI is the main model; configuration files also define the paper's ablations
and single-task models. Variants change
inputs, weighting, co-label use or active tasks.
[VARIANTS](model/VARIANTS.md) lists changes; [training flow](model/TRAINING_DECISIONS.md)
summarizes how configuration choices enter training.

Configuration merges `config_fixed.json` → `config_defaults.json` → `configs/*.json`.
`model_registry.json` records names, aliases and checkpoint provenance only;
variant changes have one source in their JSON config files.
Deployment paths are separate in `model/paths.example.json`.
See [training targets](colabel/README.md#training-targets) for annotation preparation.

```bash
python curation/generate_final_splits.py --check-only
python -m model.prepare --config all
# Only build masks when deploying without the verified released CA masks:
python -m model.build_masks --workers 8
python -m model.train --config GLAURI --print-config
python -m model.train --config GLAURI --check-inputs
bash model/train.sh GLAURI
# Submit the registered paper models:
bash model/submit_all.sh
```

Use `--paths /path/to/paths.json` with preparation/training commands, or set
`GLAURI_PATHS` for shell/Slurm commands. See [training flow](model/TRAINING_DECISIONS.md).
Checkpoint selection uses active-task H5 validation AUCs only. The bundled
inference weights predate this training-validation protocol; they are not a
newly trained checkpoint from these splits. Full training records are archived
separately. Output directories must be new. Training writes best-AUC and last weights,
effective config, parameter decisions, manifests, validation membership and logs.

## Evaluate

```bash
python -m benchmark.paper_pipeline --inputs /path/to/evaluation_inputs.json --runs-root /path/to/paired_runs --output /path/to/new_results --stage plan
python -m benchmark.paper_pipeline --inputs /path/to/evaluation_inputs.json --runs-root /path/to/paired_runs --output /path/to/new_results --stage all
```

The standard-test, weighting and ablation groups use verified memberships and
common evaluation labels; [benchmark membership and split details](benchmark/EVAL_SCRIPTS.md#benchmark-membership-and-data-splits)
are documented with the evaluation scripts. Each of
four interface types has a separate curated H5 validation cohort, sampled from
native-positive sequence clusters without model scores. AlloBench retains 25
validation and 58 test targets. Standard tests: MaSIF53, DNA129, RNA117, Zn59,
COACH420. External methods use frozen predictions.
[Evaluation instructions](benchmark/EVAL_SCRIPTS.md) explain artifacts and cache protocols.
Input/co-label ablations also evaluate AlloBench-test; single-task
models use their matching output heads only.

Plotting, SI tables, manuscript cases and human-proteome/Fig5B analysis belong to
the separate paper reproduction package. Fig1C is an annotation-cluster statistic
and remains unchanged. No paper-result directory is overwritten by training.

## Release provenance

[The model card](model/GLAURI/README.md), [asset manifest](assets/manifest.json),
[third-party notices](NOTICE.txt) describe provenance. Repository packaging does not
replace upstream model or dataset licenses.

Repository-authored source code uses the [MIT License](LICENSE). Third-party
source, pretrained weights and datasets retain their own applicable terms;
see NOTICE.txt and third_party_licenses/.
