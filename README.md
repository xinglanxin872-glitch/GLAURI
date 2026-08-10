# GLAURI

A four-channel per-residue protein interface predictor: one model, one forward pass, four
outputs — protein–protein, protein–nucleic-acid (DNA/RNA), protein–ion, and protein–ligand
binding-site probability, all per residue, no separate model per interface type.

## Quickstart

```bash
conda env create -f environment.yml
conda activate glauri
# PLM weights are pulled from your local HuggingFace/torch hub cache the first time you
# run with plm="esm3" (the shipped checkpoint) -- see environment.yml's esm3lib section
# if this is a fresh machine.

python3 infer_one.py _smoketest/apo_4x8m_A.pdb --out-prefix /tmp/apo_4x8m_A
```

This scores a single protein chain and writes four B-factor-colored PDBs
(`<prefix>-{protein,nn,ion,ligand}.pdb`, B-factor = per-residue probability in [0,1],
scored chain only — everything else in the output PDB gets B-factor -1.00 so you can grey
it out in PyMOL: `color grey80, b < 0`).

**Verify your install**: `_smoketest/README.md` has the exact expected numbers
(mean/max/n≥0.5 per channel) for that command. If your run matches, your environment is
correctly set up end to end (structure I/O → GCP model → ESM3 embedding → inference).

## Layout

```
GLAURI/
├── infer_one.py      entry point: one chain in, four channel PDBs out (see Quickstart)
├── environment.yml    conda + pip spec (torch is a CUDA nightly build, see comments there)
├── _smoketest/        reference input/output for infer_one.py, see its own README
├── src/                shared library: structure I/O, feature encoding, PLM embedding,
│                      the process_structure -> process_seq_batch -> inference pipeline
│                      every eval/case_study script in this repo is built on
├── model/              the model itself (model.py, main.py = training entry point,
│                      config.py/config_baseline.py = the shipped hyperparameters) +
│                      model/GLAURI/ = the trained checkpoint (model.pt, ~9MB)
├── curation/            builds the training H5 from raw structures (build_dataset.py) +
│                      the train/val/test split (split/split_dataset.ipynb,
│                      split/cluster/ = mmseqs clustering intermediates,
│                      final_split/ = the actual split used to train model/GLAURI)
├── colabel/             the pocket-labeling pipeline that produces training labels for
│                      the non-native-contact ("pocket") training signal -- 3-step chain,
│                      see colabel/run_smoketest.sh for a runnable end-to-end example
├── ablation/            configs + launch scripts for the ablation runs (zero-seq,
│                      zero-struct, zero-rsa, etc.)
├── benchmark/           runs GLAURI + external baselines on 5 standard benchmark tasks
│                      (protein/DNA-129/RNA-117/ion/coach420-ligand); see
│                      benchmark/EVAL_SCRIPTS.md and benchmark/results/ for output tables
└── case_study/           three worked examples (apo/holo conformational change,
                         a multi-interface hub protein, a nucleic-acid case), each with
                         a quick AUC/TPR check against baselines -- see each subdir
```

## External data you must supply

Not shipped in this repo (paths are symlinks on the machine this was packaged on —
`find . -type l` to see what's currently linked where, then point these at your own copies):

| what | used by | where it lived when packaged |
|---|---|---|
| `bio_251210.h5` | training (`model/main.py`), repo root | full structure/contact dataset, too large to ship |
| `bio_251210_rsa.h5` | training (RSA sidecar) | precomputed per-residue RSA, `curation/add_sasa_to_h5.py` output |
| `model/all_seq_dict.pkl` | training | chain_id -> sequence lookup |
| `colabel/all_colabel.pkl`, `colabel/colabel_mask_all.pkl` | training (`--use_colabel`) | pocket-labeling pipeline output, see `colabel/` |
| `model/GLAURI/mask_dict_train.pkl` / `mask_dict_valid.pkl` | training (pairwise masks) | built by `model/build_mask_dict_pairwise.py` |

None of the above are needed to just run `infer_one.py` on your own structures — only for
retraining. `model/GLAURI/model.pt` (the trained checkpoint) **is** shipped.

## Reproducing a specific piece

- **Inference on one structure**: `infer_one.py` (see Quickstart).
- **Pocket-labeling pipeline**: `colabel/run_smoketest.sh` runs the real 3-step pipeline
  end to end on a small real data slice.
- **Benchmark eval**: `benchmark/EVAL_SCRIPTS.md`.
- **Train/val/test split**: `curation/split/split_dataset.ipynb` (self-contained, reads
  `curation/split/lists/` + `curation/split/cluster/`, writes `curation/final_split/`).
- **Training**: `model/<date>_*.sh` launch scripts (SLURM), `model/main.py --help` for the
  full CLI.
