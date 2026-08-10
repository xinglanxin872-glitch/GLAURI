# colabel/

Pocket-labeling pipeline: produces the "pocket" training labels (non-native-contact
positives) consumed by `model/main.py --use_colabel`. Order of use, top to bottom.

- **`colabel_summary_merged.ipynb`** — upstream, run once. mmseqs2 (95% seq-id)
  clustering of every chain in the dataset -> `all_cluster_summary.csv`, the
  `--cluster-summary` input every step1 run below needs.

- **`step1_batch_cluster_pocket_labels.py`** — STEP 1. Per cluster: collect contact
  residues for one interface class (protein/nucleic/ion/ligand), superpose cluster
  members onto a reference chain (USalign), spatially group contacts into "pockets"
  (DBSCAN on superposed coordinates). Also caches per-ckey receptor+ligand complex
  PDBs for step 2. No label scoring here — clustering only.

- **`step2_pocket_binary_labels.py`** — STEP 2. For each pocket from step 1: load the
  cached complex PDB, mark receptor residues with any heavy atom within 3.5 Å
  (default `--cutoff`) of any ligand atom, align back to the reference sequence ->
  one 0/1 `pocket_label` string per pocket. This is the actual label computation.

- **`step3_build_pocket_store.py`** — STEP 3. Compresses the 4 per-interface-type
  binary-label CSVs (protein/nn/ion/ligand, ~95MB/145k rows combined) into a sparse
  `pocket_store.pkl` (positive positions only) that `model/data_handler.py` loads
  directly at train time.

- **`colabel_util.py`** — shared helpers step 1 still depends on: writing single-chain
  PDBs (USalign input) and receptor+ligand complex PDBs (step 2's cache), plus `ckey`
  string parsing. Used to be a full jdock/rosetta scoring pipeline; trimmed down when
  step 1 dropped scoring (see its own module docstring for the history).

- **`run_smoketest.sh`** + **`_smoketest/`** — reproducible step1→step2→step3 run on a
  small real slice of the training H5 (not synthetic data). `./run_smoketest.sh` from
  this directory (`conda activate boltz` first) reproduces it end to end and asserts
  every pocket has ≥1 positive residue.

- **`all_colabel.pkl`, `colabel_mask_all.pkl`** — step 3's output plus a chain-mapping
  sidecar, symlinked from outside this repo (external training data, not shipped —
  see the root `README.md`'s external-data table).
