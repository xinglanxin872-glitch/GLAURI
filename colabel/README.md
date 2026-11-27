# Contact-label co-label pipeline

For binding-mode definitions, the ≥4-participating-residue filter and chain-name
→ structure-set → interface-mode search, see [Binding modes](BINDING_MODES.md).
To install and launch the English browser with monomer cartoons and medoid
representative chains, follow [the browser quick start](BINDING_MODES.md#browser-search).

This directory contains the sequence preparation and hierarchical pocket-clustering
pipeline. It replaces the old USalign/coordinate grouping, docking, and geometric
contact-recalculation scripts. Input HDF5 files are always opened read-only.

## Algorithm and sources

1. Select protein receptor entries from H5 metadata; export full chain sequences.
   The default uses `all_seq_dict.pkl`. Sequences are not cropped
   to their contacting residues. `--sequence-source structure` is available when
   an explicitly different, structure-derived sequence universe is intended.
2. Cluster full sequences with MMseqs2: identity 0.95, coverage 0.9, cov-mode 0.
   Write the original five-column cluster-summary format and membership lists.
3. For each sequence cluster, select its longest usable full reference sequence;
   ties retain membership order. Read each complex's native H5 contact labels,
   globally align the receptor structure sequence to the reference, and project
   the contact labels to a reference-length binary vector. The historical
   Biopython first-optimal-alignment convention is preserved by `leftmost_align.py`.
4. Cluster those binary contact vectors with Jaccard/Tanimoto distance and
   average-linkage hierarchical clustering. Drop contacts projecting to all-zero.
   Pocket IDs are zero-based and ordered largest-pocket-first.
5. Optionally build a four-column training-label store from the pocket table.
   Column order: protein, nucleic acid, ion, ligand. Support counts distinct
   receptor structure keys, not contact rows. The denominator is the set of
   receptors with retained contacts in that cluster/interface class.

Native H5 contacts are projected and grouped directly; the H5-reading helpers
are in `contact_io.py` and alignment is implemented in `leftmost_align.py`.

Upstream structure cleaning, atomic contact extraction, and H5 construction remain
in `../curation/build_dataset.py`; they precede this directory's H5 input.
Native H5 contact records are reused as stored. Their definition is not replaced
by a new 3.5-A cutoff here. No H5 `cluster` fields are rewritten.

## Pocket clustering and downstream labels

`distance = 1 - |A intersection B| / |A union B|`.

Here A and B are sets of contacting reference residues from two contact records.
The clustering uses **distance 0.8**, equivalent to similarity 0.2. Use either
`--threshold 0.8` or `--similarity-threshold 0.2` to specify that same setting.
Passing `--threshold 0.2` selects distance 0.2 and is a different, stricter cut.
With average linkage, the cut acts on average inter-cluster distances rather
than imposing a minimum similarity on every pair of records in a pocket.

These settings define the binding modes. For downstream annotation preparation,
see [training targets](#training-targets). The browser has its
own [residue/support filters](BINDING_MODES.md#residue-and-support-filters).

## Training targets

Targets combine native contacts with annotations projected within each structure
set. Protein, nucleic-acid and ion annotations are binary unions. For a ligand
pocket with support `s` (distinct receptor keys supporting that pocket divided
by those with retained ligand contacts in the set), its contact sites receive
weight `1` when `s >= 0.5`, and `0.25 + 0.75 * s / 0.5` otherwise; non-contact
sites remain zero. Take the maximum across pockets, then the maximum with native
contacts at each residue. H5 validation uses the same targets as training;
the w/o co-label ablation uses native contacts only. The implementation is in
[label_profiles.py](../model/label_profiles.py) and
[data_handler.py](../model/data_handler.py), with parameters in
[config_fixed.json](../model/config_fixed.json).

## Run

Dependencies: `python -m pip install -r colabel/requirements.txt`. Install MMseqs2 separately and
put `mmseqs` on PATH, or pass its executable using `--mmseqs`.
The examples assume execution from the repository root. Set data/output paths
for the local machine; large datasets and generated outputs are not bundled here.

```bash
DATA=/path/to/downloaded_data
OUT=/path/to/new_colabel_outputs

python colabel/export_sequences.py \
  --h5 "$DATA/bio_251210.h5" \
  --all-seq-dict "$DATA/all_seq_dict.pkl" \
  --out-fasta "$OUT/all.fasta"

python colabel/cluster_protein_sequences.py \
  --fasta "$OUT/all.fasta" --out-prefix "$OUT/all" --threads 8

for cls in 0 1 2 3; do
  python colabel/cluster_contact_labels.py \
    --h5 "$DATA/bio_251210.h5" \
    --all-seq-dict "$DATA/all_seq_dict.pkl" \
    --cluster-csv "$OUT/all_cluster_summary.csv" \
    --ref-policy longest --threshold 0.8 \
    --contact-class "$cls" --out-dir "$OUT/interactions"
done

# Build binary non-ligand reference labels; model.prepare fills the ligand column.
python colabel/build_cluster_label_store.py \
  --in-dir "$OUT/interactions" \
  --out "$OUT/reference_labels.pkl"

# Set base_label_store and ligand_interactions in paths.json to the two outputs.
python -m model.prepare --paths /path/to/paths.json --config GLAURI
```

To reproduce the existing MMseqs-longest interaction tables, pass the existing
cluster-summary CSV directly to the contact
step. Rerunning MMseqs with another version may change membership or ordering.
The sequence-export size filter defaults to 8192 atoms and 48 residues; the
contact step preserves the contact-label rule of at least 48 residues, without an
additional atom-count filter on the supplied cluster membership.

For parallel execution, use `--shard 0/8` through `--shard 7/8` for each class,
then `python colabel/merge_pocket_interactions.py --dir "$OUT/interactions"`.
Sharding happens by whole sequence cluster; input shards are preserved.
The merger rejects missing shards and clusters present in multiple shards.
