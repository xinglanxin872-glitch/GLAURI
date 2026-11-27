# Binding modes and structure-set lookup

For installation, prebuilt-index data requirements, local/SSH launch commands
and troubleshooting, start with [Browser search](#browser-search).

A **structure set** here is the original high-similarity MMseqs2 receptor-sequence
cluster: identity 0.95, coverage 0.9, cov-mode 0. Its identifier is `cluster_names`
in the membership CSV and `cluster_name` in the contact tables. It is distinct
from the coarse 30%/80% components used for dataset splitting. A short PDB-chain
name may match multiple structure keys/assemblies; the lookup retains all matches.

Within each structure set, contact records are processed separately for protein,
nucleic-acid (`nn`), ion and ligand interfaces. Native contact vectors are aligned
to the longest usable reference sequence. Average-linkage clustering of Jaccard
distances, cut at distance 0.8, defines the existing `pocket_id` groups. Each group
is called a **binding mode** here. These are algorithmic contact patterns, not
automatically experimentally validated mechanisms. They are unrelated to the
number of partner classes predicted for a human protein (Fig5A multiplicity).

## Residue and support filters
Binding modes within each interface type are identified through fully
unsupervised clustering. Support and the number of contacting residues in each
binding mode can, to some extent, serve as indicators of confidence.
These are descriptive indicators, not calibrated confidence probabilities.
Interface types themselves come from the native contact annotations.

The lookup counts participating residues as the **union of positive reference
positions across all contact records in the mode**:

`participating_residues(mode) = {i: any ref_label_record[i] == 1}`.

The default is `len(participating_residues) >= 4`, i.e. **strictly more than 3**.
The browser shows only modes passing this adjustable filter; it does not show
all 180,600 indexed modes unconditionally. By default it adds no support cutoff
beyond support >0. The support >0.1 criterion belongs to the original statistic
and is an optional browser setting, rather than the browser default.
This is a mode-level filter after projection and aggregation. It does not require
each individual contact record to contain four residues, and does not recluster
or renumber the original modes. Indels/unmapped residues do not add reference
positions. Displayed residue names/positions use the reference sequence, **1-based**;
they are not the queried PDB chain's residue numbers. Mode ID = `pocket_id + 1`;
gaps in displayed IDs are intentional after filtering.

`support = distinct receptor keys in this mode / distinct receptor keys with
retained contacts in this structure set and interface type`.

The denominator is neither contact-row count nor the full structure-set size.
One receptor may support multiple modes. The new lookup defaults to support >0
(all existing modes); `--min-support 0.1` additionally applies the old statistic's
**strict support >0.1** rule. Neither the size filter nor support filter modifies
training labels or benchmark truth. Downstream annotation preparation is
documented under [training targets](README.md#training-targets).

## Representative chain for each mode

The English search page lists a **Representative chain (medoid)** for every
retained mode. It is the receptor chain of an actual contact record minimizing
the mean Jaccard distance to **all original contact records within that mode**.
Identical vectors retain their contact-record multiplicity; this is not an
unweighted medoid of unique vectors or a selection over distinct receptor chains.
The zero self-distance is included in the reported mean. Numerical ties within
1e-12 mean distance retain the first source contact, deterministically.

The display includes both `PDB_chain` and the complete structure key, preserving
assembly/model suffixes. Expanded details and query JSON also give the selected
contact key, partner key and mean distance. The representative must belong to
that mode and can differ between modes. It is distinct from the structure set's
longest reference chain, which defines the label coordinate system.

Index schema `glauri_binding_modes_v2` stores these medoids. Rebuild an older v1
index from the same frozen inputs before using this page; the source records,
mode IDs, residue/support filters and coordinate mappings are unchanged.

## Existing statistic and structure display

The existing binding-mode statistic counts modes with support >0.1, using sets
with at least one qualifying mode as the denominator. Before the new residue-size
filter, protein has 20,254 multi-mode sets out of 40,848 (49.6%); ligand has 11,494
out of 23,423 (49.1%). These are the existing statistic's counts, not counts after
the new ≥4-residue filter. The old figure is not overwritten by index building.

The original 6CBZ/2PYI structure display has an additional, different rule:
only modes with at least two contact records can color residues; a reference
position must have coverage ≥0.5 across the mode's contact records. Overlapping
modes compete by highest coverage, then smaller pocket ID on ties. The reference
labels are aligned back to the target chain. PDB B-factor 0 is background and
`pocket_id + 1` is the mode ID; this is a categorical encoding, not a probability.
The new lookup provides both union positions and coverage ≥0.5 positions. Its
union-size filter is independent of those rendering rules. The interactive
monomer cartoon display can show either position set. It does not apply the
old two-contact rendering threshold or resolve overlaps between modes: only
the selected mode is drawn.

## Build the index

Python's standard library is sufficient; no H5, GPU or model checkpoint is read.
Use the original membership and all four matching frozen interaction tables:

```bash
python colabel/binding_modes.py build \
  --clusters /path/to/all_cluster_summary.csv \
  --tables /path/to/pocket_interactions_mmseq_longest \
  --output colabel/outputs/binding_modes/binding_modes.sqlite
```

The input directory must contain `protein_pocket_interactions.csv.gz`,
`nn_pocket_interactions.csv.gz`, `ion_pocket_interactions.csv.gz` and
`ligand_pocket_interactions.csv.gz`. The generated database preserves all modes;
filters are applied at query time. Its `.manifest.json` records input hashes,
unfiltered and filtered counts, definitions and the database hash. Builds refuse
an existing output. Input membership, pocket IDs, contacts and labels are read-only.

Generated `colabel/outputs/` is ignored by Git. To offer lookup to users, distribute
the index and manifest as explicit data assets, or provide all five frozen input
files with the matching source hashes. The current training metadata supplement
contains only the ligand interaction table, not all four tables or the original
high-similarity membership CSV; that supplement alone cannot rebuild this index.

## Query from the command line

The local frozen inputs contain 50,792 structure sets and 355,115 member
structure keys. The ≥4-residue filter retains 125,302 of 180,600 original modes:

| Interface | Original modes | ≥4 union residues | Also support >0.1 |
| --- | ---: | ---: | ---: |
| Protein | 85,787 | 73,755 | 71,304 |
| Nucleic acid | 4,680 | 3,950 | 3,912 |
| Ion | 34,169 | 11,018 | 10,881 |
| Ligand | 55,964 | 36,579 | 33,837 |

These are mode counts, not structure-set counts. Inputs are identified by the
generated manifest hashes; other source versions may give different totals.

From the repository root:

```bash
python colabel/binding_modes.py query 6CBZ_B \
  --db colabel/outputs/binding_modes/binding_modes.sqlite

python colabel/binding_modes.py query 2PYI_A \
  --db colabel/outputs/binding_modes/binding_modes.sqlite \
  --min-residues 4 --min-support 0.1 --json --output /path/to/new_query.json
```

Accepted inputs are `PDB_chain` (PDB case is normalized; chain case is preserved)
or a complete structure key such as `CB/6CBZ/B:0`. If no exact chain-case match
exists, a marked case-insensitive fallback returns all matching keys. The result
contains the full structure-set membership and, for each interface, its reference
sequence/key, original/retained mode counts and per-mode positions, support,
supporting receptor keys, contact count and up to three source-contact examples.

`query_receptor_keys` identifies which of the matched structures actually has a
retained contact record in that mode. A mode belonging to the same structure set
does not imply that the queried chain itself has experimentally observed contacts
for it. An absent interface is reported separately from modes excluded by filters.

Export all qualifying modes, one row per structure set/interface/pocket:

```bash
python colabel/binding_modes.py export \
  --db colabel/outputs/binding_modes/binding_modes.sqlite \
  --min-residues 4 --output /path/to/new_modes_ge4.csv
```

## Browser search

Run the commands below from the repository root with Python 3.10. Obtain the
separate `GLAURI_binding_modes_v2.zip` index archive and extract it into a fresh
`colabel/outputs/` directory:

```bash
python -m zipfile -e /path/to/GLAURI_binding_modes_v2.zip colabel/outputs
```

The resulting database is `colabel/outputs/binding_modes/binding_modes.sqlite`.
Check its SHA256 against the adjacent manifest. The archive is prepared locally;
its public download link has not yet been added. The
[processed dataset](https://huggingface.co/datasets/sindax/GLAURI_processed/tree/main)
currently supplies the H5 restoration package separately. The native H5 is about
132.6 GB; it is optional for reference maps and required for monomer coordinates.
The RSA sidecar, checkpoints, ESM weights and GPU are not required for this browser.

For index-only reference contact maps, tables and JSON:

```bash
python colabel/binding_modes.py serve \
  --db colabel/outputs/binding_modes/binding_modes.sqlite --index-only --port 8765
```

For those resources plus the monomer cartoon, place the restored native file at
`data/bio_251210.h5` relative to the repository root, or pass its actual location.
Install the small coordinate-viewing dependency set first:

```bash
python -m pip install -r requirements/binding-browser.txt
python colabel/binding_modes.py serve \
  --db colabel/outputs/binding_modes/binding_modes.sqlite \
  --h5 data/bio_251210.h5 --port 8765
```

Keep the terminal running and open `http://127.0.0.1:8765`; press Ctrl+C to stop.
For a remote SSH session, forward port 8765 first (for example via VS Code's
Ports panel), or run `ssh -N -L 8765:127.0.0.1:8765 username@server` on your own
computer. The server binds only to localhost. If the port is occupied, select
another with `--port` and use the corresponding browser/tunnel address.
Search shows structure-set membership, four interface types and expandable mode
details; the full result can be downloaded as JSON. Queries only read the index.

The page automatically detects coordinate availability; it supports both cases.
Without H5, sequence contact-map strips remain fully functional and cartoon
controls are hidden. `--index-only` forces this mode even when H5 exists. Restore
the **native** H5, rather than just the RSA sidecar, and restart the server to
enable coordinates. The file may live anywhere if provided with `--h5`.

Search for `6CBZ_B` or `2PYI_A`, then select an interface/mode or click
**Highlight pocket**. The reference strips and medoid tables work without H5.
Opening the HTML directly, GitHub preview or a static-file server does not run
the required Python API. If search works but the cartoon does not, check the H5
path and coordinate dependencies; try another set member if a chain has no H5
coordinates. Refresh with Ctrl+Shift+R after updating browser assets.

## Reference-sequence contact maps without H5

Each strip spans the entire reference FASTA length `L`, with reference positions
from **1 to L**, including positions missing from the queried monomer's coordinates.
It uses the indexed reference label directly, without aligning/cropping to a PDB.
The reference can differ from both the structure-set name and a mode's medoid.
The exact FASTA and its length are shown above/below the strips.

There is one strip per mode passing the browser's current residue/support filters.
Contact positions have a binary color (blue; orange for the selected mode), while
background is gray. The selected **Displayed sites** rule switches between the
reference contact union and sites with contact-record coverage ≥0.5. This changes
the visible sites, not the mode filter, which still counts union residues.
Color intensity is not a confidence probability or per-site contact frequency.
Click a strip to select its mode; hover to inspect a reference residue and position.
This view needs only the SQLite index and works without H5, NumPy or WebGL.

## Monomer cartoon and pocket highlighting

Structure display additionally requires `numpy`, `h5py` and Biopython from the
project environment. Search/export/index building still use only the standard
library. The local server loads one monomer on demand from the native H5 opened
read-only. It does not run prediction, fetch PDB coordinates, or alter contacts.
The browser bundles 3Dmol.js 3.0.0 locally, so it needs no CDN connection; its
license and source identity are in `third_party_licenses/3Dmol.txt` and
`colabel/static/manifest.json`.

After searching, a single viewer shows the queried monomer as a pure cartoon:
helical ribbons, beta-strand arrows and loops, without sticks or atomic spheres.
The selected pocket colors the cartoon orange without changing its geometry.
Use the interface/mode selectors or a table row's “Highlight pocket” button. The monomer
selector includes the matched structure keys, reference chain and all other set
members. A member without H5 coordinates produces an explicit error; other
members and the reference chain remain selectable.

The same `leftmost_align` function used in label construction globally aligns
the structure sequence to the interface reference sequence. Mapped reference
positions select actual monomer residues. Cα atoms are selected by the H5 atom
name table, not by a guessed numerical index. Missing reference positions are
reported and not assigned invented coordinates. The panel shows the mapping
count, unmapped count and sequence identity. Cartoon rendering uses the real
backbone coordinates; missing residues are not filled in.

H5 contains coordinates and structure-sequence residue order, but not original
PDB residue numbers/insertion codes. The viewer's chain A and sequential residue
IDs are display identifiers, **not original PDB chain/residue identifiers**. Hover
labels show both structure-sequence and reference-sequence positions. Original
H5 coordinates are retained, with no alignment transform or coordinate noise.
Switching to the reference structure may recover sites absent in the queried
fragment, but sites missing in that reference's coordinates remain absent.

The viewer supports rotation and zoom; “Focus pocket” zooms to the selected residues
and “Reset view” returns to the full monomer. The ≥4-residue filter always refers
to the reference union, even when the viewer selects coverage ≥0.5 positions.

“Export PyMOL cartoon” downloads a self-contained `.pml` containing the selected
monomer's coordinates and the currently mapped pocket selection. Open it with
PyMOL to display a cartoon-only monomer with orange pocket residues; original
PDB residue numbers are not reconstructed. The script runs PyMOL's `dss` to
assign secondary structure and enables fancy helices/sheets and smooth loops.
The browser itself does not require a PyMOL installation or plugin.

The browser scene is rendered with 3Dmol.js/WebGL: a PDB string reconstructed
from native H5 coordinates is loaded locally, secondary structure is inferred
from coordinates by the viewer, and cartoon geometry is colored by the mapped
residue selection. This is an interactive browser rendering. The exported PML
instead embeds the same coordinates, runs PyMOL commands, and highlights the same
mapped residue IDs; the two programs can assign slightly different secondary
structure boundaries. Neither path changes the H5 coordinates or contact labels.
