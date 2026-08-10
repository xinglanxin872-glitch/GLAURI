#!/bin/bash
# Local (non-SLURM) smoketest for the step1 -> step2 -> step3 colabel pocket pipeline.
# Slices one real 5-chain / 13-contact cluster (EO/3EO0/B:0) out of the full training H5
# into a 3MB test h5 under _smoketest/, then runs all three pipeline steps on it and
# prints a pass/fail summary based on actual pocket contents, not just exit codes.
#
# Only contact-class 0 (protein) is exercised -- classes 1/2/3 (nn/ion/ligand) share the
# same step1 code path (--contact-class just selects a column) but haven't been verified;
# step3 is fed empty placeholder CSVs for them so it can still run its 4-file merge.
#
# Usage:
#   conda activate boltz
#   ./run_smoketest.sh                 # reuse cached smoke_bio.h5 if present
#   FORCE_EXTRACT=1 ./run_smoketest.sh # re-extract from the source h5 first

set -euo pipefail

# ===================== edit paths here =====================
COLABEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${COLABEL_DIR}/_smoketest"
RUN_OUT="${TEST_DIR}/run_out"

# Source data this smoketest is sliced from -- not part of the GLAURI repo, points at the
# original working tree. Override via env if running elsewhere.
SMOKE_SRC_H5="${SMOKE_SRC_H5:-/home/lanxin/data/Interface_prediction/bio_251210.h5}"
SMOKE_SRC_CLUSTER_SUMMARY="${SMOKE_SRC_CLUSTER_SUMMARY:-/home/lanxin/data/Interface_prediction/colabel/all_cluster_summary.csv}"
SMOKE_SRC_ALL_SEQ_DICT="${SMOKE_SRC_ALL_SEQ_DICT:-/home/lanxin/data/Interface_prediction/cluster/all_seq_dict.pkl}"
export SMOKE_SRC_H5 SMOKE_SRC_CLUSTER_SUMMARY SMOKE_SRC_ALL_SEQ_DICT

USALIGN_BIN="${USALIGN_BIN:-/home/lanxin/data/USalign/USalign}"
PDB_DIR="${PDB_DIR:-/home/lanxin/data/Interface_prediction/bio_assemblies_search_1}"

PYTHON_BIN="${PYTHON_BIN:-/home/lanxin/miniconda3/envs/boltz/bin/python}"
# =============================================================

mkdir -p "${TEST_DIR}"

echo "== step 0: slice test subset out of the source h5 (skips if already present) =="
extract_args=(--out-dir "${TEST_DIR}")
[[ "${FORCE_EXTRACT:-0}" == "1" ]] && extract_args+=(--force)
"${PYTHON_BIN}" "${TEST_DIR}/extract_smoke_subset.py" "${extract_args[@]}"

rm -rf "${RUN_OUT}"

echo
echo "== step 1: pocket clustering + complex-PDB cache =="
"${PYTHON_BIN}" "${COLABEL_DIR}/step1_batch_cluster_pocket_labels.py" \
  --h5 "${TEST_DIR}/smoke_bio.h5" \
  --cluster-summary "${TEST_DIR}/smoke_cluster_summary.csv" \
  --cluster-name "EO/3EO0/B:0" \
  --out-dir "${RUN_OUT}" \
  --usalign-bin "${USALIGN_BIN}" \
  --pdb-dir "${PDB_DIR}" \
  --pdb-layout bio \
  --contact-class 0 \
  --cache-dir "${RUN_OUT}/cache" \
  --all-seq-dict "${TEST_DIR}/smoke_all_seq_dict.pkl" \
  --align-workers 2

echo
echo "== step 2: per-pocket binary contact labels (--complex-pdb-dir mode) =="
"${PYTHON_BIN}" "${COLABEL_DIR}/step2_pocket_binary_labels.py" \
  --details-dir "${RUN_OUT}/details" \
  --complex-pdb-dir "${RUN_OUT}/cache/complex_pdb" \
  --out-csv "${RUN_OUT}/smoke_pocket_binary_labels.csv" \
  --cutoff 3.5 \
  --workers 1

echo
echo "== step 3: compress into training-ready pocket_store.pkl =="
# step3 always reads all 4 interface-type CSVs; only "protein" has real data this round,
# the other 3 are empty placeholders (header row only) so the 4-file merge still runs.
mkdir -p "${RUN_OUT}/pocket_binary_labels_final"
cp "${RUN_OUT}/smoke_pocket_binary_labels.csv" "${RUN_OUT}/pocket_binary_labels_final/protein_pocket_binary_labels.csv"
header_only="${RUN_OUT}/pocket_binary_labels_final/.header_only.csv"
head -1 "${RUN_OUT}/smoke_pocket_binary_labels.csv" > "${header_only}"
for t in nn ion ligand; do
  cp "${header_only}" "${RUN_OUT}/pocket_binary_labels_final/${t}_pocket_binary_labels.csv"
done
rm -f "${header_only}"

"${PYTHON_BIN}" "${COLABEL_DIR}/step3_build_pocket_store.py" \
  --label_dir "${RUN_OUT}/pocket_binary_labels_final" \
  --out "${RUN_OUT}/smoke_pocket_store.pkl" \
  --colabel ""

echo
echo "== verify: pocket_label must have real (non-degenerate) positive residues =="
"${PYTHON_BIN}" - "${RUN_OUT}/smoke_pocket_binary_labels.csv" << 'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
ok = True
if df.empty:
    print("[FAIL] no pocket rows produced")
    ok = False
for _, row in df.iterrows():
    n_ones = row["pocket_label"].count("1")
    status = "ok" if n_ones > 0 else "FAIL (all-zero pocket)"
    print(f"  cluster={row['cluster_id']} pocket={row['pocket_index']} "
          f"len={len(row['pocket_label'])} positives={n_ones}  [{status}]")
    ok &= n_ones > 0
print("[PASS]" if ok else "[FAIL]", "smoketest", "green" if ok else "found a degenerate pocket")
sys.exit(0 if ok else 1)
PY
