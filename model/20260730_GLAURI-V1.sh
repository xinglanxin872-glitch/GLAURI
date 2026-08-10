#!/bin/bash
#SBATCH --job-name=GLAURI_baseline_cat36rsa
#SBATCH --partition=gpu-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err

set -euo pipefail

# Baseline training script, adapted for the /home/lanxin/data/GLAURI layout.
# Output dir is ./GLAURI (i.e. /home/lanxin/data/GLAURI/model/GLAURI): it already holds the
# 20260731 run (model.pt + train.log), so with reload=True in config_runtime the run resumes
# from that checkpoint instead of starting over.
# Model = current config_baseline.py (GCP + ESM3 concat + per-residue RSA sidecar,
# h_input_dim 36+1536; RSA precomputed by curation/add_sasa_to_h5.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

out_dir=./GLAURI
mkdir -p "${out_dir}"

# --- pairwise masks: prefer a repo-local copy, fall back to the Interface_prediction checkout ---
if [ -f ./new_mask_pairwise/mask_dict_train.pkl ]; then
  MASK_SRC=./new_mask_pairwise
else
  MASK_SRC="${MASK_SRC:-../../Interface_prediction/model/new_mask_pairwise}"
fi
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_train.pkl")" "${out_dir}/mask_dict_train.pkl"
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_valid.pkl")" "${out_dir}/mask_dict_valid.pkl"

# --- dataset / RSA sidecar / seq dict / colabel: config resolves ../bio_251210{,_rsa}.h5,
# --- ../colabel/*.pkl and ./all_seq_dict.pkl relative to this directory; symlink from the
# --- Interface_prediction checkout when the GLAURI-local copy is absent ---
for f in bio_251210.h5 bio_251210_rsa.h5; do
  if [ ! -e "../${f}" ] && [ -e "../../Interface_prediction/${f}" ]; then
    ln -sfn "$(realpath "../../Interface_prediction/${f}")" "../${f}"
    echo "linked ../${f} -> ../../Interface_prediction/${f}"
  fi
done
for f in all_colabel.pkl colabel_mask_all.pkl; do
  if [ ! -e "../colabel/${f}" ] && [ -e "../../Interface_prediction/colabel/${f}" ]; then
    ln -sfn "$(realpath "../../Interface_prediction/colabel/${f}")" "../colabel/${f}"
    echo "linked ../colabel/${f} -> ../../Interface_prediction/colabel/${f}"
  fi
done
if [ ! -e ./all_seq_dict.pkl ] && [ -e ../../Interface_prediction/model/all_seq_dict.pkl ]; then
  ln -sfn "$(realpath ../../Interface_prediction/model/all_seq_dict.pkl)" ./all_seq_dict.pkl
  echo "linked ./all_seq_dict.pkl -> ../../Interface_prediction/model/all_seq_dict.pkl"
fi

# --- singularity image: repo-local first, Interface_prediction as fallback ---
if [ -f ./protein_env_v2.sif ]; then
  SIF=./protein_env_v2.sif
else
  SIF="${SIF:-../../Interface_prediction/model/protein_env_v2.sif}"
fi

# --- precheck ---
missing=0
for f in "${SIF}" ../bio_251210.h5 ../bio_251210_rsa.h5 ./all_seq_dict.pkl \
         ../colabel/all_colabel.pkl ../colabel/colabel_mask_all.pkl \
         ../curation/final_split/cluster_30_80_benchclean_train_m.pkl \
         ../curation/final_split/cluster_30_80_benchclean_val_m.pkl \
         ../curation/final_split/cluster_30_80_benchclean_train_weight.pkl; do
  if [ -e "$f" ]; then
    echo "  OK      $f"
  else
    echo "  MISSING $f" >&2
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "ERROR: missing files listed above; aborting." >&2
  exit 1
fi

# Only the training set is weighted; the validation set is an unweighted fixed 2048 subset.
singularity exec --nv --bind ~/my_extra_pkgs:/extra "${SIF}" \
  env PYTHONPATH="${PYTHONPATH:-}:/extra" PYTHONHASHSEED=42 \
  python3 main.py \
    --config config_baseline.py \
    --use_colabel \
    --out_dir "${out_dir}" \
    --train_m ../curation/final_split/cluster_30_80_benchclean_train_m.pkl \
    --valid_m ../curation/final_split/cluster_30_80_benchclean_val_m.pkl \
    --train_weight ../curation/final_split/cluster_30_80_benchclean_train_weight.pkl \
    --valid_eval_size 2048 \
    --valid_subset_seed 42
