#!/bin/bash
#SBATCH --job-name=ablation_zerorsa
#SBATCH --partition=gpu-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err

set -euo pipefail

# Ablation experiment, baseline = 20260730_2split_weightedtrain_nopocket_v6_esm3_proj128_drop03_cat36rsa_gcpdrop02.
# RSA-feature ablation: the only difference is that the config enables zero_rsa_input (node-scalar column 36 RSA is zeroed);
# the RSA sidecar ../bio_251210_rsa.h5 must still exist; the data is read as usual and zeroed at forward time,
# keeping parameter count, initialization, and batch order bit-aligned with the baseline.

out_dir=./zerorsa
mkdir -p "${out_dir}"

MASK_SRC="${MASK_SRC:-../../Interface_prediction/model/new_mask_pairwise}"
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_train.pkl")" "${out_dir}/mask_dict_train.pkl"
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_valid.pkl")" "${out_dir}/mask_dict_valid.pkl"

# Only the training set is weighted; the validation set is an unweighted fixed 2048 subset.
singularity exec --nv --bind ~/my_extra_pkgs:/extra ../../Interface_prediction/model/protein_env_v2.sif \
  env PYTHONPATH="${PYTHONPATH:-}:/extra" PYTHONHASHSEED=42 \
  python3 ../model/main.py \
    --config config_zerorsa.py \
    --use_colabel \
    --out_dir "${out_dir}" \
    --train_m ../curation/final_split/cluster_30_80_benchclean_train_m.pkl \
    --valid_m ../curation/final_split/cluster_30_80_benchclean_val_m.pkl \
    --train_weight ../curation/final_split/cluster_30_80_benchclean_train_weight.pkl \
    --valid_eval_size 2048 \
    --valid_subset_seed 42
