#!/bin/bash
#SBATCH --job-name=ablation_zerostruct
#SBATCH --partition=gpu-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err

set -euo pipefail

# Ablation experiment, baseline = 20260730_2split_weightedtrain_nopocket_v6_esm3_proj128_drop03_cat36rsa_gcpdrop02.
# Structure-feature ablation (one-hot + dihedrals): the only difference is that the config enables zero_struct_input
# (the first 35 node-scalar columns + edge scalars batch.e + edge vectors batch.xi are zeroed; the RSA column and the ESM3 segment are kept,
# and the edge_index topology and f_ij local frames are kept).

out_dir=./zerostruct
mkdir -p "${out_dir}"

MASK_SRC="${MASK_SRC:-../../Interface_prediction/model/new_mask_pairwise}"
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_train.pkl")" "${out_dir}/mask_dict_train.pkl"
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_valid.pkl")" "${out_dir}/mask_dict_valid.pkl"

# Only the training set is weighted; the validation set is an unweighted fixed 2048 subset.
singularity exec --nv --bind ~/my_extra_pkgs:/extra ../../Interface_prediction/model/protein_env_v2.sif \
  env PYTHONPATH="${PYTHONPATH:-}:/extra" PYTHONHASHSEED=42 \
  python3 ../model/main.py \
    --config config_zerostruct.py \
    --use_colabel \
    --out_dir "${out_dir}" \
    --train_m ../curation/final_split/cluster_30_80_benchclean_train_m.pkl \
    --valid_m ../curation/final_split/cluster_30_80_benchclean_val_m.pkl \
    --train_weight ../curation/final_split/cluster_30_80_benchclean_train_weight.pkl \
    --valid_eval_size 2048 \
    --valid_subset_seed 42
