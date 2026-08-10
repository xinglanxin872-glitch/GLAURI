#!/bin/bash
#SBATCH --job-name=ablation_zeroseq
#SBATCH --partition=gpu-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err

set -euo pipefail

# Ablation experiment, baseline = 20260730_2split_weightedtrain_nopocket_v6_esm3_proj128_drop03_cat36rsa_gcpdrop02.
# Sequence-feature ablation: data split, training weights, validation subset, mask, and seed are identical to the baseline;
# the only difference is that the config enables zero_seq_input (the 1536-dim ESM3 segment of the node scalars is zeroed as a block;
# the 29-dim residue one-hot, 6-dim dihedrals, 1-dim RSA, and edge features are kept).

out_dir=./zeroseq
mkdir -p "${out_dir}"

MASK_SRC="${MASK_SRC:-../../Interface_prediction/model/new_mask_pairwise}"
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_train.pkl")" "${out_dir}/mask_dict_train.pkl"
ln -sfn "$(realpath "${MASK_SRC}/mask_dict_valid.pkl")" "${out_dir}/mask_dict_valid.pkl"

# Only the training set is weighted; the validation set is an unweighted fixed 2048 subset.
singularity exec --nv --bind ~/my_extra_pkgs:/extra ../../Interface_prediction/model/protein_env_v2.sif \
  env PYTHONPATH="${PYTHONPATH:-}:/extra" PYTHONHASHSEED=42 \
  python3 ../model/main.py \
    --config config_zeroseq.py \
    --use_colabel \
    --out_dir "${out_dir}" \
    --train_m ../curation/final_split/cluster_30_80_benchclean_train_m.pkl \
    --valid_m ../curation/final_split/cluster_30_80_benchclean_val_m.pkl \
    --train_weight ../curation/final_split/cluster_30_80_benchclean_train_weight.pkl \
    --valid_eval_size 2048 \
    --valid_subset_seed 42
