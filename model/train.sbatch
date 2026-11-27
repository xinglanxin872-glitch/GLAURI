#!/usr/bin/env bash
#SBATCH --job-name=GLAURI_paper
#SBATCH --partition=gpu-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=168:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail
REPO="${GLAURI_REPO_ROOT:?Submit through model/submit_all.sh so the repo path survives Slurm spooling}"
CONFIG_ID="$("${PYTHON_BIN:-python3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["id"])' "${REPO}/model/model_registry.json" "${SLURM_ARRAY_TASK_ID:-0}")"
exec bash "${REPO}/model/run_training.sh" "${CONFIG_ID}"
