#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GLAURI_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COUNT="$("${PYTHON_BIN:-python3}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${SCRIPT_DIR}/model_registry.json")"
CMD=(sbatch --array="0-$((COUNT-1))" "$@" "${SCRIPT_DIR}/train.sbatch")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}";printf '\n';exit 0;fi
exec "${CMD[@]}"
