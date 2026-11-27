#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-GLAURI}"
shift "$(( $# > 0 ? 1 : 0 ))"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
PACKAGE_ROOT="$(cd "${REPO}/.." && pwd)"
if [[ -d "${PACKAGE_ROOT}/third_party/esm3lib" ]]; then
  export PYTHONPATH="${REPO}:${PACKAGE_ROOT}/third_party:${PYTHONPATH:-}"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "${REPO}"
CMD=("${PYTHON_BIN}" -m model.train --config "${CONFIG}")
if [[ -n "${GLAURI_PATHS:-}" ]]; then CMD+=(--paths "${GLAURI_PATHS}"); fi
CMD+=("$@")
if [[ -n "${CONTAINER:-}" ]]; then
  PREFIX=("${CONTAINER_BIN:-singularity}" exec --nv --bind "${REPO}:${REPO}")
  PREFIX+=(--bind "${PACKAGE_ROOT}:${PACKAGE_ROOT}")
  if [[ -n "${DATA_ROOT:-}" ]]; then PREFIX+=(--bind "${DATA_ROOT}:${DATA_ROOT}"); fi
  if [[ -n "${HF_HOME:-}" ]]; then PREFIX+=(--bind "${HF_HOME}:${HF_HOME}"); fi
  if [[ -n "${EXTRA_PKGS_DIR:-}" ]]; then
    PREFIX+=(--bind "${EXTRA_PKGS_DIR}:/extra")
    export PYTHONPATH="${PYTHONPATH}:/extra"
  fi
  CMD=("${PREFIX[@]}" "${CONTAINER}" env "PYTHONPATH=${PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${CMD[@]}")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}";printf '\n';exit 0;fi
exec "${CMD[@]}"
