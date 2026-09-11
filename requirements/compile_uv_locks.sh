#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_BACKEND="${IGRIS_TORCH_BACKEND:-cu130}"
TORCH_INDEX_URL="${IGRIS_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_BACKEND}}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv executable not found: ${UV_BIN}" >&2
  exit 1
fi

compile_lock() {
  local name="$1"
  shift
  "${UV_BIN}" pip compile \
    --python "${PYTHON_BIN}" \
    --custom-compile-command "requirements/compile_uv_locks.sh" \
    "$@" \
    "${REPO_ROOT}/requirements/${name}.in" \
    -o "${REPO_ROOT}/requirements/${name}.lock.txt"
}

compile_lock runtime
compile_lock sim
compile_lock ik
if "${UV_BIN}" pip compile --help 2>&1 | grep -q -- "--torch-backend"; then
  compile_lock ml --torch-backend "${TORCH_BACKEND}"
else
  compile_lock ml --index "${TORCH_INDEX_URL}" --index-strategy unsafe-best-match
fi

echo "uv lock files updated under ${REPO_ROOT}/requirements"
