#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements/runtime.in"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_NAME="${IGRIS_CONDA_ENV:-igris-teleop-runtime}"
CONDA_PREFIX_PATH=""
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RESET_ENV=0
SDK_WHEEL="$(find "${REPO_ROOT}/third_party/igris_c_sdk_public/dist" -maxdepth 1 -name 'igris_c_sdk-*.whl' | head -n 1)"

usage() {
  cat <<'USAGE'
Usage: igris_teleop/setup_runtime_conda.sh [options]

Options:
  --reset          Remove the target conda env before creating it again.
  --name NAME      Conda env name. Default: $IGRIS_CONDA_ENV or igris-teleop-runtime
  --prefix PATH    Conda env prefix path. Overrides --name.
  --python VER     Python version. Default: $PYTHON_VERSION or 3.12
  --conda PATH     Conda executable. Default: $CONDA_BIN or conda
  -h, --help       Show this help.

Environment:
  CONDA_BIN        Conda executable used when --conda is not provided.
  IGRIS_CONDA_ENV  Conda env name used when --name is not provided.
  PYTHON_VERSION   Python version used when --python is not provided.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)
      RESET_ENV=1
      shift
      ;;
    --name)
      CONDA_NAME="$2"
      CONDA_PREFIX_PATH=""
      shift 2
      ;;
    --prefix)
      CONDA_PREFIX_PATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --conda)
      CONDA_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Requirements file not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1

env_args=()
run_args=()
activate_hint=""
if [[ -n "${CONDA_PREFIX_PATH}" ]]; then
  env_args=(-p "${CONDA_PREFIX_PATH}")
  run_args=(-p "${CONDA_PREFIX_PATH}")
  activate_hint="conda activate ${CONDA_PREFIX_PATH}"
else
  env_args=(-n "${CONDA_NAME}")
  run_args=(-n "${CONDA_NAME}")
  activate_hint="conda activate ${CONDA_NAME}"
fi

env_exists() {
  "${CONDA_BIN}" run "${run_args[@]}" python -c 'import sys; print(sys.version)' >/dev/null 2>&1
}

if [[ "${RESET_ENV}" == "1" ]] && env_exists; then
  "${CONDA_BIN}" env remove -y "${env_args[@]}"
fi

if ! env_exists; then
  "${CONDA_BIN}" create -y "${env_args[@]}" "python=${PYTHON_VERSION}" pip
fi

"${CONDA_BIN}" run --no-capture-output "${run_args[@]}" python -m pip install -U pip setuptools wheel
"${CONDA_BIN}" run --no-capture-output "${run_args[@]}" python -m pip install -r "${REQUIREMENTS_FILE}"

if [[ -n "${SDK_WHEEL}" ]]; then
  "${CONDA_BIN}" run --no-capture-output "${run_args[@]}" python -m pip install "${SDK_WHEEL}"
else
  echo "igris_c_sdk wheel not found under third_party/igris_c_sdk_public/dist; skipping" >&2
fi

if [[ -d "${REPO_ROOT}/ros_ws/src/DynamixelSDK/python" ]]; then
  "${CONDA_BIN}" run --no-capture-output "${run_args[@]}" python -m pip install "${REPO_ROOT}/ros_ws/src/DynamixelSDK/python"
fi

"${CONDA_BIN}" run --no-capture-output "${run_args[@]}" python -m pip uninstall -y argparse >/dev/null 2>&1 || true

PYTHONPATH="${REPO_ROOT}" "${CONDA_BIN}" run --no-capture-output "${run_args[@]}" python - <<'PY'
import cv2
import logging_mp
import numpy
import pyrealsense2
import yaml

print("runtime conda imports ok")
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("pyrealsense2:", getattr(pyrealsense2, "__file__", "<module>"))
print("logging_mp:", getattr(logging_mp, "__file__", "<module>"))
print("yaml:", getattr(yaml, "__file__", "<module>"))
PY

"${CONDA_BIN}" run --no-capture-output "${run_args[@]}" python -m pip check

if [[ -n "${CONDA_PREFIX_PATH}" ]]; then
  echo "runtime conda env ready: ${CONDA_PREFIX_PATH}"
else
  echo "runtime conda env ready: ${CONDA_NAME}"
fi
echo "activate with: ${activate_hint}"
