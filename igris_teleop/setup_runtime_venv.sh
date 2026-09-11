#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_VENV="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_BIN="${UV_BIN:-uv}"
REQUIREMENTS_IN="${REPO_ROOT}/requirements/runtime.in"
REQUIREMENTS_LOCK="${REPO_ROOT}/requirements/runtime.lock.txt"
SDK_WHEEL=""
for SDK_DIST_DIR in \
  "${REPO_ROOT}/third_party/igris_c_sdk_public/dist" \
  "${REPO_ROOT}/ros_ws/src/igris_c_ros_bridge/thirdparty/igris_c_sdk_public/dist"
do
  if [[ -d "${SDK_DIST_DIR}" ]]; then
    SDK_WHEEL="$(find "${SDK_DIST_DIR}" -maxdepth 1 -name 'igris_c_sdk-*.whl' | head -n 1)"
    if [[ -n "${SDK_WHEEL}" ]]; then
      break
    fi
  fi
done
RESET_ENV=0

usage() {
  cat <<'USAGE'
Usage: igris_teleop/setup_runtime_venv.sh [options]

Options:
  --reset          Remove the target venv before creating it again.
  --venv PATH      Target venv path. Default: <repo>/.venv
  --python PATH    Python executable. Default: $PYTHON_BIN or python3.12
  -h, --help       Show this help.

Environment:
  PYTHON_BIN       Python executable used when --python is not provided.
  UV_BIN           uv executable. Default: uv
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)
      RESET_ENV=1
      shift
      ;;
    --venv)
      RUNTIME_VENV="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_IN}" ]]; then
  echo "Requirements input file not found: ${REQUIREMENTS_IN}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
USE_UV=0
if command -v "${UV_BIN}" >/dev/null 2>&1 && [[ -f "${REQUIREMENTS_LOCK}" ]]; then
  USE_UV=1
fi

if [[ "${RESET_ENV}" == "1" && -e "${RUNTIME_VENV}" ]]; then
  if [[ -f "${RUNTIME_VENV}/pyvenv.cfg" ]]; then
    rm -rf "${RUNTIME_VENV}"
  else
    echo "Refusing to remove non-venv path: ${RUNTIME_VENV}" >&2
    exit 1
  fi
fi

if [[ ! -f "${RUNTIME_VENV}/pyvenv.cfg" ]]; then
  if [[ "${USE_UV}" == "1" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_BIN}" "${RUNTIME_VENV}"
  else
    "${PYTHON_BIN}" -m venv "${RUNTIME_VENV}"
  fi
fi

if [[ "${USE_UV}" == "1" ]]; then
  "${UV_BIN}" pip sync --python "${RUNTIME_VENV}/bin/python" "${REQUIREMENTS_LOCK}"
else
  "${RUNTIME_VENV}/bin/python" -m pip install -U pip setuptools wheel
  "${RUNTIME_VENV}/bin/python" -m pip install -r "${REQUIREMENTS_IN}"
fi

if [[ -n "${SDK_WHEEL}" ]]; then
  if [[ "${USE_UV}" == "1" ]]; then
    "${UV_BIN}" pip install --python "${RUNTIME_VENV}/bin/python" "${SDK_WHEEL}"
  else
    "${RUNTIME_VENV}/bin/python" -m pip install "${SDK_WHEEL}"
  fi
else
  echo "igris_c_sdk wheel not found under third_party/igris_c_sdk_public/dist; skipping" >&2
fi

if [[ -d "${REPO_ROOT}/ros_ws/src/DynamixelSDK/python" ]]; then
  if [[ "${USE_UV}" == "1" ]]; then
    "${UV_BIN}" pip install --python "${RUNTIME_VENV}/bin/python" "${REPO_ROOT}/ros_ws/src/DynamixelSDK/python"
  else
    "${RUNTIME_VENV}/bin/python" -m pip install "${REPO_ROOT}/ros_ws/src/DynamixelSDK/python"
  fi
fi

if [[ "${USE_UV}" == "1" ]]; then
  "${UV_BIN}" pip uninstall --python "${RUNTIME_VENV}/bin/python" argparse >/dev/null 2>&1 || true
else
  "${RUNTIME_VENV}/bin/python" -m pip uninstall -y argparse >/dev/null 2>&1 || true
fi

PYTHONPATH="${REPO_ROOT}" "${RUNTIME_VENV}/bin/python" - <<'PY'
import cv2
import logging_mp
import numpy
import pyrealsense2
import yaml

print("runtime venv imports ok")
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("pyrealsense2:", getattr(pyrealsense2, "__file__", "<module>"))
print("logging_mp:", getattr(logging_mp, "__file__", "<module>"))
print("yaml:", getattr(yaml, "__file__", "<module>"))
PY

if [[ "${USE_UV}" == "1" ]]; then
  "${UV_BIN}" pip check --python "${RUNTIME_VENV}/bin/python"
else
  "${RUNTIME_VENV}/bin/python" -m pip check
fi

echo "runtime venv ready: ${RUNTIME_VENV}"
echo "activate with: source ${RUNTIME_VENV}/bin/activate"
