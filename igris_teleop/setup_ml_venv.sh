#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_VENV="${REPO_ROOT}/.venv-ml"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_BIN="${UV_BIN:-uv}"
TORCH_BACKEND="${IGRIS_TORCH_BACKEND:-cu130}"
TORCH_INDEX_URL="${IGRIS_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_BACKEND}}"
REQUIREMENTS_IN="${REPO_ROOT}/requirements/ml.in"
REQUIREMENTS_LOCK="${REPO_ROOT}/requirements/ml.lock.txt"
GRADCAM_SRC="${REPO_ROOT}/vendor_patches/lerobot_modeling_act_gradcam.py"
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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${ML_VENV}/pyvenv.cfg" ]]; then
  if command -v "${UV_BIN}" >/dev/null 2>&1 && [[ -f "${REQUIREMENTS_LOCK}" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_BIN}" "${ML_VENV}"
  else
    "${PYTHON_BIN}" -m venv "${ML_VENV}"
  fi
fi

USE_UV=0
if command -v "${UV_BIN}" >/dev/null 2>&1 && [[ -f "${REQUIREMENTS_LOCK}" ]]; then
  USE_UV=1
fi

if [[ "${USE_UV}" == "1" ]]; then
  if "${UV_BIN}" pip sync --help 2>&1 | grep -q -- "--torch-backend"; then
    "${UV_BIN}" pip sync --python "${ML_VENV}/bin/python" --torch-backend "${TORCH_BACKEND}" "${REQUIREMENTS_LOCK}"
  else
    "${UV_BIN}" pip sync \
      --python "${ML_VENV}/bin/python" \
      --index "${TORCH_INDEX_URL}" \
      --index-strategy unsafe-best-match \
      "${REQUIREMENTS_LOCK}"
  fi
else
  if [[ ! -f "${REQUIREMENTS_IN}" ]]; then
    echo "Requirements input file not found: ${REQUIREMENTS_IN}" >&2
    exit 1
  fi

  "${ML_VENV}/bin/python" -m pip install -U pip setuptools wheel

  "${ML_VENV}/bin/python" -m pip install \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url "${TORCH_INDEX_URL}"

  "${ML_VENV}/bin/python" -m pip install \
    torchcodec==0.8.0 \
    --index-url "${TORCH_INDEX_URL}"

  "${ML_VENV}/bin/python" -m pip install -r "${REQUIREMENTS_IN}"
fi

if [[ -n "${SDK_WHEEL}" ]]; then
  if [[ "${USE_UV}" == "1" ]]; then
    "${UV_BIN}" pip install --python "${ML_VENV}/bin/python" "${SDK_WHEEL}"
  else
    "${ML_VENV}/bin/python" -m pip install "${SDK_WHEEL}"
  fi
fi

if [[ "${USE_UV}" == "1" ]]; then
  install_action_lipo=("${UV_BIN}" pip install --python "${ML_VENV}/bin/python" action_lipo)
else
  install_action_lipo=("${ML_VENV}/bin/python" -m pip install action_lipo)
fi

if ! "${install_action_lipo[@]}"; then
  echo "optional package install failed: action_lipo" >&2
fi

if [[ -f "${GRADCAM_SRC}" ]]; then
  SITE_PACKAGES="$("${ML_VENV}/bin/python" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"
  ACT_DIR="${SITE_PACKAGES}/lerobot/policies/act"
  if [[ -d "${ACT_DIR}" ]]; then
    cp "${GRADCAM_SRC}" "${ACT_DIR}/modeling_act_gradcam.py"
  else
    echo "LeRobot ACT package directory not found: ${ACT_DIR}" >&2
    exit 1
  fi
fi

PYTHONPATH="${REPO_ROOT}" "${ML_VENV}/bin/python" - <<'PY'
import cv2
import anytree
import nlopt
import lerobot
import logging_mp
import numpy as np
import pandas as pd
import pinocchio as pin
import pyarrow
import pytransform3d
import torch
import torchvision
import trimesh
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act_gradcam import ACTPolicy
from igris_teleop.workers.worker_hand import HandWorker

try:
    import torchcodec
    torchcodec_status = getattr(torchcodec, "__version__", "<unknown>")
except Exception as exc:
    torchcodec_status = f"unavailable: {exc}"

print("ml venv imports ok")
print("python torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("torchcodec:", torchcodec_status)
print("lerobot:", getattr(lerobot, "__version__", "<unknown>"))
print("pinocchio:", getattr(pin, "__version__", "<unknown>"))
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("pyarrow:", pyarrow.__version__)
print("cv2:", cv2.__version__)
print("nlopt:", getattr(nlopt, "__version__", "<unknown>"))
print("pytransform3d:", getattr(pytransform3d, "__version__", "<unknown>"))
print("anytree:", getattr(anytree, "__version__", "<unknown>"))
print("trimesh:", getattr(trimesh, "__version__", "<unknown>"))
print("logging_mp:", getattr(logging_mp, "__file__", "<module>"))
print("lerobot dataset class:", LeRobotDataset.__name__)
print("gradcam ACTPolicy:", ACTPolicy.__name__)
print("hand worker:", HandWorker.__name__)
PY

if [[ "${USE_UV}" == "1" ]]; then
  "${UV_BIN}" pip check --python "${ML_VENV}/bin/python"
else
  "${ML_VENV}/bin/python" -m pip check
fi

echo "ML venv ready: ${ML_VENV}"
