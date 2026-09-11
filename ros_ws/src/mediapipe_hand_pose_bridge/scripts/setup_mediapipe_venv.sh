#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_WS_ROOT="$(cd -- "${PACKAGE_ROOT}/../.." && pwd)"
REPO_ROOT="$(cd -- "${ROS_WS_ROOT}/.." && pwd)"
VENV_ROOT="${IGRIS_MEDIAPIPE_VENV:-${REPO_ROOT}/.venv-mediapipe}"
PYTHON_BIN="${IGRIS_MEDIAPIPE_BASE_PYTHON:-/usr/bin/python3}"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements/mediapipe.in"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_ROOT}"
elif grep -q '^include-system-site-packages = false$' "${VENV_ROOT}/pyvenv.cfg"; then
    "${PYTHON_BIN}" -m venv --upgrade --system-site-packages "${VENV_ROOT}"
fi

PYTHONNOUSERSITE=1 "${VENV_ROOT}/bin/python" -m pip install --upgrade pip
PYTHONNOUSERSITE=1 "${VENV_ROOT}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"

# ROS Jazzy and the venv both use Python 3.12, so sourced ROS packages remain importable.
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${ROS_WS_ROOT}/install/setup.bash" ]]; then
  source "${ROS_WS_ROOT}/install/setup.bash"
fi
set -u

PYTHONNOUSERSITE=1 "${VENV_ROOT}/bin/python" - <<'PY'
import cv2
import mediapipe
import rclpy

print(f"MediaPipe environment ready: mediapipe={mediapipe.__version__} cv2={cv2.__version__}")
PY
