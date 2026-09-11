#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_VENV="${REPO_ROOT}/.venv-sim"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
UV_BIN="${UV_BIN:-uv}"
REQUIREMENTS_IN="${REPO_ROOT}/requirements/sim.in"
REQUIREMENTS_LOCK="${REPO_ROOT}/requirements/sim.lock.txt"
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

if [[ -z "${SDK_WHEEL}" ]]; then
  echo "igris_c_sdk wheel not found under third_party/igris_c_sdk_public/dist" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

USE_UV=0
if command -v "${UV_BIN}" >/dev/null 2>&1 && [[ -f "${REQUIREMENTS_LOCK}" ]]; then
  USE_UV=1
fi

if [[ "${USE_UV}" == "1" ]]; then
  if [[ ! -f "${SIM_VENV}/pyvenv.cfg" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_BIN}" "${SIM_VENV}"
  fi
  "${UV_BIN}" pip sync --python "${SIM_VENV}/bin/python" "${REQUIREMENTS_LOCK}"
  "${UV_BIN}" pip install --python "${SIM_VENV}/bin/python" "${SDK_WHEEL}"
  "${UV_BIN}" pip check --python "${SIM_VENV}/bin/python"
else
  if [[ ! -f "${REQUIREMENTS_IN}" ]]; then
    echo "Requirements input file not found: ${REQUIREMENTS_IN}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv "${SIM_VENV}"
  "${SIM_VENV}/bin/python" -m pip install -U pip setuptools wheel
  "${SIM_VENV}/bin/python" -m pip install -r "${REQUIREMENTS_IN}"
  "${SIM_VENV}/bin/python" -m pip install "${SDK_WHEEL}"
  "${SIM_VENV}/bin/python" -m pip check
fi

echo "simulation venv ready: ${SIM_VENV}"
