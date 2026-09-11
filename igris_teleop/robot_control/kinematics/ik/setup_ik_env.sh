#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IK_VENV="${REPO_ROOT}/.venv-ik"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_BIN="${UV_BIN:-uv}"
REQUIREMENTS_IN="${REPO_ROOT}/requirements/ik.in"
REQUIREMENTS_LOCK="${REPO_ROOT}/requirements/ik.lock.txt"
RESET_ENV=0

usage() {
  cat <<'USAGE'
Usage: igris_teleop/robot_control/ik/setup_ik_env.sh [options]

Options:
  --reset          Remove the target venv before creating it again.
  --venv PATH      Target venv path. Default: <repo>/.venv-ik
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
      IK_VENV="$2"
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

export PYTHONNOUSERSITE=1
USE_UV=0
if command -v "${UV_BIN}" >/dev/null 2>&1 && [[ -f "${REQUIREMENTS_LOCK}" ]]; then
  USE_UV=1
fi

if [[ "${RESET_ENV}" == "1" && -e "${IK_VENV}" ]]; then
  if [[ -f "${IK_VENV}/pyvenv.cfg" ]]; then
    rm -rf "${IK_VENV}"
  else
    echo "Refusing to remove non-venv path: ${IK_VENV}" >&2
    exit 1
  fi
fi

if [[ ! -f "${IK_VENV}/pyvenv.cfg" ]]; then
  if [[ "${USE_UV}" == "1" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_BIN}" "${IK_VENV}"
  else
    "${PYTHON_BIN}" -m venv "${IK_VENV}"
  fi
fi

if [[ "${USE_UV}" == "1" ]]; then
  "${UV_BIN}" pip sync --python "${IK_VENV}/bin/python" "${REQUIREMENTS_LOCK}"
else
  REQUIREMENTS_FILE="${REQUIREMENTS_LOCK}"
  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    REQUIREMENTS_FILE="${REQUIREMENTS_IN}"
  fi
  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Requirements file not found: ${REQUIREMENTS_FILE}" >&2
    exit 1
  fi
  "${IK_VENV}/bin/python" -m pip install -U pip setuptools wheel
  "${IK_VENV}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"
fi

PYTHONPATH="${REPO_ROOT}" "${IK_VENV}/bin/python" - <<'PY'
import numpy as np

from igris_teleop.robot_control.kinematics.ik import prox_ik_pelvis_env as ikenv

print("prox ik dependencies:", ikenv.describe_ik_dependency_context())

solver = ikenv.IGRIS_C_UpperIK()
q0 = solver._default_init_data.copy() if solver._default_init_data is not None else solver.init_data.copy()
l0, r0, h0 = solver.get_ee_poses(q0)
q, tau = solver.solve_ik(
    l0.homogeneous,
    r0.homogeneous,
    h0.homogeneous,
    current_lr_arm_motor_q=q0,
)
assert q.shape == q0.shape, (q.shape, q0.shape)
assert tau.shape == q0.shape, (tau.shape, q0.shape)
assert np.all(np.isfinite(q)), q
assert np.all(np.isfinite(tau)), tau
print("prox ik smoke test ok")
PY

if [[ "${USE_UV}" == "1" ]]; then
  "${UV_BIN}" pip check --python "${IK_VENV}/bin/python"
else
  "${IK_VENV}/bin/python" -m pip check
fi

echo "IK venv ready: ${IK_VENV}"
echo "use with: export IGRIS_IK_PYTHON=${IK_VENV}/bin/python"
