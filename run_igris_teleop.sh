#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_VENV="${SCRIPT_DIR}/.venv"
ML_PYTHON="${SCRIPT_DIR}/.venv-ml/bin/python"
SIM_PYTHON="${SCRIPT_DIR}/.venv-sim/bin/python"
IK_PYTHON="${SCRIPT_DIR}/.venv-ik/bin/python"
VENV_ACTIVATE="${RUNTIME_VENV}/bin/activate"
LOG_DIR="${SCRIPT_DIR}/igris_artifacts/logs/desktop_launcher"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/launch_${TIMESTAMP}.log"
IGRIS_LAUNCH_OPTIONS_FILE="${IGRIS_LAUNCH_OPTIONS_FILE:-${SCRIPT_DIR}/igris_teleop_desktop_options.env}"
IGRIS_MAIN_ARGS=()

if [[ -f "${IGRIS_LAUNCH_OPTIONS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${IGRIS_LAUNCH_OPTIONS_FILE}"
fi

if ! declare -p IGRIS_MAIN_ARGS >/dev/null 2>&1; then
  IGRIS_MAIN_ARGS=()
elif [[ "$(declare -p IGRIS_MAIN_ARGS)" != declare\ -a* ]]; then
  IGRIS_MAIN_ARGS_TEXT="${IGRIS_MAIN_ARGS}"
  read -r -a IGRIS_MAIN_ARGS <<< "${IGRIS_MAIN_ARGS_TEXT}"
fi

ML_PYTHON="${IGRIS_ML_PYTHON:-${ML_PYTHON}}"
SIM_PYTHON="${IGRIS_SIM_PYTHON:-${SIM_PYTHON}}"
IK_PYTHON="${IGRIS_IK_PYTHON:-${IK_PYTHON}}"

export IGRIS_LAUNCH_OPTIONS_FILE
export IGRIS_WEB_HOST IGRIS_WEB_PORT IGRIS_WEB_OPEN_BROWSER
export IGRIS_KEEP_TERMINAL_OPEN IGRIS_ML_PYTHON IGRIS_SIM_PYTHON IGRIS_IK_PYTHON

keep_terminal_open() {
  local exit_status="$1"

  if [[ "${IGRIS_KEEP_TERMINAL_OPEN:-1}" == "0" ]]; then
    exit "${exit_status}"
  fi

  echo
  echo "[launcher] terminal kept open after exit_status=${exit_status}"
  echo "[launcher] project=${SCRIPT_DIR}"
  echo "[launcher] log_file=${LOG_FILE}"
  echo "[launcher] type 'exit' to close this terminal"
  exec /bin/bash -i
}

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv activate script not found: ${VENV_ACTIVATE}"
  echo "Run: ${SCRIPT_DIR}/igris_teleop/setup_runtime_venv.sh"
  keep_terminal_open 1
fi

if [[ ! -x "${SIM_PYTHON}" ]]; then
  echo "Simulator Python not found: ${SIM_PYTHON}"
  echo "Run: ${SCRIPT_DIR}/igris_teleop/sim/setup_sim_venv.sh"
  keep_terminal_open 1
fi

if [[ ! -x "${ML_PYTHON}" ]]; then
  echo "ML Python not found: ${ML_PYTHON}"
  echo "Run: ${SCRIPT_DIR}/igris_teleop/setup_ml_venv.sh"
  keep_terminal_open 1
fi

if [[ ! -x "${IK_PYTHON}" ]]; then
  echo "IK Python not found: ${IK_PYTHON}"
  echo "Run: ${SCRIPT_DIR}/igris_teleop/robot_control/ik/setup_ik_env.sh"
  keep_terminal_open 1
fi

cd "${SCRIPT_DIR}" || exit 1
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[launcher] $(date '+%F %T')"
echo "[launcher] project=${SCRIPT_DIR}"
echo "[launcher] log_file=${LOG_FILE}"
echo "[launcher] runtime_python=${RUNTIME_VENV}/bin/python"
echo "[launcher] ml_python=${ML_PYTHON}"
echo "[launcher] sim_python=${SIM_PYTHON}"
echo "[launcher] ik_python=${IK_PYTHON}"
echo "[launcher] options_file=${IGRIS_LAUNCH_OPTIONS_FILE}"
if ((${#IGRIS_MAIN_ARGS[@]})); then
  printf '[launcher] option_args='
  printf ' %q' "${IGRIS_MAIN_ARGS[@]}"
  printf '\n'
fi

LAUNCH_BASH_FLAGS="-ic"
if [[ "${IGRIS_KEEP_TERMINAL_OPEN:-1}" == "0" ]]; then
  LAUNCH_BASH_FLAGS="-c"
fi

/bin/bash "${LAUNCH_BASH_FLAGS}" '
cd "$1" || exit 1
if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
  source "/opt/ros/jazzy/setup.bash"
fi
if [[ -f "$1/ros_ws/install/setup.bash" ]]; then
  source "$1/ros_ws/install/setup.bash"
fi
source "$1/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export IGRIS_ML_PYTHON="${IGRIS_ML_PYTHON:-$1/.venv-ml/bin/python}"
export IGRIS_SIM_PYTHON="${IGRIS_SIM_PYTHON:-$1/.venv-sim/bin/python}"
export IGRIS_IK_PYTHON="${IGRIS_IK_PYTHON:-$1/.venv-ik/bin/python}"
if [[ -z "${CYCLONEDDS_URI:-}" && -f "$1/local_state/cyclonedds_igris_lan.xml" ]]; then
  export CYCLONEDDS_URI="file://$1/local_state/cyclonedds_igris_lan.xml"
fi
python -m igris_teleop.main "${@:2}"
' bash "${SCRIPT_DIR}" "${IGRIS_MAIN_ARGS[@]}" "$@"
status=$?

echo "[launcher] exit_status=${status}"

# Startup preflight failures already include an actionable message. Returning
# directly avoids opening a nested interactive shell for an instance/port clash.
if [[ "${status}" -eq 73 ]]; then
  exit "${status}"
fi

keep_terminal_open "${status}"
