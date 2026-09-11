#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_BIN="${UV_BIN:-uv}"
TORCH_BACKEND="${IGRIS_TORCH_BACKEND:-cu130}"

RESET_ENVS=0
REFRESH_LOCKS=0
ENSURE_UV=1
INSTALL_RUNTIME=1
INSTALL_SIM=1
INSTALL_IK=1
INSTALL_ML=1
INSTALL_DESKTOP=1

usage() {
  cat <<'USAGE'
Usage: ./install_igris_teleop.sh [options]

Installs uv if needed, prepares the project venvs with uv, and installs the
desktop launcher.

Options:
  --reset            Remove selected venvs before reinstalling them.
  --refresh-locks    Recompile requirements/*.lock.txt before installing.
  --skip-venvs       Do not install runtime/sim/IK/ML venvs.
  --skip-runtime     Do not install .venv.
  --skip-sim         Do not install .venv-sim.
  --skip-ik          Do not install .venv-ik.
  --skip-ml          Do not install .venv-ml.
  --skip-desktop     Do not install the desktop launcher.
  --desktop-only     Install uv if needed and the desktop launcher only.
  --skip-uv-install  Do not install uv automatically when it is missing.
  --python PATH      Python executable for new venvs. Default: python3.12.
  --uv PATH          uv executable. Default: uv.
  --torch-backend X  uv torch backend for ML locks/install. Default: cu130.
  -h, --help         Show this help.

Environment:
  PYTHON_BIN          Same as --python.
  UV_BIN              Same as --uv.
  IGRIS_TORCH_BACKEND Same as --torch-backend.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)
      RESET_ENVS=1
      shift
      ;;
    --refresh-locks)
      REFRESH_LOCKS=1
      shift
      ;;
    --skip-venvs)
      INSTALL_RUNTIME=0
      INSTALL_SIM=0
      INSTALL_IK=0
      INSTALL_ML=0
      shift
      ;;
    --skip-runtime)
      INSTALL_RUNTIME=0
      shift
      ;;
    --skip-sim)
      INSTALL_SIM=0
      shift
      ;;
    --skip-ik)
      INSTALL_IK=0
      shift
      ;;
    --skip-ml)
      INSTALL_ML=0
      shift
      ;;
    --skip-desktop)
      INSTALL_DESKTOP=0
      shift
      ;;
    --desktop-only)
      INSTALL_RUNTIME=0
      INSTALL_SIM=0
      INSTALL_IK=0
      INSTALL_ML=0
      INSTALL_DESKTOP=1
      shift
      ;;
    --skip-uv-install)
      ENSURE_UV=0
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --uv)
      UV_BIN="$2"
      shift 2
      ;;
    --torch-backend)
      TORCH_BACKEND="$2"
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

step() {
  echo
  echo "==> $*"
}

find_uv_on_disk() {
  local candidate
  for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

normalize_uv_bin() {
  if command -v "${UV_BIN}" >/dev/null 2>&1; then
    UV_BIN="$(command -v "${UV_BIN}")"
    return 0
  fi

  if [[ "${UV_BIN}" == "uv" ]]; then
    local disk_uv
    if disk_uv="$(find_uv_on_disk)"; then
      UV_BIN="${disk_uv}"
      export PATH="$(dirname "${UV_BIN}"):${PATH}"
      return 0
    fi
  fi

  return 1
}

ensure_uv() {
  if normalize_uv_bin; then
    echo "Using uv: ${UV_BIN}"
    "${UV_BIN}" --version
    return 0
  fi

  if [[ "${ENSURE_UV}" != "1" ]]; then
    echo "uv not found; continuing because --skip-uv-install was set."
    return 0
  fi

  if [[ "${UV_BIN}" != "uv" ]]; then
    echo "uv executable not found: ${UV_BIN}" >&2
    exit 1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv automatically." >&2
    exit 1
  fi

  step "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

  if ! normalize_uv_bin; then
    echo "uv installation finished, but uv is still not on PATH." >&2
    exit 1
  fi

  echo "Using uv: ${UV_BIN}"
  "${UV_BIN}" --version
}

assert_python_exists() {
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
}

assert_no_venv_processes() {
  local venv_path="$1"
  local matches

  if ! command -v ps >/dev/null 2>&1 || ! command -v awk >/dev/null 2>&1; then
    return 0
  fi

  matches="$(ps -eo pid=,comm=,args= | awk -v needle="${venv_path}/bin/python" '$2 !~ /awk$/ && index($0, needle) > 0 { print }')"
  if [[ -n "${matches}" ]]; then
    echo "Refusing to reset venv while Python processes are still running: ${venv_path}" >&2
    echo "${matches}" >&2
    exit 1
  fi
}

reset_venv_if_selected() {
  local venv_path="$1"

  if [[ "${RESET_ENVS}" != "1" || ! -e "${venv_path}" ]]; then
    return 0
  fi

  if [[ ! -f "${venv_path}/pyvenv.cfg" ]]; then
    echo "Refusing to remove non-venv path: ${venv_path}" >&2
    exit 1
  fi

  assert_no_venv_processes "${venv_path}"
  rm -rf "${venv_path}"
}

desktop_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

install_desktop_launcher() {
  local app_id="${IGRIS_DESKTOP_APP_ID:-igris-teleop.desktop}"
  local app_name="${IGRIS_DESKTOP_APP_NAME:-IGRIS Teleop}"
  local desktop_dir="${XDG_DESKTOP_DIR:-${HOME}/Desktop}"
  local applications_dir="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
  local launcher_path="${REPO_ROOT}/run_igris_teleop.sh"
  local options_path="${REPO_ROOT}/igris_teleop_desktop_options.env"
  local log_dir="${REPO_ROOT}/igris_artifacts/logs/desktop_launcher"
  local install_desktop_icon="${IGRIS_INSTALL_DESKTOP_ICON:-1}"
  local desktop_entry="${applications_dir}/${app_id}"
  local desktop_file_content

  if [[ ! -x "${launcher_path}" ]]; then
    echo "Launcher script not executable: ${launcher_path}" >&2
    echo "Run: chmod +x ${launcher_path}" >&2
    exit 1
  fi

  mkdir -p "${applications_dir}" "${log_dir}"

  if [[ "${install_desktop_icon}" == "1" ]]; then
    mkdir -p "${desktop_dir}"
  fi

  desktop_file_content="$(cat <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=${app_name}
Comment=Launch IGRIS Teleop Web UI
Exec=$(desktop_quote "${launcher_path}")
Path=${REPO_ROOT}
Icon=applications-engineering
Terminal=true
Categories=Science;Robotics;
StartupNotify=false
Actions=EditLaunchOptions;OpenProjectFolder;OpenLauncherLogs;

[Desktop Action EditLaunchOptions]
Name=Edit Launch Options
Exec=xdg-open $(desktop_quote "${options_path}")

[Desktop Action OpenProjectFolder]
Name=Open Project Folder
Exec=xdg-open $(desktop_quote "${REPO_ROOT}")

[Desktop Action OpenLauncherLogs]
Name=Open Launcher Logs
Exec=xdg-open $(desktop_quote "${log_dir}")
EOF
)"

  printf '%s\n' "${desktop_file_content}" > "${desktop_entry}"
  chmod +x "${desktop_entry}"

  if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "${desktop_entry}"
  fi

  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${applications_dir}" >/dev/null 2>&1 || true
  fi

  if [[ "${install_desktop_icon}" == "1" ]]; then
    local desktop_icon="${desktop_dir}/${app_id}"
    printf '%s\n' "${desktop_file_content}" > "${desktop_icon}"
    chmod +x "${desktop_icon}"
    if command -v gio >/dev/null 2>&1; then
      gio set "${desktop_icon}" metadata::trusted true >/dev/null 2>&1 || true
    fi
    echo "Desktop launcher installed: ${desktop_icon}"
  fi

  echo "Application launcher installed: ${desktop_entry}"
}

cd "${REPO_ROOT}"

ensure_uv

if [[ "${INSTALL_RUNTIME}" == "1" || "${INSTALL_IK}" == "1" || "${INSTALL_ML}" == "1" ]]; then
  assert_python_exists
fi

if [[ "${REFRESH_LOCKS}" == "1" ]]; then
  if ! normalize_uv_bin; then
    echo "--refresh-locks requires uv." >&2
    exit 1
  fi
  step "Refreshing uv lock files"
  PYTHON_BIN="${PYTHON_BIN}" UV_BIN="${UV_BIN}" IGRIS_TORCH_BACKEND="${TORCH_BACKEND}" \
    "${REPO_ROOT}/requirements/compile_uv_locks.sh"
fi

if [[ "${INSTALL_RUNTIME}" == "1" ]]; then
  reset_venv_if_selected "${REPO_ROOT}/.venv"
  step "Installing runtime venv"
  UV_BIN="${UV_BIN}" "${REPO_ROOT}/igris_teleop/setup_runtime_venv.sh" --python "${PYTHON_BIN}"
fi

if [[ "${INSTALL_SIM}" == "1" ]]; then
  reset_venv_if_selected "${REPO_ROOT}/.venv-sim"
  step "Installing simulator venv"
  SIM_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${SIM_PYTHON_BIN}" ]]; then
    SIM_PYTHON_BIN="${PYTHON_BIN}"
  fi
  PYTHON_BIN="${SIM_PYTHON_BIN}" UV_BIN="${UV_BIN}" \
    "${REPO_ROOT}/igris_teleop/sim/setup_sim_venv.sh"
fi

if [[ "${INSTALL_IK}" == "1" ]]; then
  reset_venv_if_selected "${REPO_ROOT}/.venv-ik"
  step "Installing IK venv"
  UV_BIN="${UV_BIN}" "${REPO_ROOT}/igris_teleop/robot_control/ik/setup_ik_env.sh" --python "${PYTHON_BIN}"
fi

if [[ "${INSTALL_ML}" == "1" ]]; then
  reset_venv_if_selected "${REPO_ROOT}/.venv-ml"
  step "Installing ML venv"
  PYTHON_BIN="${PYTHON_BIN}" UV_BIN="${UV_BIN}" IGRIS_TORCH_BACKEND="${TORCH_BACKEND}" \
    "${REPO_ROOT}/igris_teleop/setup_ml_venv.sh"
fi

if [[ "${INSTALL_DESKTOP}" == "1" ]]; then
  step "Installing desktop launcher"
  install_desktop_launcher
fi

echo
echo "IGRIS Teleop install complete."
