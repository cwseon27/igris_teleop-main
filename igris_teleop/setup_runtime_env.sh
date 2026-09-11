#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: igris_teleop/setup_runtime_env.sh <venv|conda> [backend options]

Examples:
  igris_teleop/setup_runtime_env.sh venv --reset
  igris_teleop/setup_runtime_env.sh conda --reset --name igris-teleop-runtime
  igris_teleop/setup_runtime_env.sh conda --reset --prefix ./.conda-runtime
USAGE
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

backend="$1"
shift

case "${backend}" in
  venv)
    exec "${SCRIPT_DIR}/setup_runtime_venv.sh" "$@"
    ;;
  conda)
    exec "${SCRIPT_DIR}/setup_runtime_conda.sh" "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown backend: ${backend}" >&2
    usage >&2
    exit 2
    ;;
esac
