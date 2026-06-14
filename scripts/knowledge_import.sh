#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLOBAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|--mode)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        exit 2
      fi
      GLOBAL_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

exec "${PROJECT_ROOT}/scripts/voice.sh" "${GLOBAL_ARGS[@]}" knowledge-import "$@"
