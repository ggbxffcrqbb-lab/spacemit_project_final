#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${PROJECT_ROOT}/benchmarks/voice_mode_compare_${STAMP}.json"

python "${PROJECT_ROOT}/benchmarks/voice_mode_compare.py" --output "${OUTPUT}" "$@"
echo "[OK] benchmark report saved to ${OUTPUT}"
