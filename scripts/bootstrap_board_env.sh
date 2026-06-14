#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements-board.txt"
