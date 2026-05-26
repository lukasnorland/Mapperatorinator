#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[run_api] ERROR: venv '$VENV_DIR' not found."
  echo "[run_api] Create it first (example): python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ -f ".env" ]]; then
  echo "[run_api] Loading .env into environment"
  set -a
  # Basic .env loader: KEY=VALUE lines
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

echo "[run_api] Starting Skeleton Design SSE API"
echo "[run_api] POST http://127.0.0.1:8080/api/skeleton-design"
exec python skeleton_design_api.py

