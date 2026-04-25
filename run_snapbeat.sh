#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[run_snapbeat] ERROR: venv '$VENV_DIR' not found."
  echo "[run_snapbeat] Run: bash run_local.sh   (it creates .venv and installs deps)"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ -f ".env" ]]; then
  echo "[run_snapbeat] Loading .env into environment"
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash run_snapbeat.sh "<audio_path>" [hydra overrides...]

Examples:
  bash run_snapbeat.sh "/path/song.mp3" difficulty=5.0 keycount=4 game_code=MT3
  bash run_snapbeat.sh "/path/song.wav" output_path="./out" snapbeat_song_name="My Song"

Notes:
  - `snapbeat_inference.py` uses Hydra. Any extra args after audio_path are passed through.
  - game_code must be one of: MT3, BH, DR (BH/DR require an explicit lora_path unless registered).
EOF
  exit 2
fi

AUDIO_PATH="$1"
shift

exec python snapbeat_inference.py "audio_path=$AUDIO_PATH" "$@"

