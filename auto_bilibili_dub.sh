#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/sil/workspace/conda_envs/veddo-tts/bin/python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export COSYVOICE_ROOT="${COSYVOICE_ROOT:-/home/sil/workspace/CosyVoice}"
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

exec "$PYTHON" "$SCRIPT_DIR/auto_bilibili_dub.py" "$@"
