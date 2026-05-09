#!/usr/bin/env bash
set -euo pipefail

# Recreate the local GPU TTS environment used by auto_bilibili_dub.py.
# The commands are idempotent where practical, but the CosyVoice model download
# is large, so this script skips it when the target directory already exists.

ENV_DIR="${ENV_DIR:-/home/sil/workspace/conda_envs/veddo-tts}"
COSYVOICE_DIR="${COSYVOICE_DIR:-/home/sil/workspace/CosyVoice}"
MODEL_DIR="${MODEL_DIR:-${COSYVOICE_DIR}/pretrained_models/CosyVoice-300M-SFT}"
PYTHON="${ENV_DIR}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python not found: $PYTHON" >&2
  echo "Create the conda env first, for example: conda create -p $ENV_DIR python=3.10 -y" >&2
  exit 1
fi

"$PYTHON" -m pip install -U pip setuptools wheel
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
"$PYTHON" -m pip install \
  yt-dlp openai srt pyyaml soundfile librosa numpy scipy tqdm requests \
  secretstorage jeepney \
  conformer diffusers gdown grpcio hydra-core HyperPyYAML inflect lightning==2.2.4 \
  matplotlib modelscope==1.20.0 networkx==3.1 onnx onnxruntime-gpu pydantic rich \
  tensorboard transformers==4.51.3 x-transformers wetext wget
"$PYTHON" -m pip install --no-build-isolation --no-deps openai-whisper==20231117 tiktoken

if [[ ! -d "$COSYVOICE_DIR/.git" ]]; then
  git clone https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_DIR"
fi
git -C "$COSYVOICE_DIR" submodule update --init --recursive

if [[ ! -f "$MODEL_DIR/cosyvoice.yaml" ]]; then
  PYTHONPATH="${COSYVOICE_DIR}:${COSYVOICE_DIR}/third_party/Matcha-TTS" "$PYTHON" - <<PY
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='${MODEL_DIR}')
PY
fi

MODEL_DIR="$MODEL_DIR" PYTHONPATH="${COSYVOICE_DIR}:${COSYVOICE_DIR}/third_party/Matcha-TTS" "$PYTHON" - <<'PY'
import os
from cosyvoice.cli.cosyvoice import CosyVoice
model = CosyVoice(os.environ['MODEL_DIR'], load_jit=False, load_trt=False, fp16=True)
print('CosyVoice speakers:', ', '.join(model.list_available_spks()))
PY
