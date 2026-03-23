#!/usr/bin/env bash
set -euo pipefail

CUDA_NUM=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\)\.\([0-9]\+\).*/\1\2/p' | head -n1)

if [ -z "${CUDA_NUM}" ]; then
  echo "Could not detect CUDA version from nvidia-smi."
  exit 1
fi

if [ "${CUDA_NUM}" -ge 130 ]; then
  TORCH_CHANNEL="cu130"
elif [ "${CUDA_NUM}" -ge 128 ]; then
  TORCH_CHANNEL="cu128"
else
  TORCH_CHANNEL="cu126"
fi

echo "Torch channel: ${TORCH_CHANNEL}"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.9.1 \
  torchvision==0.24.1 \
  --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"

python -m pip install \
  numpy==1.26.4 \
  accelerate==1.11.0 \
  dashscope==1.25.10 \
  decord==0.6.0 \
  diffusers==0.36.0 \
  easydict==1.13 \
  einops==0.8.1 \
  ftfy==6.3.1 \
  gradio==6.9.0 \
  imageio==2.37.2 \
  imageio-ffmpeg==0.6.0 \
  matplotlib==3.10.7 \
  opencv-python==4.11.0.86 \
  pillow==11.3.0 \
  pyyaml==6.0.3 \
  safetensors==0.6.2 \
  timm==1.0.22 \
  tokenizers==0.22.1 \
  tqdm==4.67.1 \
  transformers==4.57.1

python -m pip install --no-deps git+https://github.com/Wan-Video/Wan2.1.git

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

for name in ["imageio", "matplotlib", "wan"]:
    try:
        __import__(name)
        print(f"{name}: ok")
    except Exception as e:
        print(f"{name}: failed -> {e}")
PY