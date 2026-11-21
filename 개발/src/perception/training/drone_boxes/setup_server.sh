#!/usr/bin/env bash
# 학습 서버(리눅스 + NVIDIA GPU)에서 한 번 돌린다: 가상환경을 만들고 torch(CUDA) · ultralytics 를 깐다.
#   bash setup_server.sh                     # 기본 가상환경 ~/.venvs/maicon_drone
#   VENV=~/venvs/x bash setup_server.sh      # 다른 곳에
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash setup_server.sh   # 드라이버가 오래돼 CUDA 판을 골라야 할 때
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/.venvs/maicon_drone}"
PY="${PYTHON:-python3}"

echo "== 가상환경: $VENV ($($PY --version))"
if [ ! -x "$VENV/bin/python" ]; then
  if ! "$PY" -m venv "$VENV" 2>/dev/null; then
    # ensurepip 가 없는 배포판 (sudo 없이): pip 없이 만들고 get-pip 으로 넣는다
    rm -rf "$VENV"; "$PY" -m venv --without-pip "$VENV"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.$$.py && "$VENV/bin/python" /tmp/get-pip.$$.py -q && rm -f /tmp/get-pip.$$.py
  fi
fi
PIP="$VENV/bin/python -m pip"
$PIP install -q -U pip

if [ -n "${TORCH_INDEX:-}" ]; then
  echo "== torch ($TORCH_INDEX)"; $PIP install -q --index-url "$TORCH_INDEX" torch torchvision
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "== GPU:"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  echo "== torch (CUDA 판, PyPI 기본)"; $PIP install -q torch torchvision
else
  echo "!! nvidia-smi 가 없습니다 — CPU 판 torch 를 깝니다 (학습이 아주 느립니다)"
  $PIP install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi
echo "== ultralytics 외"; $PIP install -q -r "$HERE/requirements.txt"

# GUI 라이브러리(libGL)가 없는 서버에서는 opencv-python 이 안 읽힌다 → headless 판으로 바꾼다
if ! "$VENV/bin/python" -c "import cv2" 2>/dev/null; then
  echo "== cv2 를 못 읽어 opencv-python-headless 로 바꿉니다"
  $PIP uninstall -q -y opencv-python || true
  $PIP install -q opencv-python-headless
fi

"$VENV/bin/python" - <<'PY'
import torch, ultralytics, cv2
print("torch", torch.__version__, "· CUDA", torch.cuda.is_available(), "·", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")
print("ultralytics", ultralytics.__version__, "· opencv", cv2.__version__)
PY
echo "== 끝. 학습: bash $HERE/run.sh <데이터셋 이름>"
