#!/usr/bin/env bash
# 학습 서버에서: 조명 변형 → 학습 → 평가 → 내보내기 를 한 번에, 뒤에서(nohup) 돌린다 — ssh 가 끊겨도 계속 돈다.
#   bash run.sh <데이터셋 이름 또는 경로> [실행 이름] [키=값 ...]
#     데이터셋 이름이면 ../datasets/<이름> (push.sh 가 올린 곳)
#     키=값 은 config/train.yaml 덮어쓰기 (예: epochs=50 batch=8 device=0 model=yolo11s-obb.pt)
#   FOREGROUND=1 bash run.sh ...      # 앞에서 돌린다 (로그를 바로 본다)
#   주의: Windows 의 SSH 로 들어가는 WSL 서버(예: 팀 학습 PC <학습PC IP>)는 ssh 가 끊기면 WSL 이 통째로 꺼져
#         nohup · tmux 로 띄운 것도 죽는다 (2026-11-19 확인) → FOREGROUND=1 로 ssh 를 학습이 끝날 때까지 열어 둔다
#         (또는 그 PC 에서 WSL 터미널을 하나 열어 둔다)
#   NO_AUGMENT=1 bash run.sh ...      # 조명 변형 없이
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/.venvs/maicon_drone}"
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 2; }
DS="$1"; shift
[ -d "$DS" ] || DS="$HERE/../datasets/$DS"
[ -f "$DS/data.yaml" ] || { echo "데이터셋이 없습니다: $DS (data.yaml)"; exit 2; }
NAME="$(date +%Y%m%d_%H%M%S)"
if [ $# -ge 1 ] && [[ "$1" != *=* ]]; then NAME="$1"; shift; fi   # 두 번째가 키=값 이면 이름 없이 덮어쓰기
AUG="--augment"; [ -n "${NO_AUGMENT:-}" ] && AUG=""
mkdir -p "$HERE/runs"
LOG="$HERE/runs/$NAME.log"
CMD=("$VENV/bin/python" -u "$HERE/train.py" --data "$DS" $AUG --name "$NAME" "$@")
if [ -n "${FOREGROUND:-}" ]; then
  "${CMD[@]}" 2>&1 | tee "$LOG"
else
  if grep -qi microsoft /proc/version 2>/dev/null && [ -n "${SSH_CONNECTION:-}${SSH_CLIENT:-}" ]; then
    echo "!! WSL 위의 ssh 세션입니다 — ssh 가 끊기면 WSL 이 꺼져 학습도 멈출 수 있습니다. FOREGROUND=1 로 돌리고 세션을 열어 두세요"
  fi
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo "$!" > "$HERE/runs/$NAME.pid"
  echo "학습을 뒤에서 시작했습니다: $NAME (pid $!)"
  echo "  로그 보기:   tail -f $LOG"
  echo "  GPU 보기:    watch -n 2 nvidia-smi"
  echo "  멈추기:      kill \$(cat $HERE/runs/$NAME.pid)"
  echo "  끝나면 내 PC 에서:  bash pull.sh <ssh 대상> $NAME"
fi
