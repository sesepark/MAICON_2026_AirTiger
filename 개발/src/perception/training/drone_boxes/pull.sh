#!/usr/bin/env bash
# 내 PC(WSL)에서: 학습이 끝난 실행의 결과(가중치 · 평가 · 그래프)를 내려받는다 (중간 체크포인트는 빼고).
#   bash pull.sh <ssh 대상> <실행 이름> [서버 폴더] [내 PC 폴더]
#     내 PC 폴더 기본: perception/data/drone_models/<실행 이름> (git 에 안 올라감)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH_T="${1:-${TRAIN_SSH:-}}"; NAME="${2:-}"; R="${3:-${TRAIN_DIR:-~/maicon_drone_train}}"   # ~ 는 서버에서 풀린다 (ssh 로 들어간 곳이 홈이 아닌 서버도 있다 — 예: WSL)
[ -n "$SSH_T" ] && [ -n "$NAME" ] || { sed -n '2,5p' "$0"; exit 2; }
DST="${4:-$HERE/../../data/drone_models/$NAME}"
mkdir -p "$DST"
rsync -az --exclude 'weights/last.pt' --exclude 'weights/epoch*.pt' "$SSH_T:$R/code/runs/$NAME/" "$DST/"
rsync -az "$SSH_T:$R/code/runs/$NAME.log" "$DST/" 2>/dev/null || true
echo "== 받았습니다: $DST"
[ -f "$DST/eval_val/eval.md" ] && cat "$DST/eval_val/eval.md"
ls "$DST/export" 2>/dev/null || true
