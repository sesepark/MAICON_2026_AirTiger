#!/usr/bin/env bash
# 내 PC(WSL)에서: 학습 코드와 데이터셋을 학습 서버로 올린다.
#   bash push.sh <ssh 대상> <데이터셋 폴더> [서버 폴더]
#     ssh 대상: user@호스트 또는 ~/.ssh/config 의 이름 (환경변수 TRAIN_SSH 로도)
#     데이터셋 폴더: 웹 화면 「내보내기」 결과 (perception/data/drone_datasets/<시각>_undist_map)
#     서버 폴더: 기본 ~/maicon_drone_train (TRAIN_DIR) → <서버 폴더>/code · <서버 폴더>/datasets/<데이터셋 이름>
#   WEIGHTS=yolo11n-obb.pt bash push.sh ...   # 서버에 인터넷이 없으면 사전학습 가중치도 같이 올린다 (code/ 에 놓인다)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH_T="${1:-${TRAIN_SSH:-}}"; DS="${2:-}"; R="${3:-${TRAIN_DIR:-~/maicon_drone_train}}"   # ~ 는 서버에서 풀린다 (ssh 로 들어간 곳이 홈이 아닌 서버도 있다 — 예: WSL)
[ -n "$SSH_T" ] && [ -n "$DS" ] || { sed -n '2,8p' "$0"; exit 2; }
[ -f "$DS/data.yaml" ] || { echo "데이터셋이 아닙니다 (data.yaml 없음): $DS"; exit 2; }
NAME="$(basename "${DS%/}")"
echo "== $SSH_T:$R 로 올림 (코드 · 데이터셋 $NAME)"
ssh "$SSH_T" "mkdir -p $R/code $R/datasets"
rsync -az --delete --exclude runs/ --exclude __pycache__/ --exclude '*.pyc' "$HERE/" "$SSH_T:$R/code/"
rsync -az "${DS%/}/" "$SSH_T:$R/datasets/$NAME/"
[ -n "${WEIGHTS:-}" ] && rsync -az "$WEIGHTS" "$SSH_T:$R/code/"
echo "== 올렸습니다. 서버에서:"
echo "  ssh $SSH_T"
echo "  bash $R/code/setup_server.sh          # 처음 한 번"
echo "  bash $R/code/run.sh $NAME             # 학습 (뒤에서 돈다)"
