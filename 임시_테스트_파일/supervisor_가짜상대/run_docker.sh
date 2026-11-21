#!/usr/bin/env bash
# supervisor 를 가짜 상대(fake_world.py)와 함께 ROS 2 Humble 컨테이너에서 한 경기 돌린다.
#   임시_테스트_파일/supervisor_가짜상대/run_docker.sh [정찰방식 face|place] [경기 길이 초] [목표거점 게시 시각 초] [자리]
# 실제 로봇 · 실제 인식이 아니다. supervisor 의 메시지 주고받기와 흐름만 본다.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:-face}"; running="${2:-300}"; target_at="${3:-240}"; slot="${4:-B-1}"
image="${MAICON_ROS_IMAGE:-maicon-mock-ros}"
docker run --rm -e ROS_LOCALHOST_ONLY=1 -v "$repo":/repo:ro "$image" bash -c "
set -e
source /opt/ros/humble/setup.bash
mkdir -p /tmp/ws/src
cp -r /repo/개발/src/team_interfaces /repo/개발/src/supervisor /repo/임시_테스트_파일/supervisor_가짜상대/mil_msgs /tmp/ws/src/
cd /tmp/ws
colcon build --packages-select team_interfaces mil_msgs supervisor > /tmp/build.log 2>&1 || { tail -40 /tmp/build.log; exit 1; }
source install/setup.bash
export PYTHONPATH=/repo/개발/src/supervisor/test:\$PYTHONPATH
echo '{\"scout_mode\": \"$mode\", \"mission\": {\"target_at_s\": $target_at}}' > /tmp/supervisor.json   # 거점 줄은 지도 메시지에 들어 있다
SUPERVISOR_CONFIG=/tmp/supervisor.json ros2 run supervisor supervisor_node > /tmp/sup.log 2>&1 &
sup=\$!
python3 /repo/임시_테스트_파일/supervisor_가짜상대/fake_world.py --slot $slot --running $running --target-at $target_at > /tmp/world.log 2>&1 || true
code=0; for i in 1 2 3 4 5 6 7 8; do if kill -0 \$sup 2>/dev/null; then sleep 0.5; else wait \$sup || code=\$?; break; fi; done
if kill -0 \$sup 2>/dev/null; then echo 'supervisor 가 stop 뒤 4초 안에 끝나지 않음'; kill -INT \$sup; fi
echo \"=== supervisor 종료 코드 \$code\"
echo '=== supervisor 기록'; grep -v 'INFO.*경기 신호' /tmp/sup.log | sed -e 's/^\[INFO\] \[[0-9.]*\] \[supervisor\]: //' | tail -40
echo '=== 가짜 상대'; grep -E 'FAKE_WORLD_SUMMARY|Traceback|Error|error' -A3 /tmp/world.log | tail -30
"
