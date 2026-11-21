#!/usr/bin/env bash
# 경기 도중에 supervisor 를 강제로 끄고 다시 켰을 때, 저장해 둔 상태를 읽어 이어 가는지 본다 (가짜 상대와 5분 경기).
#   임시_테스트_파일/supervisor_가짜상대/run_restart.sh [정찰방식] [끄는 시각 초]
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:-place}"; kill_at="${2:-90}"
image="${MAICON_ROS_IMAGE:-maicon-mock-ros}"
docker run --rm -e ROS_LOCALHOST_ONLY=1 -v "$repo":/repo:ro "$image" bash -c "
set -e
source /opt/ros/humble/setup.bash
mkdir -p /tmp/ws/src /tmp/run
cp -r /repo/개발/src/team_interfaces /repo/개발/src/supervisor /repo/임시_테스트_파일/supervisor_가짜상대/mil_msgs /tmp/ws/src/
cd /tmp/ws
colcon build --packages-select team_interfaces mil_msgs supervisor > /tmp/build.log 2>&1 || { tail -40 /tmp/build.log; exit 1; }
source install/setup.bash
export PYTHONPATH=/repo/개발/src/supervisor/test:\$PYTHONPATH
cd /tmp/run
echo '{\"scout_mode\": \"$mode\"}' > supervisor.json
python3 /repo/임시_테스트_파일/supervisor_가짜상대/fake_world.py --slot B-1 --running 300 --target-at 240 > /tmp/world.log 2>&1 &
world=\$!
ros2 run supervisor supervisor_node > /tmp/sup1.log 2>&1 &
sleep $((kill_at + 4))
pkill -9 -f 'supervisor_node' || true
echo \"=== ${kill_at}초에 supervisor 를 강제로 끔. 저장 파일: \$(ls -la supervisor_state.json 2>&1 | cut -c1-80)\"
sleep 3
ros2 run supervisor supervisor_node > /tmp/sup2.log 2>&1 &
wait \$world || true
sleep 2
echo '=== 끄기 전 기록'; sed -e 's/^\[INFO\] \[[0-9.]*\] \[supervisor\]: //' /tmp/sup1.log | grep -v '하는 일: wait' | tail -8
echo '=== 다시 켠 뒤 기록'; sed -e 's/^\[INFO\] \[[0-9.]*\] \[supervisor\]: //' /tmp/sup2.log | grep -v '하는 일: wait' | tail -14
echo '=== 가짜 상대'; grep -a -E 'FAKE_WORLD_SUMMARY|Traceback' -A2 /tmp/world.log | cut -c1-700 | tail -4
"
