#!/usr/bin/env bash
# 로봇 컨테이너 안에서 돌린다 (robot.sh 가 부른다. 손으로 해도 된다: sudo ugv-shell 뒤에).
#   bash run.sh build     team_interfaces · navigation 을 이 시험 전용 작업공간에 빌드
#   bash run.sh start     (라이다가 꺼져 있으면 라이다) · SLAM · 주행 · 웹 시험기 · 기록(rosbag)을 켠다
#   bash run.sh stop      위에서 켠 것을 끈다 (주행 노드는 끝나면서 정지 명령을 보낸다)
#   bash run.sh status    켜져 있는지와 로그 끝부분
# 환경변수: MAX_V(기본 0.25 m/s) · PORT(5960) · DRIVE_ARGS · SLAM_ARGS (ROS 파라미터를 더 줄 때, 예: DRIVE_ARGS="-p min_turn_w:=1.0")
set -eo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd)
WS=$(cd "$TOOLS/../../../.." && pwd)        # .../nav_web
cd "$WS"
source /opt/ros/humble/setup.bash
[[ -f /home/ws/ugv_ws/install/local_setup.bash ]] && source /home/ws/ugv_ws/install/local_setup.bash
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"   # 로봇의 기본 노드들이 컨테이너 안에서만 통신한다
export PYTHONUNBUFFERED=1

stop_all() {
    local file="$WS/runs/latest/pids" name pid
    [[ -f $file ]] || return 0
    while read -r name pid; do
        kill -INT -- "-$pid" 2>/dev/null || true
    done < "$file"
    sleep 2
    while read -r name pid; do
        if kill -0 -- "-$pid" 2>/dev/null; then
            echo "강제 종료: $name"
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    done < "$file"
    mv "$file" "$file.stopped"
}

case ${1:-} in
build)
    colcon build --base-paths src --packages-select team_interfaces navigation 2>&1 | tee build.log | tail -6
    ;;
start)
    [[ -f install/local_setup.bash ]] || { echo '먼저: bash run.sh build' >&2; exit 1; }
    source install/local_setup.bash
    stop_all
    run="$WS/runs/$(date +%m%d_%H%M%S)"
    mkdir -p "$run"
    ln -sfn "$run" "$WS/runs/latest"
    launch() {
        local name=$1
        shift
        setsid nohup "$@" > "$run/$name.log" 2>&1 &
        echo "$name $!" >> "$run/pids"
    }
    if timeout 8 ros2 topic info /scan 2>/dev/null | grep -q 'Publisher count: [1-9]'; then
        echo '라이다: 이미 켜져 있음'
    else
        echo '라이다: 꺼져 있어 LD19 드라이버를 켠다'
        launch lidar ros2 launch ldlidar ld19.launch.py
    fi
    launch slam ros2 run navigation navigation_slam --ros-args -p debug:=true $SLAM_ARGS
    launch drive ros2 run navigation navigation_drive --ros-args -p debug_hz:=5.0 -p max_v:="${MAX_V:-0.25}" $DRIVE_ARGS
    launch web python3 -u "$TOOLS/web_drive.py" --port "${PORT:-5960}" --config "$WS/map.json"
    launch bag ros2 bag record -s sqlite3 -o "$run/bag" /odom /odom/odom_raw /scan /imu/data /cmd_vel \
        /team/navigation/pose /team/navigation/map /team/navigation/debug /team/perception/field_map \
        /team/server/state /team/control/state /team/mission/state /team/node_status
    sleep 4
    cat "$run/pids"
    echo "기록: $run"
    echo "화면: http://$(hostname -I | awk '{print $1}'):${PORT:-5960}"
    ;;
stop)
    stop_all
    echo '끔'
    ;;
status)
    file="$WS/runs/latest/pids"
    if [[ -f $file ]]; then
        while read -r name pid; do
            if kill -0 -- "-$pid" 2>/dev/null; then echo "켜짐  $name"; else echo "죽음  $name"; fi
        done < "$file"
    else
        echo '켜 둔 것이 없음'
    fi
    for name in slam drive web lidar; do
        [[ -f $WS/runs/latest/$name.log ]] && { echo "--- $name.log"; tail -n 6 "$WS/runs/latest/$name.log"; }
    done
    ;;
*)
    echo 'Usage: bash run.sh {build|start|stop|status}' >&2
    exit 2
    ;;
esac
