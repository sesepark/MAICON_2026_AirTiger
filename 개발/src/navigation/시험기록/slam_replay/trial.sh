#!/usr/bin/env bash
set -eo pipefail
if [[ ${1:-} == prepare && $# == 1 ]]; then
    [[ -f source.tar.gz ]]
    mkdir -p src
    tar -xzf source.tar.gz -C src
    source /opt/ros/humble/setup.bash
    colcon build --base-paths src --packages-select team_interfaces navigation \
        --event-handlers console_direct+ 2>&1 | tee build.log
    exit
fi

[[ $# -ge 2 ]] || { echo 'Usage: trial.sh {probe|record|mark} RUN_ID [LABEL]' >&2; exit 2; }
action=$1
id=$2
[[ $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
# The staged tool lives under src/navigation/시험기록/slam_replay.
workspace=$(cd "$(dirname "$0")/../../../.." && pwd)
run="$workspace/runs/$id"
if [[ $action == mark ]]; then
    [[ $# == 3 && $3 != *$'\t'* && $3 != *$'\n'* ]] || exit 2
    [[ -d $run ]]
    printf '%s\t%s\n' "$(date +%s%N)" "$3" >> "$run/marks.tsv"
    exit
fi
source /opt/ros/humble/setup.bash
source "$workspace/install/local_setup.bash"
# The factory stack is local to the container.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
case $action in
    probe)
        mkdir -p "$workspace/runs"
        {
            date -Ins
            df -h "$workspace"
            timeout 10s ros2 node list
            timeout 10s ros2 topic list -t
            for topic in /odom /scan /tf /tf_static /camera_info /image_raw \
                /team/server/uav_image /team/server/state /team/perception/field_map \
                /team/navigation/map /team/navigation/pose /cmd_vel; do
                printf '\n--- %s ---\n' "$topic"
                timeout 8s ros2 topic info -v "$topic" || true
            done
            timeout 5s ros2 run tf2_ros tf2_echo base_link base_lidar_link || true
        } 2>&1 | tee "$workspace/runs/$id.probe.log"
        ;;
    record)
        mkdir -p "$workspace/runs"
        nodes=$(timeout 10s ros2 node list)
        if [[ $nodes =~ (^|[[:space:]])/slam_trial($|[[:space:]]) ]]; then
            echo 'Existing experiment SLAM node detected. Stop the duplicate before recording.' >&2
            exit 1
        fi
        bash "$(dirname "$0")/record.sh" "$run"
        ;;
    *) exit 2 ;;
esac
