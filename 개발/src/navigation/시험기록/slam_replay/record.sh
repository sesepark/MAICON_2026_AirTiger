#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: bash record.sh NEW_RUN_DIRECTORY" >&2
    exit 2
fi
command -v ros2 >/dev/null
ros2 pkg executables navigation | awk '$2 == "navigation_slam" { found=1 } END { exit !found }'
run_dir=$1
tools_dir=$(cd "$(dirname "$0")" && pwd)
slam_executable="$(ros2 pkg prefix navigation)/lib/navigation/navigation_slam"
[[ -x $slam_executable ]]
mkdir -- "$run_dir"
bag_pid=
slam_pid=

cleanup() {
    if [[ -n $slam_pid ]]; then
        kill -INT "$slam_pid" 2>/dev/null || true
    fi
    if [[ -n $bag_pid ]]; then
        kill -INT "$bag_pid" 2>/dev/null || true
        wait "$bag_pid" || true
    fi
    if [[ -n $slam_pid ]]; then
        wait "$slam_pid" || true
    fi
    if [[ -f $run_dir/bag/metadata.yaml ]]; then
        ros2 bag info "$run_dir/bag" > "$run_dir/bag_info.txt" 2>&1 || true
        python3 "$tools_dir/bag_summary.py" "$run_dir/bag" \
            > "$run_dir/receive_timing.json" 2> "$run_dir/summary.log" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
    date -Ins
    git rev-parse HEAD || true
    git status --short || true
    printf 'ROS_DISTRO=%s\nROS_DOMAIN_ID=%s\n' "${ROS_DISTRO:-}" "${ROS_DOMAIN_ID:-}"
    ros2 topic list -t
    sha256sum "$slam_executable"
} > "$run_dir/environment.txt" 2>&1

env --default-signal=INT ros2 bag record -s sqlite3 -o "$run_dir/bag" \
    /odom /odom/odom_raw /scan /imu/data /tf /tf_static \
    /image_raw /camera_info /team/server/uav_image \
    /team/server/state /team/perception/observations \
    /team/perception/field_map /slam_trial/map /slam_trial/pose /cmd_vel \
    > "$run_dir/recorder.log" 2>&1 &
bag_pid=$!

# Launch the installed entry point directly so its PID owns ROS shutdown.
env --default-signal=INT "$slam_executable" --ros-args \
    -r __node:=slam_trial -r /team/navigation/map:=/slam_trial/map \
    -r /team/navigation/pose:=/slam_trial/pose > "$run_dir/slam.log" 2>&1 &
slam_pid=$!
printf 'Recording in %s. Use Ctrl+C to finish and write the summary.\n' "$run_dir"
wait -n "$bag_pid" "$slam_pid"
