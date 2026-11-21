#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo 'Usage: bash robot.sh {stage|shell|collect} UGV_IP [EXPERIMENT_ID [DESTINATION]]' >&2
    exit 2
}
[[ $# -ge 2 ]] || usage
action=$1
host=$2
[[ $host =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage
IFS=. read -r -a octets <<< "$host"
for i in "${!octets[@]}"; do
    [[ ${octets[i]} =~ ^[0-9]{1,3}$ ]] || usage
    octets[i]=$((10#${octets[i]}))
    (( octets[i] <= 255 )) || usage
done
printf -v remote 'team@%d.%d.%d.%d' "${octets[@]}"
if [[ $action == shell && $# == 2 ]]; then
    exec ssh -t "$remote" 'sudo ugv-shell'
fi
[[ $# -ge 3 ]] || usage
id=$3
[[ $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || usage
remote_dir="/home/jetson/team/slam_trial_$id"

case $action in
    stage)
        [[ $# == 3 ]] || usage
        root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
        archive=$(mktemp)
        trap 'rm -f -- "$archive"' EXIT
        # Include the actual local sources, including uncommitted experiment tools.
        tar -czf "$archive" -C "$root/개발/src" \
            --exclude=__pycache__ --exclude=.pytest_cache \
            navigation/navigation navigation/package.xml navigation/resource \
            navigation/setup.cfg navigation/setup.py navigation/test navigation/시험기록/slam_replay \
            team_interfaces
        revision=$(git -C "$root" rev-parse HEAD)
        ssh "$remote" "mkdir '$remote_dir'"
        scp "$archive" "$remote:$remote_dir/source.tar.gz"
        ssh "$remote" "printf '%s\n' '$revision' > '$remote_dir/revision.txt'"
        printf 'Container workspace: /home/ws/team/slam_trial_%s\n' "$id"
        ;;
    collect)
        [[ $# == 4 ]] || usage
        mkdir -- "$4"
        scp -r "$remote:$remote_dir/runs" "$remote:$remote_dir/source.tar.gz" \
            "$remote:$remote_dir/revision.txt" "$remote:$remote_dir/build.log" "$4/"
        ;;
    *) usage ;;
esac
