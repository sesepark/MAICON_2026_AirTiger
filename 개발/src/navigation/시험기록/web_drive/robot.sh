#!/usr/bin/env bash
# PC(Git Bash)에서 로봇에 올리고 · 빌드하고 · 켜고 · 끈다. 팀 와이파이에 연결한 상태에서, SSH 비밀번호는 물을 때마다 넣는다.
#   bash robot.sh deploy <로봇 IP>          지금 PC 의 소스(올리지 않은 수정 포함)를 올리고 컨테이너 안에서 빌드
#   bash robot.sh start  <로봇 IP>          라이다 · SLAM · 주행 · 웹 시험기를 켠다 → http://<로봇 IP>:5960
#   bash robot.sh stop   <로봇 IP>
#   bash robot.sh status <로봇 IP>
#   bash robot.sh logs   <로봇 IP> <받을 폴더>   마지막 실행의 로그 · rosbag 을 가져온다
# start 에 속도 제한 등을 주려면: MAX_V=0.3 DRIVE_ARGS="-p min_turn_w:=1.0" bash robot.sh start <IP>
# 로봇의 작업공간은 /home/jetson/team/nav_web (컨테이너 안에서는 /home/ws/team/nav_web). 경기용 ~/team/ws 는 건드리지 않는다.
set -euo pipefail
usage() { sed -n '2,9p' "$0" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
action=$1
host=$2
[[ $host =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage
IFS=. read -r -a octets <<< "$host"
printf -v host '%d.%d.%d.%d' "$((10#${octets[0]}))" "$((10#${octets[1]}))" "$((10#${octets[2]}))" "$((10#${octets[3]}))"
remote="team@$host"
outside=/home/jetson/team/nav_web
inside=/home/ws/team/nav_web
tools=$inside/src/navigation/시험기록/web_drive

# 표준 입력으로 준 명령을 컨테이너 안에서 실행한다 (ugv-shell 은 인자를 받지 않는다)
in_container() { ssh "$remote" 'sudo /usr/local/sbin/ugv-shell' 2>&1 | grep -v -e '^root@.*# ' -e 'job control' -e 'process group' || true; }

case $action in
deploy)
    root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
    archive=$(mktemp)
    stage=$(mktemp -d)
    trap 'rm -rf -- "$archive" "$stage"' EXIT
    # 드론 사진 → 지도: perception 의 함수 3개와 연습 경기장 샘플 사진 (drone_map.py 가 src/perception 에서 찾는다).
    # main 에 합쳐졌으면 작업 트리에서, 아니면 git 의 origin/perception 에서 꺼낸다
    perception_files="perception/drone_boundary.py perception/drone_boxes.py perception/drone_field_map.py test/data/drone_test1.jpg test/data/drone_test2.jpg"
    for rel in $perception_files; do
        mkdir -p "$stage/perception/$(dirname "$rel")"
        if [[ -f $root/개발/src/perception/$rel ]]; then
            cp "$root/개발/src/perception/$rel" "$stage/perception/$rel"
        elif ! git -C "$root" show "origin/perception:개발/src/perception/$rel" > "$stage/perception/$rel" 2>/dev/null; then
            rm -f "$stage/perception/$rel"
            echo "주의: perception 의 $rel 이 없음 — 드론 사진 → 지도를 쓸 수 없을 수 있다 (git fetch 로 perception 브랜치를 받을 것)" >&2
        fi
    done
    tar -czf "$archive" -C "$root/개발/src" --exclude=__pycache__ --exclude=map.local.json --exclude=_perception --exclude=secrets.json \
        navigation/navigation navigation/package.xml navigation/resource navigation/setup.cfg navigation/setup.py \
        navigation/시험기록/web_drive team_interfaces -C "$stage" perception
    revision="$(git -C "$root" rev-parse --short=12 HEAD) (+수정 $(git -C "$root" status --porcelain -- 개발 | wc -l | tr -d ' ')개)"
    ssh "$remote" "mkdir -p '$outside'"
    scp "$archive" "$remote:$outside/source.tar.gz"
    in_container <<EOF
cd $inside && rm -rf src && mkdir src && tar -xzf source.tar.gz -C src && echo '$revision' > revision.txt && bash $tools/run.sh build
exit
EOF
    ;;
start)
    in_container <<EOF
MAX_V='${MAX_V:-0.25}' PORT='${PORT:-5960}' DRIVE_ARGS='${DRIVE_ARGS:-}' SLAM_ARGS='${SLAM_ARGS:-}' bash $tools/run.sh start
exit
EOF
    echo "화면: http://$host:${PORT:-5960}"
    ;;
stop|status)
    in_container <<EOF
bash $tools/run.sh $action
exit
EOF
    ;;
logs)
    [[ $# == 3 ]] || usage
    mkdir -p -- "$3"
    in_container <<EOF
cd $inside/runs && tar -czf ../last_run.tgz -h latest && chmod a+r ../last_run.tgz
exit
EOF
    scp "$remote:$outside/last_run.tgz" "$3/"
    echo "받음: $3/last_run.tgz"
    ;;
*) usage ;;
esac
