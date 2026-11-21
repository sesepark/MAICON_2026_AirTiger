#!/usr/bin/env bash
# 로봇에서와 같은 방식으로 한 경기를 돌린다: team_node.py(개발/robot)가 server_bridge · llm_worker 와 가짜 상대를 자식으로 켜고,
# 자기 안에서 supervisor 를 돌리고, stop 뒤 제때 끝나는지 본다. 가짜 상대(fake_world.py)는 로봇 · navigation · perception · 경기 서버(HTTP)를 흉내 낸다.
#   임시_테스트_파일/supervisor_가짜상대/run_team_node.sh [정찰방식 place|face] [경기 길이 초] [목표거점 게시 시각 초] [자리]
# 환경변수 (모두 없어도 된다)
#   LLM_KEY_FILE=<키가 한 줄 든 파일>  진짜 LLM 에 묻는다 (키는 어디에도 찍지 않는다). 없으면 llm_worker 는 낱말 해석만 한다
#   LLM_WAIT_S=<초>                   llm_worker 가 LLM 답을 기다리는 시간 (짧게 주면 「먼저 낱말 해석, 뒤에 LLM 답으로 교체」 길을 본다)
#   NO_LLM_WORKER=1                   llm_worker 를 켜지 않는다 (supervisor 가 낱말 해석기를 직접 부르는 길을 본다)
#   STANDBY_S=<초> QUIET_NODES=perception   시작 대기 길이와, 상태를 내보내지 않을 프로그램 (준비완료 시한 12초를 볼 때: STANDBY_S=16)
#   PRACTICE=1                        참가팀 포털의 팀 테스트 흉내: 로봇은 연습 모드로 남아 경기 신호를 주지 않고, 경기는 시험 포트(8100)에서 열린다
#   DRONE_CAMERA=test1                server_bridge 설정에 연습용 카메라 이름을 적는다 (시험 포트를 쓸 때만 붙어 나가야 한다)
#   OUT_DIR=<폴더>                    기록 폴더(~/team/log 에 해당)를 여기로 복사해 둔다
# 실제 서버 · 실제 로봇이 아니다.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:-place}"; running="${2:-25}"; target_at="${3:-15}"; slot="${4:-B-1}"
image="${MAICON_ROS_IMAGE:-maicon-mock-ros}"
mounts=(-v "$repo":/repo:ro)
[ -n "${LLM_KEY_FILE:-}" ] && mounts+=(-v "$LLM_KEY_FILE":/run/llm_key:ro)
[ -n "${OUT_DIR:-}" ] && mkdir -p "$OUT_DIR" && mounts+=(-v "$OUT_DIR":/out)
standby="${STANDBY_S:-4}"; quiet="${QUIET_NODES:-}"
practice=""; [ -n "${PRACTICE:-}" ] && practice="'--practice', '--test-port', '8100',"
docker run --rm -e ROS_LOCALHOST_ONLY=1 -e LLM_WAIT_S="${LLM_WAIT_S:-}" -e NO_LLM_WORKER="${NO_LLM_WORKER:-}" -e DRONE_CAMERA="${DRONE_CAMERA:-}" "${mounts[@]}" "$image" bash -c "
set -e
source /opt/ros/humble/setup.bash
mkdir -p /tmp/team/ws/src
cp -r /repo/개발/src/team_interfaces /repo/개발/src/supervisor /repo/개발/src/server_bridge /repo/개발/src/llm_worker /repo/임시_테스트_파일/supervisor_가짜상대/mil_msgs /tmp/team/ws/src/
cd /tmp/team/ws && colcon build > /tmp/build.log 2>&1 || { tail -40 /tmp/build.log; exit 1; }
cp /repo/개발/robot/team_node.py /tmp/team/
cd /tmp/team
python3 - <<'PY'
import json, os
world = {'name': 'fake_world', 'restart': False,
         'cmd': ['python3', '/repo/임시_테스트_파일/supervisor_가짜상대/fake_world.py', '--slot', '$slot', '--standby', '$standby', '--running', '$running', '--report', '4',
                 '--target-at', '$target_at', '--http', '8000', $practice '--quiet-nodes', '$quiet'],
         'env': {'PYTHONPATH': '/repo/개발/src/supervisor/test:\${PYTHONPATH}'}}
programs = [world, {'name': 'server_bridge', 'package': 'server_bridge', 'executable': 'server_bridge_node'}]
if not os.environ.get('NO_LLM_WORKER'):
    programs.append({'name': 'llm_worker', 'package': 'llm_worker', 'executable': 'llm_worker_node'})
json.dump({'programs': programs}, open('programs.json', 'w'), ensure_ascii=False)
json.dump({'scout_mode': '$mode', 'mission': {'target_at_s': $target_at}}, open('supervisor.json', 'w'))
secrets = {'server_url': 'http://127.0.0.1:8000', 'mission_code': 'TEST-CODE-1234'}
if os.path.exists('/run/llm_key'):
    secrets['llm_api_key'] = open('/run/llm_key').read().strip()
json.dump(secrets, open('secrets.json', 'w'))
if os.environ.get('DRONE_CAMERA'):
    json.dump({'drone_camera': os.environ['DRONE_CAMERA']}, open('server_bridge.json', 'w'))
if os.environ.get('LLM_WAIT_S'):
    json.dump({'wait_s': float(os.environ['LLM_WAIT_S'])}, open('llm_worker.json', 'w'))
PY
code=0; python3 team_node.py > /tmp/team/out.log 2>&1 || code=\$?
ended=\$(date +%s.%N)
session=\$(ls -d /tmp/team/log/*/ | tail -1)
echo \"=== team_node 종료 코드 \$code\"
stop_at=\$(grep -a '경기 신호' \${session}fake_world.log 2>/dev/null | grep -a 'stop' | sed -E 's/^\[INFO\] \[([0-9.]+)\].*/\1/' | tail -1)
python3 -c \"
s = '\$stop_at'
print('=== stop 신호부터 team_node 가 끝날 때까지 %.2f 초' % (float('\$ended') - float(s)) if s else '=== stop 신호 기록을 못 찾음')\"
sleep 1; echo \"=== 남은 팀 프로세스: \$(ps -eo comm,args | awk '\$1 ~ /^python3/ && /fake_world.py|team_node.py|server_bridge_node|llm_worker_node/' | wc -l)\"
echo '=== team_node 출력'; grep -a -E 'team_node|supervisor 켜짐|M1|하는 일|발사|지정표적|준비|단계:|Traceback|Error|예외' /tmp/team/out.log | sed -e 's/^\[[A-Z]*\] \[[0-9.]*\] \[supervisor\]: //' | cut -c1-230 | head -30
echo '=== server_bridge'; sed -e 's/^\[[A-Z]*\] \[[0-9.]*\] \[server_bridge\]: //' \${session}server_bridge.log 2>/dev/null | cut -c1-200 | tail -9
echo '=== llm_worker'; sed -e 's/^\[[A-Z]*\] \[[0-9.]*\] \[llm_worker\]: //' \${session}llm_worker.log 2>/dev/null | cut -c1-260 | tail -6
echo '=== 기록 폴더'; (cd \$session && find . -maxdepth 2 | sort | head -40 && echo \"사진 \$(ls supervisor/photos 2>/dev/null | wc -l)장 · 드론 화면 \$(ls server_bridge/drone 2>/dev/null | wc -l)장 · 일어난 일 \$(wc -l < supervisor/events.jsonl)줄 · 서버 요청 \$(wc -l < server_bridge/http.jsonl)줄 · 크기 \$(du -sh . | cut -f1)\"; cat version.json | tr -d '\n' | cut -c1-420; echo)
python3 - <<'PY'
import os
needles = {'임무코드': 'TEST-CODE-1234'}
if os.path.exists('/run/llm_key'):
    needles['LLM 키'] = open('/run/llm_key').read().strip()
hits = {name: 0 for name in needles}
for folder, _dirs, files in os.walk('/tmp/team'):
    if '/ws' in folder:
        continue
    for name in files:
        if name == 'secrets.json':
            continue
        try:
            data = open(os.path.join(folder, name), 'rb').read()
        except OSError:
            continue
        for label, needle in needles.items():
            hits[label] += data.count(needle.encode())
print('=== 기록 · 출력에 찍힌 횟수 (0 이어야 한다):', hits)
PY
echo '=== 가짜 상대'; grep -a -E 'FAKE_WORLD_SUMMARY|Traceback|Error' -A2 \${session}fake_world.log 2>&1 | cut -c1-700 | tail -6
if [ -d /out ]; then rm -rf /out/log && cp -r /tmp/team/log /out/log && cp /tmp/team/out.log /out/out.log && chmod -R a+rwX /out; fi
"
