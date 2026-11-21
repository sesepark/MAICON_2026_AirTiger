#!/usr/bin/env bash
# 드론 사진 → 지도가 끝까지 전달되는지 컨테이너에서 본다. team_node.py 가 로봇에서와 같은 방식으로 켠다:
#   서버 흉내(드론 사진 = perception 의 시험 사진) → 진짜 server_bridge → 진짜 perception(/team/perception/field_map)
#   → 진짜 navigation_slam(/team/navigation/map) → 진짜 supervisor
#   임시_테스트_파일/supervisor_가짜상대/run_map_chain.sh [드론 사진 파일(저장소 기준 경로)] [자리] [경기 길이 초]
# 환경변수: MAP_PARAMS="-p map_separate_process:=false" 처럼 perception 에 줄 설정, OUT_DIR=<폴더> 기록 폴더를 복사해 둘 곳
#   STANDBY_S=<초>   시작 전 대기 길이 (기본 5). 인식 모델이 없는 컨테이너에서는 perception 이 ERROR 라서, 준비완료는 12초 시한 뒤에야 나간다 → 그것을 보려면 16
#   REAL_PROGRAMS=1  저장소의 개발/robot/programs.json 을 그대로 쓴다 (다섯 프로그램 전부 — 로봇에 올리기 전에 함께 켜지는지 보는 용도. 가짜 상대는 로봇과 서버만 흉내 낸다)
# 사진 판독 모델과 라이다가 없다 → perception 은 사진 판독을 ERROR 로 알리고, navigation 은 위치를 내지 못한다 (지도 전달만 본다).
# 실제 서버 · 실제 로봇이 아니다.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
jpeg="${1:-개발/src/perception/test/data/drone_test2.jpg}"; slot="${2:-B-1}"; running="${3:-40}"
image="${MAICON_ROS_IMAGE:-maicon-mock-ros}"
standby="${STANDBY_S:-5}"
mounts=(-v "$repo":/repo:ro)
[ -n "${OUT_DIR:-}" ] && mkdir -p "$OUT_DIR" && mounts+=(-v "$OUT_DIR":/out)
docker run --rm -e ROS_LOCALHOST_ONLY=1 -e MAP_PARAMS="${MAP_PARAMS:-}" -e REAL_PROGRAMS="${REAL_PROGRAMS:-}" "${mounts[@]}" "$image" bash -c "
set -e
source /opt/ros/humble/setup.bash
mkdir -p /tmp/team/ws/src
cp -r /repo/개발/src/team_interfaces /repo/개발/src/supervisor /repo/개발/src/server_bridge /repo/개발/src/llm_worker /repo/개발/src/perception /repo/개발/src/navigation /repo/임시_테스트_파일/supervisor_가짜상대/mil_msgs /tmp/team/ws/src/
cd /tmp/team/ws && colcon build > /tmp/build.log 2>&1 || { tail -40 /tmp/build.log; exit 1; }
cp /repo/개발/robot/team_node.py /repo/개발/robot/lidar.sh /tmp/team/
cd /tmp/team
python3 - <<'PY'
import json, os
world = {'name': 'fake_world', 'restart': False,
         'cmd': ['python3', '/repo/임시_테스트_파일/supervisor_가짜상대/fake_world.py', '--slot', '$slot', '--standby', '$standby', '--running', '$running', '--report', '4',
                 '--target-at', '30', '--http', '8000', '--no-perception', '--no-slam', '--quiet-nodes', 'navigation', '--drone-jpeg', '/repo/$jpeg'],
         'env': {'PYTHONPATH': '/repo/개발/src/supervisor/test:\${PYTHONPATH}'}}
perception = {'name': 'perception', 'package': 'perception', 'executable': 'perception_node'}
extra = os.environ.get('MAP_PARAMS', '').split()
if extra:
    perception['args'] = ['--ros-args'] + extra
programs = [world, {'name': 'navigation_slam', 'package': 'navigation', 'executable': 'navigation_slam'}, perception,
            {'name': 'server_bridge', 'package': 'server_bridge', 'executable': 'server_bridge_node'},
            {'name': 'llm_worker', 'package': 'llm_worker', 'executable': 'llm_worker_node'}]
if os.environ.get('REAL_PROGRAMS'):
    world['cmd'].append('--no-drive')
    real = json.load(open('/repo/개발/robot/programs.json', encoding='utf-8'))['programs']
    programs = [world] + real
    print('=== 저장소의 프로그램 목록 그대로:', [(p['name'], p.get('enabled', True)) for p in real])
json.dump({'programs': programs}, open('programs.json', 'w'), ensure_ascii=False)
json.dump({'scout_mode': 'place'}, open('supervisor.json', 'w'))
json.dump({'server_url': 'http://127.0.0.1:8000', 'mission_code': 'TEST-CODE-1234'}, open('secrets.json', 'w'))
PY
code=0; python3 team_node.py > /tmp/team/out.log 2>&1 || code=\$?
session=\$(ls -d /tmp/team/log/*/ | tail -1)
echo \"=== team_node 종료 코드 \$code\"
echo '=== perception 이 남긴 말'; sed -e 's/^\[[A-Z]*\] \[[0-9.]*\] \[perception\]: //' \${session}perception.log 2>/dev/null | cut -c1-230 | tail -5
echo '=== navigation_slam 이 남긴 말'; sed -e 's/^\[[A-Z]*\] \[[0-9.]*\] \[navigation_slam\]: //' \${session}navigation_slam.log 2>/dev/null | cut -c1-230 | tail -4
source /tmp/team/ws/install/setup.bash
PYTHONPATH=/repo/개발/src/supervisor:\$PYTHONPATH python3 - \$session <<'PY'
import json, os, sys
from collections import Counter
session = sys.argv[1]
def lines(path):
    try:
        return [json.loads(x) for x in open(os.path.join(session, path), encoding='utf-8') if x.strip()]
    except OSError:
        return []
tries = [r for r in lines('perception/map.jsonl')]
print('=== perception 의 지도 시도')
for r in tries:
    print('   ', {k: r.get(k) for k in ('e', 'ok', 'how', 'took_s', 'seconds', 'boundary_s', 'boxes', 'strip', 'cm_half', 'tries', 'frames', 'error')})
events = lines('supervisor/events.jsonl')
start = next((e for e in events if e.get('e') == 'phase' and e.get('phase') == 'running'), None) or next((e for e in events if 'running' in json.dumps(e)), None)
got = next((e for e in events if e.get('e') == 'map'), None)
if got is None:
    print('=== supervisor 가 지도를 받지 못함. 기록에 있는 일의 종류:', Counter(e.get('e') for e in events).most_common(12))
else:
    when = None
    for key in ('t', 'clock', 'elapsed', 'wall'):
        if start is not None and key in got and key in start and isinstance(got[key], (int, float)) and isinstance(start[key], (int, float)):
            when = '%s 기준 시작 뒤 %.1f초' % (key, got[key] - start[key]); break
    print('=== supervisor 가 받은 지도: 상자 %d개 · 크기 %s · 거점 %s · %s' % (len(got['boxes']), got['size'], sorted(got['bases']), when))
try:
    from supervisor import arena as A
    data = json.load(open(os.path.join(session, 'perception/field_map.json'), encoding='utf-8'))
    from perception.drone_field_map import field_map_msg
    fmap = A.field_map_from_msg(field_map_msg(data), A.ArenaPlan(), None)
    ours = Counter(f.sector for f in fmap.faces)
    theirs = Counter(f['sector'] for f in data['faces'])
    mine = {(f.box, f.side): f.sector for f in fmap.faces}
    yours = {(f['box'], f['side']): (f['sector'] or None) for f in data['faces']}
    print('=== 면 목록 견주기: supervisor %d개 · perception %d개 · supervisor 에만 %s · perception 에만 %s · 구역이 다른 면 %s' % (
        len(mine), len(yours), sorted(set(mine) - set(yours)), sorted(set(yours) - set(mine)), sorted((k, mine[k], yours[k]) for k in set(mine) & set(yours) if mine[k] != yours[k])))
    print('=== 면이 속한 구역 — supervisor 가 정한 것:', dict(sorted(ours.items(), key=lambda kv: str(kv[0]))))
    print('                    perception 이 적은 것:', dict(sorted(theirs.items(), key=lambda kv: str(kv[0]))))
    print('    구역 (supervisor 가 쓰는 값):', {k: [round(v, 2) for v in r] for k, r in sorted(fmap.sectors.items())})
    print('    거점 (supervisor 가 쓰는 값):', {k: [round(v, 2) for v in r] for k, r in sorted(fmap.bases.items())})
except Exception as error:
    print('=== 구역 견주기를 하지 못함:', type(error).__name__, error)
PY
echo '=== 가짜 상대가 본 것'; grep -a -E 'FAKE_WORLD_SUMMARY' \${session}fake_world.log 2>&1 | cut -c1-600 | tail -1
echo '=== 프로그램별 기록 (줄 수 · Traceback 수)'; for f in \${session}*.log; do echo \"   \$(basename \$f) \$(wc -l < \$f)줄 · Traceback \$(grep -a -c Traceback \$f || true)\"; done
echo '=== 다시 켜진 프로그램'; grep -a -E '다시 켬|다시 켭|restart|죽음|끝남' /tmp/team/out.log | cut -c1-200 | head -6
echo '=== team_node 출력에서'; grep -a -E 'Traceback|Error|예외|하는 일|준비' /tmp/team/out.log | sed -e 's/^\[[A-Z]*\] \[[0-9.]*\] \[supervisor\]: //' | cut -c1-200 | head -12
sleep 1; echo \"=== 남은 프로세스: \$(ps -eo comm,args | awk '\$1 ~ /^python3/ && /fake_world.py|team_node.py|_node|navigation_slam|map_maker/' | wc -l)\"
if [ -d /out ]; then rm -rf /out/log && cp -r /tmp/team/log /out/log && cp /tmp/team/out.log /out/out.log && chmod -R a+rwX /out; fi
"
