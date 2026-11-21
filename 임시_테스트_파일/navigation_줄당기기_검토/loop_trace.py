"""loop_compare.py 의 한 판을 자세히: 목표마다 결과 · 닿은 자리 · 그때의 경로 · 단계.

    python loop_trace.py <navigation 패키지 경로> <4a5e3d0 이전 grid.py> old|new <지도 1차|2차|3차> <seed>
"""
import importlib.util
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(sys.argv[1]).resolve()
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
which, map_name, seed = sys.argv[3], sys.argv[4], int(sys.argv[5])
sys.path.insert(0, str(pkg / 'test'))
import test_team_test_replay as T  # noqa: E402

W = T.W
s = importlib.util.spec_from_file_location('grid_old', sys.argv[2])
old_mod = importlib.util.module_from_spec(s)
s.loader.exec_module(old_mod)
# loop_compare 의 지도 · 자리 고르기를 그대로 쓴다
sys.argv = [sys.argv[0], str(pkg), sys.argv[2], which, '--seeds', '1-0', '--maps', map_name]
import loop_compare as LC  # noqa: E402  (seeds 1-0 = 빈 범위 → 아무것도 안 돌린다)

boxes = LC.MAPS[map_name]
world = W.World(boxes=boxes, hidden=W.PILLARS, **W.ARENA)
robot = W.SimRobot(world, W.slot_start_pose('R-2'), odom_offset=1.0, slip=0.85, seed=seed, **W.REAL_ROBOT)
loop = W.ClosedLoop(world, robot)
mod = loop.drive_module
if which == 'old':
    mod.Grid = old_mod.Grid
events = []
step = loop.step
last = {'contacts': 0}


def traced():
    step()
    r = loop.robot
    n = r.collisions - r.hidden_contacts
    if n != last['contacts']:
        job = loop.drive._job
        events.append((loop.t, round(r.x, 3), round(r.y, 3), round(math.degrees(r.yaw), 1), loop.cmd,
                       job.phase if job else None, [tuple(round(v, 2) for v in q) for q in (job._path or [])][:6] if job else None))
        last['contacts'] = n


loop.step = traced
loop.start_from_slot('R-2', W.base_region('R-2'))
loop.run(1.5)
targets = LC.SPOTS_2 if map_name == '2차' else LC.spots_for(boxes, seed, loop.drive.grid)
print('상자', boxes)
print('자리', targets)
for x, y in targets:
    began = loop.t
    n0 = len(events)
    res = T.go(loop, x, y, cap_s=40.0)
    r = loop.robot
    print('→ (%.3f, %.3f): %s %s · %.1f s · 끝 (%.3f, %.3f, %.0f°) · 닿은 주기 %d' % (
        x, y, T.W and {1: 'REACHED', 3: 'BLOCKED', 4: 'TIMED_OUT'}.get(res.outcome, res.outcome) if res else None, res.detail if res else '',
        loop.t - began, r.x, r.y, math.degrees(r.yaw), len(events) - n0))
    for e in events[n0:n0 + 4]:
        print('     닿음 t=%.2f 자리 (%.3f, %.3f, %.1f°) 명령 %s 단계 %s 경로 %s' % (e[0] - began, *e[1:]))
    if len(events) - n0 > 4:
        e = events[-1]
        print('     … 마지막 t=%.2f 자리 (%.3f, %.3f, %.1f°) 명령 %s 단계 %s 경로 %s' % (e[0] - began, *e[1:]))
