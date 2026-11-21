"""상자 면 곁(가운데에서 면까지 d)의 자리에 와서 면과 나란한 방향으로 서라는 목표 — 도착 방향 맞추기가 상자에 걸리는가.

    python align_near.py <navigation 패키지 경로>
2차 지도, R-2 출발, seed 1. 상자 6개 × 면 4개 중 쓸 수 있는 자리, d = 0.19 / 0.22 m, 방향 = 면과 나란히(+90°).
"""
import json
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(pkg / 'test'))
os.chdir(pkg / 'test')
import test_team_test_replay as T  # noqa: E402

W = T.W
logs = pkg.parents[2] / '로그_파일' / '1118_팀테스트_경기장1_R2' / '2차_2223' / 'supervisor' / 'events.jsonl'
boxes = next(tuple(tuple(b) for b in e['boxes']) for e in map(json.loads, open(logs, encoding='utf-8')) if e.get('e') == 'map')
NAMES = {1: 'REACHED', 2: 'CANCELED', 3: 'BLOCKED', 4: 'TIMED_OUT'}

world = W.World(boxes=boxes, hidden=W.PILLARS, **W.ARENA)
robot = W.SimRobot(world, W.slot_start_pose('R-2'), odom_offset=1.0, slip=0.85, seed=1, **W.REAL_ROBOT)
loop = W.ClosedLoop(world, robot)
loop.start_from_slot('R-2', W.base_region('R-2'))
loop.run(1.5)
grid = loop.drive.grid
spots = []
for bx, by in boxes[:8]:
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for d in (0.19, 0.22):
            x, y = bx + dx * (0.2 + d), by + dy * (0.2 + d)
            yaw = math.degrees(math.atan2(dx, -dy))   # 면과 나란히
            if 0.4 < x < 5.7 and 0.4 < y < 4.6 and grid.spec.distance_to_structure(x, y) >= d - 0.005:
                spots.append((round(x, 3), round(y, 3), round(yaw, 1), d))
rows = []
for x, y, yaw, d in spots[:24]:
    r = loop.robot
    began, c0, pinned = loop.t, r.collisions, 0
    step = loop.step

    def counted(step=step):
        global pinned
        k = r.collisions
        step()
        v, w = loop.drive._last_cmd
        if abs(v) < 1e-6 and abs(w) > 1e-6 and r.collisions > k:
            pinned += 1
    loop.step = counted
    res = T.go(loop, x, y, yaw, yaw_tol=0.10, cap_s=40.0)
    loop.step = step
    out = dict(spot=(x, y, yaw), d=d, outcome=NAMES.get(res.outcome, res.outcome) if res else None, t=round(loop.t - began, 1),
               pinned=pinned, contacts=r.collisions - c0, detail=(res.detail if res else '')[:50])
    rows.append(out)
    print(out, flush=True)
n = len(rows)
print('REACHED %d/%d (방향 못 맞춤 detail %d) · BLOCKED %d · TIMED_OUT %d · 평균 %.1f s · 못 돈 주기 합 %d · 닿은 주기 합 %d' % (
    sum(o['outcome'] == 'REACHED' for o in rows), n, sum('돌 자리' in o['detail'] or '끝냄' in o['detail'] for o in rows),
    sum(o['outcome'] == 'BLOCKED' for o in rows), sum(o['outcome'] == 'TIMED_OUT' for o in rows),
    sum(o['t'] for o in rows) / n, sum(o['pinned'] for o in rows), sum(o['contacts'] for o in rows)))
