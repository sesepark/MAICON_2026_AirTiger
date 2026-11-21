"""서서 돌기 · 도착 방향 맞추기를 두 가지 로봇 흉내로 견준다 (2026-11-19 실차 회전 실측 뒤).

    python rot_bench.py <navigation 패키지 경로> <A|B> [seed ...]
A = sim_world.REAL_ROBOT (지금까지의 실차 흉내: 명령만큼 돈다), B = REAL_ROBOT_1119 (실측: 1.4 rad/s 밑은 안 돌고 명령의 0.35 배).
1118 2차 지도 · R-2 출발에서 상자 면 곁 24자리(가운데에서 면까지 0.19 · 0.22 m, 면과 나란한 방향) + 트인 자리 8곳(도착 방향을 90 · 180° 돌려야 함)을
한 판에 이어서 보낸다 (방향 허용 10°, supervisor 기본 0.17 rad). 목표마다 결과 · 걸린 시간 · 방향 맞추기 시간 · 남은 방향 · 닿은 주기.
"""
import collections
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(sys.argv[1]).resolve()
model = sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]] or [1]
sys.path.insert(0, str(pkg / 'test'))
os.chdir(pkg / 'test')
import test_team_test_replay as T  # noqa: E402

W = T.W
NAMES = {1: 'REACHED', 2: 'CANCELED', 3: 'BLOCKED', 4: 'TIMED_OUT'}
ROBOT = W.REAL_ROBOT if model == 'A' else W.REAL_ROBOT_1119
boxes = T.BOXES_1118_2
tot = collections.Counter()
for seed in seeds:
    world = W.World(boxes=boxes, hidden=W.PILLARS, **W.ARENA)
    robot = W.SimRobot(world, W.slot_start_pose('R-2'), odom_offset=1.0, slip=0.85, seed=seed, **ROBOT)
    loop = W.ClosedLoop(world, robot, drive_params={k: float(v) for k, v in (kv.split('=') for kv in os.environ.get('DRIVE_SET', '').split(',') if kv)})
    loop.start_from_slot('R-2', W.base_region('R-2'))
    loop.run(1.5)
    grid = loop.drive.grid
    spots = []
    for bx, by in boxes[:8]:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for d in (0.19, 0.22):
                x, y = bx + dx * (0.2 + d), by + dy * (0.2 + d)
                yaw = math.degrees(math.atan2(dx, -dy))
                if 0.4 < x < 5.7 and 0.4 < y < 4.6 and grid.spec.distance_to_structure(x, y) >= d - 0.005:
                    spots.append((round(x, 3), round(y, 3), round(yaw, 1)))
    spots = spots[:24]
    opens = [(3.9, 2.2, 90.0), (3.9, 2.2, -90.0), (2.4, 2.2, 180.0), (2.4, 2.2, 0.0), (2.5, 3.3, -90.0), (2.5, 3.3, 90.0), (5.0, 2.6, 0.0), (5.0, 2.6, 180.0)]
    opens = [s for s in opens if grid.spec.distance_to_structure(s[0], s[1]) > 0.35]
    phase_t = collections.Counter()
    step0 = loop.step

    def step():
        job = getattr(loop.drive, '_job', None)
        if job is not None and not job.done:
            phase_t[job.phase.split('+')[0]] += loop.DT
        step0()
    loop.step = step
    rows = []
    for x, y, yaw in spots + opens:
        r = loop.robot
        began, c0 = loop.t, r.collisions
        phase_t.clear()
        res = T.go(loop, x, y, yaw, yaw_tol=0.17, cap_s=40.0)
        err = T.heading_error_deg(loop, yaw)
        rows.append(dict(outcome=NAMES.get(res.outcome, res.outcome) if res else None, t=loop.t - began, align=phase_t['aligning'] + phase_t['settling'],
                         rot=phase_t['rotating'], err=err, contacts=r.collisions - c0, detail=(res.detail if res else '')[:60]))
    n = len(rows)
    reached = sum(o['outcome'] == 'REACHED' for o in rows)
    within = sum(o['err'] <= 10.5 for o in rows)
    line = dict(seed=seed, n=n, reached=reached, yaw_ok=within, t=sum(o['t'] for o in rows), align=sum(o['align'] for o in rows),
                rot=sum(o['rot'] for o in rows), contacts=sum(o['contacts'] for o in rows), max_err=max(o['err'] for o in rows))
    for k, v in line.items():
        if k not in ('seed', 'max_err'):
            tot[k] += v
    tot['max_err'] = max(tot['max_err'], line['max_err'])
    print('model %s seed %d: REACHED %d/%d · 방향 10° 안 %d · 걸린 시간 합 %.0f s · 맞추기(aligning+settling) %.0f s · 서서 돌기(rotating) %.0f s · 닿은 주기 %d · 남은 방향 최대 %.0f°'
          % (model, seed, reached, n, within, line['t'], line['align'], line['rot'], line['contacts'], line['max_err']), flush=True)
    for o in rows:
        if o['outcome'] != 'REACHED' or o['err'] > 10.5 or o['contacts']:
            print('   ', {k: (round(v, 1) if isinstance(v, float) else v) for k, v in o.items()})
print('TOTAL model %s: REACHED %d/%d · 방향 10° 안 %d · 시간 %.0f s · 맞추기 %.0f s · 서서 돌기 %.0f s · 닿은 주기 %d · 남은 방향 최대 %.0f°'
      % (model, tot['reached'], tot['n'], tot['yaw_ok'], tot['t'], tot['align'], tot['rot'], tot['contacts'], tot['max_err']))
