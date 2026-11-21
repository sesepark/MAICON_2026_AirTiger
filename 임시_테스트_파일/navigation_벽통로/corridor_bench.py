"""2026-11-19 14:52 실차(연습 경기장 1, B-1)의 정찰 목표를 그 판 지도(6.12 × 4.80, 상자 14개)의 가상 경기장에서 차례로 다시 보낸다.
상자 2 · 3(위 벽 밑) · 12 · 13(아래 벽 위)과 벽 사이 0.6 m 통로의 정찰 자리(벽에서 0.18 · 0.23 m)에서 「돌 자리 없음」 · 시간 초과가 나던 것.

    python corridor_bench.py <navigation 패키지 경로> <A|B> [seed ...]
A = sim_world.REAL_ROBOT, B = REAL_ROBOT_1119 (실측 회전). 시한은 supervisor 와 같게 10 s + 거리 / 0.2 m/s.
"""
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(sys.argv[1]).resolve()
model = sys.argv[2]
seeds = [int(s) for s in sys.argv[3:]] or [1]
sys.path.insert(0, str(pkg / 'test'))
sys.path.insert(0, str(pkg / '시험기록' / 'web_drive'))
os.chdir(pkg / 'test')
import arena_default as AD  # noqa: E402
import sim_world as W  # noqa: E402

NAMES = {1: 'REACHED', 2: 'CANCELED', 3: 'BLOCKED', 4: 'TIMED_OUT', 6: 'INVALID'}
BOXES_1452 = ((1.425, 4.035), (1.856, 3.997), (4.27, 4.005), (4.666, 4.016), (3.054, 3.41), (1.448, 2.404), (1.852, 2.392),
              (4.249, 2.387), (4.632, 2.37), (3.041, 1.506), (1.465, 0.816), (1.871, 0.835), (4.27, 0.799), (4.638, 0.821))
# (번호, x, y, 방향°, 허용°) — supervisor/events.jsonl 의 go (normal) 그대로. 9 · 10 · 17 · 18 이 벽 통로
GOALS = [(1, 1.825, 3.425, -0.9, 36), (2, 2.425, 2.625, 42.9, 36), (3, 3.025, 2.625, -51.0, 10), (4, 3.625, 3.025, 92.2, 10),
         (6, 4.025, 3.225, -58.9, 24.3), (7, 3.025, 4.025, -1.1, 36), (8, 2.425, 4.425, -52.4, 36), (9, 3.825, 4.625, -43.3, 36),
         (10, 4.225, 4.625, -42.9, 36), (11, 5.225, 3.625, 132.6, 36), (12, 5.225, 2.825, -130.8, 36), (13, 5.225, 1.225, -133.7, 36),
         (14, 4.025, 1.625, -1.3, 10), (15, 3.625, 1.825, -140.2, 36), (16, 3.625, 1.625, -44.1, 36), (17, 4.225, 0.225, 43.7, 36),
         (18, 3.825, 0.225, 40.1, 36), (19, 3.025, 1.025, -12.2, 36), (20, 2.625, 0.825, 49.1, 36), (21, 1.825, 1.825, -17.4, 36)]
WALL = {9, 10, 17, 18}
ROBOT = W.REAL_ROBOT if model == 'A' else W.REAL_ROBOT_1119
H = 4.80
tot = dict(n=0, ok=0, wall_ok=0, t=0.0, wall_t=0.0, contacts=0)
for seed in seeds:
    world = W.World(6.12, H, 0.40, BOXES_1452, hidden=AD.pillars(6.12, H))
    region = AD.base_region('B-1', 6.12, H)
    robot = W.SimRobot(world, ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2, 0.0), odom_offset=1.0, slip=0.85, seed=seed, **ROBOT)
    import json as _json
    loop = W.ClosedLoop(world, robot, drive_params=_json.loads(os.environ.get('NAV_DRIVE_PARAMS', '{}')))   # 주행 파라미터를 바꿔 볼 때
    loop.start_from_slot('B-1', region)
    loop.run(2.0)
    for no, gx, gy, yaw, tol in GOALS:
        t0, c0 = loop.t, robot.collisions
        timeout = 10.0 + math.hypot(gx - robot.x, gy - robot.y) / 0.2
        res = loop.drive_to(gx, gy, limit_s=timeout + 3.0, yaw=math.radians(yaw), yaw_tol=math.radians(tol), timeout_s=timeout)
        e = math.degrees(abs(math.atan2(math.sin(robot.yaw - math.radians(yaw)), math.cos(robot.yaw - math.radians(yaw)))))
        d = math.hypot(robot.x - gx, robot.y - gy)
        out = NAMES.get(res.outcome, res.outcome) if res else None
        ok = out == 'REACHED' and d <= 0.27 and e <= tol + 0.5
        spent = loop.t - t0
        tot['n'] += 1; tot['ok'] += ok; tot['t'] += spent; tot['contacts'] += robot.collisions - c0
        if no in WALL:
            tot['wall_ok'] += ok; tot['wall_t'] += spent
        wall = min(robot.y, H - robot.y)
        print('%s seed %d #%2d %-9s %5.1f s  자리 %.2f m 방향 %3.0f°(허용 %2.0f)  벽까지 %.2f  닿음 %3d  %s' % (
            model, seed, no, out, spent, d, e, tol, wall, robot.collisions - c0, (res.detail if res else '')[:70]), flush=True)
print('TOTAL %s: 자리·방향까지 맞춘 목표 %d/%d · 벽 통로 4곳 × seed %d/%d · 걸린 시간 %.0f s (벽 통로 %.0f s) · 닿은 주기 %d' % (
    model, tot['ok'], tot['n'], tot['wall_ok'], 4 * len(seeds), tot['t'], tot['wall_t'], tot['contacts']))
