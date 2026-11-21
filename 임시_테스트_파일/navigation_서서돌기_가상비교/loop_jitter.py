"""4a5e3d0 전/후 grid.py 로 가상 경기장(실차 흉내)을 돌려 견준다 — SLAM · 주행 노드 그대로, 격자 class 만 바꾼다.

    python loop_compare.py <navigation 패키지 경로> <4a5e3d0 이전 grid.py> old|new [--seeds 1-8] [--maps 2차] [--out 파일]

- 지도 · 세계: 2026-11-18 팀 테스트 1·2·3차 perception 지도의 상자 (= 실제 상자), 지도에 없는 고정 기둥 10개 (sim_world.PILLARS)
- 차체: REAL_ROBOT (지연 0.30 s, 1차 지연 0.20 s, 서서 돌기 마찰 0.45 s), slip 0.85, 출발 R-2
- 목표: 2차는 그 판의 정찰 자리 7곳(test_team_test_replay 의 스침 시험과 같다), 1·3차는 상자 면에서 0.7 ~ 1.0 m 떨어진 무작위 자리 7곳.
  방향은 주지 않는다 (방향 맞추기 멈춤과 섞이지 않게). 시한 min(40, 15 + 1 m 에 10 s)
- 잰 것: 닿은 주기(지도 상자 · 벽 / 기둥), 차체에서 지도 상자까지 가장 가까운 거리, 도착 수, 걸린 시간, 계획 시간
"""
import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('package')
ap.add_argument('old_grid')
ap.add_argument('which', choices=('old', 'new'))
ap.add_argument('--seeds', default='1-8')
ap.add_argument('--maps', default='1차,2차,3차')
ap.add_argument('--out')
ap.add_argument('--jitter', type=float, default=0.06)  # 실제 상자를 지도보다 ±이만큼 어긋나게 (드론 지도 오차 흉내)
a = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(a.package).resolve()
sys.path.insert(0, str(pkg / 'test'))
import test_team_test_replay as T  # noqa: E402

W = T.W
spec_old = importlib.util.spec_from_file_location('grid_old', a.old_grid)
old_mod = importlib.util.module_from_spec(spec_old)
spec_old.loader.exec_module(old_mod)

logs = pkg.parents[2] / '로그_파일' / '1118_팀테스트_경기장1_R2'
MAPS = {}
for run in ('1차_2211', '2차_2223', '3차_2339'):
    for line in open(logs / run / 'supervisor' / 'events.jsonl', encoding='utf-8'):
        e = json.loads(line)
        if e.get('e') == 'map':
            MAPS[run[:2]] = tuple(tuple(b) for b in e['boxes'])
            break
SPOTS_2 = ((4.225, 2.225), (4.025, 1.825), (3.625, 1.025), (2.625, 2.625), (2.825, 2.825), (4.025, 3.625), (2.625, 3.425))


def spots_for(boxes, seed, grid):
    rng = random.Random(1000 + seed)
    out = []
    while len(out) < 7:
        bx, by = rng.choice(boxes)
        side = rng.randrange(4)
        d = 0.2 + rng.uniform(0.7, 1.0)
        dx, dy = ((1, 0), (-1, 0), (0, 1), (0, -1))[side]
        off = rng.uniform(-0.15, 0.15)
        x, y = bx + dx * d + dy * off, by + dy * d + dx * off
        if 0.35 < x < 6.12 - 0.35 and 0.35 < y < 5.0 - 0.35 and grid.free(x, y) and all(math.hypot(x - px, y - py) > 0.25 for px, py in W.PILLARS):
            out.append((round(x, 3), round(y, 3)))
    return out


def seeds(text):
    lo, _, hi = text.partition('-')
    return range(int(lo), int(hi or lo) + 1)


results = []
for map_name in a.maps.split(','):
    boxes = MAPS[map_name]
    for seed in seeds(a.seeds):
        jr = random.Random(500 + seed)
        truth = tuple((round(x + jr.uniform(-a.jitter, a.jitter), 3), round(y + jr.uniform(-a.jitter, a.jitter), 3)) for x, y in boxes)
        world = W.World(boxes=truth, hidden=W.PILLARS, **W.ARENA)
        robot = W.SimRobot(world, W.slot_start_pose('R-2'), odom_offset=1.0, slip=0.85, seed=seed, **W.REAL_ROBOT)
        loop = W.ClosedLoop(world, robot)
        _pm = loop.publish_map
        loop.publish_map = lambda bases=(), self_pose=None, others=(), boxes=None, _pm=_pm, _b=boxes: _pm(bases, self_pose, others, _b if boxes is None else boxes)
        mod = loop.drive_module
        plan_ms = []
        base = old_mod.Grid if a.which == 'old' else mod.Grid

        class Timed(base):
            def plan(self, *args, **kw):
                t0 = time.perf_counter()
                try:
                    return super().plan(*args, **kw)
                finally:
                    plan_ms.append((time.perf_counter() - t0) * 1000.0)

        mod.Grid = Timed
        stats = dict(box_min=float('inf'))
        step = loop.step

        def counted():
            step()
            r = loop.robot
            stats['box_min'] = min(stats['box_min'], world.body_clearance(r.x, r.y, r.yaw, walls=False, items=list(world.boxes)))

        loop.step = counted
        loop.start_from_slot('R-2', W.base_region('R-2'))
        loop.run(1.5)
        targets = SPOTS_2 if map_name == '2차' else spots_for(boxes, seed, loop.drive.grid)
        began, reached = loop.t, 0
        for x, y in targets:
            res = T.go(loop, x, y, cap_s=40.0)
            reached += int(res is not None and res.outcome == T.REACHED)
        r = loop.robot
        row = dict(map=map_name, seed=seed, which=a.which, reached=reached, n=len(targets), time=loop.t - began,
                   contacts=r.collisions - r.hidden_contacts, hidden=r.hidden_contacts, box_min=stats['box_min'],
                   body_min=r.min_body_clearance, plans=len(plan_ms), plan_med=sorted(plan_ms)[len(plan_ms) // 2] if plan_ms else 0.0,
                   plan_max=max(plan_ms) if plan_ms else 0.0)
        results.append(row)
        print('%s seed %2d %s: 도착 %d/%d · %5.1f s · 닿음 지도 %d · 기둥 %d · 상자까지 최소 %.3f m · 계획 %d번 중앙 %.1f ms 최대 %.1f ms' % (
            map_name, seed, a.which, reached, len(targets), row['time'], row['contacts'], row['hidden'], row['box_min'],
            row['plans'], row['plan_med'], row['plan_max']), flush=True)
if a.out:
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
