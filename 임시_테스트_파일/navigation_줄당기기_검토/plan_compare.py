"""4a5e3d0 전/후 grid.py 의 plan 비교 — 경로만 (차체는 안 돌린다).

    python plan_compare.py <navigation 패키지 경로> <4a5e3d0 이전 grid.py> [--pairs 300] [--seed 1]

지도: 2026-11-18 팀 테스트 1·2·3차 perception 지도의 상자 (supervisor/events.jsonl 의 map), 경기장 6.12 × 5.00, 상자 0.40.
한 판에 무작위 (출발, 목표) 쌍 — 둘 다 막힌 칸이 아닌 곳. 같은 쌍을 전 · 후 격자로 짠다.
잰 것: 계획 시간(ms), 길이, 경로 위 점에서 가장 가까운 벽 · 상자 면까지의 거리(가운데 기준, 최소),
그 거리가 0.20 m(차체 반 대각선 0.17 + 3 cm) 안으로 들어간 경로 비율, 꺾이는 점 수, 길 없음.
--extra 를 주면 무작위 임시 장애물(점 하나 = 반지름 obstacle_radius + grow 원)도 넣는다.
"""
import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('package')
ap.add_argument('old_grid')
ap.add_argument('--pairs', type=int, default=300)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--extra', type=int, default=0, help='판마다 임시 장애물 점 수')
ap.add_argument('--logs', default=None, help='로그_파일/1118_팀테스트_경기장1_R2 경로 (기본: 패키지에서 찾음)')
a = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(a.package).resolve()
sys.path.insert(0, str(pkg))
from navigation.drive.fieldmap import MapSpec  # noqa: E402
from navigation.drive import grid as new_mod  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402

spec_old = importlib.util.spec_from_file_location('grid_old', a.old_grid)
old_mod = importlib.util.module_from_spec(spec_old)
spec_old.loader.exec_module(old_mod)

logs = Path(a.logs) if a.logs else pkg.parents[2] / '로그_파일' / '1118_팀테스트_경기장1_R2'
maps = []
for run in ('1차_2211', '2차_2223', '3차_2339'):
    for line in open(logs / run / 'supervisor' / 'events.jsonl', encoding='utf-8'):
        e = json.loads(line)
        if e.get('e') == 'map':
            maps.append((run, tuple(tuple(b) for b in e['boxes'])))
            break
P = DriveParams()
HALF_DIAG = math.hypot(P.robot_half_length, P.robot_half_width)  # 0.170


def clearance_along(spec, pts, start):
    """경로(start → pts) 위 2.5 cm 마다 가장 가까운 벽 · 상자 면까지 거리의 최소."""
    best = float('inf')
    prev = start
    for q in pts:
        n = max(1, int(math.hypot(q[0] - prev[0], q[1] - prev[1]) / 0.025))
        for k in range(n + 1):
            t = k / n
            best = min(best, spec.distance_to_structure(prev[0] + (q[0] - prev[0]) * t, prev[1] + (q[1] - prev[1]) * t))
        prev = q
    return best


def length(start, pts):
    total, prev = 0.0, start
    for q in pts:
        total += math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
    return total


rng = random.Random(a.seed)
rows = {'old': [], 'new': []}
for run, boxes in maps:
    spec = MapSpec(6.12, 5.0, 0.4, boxes, 'arena')
    kw = dict(radius=P.robot_radius, margin=P.inflate_margin, cell=P.cell, soft_m=P.soft_m, soft_cost=P.soft_cost)
    t0 = time.perf_counter(); g_old = old_mod.Grid(spec, **kw); build_old = time.perf_counter() - t0
    t0 = time.perf_counter(); g_new = new_mod.Grid(spec, **kw); build_new = time.perf_counter() - t0
    for _ in range(a.pairs):
        while True:
            s = (rng.uniform(0.2, 5.92), rng.uniform(0.2, 4.8))
            g = (rng.uniform(0.2, 5.92), rng.uniform(0.2, 4.8))
            if g_new.free(*s) and g_new.free(*g) and math.hypot(g[0] - s[0], g[1] - s[1]) > 0.5:
                break
        extra = set()
        for _k in range(a.extra):
            ox, oy = rng.uniform(0.3, 5.8), rng.uniform(0.3, 4.7)
            if math.hypot(ox - s[0], oy - s[1]) < 0.5 or math.hypot(ox - g[0], oy - g[1]) < 0.5:
                continue
            extra.update(g_new.cells_in_disc(ox, oy, P.obstacle_radius + g_new.grow))
        for name, grid in (('old', g_old), ('new', g_new)):
            t0 = time.perf_counter()
            path = grid.plan(s, g, extra)
            dt = (time.perf_counter() - t0) * 1000.0
            if path is None:
                rows[name].append(dict(run=run, ms=dt, ok=False))
                continue
            rows[name].append(dict(run=run, ms=dt, ok=True, len=length(s, path), pts=len(path), clear=clearance_along(spec, path, s)))
    print('%s: 상자 %d개, 격자 만들기 전 %.0f ms · 후 %.0f ms' % (run, len(boxes), build_old * 1000, build_new * 1000))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


print('쌍 %d개 (판마다 %d), 임시 장애물 점 %d' % (len(rows['new']), a.pairs, a.extra))
print('%-4s %8s %8s %8s %8s | %7s %7s | %8s %8s %9s | %6s' % ('', 'ms 중앙', 'ms p95', 'ms 최대', '길없음', '길이 m', '점 수', '여유 최소', '여유 p5', '<0.20 m', ''))
for name in ('old', 'new'):
    r = rows[name]
    ok = [x for x in r if x['ok']]
    ms = [x['ms'] for x in r]
    print('%-4s %8.2f %8.2f %8.2f %8d | %7.3f %7.1f | %8.3f %8.3f %8.1f%% |' % (
        '전' if name == 'old' else '후', statistics.median(ms), pct(ms, 0.95), max(ms), len(r) - len(ok),
        statistics.mean(x['len'] for x in ok), statistics.mean(x['pts'] for x in ok),
        min(x['clear'] for x in ok), pct([x['clear'] for x in ok], 0.05), 100.0 * sum(x['clear'] < HALF_DIAG + 0.03 for x in ok) / len(ok)))
# 같은 쌍끼리
pair = [(o, n) for o, n in zip(rows['old'], rows['new']) if o['ok'] and n['ok']]
longer = [n['len'] / o['len'] - 1.0 for o, n in pair]
gain = [n['clear'] - o['clear'] for o, n in pair]
slower = [n['ms'] / max(o['ms'], 1e-6) for o, n in zip(rows['old'], rows['new'])]
print('같은 쌍: 길이 +%.1f%% (중앙) · 최대 +%.1f%%, 최소 여유 +%.3f m (중앙) · 줄어든 쌍 %d개 (가장 많이 %.3f m), 계획 시간 ×%.2f (중앙) · ×%.2f (최대)' % (
    100 * statistics.median(longer), 100 * max(longer), statistics.median(gain), sum(g < -1e-6 for g in gain), min(gain),
    statistics.median(slower), max(slower)))
print('길 있음/없음이 갈린 쌍: %d' % sum(o['ok'] != n['ok'] for o, n in zip(rows['old'], rows['new'])))
