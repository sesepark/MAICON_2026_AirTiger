"""4a5e3d0 전/후 grid.py 의 plan 을 몇 가지 정해 둔 장면에서 견준다 (경로 점 · 여유 · 시간 · 함수 호출 수).

    python plan_cases.py <navigation 패키지 경로> <4a5e3d0 이전 grid.py>
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(pkg))
from navigation.drive.fieldmap import MapSpec  # noqa: E402
from navigation.drive import grid as new_mod  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402

s = importlib.util.spec_from_file_location('grid_old', sys.argv[2])
old_mod = importlib.util.module_from_spec(s)
s.loader.exec_module(old_mod)
P = DriveParams()
KW = dict(radius=P.robot_radius, margin=P.inflate_margin, cell=P.cell, soft_m=P.soft_m, soft_cost=P.soft_cost)
logs = pkg.parents[2] / '로그_파일' / '1118_팀테스트_경기장1_R2'


def boxes_of(run):
    for line in open(logs / run / 'supervisor' / 'events.jsonl', encoding='utf-8'):
        e = json.loads(line)
        if e.get('e') == 'map':
            return tuple(tuple(b) for b in e['boxes'])


def clearance(spec, start, pts):
    best, where, prev = float('inf'), None, start
    for q in pts:
        n = max(1, int(math.hypot(q[0] - prev[0], q[1] - prev[1]) / 0.02))
        for k in range(n + 1):
            t = k / n
            x, y = prev[0] + (q[0] - prev[0]) * t, prev[1] + (q[1] - prev[1]) * t
            d = spec.distance_to_structure(x, y)
            if d < best:
                best, where = d, (round(x, 2), round(y, 2))
        prev = q
    return best, where


def interior_clearance(spec, start, pts, skip=0.30):
    """출발 · 도착에서 skip 안을 뺀 경로의 최소 여유 (출발 · 목표 자리가 원래 벽 가까이인 경우를 빼고 줄 당기기만 보려고)."""
    goal = pts[-1]
    best, prev = float('inf'), start
    for q in pts:
        n = max(1, int(math.hypot(q[0] - prev[0], q[1] - prev[1]) / 0.02))
        for k in range(n + 1):
            t = k / n
            x, y = prev[0] + (q[0] - prev[0]) * t, prev[1] + (q[1] - prev[1]) * t
            if math.hypot(x - start[0], y - start[1]) < skip or math.hypot(x - goal[0], y - goal[1]) < skip:
                continue
            best = min(best, spec.distance_to_structure(x, y))
        prev = q
    return best


def count_calls(grid):
    counts = {'segment_free': 0, '_segment_roomy': 0}
    for name in counts:
        if hasattr(grid, name):
            f = getattr(grid, name)

            def wrap(*a, _f=f, _n=name, **k):
                counts[_n] += 1
                return _f(*a, **k)
            setattr(grid, name, wrap)
    return counts


def show(title, spec, start, goal, extra_pts=()):
    print('■ %s  출발 %s → 목표 %s%s' % (title, start, goal, '  임시 장애물 %s' % (list(extra_pts),) if extra_pts else ''))
    for name, mod in (('전', old_mod), ('후', new_mod)):
        g = mod.Grid(spec, **KW)
        extra = set()
        for x, y in extra_pts:
            extra.update(g.cells_in_disc(x, y, P.obstacle_radius + g.grow))
        counts = count_calls(g)
        t0 = time.perf_counter()
        path = g.plan(start, goal, extra)
        ms = (time.perf_counter() - t0) * 1000
        if path is None:
            print('   %s: 길 없음 (%.1f ms)' % (name, ms))
            continue
        c, where = clearance(spec, start, path)
        ci = interior_clearance(spec, start, path)
        ln = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip([start] + path[:-1], path))
        ext = ''
        if extra_pts:
            dmin = min(math.hypot(x - ox, y - oy) for p, q in zip([start] + path[:-1], path)
                       for k in range(21) for x, y in [(p[0] + (q[0] - p[0]) * k / 20, p[1] + (q[1] - p[1]) * k / 20)] for ox, oy in extra_pts)
            ext = ' · 임시 장애물 점까지 최소 %.3f m' % dmin
        print('   %s: 점 %2d · 길이 %.2f m · 여유 최소 %.3f m (%s) · 가운데 구간 %.3f m%s · %.1f ms · segment_free %d · roomy %d' % (
            name, len(path), ln, c, where, ci, ext, ms, counts['segment_free'], counts['_segment_roomy']))
        print('      ' + ' '.join('(%.2f,%.2f)' % q for q in path[:14]) + (' …' if len(path) > 14 else ''))


b2 = boxes_of('2차_2223')
spec2 = MapSpec(6.12, 5.0, 0.4, b2, 'arena')
# 1. 2차 판 스침 자리: (2.825, 2.825) → (4.025, 3.625) → (2.625, 3.425) 정찰 순서 중
show('2차 지도: 정찰 자리 사이 (상자 (3.07, 3.75) 옆)', spec2, (4.025, 3.625), (2.625, 3.425))
show('2차 지도: 출발 R-2 → 첫 정찰 자리', spec2, (5.92, 2.5), (4.225, 2.225))
# 2. M5 돌진: 거점 입구 대기 자리 → B-1 안 목표의 진입점
g = new_mod.Grid(spec2, **KW)
entry = g.nearest_free(0.175, 3.817, max_dist=0.6)
show('2차 지도: M5 대기 자리 (0.42, 2.50) → B-1 돌진 진입점 %s' % (tuple(round(v, 3) for v in entry),), spec2, (0.42, 2.50), entry)
# 3. 출발이 상자에 바짝 (tight · 막힌 칸 안)
bx, by = b2[3]
show('2차 지도: 상자 %s 면 0.12 m 앞에서 출발' % ((bx, by),), spec2, (bx + 0.2 + 0.12, by), (5.5, 4.5))
# 4. 40 cm 틈 (test_drive_core 의 test_40cm_gap 과 같은 꼴): 가운데 간격 0.8 m 상자 두 개 + 벽처럼 늘어선 상자
wall = tuple((x * 0.4 + 0.2, 2.5) for x in range(15) if x not in (7, 8)) + ((3.2 - 0.2 - 0.4, 2.5),)
gap_boxes = tuple(b for b in wall)
spec_gap = MapSpec(6.12, 5.0, 0.4, ((2.6, 2.5), (3.4, 2.5)) + tuple((x, 2.5) for x in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8)), 'arena')
show('40 cm 틈 (상자 가운데 2.6 · 3.4, 틈 2.8 ~ 3.2)', spec_gap, (2.0, 1.5), (4.0, 3.5))
# 5. 임시 장애물(다른 UGV 점 몇 개) 옆을 지나감 — tight 는 지도 상자 · 벽만, 임시 장애물에는 없다
show('2차 지도: 지도에 없는 물체(점 3개) 옆', spec2, (5.0, 3.0), (2.4, 3.0), extra_pts=((3.9, 3.05), (3.95, 3.0), (3.9, 2.95)))
