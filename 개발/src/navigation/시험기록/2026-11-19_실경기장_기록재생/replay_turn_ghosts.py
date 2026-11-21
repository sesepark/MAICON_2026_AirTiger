"""경기 기록의 「지도에 없는 점」 을 실제 ObstacleTracker 에 다시 넣어, 빨리 도는 동안의 헛점이 길 찾기를 얼마나 흔드는지 전 · 후로 견준다.

넣는 것: navigation/debug.jsonl state (2 Hz) 의 자세 · 지도에 없는 점(300개까지) — 그 자세로 차체 좌표로 되돌려 on_scan 에 준다.
        도는 빠르기는 앞 state 와의 방향 차이로. 광선(「비었다」 를 보는 데 쓴다)은 navigation_slam/debug.jsonl scan_raw (1 Hz) 가운데 0.3 s 안의 것.
보는 것: state 마다 그때의 임시 장애물로 지금 자리 → 목표의 길을 다시 짜서
        · 한 목표 안에서 0.5 s 사이에 길이가 0.8 m 넘게 바뀐 횟수 (실차에서 도는 쪽이 뒤집히던 때)
        · 임시 장애물을 넣으면 길이 없던 표본 수 · 막는 칸 · 기억으로만 남은 칸
전 = scan_turn_skip_rate 0 · memory_min_seen_s 0 (2026-11-19 밤 전의 규칙),  후 = 지금 기본값.

한계: 기록은 2 Hz (실차 10 Hz) 이고, 차체가 움직인 길은 그 판 그대로다 (길이 덜 바뀌었으면 덜 돌았을 것까지는 못 본다).
      옛 판은 그때의 코드로 돈 기록이다 — 점은 지금 규칙(차체 점 · 외벽 곁 · 지도 설명)으로 다시 거르지만, 그때의 SLAM 위치와 드론 지도는 그대로 쓴다.
      위치가 틀렸던 판(16:55 · 18:06)과 서서 돌지 못하던 판(17:05 ~ 17:38)의 숫자는 그 원인이 섞인 것이다.

쓰기:  python replay_turn_ghosts.py <기록 폴더>… [-v]
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
from navigation.drive.fieldmap import MapSpec, fixed_pillars  # noqa: E402
from navigation.drive.geometry import Pose2D, path_length, wrap  # noqa: E402
from navigation.drive.grid import Grid  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402


def _path(folder, node):
    for name in (os.path.join(folder, node, 'debug.jsonl'), os.path.join(folder, node + '_debug.jsonl')):
        if os.path.exists(name):
            return name
    raise FileNotFoundError('%s 의 %s 기록이 없다' % (folder, node))


def spec_of(m):
    import inspect
    fields = inspect.signature(MapSpec).parameters
    kw = dict(width=m['size'][0], height=m['size'][1], box_size=m['box_size'], boxes=tuple(tuple(b) for b in m['boxes']), frame_id='arena')
    if 'pillars' in fields:
        kw['pillars'] = tuple(fixed_pillars(m['size'][0], m['size'][1]))
    return MapSpec(**{k: v for k, v in kw.items() if k in fields})


def load(folder):
    nav = [json.loads(line) for line in open(_path(folder, 'navigation'), encoding='utf-8')]
    scans = []
    with open(_path(folder, 'navigation_slam'), encoding='utf-8') as f:
        for line in f:
            if '"scan_raw"' in line:
                r = json.loads(line)
                scans.append((r['t'], r['angle_min'], r['angle_increment'], np.array([x if x else np.nan for x in r['ranges']], dtype=float)))
    return nav, scans


def replay(nav, scans, params):
    maps = [r for r in nav if r.get('event') == 'map']
    spec = spec_of(maps[0])
    grid = Grid(spec, radius=params.robot_radius, margin=params.inflate_margin, cell=params.cell, soft_m=params.soft_m, soft_cost=params.soft_cost)
    tracker = ObstacleTracker(params)
    out = dict(flips=0, samples=0, no_way=0, cells=[], remembered=[], flip_at=[])
    start = next((r['t'] for r in nav if r.get('event') == 'goal'), 0.0)
    prev = None        # 앞 state
    last = {}          # 목표 번호 → 마지막 길이
    goal_no = None
    k = 0
    for r in nav:
        if r.get('event') == 'goal':
            goal_no = r['no']
            continue
        if r.get('event') == 'map' and r is not maps[0]:
            spec = spec_of(r)
            grid = Grid(spec, radius=params.robot_radius, margin=params.inflate_margin, cell=params.cell, soft_m=params.soft_m, soft_cost=params.soft_cost)
            continue
        if r.get('event') != 'state' or not r.get('pose'):
            continue
        t, pose = r['t'], Pose2D(*r['pose'])
        rate = 0.0
        if prev is not None and 0.0 < t - prev['t'] <= 0.8:
            rate = wrap(pose.yaw - prev['pose'][2]) / (t - prev['t'])
        prev = r
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        body = [((x - pose.x) * c + (y - pose.y) * s, -(x - pose.x) * s + (y - pose.y) * c) for x, y in (r.get('unexplained') or [])]
        # 옛 판의 점은 그때의 규칙으로 고른 것이다 → 지금 규칙으로 다시 거른다: 차체 사각형 + scan_self_margin_m 안의 점(우리 몸 — 16:55 ~ 17:38 판)은 뺀다.
        # 외벽 곁의 점(wall_explain_m)과 지도로 설명되는 점은 on_scan 이 지금 값으로 다시 거른다. 그때의 SLAM 위치 · 드론 지도가 틀렸던 것은 그대로 남는다
        margin = params.scan_self_margin_m
        if margin > 0.0:
            body = [(bx, by) for bx, by in body if abs(bx) > params.robot_half_length + margin or abs(by) > params.robot_half_width + margin]
        while k + 1 < len(scans) and scans[k + 1][0] <= t + 0.3:
            k += 1
        rays = None
        if scans and abs(scans[k][0] - t) <= 0.3:
            rays = (scans[k][1], scans[k][2], scans[k][3])
        tracker.on_scan(t, pose, body, spec, grid, rays, turn_rate=rate)
        job = r.get('job')
        if not job or job.get('mode') == 'straight' or not job.get('target'):
            continue
        extra = tracker.extra_cells(t, grid)
        target = tuple(job['target'])
        path = grid.plan((pose.x, pose.y), target, extra)
        if path is None:
            out['no_way'] += 1
            path = grid.plan((pose.x, pose.y), target)
        out['samples'] += 1
        out['cells'].append(len(extra))
        out['remembered'].append(tracker.memory_summary(t)['remembered'])
        if path is None:
            continue
        length = path_length((pose.x, pose.y), path)
        before = last.get(goal_no)
        if before is not None and t - before[0] <= 0.8 and abs(length - before[1]) > 0.8:
            out['flips'] += 1
            out['flip_at'].append((t - start, goal_no, before[1], length))
        last[goal_no] = (t, length)
    out['skipped'] = tracker.skipped_turning
    return out


def run(folder, verbose=False):
    nav, scans = load(folder)
    old = DriveParams()
    old.scan_turn_skip_rate = 0.0
    old.memory_min_seen_s = 0.0
    logged = 0   # 그 판이 실제로 낸 값 (기록의 남은 길이)
    prev = {}
    no = None
    for r in nav:
        if r.get('event') == 'goal':
            no = r['no']
        elif r.get('event') == 'state' and r.get('job') and r['job'].get('remaining') is not None:
            b = prev.get(no)
            if b is not None and r['t'] - b[0] <= 0.8 and abs(r['job']['remaining'] - b[1]) > 0.8:
                logged += 1
            prev[no] = (r['t'], r['job']['remaining'])
    print('== %s   (그 판의 기록에서 길이 0.8 m 넘게 바뀐 것 %d번)' % (os.path.basename(os.path.normpath(folder)), logged))
    for name, params in (('전', old), ('후', DriveParams())):
        o = replay(nav, scans, params)
        cells, mem = sorted(o['cells']), sorted(o['remembered'])
        print('   %s: 길이 바뀜 %2d번 · 임시 장애물을 넣으면 길이 없던 표본 %3d/%d · 막는 칸 가운데 %4d · 최대 %4d · 기억으로만 남은 칸 최대 %4d%s' % (
            name, o['flips'], o['no_way'], o['samples'], cells[len(cells) // 2] if cells else 0, cells[-1] if cells else 0, mem[-1] if mem else 0,
            ' · 도는 동안이라 안 만든 칸 %d' % o['skipped'] if o['skipped'] else ''))
        if verbose:
            for f in o['flip_at']:
                print('        t%+6.1f no%-2d %.2f → %.2f m' % f)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for folder in [a for a in sys.argv[1:] if not a.startswith('-')]:
        run(folder, '-v' in sys.argv)


if __name__ == '__main__':
    main()
