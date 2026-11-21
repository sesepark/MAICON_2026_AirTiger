"""경기 기록의 스캔(약 1 Hz) · 자세 · 지도를 주행 노드의 임시 장애물(ObstacleTracker)에 다시 넣고, 목표마다 출발 순간의 경로를 견준다.

  전체  : 임시 장애물 전부 (지금 스캔에 찍힌 것 + 기억으로만 남은 것) 를 막고 짠 길
  본것만: 지금 스캔에 찍힌 것(obstacle_ttl_s 안)만 막고 짠 길
  없이  : 임시 장애물 없이 (지도만)

기억으로만 남은 칸 때문에 크게 돌아가는 목표를 찾는다 (2026-11-19 17:57 본선 경기장: 1 m 앞 목표를 3.5 m 돌아가다 시한 초과).
실제 노드는 스캔을 10 Hz 로 받으므로 여기서 만든 기억은 실제보다 적다 (재생에서 돌아가면 실제로도 돌아간다).

쓰기:  python replay_obstacles.py <기록 폴더> [--set 이름=값 ...]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
from navigation.drive.fieldmap import MapSpec, fixed_pillars  # noqa: E402
from navigation.drive.geometry import Pose2D, path_length  # noqa: E402
from navigation.drive.grid import Grid  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker, scan_to_body  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402


def load(folder):
    events = []
    with open(os.path.join(folder, 'navigation', 'debug.jsonl'), encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('event') in ('map', 'goal', 'result'):
                events.append(r)
    with open(os.path.join(folder, 'navigation_slam', 'debug.jsonl'), encoding='utf-8') as f:
        for line in f:
            if '"scan_raw"' not in line and '"lidar_boxes"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('event') == 'lidar_boxes':
                events.append(r)
            elif r.get('pose'):
                r['event'] = 'scan'
                events.append(r)
    events.sort(key=lambda r: r['t'])
    return events


def run(folder, params=None, quiet=False):
    p = params or DriveParams()
    tracker = ObstacleTracker(p)
    grid = spec = None
    results = {}
    rows = []
    t0 = None
    lidar_boxes = []
    counts = []
    for r in load(folder):
        kind = r['event']
        if kind == 'lidar_boxes':
            lidar_boxes = [tuple(b) for b in r['boxes']]
            continue
        if kind == 'map':
            w, h = r['size']
            boxes = tuple(tuple(b) for b in r['boxes'])
            spec = MapSpec(w, h, r['box_size'], boxes, 'arena', tuple(fixed_pillars(w, h, r['box_size'])) if p.fixed_pillars else ())
            grid = Grid(spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
        elif kind == 'scan' and grid is not None:
            t0 = t0 if t0 is not None else r['t']
            ranges = np.array([x if x else np.nan for x in r['ranges']], dtype=float)
            pose = Pose2D(*r['pose'])
            body = scan_to_body(ranges, r['angle_min'], r['angle_increment'], p)
            if hasattr(tracker, 'on_lidar_boxes'):           # 라이다 상자를 길 찾기에 쓰는 시제품 (아직 main 에 없다)
                tracker.on_lidar_boxes(r['t'], lidar_boxes)
            tracker.on_scan(r['t'], pose, body, spec, grid, (r['angle_min'], r['angle_increment'], ranges))
            counts.append(len(tracker.extra_cells(r['t'], grid)))
        elif kind == 'result':
            results[r['no']] = r
        elif kind == 'goal' and grid is not None and r.get('pose'):
            start, goal = tuple(r['pose'][:2]), tuple(r['goal'][:2])
            target = grid.nearest_free(*goal) or goal
            full = tracker.extra_cells(r['t'], grid)
            fresh = tracker.fresh_cells(r['t'], grid) if hasattr(tracker, 'fresh_cells') else full
            lengths = []
            tight = tracker.tight_cells(r['t'], grid) if hasattr(tracker, 'tight_cells') else None
            try:
                safe = tracker.fresh_cells(r['t'], grid, Pose2D(*r['pose']))   # 차체 뒤쪽의 기억은 남긴 것
            except TypeError:
                safe = fresh
            for cells in (full, fresh, ()) + ((tight,) if tight is not None else ()):
                path = grid.plan(start, target, cells)
                lengths.append(None if path is None else round(path_length(start, path), 2))
            path = grid.plan(start, target, safe)
            safe_len = None if path is None else round(path_length(start, path), 2)
            rows.append((r['no'], round(r['t'] - (t0 or r['t']), 1), start, goal, len(full), len(fresh), lengths, safe_len))
    if not quiet:
        print('목표  t(s)   출발 → 목표                         막은 칸(전체/본것만)   길 길이 m (전체 · 본것만 · 없이)   실제 결과')
        for no, t, start, goal, n_full, n_fresh, lengths, safe_len in rows:
            res = results.get(no, {})
            flag = ''
            if lengths[0] is None and lengths[1] is not None:
                flag = '  ← 기억 때문에 길 없음'
            elif lengths[0] and lengths[1] and lengths[0] > 1.5 * lengths[1] + 0.5:
                flag = '  ← 기억 때문에 돌아감 (뒤쪽 기억은 남기면 %s m%s)' % (safe_len, ' — 옵션이 줄여 줌' if safe_len and lengths[0] > 1.5 * safe_len + 0.5 else ' — 뒤쪽 기억이라 그대로')
            print('%3d %6.1f  (%.2f, %.2f) → (%.2f, %.2f)   %5d / %5d   %s · %s · %s [좁게 %s]   %s %s%s' % (
                no, t, start[0], start[1], goal[0], goal[1], n_full, n_fresh, lengths[0], lengths[1], lengths[2], lengths[3] if len(lengths) > 3 else '-',
                res.get('outcome', '?'), res.get('elapsed', ''), flag))
    if not quiet and counts:
        none = sum(1 for row in rows if row[6][0] is None)
        print('막은 칸 (스캔마다): 가운데값 %d · 가장 많을 때 %d   |  출발 순간 길이 없는 목표 %d / %d' % (int(np.median(counts)), max(counts), none, len(rows)))
    return rows, results


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    params = DriveParams()
    for item in a.set:
        k, v = item.split('=')
        old = getattr(params, k)
        setattr(params, k, v.lower() in ('1', 'true') if isinstance(old, bool) else type(old)(float(v)))
    run(a.folder, params)
