"""경기 기록(navigation_slam/debug.jsonl 의 scan_raw · navigation/debug.jsonl 의 map)을 SLAM 정합 + 지도 고치기에 다시 넣는다.

경기 기록에는 /odom 이 없고 스캔이 약 1 Hz 뿐이라 SLAM 노드를 그대로 돌릴 수는 없다. 그래서 스캔마다
  1. 기록된 자세(와 바로 앞 스캔에서 얻은 자세) 둘레 ±search m 를 0.1 m 간격으로 훑어 가장 잘 맞는 자리를 고르고 (score_pose — 정합과 같은 점수)
  2. scan_match 로 다듬은 자세로
  3. MapFusion.observe 를 부른다 (정합에 쓰는 상자 · 무게는 그때의 fusion.structure()).
지도의 세로를 바꿔 볼 수 있다 (--height 4.8: 16:55 판은 5.00 지도로 돌았다. perception 은 상자를 정중앙 기준으로 두므로 상자 y 는 차이의 절반만큼 옮긴다).

쓰기:  python replay_log.py <기록 폴더> [--height 4.8] [--search 0.5] [--logged-pose] [--pose-check] [--png out.png] [--set 이름=값 ...]
기록 폴더 = navigation/debug.jsonl 과 navigation_slam/debug.jsonl 이 든 곳 (https://maicon-match.tlate.ai/files/<경기>/... 에서 받는다).
"""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'test')))


def _import_slam():
    try:
        import rclpy  # noqa: F401
        from navigation import slam_node
        return slam_node
    except ImportError:  # PC: 시험의 가짜 ROS 로 불러온다 (test/test_node_split.load_slam)
        from test_node_split import load_slam
        return SimpleNamespace(**load_slam()[0])   # runpy 가 준 이름표 → 모듈처럼 쓴다


sn = _import_slam()
from navigation.map_fusion import MapFusion, FusionParams, _surface_distance, _wall_distance  # noqa: E402
from navigation.drive.fieldmap import fixed_pillars  # noqa: E402


def load(folder):
    first_map = None
    with open(os.path.join(folder, 'navigation', 'debug.jsonl'), encoding='utf-8') as f:
        for line in f:
            if '"event":"map"' in line:
                first_map = json.loads(line)
                break
    scans = []
    with open(os.path.join(folder, 'navigation_slam', 'debug.jsonl'), encoding='utf-8') as f:
        for line in f:
            if '"scan_raw"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('pose'):
                scans.append(r)
    return first_map, scans


def as_scan(r):
    ranges = [x if x else float('nan') for x in r['ranges']]
    return SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link'), ranges=ranges, range_min=0.02, range_max=12.0,
                           angle_min=r['angle_min'], angle_increment=r['angle_increment'])


def run(folder, height=None, search=0.5, logged_pose=False, params=None, quiet=False, pose_check=False):
    first_map, scans = load(folder)
    width, map_height = first_map['size']
    size = first_map['box_size']
    boxes = [tuple(b) for b in first_map['boxes']]
    shift = 0.0
    if height and abs(height - map_height) > 1e-6:
        shift = (height - map_height) / 2.0
        boxes = [(x, y + shift) for x, y in boxes]
        map_height = height
    field = SimpleNamespace(width=width, height=map_height, box_size=size, boxes=[SimpleNamespace(x=x, y=y) for x, y in boxes])
    fixed = fixed_pillars(width, map_height, size)
    fusion = MapFusion(params or FusionParams())
    fusion.set_map(boxes, size, width, map_height, fixed=fixed)
    out = {'poses': [], 'explained': [], 'versions': 0, 'failed': 0, 'events': [], 'doubted': 0}
    last_out = None
    previous = None
    steps = np.arange(-search, search + 1e-9, 0.1)
    for r in scans:
        scan = as_scan(r)
        logged = (r['pose'][0], r['pose'][1] + shift, r['pose'][2])
        live, weights = fusion.structure()
        pose = logged
        if not logged_pose:
            points5 = np.array(sn.scan_points(scan), dtype=float)
            sboxes, half, w, h = sn.structure(field, boxes=live)
            wts = sn._pad_weights(weights, len(sboxes))
            best = None
            for seed in [logged] + ([previous] if previous is not None else []):
                for dyaw in (-0.2, -0.1, 0.0, 0.1, 0.2):
                    for dx in steps:
                        for dy in steps:
                            cand = (seed[0] + dx, seed[1] + dy, seed[2] + dyaw)
                            hits, err = sn.score_pose(points5, cand, sboxes, half, w, h, wts)
                            key = (-hits, err + 0.02 * math.hypot(dx, dy))   # 같은 점수면 씨앗 자리에 가까운 쪽
                            if best is None or key < best[0]:
                                best = (key, cand)
            matched = sn.scan_match(field, scan, best[1], False, boxes=live, weights=weights)
            if matched is None:
                out['failed'] += 1
                continue
            pose = matched
            previous = pose
        points = np.array(sn.scan_points(scan, fusion.params.scan_step), dtype=float)
        # --pose-check: 외벽이 다른 자리를 가리키는 스캔으로는 지도를 고치지 않는다 (slam_node 가 하는 것과 같다 — 위치를 옮기지는 않는다)
        if pose_check and sn.wall_suspect(points, pose, width, map_height) is not None:
            out['doubted'] += 1
        else:
            fusion.observe(points, pose, int(r['t'] * 1e9), sensor=(sn.LIDAR_X, 0.0))
        # 이 자세에서 지도(벽 · 기둥 · 정합에 쓰는 상자)로 설명되는 점의 비율
        world = sn.to_world(points, pose)
        live, _ = fusion.structure()
        solid = np.concatenate((np.asarray(live, dtype=float).reshape(-1, 2), np.asarray(fixed, dtype=float).reshape(-1, 2)))
        d = _wall_distance(world, width, map_height)
        if len(solid):
            d = np.minimum(d, _surface_distance(world, solid, size / 2).min(axis=1))
        reach = np.hypot(points[:, 0], points[:, 1]) < 3.5
        out['explained'].append(float((d[reach] < 0.07).mean()) if reach.any() else 0.0)
        out['poses'].append((r['t'], pose, logged))
        now = [(round(x, 2), round(y, 2)) for x, y, _ in fusion.output()]
        if last_out is not None and now != last_out:
            out['versions'] += 1
            s = fusion.summary()
            out['events'].append((round(r['t'] - scans[0]['t'], 1), len(now), s.get('rejected_boxes'), s.get('moved_boxes')))
        last_out = now
    out['summary'] = fusion.summary()
    out['fusion'] = fusion
    out['field'] = (width, map_height, size, boxes, fixed)
    out['scans'] = scans
    if not quiet:
        e = np.array(out['explained'])
        off = np.array([math.hypot(p[0] - l[0], p[1] - l[1]) for _, p, l in out['poses']]) if out['poses'] else np.zeros(1)
        print('스캔 %d (정합 실패 %d) · 지도 %.2f x %.2f · 드론 상자 %d' % (len(scans), out['failed'], width, map_height, len(boxes)))
        print('  설명되는 점 비율: 가운데값 %.2f · 아래 10%% %.2f · 0.5 아래인 스캔 %d' % (np.median(e), np.percentile(e, 10), int((e < 0.5).sum())))
        print('  기록된 자세와의 거리: 가운데값 %.2f m · 0.15 m 넘는 스캔 %d · 가장 큼 %.2f' % (np.median(off), int((off > 0.15).sum()), off.max()))
        s = out['summary']
        print('  내보내는 지도가 바뀐 횟수 %d · 확인 %d · 뺀 상자 %s' % (out['versions'], s['confirmed'], s['rejected_boxes']))
        print('  옮긴 상자 %s' % (s.get('moved_boxes'),))
        print('  더한 상자 %s' % (s['added_boxes'],))
        low = size / 2 - 0.08
        print('  그 가운데 경기장 밖으로 나간 것 %d · 외벽이 다른 자리를 가리켜 건너뛴 스캔 %d'
              % (sum(1 for x, y in s['added_boxes'] if not (low <= x <= width - low and low <= y <= map_height - low)), out['doubted']))
    return out


def draw(out, path):
    import cv2
    width, height, size, boxes, fixed = out['field']
    S, M = 100, 0.5
    acc = np.zeros((int((height + 2 * M) * S), int((width + 2 * M) * S)), np.float32)
    by_t = {t: pose for t, pose, _ in out['poses']}
    for r in out['scans']:
        pose = by_t.get(r['t'])
        if pose is None:
            continue
        pts = np.array(sn.scan_points(as_scan(r), 1), dtype=float)
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 3.5]
        w = sn.to_world(pts, pose)
        px = ((w[:, 0] + M) * S).astype(int)
        py = ((height + M - w[:, 1]) * S).astype(int)
        ok = (px >= 0) & (px < acc.shape[1]) & (py >= 0) & (py < acc.shape[0])
        np.add.at(acc, (py[ok], px[ok]), 1)
    img = np.full(acc.shape + (3,), 255, np.uint8)
    v = np.clip(acc / 3.0, 0, 1)
    img[acc > 0] = np.stack([(1 - v[acc > 0]) * 170] * 3, axis=1).astype(np.uint8)

    def T(x, y):
        return int((x + M) * S), int((height + M - y) * S)

    def box(x, y, colour, thick=1):
        cv2.rectangle(img, T(x - size / 2, y + size / 2), T(x + size / 2, y - size / 2), colour, thick)
    cv2.rectangle(img, T(0, height), T(width, 0), (0, 160, 0), 1)
    for x, y in fixed:
        box(x, y, (0, 140, 255))
    for x, y in boxes:
        box(x, y, (255, 170, 170))                       # 드론 지도 (옅은 파랑)
    for x, y, _ in out['fusion'].output():
        box(x, y, (255, 0, 0), 2)                        # 내보내는 지도 (파랑)
    for x, y in out['fusion'].added():
        box(x, y, (0, 0, 255), 2)                        # 라이다로 더한 상자 (빨강)
    poses = [p for _, p, _ in out['poses']]
    for a, b in zip(poses, poses[1:]):
        cv2.line(img, T(a[0], a[1]), T(b[0], b[1]), (0, 170, 0), 1)
    for g in range(0, 7):
        cv2.putText(img, str(g), T(g, -0.3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for g in range(0, 5):
        cv2.putText(img, str(g), T(-0.4, g), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--height', type=float, default=None)
    ap.add_argument('--search', type=float, default=0.5)
    ap.add_argument('--logged-pose', action='store_true', help='다시 맞추지 않고 기록된 자세를 그대로 쓴다')
    ap.add_argument('--png', default=None)
    ap.add_argument('--pose-check', action='store_true', help='외벽이 다른 자리를 가리키는 스캔으로는 지도를 고치지 않는다 (2026-11-19 위치 검증)')
    ap.add_argument('--set', action='append', default=[], help='FusionParams 값 바꾸기 (이름=값)')
    a = ap.parse_args()
    params = FusionParams()
    for item in a.set:
        k, v = item.split('=')
        old = getattr(params, k)
        setattr(params, k, v.lower() in ('1', 'true') if isinstance(old, bool) else type(old)(float(v)))
    result = run(a.folder, a.height, a.search, a.logged_pose, params, pose_check=a.pose_check)
    for event in result['events']:
        print('   t+%s s: 지도 상자 %d · 뺀 %s · 옮긴 %s' % event)
    if a.png:
        draw(result, a.png)
