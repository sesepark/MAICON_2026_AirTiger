"""경기 기록의 스캔(scan_raw)과 그때 내보낸 위치로 「벽으로 위치 다시 보기」(slam_node.wall_suspect → 드론 지도만으로 다시 정합)를 돌려 본다.

무엇을 보나
  - 그 경기에서 위치가 틀렸던 구간에 판정이 켜지는가, 그리고 옮길 자리가 한결같은가 (WALL_STREAK 번 이어 같은 쪽)
  - 위치가 맞던 구간에서는 켜지지 않는가 (잘못 옮기면 안 된다)
기록의 스캔은 약 1 Hz 뿐이라 「이어서 5번」 은 여기서 5초다 (로봇에서는 0.5초). 옮긴 뒤의 일(그 뒤로 정합이 새 자리를 따라가는지)은
이 도구로 볼 수 없다 — 기록된 위치는 옮기지 않은 그대로이기 때문이다. 그래서 옮김 판정이 난 스캔마다 「기록된 자리 → 옮길 자리」 만 적는다.

쓰기:  python replay_wall_check.py <기록 폴더> [--all]
기록 폴더 = navigation/debug.jsonl 과 navigation_slam/debug.jsonl 이 든 곳 (또는 navigation_debug.jsonl · navigation_slam_debug.jsonl 로 받아 둔 곳).
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'test')))


def _import_slam():
    try:
        import rclpy  # noqa: F401 — 있으면 진짜 ROS 로
        from navigation import slam_node
        return slam_node
    except ImportError:  # PC: 시험의 가짜 ROS 로 불러온다
        from test_node_split import load_slam
        return SimpleNamespace(**load_slam()[0])


sn = _import_slam()


def _path(folder, node):
    for name in (os.path.join(folder, node, 'debug.jsonl'), os.path.join(folder, node + '_debug.jsonl')):
        if os.path.exists(name):
            return name
    raise FileNotFoundError('%s 의 %s 기록이 없다' % (folder, node))


def load(folder):
    maps, start = [], None
    with open(_path(folder, 'navigation'), encoding='utf-8') as f:
        for line in f:
            if '"event":"map"' in line:
                maps.append(json.loads(line))
            elif start is None and '"event":"goal"' in line:
                start = json.loads(line)['t']
    scans = []
    with open(_path(folder, 'navigation_slam'), encoding='utf-8') as f:
        for line in f:
            if '"scan_raw"' in line:
                r = json.loads(line)
                if r.get('pose'):
                    scans.append(r)
    return maps, scans, start


def as_scan(r):
    ranges = [x if x else float('nan') for x in r['ranges']]
    return SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link'), ranges=ranges, range_min=0.02, range_max=12.0,
                           angle_min=r['angle_min'], angle_increment=r['angle_increment'])


def check(field, scan, pose):
    """slam_node 가 스캔마다 하는 것과 같은 판정. (옮길 자리 또는 None, 벽 점수 (지금, 그 자리) 또는 None)."""
    points = np.array(sn.scan_points(scan, 2), dtype=float)
    suspect = sn.wall_suspect(points, pose, field.width, field.height)
    if suspect is None:
        return None, None
    here, there, where = suspect
    second = sn.confirm_wall_candidate(field, scan, pose, where, boxes=[(b.x, b.y) for b in field.boxes])
    return second, (here, there)


def run(folder, show_all=False, quiet=False):
    maps, scans, start = load(folder)
    start = start if start is not None else scans[0]['t']
    out = {'scans': 0, 'suspect': 0, 'confirmed': 0, 'moves': [], 'ms': []}
    streak = []
    index = 0
    for r in scans:
        while index + 1 < len(maps) and maps[index + 1]['t'] <= r['t']:
            index += 1
        m = maps[index]
        field = SimpleNamespace(width=m['size'][0], height=m['size'][1], box_size=m['box_size'],
                                boxes=[SimpleNamespace(x=x, y=y) for x, y in m['boxes']])
        pose = tuple(r['pose'])
        began = time.perf_counter()
        second, scores = check(field, as_scan(r), pose)
        out['ms'].append((time.perf_counter() - began) * 1000)
        out['scans'] += 1
        t = r['t'] - start
        if scores is not None:
            out['suspect'] += 1
        if second is None:
            streak = []
            if show_all and scores is not None and not quiet:
                print('t%+7.1f (%.2f, %.2f, %4.0f°) 벽 %3.0f → %3.0f  다시 정합이 받지 않음' % (t, pose[0], pose[1], math.degrees(pose[2]), *scores))
            continue
        out['confirmed'] += 1
        shift = (second[0] - pose[0], second[1] - pose[1])
        if streak and math.hypot(shift[0] - streak[-1][0], shift[1] - streak[-1][1]) > sn.WALL_AGREE_M:
            streak = []
        streak.append(shift)
        moved = len(streak) >= sn.WALL_STREAK
        if moved:
            out['moves'].append((t, pose, second))
        if not quiet:
            print('t%+7.1f (%.2f, %.2f, %4.0f°) 벽 %3.0f → %3.0f  옮길 양 (%+.2f, %+.2f, %+3.0f°) %s'
                  % (t, pose[0], pose[1], math.degrees(pose[2]), *scores, shift[0], shift[1],
                     math.degrees(sn.wrap(second[2] - pose[2])), '← 옮김' if moved else ''))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--all', action='store_true', help='다시 정합이 받지 않은 의심도 적는다')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    for folder in args.folders:
        print('==', os.path.basename(os.path.normpath(folder)))
        out = run(folder, args.all, args.quiet)
        ms = sorted(out['ms'])
        print('   스캔 %d · 벽이 다른 자리를 가리킴 %d · 다시 정합도 받음 %d · 옮김 판정 %d (스캔 %d개 이어짐) · 계산 %.1f ms (가운데) / %.1f ms (가장 김)'
              % (out['scans'], out['suspect'], out['confirmed'], len(out['moves']), sn.WALL_STREAK, ms[len(ms) // 2] if ms else 0, ms[-1] if ms else 0))


if __name__ == '__main__':
    main()
