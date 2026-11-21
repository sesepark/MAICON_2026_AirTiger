"""경기 기록으로 「드론 사진이 중간에 끊겼다」 를 흉내 내어, 붙든 UGV(drive/ugvs.UgvTracks._held)를 라이다가 맞게 확인 · 반박하는지 본다.

끊긴 시각을 cut_step 초마다 바꿔 가며: 그때까지의 드론 사진(navigation/debug.jsonl 의 ugvs)으로 궤적을 만들고, 그 뒤로는 라이다 스캔
(navigation_slam/debug.jsonl 의 scan_raw, 약 1 Hz)만 ObstacleTracker.on_scan 에 넣는다. 참값 = 그 판에서 실제로는 계속 온 드론 사진
(붙든 자리에서 ugv_match_m 안에 UGV 가 보이면 「아직 있다」).

  잘못 지움 : 라이다가 「떠났다」 고 지웠는데 드론으로는 그 자리에 있었다
  맞게 지움 : 지웠고 드론으로도 떠났다
  못 지움   : 드론으로는 떠났는데 끝(끊긴 뒤 ugv_hold_s)까지 붙들었다 (라이다가 그 자리를 못 봤다)
  붙듦      : 드론으로도 끝까지 있었다
  [확인 · 빔 · 모름 · 돎] = 그 UGV 자리를 스캔마다 어떻게 봤나: 지도에 없는 점이 찍힘 · 광선이 지나감 · 알 수 없음(가림 · 범위 밖 · 값 없음) · 빨리 돌아 건너뜀

기록의 스캔은 1 Hz 라 실제(10 Hz)보다 반박이 늦다 (ugv_hold_clear_scans 번 = 여기서는 3 s). 빨리 도는 스캔은 주행 기록의 명령 각속도로 가른다.

쓰기:  python replay_hold.py <기록 폴더> [--cut-step 20] [--set 이름=값 ...]
       기록 폴더 = navigation/debug.jsonl · navigation_slam/debug.jsonl 이 든 곳
"""
import argparse
import bisect
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
from navigation.drive.fieldmap import MapSpec, fixed_pillars  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.grid import Grid  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker, scan_to_body  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.ugvs import UgvTracks  # noqa: E402


def rows(path, want):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not any(w in line for w in want):
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def load(folder):
    nav = rows(os.path.join(folder, 'navigation', 'debug.jsonl'), ('"map"', '"ugvs"', '"state"'))
    slam = rows(os.path.join(folder, 'navigation_slam', 'debug.jsonl'), ('"scan_raw"',))
    amap = next(r for r in nav if r.get('event') == 'map')
    frames = [r for r in nav if r.get('event') == 'ugvs']
    states = [r for r in nav if r.get('event') == 'state' and r.get('pose')]
    scans = [r for r in slam if r.get('event') == 'scan_raw' and r.get('pose')]
    return amap, frames, states, scans


def feed(tracks, frame, states, state_t, p):
    """기록의 사진 한 장을 주행 노드와 같은 규칙으로 넣는다 (그 무렵 우리가 있던 자리 = 2 Hz 상태 줄의 자세)."""
    stamp = frame['stamp']
    lo, hi = bisect.bisect_left(state_t, stamp - p.ugv_self_window_s - 0.5), bisect.bisect_right(state_t, stamp + 0.5)
    recent = [tuple(states[k]['pose'][:2]) for k in range(lo, hi)]
    tracks.on_frame(stamp, [tuple(q) for q in frame['seen']], recent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--cut-step', type=float, default=20.0)
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    p = DriveParams()
    for item in args.set:
        name, value = item.split('=')
        setattr(p, name, type(getattr(p, name))(float(value)))
    amap, frames, states, scans = load(args.folder)
    w, h = amap['size']
    spec = MapSpec(w, h, amap['box_size'], tuple(tuple(b) for b in amap['boxes']), 'arena',
                   tuple(fixed_pillars(w, h, amap['box_size'])) if p.fixed_pillars else ())
    grid = Grid(spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
    state_t = [s['t'] for s in states]
    t0, t1 = frames[0]['stamp'], frames[-1]['stamp']
    print('사진 %d장 (%.0f s) · 스캔 %d개 · ugv_hold_s %.0f · clear %.2f m × %d번' % (
        len(frames), t1 - t0, len(scans), p.ugv_hold_s, p.ugv_hold_clear_m, p.ugv_hold_clear_scans))

    def there(x, y, t):
        """참값: t 무렵의 사진(앞뒤 한 장씩)에 그 자리 ugv_match_m 안의 UGV 가 있었나. 사진이 없으면 None."""
        near = [f for f in frames if abs(f['stamp'] - t) <= 3.0]
        if not near:
            return None
        return any(math.hypot(q[0] - x, q[1] - y) <= p.ugv_match_m for f in near for q in f['seen'])

    total = {'잘못 지움': 0, '맞게 지움': 0, '못 지움': 0, '붙듦': 0, '참값 없음': 0}
    confirmed = waits = 0
    cut = t0 + 30.0
    while cut < t1 - 20.0:
        tracks = UgvTracks(p)
        tracker = ObstacleTracker(p)
        tracker.ugvs = tracks
        for f in frames:
            if f['stamp'] <= cut:
                feed(tracks, f, states, state_t, p)
        lost = tracks.stamp
        start = lost + tracks._feed() + 0.1
        held = {id(u): (u.x, u.y, u.still_s()) for u in tracks.fresh(start)}
        notes = []
        fate = {}
        seen = {key: [0, 0, 0, 0] for key in held}   # 스캔 수: 확인 · 비었다 · 알 수 없음(가림 · 범위 밖 · 값 없음) · 빨리 돌아 건너뜀
        confirm, refute = tracks.confirm, tracks.refute
        tracks.confirm = lambda u, t: (seen[id(u)].__setitem__(0, seen[id(u)][0] + 1), confirm(u, t))[1]
        tracks.refute = lambda u, clear=True: (seen[id(u)].__setitem__(1 if clear else 2, seen[id(u)][1 if clear else 2] + 1), refute(u, clear))[1]
        for s in scans:
            t = s['t']
            if t < start or t > min(t1, lost + p.ugv_hold_s):
                continue
            k = min(bisect.bisect_left(state_t, t), len(states) - 1)
            turn = abs((states[k].get('cmd') or [0.0, 0.0])[1])
            ranges = np.array([x if x else np.nan for x in s['ranges']], dtype=float)
            body = scan_to_body(ranges, s['angle_min'], s['angle_increment'], p)
            before = {id(u) for u in tracks.fresh(t)}
            if p.scan_turn_skip_rate > 0.0 and turn >= p.scan_turn_skip_rate:
                for key in before & seen.keys():
                    seen[key][3] += 1
            tracker.on_scan(t, Pose2D(*s['pose']), body, spec, grid, (s['angle_min'], s['angle_increment'], ranges), turn_rate=turn)
            for key in before - {id(u) for u in tracks.fresh(t)}:
                if key in held and key not in fate:
                    fate[key] = t
        end = min(t1, lost + p.ugv_hold_s)
        for u in tracks.tracks:
            if id(u) in held and u.lidar is not None:
                confirmed += 1
        for key, (x, y, still) in held.items():
            if key in fate:
                truth = there(x, y, fate[key])
                kind = '참값 없음' if truth is None else ('잘못 지움' if truth else '맞게 지움')
                notes.append('(%.2f, %.2f) %s +%.0f s' % (x, y, kind, fate[key] - lost))
                if kind == '맞게 지움':
                    waits += 1
            else:
                truth = there(x, y, end)
                kind = '참값 없음' if truth is None else ('붙듦' if truth else '못 지움')
                notes.append('(%.2f, %.2f) %s [확인 %d · 빔 %d · 모름 %d · 돎 %d]' % ((x, y, kind) + tuple(seen[key])))
            total[kind] += 1
        print('끊김 %4.0f s: 붙든 것 %d — %s' % (lost - t0, len(held), ' · '.join(notes) or '-'))
        cut += args.cut_step
    print('합계', total, '· 라이다로 확인된 것', confirmed)


if __name__ == '__main__':
    main()
