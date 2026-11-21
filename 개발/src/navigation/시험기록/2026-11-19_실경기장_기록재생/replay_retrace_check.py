"""경기 기록에서 시한 전에 BLOCKED 로 끝난 목표마다: 그때 되짚기 조건(executor._can_retrace)이 맞았을까.

기록의 자세(arena, 2 Hz)로 지나온 길을 다시 만든다 — 실차의 길은 로컬 궤적 · 16 Hz 라 이보다 촘촘하다. 물러난 뒤의 일은 이 도구로 볼 수 없다.

쓰기:  python replay_retrace_check.py <기록 폴더>… [-v]
기록 폴더 = navigation/debug.jsonl 이 든 곳 (또는 navigation_debug.jsonl 로 받아 둔 곳).
"""
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
from navigation.drive.executor import GoalExecutor  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.trail import Trail  # noqa: E402

P = DriveParams()
REACH = math.hypot(P.robot_half_length, P.robot_half_width) + P.sweep_margin + 0.05   # executor._needs_room


def gap(m, x, y):
    """지도(벽 · 상자)까지의 틈. 고정 기둥은 기록의 지도에 없어 보지 않는다."""
    w, h = m['size']
    half = m['box_size'] / 2
    g = min(x, w - x, y, h - y)
    for bx, by in m['boxes']:
        dx, dy = abs(x - bx) - half, abs(y - by) - half
        g = min(g, math.hypot(max(dx, 0), max(dy, 0)) if max(dx, dy) > 0 else max(dx, dy))
    return g


def run(folder, verbose=False):
    path = next(q for q in (os.path.join(folder, 'navigation', 'debug.jsonl'), os.path.join(folder, 'navigation_debug.jsonl')) if os.path.exists(q))
    nav = [json.loads(line) for line in open(path, encoding='utf-8')]
    maps = [r for r in nav if r.get('event') == 'map']
    start = next((r['t'] for r in nav if r.get('event') == 'goal'), 0)
    trail = Trail()
    last = None
    out = collections.Counter()
    rows = []
    for r in nav:
        if r.get('event') == 'state' and r.get('pose'):
            p = r['pose']
            if last is not None:   # 2 Hz 기록 사이를 0.05 m 간격으로 메운다
                q = last['pose']
                n = int(math.hypot(p[0] - q[0], p[1] - q[1]) / 0.05)
                for k in range(1, n):
                    f = k / n
                    trail.update(Pose2D(q[0] + (p[0] - q[0]) * f, q[1] + (p[1] - q[1]) * f, p[2]))
            trail.update(Pose2D(*p))
            last = r
        elif r.get('event') == 'result' and r['outcome'] == 'BLOCKED' and last is not None and r.get('pose') and maps:
            if 'BLOCKED_AT_TIMEOUT' in r['detail'] or 'INTERNAL' in r['detail']:
                continue
            p = r['pose']
            m = max((x for x in maps if x['t'] <= r['t']), key=lambda x: x['t'], default=maps[0])
            trail.update(Pose2D(*p))
            target = trail.back_target(P.retrace_reach_m)
            err = math.degrees(abs(GoalExecutor._rear_err(target))) if target else None
            room = trail.back_room()
            un = last.get('unexplained') or []
            near = any(math.hypot(x - p[0], y - p[1]) <= REACH for x, y in un)
            corridor = min(p[1], m['size'][1] - p[1]) < P.corridor_wall_band_m
            tight = corridor or gap(m, p[0], p[1]) <= REACH or near
            c, s = math.cos(p[2]), math.sin(p[2])
            behind = 0
            for x, y in un:   # 뒤쪽 띠 (obstacles.blocking_ahead(-1))
                bx, by = (x - p[0]) * c + (y - p[1]) * s, -(x - p[0]) * s + (y - p[1]) * c
                if abs(by) <= P.robot_half_width + P.stop_side_margin and 0.0 <= -bx - P.robot_half_length <= P.stop_dist + P.retrace_v * P.stop_reaction_s:
                    behind += 1
            code = 'NO_ROOM' if 'NO_ROOM' in r['detail'] else 'NO_PROGRESS' if 'NO_PROGRESS' in r['detail'] else 'OBSTACLE'
            if not tight:
                verdict = '넓은 곳 — 물러나지 않음'
            elif target is None or room < P.retrace_min_m:
                verdict = '지나온 길 없음'
            elif err > P.retrace_max_deg:
                verdict = '길이 차체 뒤가 아님'
            elif behind:
                verdict = '뒤쪽 띠에 점'
            else:
                verdict = '되짚음'
            out[verdict] += 1
            rows.append('  t%+6.1f (%.2f, %.2f, %4.0f°) %-11s %s%s' % (
                r['t'] - start, p[0], p[1], math.degrees(p[2]), code, verdict, '' if err is None else '  [뒤 축에서 %.0f°, 길 %.2f m]' % (err, room)))
    print('==', os.path.basename(os.path.normpath(folder)), dict(out))
    if verbose:
        print('\n'.join(rows))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    folders = [a for a in sys.argv[1:] if not a.startswith('-')]
    total = collections.Counter()
    for folder in folders:
        total += run(folder, '-v' in sys.argv)
    print('전체', dict(total))


if __name__ == '__main__':
    main()
