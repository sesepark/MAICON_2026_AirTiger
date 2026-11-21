"""경기 기록의 navigation/debug.jsonl(3e1b3e1 부터 남음)로 서서 돌기 고침을 확인한다. 로봇을 움직이지 않는다.

    python -I run_check.py <기록 폴더>      # <기록 폴더>/navigation/debug.jsonl

보는 것
  1. 실제로 「서서 돌라는데 안 돈」 구간 (단계 rotating · aligning, 명령 w ≠ 0, 방향 변화 < rot_stuck_rad 가 rot_stuck_s 넘게) — 새 _turn_stuck 이 잡았을 곳
  2. 실제로 돈 회전(단계 rotating · aligning 에서 위치 그대로 방향만 바뀐 곳)을 새 _structure_turn_hits 가 막았을까 (잘못 막음)
  3. 목표마다 결과 · 걸린 시간 · 그 동안의 단계 — 길게 서 있던 목표가 무엇 때문이었나
  4. 길게 선 목표의 마지막 상태에서 새 코드라면 회전을 무엇이 막는가 (지도 구조물 / 지도에 없는 점)
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '개발' / 'src' / 'navigation'))
from navigation.drive import executor as ex  # noqa: E402
from navigation.drive.fieldmap import MapSpec, fixed_pillars  # noqa: E402
from navigation.drive.geometry import Pose2D, wrap  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402

P = DriveParams()
HL, HW = P.robot_half_length, P.robot_half_width


class Pts:
    """ObstacleTracker 의 turn_hits 만 흉내: 기록의 지도에 없는 점(경기장 좌표) → 차체 기준."""

    def __init__(self, pose, world):
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        self.unexplained_body = [(c * (x - pose.x) + s * (y - pose.y), -s * (x - pose.x) + c * (y - pose.y)) for x, y in world]
        self.p = P

    def turn_hits(self, t, angle, margin):
        from navigation.drive.obstacles import ObstacleTracker
        o = ObstacleTracker(P)
        o.unexplained_body, o.scan_t = self.unexplained_body, t
        return o.turn_hits(t, angle, margin)


def main(folder):
    rows = [json.loads(line) for line in open(Path(folder) / 'navigation' / 'debug.jsonl', encoding='utf-8')]
    m = next(r for r in rows if r.get('event') == 'map')
    w, h = m['size']
    boxes = tuple(tuple(b) for b in m['boxes'])
    size = m['box_size']
    fixed = tuple(p for p in fixed_pillars(w, h, size) if not any(abs(p[0] - bx) < size / 2 and abs(p[1] - by) < size / 2 for bx, by in boxes))
    spec = MapSpec(w, h, size, boxes, 'arena', fixed=fixed)
    grid = NS(spec=spec)
    t0 = rows[0]['t']
    recs = sorted([r for r in rows if r.get('event') in ('state', 'phase') and r.get('pose')], key=lambda r: r['t'])

    def phase_of(r):
        return r['phase'] if r.get('event') == 'phase' else (r.get('job') or {}).get('phase')

    print('== %s · 지도 상자 %d · 고정 기둥 %d · 기록 %d줄' % (Path(folder).name, len(boxes), len(fixed), len(rows)))

    # 1 · 2. 서서 돌기 구간
    pinned, turns, refused = [], 0, []
    job = ex.GoalExecutor(ex.GoalSpec(0.0, 0.0, None, 0, 0.1, 0.1, 10.0), P, 0.0)
    cur = None
    for r in recs:
        ph, cmd = phase_of(r), r.get('cmd') or [0.0, 0.0]
        rot = ph in ('rotating', 'aligning') and abs(cmd[1]) > 1e-6 and abs(cmd[0]) < 1e-6
        if rot and cur is None:
            cur = [r, r]
        elif rot:
            cur[1] = r
        elif cur is not None:
            a, b = cur
            dyaw = wrap(b['pose'][2] - a['pose'][2])
            dur = b['t'] - a['t']
            moved = math.hypot(b['pose'][0] - a['pose'][0], b['pose'][1] - a['pose'][1])
            if dur >= P.rot_stuck_s and abs(dyaw) < P.rot_stuck_rad:
                pinned.append((a['t'] - t0, dur, a['pose'], ex.body_clearance(spec, *a['pose'], HL, HW)))
            if abs(dyaw) >= math.radians(8.0) and moved < 0.05:
                turns += 1
                pose = Pose2D(*a['pose'])
                if job._structure_turn_hits(pose, dyaw, grid):
                    refused.append((a['t'] - t0, a['pose'], math.degrees(dyaw), ex.body_clearance(spec, *a['pose'], HL, HW)))
            cur = None
    print('  실제로 서서 돌라는데 %.1f s 넘게 안 돈 구간: %d번' % (P.rot_stuck_s, len(pinned)))
    for t, dur, pose, gap in pinned:
        print('    t=%6.1f s · %.1f s · 자세 (%.2f, %.2f, %.0f°) · 지도 구조물과 틈 %.3f m' % (t, dur, pose[0], pose[1], math.degrees(pose[2]), gap))
    print('  실제로 서서 돈 회전 %d번 중 구조물 회전 검사가 잘못 막았을 것: %d번' % (turns, len(refused)))
    for t, pose, ang, gap in refused:
        print('    t=%6.1f s (%.2f, %.2f, %.0f°) 돈 각도 %+.0f° · 틈 %.3f m · 가운데-구조물 %.3f m' % (
            t, pose[0], pose[1], math.degrees(pose[2]), ang, gap, spec.distance_to_structure(pose[0], pose[1])))

    # 3. 목표마다
    goals = {r['no']: r for r in rows if r.get('event') == 'goal'}
    results = {r['no']: r for r in rows if r.get('event') == 'result'}
    phases = {}
    for r in rows:
        if r.get('event') == 'phase':
            phases.setdefault(r['no'], Counter())[r['phase']] += 1
    outs = Counter(results[n]['outcome'] for n in results)
    print('  목표 %d개: %s' % (len(goals), dict(outs)))
    long = [(results[n]['elapsed'], n) for n in results if results[n]['elapsed'] >= 15.0]
    for el, n in sorted(long, reverse=True)[:8]:
        res = results[n]
        print('    목표 %3d · %5.1f s · %-9s · %s · 단계 %s' % (n, el, res['outcome'], res['detail'][:60], dict(phases.get(n, {}).most_common(4))))
        # 4. 그 목표의 마지막 state 에서 새 코드라면 무엇이 회전을 막나
        st = [r for r in rows if r.get('event') == 'state' and (r.get('job') or {}).get('goal') and r.get('pose')
              and goals[n]['t'] <= r['t'] <= res['t']]
        if st:
            s = st[-1]
            pose = Pose2D(*s['pose'])
            j = ex.GoalExecutor(ex.GoalSpec(0.0, 0.0, None, 0, 0.1, 0.1, 10.0), P, 0.0)
            obs = Pts(pose, s.get('unexplained') or [])
            for ang in (math.radians(90.0), -math.radians(90.0)):
                j._free_turn(s['t'], ang, abs(ang), obs, pose, grid)
            print('        마지막 자세 (%.2f, %.2f, %.0f°) · 지도 구조물 틈 %.3f m · 가운데-구조물 %.3f m · ±90° 회전을 막는 것 %s · 지도에 없는 점 %d' % (
                pose.x, pose.y, math.degrees(pose.yaw), ex.body_clearance(spec, pose.x, pose.y, pose.yaw, HL, HW),
                spec.distance_to_structure(pose.x, pose.y), sorted(j._turn_block) or '없음', len(s.get('unexplained') or [])))


if __name__ == '__main__':
    main(sys.argv[1])
