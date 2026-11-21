"""서서 돌기 고침(executor: 구조물 회전 검사 · 안 도는 회전 감지)을 실차 기록에 대 본다. 로봇을 움직이지 않는다.

    python -I real_check.py bag <bag_0.db3> [이름]          # 웹 시험기 실차 bag (/cmd_vel · /imu/data · /team/navigation/pose · map)
    python -I real_check.py team <supervisor/events.jsonl>  # 포털 팀 테스트 기록 (0.5 s 위치 · 지도)

bag 에서 보는 것
  1. 서서 돌라는 명령(|v| < 0.01, |w| ≥ 0.3)이 시작된 뒤 IMU 방향이 3° 바뀌기까지 걸린 시간 — rot_stuck_s 와 견준다
  2. 같은 명령 동안 _turn_stuck 과 같은 규칙(rot_stuck_s 동안 rot_stuck_rad 도 안 바뀜)이 켜졌을까 — 켜졌다면 잘못 켜진 것 (실차는 돌았다)
  3. 실제로 돈 회전(시작 자세 = SLAM 위치, 각도 = IMU 로 잰 실제 회전)을 _structure_turn_hits 가 막았을까 — 막았다면 잘못 막은 것
  4. 실제 궤적 내내 차체와 지도 구조물 사이의 가장 좁은 틈
팀 테스트 기록에서는 3 · 4 를 (0.5 s 간격 위치로 서서 돈 곳을 찾아서).
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
NAV = ROOT / '개발' / 'src' / 'navigation'
sys.path.insert(0, str(NAV))
sys.path.insert(0, str(NAV / '시험기록' / '2026-11-19_map_fusion_실차재생'))
from navigation.drive import executor as ex  # noqa: E402
from navigation.drive.fieldmap import MapSpec, fixed_pillars, from_field_map  # noqa: E402
from navigation.drive.geometry import Pose2D, wrap  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402

P = DriveParams()
HL, HW = P.robot_half_length, P.robot_half_width


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def checker():
    return ex.GoalExecutor(ex.GoalSpec(0.0, 0.0, None, 0, 0.1, 0.1, 10.0), P, 0.0)


def interp(series, t):
    """series = [(t, value)] 정렬됨 → t 의 값 (각도는 펼친 값이어야 한다)."""
    lo, hi = 0, len(series) - 1
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if series[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, v0), (t1, v1) = series[lo], series[hi]
    return v0 + (v1 - v0) * (t - t0) / max(t1 - t0, 1e-9)


def unwrap(series):
    out, prev, acc = [], None, 0.0
    for t, a in series:
        if prev is not None:
            acc += wrap(a - prev)
        else:
            acc = a
        prev = a
        out.append((t, acc))
    return out


def check_bag(db, name):
    import bagread
    cmds, imu, poses, maps = [], [], [], []
    for ts, topic, m in bagread.read(db, {'/cmd_vel', '/imu/data', '/team/navigation/pose', '/team/navigation/map'}):
        t = ts / 1e9
        if topic == '/cmd_vel':
            cmds.append((t, m.linear.x, m.angular.z))
        elif topic == '/imu/data':
            imu.append((t, yaw_of(m.orientation)))
        elif topic == '/team/navigation/pose':
            poses.append((t, m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)))
        elif topic == '/team/navigation/map' and (not maps or len(m.boxes) != maps[-1][2]):
            spec = from_field_map(m, pillars=P.fixed_pillars).spec
            maps.append((t, spec, len(m.boxes)))
    imu_u = unwrap(imu)
    print('== %s: /cmd_vel %d · IMU %d · 위치 %d · 지도 %d번 (경기장 %.2f × %.2f, 상자 %d, 고정 기둥 %d)' % (
        name, len(cmds), len(imu), len(poses), len(maps), maps[-1][1].width, maps[-1][1].height, len(maps[-1][1].boxes), len(maps[-1][1].fixed)))

    def spec_at(t):
        s = maps[0][1]
        for tm, sp, _n in maps:
            if tm <= t:
                s = sp
        return s

    def pose_at(t):
        best = min(poses, key=lambda p: abs(p[0] - t))
        return best if abs(best[0] - t) < 0.3 else None

    # 서서 돌라는 명령 구간
    segs, cur = [], None
    for t, v, w in cmds:
        turning = abs(v) < 0.01 and abs(w) >= 0.3
        if turning and cur is None:
            cur = [t, t, w, abs(w)]
        elif turning:
            cur[1], cur[3] = t, max(cur[3], abs(w))
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    onsets, false_stuck, refused, rows = [], 0, 0, []
    job = checker()
    for t0, t1, w0, wmax in segs:
        y0 = interp(imu_u, t0)
        onset = None
        k = t0
        while k <= t1 + 1.0:
            if abs(interp(imu_u, k) - y0) >= math.radians(3.0):
                onset = k - t0
                break
            k += 0.02
        if onset is not None and onset <= t1 - t0 + 1.0:
            onsets.append(onset)
        # _turn_stuck 과 같은 규칙 (2026-11-19 통합 뒤): 명령 중 rot_stuck_s 창마다 돈 각의 합이 rot_stuck_rad 보다 작으면
        ref_t, ref_y, trig = t0, y0, False
        k = t0
        while k <= t1:
            y = interp(imu_u, k)
            if k - ref_t >= P.rot_stuck_s:
                if abs(y - ref_y) < P.rot_stuck_rad:
                    trig = True
                    print('    켜짐: 명령 시작 %.1f s · 길이 %.2f s · w %.2f · 창 [%.2f, %.2f] s 동안 돈 각 %.1f° · 명령 전체 동안 %.1f°' % (
                        t0 - cmds[0][0], t1 - t0, w0, ref_t - t0, k - t0, math.degrees(y - ref_y), math.degrees(interp(imu_u, t1) - y0)))
                    break
                ref_t, ref_y = k, y
            k += 1.0 / 16.0
        false_stuck += trig
        turned = interp(imu_u, t1 + 0.4) - y0  # 명령 끝 뒤 미끄러짐까지
        p = pose_at(t0)
        hit = None
        gap = None
        if p is not None and abs(turned) > math.radians(2.0):
            spec = spec_at(t0)
            grid = NS(spec=spec)
            hit = job._structure_turn_hits(Pose2D(p[1], p[2], p[3]), turned, grid)
            refused += bool(hit)
            gap = ex.body_clearance(spec, p[1], p[2], p[3], HL, HW)
        rows.append((t0 - cmds[0][0], t1 - t0, w0, wmax, onset, trig, math.degrees(turned), hit, gap,
                     spec_at(t0).distance_to_structure(p[1], p[2]) if p else None))
    print('  서서 돌기 명령 %d번 (길이 중앙 %.2f s)' % (len(segs), sorted(r[1] for r in rows)[len(rows) // 2] if rows else 0))
    if onsets:
        o = sorted(onsets)
        print('  돌기 시작까지 (IMU 3°): 중앙 %.2f s · p90 %.2f s · 최대 %.2f s (rot_stuck_s %.1f s) · %d번 잼' % (
            o[len(o) // 2], o[int(len(o) * 0.9)], o[-1], P.rot_stuck_s, len(o)))
    print('  안 도는 회전 감지가 잘못 켜졌을 회전: %d / %d' % (false_stuck, len(segs)))
    print('  구조물 회전 검사가 잘못 막았을 회전: %d / %d (실제로 돈 것)' % (refused, sum(1 for r in rows if r[7] is not None)))
    near = [r for r in rows if r[9] is not None and r[9] <= math.hypot(HL, HW) + P.structure_turn_clear]
    print('  그중 구조물 곁(가운데에서 %.2f m 안 → 검사함): %d번' % (math.hypot(HL, HW) + P.structure_turn_clear, len(near)))
    for r in near:
        print('    t=%6.1f s 명령 %.2f s · w %.2f · 돈 각도 %+.0f° · 막음 %s · 틈 %.3f m · 가운데-구조물 %.3f m' % (
            r[0], r[1], r[2], r[6], r[7], r[8], r[9]))
    gaps = []
    for t, x, y, yaw in poses[::5]:
        gaps.append((ex.body_clearance(spec_at(t), x, y, yaw, HL, HW), t - poses[0][0], x, y))
    g = min(gaps)
    print('  궤적 내내 차체-구조물 가장 좁은 틈: %.3f m (t=%.1f s, (%.2f, %.2f)) · 5 cm 안 %.1f%%' % (
        g[0], g[1], g[2], g[3], 100.0 * sum(1 for q in gaps if q[0] < 0.05) / len(gaps)))
    return rows


def check_team(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    m = next(r for r in rows if r.get('e') == 'map')
    w, h, size = m['size']
    boxes = tuple((float(x), float(y)) for x, y in m['boxes'])
    spec = MapSpec(w, h, size, boxes, 'arena', fixed=tuple(p for p in fixed_pillars(w, h, size) if not any(abs(p[0] - bx) < size / 2 and abs(p[1] - by) < size / 2 for bx, by in boxes)) if P.fixed_pillars else ())
    ticks = [(r['t'], r['pose'][0], r['pose'][1], math.radians(r['pose'][2])) for r in rows if r.get('e') == 'tick' and r.get('pose') and r.get('t') is not None]
    job = checker()
    turns, refused, cur = [], 0, None
    for a, b in zip(ticks, ticks[1:]):
        still = math.hypot(b[1] - a[1], b[2] - a[2]) < 0.03
        dyaw = wrap(b[3] - a[3])
        if still and abs(dyaw) >= math.radians(4.0):
            if cur is None:
                cur = [a, dyaw]
            else:
                cur[1] += dyaw
        elif cur is not None:
            turns.append(cur)
            cur = None
    if cur is not None:
        turns.append(cur)
    near = []
    for (t, x, y, yaw), ang in turns:
        if abs(ang) < math.radians(8.0):
            continue
        hit = job._structure_turn_hits(Pose2D(x, y, yaw), ang, NS(spec=spec))
        refused += bool(hit)
        d = spec.distance_to_structure(x, y)
        if d <= math.hypot(HL, HW) + P.structure_turn_clear:
            near.append((t, x, y, math.degrees(ang), hit, ex.body_clearance(spec, x, y, yaw, HL, HW), d))
    gaps = [(ex.body_clearance(spec, x, y, yaw, HL, HW), t, x, y) for t, x, y, yaw in ticks]
    g = min(gaps)
    big = [tr for tr in turns if abs(tr[1]) >= math.radians(8.0)]
    print('== %s: 지도 상자 %d · 고정 기둥 %d · 위치 %d개 · 서서 돈 곳 %d번' % (path, len(boxes), len(spec.fixed), len(ticks), len(big)))
    print('  구조물 회전 검사가 잘못 막았을 회전: %d / %d · 구조물 곁에서 돈 것 %d번' % (refused, len(big), len(near)))
    for r in near:
        print('    t=%6.1f s (%.2f, %.2f) 돈 각도 %+.0f° · 막음 %s · 틈 %.3f m · 가운데-구조물 %.3f m' % r)
    print('  궤적 내내 차체-구조물 가장 좁은 틈: %.3f m (t=%.1f s, (%.2f, %.2f)) · 5 cm 안 %.1f%%' % (
        g[0], g[1], g[2], g[3], 100.0 * sum(1 for q in gaps if q[0] < 0.05) / len(gaps)))


if __name__ == '__main__':
    if sys.argv[1] == 'bag':
        check_bag(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else Path(sys.argv[2]).parent.parent.name)
    else:
        check_team(sys.argv[2])
