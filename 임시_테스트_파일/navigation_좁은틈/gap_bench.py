"""40 cm 틈(규정의 최소 이격)을 지도 오차가 있을 때 지나가는가. 쓰기: python gap_bench.py <navigation 패키지 폴더>
지도의 틈 폭(map)과 실제 틈 폭(real)을 따로 준다. 스캔 = 실제 상자의 틈 쪽 면 · 앞면 위의 점 (지도와 다르면 「지도에 없는 점」 이 된다)."""
import math, os, sys
sys.stdout.reconfigure(encoding="utf-8")
PKG = os.path.abspath(sys.argv[1])
sys.path.insert(0, PKG); sys.path.insert(0, os.path.join(PKG, "test"))
import test_drive_core as D
from navigation.drive import executor as ex
from navigation.drive.fieldmap import MapSpec
from navigation.drive.geometry import Pose2D
from navigation.drive.params import DriveParams

X, YC = 3.0, 2.6

def wall(lo_face, hi_face):
    """틈의 아래 면 y = lo_face, 위 면 y = hi_face 인 상자 벽 (x = 3.0 줄)."""
    boxes = []
    y = lo_face - 0.2
    while y > -0.2:
        boxes.append((X, round(y, 3))); y -= 0.4
    y = hi_face + 0.2
    while y < 5.2:
        boxes.append((X, round(y, 3))); y += 0.4
    return boxes

def face_points(lo_face, hi_face, step=0.02):
    pts = []
    for k in range(int(0.4 / step) + 1):
        x = X - 0.2 + k * step
        pts += [(x, lo_face), (x, hi_face)]
    for k in range(int(0.5 / step) + 1):
        pts += [(X - 0.2, lo_face - k * step), (X - 0.2, hi_face + k * step), (X + 0.2, lo_face - k * step), (X + 0.2, hi_face + k * step)]
    return [(x, y, 0.002) for x, y in pts]

def body_gap(pose, real_boxes, p):
    spec = MapSpec(6.12, 5.0, 0.40, tuple(real_boxes), "arena")
    return ex.body_clearance(spec, pose.x, pose.y, pose.yaw, p.robot_half_length, p.robot_half_width)

def trial(name, map_faces, real_faces, start=(1.0, 2.6, 0.0), target=(5.0, 2.6), params=None):
    p = params or DriveParams()
    mboxes, rboxes = wall(*map_faces), wall(*real_faces)
    p, spec, grid = D.make_world(boxes=mboxes, params=p)
    job = ex.GoalExecutor(D.goal(*target, timeout_s=60.0), p, 0.0)
    robot = D.SimRobot(Pose2D(*start))
    if not job.validate(spec, grid, robot.pose):
        print(f"{name:34s} INVALID {job.detail}"); return
    discs = face_points(*real_faces)
    trace = D.run(job, robot, spec, grid, p, discs=discs, max_t=70.0)
    gaps = [body_gap(pose, rboxes, p) for _t, pose, _c, _ph, _b in trace]
    touch = sum(g < 0.0 for g in gaps)
    phases = {}
    for _t, _pose, _c, ph, _b in trace: phases[ph] = phases.get(ph, 0) + 1
    top = ", ".join(f"{k}:{v*D.DT:.0f}s" for k, v in sorted(phases.items(), key=lambda kv: -kv[1])[:3])
    print(f"{name:34s} {ex.OUTCOME_NAMES.get(job.outcome, job.outcome) if hasattr(ex,'OUTCOME_NAMES') else job.outcome!s:10s} t={trace[-1][0]:5.1f}s 최소 틈 {min(gaps)*100:+5.1f} cm 닿은 주기 {touch:3d}  [{top}]  {job.detail[:70]}")

h = 0.20
for name, m, r in [
    ("지도 0.40 · 실제 0.40", 0.40, 0.40), ("지도 0.36 · 실제 0.40", 0.36, 0.40), ("지도 0.32 · 실제 0.40", 0.32, 0.40),
    ("지도 0.30 · 실제 0.40", 0.30, 0.40), ("지도 0.44 · 실제 0.40", 0.44, 0.40), ("지도 0.50 · 실제 0.40", 0.50, 0.40),
    ("지도 0.60 · 실제 0.40", 0.60, 0.40), ("지도 0.45 · 실제 0.45", 0.45, 0.45), ("지도 0.40 · 실제 0.50", 0.40, 0.50)]:
    trial(name, (YC - m / 2, YC + m / 2), (YC - r / 2, YC + r / 2))
trial("한쪽만: 지도 아래 상자 +6 cm", (YC - 0.20 + 0.06, YC + 0.20), (YC - 0.20, YC + 0.20))
trial("한쪽만: 실제 아래 상자 +6 cm(틈 0.40)", (YC - 0.26, YC + 0.20), (YC - 0.20, YC + 0.20))
trial("비스듬히 들어감 (지도=실제 0.40)", (YC - 0.2, YC + 0.2), (YC - 0.2, YC + 0.2), start=(1.5, 1.2, 0.0), target=(4.5, 4.0))
trial("비스듬히 (지도 0.36 · 실제 0.40)", (YC - 0.18, YC + 0.18), (YC - 0.2, YC + 0.2), start=(1.5, 1.2, 0.0), target=(4.5, 4.0))
trial("한쪽만: 지도 아래 상자 +10 cm", (YC - 0.20 + 0.10, YC + 0.20), (YC - 0.20, YC + 0.20))
trial("비스듬히 (지도 0.30 · 실제 0.40)", (YC - 0.15, YC + 0.15), (YC - 0.2, YC + 0.2), start=(1.5, 1.2, 0.0), target=(4.5, 4.0))
trial("비스듬히 반대 (지도 0.32 · 실제 0.40)", (YC - 0.16, YC + 0.16), (YC - 0.2, YC + 0.2), start=(1.5, 4.0, 0.0), target=(4.5, 1.0))
trial("비스듬히, 한쪽 +10 cm", (YC - 0.10, YC + 0.20), (YC - 0.2, YC + 0.2), start=(1.5, 1.2, 0.0), target=(4.5, 4.0))
trial("옆을 보고 출발 (지도 0.34)", (YC - 0.17, YC + 0.17), (YC - 0.2, YC + 0.2), start=(2.3, 2.6, 1.57), target=(4.5, 2.6))
