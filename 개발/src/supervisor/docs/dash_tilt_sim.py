"""거점 안에 다른 로봇이 서 있을 때, 대기 자리에서의 차체 기울기에 따라 첫 돌진이 들어가는가 — navigation 의 가상 주행으로 세어 본다 (ROS 없이).

쓰는 법 (navigation 패키지 폴더에서):
    python3 ../supervisor/docs/dash_tilt_sim.py .                       # 지금 코드
    python3 ../supervisor/docs/dash_tilt_sim.py . assault_shift_out_m=-1   # 설정을 바꿔서 (이름=값,이름=값)

지도는 실제 경기장처럼 고정 기둥(거점의 양 끝)을 넣는다 — test_goal_shift.drive 의 지도에는 기둥이 없어 04:48 판(2026-11-20)이 다시 만들어지지 않았다.
글: 돌진_기울기_확인_요청_2026-11-20.md
"""
import math
import sys

root = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else ""
sys.path.insert(0, root)
sys.path.insert(0, root + "/test")

from test_drive_core import DT, SimRobot, goal, scan_of_discs  # noqa: E402
from navigation.drive import executor as ex  # noqa: E402
from navigation.drive import pose_tracker as loc  # noqa: E402
from navigation.drive.fieldmap import MapSpec, fixed_pillars  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.grid import Grid  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.ugvs import UgvTracks  # noqa: E402

UGV_R = 0.13
W, H = 6.12, 4.80


def params():
    p = DriveParams()
    for kv in variant.split(","):
        if "=" in kv:
            k, v = kv.split("=")
            cur = getattr(p, k)
            setattr(p, k, (v == "True") if isinstance(cur, bool) else type(cur)(float(v)))
    return p


def dash(p, start, target, ugv, max_t=25.0):
    """start 에서 target 으로 돌진. ugv = 서 있는 다른 UGV 의 가운데 (드론은 30 s 전부터 보았고 라이다에는 원으로 찍힌다). → (목표, 로봇, 걸린 시간)"""
    spec = MapSpec(W, H, 0.40, (), "arena", fixed=fixed_pillars(W, H))
    grid = Grid(spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
    job = ex.GoalExecutor(goal(target[0], target[1], yaw=None, mode=ex.MODE_ASSAULT, yaw_tol=0.5, timeout_s=max_t), p, 0.0)
    robot = SimRobot(start)
    if not job.validate(spec, grid, robot.pose):
        return job, robot, 0.0
    obstacles = ObstacleTracker(p)
    obstacles.ugvs = UgvTracks(p)
    me = (start.x, start.y)
    for k in range(-30, 1):
        obstacles.ugvs.on_frame(float(k), [me, ugv], [me], local=me, pose=me)
    t, k = 0.0, 0
    while not job.done and t < max_t:
        me = (robot.pose.x, robot.pose.y)
        if k % 20 == 0 and k:
            obstacles.ugvs.on_frame(t, [me, ugv], [me], local=me, pose=me)
        if k % 2 == 0:
            obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [ugv + (UGV_R,)], p), spec, grid)
        robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
        t += DT
        k += 1
    return job, robot, t


if __name__ == "__main__":
    tilts = (-40, -20, -10, 0, 3, 5, 10, 22, 30, 60, 90)   # + = 서 있는 로봇 쪽으로 기울어 섬, − = 먼 쪽
    print("설정: %s" % (variant or "지금 코드"))
    print("차체 기울기(°, + = 서 있는 로봇 쪽):  " + " ".join("%6d" % t for t in tilts))
    total = {t: [0, 0] for t in tilts}
    for ux in (0.14, 0.20, 0.26, 0.32, 0.37, 0.42):
        for uy in (2.34, 2.37, 2.40, 2.43, 2.46):
            for y in (2.78, 2.02):                      # supervisor 가 고르는 자리 (거점 끝에서 0.22 m)
                if abs(uy - y) < 0.30:
                    continue
                toward = -1.0 if uy < y else 1.0
                cells = []
                for tilt in tilts:
                    yaw = math.pi - toward * math.radians(tilt)
                    job, robot, took = dash(params(), Pose2D(0.80, y, math.atan2(math.sin(yaw), math.cos(yaw))), (0.20, y), (ux, uy))
                    ok = job.outcome == ex.REACHED and robot.pose.x <= 0.30
                    total[tilt][0] += ok
                    total[tilt][1] += 1
                    cells.append("%s%.1f" % ("O " if ok else ("X막" if job.outcome == ex.BLOCKED else "X얕"), took))
                if any(c.startswith("X") for c in cells):
                    print("  로봇 (%.2f, %.2f) 자리 y %.2f:  " % (ux, uy, y) + " ".join("%6s" % c for c in cells))
    print("들어간 판 (벽에서 0.30 m 안):       " + " ".join("%3d/%d" % tuple(total[t]) for t in tilts))
