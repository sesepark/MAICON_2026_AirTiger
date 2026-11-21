"""2026-11-19 검토 도구(p10_squeeze)의 좁은 틈 장면을 그대로 다시 돌린다 — 간단한 차체(지연 0.1 s, 회전 마찰 없음).

지도 상자 벽 x = 3.0 (y 0.2 ~ 1.4 · 3.6 ~ 4.8, 열린 곳 y 1.6 ~ 3.4) 가운데에 지도에 없는 원기둥 둘(반지름 0.12 m)을 gap 만큼 띄워 둔다.
출발 (1.0, start_y, 0°) → 목표 (5.0, 2.5). obstacle_radius 를 바꿔 가며 결과 · 시간 · 차체와 원기둥의 최소 거리를 찍는다.

    python squeeze_simple.py [obstacle_radius ...]
"""
import math
import sys
from collections import Counter
from pathlib import Path

import os  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
# NAV_ROOT = 다른 패키지 사본(시제품), PARAMS = "이름=값,이름=값" 으로 모든 판에 더할 주행 파라미터
NAV = Path(os.environ["NAV_ROOT"]) if os.environ.get("NAV_ROOT") else Path(__file__).resolve().parents[2] / "개발" / "src" / "navigation"
EXTRA = {k: float(v) for k, v in (item.split("=") for item in os.environ.get("PARAMS", "").split(",") if item)}
sys.path.insert(0, str(NAV))
from navigation.drive import executor as ex, pose_tracker as loc  # noqa: E402
from navigation.drive.fieldmap import MapSpec  # noqa: E402
from navigation.drive.geometry import Pose2D, advance  # noqa: E402
from navigation.drive.grid import Grid  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402

DT = 0.05


class Robot:
    def __init__(self, pose, tau=0.1):
        self.pose, self.v, self.w, self.tau = pose, 0.0, 0.0, tau

    def apply(self, cmd):
        k = min(1.0, DT / self.tau)
        self.v += (cmd.v - self.v) * k
        self.w += (cmd.w - self.w) * k
        self.pose = advance(self.pose, self.v, self.w, DT)


def scan_of_discs(pose, discs, p):
    body = []
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    for cx, cy, r in discs:
        for k in range(48):
            a = 2 * math.pi * k / 48
            dx, dy = cx + r * math.cos(a) - pose.x, cy + r * math.sin(a) - pose.y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            if p.scan_self_mask_m < math.hypot(bx - p.lidar_x, by) < p.scan_max_range:
                body.append((bx, by))
    return body


def body_clear(pose, disc, p):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    dx, dy = disc[0] - pose.x, disc[1] - pose.y
    bx, by = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(bx) - p.robot_half_length, 0.0), max(abs(by) - p.robot_half_width, 0.0)) - disc[2]


def run_case(radius, gap, start_y, max_t=40.0):
    p = DriveParams()
    for k, v in EXTRA.items():
        setattr(p, k, v)
    p.obstacle_radius = radius
    boxes = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4)] + [(3.0, y) for y in (3.6, 4.0, 4.4, 4.8)]
    spec = MapSpec(6.12, 5.0, 0.40, tuple(boxes), "arena")
    grid = Grid(spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
    r = 0.12
    discs = [(3.0, 2.5 - gap / 2 - r, r), (3.0, 2.5 + gap / 2 + r, r)]
    job = ex.GoalExecutor(ex.GoalSpec(5.0, 2.5, None, 0, 0.10, 0.10, max_t, "arena", "r"), p, 0.0)
    robot = Robot(Pose2D(1.0, start_y, 0.0))
    job.validate(spec, grid, robot.pose)
    ob = ObstacleTracker(p)
    t, phases, clr = 0.0, Counter(), 9.9
    while not job.done and t < max_t:
        if int(round(t / DT)) % 2 == 0:
            ob.on_scan(t, robot.pose, scan_of_discs(robot.pose, discs, p), spec, grid)
        cmd = job.step(t, robot.pose, loc.OK, grid, ob)
        clr = min(clr, min(body_clear(robot.pose, d, p) for d in discs))
        phases[job.phase] += 1
        robot.apply(cmd)
        t += DT
    return job, t, clr, phases


if __name__ == "__main__":
    radii = [float(a) for a in sys.argv[1:]] or [0.18, 0.13, 0.08, 0.05, 0.0]
    print("부풀림 합 = obstacle_radius + robot_radius 0.15 + inflate_margin 0.02")
    for radius in radii:
        for gap, sy in ((0.36, 2.5), (0.40, 2.5), (0.40, 2.62), (0.45, 2.7), (0.50, 2.75)):
            job, t, clr, phases = run_case(radius, gap, sy)
            print("r %.2f (합 %.2f) 틈 %.2f 출발 y %.2f → %-9s %5.1f s  원기둥과 최소 %.3f m  %s | %s" % (
                radius, radius + 0.17, gap, sy, ex.OUTCOME_NAMES.get(job.outcome, job.outcome) if hasattr(ex, "OUTCOME_NAMES") else job.outcome,
                t, clr, ",".join("%s %d" % kv for kv in phases.most_common(3)), job.detail[:40]))
