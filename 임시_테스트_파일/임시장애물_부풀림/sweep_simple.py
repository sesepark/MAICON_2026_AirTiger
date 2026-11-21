"""2026-11-19 검토 도구(p2_sweep · p4_deadlock)의 무작위 장면을 obstacle_radius 별로 — 간단한 차체(지연 0.1 s, 회전 마찰 없음, 원기둥).

static  : 출발 (1.0, 2.5) → 목표 (4.0, 2.5) 사이에 지도에 없는 원기둥 하나 (반지름 0.06 ~ 0.15)
cross   : 원기둥이 옆에서 0.15 ~ 0.4 m/s 로 길을 가로지른다
headon  : 원기둥이 마주 오다 선다
deadlock: 출발 바로 곁(0.30 ~ 0.70 m)에 원기둥, 출발 방향 무작위 → 목표 (4.5, y)

    python sweep_simple.py <kind> <n> <seed> [r ...]
"""
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I 로 돌려도 squeeze_simple 을 찾게
from squeeze_simple import EXTRA, DT, ObstacleTracker, Robot, body_clear, scan_of_discs, ex, loc, MapSpec, Grid, DriveParams, Pose2D

sys.stdout.reconfigure(encoding="utf-8")


def run(job, robot, spec, grid, p, discs_fn, max_t):
    ob = ObstacleTracker(p)
    t, clr = 0.0, 9.9
    while not job.done and t < max_t:
        discs = discs_fn(t)
        if int(round(t / DT)) % 2 == 0:
            ob.on_scan(t, robot.pose, scan_of_discs(robot.pose, discs, p), spec, grid)
        cmd = job.step(t, robot.pose, loc.OK, grid, ob)
        clr = min([clr] + [body_clear(robot.pose, d, p) for d in discs])
        robot.apply(cmd)
        t += DT
    return t, clr


def cases(kind, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if kind == "deadlock":
            yaw0 = rng.uniform(-0.9, 0.9)
            gy = rng.uniform(1.5, 3.5)
            d, a = rng.uniform(0.30, 0.70), rng.uniform(-0.8, 0.8)
            disc = (2.0 + d * math.cos(yaw0 + a), 2.5 + d * math.sin(yaw0 + a), rng.uniform(0.05, 0.14))
            if math.hypot(disc[0] - 2.0, disc[1] - 2.5) < disc[2] + 0.19:
                continue
            out.append(dict(start=(2.0, 2.5, yaw0), goal=(4.5, gy), mode=0, timeout=15.0, fn=lambda t, dd=disc: [dd]))
            continue
        yaw0 = rng.uniform(-1.2, 1.2)
        mode = rng.choice((0, 0, 2))
        if kind == "static":
            disc = (rng.uniform(1.4, 3.5), 2.5 + rng.uniform(-0.6, 0.6), rng.uniform(0.06, 0.15))
            fn = lambda t, d=disc: [d]
        elif kind == "cross":
            y0 = rng.choice((1.2, 3.8))
            vy = rng.uniform(0.15, 0.4) * (1 if y0 < 2.5 else -1)
            x0, t0 = rng.uniform(1.6, 3.4), rng.uniform(0.0, 4.0)
            fn = lambda t, x0=x0, y0=y0, vy=vy, t0=t0: [(x0, y0 + vy * max(0.0, t - t0), 0.12)]
        else:
            x0, vx = rng.uniform(3.0, 4.5), -rng.uniform(0.1, 0.3)
            stop_x, yoff = rng.uniform(1.8, 3.0), rng.uniform(-0.15, 0.15)
            fn = lambda t, x0=x0, vx=vx, stop_x=stop_x, yoff=yoff: [(max(stop_x, x0 + vx * t), 2.5 + yoff, 0.12)]
        out.append(dict(start=(1.0, 2.5, yaw0), goal=(4.0, 2.5), mode=mode, timeout=25.0, fn=fn))
    return out


if __name__ == "__main__":
    kind, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    radii = [float(a) for a in sys.argv[4:]] or [0.18, 0.13, 0.08, 0.05, 0.0]
    todo = cases(kind, n, seed)
    for r in radii:
        tally, coll, clrs, times = Counter(), 0, [], []
        for c in todo:
            p = DriveParams()
            for k, v in EXTRA.items():
                setattr(p, k, v)
            p.obstacle_radius = r
            spec = MapSpec(6.12, 5.0, 0.40, (), "arena")
            grid = Grid(spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
            job = ex.GoalExecutor(ex.GoalSpec(c["goal"][0], c["goal"][1], None, c["mode"], 0.10, 0.10, c["timeout"], "arena", "r"), p, 0.0)
            robot = Robot(Pose2D(*c["start"]))
            job.validate(spec, grid, robot.pose)
            t, clr = run(job, robot, spec, grid, p, c["fn"], c["timeout"])
            tally[ex.OUTCOME_NAMES.get(job.outcome, "없음(시한)")] += 1
            coll += clr < 0.0
            clrs.append(clr)
            times.append(t)
        clrs.sort()
        print("%-8s r %.2f (합 %.2f) %d판: %s · 닿음 %d · 최소 거리 가장 작은 것 %.3f · 중앙값 %.3f m · 걸림 중앙값 %.1f s" % (
            kind, r, r + 0.17, len(todo), dict(tally), coll, clrs[0], clrs[len(clrs) // 2], sorted(times)[len(times) // 2]), flush=True)
