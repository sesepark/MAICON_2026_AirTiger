"""임시 장애물 부풀림(obstacle_radius)별 비교 — SLAM 노드 + 주행 노드 닫힌 루프, 실차 흉내(sim_world.REAL_ROBOT).

장면
  gap      : 출발 칸 B-2 (0.2, 2.5, 0°) 앞 x = 1.6 에 상자 벽. 열린 곳은 하나뿐 — 상자 면과 다른 UGV 사이(box_ugv) 또는 UGV 둘 사이(ugv_ugv)의 틈 G.
             틈 가운데를 출발 y 에서 off 만큼 비켜 둔다. 목표 (3.0, 틈 가운데). 다른 UGV 의 몸 250 × 230, 라이다에는 가운데 seen 정사각형만.
  random   : 출발 칸 B-2 → 목표 (2.6, 1.6 ~ 3.4). 출발과 목표 사이에 다른 UGV 하나를 무작위로 (몸이 출발 자세와 겹치지 않게).
  replay   : test/test_team_test_replay.py 의 다른 UGV 시험 셋 + 상자 스침 (그 시험 그대로, 주행 파라미터만 바꿈)

    python compare_real.py gap    [r ...]
    python compare_real.py random [r ...] --n 20 --seed 1
    python compare_real.py replay [r ...]
r 는 obstacle_radius (격자에서 지도에 없는 점 하나를 막는 반지름 = r + robot_radius 0.15 + inflate_margin 0.02).
"""
import argparse
import math
import random
import sys
import unittest
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import os  # noqa: E402
NAV = Path(os.environ["NAV_ROOT"]) if os.environ.get("NAV_ROOT") else Path(__file__).resolve().parents[2] / "개발" / "src" / "navigation"
EXTRA = {}  # 모든 장면에 더할 주행 파라미터 (--set 이름=값)
sys.path.insert(0, str(NAV / "test"))
sys.path.insert(0, str(NAV / "시험기록" / "web_drive"))
import sim_world as W  # noqa: E402

NAMES = {1: "REACHED", 2: "CANCELED", 3: "BLOCKED", 4: "TIMED_OUT", 5: "CONTROL_LOST", 6: "INVALID", 7: "LOC_LOST", None: "없음"}
START = W.slot_start_pose("B-2")  # (0.2, 2.5, 0)


def make_loop(world, radius, seed=1):
    robot = W.SimRobot(world, START, odom_offset=1.0, slip=0.85, seed=seed, **W.REAL_ROBOT)
    loop = W.ClosedLoop(world, robot, drive_params=dict(EXTRA, obstacle_radius=radius))
    loop.start_from_slot("B-2", W.base_region("B-2"))
    loop.run(1.5)
    return loop


def drive(loop, goal, timeout_s=30.0, mode=0):
    """목표를 보내고 끝날 때까지. (결과 이름, 걸린 s, 지도에 없는 것에 닿은 주기, 그것과의 최소 거리 m, 상자 · 벽 포함 닿은 주기)"""
    world, robot = loop.world, loop.robot
    loop.allowed = True
    handle, thread, box = loop.send_goal(goal[0], goal[1], yaw=None, mode=mode, timeout_s=timeout_s)
    job = loop.job_of(handle)
    began = loop.t
    near = 9.9
    c0, h0 = robot.collisions, robot.hidden_contacts
    while job is not None and not job.done and loop.t < began + timeout_s + 5.0:
        loop.step()
        if world.unmapped():
            near = min(near, world.body_clearance(robot.x, robot.y, robot.yaw, walls=False, items=world.unmapped()))
    thread.join(1.0)
    result = box.get("result")
    outcome = NAMES.get(getattr(result, "outcome", None), str(getattr(result, "outcome", None)))
    return outcome, loop.t - began, robot.hidden_contacts - h0, near, robot.collisions - c0, getattr(result, "detail", "")


def gap_world(kind, gap, off, seen):
    """x = 1.6 의 벽. 열린 곳은 틈 하나 (가운데 y = 2.5 + off). 상자 · UGV 는 빈틈 없이 맞붙인다."""
    cy = 2.5 + off
    uh = W.UGV_SIZE[1] / 2  # UGV 의 y 반폭 0.115 (차체 250 이 x, 230 이 y)
    vehicles = []
    if kind == "box_ugv":  # 아래 = 상자 면, 위 = UGV
        low_face, high_face = cy - gap / 2, cy + gap / 2
        vehicles.append((1.6, high_face + uh))
        top_of_ugv = high_face + 2 * uh
    else:  # ugv_ugv: 아래 UGV · 위 UGV
        low_ugv_top, high_ugv_bot = cy - gap / 2, cy + gap / 2
        vehicles += [(1.6, low_ugv_top - uh), (1.6, high_ugv_bot + uh)]
        low_face = low_ugv_top - 2 * uh
        top_of_ugv = high_ugv_bot + 2 * uh
    boxes = []
    y = low_face - 0.2  # 아래 상자 줄: 맨 위 상자의 윗면 = low_face
    while y + 0.2 > 0.0:
        boxes.append((1.6, round(y, 4)))
        y -= 0.4
    y = top_of_ugv + 0.2  # 위 상자 줄: 맨 아래 상자의 아랫면 = UGV 윗면
    while y - 0.2 < 5.0:
        boxes.append((1.6, round(y, 4)))
        y += 0.4
    world = W.World(boxes=boxes, hidden=W.PILLARS, vehicles=vehicles, vehicle_seen=seen, **W.ARENA)
    return world, (3.0, cy)


def run_gap(radii):
    print("장면 gap: x 1.6 상자 벽의 유일한 틈, 목표 (3.0, 틈 가운데), 출발 B-2 (0.2, 2.5, 0°), 시한 30 s")
    print("%-8s %-5s %-5s %-5s %-6s | %-9s %6s %6s %8s" % ("종류", "틈", "비킴", "보임", "r(합)", "결과", "걸림s", "닿음", "최소거리"))
    for kind in ("box_ugv", "ugv_ugv"):
        for gap in (0.40, 0.45, 0.50):
            for off in (0.0, 0.15, 0.25):
                for seen in (0.23, 0.12):
                    for r in radii:
                        world, goal = gap_world(kind, gap, off, seen)
                        loop = make_loop(world, r)
                        out, took, hit, near, coll, detail = drive(loop, goal)
                        print("%-8s %.2f  %.2f  %.2f  %.2f(%.2f) | %-9s %6.1f %6d %8.3f  %s" % (
                            kind, gap, off, seen, r, r + 0.17, out, took, hit, near, detail[:28]), flush=True)


BOXES_1118_2 = ((1.406, 3.923), (1.834, 3.894), (4.305, 4.177), (3.068, 3.745), (4.693, 3.72), (3.301, 2.691), (1.827, 2.479),
                (2.792, 1.62), (3.228, 1.62), (1.712, 1.217), (4.194, 1.192), (1.737, 0.861), (4.2, 0.804))  # 2차 perception 지도 (재현 시험과 같음)


def _clear_of_boxes(x, y, need):
    return all(max(abs(x - bx) - 0.2, 0.0) ** 2 + max(abs(y - by) - 0.2, 0.0) ** 2 >= need * need for bx, by in BOXES_1118_2)


def run_random(radii, n, seed):
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        while True:
            gx, gy = 2.3, rng.uniform(1.6, 3.4)
            k = rng.uniform(0.25, 0.8)
            ux = START[0] + (gx - START[0]) * k + rng.uniform(-0.1, 0.1)
            uy = START[1] + (gy - START[1]) * k + rng.uniform(-0.35, 0.35)
            if (math.hypot(ux - START[0], uy - START[1]) > 0.45 and math.hypot(ux - gx, uy - gy) > 0.45
                    and _clear_of_boxes(ux, uy, 0.20) and _clear_of_boxes(gx, gy, 0.25)):
                break
        cases.append(((gx, gy), (ux, uy), rng.choice((0.12, 0.23)), rng.choice((0, 0, 2))))
    print("장면 random: 출발 B-2 → 목표 (2.3, y), 그 사이 다른 UGV 하나 (몸 250 × 230). 2차 지도 상자 13개 + 기둥. %d판, seed %d" % (n, seed))
    for r in radii:
        tally, hits, nears, times, bad = {}, 0, [], [], []
        for i, (goal, ugv, seen, mode) in enumerate(cases):
            world = W.World(boxes=BOXES_1118_2, hidden=W.PILLARS, vehicles=[ugv], vehicle_seen=seen, **W.ARENA)
            loop = make_loop(world, r, seed=i + 1)
            out, took, hit, near, coll, detail = drive(loop, goal, timeout_s=25.0, mode=mode)
            tally[out] = tally.get(out, 0) + 1
            hits += hit > 0
            nears.append(near)
            times.append(took)
            if out != "REACHED" or hit:
                bad.append((i, out, round(took, 1), hit, round(near, 3), seen, mode, [round(v, 2) for v in goal], [round(v, 2) for v in ugv]))
        nears.sort()
        print("r %.2f (합 %.2f): %s · 닿은 판 %d · 최소 거리 가장 작은 것 %.3f · 중앙값 %.3f m · 걸림 중앙값 %.1f s" % (
            r, r + 0.17, tally, hits, nears[0], nears[len(nears) // 2], sorted(times)[len(times) // 2]), flush=True)
        for b in bad:
            print("    ", b)


def run_replay(radii):
    import test_team_test_replay as T
    names = ["test_assault_into_an_occupied_base_does_not_touch_and_gives_up_early",
             "test_does_not_hit_a_ugv_coming_out_to_the_base_entrance",
             "test_backs_off_after_another_ugv_pushes_in",
             "test_does_not_graze_a_box_on_the_way_to_the_scout_spots"]
    original = W.ClosedLoop.__init__
    for r in radii:
        loops = []

        def patched(self, world, robot, drive_params=None, lidar=True, _r=r):
            params = dict(drive_params or {})
            params.setdefault("obstacle_radius", _r)
            for k, v in EXTRA.items():
                params.setdefault(k, v)
            original(self, world, robot, params, lidar)
            loops.append(self)

        W.ClosedLoop.__init__ = patched
        for name in names:
            loops.clear()
            result = unittest.TestResult()
            unittest.TestSuite([T.TeamTestReplay(name)]).run(result)
            ok = result.wasSuccessful()
            why = (result.failures or result.errors)[0][1].strip().splitlines()[-1] if not ok else ""
            rb = loops[-1].robot if loops else None
            print("r %.2f (합 %.2f) %-70s %s  닿음(지도에 없는 것) %s  상자·벽 포함 %s  차체 최소 거리 %s  %s" % (
                r, r + 0.17, name, "통과" if ok else "실패", rb.hidden_contacts if rb else "-", rb.collisions if rb else "-",
                "%.3f" % rb.min_body_clearance if rb else "-", why[:80]), flush=True)
    W.ClosedLoop.__init__ = original


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=("gap", "random", "replay"))
    ap.add_argument("radii", nargs="*", type=float)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="주행 파라미터 이름=값 (예: narrow_obstacle_radius=0.05)")
    a = ap.parse_args()
    radii = a.radii or [0.18, 0.13, 0.08, 0.05, 0.0]
    for item in a.set:
        k, v = item.split("=")
        EXTRA[k] = float(v)
    print("NAV", NAV, "추가 파라미터", EXTRA)
    if a.scene == "gap":
        run_gap(radii)
    elif a.scene == "random":
        run_random(radii, a.n, a.seed)
    else:
        run_replay(radii)
