"""자리에는 왔는데 도착 방향으로 돌려니 서 있는 물체에 걸릴 때: 기다리지 않고 한 번 비켜 서서 돈다 (params.align_room_moves, 2026-11-20). ROS 없이 돈다.

2026-11-20 01:50 실차 22 · 23번: 서 있는 UGV (3.04, 2.56) 에서 0.34 m 인 자리 (3.19, 2.87) 에서 방향을 100° 남기고 3 s 씩 두 번 서 있다가 취소됐다.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_drive_core import DT, SimRobot, goal, make_world, scan_of_discs  # noqa: E402

from navigation.drive import executor as ex  # noqa: E402
from navigation.drive import pose_tracker as loc  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.ugvs import UgvTracks  # noqa: E402

HERE = Pose2D(3.19, 2.87, -2.21)   # 23번 목표를 받은 자리 · 방향 (그 UGV 쪽을 보고 서 있다)
YAW = -0.50                        # 도착 방향 — 100° 를 더 돌아야 한다
UGV = (3.04, 2.56)


def turn_here(params, discs, drone=(), max_t=20.0):
    """HERE 에 선 채 YAW 로 돌라는 목표 (자리는 이미 맞다). discs = 라이다에 찍히는 서 있는 것들 [(x, y, r)], drone = 드론이 30 s 전부터 본 UGV 가운데들."""
    params.goal_shift_m = 0.0   # 방향 맞추기만 본다
    p, spec, grid = make_world(boxes=(), width=6.12, height=4.8, params=params)
    job = ex.GoalExecutor(goal(HERE.x, HERE.y, yaw=YAW, pos_tol=0.12, yaw_tol=0.37, timeout_s=max_t), p, 0.0)
    robot = SimRobot(HERE)
    assert job.validate(spec, grid, robot.pose)
    obstacles = ObstacleTracker(p)
    obstacles.ugvs = UgvTracks(p)
    me = (HERE.x, HERE.y)
    for k in range(-30, 1):
        obstacles.ugvs.on_frame(float(k), [me] + list(drone), [me], local=me, pose=me)
    t, k, phases, nearest = 0.0, 0, [], math.inf
    while not job.done and t < max_t:
        me = (robot.pose.x, robot.pose.y)
        if k % 20 == 0 and k:
            obstacles.ugvs.on_frame(t, [me] + list(drone), [me], local=me, pose=me)
        if k % 2 == 0:
            obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, discs, p), spec, grid)
        robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
        if not phases or phases[-1] != job.phase:
            phases.append(job.phase)
        nearest = min([nearest] + [math.hypot(robot.pose.x - x, robot.pose.y - y) - r for x, y, r in discs])
        t += DT
        k += 1
    return job, robot, t, phases, nearest


def yaw_left(robot):
    return abs(math.atan2(math.sin(robot.pose.yaw - YAW), math.cos(robot.pose.yaw - YAW)))


class AlignStandingTest(unittest.TestCase):
    def test_before_it_waited_and_ended_without_the_heading(self):
        old = DriveParams()
        old.align_room_moves = 0
        job, robot, t, phases, _near = turn_here(old, [UGV + (0.13,)], drone=[UGV])
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn(ex.NO_ROOM_AT_GOAL, job.detail)
        self.assertGreater(t, old.obstacle_give_up_s - 0.2)       # 5 s 를 서 있었다
        self.assertGreater(yaw_left(robot), 0.37)

    def test_moves_off_a_parked_ugv_once_and_turns(self):
        job, robot, t, phases, nearest = turn_here(DriveParams(), [UGV + (0.13,)], drone=[UGV])
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertNotIn(ex.NO_ROOM_AT_GOAL, job.detail)
        self.assertIn("비켜 섬", job.detail)
        self.assertLessEqual(yaw_left(robot), 0.37 + 1e-6)        # 도착 방향을 맞췄다
        self.assertLess(t, 5.0)
        self.assertEqual(sum(1 for ph in phases if ph in ("backing_up", "moving_off")), 1)   # 한 번만 비킨다 — 자리로 되돌아가 되풀이하지 않는다
        self.assertLess(math.hypot(robot.pose.x - HERE.x, robot.pose.y - HERE.y), 0.30)
        self.assertGreater(nearest, 0.17 - 0.02)                   # 그 UGV 에 차체 모서리가 닿지 않는다 (반 대각선 0.17)

    def test_a_standing_thing_seen_only_by_lidar_is_treated_the_same_after_it_has_stood(self):
        job, robot, t, phases, _near = turn_here(DriveParams(), [UGV + (0.13,)])
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLessEqual(yaw_left(robot), 0.37 + 1e-6)
        self.assertIn("비켜 섬", job.detail)

    def test_ends_quickly_when_there_is_nowhere_to_move_off(self):
        """앞뒤로 서 있는 것에 끼어 비킬 곳이 없다 → 5 s 를 기다리지 않고 obstacle_give_up_parked_s 뒤 REACHED [NO_ROOM_AT_GOAL]."""
        c, s = math.cos(HERE.yaw), math.sin(HERE.yaw)
        ahead = (HERE.x + 0.33 * c, HERE.y + 0.33 * s)
        behind = (HERE.x - 0.33 * c, HERE.y - 0.33 * s)
        p = DriveParams()
        job, robot, t, phases, _near = turn_here(p, [ahead + (0.13,), behind + (0.13,)], drone=[ahead, behind])
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn(ex.NO_ROOM_AT_GOAL, job.detail)
        self.assertIn("서 있는 UGV", job.detail)
        self.assertLess(t, p.obstacle_give_up_parked_s + 1.0)
        self.assertLess(math.hypot(robot.pose.x - HERE.x, robot.pose.y - HERE.y), 0.03)


if __name__ == "__main__":
    unittest.main()
