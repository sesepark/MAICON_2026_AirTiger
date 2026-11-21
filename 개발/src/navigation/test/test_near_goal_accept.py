"""서 있는 물체가 목표 곁을 막았는데 목표 바로 앞까지 와 있으면 그 자리에서 도착으로 끝낸다 (params.near_goal_accept_m, 2026-11-20). ROS 없이 돈다.

2026-11-20 00:00 실차 5번 목표: 서 있는 UGV 에서 0.3 m 인 정찰 자리 — 목표 0.22 m 앞에서 BLOCKED, 0.6 m 되짚어 물러나 13.6 s.
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

GOAL = (3.2, 2.9)


def drive(params, disc_x, yaw=None, mode=ex.MODE_NORMAL):
    """(1.0, 2.9) 에서 곧게 GOAL 로. 목표 너머 (disc_x, 2.80) 에 반지름 0.12 m 의 서 있는 것 — 목표 칸이 그 임시 장애물에 덮이고, 그 앞에서 선다."""
    # 목표를 곁의 빈 칸으로 옮기는 것(goal_shift_m, 2026-11-20)이 먼저 듣는다 — 여기서는 그것이 못 쓰일 때(옮길 빈 칸이 없다 · 꺼 둠)의 뒷받침을 본다
    params.goal_shift_m = 0.0
    p, spec, grid = make_world(boxes=(), params=params)
    job = ex.GoalExecutor(goal(GOAL[0], GOAL[1], yaw=yaw, mode=mode, yaw_tol=0.15, timeout_s=40.0), p, 0.0)
    robot = SimRobot(Pose2D(1.0, 2.9, 0.0))
    assert job.validate(spec, grid, robot.pose)
    obstacles = ObstacleTracker(p)
    disc = (disc_x, 2.80, 0.12)
    t, k, farthest_back = 0.0, 0, 0.0
    best_x = robot.pose.x
    while not job.done and t < 40.0:
        if k % 2 == 0:
            obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [disc], p), spec, grid)
        robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
        best_x = max(best_x, robot.pose.x)
        farthest_back = max(farthest_back, best_x - robot.pose.x)
        t += DT
        k += 1
    left = math.hypot(GOAL[0] - robot.pose.x, GOAL[1] - robot.pose.y)
    return job, t, left, farthest_back, robot


class NearGoalAcceptTest(unittest.TestCase):
    def test_blocked_just_short_of_the_goal_ends_as_reached(self):
        job, t, left, back, _robot = drive(DriveParams(), 3.55)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn("도착으로 끝냄", job.detail)
        self.assertIn("[%s]" % ex.CODE_POS_NOT_MET, job.detail)   # 허용 밖에 섰다는 것은 늘 적는다
        self.assertGreater(left, 0.10)
        self.assertLessEqual(left, DriveParams().near_goal_accept_m)
        self.assertLess(back, 0.05)                               # 되짚어 물러나지 않는다

    def test_without_it_the_same_goal_is_blocked(self):
        old = DriveParams()
        old.near_goal_accept_m = 0.0
        job, t, left, back, _robot = drive(old, 3.55)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("[%s]" % ex.CODE_OBSTACLE, job.detail)
        new, t_new, _left, _back, _r = drive(DriveParams(), 3.55)
        self.assertLessEqual(t_new, t)                            # 더 오래 걸리지 않는다

    def test_far_from_the_goal_it_is_still_blocked(self):
        job, t, left, back, _robot = drive(DriveParams(), 3.25)   # 물체가 목표 바로 앞 — 목표에서 0.35 m 넘게 떨어져 선다
        self.assertGreater(left, DriveParams().near_goal_accept_m)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)

    def test_turns_to_the_asked_heading_before_it_ends(self):
        job, t, left, back, robot = drive(DriveParams(), 3.55, yaw=-math.pi / 2)   # 물체 반대쪽으로 도는 도착 방향
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn("도착으로 끝냄", job.detail)
        self.assertIn("[%s]" % ex.CODE_POS_NOT_MET, job.detail)
        if "[%s]" % ex.CODE_YAW_NOT_MET not in job.detail:
            self.assertLess(abs(math.atan2(math.sin(robot.pose.yaw + math.pi / 2), math.cos(robot.pose.yaw + math.pi / 2))), 0.2)

    def test_the_no_progress_way_out_ends_as_reached_too(self):
        """물체가 조금 더 멀면 목표 0.11 m 앞에 선다 (허용 0.10 m 를 1 cm 넘김). 길은 있다고 보아 「막힘」 이 아니라 NO_PROGRESS 로 끝나던 장면 —
        그때도 서 있는 물체 앞이고 목표 바로 앞이면 도착으로 끝낸다."""
        old = DriveParams()
        old.near_goal_accept_m = 0.0
        before, _t, left, _back, _robot = drive(old, 3.65)
        self.assertEqual(before.outcome, ex.BLOCKED, before.detail)
        self.assertIn("[%s]" % ex.CODE_NO_PROGRESS, before.detail)
        self.assertLess(left, 0.15)
        job, _t, _left, _back, _robot = drive(DriveParams(), 3.65)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn("도착으로 끝냄", job.detail)

    def test_assault_goals_are_not_changed(self):
        """돌진(거점 진입)은 전과 같다 — 못 들어갔는데 도착이라 하지 않는다."""
        job, t, left, back, _robot = drive(DriveParams(), 3.55, mode=ex.MODE_ASSAULT)
        self.assertNotIn("도착으로 끝냄", job.detail)


if __name__ == "__main__":
    unittest.main()
