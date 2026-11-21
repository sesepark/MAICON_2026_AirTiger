"""「그 물체를 지나는 길」 에서 물체 곁을 스쳐 지날 때는 옆 틈만큼 속도를 줄인다 (params.through_pass_m, 2026-11-20). ROS 없이 돈다.

2026-11-20 00:00 실차 11번 목표: 서 있는 UGV 에서 0.3 m 인 자리로 가며, 차체 옆 0.05 ~ 0.13 m 에 있던 그 UGV 를 0.45 m/s 로 스쳤다.
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


def drive(params, disc_y):
    """(1.0, 2.9) 에서 곧게 (3.2, 2.9) 로. (2.9, disc_y) 에 반지름 0.12 m 의 서 있는 것 — 목표 칸이 그 임시 장애물에 덮여 「그 물체를 지나는 길」 이 된다.
    돌려주는 것: 목표, 그 물체 곁(x 2.6 ~ 3.0)을 지날 때의 가장 큰 속도, 그 길이 「지나는 길」 이었나."""
    params.goal_shift_m = 0.0   # 「그 물체를 지나는 길」 을 보려는 것이다 — 목표를 곁의 빈 칸으로 옮기는 것(2026-11-20)은 끈다
    params.max_v = 0.45         # 아래 속도 문턱은 max_v 0.45 때의 것이다 (2026-11-20 에 기본이 0.55 로 올랐다)
    p, spec, grid = make_world(boxes=(), params=params)
    job = ex.GoalExecutor(goal(3.2, 2.9, timeout_s=30.0), p, 0.0)
    robot = SimRobot(Pose2D(1.0, 2.9, 0.0))
    assert job.validate(spec, grid, robot.pose)
    obstacles = ObstacleTracker(p)
    disc = (2.9, disc_y, 0.12)
    t, k, fastest, through = 0.0, 0, 0.0, False
    while not job.done and t < 30.0:
        if k % 2 == 0:
            obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [disc], p), spec, grid)
        cmd = job.step(t, robot.pose, loc.OK, grid, obstacles)
        through = through or job._through_obstacle
        if 2.6 <= robot.pose.x <= 3.0:
            fastest = max(fastest, cmd.v)
        robot.apply(cmd)
        t += DT
        k += 1
    return job, fastest, through


class PassGapTest(unittest.TestCase):
    def test_side_gap_of_points_beside_the_path(self):
        p, spec, grid = make_world(boxes=())
        tr = ObstacleTracker(p)
        pose = Pose2D(2.0, 2.5, 0.0)
        tr.on_scan(0.0, pose, [(0.30, 0.20), (0.90, -0.18), (-0.50, 0.15), (0.40, 0.60)], spec, grid)
        # 앞 0.30 m · 옆 0.20 m 의 점: 차체 옆면(0.115)에서 0.085 m. 뒤쪽 점 · 멀리 옆의 점은 세지 않는다
        self.assertAlmostEqual(tr.pass_gap(0.0, 0.0, reach=0.4, width=0.15), 0.085, places=3)
        self.assertAlmostEqual(tr.pass_gap(0.0, 0.0, reach=1.0, width=0.15), 0.065, places=3)   # 더 멀리 보면 앞 0.90 m 의 점
        self.assertEqual(tr.pass_gap(0.0, 0.0, reach=0.4, width=0.05), math.inf)
        self.assertEqual(tr.pass_gap(1.0, 0.0), math.inf)                                         # 스캔이 낡았다


class ThroughPassSpeedTest(unittest.TestCase):
    def test_slows_only_as_much_as_the_side_gap_is_narrow(self):
        old = DriveParams()
        old.through_pass_m = 0.0
        # 옆 틈 0.085 m (물체 가운데가 길에서 0.32 m): 전에는 제 속도, 이제 틈만큼 줄인다 — tight_v 까지는 아니다
        before, fast_before, through = drive(old, 2.58)
        after, fast_after, through_after = drive(DriveParams(), 2.58)
        self.assertTrue(through and through_after)
        self.assertGreater(fast_before, 0.40)
        self.assertLess(fast_after, 0.32)
        self.assertGreater(fast_after, DriveParams().tight_v + 0.03)
        self.assertEqual(after.outcome, before.outcome, after.detail)
        # 옆 틈이 넓으면 (0.185 m) 줄이지 않는다
        _job, fast_wide, through_wide = drive(DriveParams(), 2.48)
        if through_wide:
            self.assertGreater(fast_wide, 0.40)


if __name__ == "__main__":
    unittest.main()
