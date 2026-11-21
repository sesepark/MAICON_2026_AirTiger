"""자리에 선 뒤 허용을 조금 넘었을 때 · 목표 곁을 지나쳤을 때 돌아서 다시 다가가기를 되풀이하지 않는다 (params.arrive_slack_m · reapproach_slack_m ·
approach_lag_s, 2026-11-20). ROS 없이 돈다.

2026-11-20 01:50 실차 25번 목표 (허용 0.10 m): 굽은 길 끝에서 0.135 ~ 0.154 m 지나쳐 섰고, 목표 쪽으로 돌기 → 0.14 m 가기(또 지나침) → 도착 방향으로 돌기를
되풀이하다 18.2 s 뒤 다른 목표로 바뀌었다.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_drive_core import DT, SimRobot, goal, make_world  # noqa: E402

from navigation.drive import executor as ex  # noqa: E402
from navigation.drive import pose_tracker as loc  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402

GOAL = (5.32, 2.40)
OLD = dict(arrive_slack_m=0.0, reapproach_slack_m=0.0, approach_lag_s=0.0)


class LaggyRobot(SimRobot):
    """명령이 delay_s 늦게 듣는 차체 (그 뒤 1차 지연 0.2 s) — 실차의 반응 지연 흉내."""

    def __init__(self, pose, delay_s):
        super().__init__(pose, tau=0.2)
        self.queue = [ex.STOP] * int(round(delay_s / DT))

    def apply(self, cmd, dt=DT):
        self.queue.append(cmd)
        super().apply(self.queue.pop(0), dt)


def drive(params, delay_s, start, yaw=0.0, max_t=35.0):
    p, spec, grid = make_world(boxes=(), width=6.12, height=4.8, params=params)
    job = ex.GoalExecutor(goal(GOAL[0], GOAL[1], yaw=yaw, pos_tol=0.10, yaw_tol=0.524, timeout_s=29.0), p, 0.0)
    robot = LaggyRobot(start, delay_s)
    assert job.validate(spec, grid, robot.pose)
    obstacles = ObstacleTracker(p)
    t, slowest_near = 0.0, []
    while not job.done and t < max_t:
        cmd = job.step(t, robot.pose, loc.OK, grid, obstacles)
        if job.phase == "driving" and job.remaining is not None and job.remaining < 0.20:
            slowest_near.append(cmd.v)
        robot.apply(cmd)
        t += DT
    return job, robot, t, slowest_near


def left(robot):
    return math.hypot(robot.pose.x - GOAL[0], robot.pose.y - GOAL[1])


class ArrivalOvershootTest(unittest.TestCase):
    START = Pose2D(4.6, 3.0, -0.2)   # 비스듬히 들어온다

    def test_a_slow_reacting_robot_no_longer_circles_the_goal(self):
        before, robot, t_before, _v = drive(DriveParams(**OLD), 0.45, self.START)
        self.assertNotEqual(before.outcome, ex.REACHED, before.detail)      # 전: 지나침 → 돌아서 다시 → 또 지나침 … 20 s 뒤 NO_PROGRESS
        self.assertGreater(t_before, 15.0)
        job, robot, t, _v = drive(DriveParams(), 0.45, self.START)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLess(t, 9.0)
        self.assertLessEqual(left(robot), 0.10 + DriveParams().reapproach_slack_m)

    def test_a_normal_arrival_is_unchanged(self):
        """곧게 들어와 허용 안에 서는 보통의 도착은 걸리는 시간도 선 자리도 전과 같다."""
        start = Pose2D(3.45, 2.76, -0.19)
        before, robot_b, t_before, _v = drive(DriveParams(**OLD), 0.30, start)
        job, robot, t, _v = drive(DriveParams(), 0.30, start)
        self.assertEqual((before.outcome, job.outcome), (ex.REACHED, ex.REACHED), job.detail)
        self.assertAlmostEqual(t, t_before, delta=0.11)
        self.assertLessEqual(left(robot), 0.10 + 1e-6)
        self.assertAlmostEqual(left(robot), left(robot_b), delta=0.01)

    def test_a_curved_final_stretch_is_taken_slower(self):
        yaw = -1.0
        bearing = yaw + math.radians(30.0)                        # 목표가 30° 왼쪽 앞 0.20 m — 서서 돌지 않고 굽어 들어간다
        start = Pose2D(GOAL[0] - 0.20 * math.cos(bearing), GOAL[1] - 0.20 * math.sin(bearing), yaw)

        def first_speed(params):
            p, spec, grid = make_world(boxes=(), width=6.12, height=4.8, params=params)
            job = ex.GoalExecutor(goal(GOAL[0], GOAL[1], pos_tol=0.10, yaw_tol=3.2, timeout_s=29.0), p, 0.0)
            self.assertTrue(job.validate(spec, grid, start))
            return job.step(0.0, start, loc.OK, grid, ObstacleTracker(p)).v

        fast, slow = first_speed(DriveParams(**OLD)), first_speed(DriveParams())
        self.assertGreater(fast, 0.38)                             # 전: 남은 0.20 m 에 0.4 m/s
        self.assertAlmostEqual(slow, (0.20 - 0.10 * ex.STOP_FRACTION) / DriveParams().approach_lag_s, delta=0.01)   # (남은 거리) / 0.5 s
        # 곧게 들어올 때는 전과 같다
        straight = Pose2D(GOAL[0] - 0.20 * math.cos(yaw), GOAL[1] - 0.20 * math.sin(yaw), yaw)
        start = straight
        self.assertAlmostEqual(first_speed(DriveParams()), first_speed(DriveParams(**OLD)), delta=1e-6)

    def _job_at(self, params, pose, reapproaches):
        p, spec, grid = make_world(boxes=(), width=6.12, height=4.8, params=params)
        job = ex.GoalExecutor(goal(GOAL[0], GOAL[1], pos_tol=0.10, yaw_tol=3.2, timeout_s=29.0), p, 0.0)
        assert job.validate(spec, grid, pose)
        job._reapproaches = reapproaches
        dist = math.hypot(GOAL[0] - pose.x, GOAL[1] - pose.y)
        return job, job._check_arrival(1.0, pose, dist, grid)

    def test_after_one_more_try_it_ends_where_it_stands(self):
        facing = Pose2D(GOAL[0] - 0.17, GOAL[1], 0.0)            # 목표가 바로 앞 0.17 m (허용 0.10)
        job, done = self._job_at(DriveParams(), facing, reapproaches=0)
        self.assertFalse(done)                                    # 처음에는 다시 다가간다
        self.assertEqual((job._state, job._reapproaches), ("drive", 1))
        job, done = self._job_at(DriveParams(), facing, reapproaches=1)
        self.assertTrue(done)                                     # 한 번 다시 다가가고도 허용 + 0.10 안이면 끝낸다
        self.assertEqual(job.outcome, ex.REACHED)
        self.assertIn("[%s]" % ex.CODE_POS_NOT_MET, job.detail)
        job, done = self._job_at(DriveParams(), Pose2D(GOAL[0] - 0.25, GOAL[1], 0.0), reapproaches=1)
        self.assertFalse(done)                                    # 그보다 멀면 계속 고친다
        job, done = self._job_at(DriveParams(reapproach_slack_m=0.0), facing, reapproaches=1)
        self.assertFalse(done)                                    # 끄면 전과 같다

    def test_a_little_past_the_goal_ends_there_when_fixing_needs_a_big_turn(self):
        behind = Pose2D(GOAL[0] + 0.17, GOAL[1], 0.0)             # 목표를 0.17 m 지나쳤다 (목표는 바로 뒤)
        job, done = self._job_at(DriveParams(), behind, reapproaches=0)
        self.assertTrue(done)
        self.assertIn("크게 돌아야", job.detail)
        job, done = self._job_at(DriveParams(arrive_slack_m=0.0), behind, reapproaches=0)
        self.assertFalse(done)                                    # 전의 여유(0.05 m)로는 다시 다가갔다 — 01:50 판 25번이 0.154 m


if __name__ == "__main__":
    unittest.main()
