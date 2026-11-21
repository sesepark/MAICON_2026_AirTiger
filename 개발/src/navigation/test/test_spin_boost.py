"""서서 돌기를 실제 회전에 맞춰 올리기 (2026-11-19 밤, M5). ROS 없이 돈다.

실차 기록(본선 경기장 10판): 서서 도는 명령 1.6 rad/s 는 실제 가운데값 0.42 ~ 0.47 rad/s, 1.4 밑은 거의 안 돈다. 21:53 판은 목표 시간의 45 % 가
서서 돌기 · 방향 맞추기였다. 가상 차체 StickyRobot 은 그 실측을 흉내 낸다 (서서 돌 때 dead 밑은 안 돌고 명령의 scale 배).
**1.6 rad/s 넘는 명령에서 실차가 얼마나 도는지는 기록에 없다** — 여기의 scale 은 가정이다. 그래서 시험은 「모자라면 올리고, 충분하면 안 올린다」 만 본다.
"""
import math
import unittest

import test_drive_core as D
from navigation.drive import executor as ex
from navigation.drive.geometry import Pose2D, advance, wrap
from navigation.drive.params import DriveParams
from navigation.drive.turn_rate import SpinRate


class StickyRobot(D.SimRobot):
    """서서 돌 때(v ≈ 0) dead 밑의 명령은 안 돌고, 그 위는 명령의 scale 배로 돈다. 가면서 도는 것은 명령대로."""

    def __init__(self, pose, dead=1.4, scale=0.27, tau=0.1):
        super().__init__(pose, tau)
        self.dead, self.scale = dead, scale

    def apply(self, cmd, dt=D.DT):
        w = cmd.w
        if abs(cmd.v) < 0.03:
            w = 0.0 if abs(w) < self.dead else w * self.scale
        k = min(1.0, dt / self.tau)
        self.v += (cmd.v - self.v) * k
        self.w += (w - self.w) * k
        self.pose = advance(self.pose, self.v, self.w, dt)


class SpinRateTest(unittest.TestCase):
    def feed(self, rate, slam_w=0.5, gyro_w=None, seconds=6.0):
        yaw, t = 0.0, 0.0
        while t < seconds:
            rate.on_slam_yaw(t, wrap(yaw))
            for k in range(2):   # IMU 20 Hz
                if gyro_w is not None:
                    rate.on_gyro(t + 0.05 * k, gyro_w)
            yaw += slam_w * 0.1
            t += 0.1
        return t

    def test_rate_from_slam_heading(self):
        r = SpinRate()
        t = self.feed(r, slam_w=0.5)
        self.assertAlmostEqual(r.rate(t - 0.1), 0.5, delta=0.03)
        self.assertEqual(r.source, 'slam')

    def test_gyro_is_used_only_after_it_agrees_with_slam(self):
        r = SpinRate()
        self.assertFalse(r.gyro_trusted)
        t = self.feed(r, slam_w=0.5, gyro_w=-0.52)   # 부호가 반대여도 크기가 맞으면 된다 (장착 방향)
        self.assertTrue(r.gyro_trusted, r.summary())
        self.assertAlmostEqual(r.rate(t - 0.1), 0.52, delta=0.02)
        self.assertEqual(r.source, 'gyro')

    def test_a_frozen_gyro_is_never_used(self):
        """launch 주석대로 IMU 값이 고정돼 있으면(정지 때의 −0.006 rad/s 그대로) 쓰지 않고 SLAM 방향으로 잰다."""
        r = SpinRate()
        t = self.feed(r, slam_w=0.5, gyro_w=-0.006)
        self.assertFalse(r.gyro_trusted)
        self.assertGreaterEqual(r.gyro_bad, 3)
        self.assertAlmostEqual(r.rate(t - 0.1), 0.5, delta=0.03)
        self.assertEqual(r.source, 'slam')

    def test_falls_back_to_pose_samples_and_none_when_stale(self):
        r = SpinRate()
        for k in range(12):
            r.on_pose_yaw(0.05 * k, 0.02 * k)
        self.assertAlmostEqual(r.rate(0.55), 0.4, delta=0.03)
        self.assertEqual(r.source, 'pose')
        self.assertIsNone(r.rate(2.0))


def boosted(**more):
    """올리기를 켠 파라미터 (기본은 끔 — params.spin_max_w)."""
    p = DriveParams()
    p.rotate_max_w, p.rotate_kick_w = 2.2, 2.8   # 이 시험들을 쓴 때의 값 (2026-11-20 에 기본이 2.6 · 3.0 으로 올랐다)
    p.spin_max_w = 2.8   # rotate_max_w(2.2) 보다 커야 켜진다
    for name, value in more.items():
        setattr(p, name, value)
    return p


def spin_goal(yaw, **more):
    return D.goal(3.0, 2.5, yaw=yaw, yaw_tol=0.17, timeout_s=40.0, **more)


class SpinBoostTest(unittest.TestCase):
    def turn(self, robot, params=None, boxes=(), yaw=math.pi, start=(3.0, 2.5, 0.0)):
        p, spec, grid = D.make_world(boxes=boxes, params=params)
        job = ex.GoalExecutor(spin_goal(yaw), p, 0.0)
        robot.pose = Pose2D(*start)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = D.run(job, robot, spec, grid, p, max_t=45.0)
        spins = [abs(c.w) for _t, _pose, c, _ph, _b in trace if abs(c.v) < 1e-6 and abs(c.w) > 1e-6]
        return job, trace, p, spins

    def test_slow_robot_gets_a_stronger_command_and_turns_sooner(self):
        off = DriveParams()                 # 기본 = 끔 (전과 같음)
        self.assertLessEqual(off.spin_max_w, off.rotate_max_w)
        job0, trace0, p0, spins0 = self.turn(StickyRobot(Pose2D(0, 0, 0)), off)
        self.assertIsNotNone(job0.turn.last_rate, "꺼 두어도 실제 빠르기는 잰다")
        job1, trace1, p1, spins1 = self.turn(StickyRobot(Pose2D(0, 0, 0)), boosted())
        self.assertEqual(job0.outcome, ex.REACHED, job0.detail)
        self.assertEqual(job1.outcome, ex.REACHED, job1.detail)
        self.assertLessEqual(max(spins0), off.rotate_max_w + 1e-6)
        self.assertGreater(max(spins1), p1.rotate_max_w + 0.3)
        self.assertLessEqual(max(spins1), p1.spin_max_w + 1e-6)
        self.assertLess(trace1[-1][0], trace0[-1][0], "180° 를 더 빨리 돈다")
        self.assertLess(abs(wrap(math.pi - trace1[-1][1].yaw)), 0.17 + 0.02, "넘어가지 않고 허용 안에 선다")
        self.assertNotIn("YAW_NOT_MET", job1.detail)

    def test_robot_that_turns_as_commanded_is_not_boosted(self):
        job, _trace, p, spins = self.turn(D.SimRobot(Pose2D(0, 0, 0)), boosted())
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLessEqual(max(spins), p.rotate_max_w + 1e-6)

    def test_no_boost_beside_a_box(self):
        """상자 면에서 0.22 m (차체 반 대각선 0.17 + spin_clear_m 0.08 안): 걸렸을 수 있는 자리에서는 더 세게 밀지 않는다."""
        job, _trace, p, spins = self.turn(StickyRobot(Pose2D(0, 0, 0)), boosted(), boxes=((3.0, 2.08),), yaw=math.pi / 2)
        self.assertTrue(spins)
        self.assertLessEqual(max(spins), p.rotate_max_w + 1e-6)

    def test_boost_is_carried_to_the_next_goal(self):
        p, spec, grid = D.make_world(boxes=(), params=boosted())
        model = ex.TurnModel()
        robot = StickyRobot(Pose2D(3.0, 2.5, 0.0))
        first = []
        for yaw in (math.pi, 0.0):
            job = ex.GoalExecutor(spin_goal(yaw), p, 0.0, model)
            self.assertTrue(job.validate(spec, grid, robot.pose))
            trace = D.run(job, robot, spec, grid, p, max_t=45.0)
            spins = [abs(c.w) for _t, _pose, c, _ph, _b in trace if abs(c.v) < 1e-6 and abs(c.w) > 1e-6]
            first.append(spins[0])
        self.assertGreater(model.boost, 0.2)
        self.assertGreater(first[1], first[0] + 0.1, "두 번째 목표는 배운 양에서 시작한다")

    def test_path_rotation_is_boosted_too(self):
        """출발하며 목표 쪽으로 도는 회전(rotating)도 같다."""
        p, spec, grid = D.make_world(boxes=(), params=boosted())
        job = ex.GoalExecutor(D.goal(1.0, 2.5, timeout_s=40.0), p, 0.0)
        robot = StickyRobot(Pose2D(3.0, 2.5, 0.0))   # 목표가 등 뒤
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = D.run(job, robot, spec, grid, p, max_t=45.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        spins = [abs(c.w) for _t, _pose, c, ph, _b in trace if ph == "rotating"]
        self.assertGreater(max(spins), p.rotate_max_w + 0.3)

    def test_path_rotation_close_to_the_goal_is_not_boosted(self):
        """목표 바로 앞(남은 경로 spin_far_m 안)의 회전은 전과 같다 — 도착 자세가 달라지지 않게."""
        p, spec, grid = D.make_world(boxes=(), params=boosted())
        job = ex.GoalExecutor(D.goal(2.5, 2.5, timeout_s=40.0), p, 0.0)
        robot = StickyRobot(Pose2D(3.0, 2.5, 0.0))   # 0.5 m 뒤
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = D.run(job, robot, spec, grid, p, max_t=45.0)
        spins = [abs(c.w) for _t, _pose, c, ph, _b in trace if ph == "rotating"]
        self.assertTrue(spins)
        self.assertLessEqual(max(spins), p.rotate_max_w + 1e-6)


if __name__ == '__main__':
    unittest.main()
