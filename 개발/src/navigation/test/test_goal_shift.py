"""서 있는 물체가 목표 칸을 덮고 있으면 목표 곁의 빈 칸으로 옮겨 간다 (params.goal_shift_m, 2026-11-20). ROS 없이 돈다.

2026-11-20 01:50 실차: 가운데 거점 안 (5.79, 2.31) 에 다른 UGV 가 서 있었다 — 그 거점으로의 돌진이 네 번 BLOCKED, 첫 돌진부터 들어가기까지 36 s.
같은 판 정찰 22번: 서 있는 UGV (3.04, 2.56) 에서 0.33 m 인 자리 (3.23, 2.83) 에 닿았지만 그 UGV 에 걸려 도착 방향으로 못 돌았다.
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
from navigation.drive.fieldmap import MapSpec, fixed_pillars  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.grid import Grid  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.ugvs import UgvTracks  # noqa: E402

UGV_R = 0.13   # 라이다에 보이는 그 UGV 의 크기 (원으로 흉내)


def drive(params, start, target, ugv, mode=ex.MODE_NORMAL, yaw=None, drone=True, leaves_at=None, max_t=40.0, pillars=False):
    """start 에서 target 으로. ugv = 서 있는 다른 UGV 의 가운데 — 드론은 출발 30 s 전부터 그 자리에서 보았고(drone), 라이다에는 원으로 찍힌다.
    leaves_at 이 있으면 그때 그 UGV 가 사라진다 (드론 · 라이다 모두). pillars: 지도에 고정 기둥(거점의 양 끝)을 넣는다 — 실제 경기장이 그렇다.
    돌려주는 것: 목표, 로봇, 그 UGV 와 가장 가까웠던 거리, 옮긴 적이 있나."""
    p, spec, grid = make_world(boxes=(), width=6.12, height=4.8, params=params)
    if pillars:
        spec = MapSpec(6.12, 4.8, 0.40, (), "arena", fixed=fixed_pillars(6.12, 4.8))
        grid = Grid(spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
    job = ex.GoalExecutor(goal(target[0], target[1], yaw=yaw, mode=mode, yaw_tol=0.5, timeout_s=max_t), p, 0.0)
    robot = SimRobot(start)
    assert job.validate(spec, grid, robot.pose), job.detail
    obstacles = ObstacleTracker(p)
    obstacles.ugvs = UgvTracks(p)
    me = (start.x, start.y)
    if drone:
        for k in range(-30, 1):
            obstacles.ugvs.on_frame(float(k), [me, ugv], [me], local=me, pose=me)
    t, k, nearest, shifted = 0.0, 0, math.inf, False
    while not job.done and t < max_t:
        there = leaves_at is None or t < leaves_at
        me = (robot.pose.x, robot.pose.y)
        if k % 20 == 0 and k and drone:   # 드론 사진 1 s 마다
            obstacles.ugvs.on_frame(t, [me] + ([ugv] if there else []), [me], local=me, pose=me)
        if k % 2 == 0:
            obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [ugv + (UGV_R,)] if there else [], p), spec, grid)
        robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
        shifted = shifted or job._shift_home is not None
        if there:
            nearest = min(nearest, math.hypot(robot.pose.x - ugv[0], robot.pose.y - ugv[1]))
        t += DT
        k += 1
    return job, robot, nearest, shifted


class GoalShiftTest(unittest.TestCase):
    BASE_UGV = (5.79, 2.31)             # 01:50 판: 가운데 거점 안에 서 있던 UGV
    BASE_START = Pose2D(5.38, 2.30, math.radians(33))   # 33번 돌진을 받은 자리
    BASE_GOAL = (5.92, 2.60)            # 그 UGV 에서 0.32 m

    def test_assault_beside_a_ugv_parked_in_the_base_goes_in_beside_it(self):
        old = DriveParams()
        old.goal_shift_m = 0.0
        before, _robot, _near, _shifted = drive(old, self.BASE_START, self.BASE_GOAL, self.BASE_UGV, mode=ex.MODE_ASSAULT)
        self.assertEqual(before.outcome, ex.BLOCKED, before.detail)          # 전: 그 UGV 쪽으로 가다 앞에 서서 BLOCKED (실차 33번)
        job, robot, nearest, shifted = drive(DriveParams(), self.BASE_START, self.BASE_GOAL, self.BASE_UGV, mode=ex.MODE_ASSAULT)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertTrue(shifted)
        self.assertIn("옮겨 감", job.detail)
        left = math.hypot(robot.pose.x - self.BASE_GOAL[0], robot.pose.y - self.BASE_GOAL[1])
        self.assertLessEqual(left, DriveParams().assault_accept_m + 0.10)     # 거점 밖으로 멀리 옮기지 않는다
        self.assertGreater(robot.pose.x, 5.75)                                # 벽 쪽(거점 안)으로 들어갔다
        self.assertGreater(nearest, 0.30)                                     # 그 UGV 에 닿지 않는다 (가운데끼리 0.25 ~ 0.34 m 면 닿는다)

    def test_assault_goal_in_the_middle_of_the_parked_ugv_is_not_moved_far(self):
        """목표가 그 UGV 몸통 한가운데면(실차 28 · 29 · 32번) assault_accept_m 안에 빈 칸이 없다 — 옮기지 않고 전처럼 빨리 BLOCKED."""
        job, robot, _near, shifted = drive(DriveParams(), self.BASE_START, (5.92, 2.40), self.BASE_UGV, mode=ex.MODE_ASSAULT)
        self.assertFalse(shifted)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("서 있는 UGV", job.detail)

    def test_assault_keeps_clear_of_a_ugv_parked_in_the_left_base(self):
        """02:39 판: 거점 안 (0.36, 2.43) 에 선 UGV 에서 0.39 m 인 (0.20, 2.78) 로의 돌진 — 실차는 그 UGV 의 윗모서리가 진행 띠에 걸려 BLOCKED 였다.
        (가상의 UGV 는 원이라 전 흐름도 닿기는 한다 — 여기서는 옮긴 길이 그 UGV 에서 더 떨어져 지나는지를 본다.)"""
        ugv, start, target = (0.36, 2.43), Pose2D(0.80, 2.98, math.radians(-88)), (0.20, 2.78)
        old = DriveParams()
        old.goal_shift_m = 0.0
        _before, _robot, near_before, _shifted = drive(old, start, target, ugv, mode=ex.MODE_ASSAULT, max_t=25.0)
        job, robot, nearest, shifted = drive(DriveParams(), start, target, ugv, mode=ex.MODE_ASSAULT, max_t=25.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertTrue(shifted)
        self.assertGreater(nearest, near_before + 0.08)
        self.assertGreater(nearest, 0.40)
        self.assertLess(robot.pose.x, 0.40)                                   # 거점 깊이(벽에서 0.40 m) 안
        self.assertLessEqual(math.hypot(robot.pose.x - target[0], robot.pose.y - target[1]), DriveParams().assault_accept_m)

    def test_assault_goal_is_not_moved_away_from_the_wall(self):
        """04:48 판(통합 쪽에서 넣음): 가운데 거점 안 (0.14, 2.43) 에 선 UGV, 돌진 목표 (0.20, 2.78). 거점의 양 끝은 고정 기둥이라 그 깊이에서 옆으로는
        빈 칸이 없다 — 가장 가까운 빈 칸은 벽에서 먼 쪽 (0.43, 2.83) 이고, 그리로 옮기면 차체가 벽에서 0.40 m 넘는 곳에 서서 REACHED 다
        (supervisor 는 벽에서 0.30 m 안이어야 들어간 것으로 친다 — 실차는 자리를 바꿔 네 번 시도해 게시 뒤 29.6 s 에 들어갔다)."""
        ugv, start, target = (0.14, 2.43), Pose2D(0.81, 2.77, math.radians(-158)), (0.20, 2.78)
        old = DriveParams()
        old.assault_shift_out_m = -1.0
        before, was, _near, moved = drive(old, start, target, ugv, mode=ex.MODE_ASSAULT, max_t=25.0, pillars=True)
        self.assertEqual(before.outcome, ex.REACHED, before.detail)           # 전: 「도착」 이라는데 거점 깊이에 못 미쳤다 (실차 23번과 같다)
        self.assertTrue(moved)
        self.assertGreater(was.pose.x, 0.40)
        job, robot, nearest, _shifted = drive(DriveParams(), start, target, ugv, mode=ex.MODE_ASSAULT, max_t=25.0, pillars=True)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLessEqual(robot.pose.x, 0.30)                              # 벽에서 0.30 m 안 = supervisor 가 들어갔다고 치는 깊이
        self.assertNotIn("옮겨 감", job.detail)
        self.assertGreater(nearest, 0.30)                                     # 그 UGV 에 닿지 않는다

    def test_assault_goal_still_moves_along_the_wall(self):
        """벽과 나란히 옮길 빈 칸이 있으면 전처럼 옮긴다 (01:50 판 33번을 고정 기둥이 있는 지도에서) — 깊이는 그대로다."""
        job, robot, nearest, shifted = drive(DriveParams(), self.BASE_START, self.BASE_GOAL, self.BASE_UGV, mode=ex.MODE_ASSAULT, pillars=True)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertTrue(shifted)
        self.assertGreaterEqual(robot.pose.x, 6.12 - 0.30)
        self.assertGreater(nearest, 0.30)

    def test_scout_spot_beside_a_parked_ugv_is_reached_with_the_asked_heading(self):
        ugv, spot, yaw = (3.04, 2.56), (3.23, 2.83), -0.47
        job, robot, nearest, shifted = drive(DriveParams(), Pose2D(5.18, 2.86, -1.52), spot, ugv, yaw=yaw)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertTrue(shifted)
        self.assertNotIn(ex.CODE_YAW_NOT_MET, job.detail)                     # 도착 방향으로 돌 자리가 있다
        self.assertNotIn(ex.NO_ROOM_AT_GOAL, job.detail)
        self.assertLess(abs(math.atan2(math.sin(robot.pose.yaw - yaw), math.cos(robot.pose.yaw - yaw))), 0.5 + 1e-6)
        self.assertLessEqual(math.hypot(robot.pose.x - spot[0], robot.pose.y - spot[1]), DriveParams().goal_shift_m + 0.05)
        self.assertGreater(nearest, 0.34)

    def test_goes_back_to_the_asked_spot_when_the_ugv_leaves(self):
        ugv, spot = (3.04, 2.56), (3.23, 2.83)
        job, robot, _near, shifted = drive(DriveParams(), Pose2D(5.6, 2.86, math.pi), spot, ugv, leaves_at=0.5)
        self.assertTrue(shifted)                                              # 처음에는 옮겼다가
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertNotIn("옮겨 감", job.detail)                                # 그 UGV 가 떠나자 원래 자리로
        self.assertNotIn(ex.CODE_POS_NOT_MET, job.detail)
        self.assertLessEqual(math.hypot(robot.pose.x - spot[0], robot.pose.y - spot[1]), 0.10 + 1e-6)

    def test_a_thing_that_has_only_just_appeared_does_not_move_the_goal(self):
        """드론 자리가 없고 라이다에 막 찍히기 시작한 것(지나가는 물체일 수 있다)은 서 있는 물체가 아니다 — tight_still_s 가 지나기 전에는 옮기지 않는다."""
        p, spec, grid = make_world(boxes=(), width=6.12, height=4.8, params=DriveParams())
        job = ex.GoalExecutor(goal(3.23, 2.83, timeout_s=20.0), p, 0.0)
        robot = SimRobot(Pose2D(4.6, 2.86, math.pi))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        obstacles = ObstacleTracker(p)
        t, k = 0.0, 0
        while t < p.tight_still_s - 0.3:
            if k % 2 == 0:
                obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [(3.04, 2.56, UGV_R)], p), spec, grid)
            robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
            self.assertIsNone(job._shift_home)
            t += DT
            k += 1


if __name__ == "__main__":
    unittest.main()
