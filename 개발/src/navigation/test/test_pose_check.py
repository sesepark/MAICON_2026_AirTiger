"""위치 검증(SLAM) · 경기장 밖 상자 금지(map_fusion) · 진행 시계 · 지도 바뀜(주행) 시험 (2026-11-19). ROS 없이 돈다.

본선 경기장 8판 기록에서 나온 것:
  - 정합이 「성공」 이라면서 위치가 0.1 ~ 0.5 m 틀린 구간 (5판). real_scans_1119_180657.json 은 그 18:06 판의 스캔 5개와 그때 내보내던 위치 · 지도다
    (맞던 때 2개, x 가 0.4 m 틀린 때 2개, y 가 0.12 m 틀린 때 2개).
  - 위치가 틀린 동안 map_fusion 이 벽 밖에 상자를 더하고 진짜 드론 상자를 뺐다.
  - 서서 도는 데 152° 에 5 s 가 걸렸다 → 크게 돌면 그것만으로 「8 s 동안 가까워지지 못함」 에 걸린다.
"""
import json
import math
import random
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

import test_drive_core as D
import test_slam_tracking as T
from navigation.drive import executor as ex
from navigation.drive.geometry import Pose2D, wrap
from navigation.map_fusion import FusionParams, MapFusion
from test_node_split import FakeNode, load_slam

DATA = Path(__file__).resolve().parent / 'real_scans_1119_180657.json'


def as_scan(r):
    ranges = [x if x else float('nan') for x in r['ranges']]
    return NS(header=NS(frame_id='base_lidar_link'), ranges=ranges, range_min=0.02, range_max=12.0,
              angle_min=r['angle_min'], angle_increment=r['angle_increment'])


class RealScanWallCheckTest(unittest.TestCase):
    """본선 경기장 18:06 판의 스캔: 틀린 위치에서는 외벽이 진짜 자리를 가리키고, 맞는 위치에서는 가만히 있는다."""

    @classmethod
    def setUpClass(cls):
        cls.slam, _ = load_slam()
        cls.scans = json.loads(DATA.read_text(encoding='utf-8'))['scans']

    def check(self, r):
        s = self.slam
        f = r['field']
        field = NS(width=f['width'], height=f['height'], box_size=f['box_size'], boxes=[NS(x=x, y=y) for x, y in f['boxes']])
        scan, pose = as_scan(r), tuple(r['pose'])
        suspect = s['wall_suspect'](np.array(s['scan_points'](scan, 2), dtype=float), pose, field.width, field.height)
        if suspect is None:
            return None
        second = s['confirm_wall_candidate'](field, scan, pose, suspect[2], boxes=[tuple(b) for b in f['boxes']])
        return None if second is None else (second[0] - pose[0], second[1] - pose[1], wrap(second[2] - pose[2]))

    def test_leaves_a_correct_pose_alone(self):
        for r in self.scans:
            if r['kind'] == 'good':
                self.assertIsNone(self.check(r), r['t'])

    def test_finds_the_real_place_when_the_pose_slid_along_x(self):
        """99초에 0.5 m 를 뛰어 30초 동안 x 가 0.4 m 남짓 틀렸다 (그동안 BLOCKED 4번, 진짜 드론 상자 2개를 뺌)."""
        for r in self.scans:
            if r['kind'] == 'x_off':
                shift = self.check(r)
                self.assertIsNotNone(shift, r['t'])
                self.assertAlmostEqual(shift[0], -0.41, delta=0.06)
                self.assertLess(abs(shift[1]), 0.12)
                self.assertLess(abs(shift[2]), 0.10)

    def test_finds_the_real_place_when_the_pose_crept_off_the_bottom_wall(self):
        """171초부터 끝까지 y 가 0.12 m 틀렸다 (아래 벽이 장애물로 보여 통로에서 75초를 갇힘)."""
        for r in self.scans:
            if r['kind'] == 'y_off':
                shift = self.check(r)
                self.assertIsNotNone(shift, r['t'])
                self.assertAlmostEqual(shift[1], -0.11, delta=0.04)
                self.assertLess(abs(shift[0]), 0.08)

    def test_wall_search_is_cheap(self):
        """스캔마다 도는 계산이다: 자리 625 x 방향 5 개를 훑는 데 이 PC 에서 몇 ms 안쪽."""
        import time
        s, r = self.slam, self.scans[0]
        points = np.array(s['scan_points'](as_scan(r), 2), dtype=float)
        began = time.perf_counter()
        for _ in range(20):
            s['wall_candidate'](points, tuple(r['pose']), 6.12, 4.8)
        self.assertLess((time.perf_counter() - began) / 20, 0.02)


class PoseCheckNodeTest(unittest.TestCase):
    def setUp(self):
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)
        self.module, _ = load_slam()
        self.slam = self.module['SlamNode']()

    def track(self, seconds=2.0, v=0.0, w=0.0, start=(1.0, 2.0, 0.0)):
        """가상 경기장에서 seconds 동안 (v, w) 로 가며 추적시킨다. (로봇, 마지막 시각)."""
        slam = self.slam
        robot = T.Robot(start, odom_offset=0.6, imu_offset=1.0, rng=random.Random(3))
        t, dt = 10.0, 0.05
        slam._receive_odometry(robot.odom(t - 0.1))
        slam._on_wheels(int(round((t - 0.1) * 1e9)), *robot.wheels())
        slam._receive_perception_map(T.arena_map(t - 0.1, (robot.x, robot.y, robot.yaw)))
        for tick in range(1, int(round(seconds / dt)) + 1):
            robot.move(v, w, dt)
            t += dt
            slam._on_wheels(int(round(t * 1e9)), *robot.wheels())
            if tick % 2 == 0:
                slam._receive_odometry(robot.odom(t))
                slam._receive_scan(robot.scan(t))
        return robot, t

    def test_normal_tracking_never_trips_the_checks(self):
        """돌고 가는 보통 주행에서는 뜀 거르기도 벽으로 다시 보기도 켜지지 않는다 (위치도 전과 같이 맞는다)."""
        test = T.SlamTrackingTest('test_tracks_through_turns_when_odom_axes_are_rotated')
        test.slam = self.slam
        errors = test.run_script(odom_offset=math.pi / 2)
        self.assertLess(max(e[0] for e in errors), 0.05)
        checks = self.slam._pose_checks
        self.assertEqual((checks['jump_rejected'], checks['jump_forced'], checks['relocated']), (0, 0, 0), checks)

    def test_a_match_farther_than_the_wheels_rolled_is_a_jump(self):
        robot, t = self.track(seconds=2.0, v=0.3)
        now = int(round(t * 1e9))
        anchor = self.slam._scan_anchor[1]
        self.assertIsNone(self.slam._jumped(now + 100_000_000, (anchor[0] + 0.10, anchor[1], anchor[2])))       # 고치는 정도
        self.assertIsNotNone(self.slam._jumped(now + 100_000_000, (anchor[0] + 0.50, anchor[1], anchor[2])))    # 0.1 s 에 0.5 m

    def test_relocates_after_the_walls_point_elsewhere_five_scans_in_a_row(self):
        """틀린 자리(진짜에서 0.4 m)에 걸려 있을 때: 외벽이 5번 이어 같은 쪽을 가리키면 그 자리로 옮기고, 그 사이 고친 지도를 되돌린다."""
        robot, t = self.track(seconds=1.0)
        slam = self.slam
        field = slam.latest_perception_map
        wrong = (robot.x + 0.4, robot.y, robot.yaw)
        slam._fusion._added.append(self._promoted(2.6, 2.0, int(t * 1e9)))        # 틀린 동안 더한 상자
        moved = None
        for k in range(self.module['WALL_STREAK']):
            self.assertIsNone(moved)
            stamp = int(round((t + 0.1 * (k + 1)) * 1e9))
            moved = slam._wall_recheck(field, robot.scan(t + 0.1 * (k + 1)), wrong, stamp)
            if moved is None:
                self.assertTrue(slam._wall_streak)       # 의심하는 동안: 지도를 고치지 않는다
        self.assertIsNotNone(moved)
        self.assertLess(math.hypot(moved[0] - robot.x, moved[1] - robot.y), 0.05)
        self.assertEqual(slam._pose_checks['relocated'], 1)
        self.assertEqual(slam._fusion.added(), [])

    def test_does_not_relocate_on_a_single_odd_scan(self):
        robot, t = self.track(seconds=1.0)
        slam = self.slam
        wrong = (robot.x + 0.4, robot.y, robot.yaw)
        right = (robot.x, robot.y, robot.yaw)
        field = slam.latest_perception_map
        self.assertIsNone(slam._wall_recheck(field, robot.scan(t + 0.1), wrong, int((t + 0.1) * 1e9)))
        self.assertIsNone(slam._wall_recheck(field, robot.scan(t + 0.2), right, int((t + 0.2) * 1e9)))
        self.assertEqual(slam._wall_streak, [])
        self.assertEqual(slam._pose_checks['relocated'], 0)

    def test_pose_check_can_be_turned_off(self):
        self.slam.pose_check = False
        robot, t = self.track(seconds=1.0, v=0.2)
        self.assertEqual(self.slam._pose_checks['wall_suspect'], 0)
        self.assertIsNotNone(self.slam._scan_anchor)

    @staticmethod
    def _promoted(x, y, first_ns):
        from navigation.map_fusion import _Track
        return _Track(x, y, promoted=True, seen=5, first_ns=first_ns, last_ns=first_ns, span=0.35, score=3.0)


WIDTH, HEIGHT, BOX = 6.12, 4.8, 0.4


def wall_and_line(pose, line_y, x0=2.8, x1=3.2):
    """아래 벽(y = 0)과 위 벽을 본 점 + y = line_y 의 가로 선(폭 x1 − x0)을 본 점. 차체 좌표 (자세는 방향 0)."""
    xs = np.linspace(0.5, 5.5, 60)
    world = [(x, 0.0) for x in xs if not x0 - 0.3 < x < x1 + 0.3] + [(x, HEIGHT) for x in xs]
    world += [(x, line_y) for x in np.linspace(x0, x1, 21)]
    world = np.array(world, dtype=float)
    return world - np.array([pose[0], pose[1]])


class OutsideArenaTest(unittest.TestCase):
    def observe(self, fusion, line_y, seconds=3.0):
        pose = (3.0, 1.5, 0.0)
        for k in range(int(seconds / 0.2)):
            fusion.observe(wall_and_line(pose, line_y), pose, int((1.0 + 0.2 * k) * 1e9))

    def test_a_wall_seen_a_little_inside_does_not_become_a_box_outside_the_arena(self):
        """위치가 0.1 m 틀려 아래 벽이 y = 0.10 에 보인다: 그 면에 맞는 상자는 가운데가 y = −0.10 — 벽 밖이라 더하지 않는다.
        (2026-11-19 18:06 판: 그렇게 벽 밖 상자 4개가 더해져 정합을 틀린 자리에 묶었다)"""
        fusion = MapFusion()
        fusion.set_map([], BOX, WIDTH, HEIGHT)
        self.observe(fusion, 0.10)
        self.assertEqual(fusion.added(), [])
        loose = MapFusion(FusionParams(wall_overlap_m=10.0))     # 금지를 끄면 전처럼 더해진다 (이 시험이 그 규칙을 보고 있다는 확인)
        loose.set_map([], BOX, WIDTH, HEIGHT)
        self.observe(loose, 0.10)
        self.assertEqual(len(loose.added()), 1)
        self.assertLess(loose.added()[0][1], 0.0)

    def test_a_box_standing_against_the_wall_is_still_added(self):
        fusion = MapFusion()
        fusion.set_map([], BOX, WIDTH, HEIGHT)
        self.observe(fusion, 0.42)          # 벽에 붙은 상자의 앞면 (가운데 y = 0.22)
        self.assertEqual(len(fusion.added()), 1)
        self.assertAlmostEqual(fusion.added()[0][1], 0.22, delta=0.04)

    def test_distrust_undoes_what_was_changed_since(self):
        """위치가 틀렸던 것이 드러나면: 그 뒤에 더한 상자는 지우고, 뺀 드론 상자는 되살리고, 그 전에 한 것은 그대로 둔다."""
        fusion = MapFusion()
        fusion.set_map([(1.0, 1.0), (4.0, 3.0)], BOX, WIDTH, HEIGHT)
        self.observe(fusion, 0.42, seconds=3.0)                       # 1.0 ~ 3.8 s: 벽에 붙은 상자를 더한다
        early = fusion.added()
        self.assertEqual(len(early), 1)
        old, new = fusion._drone
        old.rejected, old.changed_ns = True, int(2.0e9)               # 2 s 에 뺀 것
        new.rejected, new.changed_ns = True, int(9.0e9)               # 9 s 에 뺀 것
        from navigation.map_fusion import _Track
        fusion._added.append(_Track(2.0, 3.0, promoted=True, first_ns=int(8.5e9), last_ns=int(9e9)))
        self.assertTrue(fusion.distrust(int(8.0e9)))
        self.assertEqual(fusion.added(), early)
        self.assertEqual(fusion.drone_states(), ['rejected', 'drone'])
        self.assertFalse(fusion.distrust(int(8.0e9)))                 # 더 되돌릴 것이 없다

    def test_structure_can_leave_out_lidar_added_boxes(self):
        fusion = MapFusion()
        fusion.set_map([(1.0, 1.0)], BOX, WIDTH, HEIGHT)
        self.observe(fusion, 0.42)
        self.assertEqual(len(fusion.structure()[0]), 2)
        self.assertEqual(len(fusion.structure(added=False)[0]), 1)


class SlowTurner(D.SimRobot):
    """서서 도는 속도가 실제 로봇처럼 느린 차체 (본선 경기장: 0.36 ~ 0.50 rad/s)."""

    def __init__(self, pose, turn_rate):
        super().__init__(pose)
        self.turn_rate = turn_rate

    def apply(self, cmd, dt=D.DT):
        if abs(cmd.v) < 0.02:
            cmd = ex.Command(0.0, max(-self.turn_rate, min(self.turn_rate, cmd.w)))
        super().apply(cmd, dt)


class ProgressClockTest(unittest.TestCase):
    def drive(self, turn_rate, params=None):
        p, spec, grid = D.make_world(boxes=(), params=params)
        job = ex.GoalExecutor(D.goal(1.5, 2.5, timeout_s=60.0), p, 0.0)
        robot = SlowTurner(Pose2D(3.0, 2.5, 0.0), turn_rate)       # 목표는 바로 뒤 1.5 m: 180° 를 돌아야 한다
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = D.run(job, robot, spec, grid, p)
        return job, trace

    def test_a_slow_real_turn_is_not_no_progress(self):
        """0.25 rad/s 로 180° 를 도는 데 12.6 s. 전에는 8 s 에 BLOCKED [NO_PROGRESS] 였다 — 실제로 돌고 있는 동안은 한도를 늘린다."""
        job, trace = self.drive(0.25)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        from navigation.drive.params import DriveParams
        old, _ = self.drive(0.25, DriveParams(no_progress_extra_s=0.0))
        self.assertEqual(old.outcome, ex.BLOCKED, old.detail)       # 늘리지 않으면 전처럼 끝난다 (이 시험이 그 규칙을 보고 있다는 확인)
        self.assertIn(ex.CODE_NO_PROGRESS, old.detail)

    def test_a_robot_that_cannot_turn_still_gives_up_at_the_same_time(self):
        """돌라고 해도 안 도는 때(본선 경기장 17:05 · 17:38 판)는 늘리지 않는다: 전과 같은 때에 NO_PROGRESS."""
        from navigation.drive.params import DriveParams
        job, trace = self.drive(0.05)
        old, old_trace = self.drive(0.05, DriveParams(no_progress_extra_s=0.0))
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertEqual(job.detail, old.detail)
        self.assertAlmostEqual(trace[-1][0], old_trace[-1][0], delta=0.01)

    def test_the_extra_time_is_capped(self):
        p, spec, grid = D.make_world(boxes=())
        job = ex.GoalExecutor(D.goal(1.5, 2.5), p, 0.0)
        job.phase = 'rotating'
        pose, t = Pose2D(3.0, 2.5, 0.0), 0.0
        for _ in range(100):                      # 50 s 동안 계속 도는 것으로 보여도
            t += 0.5
            pose = Pose2D(3.0, 2.5, wrap(pose.yaw + 0.3))
            job._credit_motion(t, pose)
        self.assertAlmostEqual(job._credit, p.no_progress_extra_s)


class MapChangeTest(unittest.TestCase):
    def test_goal_is_moved_again_when_a_box_lands_on_it(self):
        """가는 중에 지도의 상자가 목표 곁으로 옮겨졌다: 목표 칸을 곁의 빈 칸으로 다시 옮겨 끝까지 간다 (전: 서 있다가 NO_PROGRESS)."""
        p, spec, grid = D.make_world(boxes=((3.0, 4.0),))
        job = ex.GoalExecutor(D.goal(4.0, 2.5, timeout_s=60.0), p, 0.0)
        robot = D.SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        world = {'spec': spec, 'grid': grid}

        def on_tick(t, robot, cmd):
            if world['grid'] is grid and t >= 1.0:
                _, world['spec'], world['grid'] = D.make_world(boxes=((4.0, 2.75),))    # 목표에서 5 cm 에 상자 면
                job.on_grid_changed(world['grid'])

        obstacles = D.ObstacleTracker(p)
        t = 0.0
        while not job.done and t < 40.0:
            cmd = job.step(t, robot.pose, D.loc.OK, world['grid'], obstacles)
            on_tick(t, robot, cmd)
            robot.apply(cmd)
            t += D.DT
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertTrue(world['grid'].free(*job.target))
        self.assertLess(math.hypot(robot.pose.x - 4.0, robot.pose.y - 2.5), 0.35)

    def test_obstacle_memory_survives_a_box_change_but_not_a_new_arena(self):
        from test_node_split import load_driving
        module, _ = load_driving()
        node = module.DrivingNode()
        node._receive_map(T.arena_map(1.0, (1.0, 2.0, 0.0)))
        node.obstacles._seeds[(10, 10)] = None
        cleared = []
        node.obstacles.clear = lambda: cleared.append(True)
        node._receive_map(T.arena_map(2.0, (1.0, 2.0, 0.0), boxes=T.BOXES[:-1]))          # 상자 하나가 빠졌다
        self.assertEqual(cleared, [])
        other = T.arena_map(3.0, (1.0, 2.0, 0.0))
        other.height = 4.8                                                                  # 경기장 치수가 바뀌었다
        node._receive_map(other)
        self.assertEqual(cleared, [True])


if __name__ == '__main__':
    unittest.main()
