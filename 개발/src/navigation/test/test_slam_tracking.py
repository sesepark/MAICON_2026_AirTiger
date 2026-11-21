"""SLAM 노드(navigation_slam)의 자기 위치 추적 시험. rclpy 없이 가짜 모듈 + 간단한 경기장 시뮬로 돈다.

실제 로봇과 같게 만든 조건 (2026-11-18 실측):
- /odom orientation 은 항상 (0,0,0,1). x·y 는 바퀴 기준 내부 방향으로 적분되고, 그 축은 경기장 축과 돌아 있다.
- /scan 은 base_lidar_link (x 0.0398 m, yaw +90°), 10 Hz, 약 505점 중 뒤쪽이 가려진다.
- /imu/data yaw 는 실제 회전을 따라가지만 기준(0)은 경기장과 다르다.
"""
import math
import random
import unittest
from types import SimpleNamespace as NS

from test_node_split import FakeNode, FieldMap, load_slam

WIDTH, HEIGHT, BOX = 6.12, 5.0, 0.4
WHEEL_BASE = 0.175   # 시뮬 로봇의 바퀴 간격 = 노드 기본값 (base_node.cpp)
BOXES = ((1.2, 1.0), (1.2, 4.0), (2.2, 2.5), (3.06, 1.2), (3.06, 3.8), (3.9, 2.5), (4.9, 1.0), (4.9, 4.0))


def ray(x, y, angle):
    dx, dy = math.cos(angle), math.sin(angle)
    best = 25.0
    half = BOX / 2
    for bx, by in BOXES:
        near, far, hit = -1e9, 1e9, True
        for origin, direction, low, high in ((x, dx, bx - half, bx + half), (y, dy, by - half, by + half)):
            if abs(direction) < 1e-12:
                hit = hit and low <= origin <= high
                continue
            a, b = (low - origin) / direction, (high - origin) / direction
            near, far = max(near, min(a, b)), min(far, max(a, b))
        if hit and near <= far and near > 0:
            best = min(best, near)
    for t in ((-x / dx) if dx < 0 else ((WIDTH - x) / dx) if dx > 0 else 1e9,
              (-y / dy) if dy < 0 else ((HEIGHT - y) / dy) if dy > 0 else 1e9):
        if t > 0:
            best = min(best, t)
    return best


def stamp(t):
    sec = int(t)
    return NS(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def quat(yaw):
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


class Robot:
    """경기장 자세 (x, y, yaw) 를 움직이며 /odom · /imu/data · /scan 을 만든다."""

    def __init__(self, pose, odom_offset, imu_offset, rng):
        self.x, self.y, self.yaw = pose
        self.odom_offset = odom_offset   # 경기장 방향 - odom 내부 방향
        self.imu_offset = imu_offset
        self.ox, self.oy = 2.0, -1.0     # odom 원점은 아무 곳
        self.left, self.right = 3.0, -1.0  # /odom/odom_raw 누적 거리 원점도 아무 값
        self.rng = rng

    def move(self, v, w, dt):
        self.left += (v - w * WHEEL_BASE / 2) * dt
        self.right += (v + w * WHEEL_BASE / 2) * dt
        self.yaw += w * dt
        self.x += v * dt * math.cos(self.yaw)
        self.y += v * dt * math.sin(self.yaw)
        internal = self.yaw - self.odom_offset
        self.ox += v * dt * math.cos(internal)
        self.oy += v * dt * math.sin(internal)

    def odom(self, t):
        return NS(header=NS(frame_id='odom', stamp=stamp(t)), child_frame_id='base_footprint',
                  pose=NS(pose=NS(position=NS(x=round(self.ox, 2), y=round(self.oy, 2)), orientation=quat(0.0))))

    def imu(self, t):
        return NS(header=NS(stamp=stamp(t)), orientation=quat(self.yaw + self.imu_offset))

    def wheels(self, resolution=None):
        """(왼쪽, 오른쪽) 누적 거리. 실제 로봇은 1 cm 단위로 올라간다 (resolution=0.01)."""
        if resolution is None:
            return self.left, self.right
        return round(self.left / resolution) * resolution, round(self.right / resolution) * resolution

    def scan(self, t):
        n = 505
        increment = 2 * math.pi / n
        lx, ly = self.x + 0.0398 * math.cos(self.yaw), self.y + 0.0398 * math.sin(self.yaw)
        ranges = []
        for k in range(n):
            angle = k * increment
            r = float('inf') if 2.2 < angle < 4.1 else ray(lx, ly, self.yaw + math.pi / 2 + angle) + self.rng.gauss(0, 0.01)
            ranges.append(r)
        return NS(header=NS(frame_id='base_lidar_link', stamp=stamp(t)), ranges=ranges, angle_min=0.0,
                  angle_increment=increment, range_min=0.05, range_max=25.0)


def arena_map(t, self_pose=None, boxes=BOXES):
    field = FieldMap()
    field.header = NS(frame_id='arena', stamp=stamp(t))
    field.width, field.height, field.box_size = WIDTH, HEIGHT, BOX
    field.boxes = [NS(x=bx, y=by) for bx, by in boxes]
    field.bases = []
    if self_pose is not None:
        x, y, yaw = self_pose
        field.vehicles = [NS(role=1, SELF=1, OTHER=2, pose=NS(position=NS(x=x, y=y), orientation=quat(yaw)))]
    return field


# 제자리 왼쪽 90° → 1.2 m 직진 → 오른쪽 135° → 0.8 m 직진
SCRIPT = ((1.5, 0.0, math.pi / 3), (3.0, 0.4, 0.0), (1.5, 0.0, -math.pi / 2), (2.0, 0.4, 0.0))


class SlamTrackingTest(unittest.TestCase):
    def setUp(self):
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)
        module, _ = load_slam()
        self.slam = module['SlamNode']()

    def feed_heading(self, robot, t, source, resolution):
        if source == 'imu':
            self.slam._receive_imu(robot.imu(t))
        else:
            self.slam._on_wheels(int(round(t * 1e9)), *robot.wheels(resolution))

    def run_script(self, odom_offset, drop_scans=(), source='wheel', resolution=None, map_boxes=BOXES):
        slam = self.slam
        slam.yaw_source = source
        robot = Robot((1.0, 2.0, 0.0), odom_offset=odom_offset, imu_offset=1.0, rng=random.Random(3))
        t, dt = 10.0, 0.05
        slam._receive_odometry(robot.odom(t - 0.1))
        self.feed_heading(robot, t - 0.1, source, resolution)
        slam._receive_perception_map(arena_map(t - 0.1, (robot.x, robot.y, robot.yaw), map_boxes))
        errors = []
        tick = 0
        for duration, v, w in SCRIPT:
            for _ in range(int(round(duration / dt))):
                robot.move(v, w, dt)
                t += dt
                tick += 1
                self.feed_heading(robot, t, source, resolution)
                if tick % 2 == 0:
                    slam._receive_odometry(robot.odom(t))
                    if not any(start <= t < end for start, end in drop_scans):
                        slam._receive_scan(robot.scan(t))
                    message = slam.pose_publisher.publish.call_args
                    if message is not None:
                        pose = message.args[0].pose
                        sent = message.args[0].header.stamp
                        if abs(sent.sec + sent.nanosec * 1e-9 - t) < 1e-6:
                            yaw = math.atan2(2 * pose.orientation.w * pose.orientation.z,
                                             1 - 2 * pose.orientation.z ** 2)
                            errors.append((math.hypot(pose.position.x - robot.x, pose.position.y - robot.y),
                                           abs(math.atan2(math.sin(yaw - robot.yaw), math.cos(yaw - robot.yaw)))))
        return errors

    def test_tracks_through_turns_when_odom_axes_are_rotated(self):
        """부팅 뒤 돌아 있던 로봇 (odom 축이 경기장과 90° 차이) 도 회전·직진 내내 위치를 놓치지 않는다."""
        errors = self.run_script(odom_offset=math.pi / 2)
        self.assertGreaterEqual(len(errors), 78)   # /odom(10 Hz) 마다 거의 매번 발행 (전체 80번)
        self.assertLess(max(e[0] for e in errors), 0.05)
        self.assertLess(max(e[1] for e in errors), 0.06)
        self.assertIsNotNone(self.slam._delta)
        self.assertAlmostEqual(math.atan2(math.sin(self.slam._delta - math.pi / 2), math.cos(self.slam._delta - math.pi / 2)), 0.0, delta=0.05)

    def test_bridges_a_lidar_gap_with_calibrated_odometry(self):
        """직진 중 라이다 정합이 1 s 끊겨도 맞춘 odom 축으로 이어 간다."""
        errors = self.run_script(odom_offset=-0.7, drop_scans=((12.8, 13.8),))
        self.assertGreaterEqual(len(errors), 75)
        self.assertLess(max(e[0] for e in errors), 0.06)

    def test_tracks_with_walls_only_when_the_drone_map_has_no_boxes(self):
        """드론이 상자를 하나도 못 찾은 지도여도 외벽만으로 위치를 이어 간다 (실제 상자는 「지도에 없는 점」 이 된다)."""
        errors = self.run_script(odom_offset=math.pi / 2, map_boxes=())
        self.assertGreaterEqual(len(errors), 75)
        self.assertLess(max(e[0] for e in errors), 0.08)
        self.assertLess(max(e[1] for e in errors), 0.08)

    def test_wheel_heading_is_the_default_source(self):
        self.assertEqual(self.slam.yaw_source, 'wheel')
        self.assertAlmostEqual(self.slam.wheel_base_m, 0.175)

    def test_wheel_heading_tracks_turns_with_one_cm_encoder_steps(self):
        """실제 로봇처럼 바퀴 거리가 1 cm 단위로만 올라가도 위치를 놓치지 않는다."""
        errors = self.run_script(odom_offset=math.pi / 2, resolution=0.01)
        self.assertGreaterEqual(len(errors), 70)
        self.assertLess(max(e[0] for e in errors), 0.10)
        self.assertLess(max(e[1] for e in errors), 0.20)

    def test_wheel_heading_bridges_a_lidar_gap(self):
        errors = self.run_script(odom_offset=-0.7, drop_scans=((12.8, 13.8),))
        self.assertGreaterEqual(len(errors), 75)
        self.assertLess(max(e[0] for e in errors), 0.06)

    def test_imu_source_still_works(self):
        errors = self.run_script(odom_offset=math.pi / 2, source='imu')
        self.assertGreaterEqual(len(errors), 78)
        self.assertLess(max(e[0] for e in errors), 0.05)

    def test_wheel_turn_matches_wheel_difference_and_ignores_imu(self):
        slam = self.slam
        slam._on_wheels(1_000_000_000, 3.0, -1.0)
        slam._on_wheels(1_500_000_000, 3.0 - 0.0875, -1.0 + 0.0875)   # 오른쪽이 0.175 m 더 감 = +1 rad
        self.assertAlmostEqual(slam._turn(1_000_000_000, 1_500_000_000), 1.0, places=9)
        slam._receive_imu(NS(header=NS(stamp=stamp(2.0)), orientation=quat(2.0)))   # wheel 모드: 무시
        self.assertEqual(len(slam._imu), 2)

    def test_wheel_jump_and_non_finite_values_are_dropped(self):
        slam = self.slam
        slam._on_wheels(1_000_000_000, 0.0, 0.0)
        slam._on_wheels(1_100_000_000, 40.0, 0.0)
        slam._on_wheels(1_200_000_000, float('nan'), 0.0)
        slam._on_wheels(1_300_000_000, 0.0, 0.0)
        self.assertAlmostEqual(slam._turn(1_000_000_000, 1_300_000_000), 0.0, places=9)

    def run_start(self, robot, ready_ticks=20, move_after_lock=0.0):
        """ready(잠김, 지도 없음) → running → 0.5 s 뒤 첫 지도(출발 칸만 있고 우리 차 표시 없음) → 3번 더."""
        slam = self.slam

        def tick(t, phase):
            slam._receive_server_state(NS(run_id='r1', header=NS(stamp=stamp(t)), team_assigned=True,
                                          camp='BLUE', slot='B-2', phase=phase))
            self.feed_heading(robot, t, 'wheel', 0.01)
            slam._receive_odometry(robot.odom(t))
            slam._receive_scan(robot.scan(t))

        t = 10.0
        for _ in range(ready_ticks):
            tick(t, 'ready')
            t += 0.1
        robot.move(move_after_lock, 0.0, 1.0)
        for _ in range(5):
            tick(t, 'running')
            t += 0.1
        slam.pose_publisher.publish.assert_not_called()
        field = arena_map(t - 0.3)   # 드론 사진 시각은 받기 전이다
        field.bases = [NS(name='B-1', min_x=0.0, min_y=3.57, max_x=0.4, max_y=4.77),
                       NS(name='B-2', min_x=0.0, min_y=1.9, max_x=0.4, max_y=3.1)]
        # 항공이 우리 차를 UNKNOWN 으로만 잡았다 (10 cm 어긋나게), 상대 차 하나
        field.vehicles = [NS(role=0, SELF=1, OTHER=2, pose=NS(position=NS(x=robot.x + 0.1, y=robot.y), orientation=quat(0.0))),
                          NS(role=2, SELF=1, OTHER=2, pose=NS(position=NS(x=5.9, y=2.5), orientation=quat(0.0)))]
        slam._receive_perception_map(field)
        for _ in range(3):
            tick(t, 'running')
            t += 0.1
        return t, tick

    def last_error(self, robot):
        message = self.slam.pose_publisher.publish.call_args
        self.assertIsNotNone(message)
        pose = message.args[0].pose
        yaw = math.atan2(2 * pose.orientation.w * pose.orientation.z, 1 - 2 * pose.orientation.z ** 2)
        return (math.hypot(pose.position.x - robot.x, pose.position.y - robot.y),
                abs(math.atan2(math.sin(yaw - robot.yaw), math.cos(yaw - robot.yaw))))

    def test_starts_from_the_slot_when_the_map_arrives_after_the_start(self):
        """지도(드론 사진)는 경기 진행 중에만 온다. ready 동안 잠겨 있던 로봇은 running 에서 처음 온 지도로
        출발 칸 전체를 라이다로 찾는다. 지도에 우리 차 표시가 없어도 된다. 찾은 뒤 출발해도 따라간다."""
        robot = Robot((0.22, 2.55, 0.3), odom_offset=1.0, imu_offset=0.5, rng=random.Random(5))
        t, tick = self.run_start(robot)
        distance, heading = self.last_error(robot)
        self.assertLess(distance, 0.03)
        self.assertLess(heading, 0.03)
        vehicles = self.slam.map_publisher.publish.call_args.args[0].vehicles
        self.assertEqual(sorted(v.role for v in vehicles), [1, 2])   # UNKNOWN 이던 우리 차가 SELF 로, 두 번 나오지 않음
        for _ in range(20):   # 출발: 0.3 m/s 로 2 s 직진
            robot.move(0.3, 0.0, 0.05)
            self.feed_heading(robot, t - 0.05, 'wheel', 0.01)
            robot.move(0.3, 0.0, 0.05)
            tick(t, 'running')
            t += 0.1
        distance, heading = self.last_error(robot)
        self.assertLess(distance, 0.05)
        self.assertLess(heading, 0.05)

    def test_slot_is_not_used_after_the_robot_moved(self):
        """잠긴 뒤 10 cm 움직였으면 출발 칸이라고 보지 않는다 (틀린 자리로 초기화하지 않음)."""
        robot = Robot((0.22, 2.55, 0.0), odom_offset=0.0, imu_offset=0.0, rng=random.Random(6))
        self.run_start(robot, move_after_lock=0.10)
        self.slam.pose_publisher.publish.assert_not_called()
        self.assertIsNone(self.slam._slot_pose)

    def test_slot_is_not_used_when_started_during_the_match(self):
        """경기 중에 켜져 잠긴 것을 못 봤으면 출발 칸을 쓰지 않는다 (항공 SELF 를 기다린다)."""
        robot = Robot((0.22, 2.55, 0.0), odom_offset=0.0, imu_offset=0.0, rng=random.Random(7))
        self.run_start(robot, ready_ticks=0)
        self.slam.pose_publisher.publish.assert_not_called()

    def test_uncalibrated_odometry_is_not_trusted_far(self):
        """odom 축을 아직 모르면 5 cm 넘게 이어 붙이지 않는다 (틀린 방향으로 밀고 가지 않음)."""
        slam = self.slam
        robot = Robot((1.0, 2.0, 0.0), odom_offset=math.pi / 2, imu_offset=0.0, rng=random.Random(1))
        slam._receive_odometry(robot.odom(9.9))
        slam._receive_perception_map(arena_map(9.9, (robot.x, robot.y, robot.yaw)))
        robot.move(0.4, 0.0, 0.5)
        slam._receive_odometry(robot.odom(10.4))
        self.assertIsNone(slam._delta)
        slam.pose_publisher.publish.assert_not_called()

    def test_imu_turn_is_unwrapped(self):
        slam = self.slam
        slam.yaw_source = 'imu'
        for k in range(30):
            slam._receive_imu(NS(header=NS(stamp=stamp(1.0 + 0.05 * k)), orientation=quat(2.5 + 0.2 * k)))
        self.assertAlmostEqual(slam._turn(1_000_000_000, 2_450_000_000), 0.2 * 29, places=6)


if __name__ == '__main__':
    unittest.main()
