"""잃은 위치 다시 찾기 (slam_node._relocate, 2026-11-20). rclpy 없이 test_slam_tracking 의 가짜 경기장으로 돈다.

정합이 2 s 넘게 끊기면 씨앗이 없어지고 출발 칸 탐색도 쓸 수 없어, 전에는 위치를 영영 내지 못했다
(2026-11-19 21:53 · 22:39 판의 끝: 「정합 실패」 2 s 뒤 「처음 위치 찾기 실패」 만 이어짐).
"""
import math
import random
import unittest
from types import SimpleNamespace as NS

from test_node_split import FakeNode, load_slam
from test_slam_tracking import Robot, arena_map, stamp


def ugvs_message(t, points, frame='arena'):
    return NS(header=NS(frame_id=frame, stamp=stamp(t)),
              poses=[NS(position=NS(x=x, y=y, z=0.9), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)) for x, y in points])


class RelocateLostTest(unittest.TestCase):
    def setUp(self):
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)
        module, _ = load_slam()
        self.module = module
        self.slam = module['SlamNode']()

    def tick(self, robot, t, scan=True):
        self.slam._on_wheels(int(round(t * 1e9)), *robot.wheels(0.01))
        self.slam._receive_odometry(robot.odom(t))
        if scan:
            self.slam._receive_scan(robot.scan(t))

    def error(self, robot, t):
        """방금(t) 낸 위치의 (거리 오차, 방향 오차). 그때 낸 것이 없으면 None."""
        message = self.slam.pose_publisher.publish.call_args
        if message is None:
            return None
        sent = message.args[0].header.stamp
        if abs(sent.sec + sent.nanosec * 1e-9 - t) > 1e-6:
            return None
        pose = message.args[0].pose
        yaw = math.atan2(2 * pose.orientation.w * pose.orientation.z, 1 - 2 * pose.orientation.z ** 2)
        return (math.hypot(pose.position.x - robot.x, pose.position.y - robot.y),
                abs(math.atan2(math.sin(yaw - robot.yaw), math.cos(yaw - robot.yaw))))

    def track_then_lose(self, shove=(0.25, -0.15, 0.2)):
        """SELF 가 든 지도로 시작해 3 s 달린 뒤(odom 축을 맞춘다), 3 s 동안 스캔이 끊긴 사이 밀려 옮겨졌다 (바퀴는 안 굴렀다)."""
        slam = self.slam
        robot = Robot((1.0, 2.0, 0.0), odom_offset=math.pi / 2, imu_offset=0.0, rng=random.Random(11))
        t = 10.0
        slam._receive_odometry(robot.odom(t - 0.1))
        slam._on_wheels(int(round((t - 0.1) * 1e9)), *robot.wheels(0.01))
        slam._receive_perception_map(arena_map(t - 0.1, (robot.x, robot.y, robot.yaw)))
        for k in range(30):   # 0.3 m/s 로 가며 조금 돈다
            robot.move(0.3, 0.3 if k < 10 else 0.0, 0.1)
            t += 0.1
            self.tick(robot, t)
        self.assertLess(self.error(robot, t)[0], 0.05)
        self.assertIsNotNone(slam._delta)
        for _ in range(30):   # 스캔이 3 s 끊겼다 (PROPAGATE_NS 2 s 를 넘긴다)
            t += 0.1
            self.tick(robot, t, scan=False)
        robot.x += shove[0]
        robot.y += shove[1]
        robot.yaw += shove[2]
        return robot, t

    def test_finds_the_pose_again_around_the_last_one(self):
        robot, t = self.track_then_lose()
        errors = []
        for _ in range(30):
            t += 0.1
            self.tick(robot, t)
            errors.append(self.error(robot, t))
        self.assertEqual(self.slam._pose_checks['lost_found'], 1, self.slam._relocate_note)
        self.assertIsNotNone(errors[-1], self.slam._relocate_note)
        self.assertLess(errors[-1][0], 0.05)
        self.assertLess(errors[-1][1], 0.05)
        self.assertIn('마지막 자리 둘레', self.slam._relocate_note)

    def spun_and_shoved(self):
        """06:17 판: 서서 도는 중에 다른 로봇에 밀렸다 — 0.3 m 옮겨졌고, 바퀴가 센 방향은 실제와 2.6 rad 다르다."""
        robot, t = self.track_then_lose(shove=(0.30, 0.05, 2.6))
        errors = []
        for _ in range(60):
            t += 0.1
            self.tick(robot, t)
            errors.append(self.error(robot, t))
            if self.slam._relocate_next is not None:   # 훑는 데 걸린 시간(이 PC)만큼 다음 시도가 늦춰진다 — 2 s 간격으로 되돌린다
                self.slam._relocate_next = min(self.slam._relocate_next, int(round(t * 1e9)) + self.module['RELOCATE_RETRY_NS'])
        return errors

    def test_finds_the_pose_when_the_wheel_heading_is_far_off(self):
        """바퀴가 센 방향이 크게 틀려도 마지막으로 정합한 자리 둘레를 모든 방향으로 훑어 찾는다 — 드론 없이."""
        errors = self.spun_and_shoved()
        self.assertEqual(self.slam._pose_checks['lost_found'], 1, self.slam._relocate_note)
        self.assertIsNotNone(errors[-1], self.slam._relocate_note)
        self.assertLess(errors[-1][0], 0.05)
        self.assertLess(errors[-1][1], 0.05)

    def test_a_scan_that_does_not_fit_the_drone_map_does_not_edit_the_map(self):
        """틀린 자리에서는 스캔이 드론 지도에 맞지 않는다 → 그 스캔으로는 지도를 고치지 않는다 (FUSION_MIN_DRONE_SHARE)."""
        robot, t = self.track_then_lose(shove=(0.0, 0.0, 0.0))
        field, scan = self.slam.latest_perception_map, robot.scan(t)
        self.assertTrue(self.slam._fits_drone_map(field, scan, (robot.x, robot.y, robot.yaw)))
        self.assertFalse(self.slam._fits_drone_map(field, scan, (robot.x + 0.9, robot.y + 0.6, robot.yaw + 2.5)))

    def test_without_it_the_pose_never_comes_back(self):
        """끄면(relocate_lost false) 전과 같다: 씨앗이 없어진 뒤로는 위치를 내지 못한다."""
        self.slam.relocate_lost = False
        robot, t = self.track_then_lose()
        for _ in range(30):
            t += 0.1
            self.tick(robot, t)
            self.assertIsNone(self.error(robot, t))
        self.assertEqual(self.slam._scan_note['result'], '처음 위치 찾기 실패')

    def start_mid_match(self, robot, seen, seconds=14.0):
        """경기 중에 켜졌다: 잠긴 단계를 못 봤고, 지도에는 우리 차가 없다. 드론이 본 UGV 자리(seen: 시각 → 목록)만 1 s 마다 온다."""
        slam = self.slam
        t = 10.0
        slam._receive_odometry(robot.odom(t - 0.1))
        slam._receive_perception_map(arena_map(t - 0.1))
        last = None
        for k in range(int(seconds * 10)):
            t += 0.1
            if k % 10 == 0:
                slam._receive_ugvs(ugvs_message(t - 0.3, seen(t)))
            slam._receive_server_state(NS(run_id='r1', header=NS(stamp=stamp(t)), team_assigned=True, camp='BLUE', slot='B-2', phase='running'))
            self.tick(robot, t)
            last = self.error(robot, t) or last
            # 훑는 데 걸린 시간(이 PC)만큼 다음 시도가 늦춰진다 — 시험에서는 2 s 간격으로 되돌린다
            if slam._relocate_next is not None:
                slam._relocate_next = min(slam._relocate_next, int(round(t * 1e9)) + self.module['RELOCATE_RETRY_NS'])
        return last

    def test_started_during_the_match_it_uses_the_drone_positions(self):
        robot = Robot((2.6, 1.9, 0.7), odom_offset=1.0, imu_offset=0.0, rng=random.Random(12))
        last = self.start_mid_match(robot, lambda t: [(4.4, 3.3), (robot.x + 0.06, robot.y - 0.04), (1.0, 3.2)])
        self.assertIsNotNone(last, self.slam._relocate_note)
        self.assertLess(last[0], 0.05)
        self.assertLess(last[1], 0.05)
        self.assertEqual(self.slam._pose_checks['lost_found'], 1)
        self.assertGreaterEqual(self.slam._pose_checks['lost_tries'], 2)   # 드론 자리만으로는 이어진 두 번이 같아야 받는다

    def test_no_pose_when_the_drone_does_not_see_us(self):
        """드론이 우리를 못 봤다(화면 밖): 다른 UGV 자리들에서는 스캔이 맞지 않는다 → 틀린 자리로 잡지 않고 잃은 채로 둔다."""
        robot = Robot((2.6, 1.9, 0.7), odom_offset=1.0, imu_offset=0.0, rng=random.Random(13))
        last = self.start_mid_match(robot, lambda t: [(4.4, 3.3), (1.0, 3.2)], seconds=7.0)
        self.assertIsNone(last, self.slam._relocate_note)
        self.assertEqual(self.slam._pose_checks['lost_found'], 0)
        self.assertGreaterEqual(self.slam._pose_checks['lost_tries'], 2)

    def test_no_drone_positions_and_no_last_pose_changes_nothing(self):
        """마지막 자리도 드론 자리도 없으면 전과 같다 (훑을 곳이 없다)."""
        robot = Robot((2.6, 1.9, 0.7), odom_offset=1.0, imu_offset=0.0, rng=random.Random(14))
        last = self.start_mid_match(robot, lambda t: [], seconds=3.0)
        self.assertIsNone(last)
        self.assertEqual(self.slam._pose_checks['lost_found'], 0)


if __name__ == '__main__':
    unittest.main()
