"""드론이 본 우리 자리로 위치를 의심하기 (slam_node._drone_self · _drone_recheck, 2026-11-20). rclpy 없이 test_slam_tracking 의 가짜 경기장으로 돈다.

드론은 「의심」 만 한다: 우리 차가 안 보이고 모르는 것 하나가 우리를 따라오면, 그 자리 둘레를 라이다로 훑어 뚜렷이 잘 맞을 때만 옮긴다.
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


class DronePoseCheckTest(unittest.TestCase):
    def setUp(self):
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)
        module, _ = load_slam()
        self.module = module
        self.slam = module['SlamNode']()
        self.robot = Robot((1.0, 2.0, 0.0), odom_offset=math.pi / 2, imu_offset=0.0, rng=random.Random(21))
        self.t = 10.0
        slam, robot = self.slam, self.robot
        slam._receive_odometry(robot.odom(self.t - 0.1))
        slam._on_wheels(int(round((self.t - 0.1) * 1e9)), *robot.wheels(0.01))
        slam._receive_perception_map(arena_map(self.t - 0.1, (robot.x, robot.y, robot.yaw)))

    def run_for(self, seconds, v=0.0, w=0.0, seen=None, every=1.0):
        """seconds 동안 0.1 s 마다 한 주기. seen(robot) = 그때 드론이 본 UGV 자리들 (every 초마다 한 장), None 이면 사진 없음."""
        slam, robot = self.slam, self.robot
        for k in range(int(round(seconds * 10))):
            robot.move(v, w, 0.1)
            self.t += 0.1
            slam._on_wheels(int(round(self.t * 1e9)), *robot.wheels(0.01))
            slam._receive_odometry(robot.odom(self.t))
            slam._receive_scan(robot.scan(self.t))
            if seen is not None and k % int(round(every * 10)) == 0:
                slam._receive_ugvs(ugvs_message(self.t - 0.3, seen(robot)))
            if slam._drone_next is not None:   # 훑는 데 걸린 시간(이 PC)만큼 다음 시도가 늦춰진다 — 시험에서는 2 s 간격으로 되돌린다
                slam._drone_next = min(slam._drone_next, int(round(self.t * 1e9)) + self.module['DRONE_RECHECK_NS'])

    def error(self):
        pose = self.slam.pose_publisher.publish.call_args.args[0].pose
        return math.hypot(pose.position.x - self.robot.x, pose.position.y - self.robot.y)

    def test_records_the_offset_when_the_drone_sees_us(self):
        self.run_for(4.0, v=0.3, seen=lambda r: [(r.x + 0.05, r.y - 0.03), (4.4, 3.3)])
        checks = self.slam._pose_checks
        self.assertGreaterEqual(checks['drone_seen'], 3)
        self.assertEqual((checks['drone_missed'], checks['drone_doubt'], checks['drone_tries']), (0, 0, 0))
        off = self.slam._debug_state()['ugvs']['self_off']
        self.assertLess(math.hypot(off[0] - 0.05, off[1] + 0.03), 0.20)   # 사진은 0.3 s 전 것이라 간 만큼은 다르다
        self.assertIsNone(self.slam._drone_stray)

    def test_not_seeing_us_alone_raises_no_doubt(self):
        """우리 차가 안 보일 뿐이면 (화면 밖 · 오른쪽 벽 쪽) 아무 일도 없다. 전부터 알던 다른 UGV 가 곁에 있어도 같다."""
        self.run_for(3.0, v=0.3, seen=lambda r: [(r.x, r.y), (r.x + 0.6, r.y + 0.3)])   # 우리와, 곁을 따라오는 다른 UGV
        self.run_for(8.0, v=0.1, seen=lambda r: [(r.x + 0.6, r.y + 0.3)])               # 우리는 안 보이고 그 UGV 만
        checks = self.slam._pose_checks
        self.assertGreaterEqual(checks['drone_missed'], 5)
        self.assertEqual((checks['drone_doubt'], checks['drone_tries'], checks['drone_moved']), (0, 0, 0))
        self.assertLess(self.error(), 0.05)

    def test_a_stranger_following_us_is_checked_by_lidar_and_changes_nothing(self):
        """모르는 것 하나가 우리를 따라온다 → 의심하고 그 자리를 라이다로 훑지만, 지금 자리가 맞으므로 옮기지 않는다."""
        self.run_for(3.0, v=0.3, seen=lambda r: [(r.x, r.y)])
        self.run_for(10.0, v=0.05, seen=lambda r: [(r.x + 0.5, r.y - 0.3)])
        checks = self.slam._pose_checks
        self.assertEqual(checks['drone_doubt'], 1)
        self.assertGreaterEqual(checks['drone_tries'], 2)
        self.assertEqual(checks['drone_moved'], 0)
        self.assertLess(self.error(), 0.05)
        stray = self.slam._debug_state()['ugvs']['stray']
        self.assertLess(math.hypot(stray[0] - 0.5, stray[1] + 0.3), 0.15)
        self.assertGreaterEqual(stray[2], self.module['DRONE_DOUBT_FRAMES'])

    def test_can_be_turned_off_but_still_records(self):
        self.slam.drone_pose_check = False
        self.run_for(3.0, v=0.3, seen=lambda r: [(r.x, r.y)])
        self.run_for(10.0, v=0.05, seen=lambda r: [(r.x + 0.5, r.y - 0.3)])
        checks = self.slam._pose_checks
        self.assertEqual(checks['drone_doubt'], 1)
        self.assertEqual((checks['drone_tries'], checks['drone_moved']), (0, 0))

    def test_moves_when_the_scan_fits_clearly_better_where_the_drone_points(self):
        """SLAM 이 0.5 m 틀린 자리를 내고 있고 드론의 「모르는 것」 이 진짜 자리를 가리킨다 → 두 번 이어 그 자리가 뚜렷이 나으면 옮긴다."""
        slam, robot = self.slam, self.robot
        self.run_for(4.0, v=0.3, w=0.2)
        self.assertLess(self.error(), 0.05)
        field = slam.latest_perception_map
        wrong = (robot.x + 0.5, robot.y + 0.1, robot.yaw)
        moved = None
        for _ in range(2):
            self.t += 2.0
            ns = int(round(self.t * 1e9))
            slam._on_wheels(ns, *robot.wheels(0.01))
            slam._receive_odometry(robot.odom(self.t))
            slam._drone_stray = (ns, -0.5, -0.1, self.module['DRONE_DOUBT_FRAMES'], ns)   # 드론: 우리 자리에서 (−0.5, −0.1) 에 모르는 것
            slam._drone_next = None
            moved = slam._drone_recheck(field, robot.scan(self.t), wrong, ns)
        self.assertIsNotNone(moved)
        self.assertLess(math.hypot(moved[0] - robot.x, moved[1] - robot.y), 0.05)
        self.assertEqual(slam._pose_checks['drone_moved'], 1)
        self.assertEqual(slam._pose_checks['drone_tries'], 2)   # 한 번으로는 옮기지 않는다


if __name__ == '__main__':
    unittest.main()
