"""뜀 거르기: 외벽이 새 자리를 뚜렷이 편들면 버리지 않는다 (2026-11-19 밤). ROS 없이 돈다.

22:39 판 128 ~ 132 s: 왼쪽 벽 곁에 선 동안 정합이 x = 0.99 와 0.86 을 오갔고, 0.86 쪽이 「/odom 이 간 거리보다 0.01 m 더 뜀」 으로 13번 버려졌다
(서 있을 때의 한도 0.12 m, 두 자리 차이 0.13 m). 그 스캔의 벽 점수는 0.99 에서 49, 0.86 에서 92 — 지키던 자리가 틀린 쪽이었다. 그 뒤 46 s 동안
그 둘레에서 BLOCKED 4번. real_scans_1119_jump.json = 그 판의 스캔 2개(kept_wrong) + 18:06 판 99.2 s(틀린 쪽으로 뛰기 시작한 스캔, jump_wrong).
"""
import json
import math
import random
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

import test_slam_tracking as T
from test_node_split import FakeNode, load_slam

DATA = Path(__file__).resolve().parent / 'real_scans_1119_jump.json'


def as_scan(r):
    ranges = [x if x else float('nan') for x in r['ranges']]
    return NS(header=NS(frame_id='base_lidar_link'), ranges=ranges, range_min=0.02, range_max=12.0,
              angle_min=r['angle_min'], angle_increment=r['angle_increment'])


class RealScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slam, _ = load_slam()
        cls.scans = json.loads(DATA.read_text(encoding='utf-8'))['scans']

    def rematch(self, r):
        s, f = self.slam, r['field']
        field = NS(width=f['width'], height=f['height'], box_size=f['box_size'], boxes=[NS(x=x, y=y) for x, y in f['boxes']])
        scan, kept = as_scan(r), tuple(r['pose'])
        result = s['scan_match'](field, scan, kept, False)
        dense = np.array(s['scan_points'](scan, 2), dtype=float)
        return kept, result, dense, field

    def test_2239_the_kept_place_was_the_wrong_one(self):
        """지키던 x = 0.99 에서 다시 정합하면 0.13 m 떨어진 0.86 이 나오고(서 있을 때의 한도 0.12 m 를 넘는다), 외벽은 그쪽을 뚜렷이 편든다."""
        s = self.slam
        for r in self.scans:
            if r['kind'] != 'kept_wrong':
                continue
            kept, result, dense, field = self.rematch(r)
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result[0] - kept[0], -0.13, delta=0.03)
            self.assertGreater(math.hypot(result[0] - kept[0], result[1] - kept[1]), s['JUMP_SLACK_M'])
            self.assertTrue(s['walls_prefer'](dense, kept, result, field.width, field.height), r['t'])
            self.assertFalse(s['walls_prefer'](dense, result, kept, field.width, field.height), r['t'])   # 거꾸로는 아니다

    def test_1806_the_jump_to_the_wrong_place_is_still_rejected(self):
        """18:06 판 99.2 s (0.5 m 뜀이 시작된 스캔): 정합이 가리킨 새 자리를 외벽은 편들지 않는다 — 전처럼 버린다."""
        s = self.slam
        for r in self.scans:
            if r['kind'] != 'jump_wrong':
                continue
            kept, result, dense, field = self.rematch(r)
            self.assertIsNotNone(result)
            self.assertGreater(math.hypot(result[0] - kept[0], result[1] - kept[1]), s['JUMP_SLACK_M'])
            self.assertFalse(s['walls_prefer'](dense, kept, result, field.width, field.height))

    def test_too_few_points_never_prefers(self):
        s = self.slam
        self.assertFalse(s['walls_prefer'](np.zeros((10, 2)), (1.0, 1.0, 0.0), (1.2, 1.0, 0.0), 6.12, 4.8))


class JumpGateNodeTest(unittest.TestCase):
    """SLAM 노드의 흐름: 지키던 자리가 틀렸을 때 맞는 쪽으로 뛴 정합을 받는다."""

    def setUp(self):
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)
        self.module, _ = load_slam()
        self.slam = self.module['SlamNode']()

    def stand(self, seconds=1.0, start=(1.0, 2.0, 0.0)):
        slam = self.slam
        robot = T.Robot(start, odom_offset=0.6, imu_offset=1.0, rng=random.Random(3))
        t, dt = 10.0, 0.05
        slam._receive_odometry(robot.odom(t - 0.1))
        slam._on_wheels(int(round((t - 0.1) * 1e9)), *robot.wheels())
        slam._receive_perception_map(T.arena_map(t - 0.1, (robot.x, robot.y, robot.yaw)))
        for tick in range(1, int(round(seconds / dt)) + 1):
            robot.move(0.0, 0.0, dt)
            t += dt
            slam._on_wheels(int(round(t * 1e9)), *robot.wheels())
            if tick % 2 == 0:
                slam._receive_odometry(robot.odom(t))
                slam._receive_scan(robot.scan(t))
        return robot, t

    def test_walls_back_the_true_place_but_not_the_wrong_one(self):
        robot, t = self.stand()
        slam, field = self.slam, self.slam.latest_perception_map
        truth, wrong = (robot.x, robot.y, robot.yaw), (robot.x + 0.3, robot.y, robot.yaw)
        stamp = int(round((t + 0.1) * 1e9))
        scan = robot.scan(t + 0.1)
        self.assertIsNotNone(slam._jumped(stamp, wrong))                       # 서 있는데 0.3 m: 뜀이다
        self.assertFalse(slam._walls_back(field, scan, truth, wrong, stamp))   # 틀린 쪽으로의 뜀: 외벽이 편들지 않는다 → 버린다
        self.assertTrue(slam._walls_back(field, scan, wrong, truth, stamp))    # 틀린 자리를 지키고 있었다면: 맞는 쪽으로의 뜀은 받는다

    def test_a_wrongly_kept_place_is_left_when_the_match_jumps_to_the_walls(self):
        """지키던 자리를 0.16 m 틀리게 만들어 둔다 (22:39 판처럼). 다음 스캔의 정합은 진짜 자리로 뛴다 — 버리지 않고 받아들인다."""
        robot, t = self.stand()
        slam = self.slam
        stamp, anchor, odom = slam._scan_anchor
        wrong = (anchor[0] + 0.16, anchor[1], anchor[2])
        slam._scan_anchor = (stamp, wrong, odom)
        slam._accepted = type(slam._accepted)([(at, x + 0.16, y, then) for at, x, y, then in slam._accepted], maxlen=slam._accepted.maxlen)
        for k in range(1, 4):
            now = t + 0.1 * k
            slam._on_wheels(int(round(now * 1e9)), *robot.wheels())
            slam._receive_odometry(robot.odom(now))
            slam._receive_scan(robot.scan(now))
        checks = slam._pose_checks
        self.assertGreaterEqual(checks['jump_wall_ok'], 1, checks)
        self.assertEqual(checks['jump_rejected'], 0, checks)
        self.assertLess(math.hypot(slam._scan_anchor[1][0] - robot.x, slam._scan_anchor[1][1] - robot.y), 0.05)


if __name__ == '__main__':
    unittest.main()
