"""SLAM 노드가 드론이 본 UGV 를 주행 노드와 같은 궤적(UgvTracks)으로 다룬다 (2026-11-20). rclpy 없이 돈다.

지도 고치기(map_fusion)에 넘기기 전에 다른 UGV 에 맞은 라이다 점을 빼는 데 쓴다 (_without_ugv_points).
"""
import unittest
from types import SimpleNamespace as NS

import numpy as np

from test_node_split import FakeNode, load_slam
from test_slam_tracking import stamp


def ugvs_message(t, points, frame='arena'):
    return NS(header=NS(frame_id=frame, stamp=stamp(t)),
              poses=[NS(position=NS(x=x, y=y, z=0.9), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)) for x, y in points])


class SlamUgvMaskTest(unittest.TestCase):
    def setUp(self):
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)
        module, _ = load_slam()
        self.slam = module['SlamNode']()

    def stand(self, t, x=2.0, y=2.5):
        """그 시각에 (x, y) 에서 정합이 받아들여졌다 (드론 사진에서 우리 차를 골라낼 자리)."""
        self.slam._pose_trail.append((int(round(t * 1e9)), x, y))

    def masked(self, t, world_points, pose=(2.0, 2.5, 0.0)):
        """경기장 좌표 점들을 차체 좌표로 넘겼을 때 빠지는 점 수."""
        body = np.array([(wx - pose[0], wy - pose[1]) for wx, wy in world_points], dtype=float)
        return len(body) - len(self.slam._without_ugv_points(body, pose, int(round(t * 1e9))))

    def test_points_on_another_ugv_are_dropped_once_it_was_seen_twice(self):
        slam = self.slam
        on_ugv = [(3.0, 2.5), (3.1, 2.6), (2.9, 2.45)]
        elsewhere = [(4.5, 1.0), (2.2, 2.5)]   # 다른 곳 · 우리 차 바로 앞
        self.stand(10.0)
        slam._receive_ugvs(ugvs_message(10.0, [(2.03, 2.52), (3.0, 2.5)]))   # 우리 차 + 다른 UGV 한 대
        self.assertEqual(self.masked(10.1, on_ugv + elsewhere), 0)           # 한 번 본 것은 아직 믿지 않는다 (주행 노드와 같다)
        self.stand(11.0)
        slam._receive_ugvs(ugvs_message(11.0, [(2.03, 2.52), (3.0, 2.5)]))
        self.assertEqual(self.masked(11.1, on_ugv + elsewhere), 3)           # 그 UGV 곁의 점만 뺀다 — 우리 차 곁의 점은 그대로
        self.assertEqual(slam._ugv_masked, 3)

    def test_frame_without_our_pose_is_not_used(self):
        slam = self.slam
        for t in (10.0, 11.0):
            slam._receive_ugvs(ugvs_message(t, [(2.03, 2.52), (3.0, 2.5)]))  # 그 무렵의 자세가 없다 → 어느 것이 우리 차인지 모른다
        self.assertEqual(self.masked(11.1, [(3.0, 2.5), (2.05, 2.5)]), 0)
        self.assertIsNotNone(slam._ugv_seen)                                  # 그래도 잃은 위치를 다시 찾을 자리의 후보로는 들고 있다

    def test_wrong_frame_is_ignored(self):
        self.stand(10.0)
        self.slam._receive_ugvs(ugvs_message(10.0, [(3.0, 2.5)], frame='map'))
        self.assertIsNone(self.slam._ugv_seen)


if __name__ == '__main__':
    unittest.main()
