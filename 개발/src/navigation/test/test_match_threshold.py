"""정합 통과 기준(맞은 점 수)은 무게 없이 센다 — 실차 스캔으로 보는 회귀 시험. ROS 없이 돈다.

2026-11-19: map_fusion 은 라이다로 확인 못 한 드론 상자에 맞은 점을 0.7 로 센다. 그 점수를 통과 기준 max(12, 점의 25 %) 에
그대로 견주면, 모든 드론 상자가 확인 전인 경기 시작 때 같은 스캔 · 같은 자세가 기준 아래로 떨어진다.
실차 기록 1118_201710 (로봇 ~/team/nav_web/runs/) 을 다시 돌렸을 때 출발 칸 탐색이 105 s 동안 실패했고 정합 성공이 4011 → 1100 번으로 줄었다.
real_scan_1118_201710.json 은 그 기록에서 지도를 받은 직후의 /scan 3개와 그때의 지도 · 출발 칸이다.
"""
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from test_node_split import load_slam

DATA = Path(__file__).resolve().parent / 'real_scan_1118_201710.json'
UNCONFIRMED = 0.7   # map_fusion.FusionParams.unconfirmed_weight


class MatchThresholdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slam, _ = load_slam()
        data = json.loads(DATA.read_text(encoding='utf-8'))
        f = data['field']
        cls.field = NS(width=f['width'], height=f['height'], box_size=f['box_size'], boxes=[NS(x=x, y=y) for x, y in f['boxes']])
        cls.boxes = [tuple(b) for b in f['boxes']]
        cls.weights = np.full(len(cls.boxes), UNCONFIRMED)
        cls.region = tuple(data['region'])
        cls.scans = [NS(header=NS(frame_id=s['frame_id']), angle_min=s['angle_min'], angle_increment=s['angle_increment'],
                        range_min=s['range_min'], range_max=s['range_max'],
                        ranges=[math.inf if r is None else r for r in s['ranges']]) for s in data['scans']]

    def near(self, a, b, tolerance=0.05):
        self.assertLess(math.hypot(a[0] - b[0], a[1] - b[1]), tolerance, (a, b))
        self.assertLess(abs(math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2]))), math.radians(3.0), (a, b))

    def test_slot_search_works_while_every_box_is_unconfirmed(self):
        """경기 시작: 드론 상자가 모두 확인 전(무게 0.7)이어도 출발 칸 탐색은 무게 없을 때와 같은 자리를 찾는다."""
        for scan in (self.scans[0], self.scans[2]):
            plain = self.slam['slot_search'](self.field, scan, self.region)
            weighted = self.slam['slot_search'](self.field, scan, self.region, boxes=self.boxes, weights=self.weights)
            self.assertIsNotNone(plain)
            self.assertIsNotNone(weighted)
            self.near(weighted, plain)

    def test_scan_match_works_while_every_box_is_unconfirmed(self):
        seed = self.slam['slot_search'](self.field, self.scans[0], self.region)
        for scan in (self.scans[0], self.scans[2]):
            plain = self.slam['scan_match'](self.field, scan, seed, False)
            weighted = self.slam['scan_match'](self.field, scan, seed, False, boxes=self.boxes, weights=self.weights)
            self.assertIsNotNone(plain)
            self.assertIsNotNone(weighted)
            self.near(weighted, plain, 0.10)   # 무게는 후보 순위를 바꿀 수 있다 (이 지도는 실제 배치와 달라 y 로 5 cm 쯤 갈린다)

    def test_this_scan_is_one_where_the_weighted_score_falls_below_the_threshold(self):
        """이 스캔이 정말 그 경우인지: 무게 없이는 기준 이상, 무게를 주면 기준 아래 (그래서 무게 준 점수를 기준에 견주면 안 된다)."""
        scan = self.scans[0]
        pose = self.slam['slot_search'](self.field, scan, self.region)
        points = np.array(self.slam['scan_points'](scan), dtype=float)
        boxes, half, width, height = self.slam['structure'](self.field)
        threshold = max(12, len(points) * 0.25)
        plain = self.slam['score_pose'](points, pose, boxes, half, width, height)[0]
        weighted = self.slam['score_pose'](points, pose, boxes, half, width, height, self.weights)[0]
        self.assertGreaterEqual(plain, threshold)
        self.assertLess(weighted, threshold)
        self.assertTrue(self.slam['enough_hits'](points, pose, boxes, half, width, height))


if __name__ == '__main__':
    unittest.main()
