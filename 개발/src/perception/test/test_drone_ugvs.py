"""드론 화면의 UGV 자리(drone_ugvs) 시험 — 모델 파일 없이 가짜 예측기로.

  python3 -m unittest discover -s test -p "test_drone_ugvs.py"     (패키지 폴더에서. ROS 불필요)
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from perception import drone_ugvs  # noqa: E402
from perception.drone_boundary import Boundary  # noqa: E402

PPM = 200.0


def boundary(image_h=1000):
    """바닥 1 m = 200 px 로 곧게 내려다보는 가짜 카메라 (렌즈 왜곡 없음). image_h 를 줄이면 화면 위아래가 잘린다."""
    cut = (1000 - image_h) / 2.0
    hom = np.array([[1.0 / PPM, 0.0, 0.0], [0.0, -1.0 / PPM, 5.0 - cut / PPM], [0.0, 0.0, 1.0]])
    return Boundary(True, hom=hom, image_size=(1224, image_h), strip=0.88, cm_half=0.60)


def quad_for(view, true_xy, half=0.12):
    """참 자리(높이 ugv_height)에 있는 UGV 를 모델이 map 그림에서 찾았을 때의 꼭짓점 픽셀."""
    n = np.array(view.mv.camera[:2])
    t = np.asarray(true_xy, float) - view.shift
    plane = view.mv.plane_of_floor([n + (t - n) / view.lift])[0]
    corners = [(plane[0] - half, plane[1] + half), (plane[0] + half, plane[1] + half), (plane[0] + half, plane[1] - half), (plane[0] - half, plane[1] - half)]
    return [tuple(p) for p in view.mv.field_to_pixel(corners)]


class UgvViewTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((1000, 1224, 3), np.uint8)

    def test_height_corrected_position(self):
        view = drone_ugvs.UgvView(boundary(), 2.55, ugv_height=0.20)
        truth = [(0.6, 0.5), (5.5, 4.4), (3.06, 2.5)]
        found = view.detect(self.image, lambda img: [("ugv", 0.9 - 0.1 * k, quad_for(view, t)) for k, t in enumerate(truth)])
        self.assertEqual(len(found), 3)
        for (x, y, _s), (tx, ty) in zip(found, truth):
            self.assertAlmostEqual(x, tx, delta=0.005)
            self.assertAlmostEqual(y, ty, delta=0.005)
        # 높이를 안 고치면(0.40 평면 그대로) 가장자리에서 10 cm 넘게 안쪽으로 끌린다
        flat = drone_ugvs.UgvView(boundary(), 2.55, ugv_height=0.40)
        x, y, _s = flat.detect(self.image, lambda img: [("ugv", 0.9, quad_for(view, truth[0]))])[0]
        self.assertGreater(np.hypot(x - truth[0][0], y - truth[0][1]), 0.10)

    def test_arena_shift_matches_map(self):
        view = drone_ugvs.UgvView(boundary(), 2.55, arena=(6.12, 4.80))
        self.assertTrue(np.allclose(view.shift, (0.0, -0.10)))
        x, y, _s = view.detect(self.image, lambda img: [("ugv", 0.9, quad_for(view, (1.0, 2.4)))])[0]
        self.assertAlmostEqual(x, 1.0, delta=0.005)
        self.assertAlmostEqual(y, 2.4, delta=0.005)   # 4.80 지도의 정중앙 = 5.00 경계의 정중앙

    def test_filters(self):
        view = drone_ugvs.UgvView(boundary(), 2.55, conf=0.5)
        dets = [("ugv", 0.9, quad_for(view, (2.0, 2.0))), ("ugv", 0.6, quad_for(view, (2.05, 2.0))),   # 겹침 → 점수 높은 쪽
                ("ugv", 0.3, quad_for(view, (4.0, 1.0))), ("box", 0.9, quad_for(view, (4.0, 3.0)))]   # 점수 낮음 · 다른 종류
        found = view.detect(self.image, lambda img: dets)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0][0], 2.0, delta=0.005)
        self.assertEqual(view.detect(self.image, lambda img: []), [])

    def test_seen_range(self):
        """화면 위아래가 잘리면 그만큼 보이는 세로 범위가 준다 (높이가 있는 점이라 바닥보다 조금 더 안쪽까지만 보인다)."""
        low, high = drone_ugvs.UgvView(boundary(), 2.55).seen_y()
        cut_low, cut_high = drone_ugvs.UgvView(boundary(image_h=800), 2.55).seen_y()   # 화면 위아래 0.5 m 씩 잘림
        self.assertLess(low, 0.30)
        self.assertGreater(high, 4.70)
        self.assertAlmostEqual(cut_low - low, 0.46, delta=0.06)
        self.assertAlmostEqual(high - cut_high, 0.46, delta=0.06)


if __name__ == "__main__":
    unittest.main()
