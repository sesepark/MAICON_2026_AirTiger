"""상자 찾기 모델 경로 시험: map 그림(drone_map_view) · 모델 출력 → 상자(drone_box_model) · 지도 연결(map_maker, box_detector="model").

  python3 -m unittest discover -s test -p "test_drone_box_model.py"     (패키지 폴더에서. ROS 불필요, 약 30초)

모델 파일 없이 돈다 — 가짜 예측기를 넣는다. 학습한 가중치(data/drone_models/*/export/*.pt)와 ultralytics 가 있으면 진짜 모델로도 본다.
"""
import glob
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from perception import drone_box_model, map_maker  # noqa: E402
from perception.drone_boundary import detect_boundary, read_image  # noqa: E402
from perception.drone_map_view import MapView  # noqa: E402

DATA = os.path.join(HERE, "data")


def quads_at(view, centers, side=0.40):
    """경기장 가운데들 → map 그림 꼭짓점 픽셀 (모델이 낼 법한 출력)."""
    out = []
    for x, y in centers:
        h = side / 2.0
        out.append([tuple(p) for p in view.field_to_pixel([(x - h, y + h), (x + h, y + h), (x + h, y - h), (x - h, y - h)])])
    return out


class ThreadLimitTests(unittest.TestCase):
    """ultralytics 가 되돌려 놓는 torch 스레드 수를 계산 프로세스의 OMP_NUM_THREADS 로 다시 묶는다 (drone_box_model.limit_threads)."""

    def setUp(self):
        self.before, self.torch = os.environ.get("OMP_NUM_THREADS"), sys.modules.get("torch")
        self.calls = []
        sys.modules["torch"] = type(sys)("torch")
        sys.modules["torch"].set_num_threads = self.calls.append

    def tearDown(self):
        os.environ.pop("OMP_NUM_THREADS", None)
        if self.before is not None:
            os.environ["OMP_NUM_THREADS"] = self.before
        sys.modules.pop("torch", None)
        if self.torch is not None:
            sys.modules["torch"] = self.torch

    def test_follows_the_environment_and_does_nothing_without_it(self):
        os.environ.pop("OMP_NUM_THREADS", None)
        self.assertIsNone(drone_box_model.limit_threads())
        os.environ["OMP_NUM_THREADS"] = "1"
        self.assertEqual(drone_box_model.limit_threads(), 1)
        os.environ["OMP_NUM_THREADS"] = "many"
        self.assertIsNone(drone_box_model.limit_threads())
        self.assertEqual(self.calls, [1])

    def test_load_limits_the_threads_after_the_first_run(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        order = []

        class Fake(drone_box_model.BoxModel):
            def predict(self, bgr, conf=0.05):
                order.append("predict")
                return []

        ultra, real = type(sys)("ultralytics"), sys.modules.get("ultralytics")
        ultra.YOLO = lambda path, task=None: type("M", (), {"names": {0: "box"}})()
        sys.modules["ultralytics"] = ultra
        sys.modules["torch"].set_num_threads = lambda n: order.append(("threads", n))
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt") as fh:
                Fake(fh.name).load()
        finally:
            sys.modules.pop("ultralytics", None)
            if real is not None:
                sys.modules["ultralytics"] = real
        self.assertEqual(order, ["predict", ("threads", 1)])


class MapViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = read_image(os.path.join(DATA, "drone_test2.jpg"))
        cls.boundary = detect_boundary(cls.image)
        cls.view = MapView(cls.boundary, 2.6)

    def test_size_and_round_trips(self):
        v = self.view
        self.assertEqual(v.size, (612, 500))
        pts = np.array([(0.0, 0.0), (6.12, 5.0), (1.234, 3.21)])
        self.assertTrue(np.allclose(v.pixel_to_field(v.field_to_pixel(pts)), pts))
        self.assertTrue(np.allclose(v.field_to_pixel([(0.0, 5.0)]), [(0.0, 0.0)]))  # 왼쪽 위 = (0, H)
        self.assertTrue(np.allclose(v.plane_of_floor(v.floor_of(pts)), pts))
        self.assertAlmostEqual(v.s, 2.6 / 2.2)

    def test_box_top_lands_on_its_footprint(self):
        """높이 0.4 m 인 점은 바닥 점 N + (P − N)·s 와 같은 원본 픽셀에 찍힌다 → map 그림에서는 P 자리에 그려진다."""
        v = self.view
        P = np.array([(1.0, 1.0), (5.0, 4.2), (3.06, 2.5)])
        src_floor = self.boundary.field_to_pixel(v.floor_of(P))  # 원본에서 윗면 점이 찍힌 픽셀
        mu, mv = v.maps()
        uv = np.round(v.field_to_pixel(P)).astype(int)
        for (u, w), s in zip(uv, src_floor):
            self.assertAlmostEqual(float(mu[w, u]), s[0], delta=1.0)
            self.assertAlmostEqual(float(mv[w, u]), s[1], delta=1.0)

    def test_render(self):
        img = self.view.render(self.image)
        self.assertEqual(img.shape, (500, 612, 3))
        self.assertGreater(float(img[200:300, 250:350].mean()), 20)  # 가운데는 찍힌다

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            MapView(self.boundary, 0.3)


class ModelPathTests(unittest.TestCase):
    """map_maker.compute(box_detector="model") — 가짜 예측기로."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            cls.jpeg = fh.read()
        cls.rules, cls.rules_info = map_maker.compute(cls.jpeg, 1234.5, "slot", box_detector="rules")
        cls.centers = [(b["x"], b["y"]) for b in cls.rules["boxes"]]
        boundary = detect_boundary(read_image(os.path.join(DATA, "drone_test2.jpg")))
        cls.view = MapView(boundary, 2.6)  # 그림 픽셀 ↔ 경기장은 높이와 상관없다 (평면 좌표)

    def fake(self, extra=()):
        quads = quads_at(self.view, self.centers)

        def predict(img):
            self.assertEqual(img.shape, (500, 612, 3))  # 모델은 map 그림을 받는다
            return [("box", 0.9, q) for q in quads] + list(extra)

        return predict

    def test_same_map_as_rules_when_boxes_agree(self):
        data, info = map_maker.compute(self.jpeg, 1234.5, "slot", box_detector="model", predict=self.fake())
        self.assertEqual(info["box_detector"], "model")
        self.assertTrue(info["ok"], info)
        self.assertEqual(info["height_source"], "measured")
        self.assertEqual(len(data["boxes"]), len(self.rules["boxes"]))
        for a, b in zip(data["boxes"], self.rules["boxes"]):  # 같은 순서 (위 줄부터 · 왼쪽부터)
            self.assertAlmostEqual(a["x"], b["x"], places=6)
            self.assertAlmostEqual(a["y"], b["y"], places=6)
        self.assertEqual(data["faces"], self.rules["faces"])
        self.assertEqual(data["sectors"], self.rules["sectors"])
        self.assertEqual(data["header"], {"frame_id": "arena", "stamp": 1234.5})

    def test_filters(self):
        q = quads_at(self.view, [(3.06, 2.5)])[0]
        wall = quads_at(self.view, [(0.3, 2.5)])[0]  # 끝 띠 안
        big = quads_at(self.view, [(2.0, 2.5)], side=0.8)[0]
        dup = quads_at(self.view, [(self.centers[0][0] + 0.05, self.centers[0][1])])[0]
        extra = [("stripe", 0.95, q), ("box", 0.1, q), ("box", 0.9, wall), ("box", 0.9, big), ("box", 0.5, dup)]
        image = read_image(os.path.join(DATA, "drone_test2.jpg"))
        bv, mv = drone_box_model.detect_boxes_model(image, self.view.boundary, self.fake(extra), 2.6, "given")
        self.assertEqual(len(bv.boxes), len(self.centers))
        reasons = " / ".join(r["reason"] for r in bv.rejected)
        for word in ("종류 stripe", "끝 띠", "크기", "겹침"):
            self.assertIn(word, reasons)
        self.assertTrue(bv.ok)
        self.assertEqual(len(bv.debug["yaw_deg"]), len(bv.boxes))
        bv2, _ = drone_box_model.detect_boxes_model(image, self.view.boundary, self.fake(), 2.65, "default")
        self.assertFalse(bv2.ok)  # 높이를 기본값으로 짐작했으면 지도로 내지 않는다 (drone_boxes 와 같은 기준)

    def test_camera_height_is_cached_then_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "drone_boundary.json")
            sources = []
            for _ in range(map_maker.HEIGHT_MIN + 1):
                _data, info = map_maker.compute(self.jpeg, 1.0, "slot", None, path, box_detector="model", predict=self.fake())
                sources.append(info["height_source"])
            self.assertEqual(sources, ["measured"] * map_maker.HEIGHT_MIN + ["cache"])
            saved = map_maker.load_boundaries(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(len(saved[0]["camera_heights"]), map_maker.HEIGHT_MIN)
            _data, info = map_maker.compute(self.jpeg, 1.0, "slot", 2.7, path, box_detector="model", predict=self.fake())
            self.assertEqual((info["height_source"], info["camera_height"]), ("given", 2.7))
            _data, _info = map_maker.compute(self.jpeg, 1.0, "slot", None, path, box_detector="rules")  # 규칙으로 돌려도 쌓인 높이는 남는다
            self.assertEqual(len(map_maker.load_boundaries(path)[0]["camera_heights"]), map_maker.HEIGHT_MIN)

    def test_missing_model_falls_back_to_rules(self):
        from unittest import mock

        with mock.patch.object(drone_box_model, "bundled_model", return_value=None):  # git 에 넣은 모델도 없을 때
            data, info = map_maker.compute(self.jpeg, 1.0, "slot", box_detector="model", box_model="/nonexistent/drone_boxes.engine")
        self.assertEqual(info["box_detector"], "rules")
        self.assertTrue(info["ok"], info)
        self.assertTrue(any("상자 모델을 못 씀" in n for n in info["notes"]), info["notes"])
        self.assertEqual(len(data["boxes"]), len(self.rules["boxes"]))


    def test_bundled_model_is_in_git(self):
        """git 에 넣은 모델 파일이 있고, 기본 자리에 파일이 없으면 그것을 고른다 (읽기는 ultralytics 가 있을 때만)."""
        path = drone_box_model.bundled_model()
        self.assertIsNotNone(path)
        self.assertTrue(path.endswith(drone_box_model.BUNDLED_NAME))


WEIGHTS = sorted(glob.glob(os.path.join(PKG, "data", "drone_models", "*", "export", "*.pt")))


@unittest.skipUnless(WEIGHTS, "학습한 가중치가 없다 (data/drone_models/*/export/*.pt)")
class RealModelTests(unittest.TestCase):
    def test_real_model_agrees_with_rules(self):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.skipTest("ultralytics 없음")
        for name in ("drone_test1.jpg", "drone_test2.jpg"):
            with open(os.path.join(DATA, name), "rb") as fh:
                jpeg = fh.read()
            rules, _ = map_maker.compute(jpeg, 1.0, "slot", box_detector="rules")
            data, info = map_maker.compute(jpeg, 1.0, "slot", box_detector="model", box_model=WEIGHTS[-1])
            self.assertEqual(info["box_detector"], "model", info["notes"])
            self.assertTrue(info["ok"], info)
            far = [b for b in data["boxes"] if min(math.dist((b["x"], b["y"]), (r["x"], r["y"])) for r in rules["boxes"]) > 0.15]
            self.assertLessEqual(len(far), 1, (name, far))
            self.assertGreaterEqual(len(data["boxes"]), len(rules["boxes"]) - 1, name)


if __name__ == "__main__":
    unittest.main()
