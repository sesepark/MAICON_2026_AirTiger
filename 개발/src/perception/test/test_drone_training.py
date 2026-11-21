"""training/drone_boxes 시험: 조명 변형 · 시야 끝 자르기 · 데이터셋 늘리기 · 경기장 좌표 평가 (학습 자체는 GPU 서버에서 — 여기서는 하지 않는다).

  python -m unittest discover -s test -p "test_drone_training.py"     (패키지 폴더에서. numpy · OpenCV · PyYAML 이 없으면 건너뛴다)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "training", "drone_boxes"))

try:
    import cv2
    import numpy as np
    import yaml  # noqa: F401

    import augment
    import edge_cut
    import evaluate
    import photometric
except ImportError as e:  # 학습 도구 의존성이 없는 PC
    raise unittest.SkipTest("numpy · OpenCV · PyYAML 필요: %s" % e)


def make_image(seed=0):
    """map 그림 흉내: 612 × 500, 위 · 아래는 시야 밖 검정, 가운데 회색 바닥에 어두운 40 px 상자 둘."""
    rng = np.random.default_rng(seed)
    img = np.full((500, 612, 3), 120, np.uint8)
    img += rng.integers(0, 10, img.shape, dtype=np.uint8)
    img[:40] = 0
    img[460:] = 0
    img[100:140, 100:140] = 30
    img[300:340, 400:440] = 30
    return img


BOXES = [[(100, 100), (140, 100), (140, 140), (100, 140)], [(400, 300), (440, 300), (440, 340), (400, 340)]]
O2F = [[0.01, 0, 0], [0, -0.01, 5.0], [0, 0, 1]]


def make_dataset(root, n_train=3, n_val=2):
    for part, n in (("train", n_train), ("val", n_val)):
        for sub in ("images", "labels", "fieldmaps"):
            os.makedirs(os.path.join(root, sub, part), exist_ok=True)
        for i in range(n):
            stem = "s__test1__%s%d" % (part, i)
            augment.write_image(os.path.join(root, "images", part, stem + ".jpg"), make_image(i))
            with open(os.path.join(root, "labels", part, stem + ".txt"), "w") as fh:
                for pts in BOXES:
                    fh.write("0 " + " ".join("%.6f %.6f" % (u / 612, v / 500) for u, v in pts) + "\n")
            with open(os.path.join(root, "fieldmaps", part, stem + ".json"), "w") as fh:
                json.dump({"frame_id": "arena", "out_to_field": O2F, "boxes": []}, fh)
    with open(os.path.join(root, "data.yaml"), "w") as fh:
        fh.write("path: /somewhere/else\ntrain: images/train\nval: images/val\nnames:\n  0: box\n  1: stripe\n")


class PhotometricTest(unittest.TestCase):
    def test_all_ops_keep_shape_and_black_outside(self):
        img = make_image()
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), "training", "drone_boxes", "config", "augment.yaml"), encoding="utf-8"))
        out, used = photometric.apply(img, cfg["ops"], np.random.default_rng(1), force=True)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(set(used), set(cfg["ops"]))
        self.assertEqual(int(out[:40].max()), 0)  # 시야 밖은 연막 · 밝기를 걸어도 검정
        self.assertGreater(float(np.abs(out[200:300].astype(int) - img[200:300].astype(int)).mean()), 3)  # 안쪽은 바뀐다

    def test_seed_repeats(self):
        img = make_image()
        ops = {"exposure": {"p": 1, "range": [0.5, 1.5]}, "noise": {"p": 1, "sigma": [2, 5]}}
        a, _ = photometric.apply(img, ops, np.random.default_rng(7))
        b, _ = photometric.apply(img, ops, np.random.default_rng(7))
        self.assertTrue(np.array_equal(a, b))

    def test_fixed_values_and_unknown(self):
        img = make_image()
        dark, _ = photometric.apply(img, {"exposure": {"range": 0.5}}, np.random.default_rng(0), force=True)
        self.assertAlmostEqual(float(dark[200, 300, 0]), round(float(img[200, 300, 0]) * 0.5), delta=1)
        with self.assertRaises(ValueError):
            photometric.apply(img, {"sparkle": {}}, np.random.default_rng(0))

    def test_stress_conditions_run(self):
        img = make_image()
        for name, ops in photometric.STRESS.items():
            out, used = photometric.apply(img, ops, np.random.default_rng(0), force=True)
            self.assertEqual(out.shape, img.shape, name)
            self.assertEqual(set(used), set(ops), name)


class AugmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="drone_train_test_")
        self.ds = os.path.join(self.tmp, "ds")
        make_dataset(self.ds)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_augment_multiplies_train_only(self):
        out = os.path.join(self.tmp, "ds_aug")
        log = augment.augment(self.ds, out, {"exposure": {"p": 1, "range": [0.5, 1.5]}}, copies=3, seed=0)
        self.assertEqual(log["counts"], {"train": 3 * 4, "val": 2})
        self.assertEqual(len(os.listdir(os.path.join(out, "images", "train"))), 12)
        self.assertEqual(len(os.listdir(os.path.join(out, "labels", "train"))), 12)
        self.assertEqual(sorted(os.listdir(os.path.join(out, "images", "val"))), sorted(os.listdir(os.path.join(self.ds, "images", "val"))))
        same = lambda a, b: open(a).read() == open(b).read()  # noqa: E731
        self.assertTrue(same(os.path.join(self.ds, "labels", "train", "s__test1__train0.txt"),
                             os.path.join(out, "labels", "train", "s__test1__train0__aug2.txt")))  # 기하를 안 건드리니 라벨 그대로
        d = yaml.safe_load(open(os.path.join(out, "data.yaml")))
        self.assertEqual(d["path"], os.path.abspath(out).replace("\\", "/"))
        self.assertEqual(d["names"], {0: "box", 1: "stripe"})
        with self.assertRaises(SystemExit):  # 비어 있지 않은 출력에는 쓰지 않는다
            augment.augment(self.ds, out, {}, copies=1, seed=0)

    def test_drop_invisible_labels(self):
        lab = os.path.join(self.ds, "labels", "train", "s__test1__train0.txt")
        with open(lab, "a") as fh:  # 시야 밖(검정 띠, v < 40) 줄무늬 블록 라벨 하나
            fh.write("1 " + " ".join("%.6f %.6f" % (u / 612, v / 500) for u, v in [(10, 0), (50, 0), (50, 35), (10, 35)]) + "\n")
        out = os.path.join(self.tmp, "ds_aug2")
        log = augment.augment(self.ds, out, {}, copies=1, seed=0, min_visible=0.05)
        self.assertEqual(log["dropped_labels"], {"train": 1, "val": 0})
        for name in ("s__test1__train0.txt", "s__test1__train0__aug0.txt"):
            lines = open(os.path.join(out, "labels", "train", name)).read().split("\n")
            self.assertEqual(len([x for x in lines if x]), 2, name)  # 보이는 상자 둘만 남는다
        log = augment.augment(self.ds, os.path.join(self.tmp, "ds_aug3"), {}, copies=0, seed=0, min_visible=0)
        self.assertEqual(log["dropped_labels"]["train"], 0)  # 0 이면 끔

    def test_evaluate_with_fake_predictor(self):
        def perfect(img):
            return [(0, 0.9, pts) for pts in BOXES]

        res = evaluate.evaluate(perfect, self.ds, "val", stress=True, out=os.path.join(self.tmp, "ev"), log=lambda *_: None)
        base = res["conditions"]["원본"]["전체"]
        self.assertEqual((base["tp"], base["fp"], base["fn"]), (4, 0, 0))
        self.assertAlmostEqual(base["err_cm_max"], 0.0, places=3)
        self.assertEqual(len(res["conditions"]), 1 + len(photometric.STRESS))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "ev", "eval.md")))

        def shifted(img):  # 15 px = 15 cm 밀린 예측 + 엉뚱한 곳 하나 → 10 cm 안 맞춤에서 모두 틀림
            return [(0, 0.9, [(u + 15, v) for u, v in pts]) for pts in BOXES] + [(0, 0.5, [(500, 50), (540, 50), (540, 90), (500, 90)])]

        res = evaluate.evaluate(shifted, self.ds, "val", stress=False, log=lambda *_: None)
        s = res["conditions"]["원본"]["전체"]
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (0, 6, 4))

        def near(img):  # 5 px 밀림 → 맞음, 오차 5 cm
            return [(0, 0.9, [(u + 5, v) for u, v in pts]) for pts in BOXES]

        s = evaluate.evaluate(near, self.ds, "val", stress=False, log=lambda *_: None)["conditions"]["원본"]["전체"]
        self.assertEqual(s["tp"], 4)
        self.assertAlmostEqual(s["err_cm_median"], 5.0, places=3)

    def test_yaw_independent_of_vertex_order(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for k in range(4):
            self.assertAlmostEqual(evaluate.yaw_deg(sq[k:] + sq[:k]), 0.0, places=6)
            self.assertAlmostEqual(evaluate.yaw_deg(list(reversed(sq[k:] + sq[:k]))), 0.0, places=6)
        c, s = 0.9659258, 0.2588190  # 15°
        rot = [(x * c - y * s, x * s + y * c) for x, y in sq]
        self.assertAlmostEqual(evaluate.yaw_deg(rot), 15.0, places=3)
        self.assertAlmostEqual(evaluate.yaw_diff(44.0, -44.0), 2.0, places=6)


def box_line(x0, y0, x1, y1, w=612, h=500, cls=0):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return "%d " % cls + " ".join("%.6f %.6f" % (x / w, y / h) for x, y in pts)


class EdgeCutTest(unittest.TestCase):
    def setUp(self):
        self.img = np.full((500, 612, 3), 120, np.uint8)
        self.img[:20] = 0
        self.lines = [box_line(250, 60, 290, 100), box_line(250, 240, 290, 280), box_line(250, 420, 290, 460)]

    def test_cut_keeps_full_label_of_partly_visible_box(self):
        cfg = {"sides": {"top": 1.0}, "visible": [0.3, 0.3], "tilt": 0.0, "bend": 0.0, "second": 0.0}
        out, keep, info = edge_cut.cut(self.img, self.lines, cfg, np.random.default_rng(0), force=True)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(info["cuts"][0]["side"], "top")
        self.assertEqual(keep, self.lines)  # 위 상자는 30 % 남아 전체 라벨 그대로, 나머지는 손대지 않음
        self.assertTrue((out[:80, 250:290] == 0).all())  # 위 상자 위쪽 70 % 가 검다 (60 + 0.7 * 40 = 88)
        self.assertTrue((out[95:100, 250:290] == 120).all())
        self.assertTrue((out[240:280, 250:290] == 120).all())

    def test_mostly_hidden_box_is_blacked_and_dropped(self):
        cfg = {"sides": {"top": 1.0}, "visible": [0.3, 0.3], "keep_min": 0.5, "tilt": 0.0, "bend": 0.0, "second": 0.0}
        out, keep, info = edge_cut.cut(self.img, self.lines, cfg, np.random.default_rng(0), force=True)
        self.assertEqual(keep, self.lines[1:])
        self.assertEqual(info["dropped"], 1)
        self.assertTrue((out[60:100, 250:290] == 0).all())

    def test_depth_limit_and_probability(self):
        cfg = {"sides": {"top": 1.0}, "max_depth": 0.05}  # 위 끝에서 5 % 안쪽 상자가 없다 → 자르지 않는다
        out, keep, info = edge_cut.cut(self.img, self.lines, cfg, np.random.default_rng(0), force=True)
        self.assertIsNone(info)
        self.assertIs(out, self.img)
        out, keep, info = edge_cut.cut(self.img, self.lines, {"p": 0.0}, np.random.default_rng(0))
        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
