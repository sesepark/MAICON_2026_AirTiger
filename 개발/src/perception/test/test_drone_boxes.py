"""드론 화면 상자 찾기 (perception/drone_boxes.py) 시험.

1) 합성 장면: 경계(바닥 대응)와 카메라(경기장 정중앙 위 높이 Hc)를 정해 두고 40 cm 상자를 그려서, 위치 · 카메라 높이를 맞히는지.
   정답을 아는 유일한 시험이다 — 시차 식(윗면 = N + s·(P − N))과 옆면으로 s 재기를 검사한다.
2) 연습 화면 test1 · test2 (test/data, 2026-11-18): 상자 수와 위치가 그때 확인한 값과 같은지 (회귀 시험).
   정답은 사람이 그림으로 확인한 상자 14개씩. 좌표는 이 코드가 낸 값을 적었다 — 경계 지도의 축척 오차(A4: y 비율 1.02~1.15)를
   그대로 물려받으므로 실제 m 와는 몇 cm 다를 수 있다.
3) 연습 화면에서 물리적으로 맞아야 하는 것: 붙어 있는 두 상자의 간격 0.40 m, test2 배치의 좌우 대칭.
4) 2026-11-19 2경기장 실차 두 판에서 나온 것: 붙은 상자 줄(윗면 사이 모서리가 안 보임) · 격자에 맞춘 상자 자리 · 지도에 적는 벽 크기.
   합성 장면으로 보고, 그날의 드론 사진이 이 PC 에 있으면(연습경기 기록 폴더) 그것으로도 본다 — 사진은 저장소에 넣지 않는다.
ROS 없이 numpy · OpenCV 만으로 돈다:  python3 -m unittest test.test_drone_boxes   (약 15초)
"""
import os
import sys
import unittest

import numpy as np

try:
    import cv2
except ImportError:  # OpenCV 가 없는 곳에서는 건너뛴다
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

if cv2 is not None:
    from perception.drone_boundary import Boundary, Lens, detect_boundary, read_image
    from perception.drone_boxes import BoxParams, _frame_text, _grid_divisors, detect_boxes

DATA = os.path.join(HERE, "data")
W, H = 6.12, 5.0
_cache = {}


def real(name):
    if name not in _cache:
        img = read_image(os.path.join(DATA, name))
        assert img is not None, name
        b = detect_boundary(img)
        _cache[name] = (img, b, detect_boxes(img, b))
    return _cache[name]


# ------------------------------------------------------------------ 합성 장면
# 옆면 그림 대신 칠할 색 (BGR). 노란 계열은 뺐다 — 노란색은 거점 테두리 · 줄무늬 블록을 거르는 데 쓴다 (실제 옆면은 풍경 · 물체 사진)
FACE_COLOURS = [(180, 120, 60), (90, 150, 70), (60, 80, 150), (150, 150, 155), (200, 200, 205), (70, 60, 90), (140, 90, 160), (120, 170, 200)]
SYN_BOXES = [(1.45, 4.10), (1.85, 4.10), (4.25, 4.12), (4.65, 4.12), (1.47, 2.48), (1.87, 2.48), (4.25, 2.50), (4.65, 2.50),
             (3.06, 3.15), (3.06, 1.80), (1.45, 0.90), (1.85, 0.90), (4.30, 0.88), (2.20, 3.40)]


def synthetic(boxes, hc=2.6, seed=0, plain=None):
    """바닥 대응을 정한 경계 + 그 경계로 그린 화면 (BGR). 카메라는 경기장 정중앙 (W/2, H/2) 위 hc m.

    plain: {상자 번호: 변 글자("S" · "E" · "N" · "W" 를 이어 붙인 것)} — 그 변의 테두리 글자 띠를 그리지 않는다 (붙은 두 윗면 사이가 안 보이는 장면).
    """
    rng = np.random.RandomState(seed)
    size = (1280, 720)
    src = np.float32([(0, 0), (W, 0), (W, H), (0, H)])
    dst = np.float32([(290, 700), (1000, 712), (992, 18), (300, 26)])  # 조금 기운 원근
    f2p = cv2.getPerspectiveTransform(src, dst)
    hom = np.linalg.inv(f2p)
    hom /= hom[2, 2]
    b = Boundary(True, k=0.0, hom=hom, image_size=size, lens=Lens.simple(0.0, size), strip=0.88, cm_half=0.60, circle_r=0.62, field_w=W, field_h=H)
    N = (W / 2, H / 2)

    def px(pts, z=0.0):
        k = hc / (hc - z)
        q = [(N[0] + (x - N[0]) * k, N[1] + (y - N[1]) * k) for x, y in pts]
        return np.round(b.field_to_pixel(q)).astype(np.int32)

    img = np.full((size[1], size[0], 3), (170, 185, 195), np.uint8)  # 경기장 밖 (밝은 바닥)
    cv2.fillConvexPoly(img, px([(0, 0), (W, 0), (W, H), (0, H)]), (105, 105, 108))
    lw = 3
    for x0 in (0.88, W / 2 - 0.6, W / 2, W / 2 + 0.6, W - 0.88):
        cv2.polylines(img, [px([(x0, 0), (x0, H)])], False, (225, 225, 225), lw)
    for y0 in (H / 3, 2 * H / 3):
        cv2.polylines(img, [px([(0.88, y0), (W - 0.88, y0)])], False, (225, 225, 225), lw)
    t = np.linspace(0, 2 * np.pi, 90)
    cv2.polylines(img, [px(list(zip(W / 2 + 0.62 * np.cos(t), H / 2 + 0.62 * np.sin(t))))], True, (225, 225, 225), lw)
    a = 0.2
    for bx, by in sorted(boxes, key=lambda p: -np.hypot(p[0] - N[0], p[1] - N[1])):
        c = {"SW": (bx - a, by - a), "SE": (bx + a, by - a), "NE": (bx + a, by + a), "NW": (bx - a, by + a)}
        faces = {"E": ("SE", "NE", bx + a < N[0]), "W": ("NW", "SW", bx - a > N[0]), "N": ("NE", "NW", by + a < N[1]), "S": ("SW", "SE", by - a > N[1])}
        for k1, k2, vis in faces.values():
            if not vis:
                continue
            quad = np.concatenate([px([c[k1], c[k2]]), px([c[k2], c[k1]], 0.4)])
            cv2.fillConvexPoly(img, quad, FACE_COLOURS[rng.randint(len(FACE_COLOURS))])
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, quad, 1)
            for _ in range(6):  # 그림처럼 보이게 무늬
                p1 = tuple(int(v) for v in quad[rng.randint(4)])
                p2 = tuple(int(v) for v in quad[rng.randint(4)])
                layer = img.copy()
                cv2.line(layer, p1, p2, tuple(int(v) for v in rng.randint(0, 255, 3)), 2)
                img[mask > 0] = layer[mask > 0]
        top = [c["SW"], c["SE"], c["NE"], c["NW"]]
        cv2.fillConvexPoly(img, px(top, 0.4), (38, 38, 42))
        r = a - 0.025  # 윗면 테두리 글자 띠 (점선)
        bare = (plain or {}).get(boxes.index((bx, by)), "")
        for side, (x1, y1), (x2, y2) in zip("SENW", [(-r, -r), (r, -r), (r, r), (-r, r)], [(r, -r), (r, r), (-r, r), (-r, -r)]):
            if side in bare:
                continue
            for u in np.arange(0.05, 0.95, 0.12):
                p1 = (bx + x1 + (x2 - x1) * u, by + y1 + (y2 - y1) * u)
                p2 = (bx + x1 + (x2 - x1) * (u + 0.07), by + y1 + (y2 - y1) * (u + 0.07))
                cv2.polylines(img, [px([p1, p2], 0.4)], False, (200, 200, 200), 1)
        cv2.polylines(img, [px([(bx - 0.03, by), (bx + 0.03, by)], 0.4)], False, (200, 200, 200), 1)
        cv2.polylines(img, [px([(bx, by - 0.03), (bx, by + 0.03)], 0.4)], False, (200, 200, 200), 1)
    noise = rng.normal(0, 3, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, b


def match(found, truth, tol):
    """truth 마다 가장 가까운 found 와의 거리. 짝이 없으면 inf."""
    out = []
    for tx, ty in truth:
        d = min((np.hypot(fx - tx, fy - ty) for fx, fy in found), default=np.inf)
        out.append(d)
    return np.array(out)


@unittest.skipIf(cv2 is None, "OpenCV 없음")
class TestSynthetic(unittest.TestCase):
    def test_positions_and_height(self):
        for hc in (2.6, 3.5):
            img, b = synthetic(SYN_BOXES, hc=hc)
            v = detect_boxes(img, b)
            with self.subTest(hc=hc):
                self.assertEqual(v.height_source, "measured", v.notes)
                self.assertAlmostEqual(v.camera[2], hc, delta=0.15 * hc)
                self.assertEqual(len(v.boxes), len(SYN_BOXES), [(round(x, 2), round(y, 2)) for x, y in v.positions()])
                err = match(v.positions(), SYN_BOXES, 0.05)
                self.assertLess(err.max(), 0.05, err.round(3))

    def test_given_height(self):
        """높이를 주면 재지 않고 그 값을 쓴다. 남는 오차는 윗면 변 측정에서 온다 (±1 cm 화면 픽셀)."""
        img, b = synthetic(SYN_BOXES, hc=2.6, seed=1)
        v = detect_boxes(img, b, BoxParams(camera_height=2.6))
        self.assertEqual(v.height_source, "given")
        err = match(v.positions(), SYN_BOXES, 0.05)
        self.assertLess(err.max(), 0.05, err.round(3))

    def test_no_boxes(self):
        img, b = synthetic([], hc=2.6)
        v = detect_boxes(img, b)
        self.assertEqual(v.boxes, [])
        self.assertFalse(v.ok)


# ------------------------------------------------------------------ 연습 화면
# 이 코드로 낸 값. 상자 수는 그림으로 확인했다 (test1 · test2 각 14개, 2026-11-18). 거르기 · s 를 바꾸면 다시 확인할 것.
# 2026-11-19 에 다시 적음: 상자 자리를 바닥 격자에 맞춰 내면서(drone_boxes._grid_divisors) 좌표가 최대 5.5 cm (test1) · 2.9 cm (test2) 달라졌다.
#   그 전 값(s 하나로 나눈 값)은 test1 의 상자들을 격자점보다 가로로 4 cm 쯤 바깥에 뒀다 (가운데에서 ±1.24 · ±1.64 m → 지금 ±1.20 ~ 1.23 · ±1.58 ~ 1.60).
#   그 전 값: test2 (1.483, 4.119) (1.873, 4.095) (4.247, 4.144) (4.620, 4.112) (3.092, 3.131) (1.486, 2.489) (1.904, 2.475) (4.241, 2.485) (4.622, 2.498)
#                   (3.075, 1.786) (1.458, 0.896) (1.860, 0.888) (4.277, 0.820) (4.666, 0.848)
#            test1 (1.822, 4.113) (4.303, 4.130) (4.696, 4.126) (4.800, 3.700) (3.890, 2.917) (1.415, 2.487) (1.790, 2.502) (4.317, 2.494) (4.683, 2.491)
#                   (3.909, 2.055) (4.271, 2.062) (1.429, 0.872) (1.796, 0.860) (4.313, 0.862)
EXPECTED = {
    "drone_test2.jpg": [(1.464, 4.099), (1.858, 4.076), (4.262, 4.124), (4.639, 4.092), (3.092, 3.117), (1.466, 2.490), (1.890, 2.475),
                        (4.256, 2.485), (4.641, 2.498), (3.083, 1.799), (1.438, 0.916), (1.845, 0.908), (4.292, 0.841), (4.686, 0.868)],
    "drone_test1.jpg": [(1.858, 4.085), (4.266, 4.102), (4.649, 4.098), (4.750, 3.679), (3.866, 2.910), (1.463, 2.487), (1.827, 2.502),
                        (4.281, 2.494), (4.636, 2.492), (3.884, 2.063), (4.235, 2.069), (1.477, 0.900), (1.833, 0.888), (4.277, 0.890)],
}


@unittest.skipIf(cv2 is None, "OpenCV 없음")
class TestPractice(unittest.TestCase):
    def test_boxes_match(self):
        for name, want in EXPECTED.items():
            with self.subTest(name=name):
                _img, _b, v = real(name)
                self.assertTrue(v.ok, v.notes)
                self.assertEqual(len(v.boxes), len(want), v.positions())
                err = match(v.positions(), want, 0.05)
                self.assertLess(err.max(), 0.05, err.round(3))

    def test_camera_height(self):
        """천장 카메라 (현장 확인 2026-11-18). 연습 화면 두 대 모두 2.5 ~ 2.7 m 로 잰다 (합성 장면 시험에서 잰 값의 치우침 −2 %)."""
        for name in EXPECTED:
            with self.subTest(name=name):
                _img, _b, v = real(name)
                self.assertEqual(v.height_source, "measured")
                self.assertGreater(v.faces_used, 5)
                self.assertTrue(2.3 < v.camera[2] < 3.0, v.camera)

    def test_touching_pairs(self):
        """test2 의 붙은 상자 쌍 6개: 가운데 사이가 0.40 m (지도 x 축척 오차 · 윗면 변 측정 오차로 ±5 cm)."""
        _img, _b, v = real("drone_test2.jpg")
        pos = v.positions()
        pairs = 0
        for i, (x1, y1) in enumerate(pos):
            for x2, y2 in pos[i + 1:]:
                if abs(y1 - y2) < 0.1 and abs(x1 - x2) < 0.6:
                    pairs += 1
                    self.assertAlmostEqual(abs(x1 - x2), 0.40, delta=0.05)
        self.assertEqual(pairs, 6)

    def test_mirror_symmetry(self):
        """test2 배치는 좌우 대칭 (지정표적 규칙). 윗면이 화면 밖으로 잘린 상자는 빼고 ±6 cm (실제 놓인 자리 차이도 들어 있다)."""
        _img, _b, v = real("drone_test2.jpg")
        whole = [(b.x, b.y) for b in v.boxes if b.cut is None]
        for x, y in whole:
            if abs(x - W / 2) < 0.3:
                continue
            twin = [(x2, y2) for x2, y2 in whole if abs(W - x2 - x) < 0.15 and abs(y2 - y) < 0.15]
            if not twin:
                continue
            d = min(np.hypot(W - x2 - x, y2 - y) for x2, y2 in twin)
            self.assertLess(d, 0.06, (x, y))


# ------------------------------------------------------------------ 붙은 상자 줄 · 격자 · 벽 크기 (2026-11-19)
# 따로 찾히는 상자들 (윗면 크기의 중앙값을 내는 데 쓰인다). 모두 20 cm 격자점 (경기장 정중앙에서 0.2 m 의 정수배)
GRID_BOXES = [(1.46, 4.10), (1.86, 4.10), (4.26, 4.10), (4.66, 4.10), (1.46, 2.50), (4.66, 2.50), (1.46, 0.90), (1.86, 0.90), (4.26, 0.90), (4.66, 0.90)]


def found_near(view, x, y, tol=0.05):
    """(x, y) 에서 tol 안에 있는 상자 (없으면 None)."""
    near = [b for b in view.boxes if np.hypot(b.x - x, b.y - y) < tol]
    return near[0] if near else None


@unittest.skipIf(cv2 is None, "OpenCV 없음")
class TestJoined(unittest.TestCase):
    """맞닿은 두 윗면 사이의 모서리가 안 보이는 줄. 정사각형 찾기는 둘 다 놓치고(joined=False 로 확인), 줄 전체의 바깥 변으로 찾아야 한다."""

    def check(self, boxes, pair, plain=None):
        for hc in (2.6, 3.0):
            with self.subTest(hc=hc):
                img, b = synthetic(boxes, hc=hc, plain=plain)
                before = detect_boxes(img, b, BoxParams(joined=False))
                self.assertEqual([found_near(before, x, y) for x, y in pair], [None, None], "이 장면은 고치기 전에도 찾아져서 시험이 되지 않는다")
                v = detect_boxes(img, b)
                self.assertEqual(len(v.boxes), len(boxes), [(round(x, 2), round(y, 2)) for x, y in v.positions()])
                for x, y in pair:
                    box = found_near(v, x, y)
                    self.assertIsNotNone(box, ((x, y), v.debug.get("joined")))
                    self.assertTrue(box.joined)
                self.assertLess(match(v.positions(), boxes, 0.05).max(), 0.05)
                self.assertEqual(sum(1 for box in v.boxes if box.joined), 2)

    def test_pair_along_the_centre_line(self):
        """10:35 판: 가운데 선 위에 세로로 붙은 두 상자. 두 윗면이 같은 검은 판이라 사이에 밝기 차이가 없다 (그 변의 글자 띠도 안 보임)."""
        pair = [(3.06, 1.50), (3.06, 1.10)]
        boxes = GRID_BOXES + pair
        self.check(boxes, pair, plain={len(GRID_BOXES): "S", len(GRID_BOXES) + 1: "N"})

    def test_pair_across_the_centre_line(self):
        """10:50 판: 가운데 선을 사이에 두고 가로로 붙은 두 상자. 사이 모서리가 칠한 선 위라 선을 지울 때 함께 지워진다.

        합성 장면은 선이 가늘고 반듯해 그 변의 글자 띠가 지운 띠 밖으로 조금 남는다 (그것만으로 찾아진다) → 실제 사진처럼 안 보이게 뺀다.
        """
        pair = [(2.86, 1.30), (3.26, 1.30)]
        self.check(GRID_BOXES + pair, pair, plain={len(GRID_BOXES): "E", len(GRID_BOXES) + 1: "W"})

    def test_three_in_a_row(self):
        pair = [(3.46, 3.50), (3.86, 3.50), (4.26, 3.50)]
        boxes = GRID_BOXES + pair
        n = len(GRID_BOXES)
        img, b = synthetic(boxes, hc=2.6, plain={n: "E", n + 1: "WE", n + 2: "W"})
        v = detect_boxes(img, b)
        self.assertEqual(len(v.boxes), len(boxes), [(round(x, 2), round(y, 2)) for x, y in v.positions()])
        self.assertLess(match(v.positions(), boxes, 0.05).max(), 0.05)
        self.assertEqual(sum(1 for box in v.boxes if box.joined), 3)

    def test_frame_text_tells_a_top_from_bare_floor(self):
        """상자들 사이의 빈 바닥도 바깥 변이 곧은 직사각형이다 (test1 에서 그렇게 없는 상자 둘이 나왔다) → 윗면 테두리 글자가 보여야 받는다."""
        img, b = synthetic(GRID_BOXES, hc=2.6)
        v = detect_boxes(img, b)
        p = BoxParams()
        R = v.debug["raster"]
        for box in v.boxes:
            self.assertGreater(_frame_text(R, p, box.top), p.joined_min_frame, (box.x, box.y))
        for x, y in ((3.36, 1.2), (2.2, 3.4), (2.8, 1.3), (1.2, 1.7)):  # 빈 바닥 — 칠한 선에 걸친 곳도
            self.assertLess(_frame_text(R, p, (x - 0.24, x + 0.24, y - 0.24, y + 0.24)), p.joined_min_frame, (x, y))

    def test_local_practice_pictures(self):
        """2026-11-19 2경기장 두 판의 드론 사진 (이 PC 의 연습경기 기록 폴더에 있을 때만). 지도에 없던 가운데 선 위의 두 상자가 나와야 한다."""
        root = os.environ.get("MAICON_SESSIONS", os.path.join(os.path.expanduser("~"), ".config", "maicon-match", "sessions"))
        cases = {"20261119_103514": (13, [(3.07, 1.73), (3.07, 1.30)]),  # 라이다가 본 동쪽 면: x 3.28 ~ 3.31, y 1.01 ~ 1.85 (아래 벽 기준 — 지도보다 0.1 m 아래)
                 "20261119_104911": (10, [(2.82, 1.32), (3.22, 1.32)])}
        seen = 0
        for session, (count, pair) in cases.items():
            img = read_image(os.path.join(root, session, "perception", "drone_used.jpg"))
            if img is None:
                continue
            seen += 1
            with self.subTest(session=session):
                v = detect_boxes(img, detect_boundary(img))
                self.assertTrue(v.ok, v.notes)
                self.assertEqual(len(v.boxes), count, [(round(x, 2), round(y, 2)) for x, y in v.positions()])
                for x, y in pair:
                    self.assertIsNotNone(found_near(v, x, y, 0.08), (x, y))
        if not seen:
            self.skipTest("그날의 드론 사진이 이 PC 에 없음")


def cut_top(img, b, y_cut, repaint=()):
    """합성 장면의 위쪽을 잘라 낸다 (경기장 좌표 y_cut 보다 위는 화면 밖) → (잘린 화면, 그 화면의 경계). 카메라가 경기장을 다 비추지 않는 것.

    repaint: 그 상자들의 남쪽 옆면을 무늬가 뚜렷한 그림으로 다시 칠한다 (합성 장면의 옆면 색은 무작위라 회색이 걸리면 바닥과 구별되지 않는다).
    """
    hc, n = 2.6, (W / 2, H / 2)
    img = img.copy()
    for bx, by in repaint:
        foot = [(bx - 0.2, by - 0.2), (bx + 0.2, by - 0.2)]
        k = hc / (hc - 0.4)
        top = [(n[0] + (x - n[0]) * k, n[1] + (y - n[1]) * k) for x, y in foot]
        quad = np.round(b.field_to_pixel(foot + top[::-1])).astype(np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, quad, 1)
        yy, xx = np.nonzero(mask)
        img[yy, xx] = np.where((((xx // 6) + (yy // 6)) % 2 == 0)[:, None], (90, 60, 190), (200, 120, 40))  # 붉은색 · 파랑 바둑무늬 (노랑 계열은 줄무늬 블록으로 걸러진다)
    row = int(round(float(b.field_to_pixel([(W / 2, y_cut)])[0][1])))
    shift = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(row)], [0.0, 0.0, 1.0]])  # 잘린 화면의 픽셀 → 원래 화면의 픽셀
    size = (img.shape[1], img.shape[0] - row)
    cut = Boundary(True, k=0.0, hom=b.hom @ shift, image_size=size, lens=Lens.simple(0.0, size), strip=b.strip, cm_half=b.cm_half,
                   circle_r=b.circle_r, field_w=W, field_h=H)
    return np.ascontiguousarray(img[row:]), cut


@unittest.skipIf(cv2 is None, "OpenCV 없음")
class TestBorder(unittest.TestCase):
    """화면 끝에 걸려 윗면의 먼 변이 화면 밖인 상자 — 카메라 쪽 옆면의 바닥선으로 찾는다 (_border_boxes)."""

    TOP_ROW = [(1.46, 4.10), (4.66, 4.10)]  # 서로 떨어진 상자 둘 (붙은 상자는 옆면 그림이 달라야 나뉘는데, 여기서는 같은 무늬로 칠한다)

    def test_boxes_cut_by_the_picture_border_are_found_by_their_side_face(self):
        boxes = self.TOP_ROW + GRID_BOXES[4:]
        img, b = synthetic(boxes, hc=2.6)
        cut_img, cut_b = cut_top(img, b, 4.03, repaint=self.TOP_ROW)  # 위쪽 줄 상자는 옆면 아래 13 cm 만 남는다 (윗면은 화면 밖)
        before = detect_boxes(cut_img, cut_b, BoxParams(border=False))
        self.assertEqual([found_near(before, x, y, 0.1) for x, y in self.TOP_ROW], [None] * 2, "이 장면은 고치기 전에도 찾아져서 시험이 되지 않는다")
        v = detect_boxes(cut_img, cut_b)
        self.assertEqual(len(v.boxes), len(boxes), [(round(x, 2), round(y, 2)) for x, y in v.positions()])
        for x, y in self.TOP_ROW:
            box = found_near(v, x, y, 0.06)
            self.assertIsNotNone(box, ((x, y), [t for t in v.debug.get("border", []) if t["side"] == "N"]))
            self.assertTrue(box.border)
            self.assertEqual(box.cut, "N")
        for x, y in GRID_BOXES[4:]:  # 다 보이는 상자는 전과 같은 길로 찾는다
            self.assertFalse(found_near(v, x, y, 0.05).border)

    def test_boxes_against_the_wall_are_left_out_unless_allowed(self):
        """벽에서 0.45 m 안(가운데 기준)은 내지 않는다 — 객체 상자는 벽에서 40 cm 넘게 띄워 놓고, 그 자리에서는 UGV 가 상자처럼 보이기 쉽다."""
        flush = [(3.86, 4.75)]
        img, b = synthetic(GRID_BOXES + flush, hc=2.6)
        cut_img, cut_b = cut_top(img, b, 4.72, repaint=flush)
        self.assertIsNone(found_near(detect_boxes(cut_img, cut_b), 3.86, 4.75, 0.1))
        allowed = detect_boxes(cut_img, cut_b, BoxParams(border_wall_gap=0.2))
        box = found_near(allowed, 3.86, 4.75, 0.06)
        self.assertIsNotNone(box, allowed.debug.get("border"))
        self.assertTrue(box.border)

    def test_whole_pictures_get_no_border_boxes(self):
        for name in EXPECTED:
            with self.subTest(name=name):
                _img, _b, v = real(name)
                self.assertEqual([b for b in v.boxes if b.border], [])

    def test_local_flush_boxes(self):
        """11:47 판과 같은 배치(10:50 판): 위쪽 벽에 붙여 놓은 상자 셋은 옆면 아래쪽만 보인다. 기본값으로는 내지 않고, border_wall_gap 을 낮추면 낸다."""
        root = os.environ.get("MAICON_SESSIONS", os.path.join(os.path.expanduser("~"), ".config", "maicon-match", "sessions"))
        img = read_image(os.path.join(root, "20261119_104911", "perception", "drone_used.jpg"))
        if img is None:
            self.skipTest("그날의 드론 사진이 이 PC 에 없음")
        b = detect_boundary(img)
        self.assertEqual([x for x in detect_boxes(img, b).boxes if x.border], [])
        v = detect_boxes(img, b, BoxParams(border_wall_gap=0.2))
        for x, y in ((2.62, 4.70), (3.49, 4.68), (4.68, 4.71)):
            box = found_near(v, x, y, 0.08)
            self.assertIsNotNone(box, (x, y, [(round(q.x, 2), round(q.y, 2)) for q in v.boxes if q.border]))
            self.assertEqual((box.border, box.cut), (True, "N"))


@unittest.skipIf(cv2 is None, "OpenCV 없음")
class TestGrid(unittest.TestCase):
    """상자 자리를 바닥의 20 cm 격자에 맞춰 내기 (_grid_divisors)."""

    N = (W / 2, H / 2)

    def tops(self, centres, kx, ky, half=0.24):
        out = []
        for x, y in centres:
            tx, ty = self.N[0] + (x - self.N[0]) * kx, self.N[1] + (y - self.N[1]) * ky
            out.append((tx - half, tx + half, ty - half, ty + half))
        return out

    def test_divisors_follow_the_grid(self):
        div, source, _info = _grid_divisors(self.tops(GRID_BOXES, 1.21, 1.15), self.N, self.N, 1.18, BoxParams())
        self.assertEqual(source, ("grid", "grid"))
        self.assertAlmostEqual(div[0], 1.21, delta=0.006)
        self.assertAlmostEqual(div[1], 1.15, delta=0.006)

    def test_boxes_off_the_grid_keep_s(self):
        rng = np.random.RandomState(3)
        loose = [(x + rng.uniform(-0.1, 0.1), y + rng.uniform(-0.1, 0.1)) for x, y in GRID_BOXES]
        div, source, _info = _grid_divisors(self.tops(loose, 1.21, 1.15), self.N, self.N, 1.18, BoxParams())
        self.assertEqual((div, source), ((1.18, 1.18), ("s", "s")))

    def test_too_few_boxes_keep_s(self):
        div, source, _info = _grid_divisors(self.tops(GRID_BOXES[:3], 1.21, 1.15), self.N, self.N, 1.18, BoxParams())
        self.assertEqual((div, source), ((1.18, 1.18), ("s", "s")))

    def test_map_scale_error_is_removed(self):
        """바닥 지도가 가로로 4 % 크고 세로로 2 % 작게 맞춰졌어도 (경계 맞추기의 축척 오차) 상자 자리는 정중앙 기준으로 맞게 나온다."""
        boxes = GRID_BOXES + [(2.86, 3.30), (3.26, 1.70)]
        img, b = synthetic(boxes, hc=2.6)
        ax, ay = 1.04, 0.98
        stretch = np.array([[ax, 0.0, W / 2 * (1 - ax)], [0.0, ay, H / 2 * (1 - ay)], [0.0, 0.0, 1.0]])
        wrong = Boundary(True, k=0.0, hom=stretch @ b.hom, image_size=b.image_size, lens=b.lens, strip=W / 2 - (W / 2 - 0.88) * ax,
                         cm_half=0.60 * ax, circle_r=0.62 * (ax + ay) / 2, field_w=W, field_h=H)
        plain = detect_boxes(img, wrong, BoxParams(grid_cell=0.0))
        self.assertGreater(match(plain.positions(), boxes, 0.05).max(), 0.05)  # s 하나로 나누면 축척 오차가 그대로 남는다
        v = detect_boxes(img, wrong)
        self.assertEqual(v.divisor_source, ("grid", "grid"), v.debug.get("grid"))
        self.assertEqual(len(v.boxes), len(boxes))
        self.assertLess(match(v.positions(), boxes, 0.05).max(), 0.04)
        self.assertAlmostEqual(v.divisor[0] / v.divisor[1], ax / ay, delta=0.02)

    def test_given_height_is_not_overridden(self):
        img, b = synthetic(GRID_BOXES, hc=2.6)
        v = detect_boxes(img, b, BoxParams(camera_height=2.6))
        self.assertEqual(v.divisor_source, ("s", "s"))
        self.assertAlmostEqual(v.divisor[0], 2.6 / 2.2, places=6)


# ------------------------------------------------------------------ field_map 형식
MSG_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "team_interfaces", "msg")


def msg_fields(name):
    """team_interfaces/msg/<name>.msg 의 칸 이름 (상수 · 주석 제외)."""
    out = []
    with open(os.path.join(MSG_DIR, name + ".msg"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line and "=" not in line.split()[1]:
                out.append(line.split()[1])
    return out


@unittest.skipIf(cv2 is None, "OpenCV 없음")
class TestFieldMap(unittest.TestCase):
    def test_fields_match_msg(self):
        """dict 칸 이름이 FieldMap · FieldRegion · FieldFace 메시지 정의와 같다."""
        img, b = synthetic(SYN_BOXES, hc=2.6)
        fm = detect_boxes(img, b).field_map(b, stamp=1763452800.25, base_naming="slot")
        self.assertEqual(sorted(fm), sorted(msg_fields("FieldMap")))
        self.assertEqual(sorted(fm["sectors"][0]), sorted(msg_fields("FieldRegion")))
        self.assertEqual(sorted(fm["bases"][0]), sorted(msg_fields("FieldRegion")))
        self.assertEqual(sorted(fm["faces"][0]), sorted(msg_fields("FieldFace")))
        self.assertEqual(fm["header"], {"frame_id": "arena", "stamp": 1763452800.25})

    def test_content(self):
        img, b = synthetic(SYN_BOXES, hc=2.6)
        v = detect_boxes(img, b)
        fm = v.field_map(b)
        self.assertEqual([(p["x"], p["y"]) for p in fm["boxes"]], [(round(x, 4), round(y, 4)) for x, y in v.positions()])
        self.assertEqual((fm["width"], fm["height"], fm["box_size"]), (W, H, 0.4))
        self.assertEqual([r["name"] for r in fm["sectors"]], ["B-S1", "B-S2", "B-S3", "CM", "R-S1", "R-S2", "R-S3"])
        self.assertTrue(all(r["row"] == 0 for r in fm["sectors"]))
        self.assertEqual(fm["bases"], [])  # 기본은 비움 — 받는 쪽마다 이름을 다르게 읽는다 (drone_field_map.py 참고)
        self.assertEqual(fm["vehicles"], [])
        # 붙은 쌍 5개(B-S1 · R-S1 · B-2 · R-2 · B-3)의 안쪽 면 10개는 빠진다
        self.assertEqual(len(fm["faces"]), 4 * len(SYN_BOXES) - 10)
        for f in fm["faces"]:
            box = fm["boxes"][f["box"]]
            self.assertAlmostEqual(f["cx"], box["x"] + 0.2 * f["nx"], places=3)
            self.assertAlmostEqual(f["cy"], box["y"] + 0.2 * f["ny"], places=3)
            self.assertEqual(f["image_ref"], "")
        for (x, y), want in (((1.45, 4.10), "B-S1"), ((3.06, 3.15), "CM"), ((4.30, 0.88), "R-S3"), ((1.87, 2.48), "B-S2")):
            i = int(np.argmin([np.hypot(p["x"] - x, p["y"] - y) for p in fm["boxes"]]))
            self.assertEqual({f["sector"] for f in fm["faces"] if f["box"] == i}, {want}, (x, y))

    def test_bases(self):
        img, b = synthetic([], hc=2.6)
        v = detect_boxes(img, b)
        slots = v.field_map(b, base_naming="slot")["bases"]
        self.assertEqual([(r["name"], r["row"]) for r in slots], [("B-1", 1), ("B-2", 2), ("B-3", 3), ("R-1", 1), ("R-2", 2), ("R-3", 3)])
        top = slots[0]
        self.assertEqual((top["min_x"], top["max_x"]), (0.0, 0.4))
        self.assertAlmostEqual((top["min_y"] + top["max_y"]) / 2, H * 5 / 6, places=3)
        kinds = {"BLUE": ["SUPPLY_DEPOT", "COMMAND_POST", "MAINTENANCE"], "RED": ["COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT"]}
        report = v.field_map(b, base_naming="report", base_kinds=kinds)["bases"]
        self.assertEqual(report[0]["name"], "BLUE_SUPPLY_DEPOT")
        self.assertEqual(report[3]["name"], "RED_COMMAND_POST")
        with self.assertRaises(ValueError):
            v.field_map(b, base_naming="report")

    def test_arena_size(self):
        """지도에 적는 벽 크기(arena)를 주면: 상자는 정중앙 기준 자리를 지키고, 벽 · 줄 경계 · 거점은 그 크기를 따른다. 안 주면 전과 같다."""
        img, b = synthetic(SYN_BOXES, hc=2.6)
        v = detect_boxes(img, b)
        old = v.field_map(b, base_naming="slot")
        new = v.field_map(b, base_naming="slot", arena=(6.0, 4.8))
        self.assertEqual((old["width"], old["height"]), (W, H))
        self.assertEqual((new["width"], new["height"]), (6.0, 4.8))
        for p0, p1 in zip(old["boxes"], new["boxes"]):
            self.assertAlmostEqual(p1["x"] - 3.0, p0["x"] - W / 2, places=3)
            self.assertAlmostEqual(p1["y"] - 2.4, p0["y"] - H / 2, places=3)
        self.assertEqual(len(new["faces"]), len(old["faces"]))
        sectors = {r["name"]: r for r in new["sectors"]}
        self.assertAlmostEqual(sectors["B-S1"]["min_y"], 3.2, places=3)  # 줄 경계 = 세로를 셋으로 나눈 곳
        self.assertAlmostEqual(sectors["CM"]["max_x"] - sectors["CM"]["min_x"], 1.2 * 6.0 / W, places=3)  # 구역선은 가로 비율만큼
        self.assertAlmostEqual((sectors["CM"]["max_x"] + sectors["CM"]["min_x"]) / 2, 3.0, places=3)
        top = {r["name"]: r for r in new["bases"]}["B-1"]
        self.assertAlmostEqual((top["min_y"] + top["max_y"]) / 2, 4.0, places=3)

    def test_practice_faces(self):
        """test2: 상자 14개 · 붙은 쌍 6개 → 면 56 − 12 = 44."""
        _img, b, v = real("drone_test2.jpg")
        fm = v.field_map(b)
        self.assertEqual(len(fm["boxes"]), 14)
        self.assertEqual(len(fm["faces"]), 44)
        self.assertTrue(all(f["sector"] for f in fm["faces"]))

    def test_msg(self):
        """ROS 2 환경(team_interfaces 빌드)에서만: dict → team_interfaces/FieldMap."""
        try:
            import team_interfaces.msg  # noqa: F401
        except ImportError:
            self.skipTest("team_interfaces 없음 (ROS 2 환경에서만)")
        from perception.drone_field_map import field_map_msg

        img, b = synthetic(SYN_BOXES, hc=2.6)
        fm = detect_boxes(img, b).field_map(b, stamp=10.5, base_naming="slot")
        m = field_map_msg(fm)
        self.assertEqual((m.header.frame_id, m.header.stamp.sec, m.header.stamp.nanosec), ("arena", 10, 500000000))
        self.assertEqual(len(m.boxes), len(fm["boxes"]))
        self.assertEqual(len(m.faces), len(fm["faces"]))
        self.assertEqual(m.bases[0].name, "B-1")


if __name__ == "__main__":
    unittest.main()
