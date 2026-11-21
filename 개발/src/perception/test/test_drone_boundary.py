"""드론 화면 경계 검출 (perception/drone_boundary.py) 시험.

시험 그림 (test/data, 연습 경기장 2026-11-18):
  drone_test1.jpg          test1 카메라 · 가운데에 체커판 2장
  drone_test2.jpg          test2 카메라 · 평소 화면
  drone_test2_checker.jpg  test2 카메라 · 가운데에 체커판 2장
  drone_test2_a4.jpg       test2 카메라 · A4 용지 13장을 경기장 곳곳에 (축척 정답지 — 297 × 210 mm)
본선 카메라 (2026-11-19 공식 테스트 경기의 첫 사진 — 사진은 넣지 않고 **흰 선 픽셀만** 남긴 흑백 그림, 경기장 벽 밖은 지움. lines_picture 가 읽는다):
  drone_main_lines.png        18:06 판의 첫 사진. RED 쪽 끝 띠 선(화면 u ≈ 952)과 그 20 px 바깥의 거점 테두리 선(u ≈ 972)이 둘 다 봉우리로 뽑힌다
  drone_main_lines_faint.png  16:55 판의 셋째 사진. 끝 띠 선의 열 합이 문턱에 못 미쳐(29 < 30) 봉우리로 뽑히지 않는다 — 거점 테두리 선만 뽑힌다
ROS 없이 numpy · OpenCV 만으로 돈다:  python3 -m unittest discover -s test   (약 30초)
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
    from perception import drone_boundary
    from perception.drone_boundary import Boundary, BoundaryParams, Lens, _distort, _undistort, detect_boundary, read_image

DATA = os.path.join(HERE, "data")
_cache = {}


def lines_picture(name):
    """흰 선 픽셀만 남긴 흑백 그림 → 경계 찾기에 넣을 BGR 그림 (어두운 바닥에 흰 선). 흰 선 고르기(line_mask)가 같은 픽셀을 다시 낸다."""
    lines = cv2.imdecode(np.fromfile(os.path.join(DATA, name), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert lines is not None, name
    img = np.full(lines.shape + (3,), 70, np.uint8)
    img[lines > 0] = 255
    return img


def boundary(name):
    if name not in _cache:
        img = lines_picture(name) if name.endswith(".png") else read_image(os.path.join(DATA, name))
        assert img is not None, name
        _cache[name] = (img, detect_boundary(img))
    return _cache[name]


# 2026-11-18 에 확인한 모서리 (그림으로 눈으로 확인 · A4 로 축척 검증). 화면 px, (0,0) · (W,0) · (W,H) · (0,H)
EXPECTED_CORNERS = {
    "drone_test2.jpg": [(287, 716), (1008, 750), (1024, -15), (294, -18)],
    "drone_test2_a4.jpg": [(287, 716), (1007, 750), (1022, -15), (295, -18)],
    "drone_test2_checker.jpg": [(287, 716), (1008, 750), (1023, -14), (294, -18)],
    "drone_test1.jpg": [(279, 723), (984, 737), (1005, -39), (272, -42)],
}
CORNER_TOL_PX = 15  # test1 은 화면마다 확장 렌즈값이 흔들려 모서리가 10 px 남짓 움직인다


def a4_sheets(img, b):
    """화면의 흰 A4 용지 → [(윤곽 px, 실제 x 길이 m, 실제 y 길이 m 또는 None)]. 흰 중앙선에 걸친 용지는 y 를 쓰지 않는다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cand = cv2.morphologyEx(((hsv[..., 2] > 225) & (hsv[..., 1] < 40)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _lab, st, cen = cv2.connectedComponentsWithStats(cand)
    out = []
    up = 8
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if not (12 <= w <= 70 and 12 <= h <= 70 and a > 0.6 * w * h):
            continue
        fx, fy = b.pixel_to_field([cen[k]])[0]
        if not (-0.1 < fx < b.field_w + 0.1 and -0.3 < fy < b.field_h + 0.3):
            continue
        x0, y0 = max(0, x - 8), max(0, y - 8)
        win = gray[y0:y + h + 8, x0:x + w + 8]
        big = cv2.resize(win, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        ring = np.concatenate([win[:3].ravel(), win[-3:].ravel(), win[:, :3].ravel(), win[:, -3:].ravel()])
        m = (big > (np.median(ring) + np.percentile(win, 90)) / 2).astype(np.uint8)
        _nn, ll, ss, _ = cv2.connectedComponentsWithStats(m)
        cnt, _ = cv2.findContours((ll == 1 + np.argmax(ss[1:, 4])).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = cnt[0].reshape(-1, 2) / up + [x0, y0] - 0.5 + 0.5 / up
        f = b.pixel_to_field(pts)
        land = np.ptp(f[:, 0]) > np.ptp(f[:, 1])
        tx, ty = (0.297, 0.210) if land else (0.210, 0.297)
        out.append((pts, tx, None if abs(fx - b.field_w / 2) < 0.15 else ty))
    return out


# 경기장 바닥 격자점 (끝 띠 안쪽 · 0.2 m 간격) — 두 경계가 같은 바닥 점을 얼마나 다르게 보는지 잴 때 쓴다
GRID = np.stack(np.meshgrid(np.arange(0.9, 5.21, 0.2), np.arange(0.4, 4.41, 0.2)), -1).reshape(-1, 2)


def cut_sides(img, frac):
    """화면 좌우를 폭의 frac 씩 잘라낸다 (평가 경기장처럼 끝 띠 선이 화면 밖으로) → (잘린 화면, 왼쪽에서 잘린 px)."""
    cut = int(round(frac * img.shape[1]))
    return np.ascontiguousarray(img[:, cut:img.shape[1] - cut]), cut


def frame_error_cm(whole, part, cut, size):
    """원본 화면의 경계 whole 과 잘린 화면의 경계 part 가 같은 바닥 점을 얼마나 다르게 보는가 → (중앙값, 최댓값) cm. 잘린 화면에 보이는 격자점만."""
    uv = whole.field_to_pixel(GRID) - [cut, 0]
    inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
    d = np.hypot(*(part.pixel_to_field(uv[inside]) - GRID[inside]).T) * 100.0
    return float(np.median(d)), float(d.max())


@unittest.skipIf(cv2 is None, "OpenCV 없음")
class DroneBoundaryTest(unittest.TestCase):
    def test_boundary_found_and_corners(self):
        for name, expected in sorted(EXPECTED_CORNERS.items()):
            with self.subTest(name=name):
                _img, b = boundary(name)
                self.assertTrue(b.ok, b.notes)
                got = b.corners_px()
                err = np.abs(got - np.array(expected, float)).max()
                self.assertLessEqual(err, CORNER_TOL_PX, got.round(1).tolist())
                self.assertLess(b.rms_m, 0.02)

    def test_sector_lines_are_the_real_ones(self):
        """끝 띠 선을 거점 테두리로 착각하지 않는다 (끝 띠 0.3 m 근처가 나오면 착각 — 2026-11-18 에 있었다)."""
        for name in EXPECTED_CORNERS:
            with self.subTest(name=name):
                _img, b = boundary(name)
                self.assertTrue(0.75 <= b.strip <= 1.0, b.strip)
                self.assertTrue(0.55 <= b.cm_half <= 0.7, b.cm_half)
                self.assertLess(abs(b.circle_r - b.cm_half), 0.05)  # 원이 CM 선에 거의 닿는다

    def test_cameras_agree_on_floor_drawing(self):
        """두 카메라가 따로 잰 바닥 그림 치수가 맞아야 한다 (같은 경기장)."""
        _i1, b1 = boundary("drone_test1.jpg")
        _i2, b2 = boundary("drone_test2.jpg")
        self.assertLess(abs(b1.strip - b2.strip), 0.06)
        self.assertLess(abs(b1.cm_half - b2.cm_half), 0.05)
        self.assertLess(abs(b1.circle_r - b2.circle_r), 0.05)

    def test_scale_matches_a4_sheets(self):
        """경기장 곳곳의 A4 용지 크기로 축척 검사. 단순 렌즈 모델(k1 하나)은 가로 σ 10 % — 확장 모델은 5 % 안이어야 한다."""
        img, b = boundary("drone_test2_a4.jpg")
        sheets = a4_sheets(img, b)
        self.assertGreaterEqual(len(sheets), 10)
        rx, ry = [], []
        for pts, tx, ty in sheets:
            f = b.pixel_to_field(pts)
            rx.append(np.ptp(f[:, 0]) / tx)
            if ty is not None:
                ry.append(np.ptp(f[:, 1]) / ty)
        rx, ry = np.array(rx), np.array(ry)
        self.assertTrue(np.std(rx) < 0.05 and np.abs(rx - 1).max() < 0.12, rx.round(3).tolist())
        self.assertTrue(np.std(ry) < 0.05 and np.abs(ry - 1).max() < 0.18, ry.round(3).tolist())

    def test_lens_roundtrip(self):
        lens = Lens(-0.5, 0.4, 0.52, 622.0, 377.0, 640.0 ** 2)
        pts = np.array([(300.0, 20.0), (640.0, 360.0), (1000.0, 700.0), (500.0, 600.0)])
        self.assertLess(np.abs(_distort(_undistort(pts, lens), lens) - pts).max(), 1e-3)

    def test_simple_lens_roundtrip_closed_form(self):
        """k2 = 0 이면 _distort 가 닫힌 식을 쓴다 — 되돌리기가 맞고, 접히는 바깥은 NaN."""
        lens = Lens.simple(-0.5, (1280, 720))
        pts = np.array([(5.0, 5.0), (640.0, 360.0), (1275.0, 715.0), (100.0, 600.0)])
        self.assertLess(np.abs(_distort(_undistort(pts, lens), lens) - pts).max(), 1e-6)
        fold = Lens.simple(0.5, (1280, 720))
        self.assertTrue(np.isnan(_distort(np.array([(640.0 + 2000.0, 360.0)]), fold)).all())

    def test_prior_gives_same_answer_fast(self):
        """지난 경계(to_dict 로 저장한 것)에서 출발하면 같은 화면에서 같은 답을, 처음 찾기보다 빨리."""
        import time

        img, b = boundary("drone_test2.jpg")
        saved = b.to_dict()
        self.assertIsNotNone(Boundary.from_dict(saved))
        began = time.perf_counter()
        b2 = detect_boundary(img, prior=saved)
        took = time.perf_counter() - began
        self.assertTrue(b2.ok, b2.notes)
        self.assertEqual(b2.debug["via"], "prior")
        self.assertLess(np.abs(np.asarray(b2.corners_px()) - np.asarray(b.corners_px())).max(), 1e-6)
        self.assertAlmostEqual(b2.strip, b.strip, places=9)
        self.assertLess(took, 3.0)
        # 같은 카메라의 다른 화면 (체커판이 놓인 것)
        img3, b3 = boundary("drone_test2_checker.jpg")
        b4 = detect_boundary(img3, prior=saved)
        self.assertTrue(b4.ok, b4.notes)
        self.assertEqual(b4.debug["via"], "prior")
        self.assertLess(np.abs(np.asarray(b4.corners_px()) - np.asarray(b3.corners_px())).max(), CORNER_TOL_PX)

    def test_prior_from_other_camera_falls_back_to_search(self):
        img1, b1 = boundary("drone_test1.jpg")
        _img2, b2 = boundary("drone_test2.jpg")
        b = detect_boundary(img1, prior=[b2.to_dict()])
        self.assertTrue(b.ok, b.notes)
        self.assertEqual(b.debug["via"], "search")
        self.assertTrue(any("지난 경계" in n for n in b.notes), b.notes)
        self.assertLess(np.abs(np.asarray(b.corners_px()) - np.asarray(b1.corners_px())).max(), 1e-6)

    def test_floor_drawing_ratios_are_checked(self):
        """맞게 찾은 경계는 바닥 그림의 비율을 지킨다 (원 / CM 반폭 ≈ 1.02, (가운데 ~ 끝 띠 선) / CM 반폭 ≈ 3.6) — 검사에 적혀 있어야 한다."""
        p = BoundaryParams()
        for name in EXPECTED_CORNERS:
            with self.subTest(name=name):
                _img, b = boundary(name)
                self.assertTrue(p.circle_cm_range[0] <= b.circle_r / b.cm_half <= p.circle_cm_range[1])
                self.assertTrue(p.span_cm_range[0] <= (b.field_w / 2 - b.strip) / b.cm_half <= p.span_cm_range[1])
                self.assertTrue(any("원 반지름 / CM 반폭" in n for n in b.notes) and any("끝 띠 선) / CM 반폭" in n for n in b.notes), b.notes)
        # 2026-11-19 에 통과해 버린 틀린 경계 둘의 치수는 범위 밖이어야 한다
        self.assertFalse(p.c_range[0] <= 1.05 <= p.c_range[1])  # 바닥 깔개의 이음매(가운데에서 ±1.04 m)를 CM 선으로 고른 것
        self.assertFalse(p.circle_cm_range[0] <= 0.688 / 0.599 <= p.circle_cm_range[1])  # 상자 모서리를 줄 경계로 고른 것 (세로 축척이 틀려 원이 안 맞음)
        self.assertFalse(p.inner_ratio[0] <= 0.49 <= p.inner_ratio[1])  # 이음매 조합의 화면 픽셀 비율

    def test_wrong_lines_are_not_accepted_on_local_pictures(self):
        """그날 틀린 선을 고르고도 통과했던 사진들 (이 PC 의 연습경기 기록 폴더에 있을 때만 — 사진은 저장소에 넣지 않는다)."""
        root = os.environ.get("MAICON_SESSIONS", os.path.join(os.path.expanduser("~"), ".config", "maicon-match", "sessions"))
        names = ("20261118_220919/server_bridge/drone/0001.jpg",  # 고치기 전: CM 반폭 1.05 m (이음매) 로 통과
                 "20261118_222257/server_bridge/drone/0051.jpg",  # 끝 띠 1.02 · CM 반폭 0.85 로 통과
                 "20261119_114701/server_bridge/drone/0001.jpg")  # 끝 띠 0.88 · CM 반폭 0.82 로 통과
        seen = 0
        for name in names:
            img = read_image(os.path.join(root, *name.split("/")))
            if img is None:
                continue
            seen += 1
            with self.subTest(name=name):
                b = detect_boundary(img)
                self.assertTrue(b.ok, b.notes)
                self.assertTrue(0.75 <= b.strip <= 0.96 and 0.56 <= b.cm_half <= 0.67, (b.strip, b.cm_half))
                self.assertLess(abs(b.circle_r / b.cm_half - 1.02), 0.05)
        if not seen:
            self.skipTest("그날의 드론 사진이 이 PC 에 없음")

    def test_prior_index_tells_which_prior_matched(self):
        img1, b1 = boundary("drone_test1.jpg")
        _img2, b2 = boundary("drone_test2.jpg")
        b = detect_boundary(img1, prior=[b2.to_dict(), b1.to_dict()])
        self.assertEqual((b.debug["via"], b.debug["prior_index"]), ("prior", 1))

    def test_reduced_fit_when_strip_lines_are_cut_off(self):
        """좌우가 잘려 끝 띠 선이 화면 밖이어도 (CM · 중앙 · CM) 세 선 + 가로선 + 원으로 맞춘다. 원본 화면의 경계와 몇 cm 안에서 같아야 한다."""
        # (그림, 좌우로 자를 비율, 오차 중앙값 한계 cm, 최대 오차 한계 cm). test1 은 세로선 5개로 맞춘 원본 경계부터가 화면을 조금만 바꿔도
        # 귀퉁이에서 5~9 cm 움직인다 (확장 렌즈값이 흔들린다 — CORNER_TOL_PX 설명) → 한계를 넓게 둔다. 2026-11-19 잰 값: 0.3/2.2 · 0.4/1.7 · 2.2/7.2
        for name, frac, med_max, worst_max in (("drone_test2.jpg", 0.33, 3.0, 8.0), ("drone_test2.jpg", 0.30, 3.0, 8.0), ("drone_test1.jpg", 0.30, 4.0, 12.0)):
            with self.subTest(name=name, frac=frac):
                img, whole = boundary(name)
                part_img, cut = cut_sides(img, frac)
                b = detect_boundary(part_img)
                self.assertTrue(b.ok, b.notes)
                self.assertTrue(b.debug.get("reduced"), b.debug.get("via"))
                self.assertEqual((b.debug["lines"]["Bs"], b.debug["lines"]["Rs"]), (0, 0))
                self.assertTrue(all(b.debug["lines"][nm] >= 20 for nm in ("Bc", "C", "Rc", "top", "bot", "circle")), b.debug["lines"])
                self.assertTrue(b.debug["strip_derived"])
                self.assertTrue(any("끝 띠 선이 화면에 없어" in n for n in b.notes), b.notes)
                self.assertTrue(0.80 <= b.strip <= 0.90, b.strip)
                self.assertTrue(0.58 <= b.cm_half <= 0.655, b.cm_half)
                med, worst = frame_error_cm(whole, b, cut, (part_img.shape[1], part_img.shape[0]))
                self.assertLess(med, med_max)
                self.assertLess(worst, worst_max)

    def test_reduced_fit_fails_rather_than_guess(self):
        """근거가 모자라면 축소 맞추기는 실패해야 한다 (틀린 경계를 ok 로 내지 않는다)."""
        img, _b = boundary("drone_test2.jpg")
        cases = {
            "CM 선 하나만 남긴 화면": np.ascontiguousarray(img[:, 610:1100]),  # 중앙선 · RED 쪽 CM 선 · RED 쪽 끝 띠 선만
            "가운데 원이 잘린 화면": np.ascontiguousarray(img[:300, 380:900]),
            "잡음": np.random.RandomState(0).randint(0, 255, (720, 1280, 3)).astype(np.uint8),
        }
        lines = np.zeros((720, 1280, 3), np.uint8) + 40  # 세로선 3개 · 가로선 2개는 있지만 원이 없는 그림
        for u in (557, 640, 723):
            cv2.line(lines, (u, 0), (u, 719), (255, 255, 255), 3)
        for v in (215, 505):
            cv2.line(lines, (0, v), (1279, v), (255, 255, 255), 3)
        cases["원이 없는 선 그림"] = lines
        for label, pic in cases.items():
            with self.subTest(case=label):
                b = detect_boundary(pic)
                self.assertFalse(b.ok, (label, b.notes))
                self.assertTrue(b.notes)

    def test_reduced_fit_does_not_touch_normal_result(self):
        """세로선 5개로 맞는 화면은 축소 맞추기를 켜든 끄든 결과가 똑같다 (축소 맞추기는 실패했을 때만 돈다). 선마다 쓴 점 수가 debug 에 남는다."""
        from unittest import mock

        for name in ("drone_test1.jpg", "drone_test2.jpg"):
            with self.subTest(name=name):
                img, b = boundary(name)
                self.assertTrue(b.ok, b.notes)
                self.assertNotIn("reduced", b.debug)
                self.assertTrue(all(b.debug["lines"][nm] >= 20 for nm in ("Bs", "Bc", "C", "Rc", "Rs", "top", "bot", "circle")), b.debug["lines"])
                with mock.patch.object(drone_boundary.REDUCED, "enabled", False):
                    off = detect_boundary(img)
                self.assertTrue(np.array_equal(np.asarray(off.corners_px()), np.asarray(b.corners_px())))
                self.assertEqual((off.strip, off.cm_half, off.circle_r, off.rms_m, off.points), (b.strip, b.cm_half, b.circle_r, b.rms_m, b.points))

    def test_picture_shifted_sideways(self):
        """화면이 한쪽으로 밀려(잘려) 렌즈 가운데가 화면 가운데에서 멀어도: 20 % 까지는 렌즈 가운데를 따라가 맞추고, 더 멀면 실패한다.

        2026-11-19 까지는 20 % 민 화면에서 확장 렌즈 모델을 버리고 단순 모델(가운데 = 화면 가운데)로 통과해 귀퉁이가 14 ~ 17 cm 틀렸다.
        """
        img, b0 = boundary("drone_test2.jpg")
        hh, ww = img.shape[:2]
        gx, gy = np.meshgrid(np.arange(0.9, 5.21, 0.4), np.arange(0.4, 4.41, 0.4))
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        for frac, must_fit in ((0.20, True), (-0.20, True), (0.32, False)):
            with self.subTest(frac=frac):
                dx = int(round(frac * ww))
                moved = cv2.warpAffine(img, np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]]), (ww, hh))
                b = detect_boundary(moved)
                if not must_fit:
                    self.assertFalse(b.ok, b.notes)
                    continue
                self.assertTrue(b.ok, b.notes)
                uv = b0.field_to_pixel(grid) + [dx, 0]
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < ww)
                err = np.hypot(*(b.pixel_to_field(uv[inside]) - grid[inside]).T)
                self.assertLess(err.max(), 0.04, err.max())

    def test_main_camera_takes_the_sector_line_not_the_base_outline(self):
        """본선 카메라: RED 쪽 끝 띠 선 바로 바깥에 거점 세 곳의 흰 테두리가 세로로 한 줄에 놓여 있다 (끝 띠 선 u ≈ 952, 테두리 u ≈ 972).
        2026-11-19 까지는 테두리 선을 고른 조합이 어림 점수가 높아 먼저 맞춰졌고, 확장 렌즈 모델이 깨져도 단순 모델로 검사를 통과해 그대로 나갔다
        (여섯 판 첫 사진 24장 중 16장) — 경기장 오른쪽 1/3 이 0.15 ~ 0.3 m 틀린다. 끝 띠 선은 테두리 선보다 훨씬 길다 (화면 위에서 아래까지)."""
        for name in ("drone_main_lines.png", "drone_main_lines_faint.png"):
            with self.subTest(name=name):
                img, b = boundary(name)
                self.assertTrue(b.ok, b.notes)
                self.assertLessEqual(abs(b.debug["combo"][0] - 349), 4, b.debug["combo"])
                self.assertLessEqual(abs(b.debug["combo"][4] - 952), 4, b.debug["combo"])
                self.assertTrue(b.debug["lens_full"], b.notes[-1])
                self.assertTrue(-0.60 <= b.lens.k1 <= -0.25 and 0.40 <= b.lens.aspect <= 0.65, b.notes[-1])  # 연습 카메라와 같은 기종의 값
                self.assertLessEqual(len(b.debug["tried"]), 2, b.debug["tried"])  # 긴 선으로 바꾼 조합을 먼저 맞춘다 — 시간이 늘지 않는다
                # 끄면 2026-11-19 의 동작: 거점 테두리 선 + 단순 렌즈, 검사는 통과
                old = detect_boundary(img, BoundaryParams(prefer_full_lens=False))
                self.assertTrue(old.ok)
                self.assertGreaterEqual(old.debug["combo"][4], 968, old.debug["combo"])
                self.assertEqual(old.lens.aspect, 1.0)
                # 두 결과는 오른쪽에서 크게 다르고 가운데 · 왼쪽은 같다
                px = b.field_to_pixel([(5.3, 0.8), (5.3, 4.0), (3.06, 2.4), (0.8, 2.4)])
                gap = np.hypot(*(old.pixel_to_field(px) - b.pixel_to_field(px)).T)
                self.assertGreater(min(gap[:2]), 0.12, gap)
                self.assertLess(max(gap[2:]), 0.04, gap)

    def test_main_camera_pictures_of_two_matches_agree(self):
        """다른 판의 사진 두 장이 같은 화면 픽셀을 같은 자리로 옮긴다 (카메라는 천장에 붙어 있다). 고치기 전에는 사진에 따라 오른쪽이 0.3 m 달랐다."""
        _a, a = boundary("drone_main_lines.png")
        _c, c = boundary("drone_main_lines_faint.png")
        gx, gy = np.meshgrid(np.linspace(0.3, 5.8, 12), np.linspace(0.5, 4.5, 9))
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        err = np.hypot(*(c.pixel_to_field(a.field_to_pixel(grid)) - grid).T)
        self.assertLess(err.max(), 0.06, err.max())

    def test_faint_sector_line_is_found_below_the_peak_threshold(self):
        """진짜 끝 띠 선이 봉우리로 뽑히지 않은 사진: 가는 선이 조금 휘어 열 합이 문턱(30)에 못 미친다 (본선 사진의 7 %).
        거점 테두리 선으로 맞춘 조합의 끝 띠 선 곁을 낮춘 문턱으로 다시 보고, 훨씬 긴 그 선으로 바꿔 맞춘다."""
        img, b = boundary("drone_main_lines_faint.png")
        p = BoundaryParams()
        mask = drone_boundary.line_mask(img, p)
        normal = [u for u, _s in drone_boundary._vertical_peaks(mask, p)]
        loose = [u for u, _s in drone_boundary._vertical_peaks(mask, p, p.peak_min * p.loose_peak_ratio)]
        self.assertFalse(any(abs(u - 952) <= 4 for u in normal), normal)
        self.assertTrue(any(abs(u - 952) <= 4 for u in loose), loose)
        self.assertLessEqual(abs(b.debug["combo"][4] - 952), 4)
        long_line = len(drone_boundary._track_vertical(mask, b.debug["combo"][4], (0, img.shape[0]), p))
        outline = len(drone_boundary._track_vertical(mask, next(u for u in normal if abs(u - 971) <= 4), (0, img.shape[0]), p))
        self.assertGreater(long_line, p.longer_line_ratio * outline, (long_line, outline))
        # 낮춘 문턱을 쓰지 않으면(1.0) 그 선을 못 찾아 전처럼 거점 테두리 선으로 나간다 — 그래도 실패하지는 않는다
        without = detect_boundary(img, BoundaryParams(loose_peak_ratio=1.0))
        self.assertTrue(without.ok)
        self.assertGreaterEqual(without.debug["combo"][4], 968)

    def test_stale_prior_with_the_base_outline_is_not_repeated(self):
        """로봇의 저장 파일에 남아 있을 수 있는 옛 경계(거점 테두리 선 · 단순 렌즈)를 지난 경계로 줘도 그 선을 되풀이하지 않는다:
        지난 경계의 선 조합으로 맞춘 것이 확장 렌즈 모델을 못 맞추면 처음부터 찾아 본다. 맞게 찾은 경계를 지난 경계로 주면 전처럼 그것으로 맞춘다."""
        for name in ("drone_main_lines.png", "drone_main_lines_faint.png"):
            with self.subTest(name=name):
                img, good = boundary(name)
                stale = detect_boundary(img, BoundaryParams(prefer_full_lens=False))
                self.assertGreaterEqual(stale.debug["combo"][4], 968)
                b = detect_boundary(img, prior=stale.to_dict())
                self.assertTrue(b.ok, b.notes)
                self.assertEqual(b.debug["via"], "search")
                self.assertLessEqual(abs(b.debug["combo"][4] - 952), 4, b.debug["combo"])
                self.assertTrue(b.debug["lens_full"])
                again = detect_boundary(img, prior=good.to_dict())
                self.assertTrue(again.ok, again.notes)
                self.assertIn(again.debug["via"], ("prior", "prior_refit"))
                px = good.field_to_pixel([(5.3, 0.8), (5.3, 4.0), (3.06, 2.4), (0.8, 2.4)])
                # 선 점을 다르게 모아 다시 맞추므로 확장 렌즈값이 조금 달라진다 — 오른쪽 귀퉁이가 5 cm 남짓 (거점 테두리 선으로 맞춘 것과의 차이는 15 ~ 30 cm)
                self.assertLess(np.hypot(*(again.pixel_to_field(px) - good.pixel_to_field(px)).T).max(), 0.08)

    def test_five_line_rule_is_shared(self):
        """끝 띠 선을 바꿔 만든 조합도 처음 조합과 같은 조건(순서 · 좌우 대칭 · 안팎 비)을 지켜야 한다."""
        p = BoundaryParams()
        self.assertTrue(drone_boundary._five_ok((349, 570, 655, 744, 952), 1280, p))
        self.assertTrue(drone_boundary._five_ok((349, 570, 655, 744, 972), 1280, p))
        self.assertFalse(drone_boundary._five_ok((349, 570, 655, 744, 1081), 1280, p))  # 벽 — 좌우 대칭이 아니다
        self.assertFalse(drone_boundary._five_ok((600, 570, 655, 744, 952), 1280, p))  # 순서가 틀리다

    def test_unicode_path_and_bad_input(self):
        self.assertIsNone(read_image(os.path.join(DATA, "없는_파일.jpg")))
        b = detect_boundary(np.zeros((720, 1280, 3), np.uint8))
        self.assertFalse(b.ok)
        self.assertTrue(b.notes)


if __name__ == "__main__":
    unittest.main()
