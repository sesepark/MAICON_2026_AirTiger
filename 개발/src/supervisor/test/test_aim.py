"""사진으로 조준하기 (supervisor/aim.py 와 mission 의 지정표적 타격) 시험.

쓰는 숫자
  레이저 점        2026-11-19 레이저 캘리브레이션의 측정 기록 16개
  팬틸트 ↔ 픽셀    2026-11-18 측정 (로그_파일/1118_카메라_팬틸트_측정/measurements.json — 면 앞 1.07 m 에서 팬틸트를 돌리며 잰 수광부 불빛의 픽셀)
  판독 답          실차 기록의 classify 줄 (2026-11-18 23:39 · 2026-11-19 10:35 판) 과, 그 사진에서 잰 수광 돔 가운데의 픽셀 (±2픽셀)
가상 경기(SimMatch)는 실제 로봇의 성능을 뜻하지 않는다 — 흐름과 시한을 본다.
"""
import math
import unittest

from sim_world import SimMatch
from supervisor import arena as A
from supervisor.aim import AimLoop, AimParams, Receiver, find_receiver, laser_pixel, parse_sensors, pixel_error, surface_z

# (거리 m, u, v) — 측정 기록 16개 (records_used_16.json)
LASER_RECORDS = [
    (0.505, 334.51, 221.85), (0.505, 334.40, 221.82), (0.600, 334.10, 226.07), (0.600, 333.82, 226.02), (0.720, 334.09, 230.01), (0.720, 333.98, 229.67),
    (0.820, 333.70, 232.41), (0.820, 333.71, 232.46), (0.385, 334.90, 214.22), (0.385, 334.88, 213.89), (0.615, 333.84, 226.91), (0.615, 333.93, 227.32),
    (0.615, 333.95, 227.52), (0.615, 333.81, 227.18), (0.615, 333.61, 226.81), (0.615, 333.50, 226.76),
]
# 결과 표 (거리 → 레이저 점)
LASER_TABLE = [(0.4, 334.8, 215.2), (0.5, 334.3, 221.9), (0.6, 334.0, 226.3), (0.8, 333.6, 231.9), (1.0, 333.4, 235.2), (2.0, 333.0, 241.9)]

# 팬틸트를 돌렸을 때 수광부 불빛이 찍힌 픽셀: (단위, x, y). 팬 1단위 0.5°, 틸트 1단위 0.298°
PAN_SWEEP = [(-80, 568.9, 209.5), (-60, 519.2, 209.7), (-40, 469.4, 209.9), (-20, 418.7, 210.3), (0, 367.7, 210.7), (20, 316.7, 209.9), (40, 267.4, 209.3), (60, 218.1, 208.1), (80, 167.5, 207.2)]
TILT_SWEEP = [(-60, 368.0, 119.8), (-40, 367.7, 149.1), (-20, 367.0, 180.2), (0, 366.5, 210.0), (20, 365.5, 239.4), (40, 365.7, 269.6), (60, 364.7, 299.4)]
SWEEP_LENS_M = 1.07  # 그때 렌즈에서 면까지


def _interp(table, unit, col):
    unit = max(table[0][0], min(table[-1][0], unit))
    for a, b in zip(table, table[1:]):
        if a[0] <= unit <= b[0]:
            return a[col] + (b[col] - a[col]) * (unit - a[0]) / (b[0] - a[0])
    return table[-1][col]


class MeasuredHead:
    """잰 값대로 움직이는 팬틸트 머리: 보낸 각도(도)에 수광부가 사진 어디에 찍히는지.

    pan_unit_deg · tilt_unit_deg = 설정이 「1단위가 몇 도」 라고 알고 있는 값 (node.py 가 각도를 단위로 바꿀 때 쓴다). 실제 머리는 잰 표대로 돈다.
    """

    def __init__(self, pan_unit_deg=0.5, tilt_unit_deg=0.3, pan_flip=1.0, tilt_flip=1.0):
        self.unit = (pan_unit_deg, tilt_unit_deg)
        self.flip = (pan_flip, tilt_flip)

    def receiver(self, pan_deg, tilt_deg):
        pu = self.flip[0] * pan_deg / self.unit[0]
        tu = self.flip[1] * tilt_deg / self.unit[1]
        x = _interp(PAN_SWEEP, pu, 1) + _interp(TILT_SWEEP, tu, 1) - 366.5
        y = _interp(TILT_SWEEP, tu, 2) + _interp(PAN_SWEEP, pu, 2) - 210.7
        return Receiver(x, y, 13.0, "sensor", 0.9)


def run_loop(head, params=None, deg_per_px=127.4 / 640.0):
    """사진 → 고침을 끝까지 돌린다. 돌려주는 값: (loop, 마지막 팬, 틸트, 그때 수광부와 레이저 점의 차이 픽셀, 찍은 사진 수)."""
    loop = AimLoop(params or AimParams(), deg_per_px)
    laser = laser_pixel(SWEEP_LENS_M - 0.048)
    pan = tilt = 0.0
    for photos in range(1, 12):
        act = loop.see(head.receiver(pan, tilt), laser, pan, tilt)
        if act[0] == "fire":
            break
        pan, tilt = act[1], act[2]
    left = pixel_error(head.receiver(pan, tilt), laser)
    return loop, pan, tilt, math.hypot(*left), photos


class LaserPixelTests(unittest.TestCase):
    def test_matches_the_calibration_table(self):
        for z, u, v in LASER_TABLE:
            got = laser_pixel(z)
            self.assertAlmostEqual(got[0], u, delta=0.11)
            self.assertAlmostEqual(got[1], v, delta=0.11)

    def test_fits_the_sixteen_records(self):
        sq = [(laser_pixel(z)[0] - u) ** 2 + (laser_pixel(z)[1] - v) ** 2 for z, u, v in LASER_RECORDS]
        self.assertLess(math.sqrt(sum(sq) / len(sq)), 0.45)  # 기록서의 값 0.36 픽셀
        self.assertLess(math.sqrt(max(sq)), 1.0)

    def test_far_point_is_not_the_image_centre(self):
        u, v = laser_pixel(1e6, AimParams(z_max_m=1e9))
        self.assertAlmostEqual(u, 332.5, delta=0.05)  # 가운데(320, 240)에서 오른쪽 12.5 · 아래 8.6 픽셀
        self.assertAlmostEqual(v, 248.6, delta=0.05)

    def test_constants_come_from_the_parameters(self):
        p = AimParams(laser_u0=300.0, laser_ku=2.0, laser_v0=200.0, laser_kv=-10.0)
        self.assertEqual(laser_pixel(0.5, p), (304.0, 180.0))

    def test_distance_is_clamped_and_measured_from_the_body_centre(self):
        self.assertAlmostEqual(surface_z(0.9), 0.9 - 0.05 - 0.048)  # 차체 가운데 → 렌즈(0.05 앞) → 식의 기준점(0.048 더 앞)
        self.assertEqual(surface_z(0.1), AimParams().z_min_m)
        self.assertEqual(laser_pixel(0.0), laser_pixel(AimParams().z_min_m))


class ParseSensorsTests(unittest.TestCase):
    def test_reads_the_text_perception_appends(self):
        got = parse_sensors("2 pictures in 56 ms | sensors=[[0.3125,0.2917,0.0312,0.0417,0.95],[0.8,0.5,0.02,0.03,0.4]]")
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0], {"x": 0.3125, "y": 0.2917, "w": 0.0312, "h": 0.0417, "conf": 0.95})

    def test_old_perception_and_broken_text(self):
        self.assertIsNone(parse_sensors("2 pictures in 56 ms"))  # 수광부를 싣지 않는 옛 perception
        self.assertEqual(parse_sensors("0 pictures in 50 ms | sensors=[]"), [])
        self.assertIsNone(parse_sensors("1 pictures in 50 ms | sensors=[[0.3,"))
        self.assertIsNone(parse_sensors(None))
        self.assertEqual(parse_sensors("x | sensors=[[0.3,0.2,0.01,0.01,0.5]] | more"), [{"x": 0.3, "y": 0.2, "w": 0.01, "h": 0.01, "conf": 0.5}])


def sensor(u, v, size=14.0, conf=0.8):
    return {"x": u / 640.0, "y": v / 480.0, "w": size / 640.0, "h": size / 480.0, "conf": conf}


# ---- 실차 기록의 판독 답 (events.jsonl 의 classify 줄 그대로)
# 2026-11-19 10:35 판 5번 사진: 가운데에 감시초소. 그 사진에서 잰 수광 돔 가운데 (329.5, 202.5)
P5 = [
    {"kind": "missile", "x": 0.031, "y": 0.508, "w": 0.062, "h": 0.217, "conf": 0.94, "cut": True},
    {"kind": "unknown", "x": 0.187, "y": 0.493, "w": 0.127, "h": 0.117, "conf": 0.28, "cut": False},
    {"kind": "non_object", "x": 0.363, "y": 0.488, "w": 0.08, "h": 0.111, "conf": 0.77, "cut": False},
    {"kind": "outpost", "x": 0.505, "y": 0.501, "w": 0.193, "h": 0.274, "conf": 0.95, "cut": False},
    {"kind": "drum", "x": 0.744, "y": 0.502, "w": 0.089, "h": 0.171, "conf": 0.92, "cut": False},
    {"kind": "military_truck", "x": 0.816, "y": 0.503, "w": 0.068, "h": 0.167, "conf": 0.94, "cut": False},
    {"kind": "unknown", "x": 0.864, "y": 0.503, "w": 0.046, "h": 0.141, "conf": 0.33, "cut": False},
]
P5_DOME = (329.5, 202.5)
# 같은 판 10번 사진: 감시초소 둘(왼쪽 것은 잘림)과 레이더, 앞에 다른 UGV(옆에 흰 돔). 잰 돔: 가운데 감시초소 (313, 217), 레이더 (553, 213), UGV 의 돔 (236, 293)
P10 = [
    {"kind": "outpost", "x": 0.029, "y": 0.504, "w": 0.047, "h": 0.194, "conf": 0.67, "cut": True},
    {"kind": "tank", "x": 0.076, "y": 0.494, "w": 0.057, "h": 0.096, "conf": 0.84, "cut": False},
    {"kind": "outpost", "x": 0.493, "y": 0.49, "w": 0.056, "h": 0.119, "conf": 0.83, "cut": False},
    {"kind": "non_object", "x": 0.548, "y": 0.489, "w": 0.061, "h": 0.119, "conf": 0.89, "cut": False},
    {"kind": "military_truck", "x": 0.73, "y": 0.503, "w": 0.128, "h": 0.189, "conf": 0.96, "cut": False},
    {"kind": "radar", "x": 0.863, "y": 0.51, "w": 0.146, "h": 0.214, "conf": 0.94, "cut": False},
]
P10_OUTPOST_DOME, P10_RADAR_DOME, P10_UGV_DOME = (313.0, 217.0), (553.0, 213.0), (236.0, 293.0)
# 같은 판 7번 사진: 감시초소가 아주 가까워 왼쪽이 잘렸다. 잰 돔 (68.5, 162)
P7 = [
    {"kind": "outpost", "x": 0.176, "y": 0.566, "w": 0.351, "h": 0.759, "conf": 0.93, "cut": True},
    {"kind": "drum", "x": 0.539, "y": 0.507, "w": 0.14, "h": 0.236, "conf": 0.94, "cut": False},
    {"kind": "military_truck", "x": 0.629, "y": 0.505, "w": 0.055, "h": 0.226, "conf": 0.91, "cut": False},
]
P7_DOME = (68.5, 162.0)
# 2026-11-18 23:39 판 1번 사진 (그때의 모델은 그림 상자를 돔 아래만 잡았다): 통신시설과 감시초소. 잰 돔: 통신시설 (131, 177), 감시초소 (323, 210)
OLD = [
    {"kind": "comms", "x": 0.182, "y": 0.574, "w": 0.276, "h": 0.349, "conf": 0.87, "cut": False},
    {"kind": "outpost", "x": 0.512, "y": 0.528, "w": 0.087, "h": 0.158, "conf": 0.82, "cut": False},
    {"kind": "unknown", "x": 0.759, "y": 0.509, "w": 0.042, "h": 0.068, "conf": 0.23, "cut": False},
    {"kind": "unknown", "x": 0.939, "y": 0.522, "w": 0.089, "h": 0.122, "conf": 0.43, "cut": False},
]
OLD_COMMS_DOME, OLD_OUTPOST_DOME = (131.0, 177.0), (323.0, 210.0)
AIMED = laser_pixel(0.8)  # 겨눈 점: 0.9 m 앞에서의 레이저 점 (333.6, 231.9)


class FindReceiverTests(unittest.TestCase):
    def test_takes_the_sensor_on_top_of_the_wanted_picture(self):
        """수광부 탐지가 표적 그림의 위쪽 가운데에 있으면 그것. 다른 표적 · UGV 의 돔은 고르지 않는다."""
        sensors = [sensor(*P10_UGV_DOME, size=30.0, conf=0.9), sensor(*P10_OUTPOST_DOME, size=8.0, conf=0.5), sensor(*P10_RADAR_DOME, size=9.0)]
        got = find_receiver(P10, sensors, "outpost", AIMED)
        self.assertEqual((got.how, round(got.u), round(got.v)), ("sensor", 313, 217))
        got = find_receiver(P10, sensors, "radar", (540.0, 240.0))  # 레이더 면을 겨누고 있을 때
        self.assertEqual((got.how, round(got.u), round(got.v)), ("sensor", 553, 213))

    def test_of_two_pictures_of_the_kind_takes_the_one_aimed_at(self):
        got = find_receiver(P10, [], "outpost", AIMED)  # 왼쪽 끝의 잘린 감시초소가 아니라 가운데 것
        self.assertEqual(got.how, "box")
        self.assertAlmostEqual(got.u, 0.493 * 640, delta=0.5)

    def test_without_a_sensor_uses_the_fixed_spot_of_the_picture_box(self):
        """수광부 탐지가 없으면 그림 상자의 위에서 20% · 가로 가운데. 실차 사진에서 잰 돔과 8픽셀 안 (0.9 m 에서 돔 반지름이 8픽셀쯤)."""
        for pictures, kind, near, dome in ((P5, "outpost", AIMED, P5_DOME), (P10, "outpost", AIMED, P10_OUTPOST_DOME), (P10, "radar", (540.0, 240.0), P10_RADAR_DOME)):
            got = find_receiver(pictures, None, kind, near)
            self.assertEqual(got.how, "box")
            self.assertLess(math.hypot(got.u - dome[0], got.v - dome[1]), 8.0, (kind, got, dome))
        self.assertIsNone(find_receiver(P5, None, "outpost", AIMED, AimParams(use_box=False)))

    def test_cut_picture_gives_no_fixed_spot_but_its_sensor_is_used(self):
        near = (80.0, 200.0)
        self.assertIsNone(find_receiver(P7, [], "outpost", near))  # 잘린 상자의 가운데는 면의 가운데가 아니다
        got = find_receiver(P7, [sensor(*P7_DOME, size=40.0)], "outpost", near)
        self.assertEqual((got.how, got.size), ("sensor", 40.0))

    def test_sensor_just_above_the_box_counts(self):
        """2026-11-18 의 모델처럼 그림 상자가 돔 아래만 잡아도, 바로 위의 수광부 탐지는 그 그림의 것으로 본다."""
        sensors = [sensor(*OLD_COMMS_DOME, size=27.0), sensor(*OLD_OUTPOST_DOME, size=15.0)]
        got = find_receiver(OLD, sensors, "outpost", AIMED)
        self.assertEqual((got.how, round(got.u), round(got.v)), ("sensor", 323, 210))
        got = find_receiver(OLD, sensors, "comms", (150.0, 240.0))
        self.assertEqual((got.how, round(got.u), round(got.v)), ("sensor", 131, 177))

    def test_low_confidence_and_far_sensors_are_ignored(self):
        self.assertEqual(find_receiver(P5, [sensor(*P5_DOME, conf=0.1)], "outpost", AIMED).how, "box")
        self.assertEqual(find_receiver(P5, [sensor(520.0, 205.0)], "outpost", AIMED).how, "box")  # 옆 그림 위의 탐지
        # 표적 그림이 답에 없을 때: 겨눈 점 가까이의 수광부만
        self.assertEqual(find_receiver([], [sensor(350.0, 215.0)], "outpost", AIMED).how, "near")
        self.assertIsNone(find_receiver([], [sensor(*P10_UGV_DOME)], "outpost", AIMED))
        self.assertIsNone(find_receiver([], [], "outpost", AIMED))
        self.assertIsNone(find_receiver(None, None, "outpost", AIMED))

    def test_picture_read_as_another_kind_still_serves_when_it_is_under_the_aim(self):
        """가까이서 종류를 모름으로 답해도, 겨눈 점이 들어 있는 그림이면 그 위쪽의 수광부를 쓴다."""
        pictures = [dict(P5[3], kind="unknown", conf=0.3)]
        self.assertEqual(find_receiver(pictures, [sensor(*P5_DOME)], "outpost", AIMED).how, "sensor")
        self.assertIsNone(find_receiver(pictures, [], "outpost", (100.0, 100.0)))  # 겨눈 점이 그 그림 밖
        self.assertEqual(find_receiver(pictures, [], "outpost", AIMED).how, "box")  # 모름으로 읽힌 그림은 상자의 자리도 쓴다
        other = [dict(P5[3], kind="tank")]  # 다른 종류로 분명히 읽혔다 → 수광부 탐지가 있을 때만 쓴다
        self.assertEqual(find_receiver(other, [sensor(*P5_DOME)], "outpost", AIMED).how, "sensor")
        self.assertIsNone(find_receiver(other, [], "outpost", AIMED))


class AimLoopTests(unittest.TestCase):
    def test_signs_follow_the_measured_head(self):
        """잰 표: 팬 + 면 불빛이 왼쪽으로, 틸트 + 면 아래로 간다 → 오른쪽 위의 수광부에는 팬 + · 틸트 + 를 보낸다."""
        loop = AimLoop(AimParams(), 127.4 / 640.0)
        act = loop.see(Receiver(367.7, 210.7, 13.0, "sensor"), (333.4, 235.5), 0.0, 0.0)
        self.assertEqual(act[0], "turn")
        self.assertAlmostEqual(act[1], 34.3 * 127.4 / 640.0, delta=0.05)  # 팬 +6.8°
        self.assertAlmostEqual(act[2], 24.8 * 127.4 / 640.0, delta=0.05)  # 틸트 +4.9°

    def test_one_correction_is_enough_with_the_measured_units(self):
        loop, pan, tilt, left, photos = run_loop(MeasuredHead())
        self.assertTrue(loop.converged)
        self.assertEqual((loop.corrections, photos), (1, 2))
        self.assertLess(left, 4.0)
        self.assertEqual(loop.relearned, [])

    def test_converges_when_the_gain_is_off_by_forty_percent(self):
        for factor in (0.6, 0.8, 1.25, 1.4):  # 팬틸트 1단위의 각도를 이만큼 틀리게 알고 있을 때
            loop, _pan, _tilt, left, _photos = run_loop(MeasuredHead(pan_unit_deg=0.5 * factor, tilt_unit_deg=0.3 * factor))
            self.assertTrue(loop.converged, factor)
            self.assertLessEqual(loop.corrections, 3, factor)
            self.assertLess(left, 4.0, factor)

    def test_converges_when_the_tilt_unit_is_half_or_double(self):
        """틸트 1단위는 자료마다 0.3° 와 0.15° 로 다르다. 어느 쪽으로 틀려도 고친 결과를 보고 다시 정한다."""
        for unit in (0.15, 0.6):
            loop, _pan, _tilt, left, _photos = run_loop(MeasuredHead(tilt_unit_deg=unit))
            self.assertTrue(loop.converged, unit)
            self.assertLessEqual(loop.corrections, 3, unit)
            self.assertLess(left, 4.0, unit)
            self.assertEqual(loop.relearned[0][0], "tilt")
        # 배우지 않으면 두 배로 넘겨 돌 때 영영 모이지 않는다
        loop, _pan, _tilt, left, _photos = run_loop(MeasuredHead(tilt_unit_deg=0.15), AimParams(learn=False))
        self.assertFalse(loop.converged)
        self.assertGreater(left, 15.0)

    def test_converges_when_a_direction_is_reversed(self):
        for flips in ((1.0, -1.0), (-1.0, 1.0)):
            loop, _pan, _tilt, left, _photos = run_loop(MeasuredHead(pan_flip=flips[0], tilt_flip=flips[1]))
            self.assertTrue(loop.converged, flips)
            self.assertLessEqual(loop.corrections, 3, flips)
            self.assertLess(left, 4.0, flips)
            self.assertLess(loop.k[0 if flips[0] < 0 else 1], 0.0)

    def test_stops_after_three_corrections(self):
        class Wild:  # 탐지 자리가 사진마다 크게 튄다 → 모이지 않는다
            n = 0

            def receiver(self, pan, tilt):
                Wild.n += 1
                return Receiver(300.0 + 37.0 * (Wild.n % 3), 200.0 + 23.0 * (Wild.n % 2), 13.0, "sensor")

        loop, _pan, _tilt, _left, photos = run_loop(Wild(), AimParams(learn=False))
        self.assertEqual((loop.corrections, photos), (3, 4))
        self.assertFalse(loop.converged)

    def test_misses_are_counted_and_cleared_by_a_sighting(self):
        loop = AimLoop()
        self.assertEqual(loop.see(None, AIMED, 0.0, 0.0), ("look",))
        self.assertEqual(loop.see(None, AIMED, 0.0, 0.0), ("look",))
        self.assertEqual(loop.misses, 2)
        self.assertEqual(loop.see(Receiver(AIMED[0] + 1.0, AIMED[1], 13.0, "sensor"), AIMED, 0.0, 0.0), ("fire",))
        self.assertEqual((loop.misses, loop.converged), (0, True))

    def test_stays_inside_the_pan_tilt_limits(self):
        loop = AimLoop(AimParams(), 0.2, pan_limit=50.0, tilt_limit=30.0)
        act = loop.see(Receiver(500.0, 100.0, 13.0, "sensor"), AIMED, 45.0, 28.0)
        self.assertEqual(act, ("turn", 50.0, 30.0))
        self.assertEqual(loop.see(Receiver(500.0, 100.0, 13.0, "sensor"), AIMED, 50.0, 30.0), ("fire",))  # 더 돌릴 수 없다
        self.assertFalse(loop.converged)

    def test_box_then_sensor_does_not_teach_a_wrong_gain(self):
        """앞 사진은 그림 상자의 자리, 뒤 사진은 수광부 탐지로 찾았으면 둘의 자리 차이를 팬틸트가 움직인 것으로 읽지 않는다."""
        loop = AimLoop(AimParams(), 0.2)
        loop.see(Receiver(AIMED[0] + 40.0, AIMED[1] - 30.0, 0.0, "box"), AIMED, 0.0, 0.0)
        loop.see(Receiver(AIMED[0] + 12.0, AIMED[1] - 9.0, 13.0, "sensor"), AIMED, 8.0, 6.0)
        self.assertEqual(loop.relearned, [])
        self.assertEqual(loop.k, [0.2, 0.2])


class StrikeBySightTests(unittest.TestCase):
    """가상 경기에서 지정표적을 쏜다. 가상 카메라 · 레이저는 sim_world.SimMatch 의 참 모형 (수광 돔 반지름 2.5 cm)."""

    HIT_M = 0.025

    @staticmethod
    def begin(m, legacy=True):
        """지도와 M1 해석이 온 뒤, 정찰을 기다리지 않고 표적 면을 알려 준다 (정찰은 이 시험의 대상이 아니다 — test_mission.py 가 본다)."""
        m.run(running_s=3.0, report_s=0.0)
        camp, kind = A.target_kind(m.order_truth["target"])
        m.mission.target_face = next(f for f in m.fmap.faces if m.truth[f.key] == kind and A.side_of(m.fmap, f.cx) == camp)
        if legacy:  # 이 시험들은 조준이 맞는지를 본다 → 발 수는 전처럼 늘 target_shots 번 (레이저 점 확인 없이). 새 발사 규칙은 test_strike_rules.py
            m.can_watch = False
            m.mission.p.unconfirmed_shots = m.mission.p.target_shots
        return m

    def strike(self, **world):
        aim, mission = world.pop("aim", {}), world.pop("mission", {})
        m = SimMatch(mode="face", slot="B-1", **world)
        for key, value in aim.items():
            setattr(m.mission.p.aim, key, value)
        for key, value in mission.items():
            setattr(m.mission.p, key, value)
        return self.begin(m).run(running_s=100.0, report_s=0.0)

    @staticmethod
    def longest(m):
        """발사 한 번 앞에서 사진 조준에 쓰는 가장 긴 시간: 시한 + 그때 기다리던 답 하나 + 여유."""
        return m.mission.p.aim.budget_s + m.mission.p.aim.answer_wait_s + 1.0

    def aim_notes(self, m):
        return [text for _t, text in m.mission.log if "조준" in text or "팬틸트" in text]

    def test_every_shot_lands_on_the_receiver(self):
        m = self.strike()
        self.assertEqual(len(m.fires), m.mission.p.target_shots)
        self.assertTrue(all(off is not None and off < self.HIT_M for off in m.beam_off), m.beam_off)
        self.assertGreaterEqual(len(m.aim_photos), len(m.fires))  # 발마다 한 장은 확인한다

    def test_whole_match_finds_the_target_then_hits_it(self):
        """정찰로 표적 면을 찾은 뒤 쏘는 경기 전체 흐름에서도 맞는다."""
        m = SimMatch(mode="face", slot="B-1", can_watch=False, params={"unconfirmed_shots": 3}).run(running_s=220.0)
        self.assertEqual(len(m.fires), m.mission.p.target_shots, m.mission.log[-6:])
        self.assertTrue(all(off is not None and off < self.HIT_M for off in m.beam_off), m.beam_off)

    def test_old_geometry_misses_where_sight_hits(self):
        """전의 방식(사진 없이, 전의 틸트 식): 잰 값에 맞춘 이 가상 머리에서는 수광부 아래를 지나 빗나간다 — 수광부 높이를 전의 값 0.35 m 로 두면 약 3 cm,
        사진에서 잰 0.32 m 로 두면 약 5 cm (실제로 M2 를 못 맞히던 것과 같은 쪽). 사진으로 고치면 맞고, 수광부가 가정보다 6 cm 높거나 5 cm 낮아도 맞는다."""
        for height, low, high in ((0.35, 0.02, 0.045), (0.32, 0.04, 0.07)):
            blind = self.strike(aim={"enabled": False}, mission={"camera_height": 0.0, "target_receiver_height": height})
            self.assertEqual(len(blind.fires), blind.mission.p.target_shots)
            self.assertTrue(all(low < off < high for off in blind.beam_off), (height, blind.beam_off))
            self.assertEqual(blind.aim_photos, [])
        for height in (0.32, 0.38, 0.27):
            sighted = self.strike(receiver_h=height)
            self.assertEqual(len(sighted.fires), sighted.mission.p.target_shots)
            self.assertTrue(all(off < self.HIT_M for off in sighted.beam_off), (height, sighted.beam_off, self.aim_notes(sighted)))

    def test_computed_tilt_agrees_with_the_two_measurements(self):
        """계산한 틸트(사진을 보기 전의 각도): 2026-11-18 측정 자리(렌즈 앞 1.07 m, 틸트 0 에서 수광부가 y = 210)에서 수광부를 레이저 점 높이(y = 235.5)로
        내리려면 25.5픽셀 = 틸트 17단위(5.1°)를 올려야 한다. 전의 식은 1.5° 였다."""
        from supervisor.faces import CameraParams
        from supervisor.fieldmap import Face
        from supervisor.mission import Mission, MissionParams

        face = Face(0, "W", 2.0, 1.0, -1.0, 0.0, None)
        pose = (2.0 - 1.07 - 0.05, 1.0, 0.0)  # 렌즈가 면에서 1.07 m
        new = Mission(MissionParams(), cam=CameraParams()).aim_angles(pose, face)
        self.assertAlmostEqual(new[0], 0.0, delta=0.01)
        self.assertAlmostEqual(new[1], (235.5 - 210.3) * 127.4 / 640.0, delta=0.4)
        old = Mission(MissionParams(camera_height=0.0), cam=CameraParams()).aim_angles(pose, face)
        self.assertAlmostEqual(old[1], math.degrees(math.atan2(0.32 - 0.29, 1.12)), delta=0.01)

    def test_hits_with_a_poor_pose(self):
        """위치 값이 6 cm · 5° 씩 흔들려도 세 발을 다 쏘고 맞는다 (전에는 쏘는 도중 위치 값이 튀면 다시 가려다 남은 발을 버렸다)."""
        for seed in (1, 2, 3):
            m = self.strike(pose_xy=0.06, pose_yaw_deg=5.0, seed=seed)
            self.assertEqual(len(m.fires), m.mission.p.target_shots, (seed, m.mission.log[-6:]))
            self.assertTrue(all(off < self.HIT_M for off in m.beam_off), (seed, m.beam_off, self.aim_notes(m)))

    def test_hits_when_the_tilt_unit_or_direction_is_not_what_the_settings_say(self):
        for world in ({"tilt_scale": 0.5}, {"tilt_scale": 2.0}, {"tilt_scale": 1.4, "pan_scale": 0.7}, {"tilt_scale": -1.0}):
            m = self.strike(receiver_h=0.38, pose_yaw_deg=4.0, **world)
            self.assertEqual(len(m.fires), m.mission.p.target_shots, world)
            # 첫 발은 고칠 횟수(3번)를 다 쓰고도 덜 맞을 수 있다. 배운 값이 남으므로 그 뒤의 발은 맞아야 한다
            self.assertTrue(all(off < self.HIT_M for off in m.beam_off[1:]), (world, m.beam_off, self.aim_notes(m)))

    def test_falls_back_to_the_picture_box_when_no_sensor_is_detected(self):
        m = self.strike(sensor_rate=0.0, receiver_h=0.33)
        self.assertEqual(len(m.fires), m.mission.p.target_shots)
        self.assertTrue(any("수광부(그림 상자에서 어림)" in text for text in self.aim_notes(m)), self.aim_notes(m))
        self.assertTrue(all(off < self.HIT_M for off in m.beam_off), m.beam_off)

    def test_fires_by_geometry_when_the_photos_show_nothing(self):
        """조준 사진에서 아무것도 못 찾거나 판독이 실패해도 발사 수는 그대로다 (전의 방식으로 쏜다). 정해진 시간 안에."""
        for world in ({"aim_answers": False},):  # 판독은 되는데 표적이 안 보이는 경우는 쏘지 않고 물러난다 (test_strike_rules.py)
            m = self.strike(**world)
            self.assertEqual(len(m.fires), m.mission.p.target_shots, world)
            self.assertTrue(any("계산한 각도로 쏜다" in text for text in self.aim_notes(m)), self.aim_notes(m))
            # 계산한 각도도 잰 값에 맞춘 것이라 전의 식(약 5 cm 아래)보다는 가깝다. 위치 값의 오차만큼은 남는다 (돔의 반지름 2.5 cm 를 넘나든다)
            self.assertTrue(all(off < 0.04 for off in m.beam_off), m.beam_off)
            self.assertLess(m.fires[0][0] - m.aim_photos[0][0], self.longest(m))

    def test_fires_within_the_time_limit_when_perception_never_answers(self):
        m = self.begin(SimMatch(mode="face", slot="B-1"))
        asked = []
        m.ports.shoot_aim = lambda tag: asked.append((m.now, tag)) or True  # 사진은 보냈는데 답이 오지 않는다
        m.run(running_s=100.0, report_s=0.0)
        self.assertEqual(len(m.fires), m.mission.p.target_shots)
        self.assertLess(m.fires[0][0] - asked[0][0], self.longest(m))

    def test_fires_within_the_time_limit_when_no_photo_can_be_taken(self):
        m = self.begin(SimMatch(mode="face", slot="B-1"))
        tried = []
        m.ports.shoot_aim = lambda tag: tried.append(m.now) and False  # 카메라 사진이 없다
        m.run(running_s=100.0, report_s=0.0)
        self.assertEqual(len(m.fires), m.mission.p.target_shots)
        self.assertLess(m.fires[0][0] - tried[0], self.longest(m))

    def test_late_answer_for_an_old_photo_is_ignored(self):
        from supervisor.mission import Mission

        mission = Mission()
        mission._strike = {"spot": (0.0, 0.0), "since": 0.0, "aimed_at": 0.0, "bad": [], "pan": 0.0, "tilt": 0.0}
        a = mission._aim_state(mission._strike)
        a.update(state="wait", tag=("aim", 1, 5), laser=AIMED)

        class F:
            t = 1.0

        mission.designated = "RED_OUTPOST"
        mission._aim_answer(F(), ("aim", 1, 4), {"pictures": P5, "sensors": [sensor(*P5_DOME)]})  # 앞 사진의 답
        self.assertEqual(a["state"], "wait")
        mission._aim_answer(F(), ("aim", 1, 5), {"pictures": P5, "sensors": [sensor(*P5_DOME)]})
        self.assertEqual(a["state"], "turn")
        self.assertTrue(a["held"])


if __name__ == "__main__":
    for world in ({}, {"receiver_h": 0.38}, {"pose_xy": 0.06, "pose_yaw_deg": 5.0, "seed": 3}, {"tilt_scale": 0.5, "receiver_h": 0.38}, {"tilt_scale": -1.0, "receiver_h": 0.38},
                  {"sensor_rate": 0.0}, {"sensor_rate": 0.0, "aim_pictures": False}):
        m = StrikeBySightTests.begin(SimMatch(mode="face", slot="B-1", **world)).run(running_s=100.0, report_s=0.0)
        print(world or "기본", "→ 발사 %d, 수광부 가운데에서 벗어난 거리(cm) %s, 조준 사진 %d장" % (len(m.fires), [None if v is None else round(v * 100, 1) for v in m.beam_off], len(m.aim_photos)))
        for t, text in m.mission.log:
            if "조준" in text or "팬틸트" in text:
                print("   ", t, text)
    unittest.main()
