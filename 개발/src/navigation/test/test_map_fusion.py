"""라이다로 드론 지도 상자를 고치는 계산(navigation/map_fusion.py) 시험. ROS 없이 돈다.

가상 경기장(시험기록/web_drive/sim_world.World 의 광선 추적)에서 로봇 자세마다 라이다 점을 만들어 넣고,
  - 드론 지도에만 있는 상자(없는 상자)가 빠지는지
  - 드론 지도에 없는 상자(빠진 상자)가 더해지는지 — 정합에 쓰는 상자(structure)에만, 내보내는 지도(output)에는 아님
  - 조금 어긋난 진짜 상자를 빼지 않는지, 진짜 상자에 붙은 없는 상자는 빼는지, 차체만 한 물체는 더하지 않는지
  - 가려진 상자 · 한 번만 보인 것 · 움직이는 물체 · 틀린 자세 · 빠른 회전에는 지도를 바꾸지 않는지
  - 정합에 쓰는 상자 목록 · 가중치가 상태를 따르는지
를 본다.
"""
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / '시험기록' / 'web_drive'))
import sim_world as W  # noqa: E402
from navigation.map_fusion import FusionParams, MapFusion  # noqa: E402

WIDTH, HEIGHT, BOX = 6.12, 5.0, 0.4
SENSOR = (0.0398, 0.0)       # base_link → base_lidar_link (slam_node.scan_points 와 같다)
REAL = [(1.4, 1.0), (2.2, 2.4), (3.0, 1.4), (3.0, 3.4), (3.8, 2.4)]
PHANTOM = (1.8, 3.6)          # 드론 지도에만 있는 상자 (실제로는 빈 바닥)
HIDDEN = (1.8, 3.6)           # 드론 지도에 빠진 상자 (같은 자리를 반대로 쓴다)
# 왼쪽 아래에서 위쪽을 보며 천천히 옮겨 가는 자세들 (0.2 s 간격)
PATH = [(0.9 + 0.05 * k, 2.4 + 0.03 * k, 0.6) for k in range(10)]


def body_scan(world, pose, step=2, noise=0.01, seed=0):
    """자세 pose 에서 본 라이다 점 (차체 좌표, (P, 2)). 뒤쪽 90° 는 가린다 (실제 드라이버처럼)."""
    rng = np.random.default_rng(seed)
    count = 505
    theta = np.arange(0, count, step) * (2 * math.pi / count)
    theta = theta[(theta < math.radians(135)) | (theta > math.radians(225))]
    x, y, yaw = pose
    ox = x + SENSOR[0] * math.cos(yaw) - SENSOR[1] * math.sin(yaw)
    oy = y + SENSOR[0] * math.sin(yaw) + SENSOR[1] * math.cos(yaw)
    r = world.rays(ox, oy, yaw + theta) + rng.normal(0.0, noise, len(theta))
    return np.stack((SENSOR[0] + r * np.cos(theta), SENSOR[1] + r * np.sin(theta)), axis=1)


def world_of(boxes):
    return W.World(WIDTH, HEIGHT, BOX, boxes)


def without_walls(points, pose, margin=0.12):
    """외벽 가까이에 찍힌 점을 뺀 스캔 (상자에 가려 벽이 안 보이는 자리 흉내)."""
    x, y, yaw = pose
    wx = x + math.cos(yaw) * points[:, 0] - math.sin(yaw) * points[:, 1]
    wy = y + math.sin(yaw) * points[:, 0] + math.cos(yaw) * points[:, 1]
    return points[(wx > margin) & (wx < WIDTH - margin) & (wy > margin) & (wy < HEIGHT - margin)]


def near(boxes, target, tolerance=0.05):
    return [b for b in boxes if math.hypot(b[0] - target[0], b[1] - target[1]) < tolerance]


class MapFusionTest(unittest.TestCase):
    def setUp(self):
        self.fusion = MapFusion(FusionParams(observe_every=1))

    def feed(self, world, poses, start_s=10.0, dt=0.2, **kwargs):
        changed = False
        for k, pose in enumerate(poses):
            t_ns = int(round((start_s + k * dt) * 1e9))
            changed |= self.fusion.observe(body_scan(world, pose, seed=k), pose, t_ns, sensor=SENSOR, **kwargs)
        return changed

    def output(self):
        return [(x, y) for x, y, _ in self.fusion.output()]

    def matching(self):
        """정합에 쓰는 상자 [(x, y)]."""
        return [tuple(b) for b in self.fusion.structure()[0].tolist()]

    # ------------------------------------------------------------------ 없는 상자 빼기
    def test_phantom_box_is_removed_and_real_boxes_are_confirmed(self):
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.assertTrue(self.feed(world_of(REAL), PATH[:8]))
        out = self.output()
        self.assertEqual(near(out, PHANTOM), [])
        for box in REAL:
            self.assertEqual(len(near(out, box, 1e-6)), 1, box)
        states = self.fusion.drone_states()
        self.assertEqual(states[len(REAL)], 'rejected')
        self.assertEqual(states[1], 'confirmed')     # (2.2, 2.4) 는 바로 앞에 보인다

    def test_one_contradicting_scan_is_not_enough(self):
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), PATH[:1])
        self.assertEqual(len(near(self.output(), PHANTOM)), 1)
        self.assertEqual(self.fusion.drone_states()[len(REAL)], 'drone')

    def test_box_hidden_behind_another_box_is_kept(self):
        """보이지 않는(가려진) 상자는 빈 바닥이라는 증거가 없으니 그대로 둔다."""
        behind = (2.75, 2.4)                       # (2.2, 2.4) 바로 뒤
        self.fusion.set_map(REAL + [behind], BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), [(1.0, 2.4, 0.0)] * 6)
        self.assertEqual(len(near(self.output(), behind, 1e-6)), 1)
        self.assertEqual(self.fusion.drone_states()[len(REAL)], 'drone')

    def test_removed_box_comes_back_when_the_lidar_sees_it(self):
        """잘못 뺀 상자는 라이다에 다시 맞으면 되살린다 (빼는 것이 되돌릴 수 없는 결정이 아니다)."""
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), PATH[:8])
        self.assertEqual(near(self.output(), PHANTOM), [])
        self.feed(world_of(REAL + [PHANTOM]), PATH[:8], start_s=20.0)
        # 1 s 쯤 뒤 라이다로 맞춘 자리에 돌아온다 (revive 3.0 에 닿기 전에 「옮기기」 가 먼저 되살린다 — 드론 자리와 몇 mm 다를 수 있다)
        self.assertEqual(len(near(self.output(), PHANTOM, 0.03)), 1)

    def test_rejected_phantom_does_not_come_back_for_a_vehicle(self):
        """뺀 상자의 면 자리에 다른 UGV(25 × 23 cm)가 서 있어도 되살리지 않는다 (2026-11-19 본선 경기장 16:55 기록: 빠졌다 되살아났다 또 빠짐)."""
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), PATH[:8])
        version = self.fusion.version
        self.assertEqual(near(self.output(), PHANTOM), [])
        ugv = (PHANTOM[0] - 0.08, PHANTOM[1] - 0.20 + 0.115, 0.25, 0.23)   # 그 상자의 아래 면 자리에 선 UGV
        self.feed(world_of(REAL + [ugv]), PATH[:8] + PATH[:8], start_s=20.0)
        self.assertEqual(self.fusion.version, version)
        self.assertEqual(near(self.output(), PHANTOM, 0.3), [])

    # ------------------------------------------------------------------ 빠진 상자 더하기
    def test_missing_box_is_added_for_matching_but_not_published(self):
        """빠진 상자는 정합에 쓰는 상자에만 더한다. 내보내는 지도(주행 노드가 「벽」 으로 다루는 것)에는 넣지 않는다."""
        self.fusion.set_map(REAL, BOX, WIDTH, HEIGHT)
        self.assertTrue(self.feed(world_of(REAL + [HIDDEN]), PATH[:8]))
        added = self.fusion.added()
        self.assertEqual(len(added), 1, added)
        self.assertLess(math.hypot(added[0][0] - HIDDEN[0], added[0][1] - HIDDEN[1]), 0.05)
        self.assertEqual(len(near(self.matching(), HIDDEN)), 1)
        boxes, weights = self.fusion.structure()
        self.assertEqual(len(boxes), len(REAL) + 1)
        self.assertEqual(weights[-1], 1.0)
        self.assertEqual(self.output(), REAL)            # 내보내는 지도는 드론 지도 그대로
        self.assertFalse(self.fusion.changed())
        self.assertEqual(self.fusion.version, 0)

    def test_missing_box_is_not_added_from_a_short_look(self):
        self.fusion.set_map(REAL, BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL + [HIDDEN]), PATH[:2])
        self.assertEqual(self.fusion.added(), [])
        self.assertEqual(len(self.matching()), len(REAL))

    def test_moving_object_is_not_added(self):
        """다른 UGV 처럼 움직이는 것은 한 자리에 머물지 않으므로 상자가 되지 않는다."""
        self.fusion.set_map(REAL, BOX, WIDTH, HEIGHT)
        for k in range(10):
            mover = (1.4 + 0.15 * k, 3.9)
            pose = PATH[k]
            self.fusion.observe(body_scan(world_of(REAL + [mover]), pose, seed=k), pose, int((10 + 0.2 * k) * 1e9), sensor=SENSOR)
        self.assertEqual(self.fusion.added(), [])
        self.assertEqual(len(self.matching()), len(REAL))

    def test_standing_vehicle_sized_object_is_not_added(self):
        """서 있는 다른 UGV(차체 25 × 23 cm)는 오래 보여도 상자가 되지 않는다 — 40 cm 상자라면 30 cm 넘는 폭으로 보인다."""
        self.fusion.set_map(REAL, BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL + [(HIDDEN[0], HIDDEN[1], 0.25, 0.23)]), PATH)
        self.assertEqual(self.fusion.added(), [])
        self.assertEqual(self.output(), REAL)

    def test_added_box_is_removed_after_it_leaves(self):
        self.fusion.set_map(REAL, BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL + [HIDDEN]), PATH[:8])
        self.assertEqual(len(self.fusion.added()), 1)
        self.assertTrue(self.feed(world_of(REAL), PATH, start_s=20.0))
        self.assertEqual(self.fusion.added(), [])
        self.assertEqual(self.matching(), REAL)
        self.assertEqual(self.output(), REAL)

    def test_box_in_the_same_place_as_a_known_one_is_not_added_twice(self):
        self.fusion.set_map(REAL, BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), PATH)
        self.assertEqual(self.output(), REAL)
        self.assertEqual(self.matching(), REAL)

    # ------------------------------------------------------------------ 증거 세는 법 (2026-11-19 실차 기록으로 고친 것)
    def test_shifted_real_box_is_kept(self):
        """드론 지도보다 20 cm 옆에 있는 진짜 상자: 면 한쪽에는 점이 맞고 다른 쪽으로는 광선이 지나간다.
        광선이 지나간다는 것만 보고 빼면 실제로 있는 물체가 지도에서 사라진다 (실차 기록 1118_202820 의 (3.5, 2.1))."""
        real = [b for b in REAL if b != (2.2, 2.4)] + [(2.2, 2.4)]
        drawn = real[:-1] + [(2.2, 2.6)]                  # 드론 지도는 마지막 상자를 y 로 20 cm 옮겨 그렸다
        self.fusion.set_map(drawn, BOX, WIDTH, HEIGHT)
        self.feed(world_of(real), PATH)
        self.assertNotEqual(self.fusion.drone_states()[-1], 'rejected')
        self.assertEqual(len(near(self.output(), (2.2, 2.6), 1e-6)), 1)
        self.assertGreater(self.fusion._drone[-1].score, 0.0)

    def test_phantom_touching_a_real_box_is_removed(self):
        """진짜 상자 앞에 붙여 그린 없는 상자(그림자 등): 그 속을 지난 광선이 진짜 상자 면 = 없는 상자의 뒤쪽 면 자리에 찍힌다.
        그 점들은 없는 상자의 증거가 아니다."""
        phantom = (1.8, 2.4)                               # 진짜 (2.2, 2.4) 의 왼쪽에 맞닿음
        self.fusion.set_map(REAL + [phantom], BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), [(0.9, 2.4 + 0.01 * k, 0.0) for k in range(8)])
        self.assertEqual(self.fusion.drone_states()[-1], 'rejected')
        self.assertEqual(near(self.output(), phantom), [])
        self.assertEqual(len(near(self.output(), (2.2, 2.4), 1e-6)), 1)

    def test_ray_clipping_a_corner_is_not_evidence(self):
        """줄인 상자 귀퉁이를 through_chord_m 보다 짧게 스친 광선으로는 점수를 깎지 않는다 (자세 · 각도가 조금만 틀려도 생긴다)."""
        self.fusion.set_map([(3.0, 2.5)], BOX, WIDTH, HEIGHT)
        # 원점 (1, 2.5) 에서 줄인 속(x 2.86 ~ 3.14, y 2.36 ~ 2.64)의 왼쪽 위 귀퉁이를 3 ~ 6 cm 만 지나 x = 5 까지 가는 광선 셋
        clip = [(4.0, 4.0 * slope) for slope in (0.0730, 0.0735, 0.0740)]
        wall = [(-0.5 + 0.05 * k, 2.5) for k in range(30)]              # 믿을 벽 점 (위쪽 벽 y = 5, 상자를 지나지 않는 방향)
        points = np.array(clip + wall)
        for k in range(6):
            self.fusion.observe(points, (1.0, 2.5, 0.0), int((10 + 0.2 * k) * 1e9))
        self.assertEqual(self.fusion.last['skip'], '')
        self.assertEqual(self.fusion._drone[0].score, 0.0)
        # 같은 광선이 속을 깊이(가운데로) 지나가면 증거가 된다
        deep = np.array([(4.0, 4.0 * slope) for slope in (0.000, 0.005, 0.010)] + wall)
        self.fusion.observe(deep, (1.0, 2.5, 0.0), int(12e9))
        self.assertLess(self.fusion._drone[0].score, 0.0)

    # ------------------------------------------------------------------ 바꾸지 않을 때 (⑥ 안전장치)
    def test_wrong_pose_does_not_change_the_map(self):
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        world = world_of(REAL + [HIDDEN])
        for k, (x, y, yaw) in enumerate(PATH):
            self.fusion.observe(body_scan(world, (x, y, yaw), seed=k), (x + 0.30, y - 0.2, yaw + 0.1), int((10 + 0.2 * k) * 1e9), sensor=SENSOR)
        self.assertEqual(self.fusion.version, 0)
        self.assertEqual(self.output(), REAL + [PHANTOM])
        self.assertEqual(self.fusion.drone_states(), ['drone'] * (len(REAL) + 1))

    def test_confirmed_boxes_keep_the_map_updating_where_walls_are_hidden(self):
        """외벽이 안 보여도 라이다로 확인한 상자가 충분히 보이면 계속 고친다 (실차 기록: 관측의 77 % 에서 벽 점이 10개 미만)."""
        behind = (2.5, 4.6)                                           # 없는 상자 뒤에 있는 진짜 상자 (벽 대신 광선이 끝나는 곳)
        real = REAL + [behind]
        self.fusion.set_map(real + [PHANTOM], BOX, WIDTH, HEIGHT)
        world = world_of(real)
        self.feed(world, PATH[:3])                                    # 벽이 보이는 동안 진짜 상자를 확인
        self.assertEqual(self.fusion.drone_states()[1], 'confirmed')
        self.assertEqual(self.fusion.drone_states()[-1], 'drone')
        for k, pose in enumerate(PATH[3:] + PATH[3:]):
            points = without_walls(body_scan(world, pose, seed=k), pose)
            self.fusion.observe(points, pose, int((11 + 0.2 * k) * 1e9), sensor=SENSOR)
            self.assertEqual(self.fusion.last['skip'], '', k)
        self.assertEqual(self.fusion.drone_states()[-1], 'rejected')

    def test_trusting_confirmed_boxes_can_be_turned_off(self):
        """trust_confirmed_boxes=False 면 처음 만든 대로 외벽 점만 믿는다: 벽이 안 보이는 스캔으로는 고치지 않는다."""
        self.fusion = MapFusion(FusionParams(observe_every=1, trust_confirmed_boxes=False))
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        world = world_of(REAL)
        self.feed(world, PATH[:3])
        self.assertEqual(self.fusion.drone_states()[1], 'confirmed')
        pose = PATH[4]
        self.assertFalse(self.fusion.observe(without_walls(body_scan(world, pose), pose), pose, int(12e9), sensor=SENSOR))
        self.assertIn('믿기 어려움', self.fusion.last['skip'])

    def test_nothing_changes_without_walls_or_confirmed_boxes(self):
        """외벽도 확인한 상자도 안 보이면 자세가 맞다는 근거가 없다 → 고치지 않는다."""
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        world = world_of(REAL)
        for k, pose in enumerate(PATH):
            points = without_walls(body_scan(world, pose, seed=k), pose)
            self.assertFalse(self.fusion.observe(points, pose, int((10 + 0.2 * k) * 1e9), sensor=SENSOR))
        self.assertIn('믿기 어려움', self.fusion.last['skip'])
        self.assertEqual(self.fusion.drone_states(), ['drone'] * (len(REAL) + 1))

    def test_fast_turn_is_skipped(self):
        """빠르게 도는 동안의 스캔은 한 바퀴 안에서 자세가 바뀌어 번진다 → 지도 고치기에 쓰지 않는다."""
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.assertFalse(self.feed(world_of(REAL), PATH, yaw_rate=1.5))
        self.assertEqual(self.output(), REAL + [PHANTOM])

    def test_observe_every_skips_scans(self):
        fusion = MapFusion(FusionParams(observe_every=3))
        self.assertEqual([fusion.wants_scan() for _ in range(6)], [True, False, False, True, False, False])

    # ------------------------------------------------------------------ 정합에 쓰는 구조
    def test_structure_follows_the_box_states(self):
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        boxes, weights = self.fusion.structure()
        self.assertEqual(boxes.shape, (len(REAL) + 1, 2))
        self.assertTrue(np.all(weights == self.fusion.params.unconfirmed_weight))
        self.feed(world_of(REAL), PATH[:8])
        boxes, weights = self.fusion.structure()
        self.assertEqual(len(boxes), len(REAL))                       # 뺀 상자는 정합에도 쓰지 않는다
        self.assertEqual(near([tuple(b) for b in boxes], PHANTOM), [])
        confirmed = [w for b, w in zip(boxes, weights) if math.hypot(b[0] - 2.2, b[1] - 2.4) < 1e-6]
        self.assertEqual(confirmed, [1.0])

    # ------------------------------------------------------------------ 옮겨진 상자 (드론 지도의 자리가 틀린 상자)
    def moved_map(self):
        """진짜 상자 HIDDEN 을 드론 지도는 0.28 m 어긋난 자리에 그렸다 (화면 끝에서 잘린 상자 — 2026-11-19 모의 경기 3차: (4.18, 3.72) ↔ (4.26, 3.99)).

        그 판처럼 로봇 쪽으로 튀어나오게 어긋났다 (튀어나온 데로 광선이 지나가 「없다」 는 증거가 쌓인다. 진짜 상자 뒤에 숨은 쪽으로 어긋나면
        보이지 않으므로 빼지도 옮기지도 못한다 — 그때는 로봇이 다른 쪽에서 볼 때까지 그대로다).
        """
        wrong = (HIDDEN[0] - 0.10, HIDDEN[1] - 0.26)
        self.fusion.set_map(REAL + [wrong], BOX, WIDTH, HEIGHT)
        return wrong

    def test_box_drawn_in_the_wrong_place_is_moved_to_where_the_lidar_sees_it(self):
        wrong = self.moved_map()
        self.assertTrue(self.feed(world_of(REAL + [HIDDEN]), PATH + PATH[::-1] + PATH))
        out = self.fusion.output()
        self.assertEqual(len(out), len(REAL) + 1)                      # 상자 수는 그대로다 — 빼고 끝나지 않는다
        x, y, source = out[len(REAL)]
        self.assertEqual(source, len(REAL))                            # 같은 드론 지도 번호로 나간다
        self.assertLess(math.hypot(x - HIDDEN[0], y - HIDDEN[1]), 0.06)  # 라이다로 맞춘 자리
        self.assertEqual(near(self.output(), wrong, 0.10), [])         # 틀린 자리에는 없다
        self.assertTrue(self.fusion.changed())
        self.assertEqual(self.fusion.drone_states()[len(REAL)], 'confirmed')
        self.assertEqual(len(near(self.fusion.added(), HIDDEN, 0.06)), 1)  # supervisor 에는 「라이다로 찾은 상자」 로도 알린다
        moved = self.fusion.summary()['moved_boxes']
        self.assertEqual(len(moved), 1)
        self.assertLess(math.hypot(moved[0][0] - wrong[0], moved[0][1] - wrong[1]), 0.01)

    def test_moved_box_stays_moved(self):
        """옮긴 뒤에 계속 봐도, 같은 드론 지도가 다시 와도 틀린 자리로 돌아가지 않는다 (그 판: 뺀 상자가 110초 뒤에 되살아났다)."""
        wrong = self.moved_map()
        world = world_of(REAL + [HIDDEN])
        self.feed(world, PATH + PATH[::-1] + PATH)
        version = self.fusion.version
        self.feed(world, PATH[::-1] + PATH, start_s=30.0)
        self.assertEqual(self.fusion.version, version)
        self.assertEqual(near(self.output(), wrong, 0.10), [])
        self.fusion.set_map(REAL + [wrong], BOX, WIDTH, HEIGHT)        # 같은 드론 지도가 다시 왔다
        self.assertEqual(near(self.output(), wrong, 0.10), [])
        self.assertEqual(len(near(self.output(), HIDDEN, 0.06)), 1)
        self.assertTrue(self.fusion.changed())

    def test_confirmed_box_is_not_moved_by_something_beside_it(self):
        """라이다로 확인한 상자는 옮기지 않는다 — 그 곁에 선 다른 UGV 때문에 진짜 상자가 끌려가면 안 된다."""
        self.fusion.set_map(REAL, BOX, WIDTH, HEIGHT)
        beside = (REAL[1][0], REAL[1][1] + 0.33)                       # (2.2, 2.4) 상자에 붙어 선 차체만 한 물체
        self.feed(world_of(REAL), PATH)
        self.assertEqual(self.fusion.drone_states()[1], 'confirmed')
        before = self.output()
        self.feed(world_of(REAL + [beside]), PATH + PATH[::-1], start_s=20.0)
        for a, b in zip(before, self.output()[:len(before)]):
            self.assertLess(math.hypot(a[0] - b[0], a[1] - b[1]), 1e-9)
        self.assertEqual(self.fusion.summary()['moved_boxes'], [])

    def test_moving_can_be_switched_off(self):
        self.fusion = MapFusion(FusionParams(observe_every=1, move_m=0.0, overlap_m=0.0))
        wrong = self.moved_map()
        self.feed(world_of(REAL + [HIDDEN]), PATH + PATH[::-1] + PATH)
        self.assertEqual(self.fusion.summary()['moved_boxes'], [])
        self.assertEqual(near(self.output(), HIDDEN, 0.06), [])        # 전처럼: 내보내는 지도에는 진짜 자리의 상자가 없다
        self.assertEqual(len(near(self.fusion.added(), HIDDEN, 0.06)), 1)
        del wrong

    def test_new_drone_map_keeps_what_the_lidar_learned(self):
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), PATH[:8])
        jittered = [(x + 0.02, y - 0.02) for x, y in REAL + [PHANTOM]]
        self.fusion.set_map(jittered, BOX, WIDTH, HEIGHT)
        states = self.fusion.drone_states()
        self.assertEqual(states[len(REAL)], 'rejected')
        self.assertEqual(states[1], 'confirmed')
        self.assertEqual(len(self.output()), len(REAL))

    def test_reset_forgets_everything(self):
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.feed(world_of(REAL), PATH[:8])
        self.fusion.reset()
        self.assertEqual(self.fusion.output(), [])
        self.fusion.set_map(REAL + [PHANTOM], BOX, WIDTH, HEIGHT)
        self.assertEqual(self.fusion.drone_states(), ['drone'] * (len(REAL) + 1))

    def test_no_map_means_no_work(self):
        self.assertFalse(self.fusion.observe(np.zeros((50, 2)) + 1.0, (1.0, 1.0, 0.0), 1, sensor=SENSOR))

    # ------------------------------------------------------------------ 계산량
    def test_observe_is_cheap(self):
        """로봇 CPU 에서 스캔 정합이 약 35 ms 다. 지도 고치기는 그보다 훨씬 가벼워야 한다 (이 PC 기준 평균 8 ms 미만)."""
        boxes = [(0.8 + 0.6 * i, 0.8 + 0.85 * j) for i in range(8) for j in range(5)][:20]
        world = world_of(boxes[:-3] + [(5.5, 4.4)])
        self.fusion.set_map(boxes, BOX, WIDTH, HEIGHT)
        pose = (1.1, 2.5, 0.3)
        scans = [body_scan(world, pose, seed=k) for k in range(20)]
        self.fusion.observe(scans[0], pose, 1, sensor=SENSOR)
        started = time.perf_counter()
        for k, points in enumerate(scans):
            self.fusion.observe(points, pose, int((1 + 0.2 * k) * 1e9), sensor=SENSOR)
        self.assertLess((time.perf_counter() - started) / len(scans), 0.008)


if __name__ == '__main__':
    unittest.main()
