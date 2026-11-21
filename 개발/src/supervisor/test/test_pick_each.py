"""덤 적기: 겨눈 면 말고 같은 사진에 찍힌 다른 면도 자리 · 크기로 붙여 적는다 (scout.pick_each · FaceBook.add 의 bonus)."""
import math
import unittest

import sim_world
from supervisor import arena as A
from supervisor.faces import CameraParams, FaceBook, expected_faces
from supervisor.scout import FaceScout, ScoutParams, pick_centered, pick_each

CAM = CameraParams()


def sized(e, kind, dx=0.0, scale=1.0, cut=False, conf=0.9):
    """면 e 가 찍힐 자리 · 크기대로 만든 그림 (실차의 답 모양: 종류, 가로 자리, 잘림, 가로 크기, 세로 크기, 확신)."""
    return (kind, e.x + dx, cut, e.width * scale, e.height * scale, conf)


class TwoTouchingBoxes(unittest.TestCase):
    """나란히 붙은 상자 둘의 남쪽 면을 1 m 앞에서 본다 (렌즈가 면에서 1.0 m). 오른쪽 뒤 멀리 상자 하나가 더 있다."""

    def setUp(self):
        self.fmap = A.field_map([(3.0, 3.5), (3.4, 3.5), (4.3, 4.6)])
        self.pose = (3.2, 2.25, math.radians(90.0))
        self.exp = expected_faces(self.fmap, self.pose, 0.0, CAM)
        self.by = {e.face.key: e for e in self.exp}
        self.left, self.right = self.by[(0, "S")], self.by[(1, "S")]
        self.p = ScoutParams()

    def test_expected_size_matches_geometry(self):
        e = self.left
        self.assertAlmostEqual(e.fraction, 1.0)
        # 세로: 0.4 m 를 가장 가까운 곳의 거리에서 본 각도 / 사진의 세로 화각 (가로 화각 × 0.75)
        self.assertAlmostEqual(e.height, 2.0 * math.atan2(0.2, 1.0) / (math.radians(CAM.hfov_deg) * CAM.aspect), delta=0.02)
        self.assertGreater(e.width, 0.12)
        self.assertLess(e.width, 0.20)

    def test_adjacent_faces_get_their_own_pictures(self):
        pics = [sized(self.left, "tank", dx=0.01), sized(self.right, "drum", dx=-0.012)]
        got = {e.face.key: k for e, k in pick_each(self.exp, pics, self.p)}
        self.assertEqual(got, {(0, "S"): "tank", (1, "S"): "drum"})
        # 겨눈 면은 따로 정하므로 skip 으로 뺀다
        got = {e.face.key: k for e, k in pick_each(self.exp, pics, self.p, skip=((0, "S"),))}
        self.assertEqual(got, {(1, "S"): "drum"})

    def test_far_picture_at_the_face_position_is_refused_by_size(self):
        # 상자가 옮겨져 그 자리에 먼 상자의 그림이 찍혔다: 자리는 맞지만 크기가 0.4배
        pics = [sized(self.left, "tank", scale=0.4), sized(self.right, "drum")]
        got = {e.face.key: k for e, k in pick_each(self.exp, pics, self.p)}
        self.assertEqual(got, {(1, "S"): "drum"})
        # 겨눈 면도 같다: 크기를 주면 거르고, 크기를 안 보면(옛 방식) 적힌다
        three = [q[:3] for q in pics]
        self.assertIsNone(pick_centered(self.exp, pics, (0, "S"), self.p.center_tol, self.p.center_sep, self.p.aim_height))
        self.assertEqual(pick_centered(self.exp, three, (0, "S"), self.p.center_tol, self.p.center_sep, self.p.aim_height)[1], "tank")
        self.assertEqual(pick_centered(self.exp, pics, (1, "S"), self.p.center_tol, self.p.center_sep, self.p.aim_height)[1], "drum")

    def test_wall_backdrop_without_a_face_is_ignored(self):
        # 벽의 배경 그림: 어느 면의 자리도 아닌 곳에 가로로 긴 그림
        pics = [(A.NON_OBJECT, 0.08, False, 0.20, 0.11, 0.8), sized(self.left, "tank"), sized(self.right, "drum"), (A.NON_OBJECT, 0.93, False, 0.18, 0.12, 0.9)]
        got = {e.face.key: k for e, k in pick_each(self.exp, pics, self.p)}
        self.assertEqual(got, {(0, "S"): "tank", (1, "S"): "drum"})

    def test_skips_what_cannot_be_told_apart(self):
        e = self.left
        # 「모름」 · 잘린 그림 · 확신이 낮은 그림
        for pic in (sized(e, A.UNKNOWN), sized(e, "tank", cut=True), sized(e, "tank", conf=0.3)):
            self.assertEqual(pick_each(self.exp, [pic], self.p), [])
        # 자리가 허용(0.05)보다 멀다
        self.assertEqual(pick_each(self.exp, [sized(e, "tank", dx=-0.06)], self.p), [])
        # 한 면 자리 가까이에 크기가 맞는 그림이 둘 → 어느 것인지 모른다
        self.assertEqual(pick_each(self.exp, [sized(e, "tank", dx=-0.03), sized(e, "drum", dx=0.03)], self.p), [])

    def test_picture_between_two_faces_is_not_recorded(self):
        # 2 m 떨어져서 보면 붙은 두 면이 사진에서 가깝다 → 두 면 사이에 찍힌 그림은 어느 면의 것인지 모른다
        far_view = expected_faces(self.fmap, (3.2, 1.25, math.radians(90.0)), 0.0, CAM)
        by = {e.face.key: e for e in far_view}
        left, right = by[(0, "S")], by[(1, "S")]
        p = ScoutParams(each_range_m=3.0)
        mid = (left.x + right.x) / 2.0
        self.assertLess(abs(mid - left.x), p.each_tol)
        self.assertEqual(pick_each(far_view, [("tank", mid, False, left.width, left.height, 0.9)], p), [])
        # 제자리에 찍힌 그림은 적힌다 (이웃 면은 each_sep 밖)
        self.assertGreater(abs(right.x - left.x), p.each_sep)
        self.assertEqual([(e.face.key, k) for e, k in pick_each(far_view, [sized(left, "tank")], p)], [((0, "S"), "tank")])

    def test_far_and_steep_faces_are_not_recorded(self):
        far = self.by[(2, "S")]  # 2.4 m 떨어진 상자 (그 서쪽 면은 거의 옆에서 보여 가로 크기로 가려진다)
        self.assertGreater(far.dist, self.p.each_range_m)
        self.assertEqual(pick_each(self.exp, [sized(far, "missile")], self.p), [])
        wide = ScoutParams(each_range_m=3.0)
        self.assertEqual([(e.face.key, k) for e, k in pick_each(self.exp, [sized(far, "missile")], wide)], [((2, "S"), "missile")])

    def test_pictures_without_size_give_no_bonus(self):
        three = [("tank", self.left.x, False), ("drum", self.right.x, False)]
        self.assertEqual(pick_each(self.exp, three, self.p), [])

    # ---- FaceScout.on_answer 까지
    def scout(self, **kw):
        book = FaceBook(self.fmap, "BLUE")
        return FaceScout(self.fmap, None, book, CAM, ScoutParams(**kw)), book

    def test_on_answer_records_aimed_face_and_bonus(self):
        scout, book = self.scout()
        notes = []
        scout.trace = notes.append
        backdrop = (A.NON_OBJECT, 0.08, False, 0.20, 0.11, 0.8)  # 면이 없는 그림이 하나 있으면 사진 전체 짝짓기는 버려진다 (실차와 같다)
        scout.on_answer(("face", (0, "S"), 0), self.pose, 0.0, [backdrop, sized(self.left, "tank"), sized(self.right, "drum")])
        self.assertEqual((book.kind((0, "S")), book.kind((1, "S"))), ("tank", "drum"))
        self.assertEqual(scout.bonus, 1)
        self.assertEqual([(q["box"], q["side"], q["kind"], q.get("bonus", False)) for q in notes[-1]["pairs"]], [(0, "S", "tank", False), (1, "S", "drum", True)])
        self.assertIn("h", notes[-1]["pictures"][0])
        # 덤의 표는 겨눠 얻은 답보다 작다
        self.assertLess(sum(book.votes[(1, "S")].values()), 0.3)
        self.assertGreaterEqual(sum(book.votes[(0, "S")].values()), 0.3)

    def test_no_bonus_when_the_aimed_face_was_not_read(self):
        # 겨눈 면의 그림이 제자리에 없다 (사진이 통째로 밀렸을 수 있다) → 덤도 적지 않는다
        backdrop = (A.NON_OBJECT, 0.08, False, 0.20, 0.11, 0.8)
        pics = [backdrop, sized(self.right, "drum")]
        scout, book = self.scout()
        scout.on_answer(("face", (0, "S"), 0), self.pose, 0.0, pics)
        self.assertEqual((book.kind((0, "S")), book.kind((1, "S"))), (None, None))
        scout, book = self.scout(each_need_aim=False)
        scout.on_answer(("face", (0, "S"), 0), self.pose, 0.0, pics)
        self.assertEqual((book.kind((0, "S")), book.kind((1, "S"))), (None, "drum"))

    def test_three_value_pictures_still_work(self):
        # 크기 없는 답 (가상 시험 · 옛 형식): 사진 전체 짝짓기 → 안 되면 겨눈 면만. 예전과 같다
        scout, book = self.scout()
        scout.on_answer(("face", (0, "S"), 0), self.pose, 0.0, [("tank", self.left.x, False), ("drum", self.right.x, False)])
        self.assertEqual(book.kind((0, "S")), "tank")
        scout, book = self.scout()
        scout.on_answer(("face", (0, "S"), 0), self.pose, 0.0, [(A.NON_OBJECT, 0.08, False), ("tank", self.left.x, False), ("drum", self.right.x, False)])
        self.assertEqual((book.kind((0, "S")), book.kind((1, "S"))), ("tank", None))


class BonusVotes(unittest.TestCase):
    def setUp(self):
        self.fmap = A.field_map([(3.0, 2.5)])
        self.book = FaceBook(self.fmap, "BLUE")
        self.exp = next(e for e in expected_faces(self.fmap, (3.0, 1.45, math.radians(90.0)), 0.0, CAM) if e.face.key == (0, "S"))

    def test_bonus_cannot_overturn_an_earlier_answer(self):
        self.assertTrue(self.book.add(self.exp, "tank"))
        self.assertFalse(self.book.add(self.exp, "drum", bonus=True))  # 다른 종류의 덤은 적지 않는다
        self.assertFalse(self.book.add(self.exp, "drum", bonus=True))
        self.assertEqual(self.book.kind((0, "S")), "tank")
        self.assertTrue(self.book.add(self.exp, "tank", bonus=True))  # 같은 종류면 표를 보탠다
        self.assertAlmostEqual(self.book.votes[(0, "S")]["tank"], 1.0 + FaceBook.bonus_weight)

    def test_aimed_answer_beats_an_earlier_bonus(self):
        self.assertTrue(self.book.add(self.exp, "drum", bonus=True))
        self.assertEqual(self.book.kind((0, "S")), "drum")
        self.assertIn((0, "S"), self.book.votes)  # 덤만으로도 「적힌 면」 이 된다 (다시 겨누러 가지 않는다)
        self.book.add(self.exp, "tank")
        self.assertEqual(self.book.kind((0, "S")), "tank")

    def test_bonus_unknown_is_not_counted_and_limit_is_used(self):
        self.assertFalse(self.book.add(self.exp, A.UNKNOWN, bonus=True))
        self.assertNotIn((0, "S"), self.book.looks)
        self.assertFalse(self.book.add(self.exp, "tank", bonus=True, limit=(0.5, 55.0)))  # 0.8 m 는 0.5 m 한도 밖
        self.assertTrue(self.book.add(self.exp, "tank", bonus=True, limit=(1.6, 55.0)))


class SizedPerception(sim_world.FakePerception):
    """가짜 perception 을 실차처럼: 그림마다 크기를 달고, 면이 없는 곳에 벽 배경 그림을 두 장 섞는다 (사진 전체 짝짓기가 버려지게)."""

    def look(self, pose, pan_deg=0.0):
        pics, keys = [], []
        exp = expected_faces(self.fmap, pose, pan_deg, self.cam)
        for e in exp:
            clear = e.fraction >= 0.6 and e.incidence_deg <= 70.0
            if not clear:
                continue
            grow = 1.0 + self.rng.uniform(0.0, 0.2)  # 실차: 답한 그림이 계산한 크기의 0.93 ~ 1.25배
            pics.append((self.truth[e.face.key], e.x + self.rng.gauss(0.0, self.x_noise), e.fraction < 0.95, e.width * grow, e.height * grow, 0.9))
            keys.append(e.face.key)
        for _ in range(2):
            x = self.rng.uniform(0.05, 0.95)
            if all(abs(x - e.x) > 0.12 for e in exp):
                pics.append((A.NON_OBJECT, x, False, 0.18, 0.11, 0.8))
                keys.append(None)
        order = sorted(range(len(pics)), key=lambda k: pics[k][1])
        return [pics[k] for k in order], [keys[k] for k in order]


class WholeScoutWithSizedPictures(unittest.TestCase):
    """정찰 한 번을 끝까지: 실차처럼 답이 오면(크기 + 벽 그림) 덤이 적히고, 덤을 끈 것보다 못 본 면이 늘지 않는다.

    위치 오차를 넣은 판은 한 판 한 판의 결과가 크게 흔들려(같은 설정도 판마다 못 본 면이 1 ~ 4개) 견줄 수 없다 → 오차 없는 판으로 본다.
    2026-11-19 오차를 넣은 12판의 평균은 덤을 켠 것 · 끈 것이 같았다 (맞은 구역 3.67 / 3.75, 못 본 면 1.9 / 1.8, 틀리게 적힌 덤 0).
    """

    def run_both(self, **kw):
        keep = sim_world.FakePerception
        sim_world.FakePerception = SizedPerception
        try:
            on = sim_world.run_scout("face", params=ScoutParams(mode="face"), **kw)
            off = sim_world.run_scout("face", params=ScoutParams(mode="face", each_range_m=0.0), **kw)
        finally:
            sim_world.FakePerception = keep
        return on, off

    def test_coverage_does_not_drop_and_nothing_is_wrong(self):
        for kw in ({}, {"camp": "RED", "start": (5.9, 2.5, math.pi)}):
            on, off = self.run_both(pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0, **kw)
            self.assertTrue(on["finished"], on)
            self.assertEqual(on["faces_wrong"], 0, on)
            # 남은 면 · 사진 수는 도는 순서에 따라 한두 개씩 오르내린다 (2026-11-19: 순서를 최단 경로로 바꾸고 가까운 면의 허용을 넓힌 뒤
            # 한쪽 진영에서 덤을 켠 쪽이 2면 · 1장 많았다 — 구역은 양쪽 다 4개 모두 맞고 틀린 면은 없다). 덤 때문에 크게 나빠지지만 않으면 된다
            self.assertLessEqual(on["left"], off["left"] + 2, (on, off))
            self.assertGreaterEqual(on["sectors_ok"], off["sectors_ok"], (on, off))
            self.assertLessEqual(on["shots"], off["shots"] + 2, (on, off))  # 덤으로 적힌 면은 다시 겨누지 않는다




class AimedFaceIsNotDroppedByTheWholePhotoMatch(unittest.TestCase):
    """2026-11-19 14:13 실차(1경기장 B-1)의 두 번째 사진. 지정표적(통신시설)을 사진 한가운데에 찍었고 모델도 맞게 읽었는데,
    사진 전체 짝짓기가 다른 면에 맞추느라 겨눈 면을 빼 버려 아무것도 적히지 않았다 — 그 판은 지정표적을 못 찾은 것이 되어 한 발도 못 쐈다."""

    BOXES = [(1.44, 4.01), (1.87, 3.97), (4.23, 3.98), (4.61, 4.0), (3.06, 3.4), (1.84, 2.82), (4.22, 2.82), (1.88, 2.43), (4.23, 2.36), (3.08, 1.58),
             (1.85, 1.2), (4.28, 1.17), (1.85, 0.81), (4.27, 0.74)]
    POSE = (2.872, 3.892, math.radians(28.3))
    PAN = 27.0
    PICTURES = [("comms", 0.479, False, 0.157, 0.22, 0.94), ("non_object", 0.691, False, 0.089, 0.111, 0.83), ("mortar", 0.766, False, 0.065, 0.169, 0.9),
                ("ammo_box", 0.878, True, 0.244, 0.979, 0.91)]

    def test_centred_target_is_recorded(self):
        from supervisor.grid import Grid

        fmap = A.field_map(self.BOXES, A.arena_from({"width": 6.12, "height": 4.8}), None)
        book = FaceBook(fmap, "BLUE")
        scout = FaceScout(fmap, Grid(fmap), book, params=ScoutParams())
        scout.on_answer(("face", (2, "W"), 0), self.POSE, self.PAN, list(self.PICTURES))
        self.assertEqual(book.kind((2, "W")), "comms")
        found = book.find_target("RED_COMMS")
        self.assertIsNotNone(found)
        self.assertEqual(found.key, (2, "W"))


class CloseUpAimedFace(unittest.TestCase):
    """2026-11-19 14:13 실차의 열다섯 번째 사진: 0.44 m 앞 · 55° 비스듬히 찍은 탄약상자. 그림이 0.068 밀려 있어(크기는 1.05배) 전에는 적지 못했다."""

    BOXES = AimedFaceIsNotDroppedByTheWholePhotoMatch.BOXES
    POSE = (3.535, 3.807, math.radians(-85.1))
    PAN = 39.3
    PICTURES = [("military_truck", 0.031, True, 0.062, 0.338, 0.9), ("outpost", 0.076, False, 0.049, 0.217, 0.81), ("unknown", 0.28, False, 0.028, 0.124, 0.18),
                ("non_object", 0.313, False, 0.047, 0.126, 0.85), ("ammo_box", 0.577, False, 0.497, 0.736, 0.93), ("radar", 0.961, True, 0.078, 0.214, 0.93)]

    def scout(self, **params):
        from supervisor.grid import Grid

        fmap = A.field_map(self.BOXES, A.arena_from({"width": 6.12, "height": 4.8}), None)
        book = FaceBook(fmap, "BLUE")
        return FaceScout(fmap, Grid(fmap), book, params=ScoutParams(**params)), book

    def test_close_face_is_recorded_with_the_wider_tolerance(self):
        scout, book = self.scout()
        scout.on_answer(("face", (4, "E"), 0), self.POSE, self.PAN, list(self.PICTURES))
        self.assertEqual(book.kind((4, "E")), "ammo_box")
        scout, book = self.scout(center_tol_near=None, aim_incidence_deg=55.0)  # 전의 규칙
        scout.on_answer(("face", (4, "E"), 0), self.POSE, self.PAN, list(self.PICTURES))
        self.assertIsNone(book.kind((4, "E")))

    def test_a_far_face_keeps_the_narrow_tolerance(self):
        e = type("E", (), {})()
        e.face = type("F", (), {"key": (1, "N")})()
        e.x, e.dist, e.height = 0.50, 1.5, 0.15
        far = [("tank", 0.57, False, 0.1, 0.15, 0.9)]  # 1.5 m 밖의 면에서 0.07 밀림 → 허용 0.05 를 넘는다
        self.assertIsNone(pick_centered([e], far, (1, "N"), 0.05, 0.08, (0.5, 2.0), (0.03, 0.03, 0.12)))
        e.dist = 0.45  # 같은 밀림이라도 0.45 m 앞이면 된다 (허용 0.097)
        self.assertIsNotNone(pick_centered([e], far, (1, "N"), 0.05, 0.08, (0.5, 2.0), (0.03, 0.03, 0.12)))

    def test_a_close_picture_of_the_wrong_size_does_not_veto_the_aimed_face(self):
        """모의 5차 185.9초: 겨눈 면의 전차(세로가 맞음) 옆 0.055 에 뒤 상자의 탄약 상자(세로 0.60배)가 찍혀 면을 적지 않았다 → 구역 하나(7점)."""
        from types import SimpleNamespace as N

        e = N(face=N(key=(2, "N")), x=0.50, dist=0.48, height=0.50, width=0.5, incidence_deg=45.0)
        tank = ("tank", 0.572, False, 0.40, 0.52, 0.95)  # 자리 0.072 어긋남 · 세로 1.04배
        rear = ("ammo_box", 0.627, False, 0.25, 0.30, 0.9)  # 0.055 더 멀다 · 세로 0.60배
        args = ((2, "N"), 0.05, 0.08, (0.5, 2.0), (0.03, 0.03, 0.12))
        self.assertIsNone(pick_centered([e], [tank, rear], *args))  # rival 을 주지 않으면 전과 같다
        self.assertEqual(pick_centered([e], [tank, rear], *args, (0.85, 1.45))[1], "tank")
        same = ("ammo_box", 0.627, False, 0.40, 0.50, 0.9)  # 옆 그림도 세로가 맞으면 헷갈리는 것이다
        self.assertIsNone(pick_centered([e], [tank, same], *args, (0.85, 1.45)))
        small = ("tank", 0.572, False, 0.20, 0.30, 0.95)  # 고른 그림의 세로가 맞지 않으면 적지 않는다
        self.assertIsNone(pick_centered([e], [small, rear], *args, (0.85, 1.45)))
        bare = ("ammo_box", 0.627, False)  # 크기를 모르는 옆 그림은 전처럼 막는다
        self.assertIsNone(pick_centered([e], [tank, bare], *args, (0.85, 1.45)))
        third = ("drum", 0.630, False, 0.40, 0.50, 0.9)  # 세 번째 그림이 가까이 있고 세로가 맞아도 막는다
        self.assertIsNone(pick_centered([e], [tank, rear, third], *args, (0.85, 1.45)))
