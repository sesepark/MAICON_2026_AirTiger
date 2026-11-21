"""사진 한 장의 그림들을 면들에 한꺼번에 맞추기 (align.aligned) — 멀리서 본 나란한 면을 「밀림」 을 빼고 가린다.

2026-11-19 14:52 실차의 사진 101장을 정답표와 견줘 만든 것: 판독 모델은 3 m 밖의 그림도 맞게 읽는데, 그림을 하나씩 따로 붙이는 규칙은
나란히 붙은 상자의 면을 2 m 밖에서 가리지 못해 버렸다.
"""
import math
import unittest

from supervisor import arena as A
from supervisor.align import aligned
from supervisor.faces import CameraParams, FaceBook, expected_faces
from supervisor.grid import Grid
from supervisor.scout import FaceScout, ScoutParams, pick_votes

CAM = CameraParams()


def sized(e, kind, dx=0.0, scale=1.0, cut=False, conf=0.9):
    return (kind, e.x + dx, cut, e.width * scale, e.height * scale, conf)


class FarPairTests(unittest.TestCase):
    """나란히 붙은 상자 둘의 남쪽 면을 2.6 m 앞에서 본다. 차체 방향 값이 틀려 사진이 통째로 0.04 밀렸다."""

    def setUp(self):
        self.fmap = A.field_map([(3.0, 3.5), (3.4, 3.5), (1.2, 3.9)])
        self.pose = (3.2, 0.7, math.radians(90.0))
        self.exp = expected_faces(self.fmap, self.pose, 0.0, CAM)
        self.by = {e.face.key: e for e in self.exp}
        self.left, self.right = self.by[(0, "S")], self.by[(1, "S")]

    def test_the_two_faces_are_closer_than_the_single_picture_rule_allows(self):
        self.assertLess(abs(self.left.x - self.right.x), ScoutParams().vote_sep)
        self.assertGreater(self.left.dist, 2.2)

    def test_shifted_photo_is_matched_as_a_whole(self):
        pics = [sized(self.left, "tank", dx=0.04), sized(self.right, "drum", dx=0.04)]
        self.assertEqual(pick_votes(self.exp, pics, ScoutParams()), [])  # 하나씩 따로 붙이면 버린다
        shift, pairs = aligned(self.exp, pics)
        self.assertAlmostEqual(shift, 0.04, places=3)
        self.assertEqual({e.face.key: kind for e, kind, _d, _q in pairs}, {(0, "S"): "tank", (1, "S"): "drum"})

    def test_one_picture_alone_is_not_matched(self):
        self.assertEqual(aligned(self.exp, [sized(self.left, "tank", dx=0.04)]), (None, []))

    def test_cut_unknown_and_unsure_pictures_are_not_used(self):
        pics = [sized(self.left, "tank", cut=True), sized(self.right, "drum")]
        self.assertEqual(aligned(self.exp, pics), (None, []))
        pics = [sized(self.left, A.UNKNOWN), sized(self.right, "drum")]
        self.assertEqual(aligned(self.exp, pics), (None, []))
        pics = [sized(self.left, "tank", conf=0.3), sized(self.right, "drum")]
        self.assertEqual(aligned(self.exp, pics, min_conf=0.8), (None, []))

    def test_far_picture_of_another_size_is_ignored(self):
        # 벽의 배경 그림(가로로 길다)과 훨씬 작은 그림은 맞추는 데 쓰지 않는다 → 남은 둘로 맞춘다
        pics = [(A.NON_OBJECT, 0.10, False, 0.30, 0.10, 0.9), sized(self.left, "tank", dx=-0.02), sized(self.right, "drum", dx=-0.02),
                sized(self.right, "missile", dx=0.2, scale=0.3)]
        _shift, pairs = aligned(self.exp, pics)
        self.assertEqual({e.face.key: kind for e, kind, _d, _q in pairs}, {(0, "S"): "tank", (1, "S"): "drum"})


class RowOfThreeTests(unittest.TestCase):
    """같은 간격으로 놓인 면 셋 가운데 그림 둘만 읽혔다 — 한 칸 밀려도 맞으므로 어느 쪽인지 알 수 없다 → 맞추지 않는다."""

    def setUp(self):
        self.fmap = A.field_map([(2.6, 3.5), (3.0, 3.5), (3.4, 3.5)])
        self.exp = expected_faces(self.fmap, (3.0, 0.9, math.radians(90.0)), 0.0, CAM)
        self.by = {e.face.key: e for e in self.exp}

    def test_two_of_three_is_ambiguous(self):
        a, b = self.by[(0, "S")], self.by[(1, "S")]
        step = b.x - a.x
        self.assertLess(abs(step), 0.14)
        pics = [sized(a, "tank", dx=step / 2.0), sized(b, "drum", dx=step / 2.0)]  # 반 칸 밀렸다: (a, b) 로도 (b, c) 로도 맞는다
        self.assertEqual(aligned(self.exp, pics, max_shift=0.07, tol=0.03), (None, []))

    def test_all_three_is_clear(self):
        a, b, c = self.by[(0, "S")], self.by[(1, "S")], self.by[(2, "S")]
        pics = [sized(a, "tank", dx=0.03), sized(b, "drum", dx=0.03), sized(c, "mortar", dx=0.03)]
        _shift, pairs = aligned(self.exp, pics)
        self.assertEqual({e.face.key: kind for e, kind, _d, _q in pairs}, {(0, "S"): "tank", (1, "S"): "drum", (2, "S"): "mortar"})


class SightingsFromFarTests(unittest.TestCase):
    """정찰에 물린 것: 멀리서 본 나란한 두 면도 서로 다른 자리의 사진 두 장이 같은 답이면 적는다 (vote_align). 끄면 적지 않는다."""

    def run_two_photos(self, **params):
        fmap = A.field_map([(4.3, 2.4), (4.7, 2.4)], A.arena_from({"width": 6.12, "height": 4.80}), None)
        book = FaceBook(fmap, "BLUE")
        scout = FaceScout(fmap, Grid(fmap), book, params=ScoutParams(**params))
        for n, pose in enumerate(((4.5, 0.2, math.radians(90.0)), (4.2, 0.3, math.radians(88.0)))):
            exp = {e.face.key: e for e in expected_faces(fmap, pose, 0.0, scout.cam)}
            pics = [sized(exp[(0, "S")], "tank", dx=0.03), sized(exp[(1, "S")], "drum", dx=0.03)]
            scout.on_pass(("pass", n), pose, 0.0, pics, at=float(n))
        return {k: book.kind(k) for k in book.votes}

    def test_recorded_only_with_the_whole_photo_match(self):
        self.assertEqual(self.run_two_photos(), {(0, "S"): "tank", (1, "S"): "drum"})
        self.assertEqual(self.run_two_photos(vote_align=False), {})


if __name__ == "__main__":
    unittest.main()
