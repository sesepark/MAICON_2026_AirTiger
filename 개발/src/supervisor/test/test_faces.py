import math
import random
import unittest

from sim_world import FakePerception, free_poses, load_layout, scout_tour
from supervisor import arena as A
from supervisor.faces import CameraParams, FaceBook, expected_faces, match


class ExpectedFacesTests(unittest.TestCase):
    def test_single_box_in_front_is_centered(self):
        fmap = A.field_map([(3.0, 2.5)])
        seen = expected_faces(fmap, (2.0, 2.5, 0.0))  # 상자의 서쪽 면을 1 m 앞 정면에서
        self.assertEqual([e.face.key for e in seen], [(0, "W")])
        self.assertTrue(seen[0].sure)
        self.assertAlmostEqual(seen[0].x, 0.5, places=2)
        self.assertLess(seen[0].incidence_deg, 1.0)

    def test_left_to_right_order(self):
        fmap = A.field_map([(3.0, 3.3), (3.0, 1.7)])  # 왼쪽(위) 상자, 오른쪽(아래) 상자
        seen = expected_faces(fmap, (1.6, 2.5, 0.0))
        self.assertEqual([e.face.key for e in seen if e.face.side == "W"], [(0, "W"), (1, "W")])
        self.assertLess(seen[0].x, seen[-1].x)

    def test_hidden_face_is_not_expected(self):
        fmap = A.field_map([(3.0, 2.5), (4.0, 2.5)])
        keys = [e.face.key for e in expected_faces(fmap, (2.0, 2.5, 0.0))]
        self.assertIn((0, "W"), keys)
        self.assertNotIn((1, "W"), keys)  # 앞 상자에 가렸다

    def test_pan_turns_the_view(self):
        fmap = A.field_map([(3.0, 2.5)])
        ahead = expected_faces(fmap, (2.0, 2.5, 0.0))[0]
        panned = expected_faces(fmap, (2.0, 2.5, 0.0), pan_deg=20.0)[0]  # 카메라를 오른쪽으로 → 면은 사진 왼쪽으로
        self.assertLess(panned.x, ahead.x)


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.fmap = A.field_map([(3.0, 3.6), (3.0, 2.5), (3.0, 1.4)])
        self.exp = [e for e in expected_faces(self.fmap, (1.5, 2.5, 0.0))]
        self.sure = [e for e in self.exp if e.sure]

    def test_exact_answer_is_paired_in_order(self):
        pics = [("tank", e.x, False) for e in self.sure]
        pairs = match(self.exp, pics)
        self.assertEqual([e.face.key for e, _k in pairs], [e.face.key for e in self.sure])

    def test_missing_sure_face_discards_the_photo(self):
        pics = [("tank", e.x, False) for e in self.sure[1:]]
        self.assertIsNone(match(self.exp, pics))

    def test_extra_picture_discards_the_photo(self):
        pics = [("tank", e.x, False) for e in self.sure] + [("drum", 0.97, False)]
        self.assertIsNone(match(self.exp, pics))

    def test_optional_face_may_be_absent(self):
        maybe = [e for e in self.exp if not e.sure]
        pics = [("tank", e.x, False) for e in self.sure]
        pairs = match(self.exp, pics)
        self.assertIsNotNone(pairs)
        self.assertTrue(all(e.sure for e, _k in pairs) or not maybe)


class FaceBookTests(unittest.TestCase):
    def test_majority_and_counts(self):
        fmap, truth = load_layout()
        book = FaceBook(fmap, "BLUE")
        face = next(f for f in book.goal)
        near = expected_faces(fmap, (face.cx + face.nx * 0.8, face.cy + face.ny * 0.8, math.atan2(-face.ny, -face.nx)))
        exp = next(e for e in near if e.face.key == face.key)
        book.add(exp, "tank")
        book.add(exp, "tank")
        book.add(exp, "drum")
        self.assertEqual(book.kind(face.key), "tank")
        self.assertEqual(book.counts()[face.sector], {"tank": 1})
        self.assertNotIn(face.key, book.remaining())
        book.add(exp, A.UNKNOWN)
        self.assertEqual(book.kind(face.key), "tank")

    def test_snapshot_roundtrip(self):
        fmap, _truth = load_layout()
        book = FaceBook(fmap, "RED")
        face = book.goal[0]
        exp = expected_faces(fmap, (face.cx + face.nx * 0.8, face.cy + face.ny * 0.8, math.atan2(-face.ny, -face.nx)))
        book.add(next(e for e in exp if e.face.key == face.key), "mortar")
        again = FaceBook(fmap, "RED")
        again.restore(book.snapshot())
        self.assertEqual(again.counts(), book.counts())


def evaluate(n=600, seed=1, pose_xy=0.03, pose_yaw_deg=2.0, assumed_hfov=120.0, true_hfov=128.0):
    """임의의 자리에서 찍은 사진 n 장: 위치 오차와 화각 오차가 있을 때 짝짓기가 얼마나 받아들이고 얼마나 틀리나.

    세는 것은 면 기록에 실제로 적히는 짝만이다 (FaceBook 의 적는 범위 안, 모름 제외).
    """
    fmap, truth = load_layout()
    eyes = FakePerception(fmap, truth, CameraParams(hfov_deg=true_hfov), seed=seed)
    cam = CameraParams(hfov_deg=assumed_hfov)
    rng = random.Random(seed + 1)
    photos = accepted = pairs_n = wrong = 0
    for pose in free_poses(fmap, n, seed):
        pics, keys = eyes.look(pose)
        if not pics:
            continue
        photos += 1
        noisy = (pose[0] + rng.gauss(0, pose_xy), pose[1] + rng.gauss(0, pose_xy), pose[2] + math.radians(rng.gauss(0, pose_yaw_deg)))
        got = match(expected_faces(fmap, noisy, 0.0, cam), pics)
        if got is None:
            continue
        accepted += 1
        left = list(zip(pics, keys))
        for exp, kind in got:
            if kind == A.UNKNOWN or exp.dist > FaceBook.record_range_m or exp.incidence_deg > FaceBook.record_incidence_deg:
                continue
            pairs_n += 1
            # 같은 종류의 그림 중 그 면에 실제로 붙어 있던 것이 있어야 맞다
            if not any(key == exp.face.key and pk == kind for (pk, _x, _c), key in left):
                wrong += 1
    return {"photos": photos, "accepted": accepted, "pairs": pairs_n, "wrong": wrong}


class MatchUnderNoiseTests(unittest.TestCase):
    def test_exact_pose_never_mislabels(self):
        r = evaluate(n=400, pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0)
        self.assertEqual(r["wrong"], 0, r)
        self.assertGreater(r["accepted"], 0.9 * r["photos"], r)

    def test_noisy_pose_and_wrong_fov_rarely_mislabels(self):
        r = evaluate(n=600)  # 위치 3 cm · 2°, 화각을 120° 로 알고 있는데 실제는 128°
        self.assertLess(r["wrong"], 0.01 * max(1, r["pairs"]), r)
        self.assertGreater(r["accepted"], 0.8 * r["photos"], r)


class ScoutTourTests(unittest.TestCase):
    def test_exact_tour_gets_all_four_sectors(self):
        fmap, truth = load_layout()
        for camp, start in (("BLUE", (0.2, 2.5, 0.0)), ("RED", (5.9, 2.5, math.pi))):
            r = scout_tour(fmap, truth, camp, start, pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0)
            self.assertEqual((r["sectors_ok"], r["faces_wrong"], r["left"]), (4, 0, 0), r)


if __name__ == "__main__":
    for kw in ({"pose_xy": 0.0, "pose_yaw_deg": 0.0, "assumed_hfov": 128.0}, {}, {"pose_xy": 0.05, "pose_yaw_deg": 4.0}):
        print(kw or "기본(위치 3cm · 2도, 화각 120 가정 / 실제 128)", evaluate(**kw))
    unittest.main()
