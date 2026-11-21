"""본 것 (표): 겨누지 않았어도 사진에 또렷이 찍힌 면을, 다른 자리에서 찍은 여러 장이 같은 답일 때 적는다.

scout.pick_votes (사진 한 장에서 「본 것」 고르기) · FaceBook.sight (모아서 적기) · FaceScout.on_answer / on_pass · mission 의 지나가며 찍은 사진 거르기.
"""
import math
import unittest

from supervisor import arena as A
from supervisor.faces import CameraParams, FaceBook, expected_faces
from supervisor.grid import Grid
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import FaceScout, ScoutParams, pick_votes

CAM = CameraParams()
BACKDROP = (A.NON_OBJECT, 0.03, False, 0.05, 0.05, 0.9)  # 면이 없는 벽 그림 — 실차의 답에는 거의 늘 섞여 있다


def sized(e, kind, dx=0.0, scale=1.0, cut=False, conf=0.9):
    """면 e 가 찍힐 자리 · 크기대로 만든 그림 (실차의 답 모양: 종류, 가로 자리, 잘림, 가로 크기, 세로 크기, 확신)."""
    return (kind, e.x + dx, cut, e.width * scale, e.height * scale, conf)


class VoteBase(unittest.TestCase):
    """RED 쪽 구역에 떨어져 놓인 상자 둘 (0번 · 1번)의 남쪽 면을 아래에서 본다. 우리는 BLUE — 두 상자의 면이 채점 면이다."""

    def setUp(self):
        self.fmap = A.field_map([(4.3, 3.6), (5.0, 3.6)])
        self.book = FaceBook(self.fmap, "BLUE")
        self.scout = FaceScout(self.fmap, None, self.book, CAM, ScoutParams())
        self.a, self.b = (0, "S"), (1, "S")
        self.assertIn(self.a, {f.key for f in self.book.goal})
        self.poses = [(4.5, 2.2, math.radians(90.0)), (4.9, 2.4, math.radians(100.0)), (4.2, 2.0, math.radians(80.0))]

    def exp(self, pose, pan=0.0):
        return {e.face.key: e for e in expected_faces(self.fmap, pose, pan, CAM)}

    def shot(self, n, kinds, at=None, pass_=True, pan=0.0, **kw):
        """poses[n] 에서 찍은 사진에 kinds = {면 key: 종류} 가 제자리 · 제 크기로 찍혔다."""
        pose = self.poses[n]
        by = self.exp(pose, pan)
        pics = [BACKDROP] + [sized(by[k], kind, **kw) for k, kind in kinds.items()]
        if pass_:
            return self.scout.on_pass(("pass", n), pose, pan, pics, at=float(n) if at is None else at)
        return pics


class PickVotesTests(VoteBase):
    def test_clear_picture_is_a_vote_and_unclear_ones_are_not(self):
        pose, p = self.poses[0], ScoutParams()
        by = self.exp(pose)
        e = by[self.a]
        expected = list(by.values())
        self.assertEqual([(x.face.key, k) for x, k in pick_votes(expected, [sized(e, "tank")], p)], [(self.a, "tank")])
        self.assertEqual(pick_votes(expected, [sized(e, "tank", cut=True)], p), [])  # 가장자리에서 잘림
        self.assertEqual(pick_votes(expected, [sized(e, "tank", conf=0.7)], p), [])  # 확신이 낮음
        self.assertEqual(pick_votes(expected, [sized(e, A.UNKNOWN)], p), [])  # 모름
        self.assertEqual(pick_votes(expected, [sized(e, "tank", scale=0.5)], p), [])  # 크기가 안 맞음 (먼 상자의 그림)
        self.assertEqual(pick_votes(expected, [sized(e, "tank", dx=0.07)], p), [])  # 자리가 안 맞음
        self.assertEqual(pick_votes(expected, [("tank", e.x, False)], p), [])  # 크기 없는 답
        self.assertEqual(pick_votes(expected, [("tank", e.x, False, e.width, e.height)], p), [])  # 확신 없는 답

    def test_too_far_or_too_oblique_is_not_a_vote(self):
        by = self.exp(self.poses[0])
        e = by[self.a]
        self.assertEqual(pick_votes(list(by.values()), [sized(e, "tank")], ScoutParams(vote_range_m=e.dist - 0.1)), [])
        self.assertEqual(pick_votes(list(by.values()), [sized(e, "tank")], ScoutParams(vote_incidence_deg=max(0.0, e.incidence_deg - 1.0))), [])

    def test_two_faces_at_the_same_place_give_no_vote_even_when_only_one_fits_by_size(self):
        # 나란히 붙은 두 상자를 멀리서 비스듬히 보면 두 면이 사진의 거의 같은 자리에 찍힌다 → 크기가 한쪽에만 맞아도 치지 않는다
        fmap = A.field_map([(4.3, 3.6), (4.7, 3.6)])
        pose = (2.6, 2.9, math.radians(20.0))
        exp = expected_faces(fmap, pose, 0.0, CAM)
        by = {e.face.key: e for e in exp}
        near, far = by[(0, "S")], by[(1, "S")]
        self.assertLess(abs(near.x - far.x), ScoutParams().vote_sep)
        self.assertEqual(pick_votes(exp, [sized(far, "missile")], ScoutParams(vote_range_m=4.0, vote_incidence_deg=80.0, vote_fraction=0.2)), [])


class SightTests(VoteBase):
    def test_single_vote_records_nothing(self):
        self.assertEqual(self.shot(0, {self.a: "tank"}), [])
        self.assertIsNone(self.book.kind(self.a))
        self.assertIn(self.a, self.book.remaining())

    def test_two_agreeing_votes_from_different_poses_record_the_face(self):
        self.shot(0, {self.a: "tank", self.b: A.NON_OBJECT})
        self.assertEqual(sorted(self.shot(1, {self.a: "tank", self.b: A.NON_OBJECT})), [(self.a, "tank"), (self.b, A.NON_OBJECT)])
        self.assertEqual((self.book.kind(self.a), self.book.kind(self.b)), ("tank", A.NON_OBJECT))  # 비객체도 적힌다 (그 면은 끝난 것이다)
        self.assertNotIn(self.a, self.book.remaining())
        self.assertEqual(self.book.voted, {self.a: "tank", self.b: A.NON_OBJECT})
        self.assertEqual(self.book.counts()[self.book.by_key[self.a].sector], {"tank": 1})
        self.assertEqual(self.scout.voted, 2)

    def test_same_photo_twice_is_one_vote(self):
        # 같은 자리 · 같은 방향 · 같은 때의 사진 두 장은 한 장으로 친다
        self.shot(0, {self.a: "tank"}, at=5.0)
        self.assertEqual(self.shot(0, {self.a: "tank"}, at=5.1), [])
        self.assertIsNone(self.book.kind(self.a))
        self.assertEqual(self.shot(0, {self.a: "tank"}, at=6.0), [(self.a, "tank")])  # 같은 자리라도 0.3초 넘게 떨어진 다른 사진이면 친다

    def test_votes_without_a_time_need_different_poses(self):
        self.scout.p.vote_apart = (0.10, 5.0, 0.0)
        self.shot(0, {self.a: "tank"}, at=5.0)
        self.assertEqual(self.shot(0, {self.a: "tank"}, at=9.0), [])
        self.assertEqual(self.shot(1, {self.a: "tank"}, at=9.5), [(self.a, "tank")])

    def test_disagreeing_vote_blocks(self):
        self.shot(0, {self.a: "tank"})
        self.assertEqual(self.shot(1, {self.a: "mortar"}), [])
        self.assertEqual(self.shot(2, {self.a: "tank"}), [])  # 한 번 갈린 면은 「본 것」 으로는 다시 적지 않는다 (겨눠 찍어야 한다)
        self.assertIsNone(self.book.kind(self.a))
        self.assertIn(self.a, self.book.remaining())

    def test_disagreeing_vote_after_recording_takes_the_record_back(self):
        self.shot(0, {self.a: "tank"})
        self.shot(1, {self.a: "tank"})
        self.assertEqual(self.book.kind(self.a), "tank")
        self.shot(2, {self.a: "mortar"})
        self.assertIsNone(self.book.kind(self.a))
        self.assertIn(self.a, self.book.remaining())
        self.assertEqual(self.book.voted, {})

    def test_aimed_record_is_not_overwritten_by_votes(self):
        pose = self.poses[0]
        aim = -math.degrees(math.atan2(self.book.by_key[self.a].cy - pose[1], self.book.by_key[self.a].cx - pose[0]) - pose[2])
        by = self.exp(pose, aim)
        self.scout.on_answer(("face", self.a, 0), pose, aim, [BACKDROP, sized(by[self.a], "tank")])
        self.assertEqual(self.book.kind(self.a), "tank")
        before = dict(self.book.votes[self.a])
        self.shot(1, {self.a: "mortar"})
        self.shot(2, {self.a: "mortar"})
        self.assertEqual(self.book.kind(self.a), "tank")
        self.assertEqual(self.book.votes[self.a], before)
        self.assertNotIn(self.a, self.book.voted)

    def test_aimed_answer_beats_an_earlier_vote_record(self):
        self.shot(0, {self.a: "tank"})
        self.shot(1, {self.a: "tank"})
        self.assertEqual(self.book.kind(self.a), "tank")
        pose = self.poses[0]
        aim = -math.degrees(math.atan2(self.book.by_key[self.a].cy - pose[1], self.book.by_key[self.a].cx - pose[0]) - pose[2])
        by = self.exp(pose, aim)
        self.scout.on_answer(("face", self.a, 0), pose, aim, [BACKDROP, sized(by[self.a], "mortar")])
        self.assertEqual(self.book.kind(self.a), "mortar")
        self.assertNotIn(self.a, self.book.voted)

    def test_aimed_photos_vote_for_the_other_faces_in_view(self):
        # 겨눠 찍은 사진 두 장(다른 자리)에 함께 찍힌 면: 겨눈 면이 안 읽혀 덤으로는 못 적어도, 두 장이 같은 답이면 적는다
        for n in (0, 1):
            pose = self.poses[n]
            by = self.exp(pose)
            self.scout.shot_taken(("face", self.a, n), 10.0 * n)
            self.scout.on_answer(("face", self.a, n), pose, 0.0, [BACKDROP, sized(by[self.b], "drum")])
        self.assertIsNone(self.book.kind(self.a))
        self.assertEqual(self.book.kind(self.b), "drum")
        self.assertEqual(self.book.voted, {self.b: "drum"})

    def test_switch_off(self):
        self.scout.p.vote_on = False
        self.shot(0, {self.a: "tank"})
        self.assertEqual(self.shot(1, {self.a: "tank"}), [])
        self.assertIsNone(self.book.kind(self.a))
        self.scout.p.vote_on, self.scout.p.vote_pass = True, False
        self.assertEqual(self.shot(2, {self.a: "tank"}), [])
        self.assertEqual(self.book.sights, {})

    def test_snapshot_keeps_the_vote_record(self):
        self.shot(0, {self.a: "tank"})
        self.shot(1, {self.a: "tank"})
        again = FaceBook(self.fmap, "BLUE")
        again.restore(self.book.snapshot())
        self.assertEqual((again.kind(self.a), again.voted), ("tank", {self.a: "tank"}))


class UnsteadyPassTests(VoteBase):
    """찍을 때 방향이 조금 흔들린 사진 (on_pass 의 steady=False): 그림 하나를 따로 붙인 것은 쓰지 않고, 사진의 그림들을 한꺼번에 맞춘 것만 쓴다."""

    def both(self, n, steady, dx=0.03):
        pose = self.poses[n]
        by = self.exp(pose)
        pics = [sized(by[self.a], "tank", dx=dx), sized(by[self.b], A.NON_OBJECT, dx=dx)]
        return self.scout.on_pass(("pass", n), pose, 0.0, pics, at=float(n), steady=steady)

    def test_single_picture_is_not_a_vote(self):
        for n in (0, 1):
            pose = self.poses[n]
            self.scout.on_pass(("pass", n), pose, 0.0, [BACKDROP, sized(self.exp(pose)[self.a], "tank")], at=float(n), steady=False)
        self.assertEqual(self.book.sights, {})
        self.shot(0, {self.a: "tank"})  # 같은 그림도 흔들리지 않은 사진이면 「본 것」 이다
        self.assertEqual(len(self.book.sights[self.a]), 1)

    def test_pictures_matched_together_are_votes(self):
        self.assertEqual(self.both(0, steady=False), [])
        self.assertEqual(sorted(self.both(1, steady=False)), [(self.a, "tank"), (self.b, A.NON_OBJECT)])
        self.assertEqual(self.book.voted, {self.a: "tank", self.b: A.NON_OBJECT})

    def test_nothing_is_used_when_the_whole_photo_match_is_off(self):
        self.scout.p.vote_align = False
        self.both(0, steady=False, dx=0.0)
        self.both(1, steady=False, dx=0.0)
        self.assertEqual(self.book.sights, {})


class StopSkipTests(unittest.TestCase):
    """찍을 면이 모두 적힌 자리는 가지 않는다 (겨눠 적었든 「본 것」 으로 적었든)."""

    def setUp(self):
        self.fmap = A.field_map([(4.3, 3.6), (5.0, 1.2)])
        self.book = FaceBook(self.fmap, "BLUE")
        self.scout = FaceScout(self.fmap, Grid(self.fmap), self.book, CAM, ScoutParams())
        self.pose = (3.2, 2.4, 0.0)

    def vote(self, key, kind="tank"):
        for n in range(2):
            self.book.sight(key, kind, (1.0 + n, 1.0, 0.0, float(n)))
        self.assertEqual(self.book.kind(key), kind)

    def test_stop_is_dropped_once_all_its_faces_are_recorded_by_votes(self):
        act = self.scout.step(self.pose, 0.0)
        self.assertEqual(act[0], "go")
        first = self.scout.stops[self.scout.i]
        for key in first[3]:
            self.vote(key)
        nxt = self.scout.step(self.pose, 0.1)
        self.assertEqual(nxt[0], "go")
        self.assertNotEqual((nxt[1], nxt[2]), (act[1], act[2]), "면이 모두 적힌 자리로는 가지 않는다")
        self.assertNotEqual(self.scout.stops[self.scout.i], first)

    def test_face_recorded_on_the_way_is_not_photographed_at_the_stop(self):
        stop = next(s for s in (self.scout.step(self.pose, 0.0), self.scout.stops[self.scout.i]) if len(s) == 4)
        self.scout.stops = [(stop[0], stop[1], stop[2], sorted(stop[3]))] + [(stop[0], stop[1], stop[2], [k for k in sorted(self.book.remaining()) if k not in stop[3]][:1])]
        keys = [k for k in self.scout.stops[0][3]] + self.scout.stops[1][3]
        self.scout.stops, self.scout.i, self.scout.queue, self.scout.body = [(stop[0], stop[1], stop[2], keys)], 0, [], None
        self.assertGreaterEqual(len(keys), 2)
        self.assertEqual(self.scout.step(self.pose, 0.0)[0], "go")  # 이 자리에서 찍을 면들이 정해졌다 (queue)
        self.vote(keys[0])  # 가는 동안 한 면이 「본 것」 으로 적혔다
        at = (stop[0], stop[1], self.scout.body)
        shots = []
        for n in range(6):
            act = self.scout.step(at, 1.0 + n)
            if act[0] != "shoot":
                break
            shots.append(act[2][1])
            self.scout.shot_taken(act[2], 1.0 + n)
        self.assertNotIn(keys[0], shots)

    def test_scout_finishes_when_everything_is_recorded_by_votes(self):
        self.scout.step(self.pose, 0.0)
        for f in self.book.goal:
            self.vote(f.key, A.NON_OBJECT)
        self.assertEqual(self.scout.step(self.pose, 0.1), ("done",))


class Ports:
    def __init__(self):
        self.aims, self.passes, self.shots, self.goal = [], [], [], None

    def go(self, x, y, yaw, **kw):
        self.goal = (x, y, yaw)
        return True

    def stop(self):
        self.goal = None

    def aim(self, pan_deg, tilt_deg):
        self.aims.append((pan_deg, tilt_deg))
        return True

    def aim_ready(self): return True
    def fire(self): return True

    def shoot(self, tag, pan_deg):
        self.shots.append((tag, pan_deg))
        return True

    def shoot_pass(self, tag):
        self.passes.append(tag)
        return True

    def look_ugvs(self): return False
    def analyze_order(self, text): return True
    def set_ready(self): return True
    def report(self, kind, payload): return True


class MissionPassTests(unittest.TestCase):
    """mission: 지나가며 찍은 사진의 답을 걸러서 「본 것」 으로 넘긴다."""

    BOXES = [(4.3, 3.6), (5.0, 3.6)]
    KEY = (0, "S")

    def setUp(self):
        self.fmap = A.field_map(self.BOXES)
        self.mission = Mission(MissionParams(pass_gap_s=1.0))
        self.ports = Ports()
        self.now = 0.0
        self.pose = (4.3, 1.1, math.radians(90.0))  # 0번 상자의 남쪽 면을 정면으로 보며 다가간다 (2.3 m → 1.5 m)

    def tick(self, answers=(), pose=None):
        self.now += 0.1
        if pose is not None:
            self.pose = pose
        self.mission.tick(Facts(now=self.now, t=1.0 + self.now, phase="running", control=True, slot="B-2", pose=self.pose, arrived=False, fmap=self.fmap,
                                answers=list(answers)), self.ports)

    def drive(self, seconds, turn_deg=0.0):
        """곧게 가는 중 (turn_deg: 한 틱에 도는 각도). 0.1 m/s."""
        for _ in range(int(round(seconds * 10))):
            x, y, yaw = self.pose
            self.tick(pose=(x + 0.01 * math.cos(yaw), y + 0.01 * math.sin(yaw), yaw + math.radians(turn_deg)))

    def next_pass(self):
        """다음 지나가며 찍은 사진이 찍힐 때까지 곧게 간다 → (tag, 찍을 때의 위치)."""
        before = len(self.ports.passes)
        for _ in range(40):
            self.drive(0.1)
            if len(self.ports.passes) > before:
                return self.ports.passes[-1], self.pose
        self.fail("지나가며 찍기가 되지 않는다")

    def answer(self, tag, pose, kind="tank"):
        e = next(e for e in expected_faces(self.fmap, pose, 0.0, CAM) if e.face.key == self.KEY)
        return (tag, pose, 0.0, [BACKDROP, sized(e, kind)])

    def answer_two(self, tag, pose, dx=0.02):
        """두 상자의 남쪽 면이 함께, 같은 만큼(dx) 밀려 찍힌 사진의 답 (방향 값이 조금 늦은 사진 — 한꺼번에 맞추기로 붙는다)."""
        by = {e.face.key: e for e in expected_faces(self.fmap, pose, 0.0, CAM)}
        return (tag, pose, 0.0, [sized(by[(0, "S")], "tank", dx=dx), sized(by[(1, "S")], "drum", dx=dx)])

    def shoot_two(self, wiggle=False, turn_deg=5.0, ticks=3, two=False):
        """지나가며 두 장을 찍고 답을 받는다 (서로 다른 자리). wiggle = 찍은 직후에 차체가 돈다 (찍을 때 막 돌기 시작했다) — 한 틱에 turn_deg 씩 ticks 틱.
        two = 두 면이 함께 찍힌 답 (answer_two)."""
        for _ in range(2):
            tag, pose = self.next_pass()
            self.tick([self.answer_two(tag, pose) if two else self.answer(tag, pose)])
            if wiggle:
                self.drive(0.1 * ticks, turn_deg=turn_deg)
                self.drive(0.1 * ticks, turn_deg=-turn_deg)
            self.drive(0.8)

    def test_pan_is_sent_to_the_middle_once_at_the_start(self):
        self.mission.p.pass_forward = False  # 「가는 동안 정면으로」 는 아래에서 따로 본다
        self.drive(1.0)
        self.assertEqual(self.ports.aims, [(0.0, 0.0)])

    def test_camera_is_turned_forward_once_per_far_goal(self):
        """정찰 자리로 가는 동안 카메라를 정면으로 돌려 둔다 — 목표마다 한 번만 (매 틱 보내면 팬틸트가 「도는 중」 으로 남아 사진을 못 찍는다)."""
        self.drive(3.0)
        self.assertTrue(self.ports.aims)
        self.assertTrue(all(a == (0.0, 0.0) for a in self.ports.aims))
        self.assertLessEqual(len(self.ports.aims), 2)  # 경기 첫머리 한 번 + 첫 목표로 가며 한 번

    def test_pass_photos_record_a_face_after_the_pan_is_known(self):
        self.drive(3.0)  # 팬을 보내고 pan_home_s 가 지났다
        self.shoot_two()
        self.assertEqual(self.mission.book.kind(self.KEY), "tank")
        self.assertIn(self.KEY, self.mission.book.voted)
        self.assertTrue(any("지나가며 찍은 사진들로 면을 적음" in text for _t, text in self.mission.log), self.mission.log)

    def test_pass_photo_before_the_first_pan_command_is_ignored(self):
        self.ports.aim = lambda pan, tilt: False  # 팬틸트 창구가 아직 안 된다 → 팬을 한 번도 보내지 못했다
        self.drive(3.0)
        self.shoot_two()
        self.assertIsNone(self.mission.book.kind(self.KEY))
        self.assertEqual(self.mission.book.sights, {})

    def test_pass_photo_right_after_the_pan_command_is_ignored(self):
        # 팬을 보낸 직후(pan_home_s 안)에 찍은 사진: 팬이 아직 돌고 있을 수 있다
        self.mission.p.pan_home_s = 60.0
        self.drive(3.0)
        self.shoot_two()
        self.assertEqual(self.mission.book.sights, {})

    def test_pass_photo_taken_while_turning_is_ignored(self):
        self.drive(3.0)
        self.mission.p.pass_turn_deg = 90.0  # 찍기 전 검사는 통과시키고, 답을 쓸 때의 검사만 본다
        self.shoot_two(wiggle=True)
        self.assertEqual(self.mission.book.sights, {})
        self.assertGreaterEqual(len(self.ports.passes), 2)

    def test_slightly_unsteady_pass_photo_is_used_only_through_the_whole_photo_match(self):
        """방향이 6° 넘게, 15° 까지 흔들린 사진: 버리지 않고 「한꺼번에 맞추기」 로 붙은 그림만 쓴다 (2026-11-19 16:01 실차: 133장 중 61장이 6° 에 걸렸다)."""
        self.drive(3.0)
        self.mission.p.pass_turn_deg = 90.0
        self.shoot_two(wiggle=True, turn_deg=4.0, ticks=2, two=True)  # 찍은 뒤 8° 까지 돌았다가 돌아옴
        self.assertEqual(self.mission.book.voted, {(0, "S"): "tank", (1, "S"): "drum"})

    def test_slightly_unsteady_pass_photo_with_one_picture_is_not_used(self):
        self.drive(3.0)
        self.mission.p.pass_turn_deg = 90.0
        self.shoot_two(wiggle=True, turn_deg=4.0, ticks=2)  # 그림 하나를 면 하나에 따로 붙이는 규칙은 사진의 밀림을 모른다 → 쓰지 않는다
        self.assertEqual(self.mission.book.sights, {})

    def test_pass_photo_taken_while_turning_is_ignored_even_with_a_whole_photo_match(self):
        self.drive(3.0)
        self.mission.p.pass_turn_deg = 90.0
        self.shoot_two(wiggle=True, turn_deg=8.0, ticks=3, two=True)  # 24° — pass_vote_turn_align_deg 를 넘는다
        self.assertEqual(self.mission.book.sights, {})

    def test_unsteady_limit_can_be_switched_off(self):
        self.drive(3.0)
        self.mission.p.pass_turn_deg = 90.0
        self.mission.p.pass_vote_turn_align_deg = 0.0  # 전처럼: 6° 를 넘으면 버린다
        self.shoot_two(wiggle=True, turn_deg=4.0, ticks=2, two=True)
        self.assertEqual(self.mission.book.sights, {})

    def test_steady_pass_photo_with_two_faces_is_recorded(self):
        self.drive(3.0)
        self.shoot_two(two=True)
        self.assertEqual(self.mission.book.voted, {(0, "S"): "tank", (1, "S"): "drum"})

    def test_single_pass_photo_records_nothing(self):
        self.drive(3.0)
        tag, pose = self.next_pass()
        self.tick([self.answer(tag, pose)])
        self.drive(1.0)
        self.assertEqual(len(self.mission.book.sights.get(self.KEY, [])), 1)
        self.assertIsNone(self.mission.book.kind(self.KEY))

    def test_answer_for_a_photo_we_did_not_take_is_ignored(self):
        self.drive(3.0)
        self.tick([self.answer(("pass", 999), self.pose)])
        self.drive(1.0)
        self.assertEqual(self.mission.book.sights, {})


class SightedCountTests(VoteBase):
    """한 장뿐인 「본 것」 (FaceBook.sighted): 표가 아니므로 정찰 계획 · 지정표적에는 끼지 않고, M3 개수(counts)에만 들어간다 (다른 답이 없을 때)."""

    def sector(self, key):
        return self.book.by_key[key].sector

    def test_single_sighting_is_counted_but_not_voted(self):
        self.shot(0, {self.a: "tank"})
        self.assertIsNone(self.book.kind(self.a))
        self.assertEqual(self.book.sighted(self.a), "tank")
        self.assertIn(self.a, self.book.remaining())  # 정찰은 그 면을 여전히 보러 간다
        self.assertEqual(self.book.counts()[self.sector(self.a)], {"tank": 1})

    def test_switch_off(self):
        self.shot(0, {self.a: "tank"})
        FaceBook.count_sighted = False
        try:
            self.assertEqual(self.book.counts()[self.sector(self.a)], {})
        finally:
            FaceBook.count_sighted = True

    def test_split_sightings_are_not_counted(self):
        self.shot(0, {self.a: "tank"})
        self.shot(1, {self.a: "mortar"})
        self.assertIsNone(self.book.sighted(self.a))
        self.assertEqual(self.book.counts()[self.sector(self.a)], {})

    def test_vote_wins_over_sighting(self):
        self.shot(0, {self.a: "tank"})
        pose = self.poses[1]
        aim = -math.degrees(math.atan2(self.book.by_key[self.a].cy - pose[1], self.book.by_key[self.a].cx - pose[0]) - pose[2])
        by = self.exp(pose, aim)
        self.scout.on_answer(("face", self.a, 0), pose, aim, [BACKDROP, sized(by[self.a], "mortar")])
        self.assertIsNone(self.book.sighted(self.a))  # 표가 생긴 면은 kind 로 본다
        self.assertEqual(self.book.counts()[self.sector(self.a)], {"mortar": 1})

    def test_gone_box_is_not_counted_from_a_sighting(self):
        self.shot(0, {self.a: "tank"})
        self.fmap.set_gone({self.a[0]})
        self.assertEqual(self.book.counts()[self.sector(self.a)], {})

    def test_single_sighted_target_is_not_a_target(self):
        target = next(k for k in sorted(A.TARGET_KINDS))
        self.shot(0, {self.a: target})
        self.assertEqual(self.book.targets(), [])  # 지정표적은 「본 것」 으로 적혔을 때(두 장)부터 — 한 장으로 사격이 끼어들지 않는다
        self.assertEqual(self.book.counts()[self.sector(self.a)], {})  # 지정표적은 객체가 아니라 세지 않는다

    def test_snapshot_keeps_single_sightings(self):
        self.shot(0, {self.a: "tank"})
        again = FaceBook(self.fmap, "BLUE")
        again.restore(self.book.snapshot())
        self.assertEqual(again.sighted(self.a), "tank")
        self.assertEqual(again.counts(), self.book.counts())
        # 다시 켜진 뒤 다른 자리의 같은 답 한 장이 오면 「본 것」 으로 적힌다 (전과 이어진다)
        x, y, h = self.book.sights[self.a][0][1:4]
        self.assertEqual(again.sight(self.a, "tank", (x + 0.5, y, h, 99.0)), "tank")

    def test_restore_of_an_older_snapshot_without_sightings(self):
        self.shot(0, {self.a: "tank"})
        data = self.book.snapshot()
        del data["sights"]
        data["votes"] = {}
        again = FaceBook(self.fmap, "BLUE")
        again.restore(data)
        self.assertEqual(again.sights, {})
        again.restore({"sights": {"0:S": [["tank", 1.0]], "9:Q": [["tank", 1.0, 1.0, 0.0, None]]}})  # 모양이 틀린 줄 · 없는 면은 버린다
        self.assertEqual(again.sights, {})


class DeferSightedTests(unittest.TestCase):
    """defer_sighted (기본 끔): 찍을 면이 모두 한 장짜리 「본 것」 인 자리는 그 자리를 처음 잡을 때 맨 뒤로 미룬다."""

    def setUp(self):
        self.fmap = A.field_map([(4.3, 3.6), (5.0, 1.2)])
        self.book = FaceBook(self.fmap, "BLUE")
        self.scout = FaceScout(self.fmap, Grid(self.fmap), self.book, CAM, ScoutParams(defer_sighted=True))
        self.pose = (3.2, 2.4, 0.0)

    def test_default_is_off(self):
        self.assertFalse(ScoutParams().defer_sighted)

    def test_stop_with_only_sighted_faces_goes_to_the_back_once(self):
        self.scout.plan(self.pose)
        first = self.scout.stops[0]
        for key in first[3]:
            self.book.sight(key, "tank", (1.0, 1.0, 0.0, 0.0))
        act = self.scout.step(self.pose, 0.0)
        self.assertEqual(act[0], "go")
        self.assertNotEqual(self.scout.stops[self.scout.i], first)
        self.assertEqual(self.scout.stops[-1], first)
        # 다른 면은 모두 적혔다 → 미룬 자리만 남는다. 거기에 다시 오면 간다 (한 번만 미룬다)
        for f in self.book.goal:
            if f.key not in first[3]:
                for n in range(2):
                    self.book.sight(f.key, A.NON_OBJECT, (1.0 + n, 1.0, 0.0, float(n)))
        self.scout.body, self.scout.queue = None, []
        act = self.scout.step(self.pose, 1.0)
        self.assertEqual((act[0], act[1], act[2]), ("go", first[0], first[1]))

    def test_goal_already_sent_is_not_changed(self):
        act = self.scout.step(self.pose, 0.0)
        current = self.scout.stops[self.scout.i]
        for key in current[3]:
            self.book.sight(key, "tank", (1.0, 1.0, 0.0, 0.0))  # 가는 동안 한 장씩 봤다
        again = self.scout.step(self.pose, 0.1)
        self.assertEqual((again[1], again[2]), (act[1], act[2]))  # 가던 자리 그대로 (navigation 의 목표를 바꾸지 않는다)


if __name__ == "__main__":
    unittest.main()
