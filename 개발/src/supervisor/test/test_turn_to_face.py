"""선 자리에서 팬이 닿지 않는 면: 다시 돌 때로 미루기 전에 차체를 그 면 쪽으로 한 번 돌려 본다 (2026-11-19 16:01 팀 테스트 B-1 뒤 — maicon-match 20261119_160134).

그 판(코드 d74b578)은 M3 가 30점 중 16점이었다. 틀린 두 구역은 면 하나씩이 모자랐다:
  - R-S1: 상자 2 의 N(미사일). 통로 자리 (3.97, 4.51) 에 4.05 까지 가서 섰고 차체 방향이 20° 쯤(허용 ±20° 의 끝)이었다 → 면이 팬 74° 방향
  - R-S3: 상자 13 의 S(탄약상자). 통로 자리 (4.68, 0.31) 에 4.76 에서 섰고 방향이 −160° 쯤(허용의 끝)이었다 → 면이 팬 62° 방향
찍을 수 있는 한도는 60°(팬 끝 48° + aim_over_deg 12°)라 둘 다 「다시 돌 때」 로 미뤄졌고, 다시 돌러 가던 중 223초에 정찰이 끝났다.
14:52 판에서도 채점 면 26개 가운데 4개가 같은 식으로 빠졌다. 고친 것: 그 자리에서 차체를 그 면 쪽으로 한 번 돌려 보고(FaceScout._turn_to),
그래도 못 겨눌 때만 미룬다. 미룬 것 · 돌린 것은 경기 기록에도 남긴다.

2026-11-19 저녁 (통합 에이전트가 main 에 합치며 고침): 그 뒤 통로 자리가 면에서 더 물러나고(corridor_back 1.25 · 12° 기울임) 팬 끝을 16° 넘는 면까지
찍게 되어, 그 판의 자세(8 cm 어긋남 · 방향 20°)로는 이제 팬이 닿는다. 그래도 같은 날 모의 5차에서 로봇이 통로 자리를 0.1 ~ 0.2 m 지나쳐 서서
면이 64° 밖으로 나가 찍지 못했다 (그 면 하나로 R-S1 7점을 잃었다). 그래서 이 시험의 자세는 「지금의 자리에서 그만큼 지나치고 방향이 틀어진」 것으로 만든다.
"""
import math
import unittest

from sim_world import SimMatch, scout_grid
from supervisor import arena as A
from supervisor.faces import FaceBook
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import PAN_LIMIT_DEG, FaceScout, ScoutParams

# 16:01 판의 perception 지도 (상자 14개)와 그 판의 구역 · 거점 치수
BOXES_1601 = [(1.426, 4.035), (1.857, 3.992), (4.257, 4.02), (4.676, 4.015), (3.098, 3.42), (1.461, 2.415), (1.844, 2.409),
              (4.247, 2.385), (4.63, 2.37), (3.026, 1.506), (1.47, 0.812), (1.861, 0.817), (3.945, 0.819), (4.366, 0.825)]
ARENA = A.ArenaPlan(width=6.12, height=4.8, end_strip=0.821, cm_width=1.2374)
START = (0.37, 4.01, 0.0)  # B-1 출발 칸에서 LiDAR 로 잡은 자리
# (면, 가던 길의 한 자세) — 선 자세는 지금의 통로 자리에서 만든다 (overshot)
CASES = [((2, "N"), (3.40, 4.10, math.radians(22.0))), ((13, "S"), (4.86, 0.75, math.radians(-100.0)))]
STOP_ERROR = 0.07  # 가상 로봇이 멈춘 자리의 오차 σ (m) — 그 판의 통로 자리 셋에서 8 · 8 · 9 cm 였다
TURN_ROOM = 0.23  # 가상 로봇이 서서 돌 수 있는 틈 (차체 반 대각선 0.17 + 선 자리의 오차)


def scout_1601(**switches):
    fmap = A.field_map(BOXES_1601, ARENA)
    book = FaceBook(fmap, "BLUE")
    scout = FaceScout(fmap, scout_grid(fmap), book, params=ScoutParams(**switches), start=START)
    scout.plan(START)
    return scout, book


def at_stop(key, **switches):
    """그 면의 통로 자리만 남긴 정찰. 다시 돌기는 꺼 둔다 (자리 하나만 보려고 — 그 자리를 떠나면 "done")."""
    scout, book = scout_1601(**switches)
    stop = next(s for s in scout.stops if s[3] == [key])
    scout.stops, scout.i, scout.passes = [stop], 0, 2
    return scout, book


def overshot(scout, book, key, far, along=0.14):
    """그 면의 통로 자리로 가다가 along 만큼 지나쳐 서고, 차체가 면의 반대쪽으로 틀어져 팬이 닿지 않게 된 자세. (통로 자리, 선 자세)"""
    act = scout.step(far, 0.0)
    assert act[0] == "go", act
    x, y, heading = act[1], act[2], act[3]
    face = book.by_key[key]
    limit = PAN_LIMIT_DEG - 2.0 + scout.p.aim_over_deg
    px, py = x + along * math.cos(heading), y + along * math.sin(heading)
    to_face = math.atan2(face.cy - py, face.cx - px)
    side = 1.0 if math.sin(to_face - heading) > 0 else -1.0  # 면이 차체의 왼쪽(+)에 있는가
    for extra in range(21, 60):  # 방향 허용(20°)을 조금 넘겨 선 것 — navigation 이 방향을 못 맞춘 채 「도착」 이라 한 때
        here = (px, py, heading - side * math.radians(extra))
        if pan_needed(here, face) > limit + 1.0:
            return (x, y, heading), here
    raise AssertionError("팬이 닿지 않는 자세를 만들지 못함: %s" % (key,))


def off(a, b):
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def pan_needed(pose, face):
    return abs(math.degrees(math.atan2(math.sin(math.atan2(face.cy - pose[1], face.cx - pose[0]) - pose[2]),
                                       math.cos(math.atan2(face.cy - pose[1], face.cx - pose[0]) - pose[2]))))


class ReplayTests(unittest.TestCase):
    """16:01 판의 두 자리를 로봇이 실제로 섰던 자세로 다시 해 본다."""

    def test_without_the_turn_the_face_is_left_for_later_as_in_that_match(self):
        for key, far in CASES:
            scout, book = at_stop(key, turn_to_face=False, retry_nearby=False)
            _spot, here = overshot(scout, book, key, far)
            scout.accept_heading()  # mission: navigation 이 방향을 못 맞춘 채 끝냈다 → 지금 방향을 받아들인다
            act = scout.step(here, 5.0)
            self.assertEqual(act[0], "done", (key, act))  # 찍지도 돌지도 않고 이 자리를 떠났다
            self.assertIn("다시 돌 때로 미룬다", scout.log[-1])
            self.assertEqual((scout.turns, scout.unaimed), (0, 1))
            self.assertIn(key, book.remaining())

    def test_turns_toward_the_face_and_shoots_it(self):
        for key, far in CASES:
            scout, book = at_stop(key)
            face = book.by_key[key]
            _spot, here = overshot(scout, book, key, far)
            scout.accept_heading()
            act = scout.step(here, 5.0)
            self.assertEqual(act[0], "go", (key, act))
            self.assertEqual((act[1], act[2]), (here[0], here[1]))  # 선 자리 그대로 — 돌기만 한다
            self.assertLess(off(act[3], math.atan2(face.cy - here[1], face.cx - here[0])), 0.01)  # 그 면을 똑바로 보는 방향
            self.assertAlmostEqual(math.degrees(act[4]), ScoutParams().turn_tol_deg)
            self.assertIn("팬이 닿지 않음", scout.log[-1])
            self.assertIn("돌려 본다", scout.log[-1])
            self.assertEqual(scout.step(here, 5.1)[:4], act[:4])  # 도는 동안 같은 목표
            side = 1.0 if math.sin(here[2] - act[3]) > 0 else -1.0  # 오던 쪽에서 허용(25°) 바로 안까지만 돌았다
            turned = (here[0], here[1], act[3] + side * math.radians(20.0))
            shot = scout.step(turned, 7.0)
            self.assertEqual((shot[0], shot[2][1]), ("shoot", key))
            self.assertLessEqual(abs(shot[1]), 25.0)
            self.assertEqual((scout.turns, scout.unaimed), (1, 0))

    def test_only_one_turn_per_face_then_it_is_left_for_later(self):
        """돌려 보라고 했는데 navigation 이 못 돌았다 (돌 자리 없음 · 방향을 못 맞춘 채 끝냄) → 한 번 더 시키지 않고 전처럼 다시 돌 때로 미룬다."""
        key, far = CASES[0]
        scout, book = at_stop(key, retry_nearby=False)
        _spot, here = overshot(scout, book, key, far)
        scout.accept_heading()
        self.assertEqual(scout.step(here, 5.0)[0], "go")
        scout.accept_heading()  # mission: 그 방향은 받아들인다 (no_turn · 도착인데 방향이 허용 밖 · 3초 서 있음)
        act = scout.step(here, 5.2)
        self.assertEqual(act[0], "done", act)
        self.assertIn("다시 돌 때로 미룬다", scout.log[-1])
        self.assertEqual((scout.turns, scout.unaimed), (1, 1))
        self.assertIn(key, book.remaining())

    def test_a_corridor_face_that_cannot_be_aimed_is_tried_from_the_other_end_right_away(self):
        """차체를 돌려도 못 겨눈 통로의 면: 맨 뒤로 미루지 않고 통로의 반대편 자리에서 바로 한 번 더 본다 (면마다 한 번)."""
        for key, far in CASES:
            scout, book = at_stop(key)
            spot, here = overshot(scout, book, key, far)
            face = book.by_key[key]
            scout.accept_heading()
            self.assertEqual(scout.step(here, 5.0)[0], "go")  # 먼저 차체를 돌려 본다
            scout.accept_heading()  # 못 돌았다
            act = scout.step(here, 5.2)
            self.assertEqual(act[0], "go", (key, act))
            self.assertIn(key, scout.nearby)
            self.assertIn("바로 가까운 자리", scout.log[-2] + scout.log[-1])
            along = (lambda p: p[0] - face.cx) if abs(face.ny) > 0.5 else (lambda p: p[1] - face.cy)
            self.assertLess(along(spot) * along((act[1], act[2])), 0.0, (key, spot, act))  # 면의 반대편 (통로를 따라)
            scout.accept_heading()
            other = (act[1], act[2], act[3] + math.radians(170.0))  # 거기서도 못 겨눴다 (차체가 거꾸로 섰다)
            scout.any_heading = True
            self.assertEqual(scout.step(other, 20.0)[0], "go")  # 그 자리에서도 한 번은 돌려 본다
            scout.accept_heading()
            self.assertEqual(scout.step(other, 20.2)[0], "done")  # 가까운 자리에서 다시 보기는 면마다 한 번뿐


class QueueTests(unittest.TestCase):
    def setUp(self):
        fmap = A.field_map([(3.06, 2.5)])
        self.book = FaceBook(fmap, "BLUE")
        self.scout = FaceScout(fmap, scout_grid(fmap), self.book, start=(0.2, 2.5, 0.0))
        self.scout.stops, self.scout._planned = [(2.4, 3.1, math.radians(-42.0), [(0, "N"), (0, "W")])], True
        self.scout.step((1.2, 2.5, 0.0), 0.0)

    def test_shoots_what_the_pan_reaches_first_then_turns_for_the_rest(self):
        scout = self.scout
        here = (2.4, 3.1, math.radians(-100.0))  # W 면은 팬 47° 방향(닿는다), N 면은 69° 방향(한도 64° 밖)
        scout.accept_heading()
        reach = {key: pan_needed(here, self.book.by_key[key]) for key in ((0, "N"), (0, "W"))}
        self.assertGreater(reach[(0, "N")], 64.0)
        self.assertLess(reach[(0, "W")], 60.0)
        first = scout.step(here, 5.0)
        self.assertEqual((first[0], first[2][1]), ("shoot", (0, "W")))  # 차례는 N 이 먼저였지만 닿는 면부터
        scout.shot_taken(first[2], 5.1)
        turn = scout.step(here, 5.2)
        self.assertEqual(turn[0], "go")
        face = self.book.by_key[(0, "N")]
        self.assertLess(off(turn[3], math.atan2(face.cy - here[1], face.cx - here[0])), 0.01)

    def test_waits_for_the_answer_and_retakes_an_unread_face_from_a_closer_spot_right_away(self):
        """찍었는데 안 적힌 면: 답을 잠깐 기다렸다가, 미루지 않고 더 가까이 · 더 정면의 자리를 바로 다음에 끼워 넣는다."""
        scout = self.scout
        here = (2.4, 3.1, math.radians(-42.0))
        scout.accept_heading()
        tags = []
        for k in range(2):  # 두 면을 차례로 찍는다
            act = scout.step(here, 5.0 + 0.3 * k)
            self.assertEqual(act[0], "shoot", act)
            scout.shot_taken(act[2], 5.0 + 0.3 * k)
            tags.append((act[2], act[1]))
        self.assertEqual(scout.step(here, 5.4), ("wait",))  # 방금 찍은 사진의 답을 기다린다
        for tag, pan in tags:
            scout.on_answer(tag, here, pan, [])  # 답은 왔는데 그림이 없다 → 안 적힌다
        act = scout.step(here, 5.5)
        self.assertEqual(act[0], "go", act)
        self.assertEqual(scout.nearby, {(0, "N"), (0, "W")})
        self.assertGreaterEqual(scout.nearby_stops, 1)
        face = self.book.by_key[scout.stops[scout.i][3][0]]
        self.assertLessEqual(math.hypot(act[1] - face.cx, act[2] - face.cy), 0.95)  # 더 가까운 자리 (0.85 m 안 + 격자 여유)
        self.assertTrue(any("바로 가까운 자리" in line for line in scout.log[-3:]), scout.log[-3:])
        late = FaceScout(self.scout.fmap, scout_grid(self.scout.fmap), FaceBook(self.scout.fmap, "BLUE"), start=(0.2, 2.5, 0.0))
        late.stops, late._planned = [(2.4, 3.1, math.radians(-42.0), [(0, "W")])], True
        late.step((1.2, 2.5, 0.0), 0.0)
        late.accept_heading()
        act = late.step(here, 5.0)
        late.shot_taken(act[2], 5.0)
        self.assertEqual(late.step(here, 5.3), ("wait",))
        self.assertNotEqual(late.step(here, 5.0 + late.p.stop_answer_wait_s + 0.1), ("wait",))  # 답이 안 와도 0.6초 넘게는 기다리지 않는다

    def test_a_face_already_read_is_not_turned_for(self):
        scout = self.scout
        here = (2.4, 3.1, math.radians(60.0))  # 두 면 다 팬 밖

        class Seen:  # 다른 사진의 덤으로 이미 적혔다
            dist, incidence_deg = 0.9, 20.0

        for key in ((0, "N"), (0, "W")):
            Seen.face = self.book.by_key[key]
            self.book.add(Seen, A.NON_OBJECT, bonus=True)
        scout.accept_heading()
        act = scout.step(here, 5.0)
        self.assertNotEqual(act[1:3], (here[0], here[1]))
        self.assertEqual((scout.turns, scout.unaimed), (0, 0))


class MissionTests(unittest.TestCase):
    def test_the_turn_goes_to_navigation_and_the_notes_reach_the_match_record(self):
        from test_blocked import Ports

        mission, ports = Mission(MissionParams()), Ports()
        fmap = A.field_map(BOXES_1601, ARENA)
        now = [0.0]

        def tick(pose, **extra):
            now[0] += 0.1
            return mission.tick(Facts(now=now[0], t=1.0 + now[0], phase="running", control=True, slot="B-1", pose=pose, arrived=ports.arrived, fmap=fmap, **extra), ports)

        tick(START)
        key, far = CASES[0]
        scout = mission.scout
        scout.stops, scout.i = [next(s for s in scout.stops if s[3] == [key])], 0
        scout.queue, scout.body, scout.any_heading = [], None, False  # 첫 틱에 잡아 둔 계획의 첫 자리를 버리고 이 자리만 본다
        scout.p.retry_in_sector = False  # 자리 하나만 본다 (다른 구역의 못 적은 면을 「떠나기 전에 다시 보기」 로 끼워 넣지 않게)
        mission._scout_goal = None
        probe, book = at_stop(key)
        _spot, here = overshot(probe, book, key, far)  # 같은 지도 · 같은 자리로 만든 「지나쳐 선 자세」
        tick(far)
        tick(far)
        asked = len(ports.goes)
        ports.reach()  # navigation: 도착 (방향은 허용의 끝)
        for _ in range(12):
            tick(here)
        turn = ports.goes[-1]
        self.assertGreater(len(ports.goes), asked)
        self.assertLess(math.hypot(turn["x"] - here[0], turn["y"] - here[1]), 0.02)  # 선 자리에서 방향만
        face = scout.by_key[key]
        self.assertLess(off(turn["yaw"], math.atan2(face.cy - here[1], face.cx - here[0])), 1.0)
        self.assertAlmostEqual(math.degrees(turn["yaw_tol"]), ScoutParams().turn_tol_deg - 4.0, places=3)  # navigation 에는 4° 좁혀 넘긴다
        notes = [text for _t, text in mission.log]
        self.assertTrue(any("팬이 닿지 않음" in n and "돌려 본다" in n for n in notes), notes)


class MatchTests(unittest.TestCase):
    """14:52 판의 배치로 가상 경기. 가상 로봇은 그 판의 실제 로봇처럼 멈춘 자리가 몇 cm 씩 어긋난다 (σ 7 cm)."""

    @staticmethod
    def play(seed, **switches):
        m = SimMatch(mode="face", slot="B-1", seed=seed, layout="layout_1452.json", turn_room_m=TURN_ROOM, scout=switches,
                     params={"m6_instead_of_m5": False})  # 거점 진입(M5)까지 보는 시험이다 — 정찰이 3분 안에 끝나도 M6 가 M5 를 대신하지 않게 둔다
        m.robot.stop_xy = STOP_ERROR
        m.run()
        book = m.mission.book
        want, got = m.m3_truth(), book.counts()
        return m, sum((9 if s == "CM" else 7) for s in book.scored if got[s] == want[s])

    def test_the_wall_side_face_is_not_lost_when_the_robot_stops_a_few_cm_off(self):
        for seed in (1, 9):
            new, points = self.play(seed)
            self.assertEqual(new.mission.book.remaining(), set(), [text for _t, text in new.mission.log if "면" in text][-6:])
            self.assertEqual(points, 30)
            self.assertTrue(new.entered)


if __name__ == "__main__":
    unittest.main()
