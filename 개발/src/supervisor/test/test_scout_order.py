"""정찰 순서(구역을 하나씩 끝내기 · 탐색 구역 먼저)와, 팬 끝을 조금 넘는 면을 팬을 끝까지 돌려 찍는 것.

2026-11-19 14:52 실차(연습 1경기장, B-1): 적은 면 16개는 모두 맞았는데 M3 는 30점 가운데 7점이었다 — 채점 구역 넷 가운데 셋에 못 적은 면이 남았다.
M3 는 구역의 종류 · 개수가 모두 맞아야 점수이므로, 면을 여러 구역에 나눠 보는 것보다 구역을 하나씩 끝내는 것이 낫다.
"""
import math
import unittest

from supervisor import arena as A
from supervisor.scout import PAN_LIMIT_DEG, ScoutParams
from test_blocked import scout_for

# 그 판의 지도 (상자 가운데 14개, 6.12 × 4.80 m): 위 · 가운데 · 아래 줄에 붙은 쌍이 둘씩, 가운데에 하나씩 둘
RUN7_BOXES = [(1.43, 4.04), (1.86, 4.0), (4.27, 4.0), (4.67, 4.02), (3.05, 3.41), (1.45, 2.4), (1.85, 2.39), (4.25, 2.39), (4.63, 2.37),
              (3.04, 1.51), (1.47, 0.82), (1.87, 0.83), (4.27, 0.8), (4.64, 0.82)]
START = (0.36, 3.98, 0.0)


def planned(order, first=None):
    fmap = A.field_map(RUN7_BOXES, A.arena_from({"width": 6.12, "height": 4.80}), None)
    scout, _grid, book = scout_for(fmap, camp="BLUE", start=START)
    scout.p = ScoutParams(order=order)
    if first is not None:
        scout.prefer(first)
    scout.plan(START)
    return scout, book


def finish_index(scout, book):
    """구역마다, 그 구역의 면을 모두 찍게 되는 자리의 차례 (0부터)."""
    need = {}
    for face in book.goal:
        need.setdefault(face.sector, set()).add(face.key)
    seen, done = set(), {}
    for n, stop in enumerate(scout.stops):
        seen |= set(stop[3])
        for name, keys in need.items():
            if name not in done and keys <= seen:
                done[name] = n
    return done, need


class SectorOrderTests(unittest.TestCase):
    def lead_of(self, scout, book, name):
        """맨 앞에서부터, 구역 name 의 면이 있는 자리가 이어지는 개수."""
        n = 0
        for stop in scout.stops:
            if not any(book.by_key[k].sector == name for k in stop[3]):
                break
            n += 1
        return n

    def test_every_mode_visits_every_stop(self):
        sets = []
        for order in ("tour", "near", "best"):
            scout, _book = planned(order)
            sets.append(sorted((round(s[0], 2), round(s[1], 2), tuple(s[3])) for s in scout.stops))
        self.assertEqual(sets[0], sets[1])
        self.assertEqual(sets[0], sets[2])

    def test_default_is_the_shortest_tour(self):
        """기본 순서는 가장 짧은 길이다 (2026-11-19 저녁 가상 경기: 계획의 6할 넘게 도는 판에서는 구역을 하나씩 끝내는 순서보다 맞은 구역이 많다)."""
        self.assertEqual(ScoutParams().order, "tour")

        def length(scout):
            pts = [START[:2]] + [(s[0], s[1]) for s in scout.stops]
            return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

        tour, _b = planned("tour")
        near, _b = planned("near")
        self.assertLess(length(tour), length(near))

    def test_the_search_sector_of_the_order_comes_first(self):
        """명령의 탐색 구역 가운데 상대 진영의 것(지정표적이 있는 곳)의 자리들을 맨 앞에 둔다. CM 은 표적이 없으므로 앞세우지 않는다."""
        for order in ("tour", "best"):
            for name in ("R-S1", "R-S2", "R-S3"):
                scout, book = planned(order, first=["CM", name])
                self.assertEqual(scout.first, (name,))
                done, need = finish_index(scout, book)
                self.assertEqual(min(done, key=done.get), name, (order, name))
                # 그 구역을 다 찍을 때까지 간 자리는 모두 그 구역의 면이 있는 자리다
                self.assertEqual(self.lead_of(scout, book, name), done[name] + 1, (order, name))

    def test_same_plan_every_time(self):
        for order in ("tour", "best"):
            a, _ = planned(order, first=["R-S1"])
            b, _ = planned(order, first=["R-S1"])
            self.assertEqual(a.stops, b.stops)

    def test_search_sector_known_after_planning_reorders_the_rest(self):
        scout, book = planned("tour")
        before = list(scout.stops)
        done, _need = finish_index(scout, book)
        late = max((name for name in done if name != "CM"), key=done.get)  # 지금 계획에서 맨 나중에 끝나는 상대 구역
        act = scout.step(START, 0.0)  # 첫 자리로 가는 중
        self.assertEqual(act[0], "go")
        scout.prefer([late])  # 명령 해석이 뒤늦게 왔다
        self.assertEqual(scout.stops, before)  # 가던 자리는 그대로
        scout.body = None  # 그 자리를 마쳤다고 치고
        scout.step(START, 1.0)
        done2, _need = finish_index(scout, book)
        self.assertEqual(min(done2, key=done2.get), late)
        self.assertEqual(sorted(map(str, scout.stops)), sorted(map(str, before)))  # 자리는 그대로, 순서만 바뀐다

    def test_leftover_faces_are_replanned_nearest_first(self):
        """다시 도는 면은 가까운 것부터다 — 먼저 돌 구역을 앞세우지 않는다 (16:01 실차: 0.45 m 옆의 면을 두고 4.3 m 떨어진 면으로 갔다)."""
        scout, book = planned("tour", first=["R-S1"])
        near, far = (13, "S"), (2, "N")
        pose = (4.2, 0.25, math.pi)  # 아래 통로, 13S 바로 옆
        scout.plan(pose, only={near, far})
        self.assertEqual(list(scout.stops[0][3]), [near])


class FullPanShotTests(unittest.TestCase):
    """자리에 왔는데 차체를 돌릴 자리가 없다 (벽 쪽 통로). 찍을 면이 팬 끝(±48°)을 조금 넘는 방향이면 팬을 끝까지 돌려 찍는다.

    그 판에서: 통로에 선 채 찍을 면이 팬 48.6° · 54.3° 방향이었는데 사진을 찍지 않고 떠났다 (둘 다 탄약상자 — 그 구역 둘이 0점).
    """

    def setUp(self):
        fmap = A.field_map([(3.06, 2.5)])
        self.scout, _grid, _book = scout_for(fmap, camp="BLUE", start=(0.2, 2.5, 0.0))
        self.key = (0, "W")
        self.scout.stops, self.scout._planned = [(2.26, 2.5, 0.0, [self.key])], True
        self.scout.step((1.0, 2.5, 0.0), 0.0)  # 가는 중

    def shot_at(self, off_deg):
        self.scout.accept_heading()  # navigation: 자리에는 왔는데 돌 자리가 없다
        return self.scout.step((2.26, 2.5, math.radians(off_deg)), 1.0)

    def test_just_beyond_pan_reach_is_shot_with_full_pan(self):
        act = self.shot_at(54.0)
        self.assertEqual(act[0], "shoot")
        self.assertAlmostEqual(act[1], PAN_LIMIT_DEG - 2.0, places=3)  # 왼쪽으로 54° 틀어졌다 → 팬은 오른쪽 끝
        self.assertEqual(act[2][1], self.key)

    def test_other_side(self):
        act = self.shot_at(-49.0)
        self.assertEqual(act[0], "shoot")
        self.assertAlmostEqual(act[1], -(PAN_LIMIT_DEG - 2.0), places=3)

    def test_far_beyond_is_left_for_later_and_said(self):
        act = self.shot_at(70.0)  # 48 + 16 을 넘는다 → 면이 사진 가장자리 밖
        self.assertNotEqual(act[0], "shoot")
        self.assertTrue(any("팬이 닿지 않음" in line for line in self.scout.log))


class CorridorStopTests(unittest.TestCase):
    """벽 쪽 통로의 면은 통로 가운데에서, 통로 방향(에서 상자 쪽으로 조금)을 본 채 서서 팬만 옆으로 돌려 찍는다.

    14:52 판: 위 통로의 자리가 벽에서 0.18 m 에 잡혀, 로봇이 벽을 보고 선 채 차체를 돌리지 못했다 (두 면을 못 적어 R-S1 이 0점).
    16:01 판(통로 가운데에 선 첫 판): 둘째 자리 둘은 찍혔고, 통로 밖에서 비스듬히 들어온 첫 자리 둘은 벽 쪽으로 18 ~ 21° 틀어지고 9 cm 지나쳐
    면이 팬 72° · 63° 방향이라 못 찍었다 → 더 뒤에, 상자 쪽으로 조금 돌려 선다.
    """

    def setUp(self):
        self.scout, self.book = planned("best")
        self.by_face = {tuple(stop[3]): stop for stop in self.scout.stops if len(stop[3]) == 1}
        self.p = self.scout.p

    def test_wall_side_faces_get_a_stop_on_the_corridor_centre_line(self):
        self.assertEqual(set(self.scout.corridor), {(2, "N"), (3, "N"), (12, "S"), (13, "S")})
        for key in self.scout.corridor:
            x, y, heading, _keys = self.by_face[(key,)]
            face = self.book.by_key[key]
            wall = 4.80 if key[1] == "N" else 0.0
            half = abs(face.cy - wall) / 2.0
            self.assertAlmostEqual(y, (face.cy + wall) / 2.0, delta=0.02)  # 면과 벽의 한가운데
            self.assertAlmostEqual(abs(x - face.cx), self.p.corridor_back * half, delta=0.02)  # 면 가운데보다 앞
            off = min(abs(math.degrees(heading)), 180.0 - abs(math.degrees(heading)))
            self.assertAlmostEqual(off, self.p.corridor_bias_deg, places=3)  # 통로 방향에서 상자 쪽으로 조금
            box_side = -1.0 if key[1] == "N" else 1.0  # 상자는 N 면의 남쪽 · S 면의 북쪽
            self.assertGreater(math.sin(heading) * box_side, 0.0)
        # 다른 면의 자리는 그대로다 (통로가 아닌 면은 통로 표에 없다)
        self.assertNotIn((2, "S"), self.scout.corridor)
        self.assertNotIn((3, "E"), self.scout.corridor)

    def test_reachable_even_when_the_robot_stops_off_and_askew(self):
        """실차에서 본 만큼(±12 cm · 정한 방향에서 ±20°) 어긋나도 면이 팬에 닿고, 다 보이고, 적을 수 있는 비스듬함 안이다."""
        from supervisor.faces import expected_faces
        for key, spots in self.scout.corridor.items():
            face = self.book.by_key[key]
            for x, y, heading in spots:
                step = (math.cos(heading), 0.0)  # 통로를 따라 (위 · 아래 통로는 x 쪽)
                for along in (-0.12, 0.0, 0.12):
                    for turn in (-20.0, 0.0, 20.0):
                        pose = (x + math.copysign(along, step[0]) if along else x, y, heading + math.radians(turn))
                        pan = self.scout._aim(pose, face)
                        self.assertIsNotNone(pan, (key, along, turn))
                        seen = {e.face.key: e for e in expected_faces(self.scout.fmap, pose, pan, self.scout.cam)}
                        self.assertIn(key, seen, (key, along, turn))
                        self.assertLessEqual(seen[key].incidence_deg, self.p.aim_incidence_deg, (key, along, turn))
                        self.assertGreaterEqual(seen[key].fraction, 0.85, (key, along, turn))

    def go_to(self, key, pose):
        """그 면의 자리가 다음 차례가 되게 하고 한 틱 돌린다."""
        scout = self.scout
        scout.i = next(n for n, stop in enumerate(scout.stops) if tuple(stop[3]) == (key,))
        scout.queue, scout.body, scout.any_heading = [], None, False
        return scout.step(pose, 0.0)

    def test_body_keeps_the_corridor_direction_and_the_pan_looks_sideways(self):
        key = (12, "S")
        act = self.go_to(key, (3.0, 1.0, 0.0))  # 서쪽에서 들어온다
        self.assertEqual(act[0], "go")
        self.assertAlmostEqual(math.degrees(act[3]), self.p.corridor_bias_deg, places=3)  # 동쪽에서 상자(북) 쪽으로 조금
        self.assertAlmostEqual(math.degrees(act[4]), self.p.corridor_yaw_tol_deg, places=3)
        x, y = act[1], act[2]
        self.assertLess(x, self.book.by_key[key].cx)  # 면보다 앞(서쪽)에 선다
        shot = self.scout.step((x, y, math.radians(-6.0)), 1.0)  # 벽 쪽으로 18° 틀어진 채 도착 (16:01 판의 첫 자리)
        self.assertEqual(shot[0], "shoot")
        self.assertEqual(shot[2][1], key)
        self.assertLess(shot[1], -30.0)  # 아래 통로를 동쪽으로 가면 상자는 왼쪽 → 팬은 왼쪽(−)
        self.assertGreaterEqual(shot[1], -(PAN_LIMIT_DEG - 2.0))

    def test_stop_follows_the_side_the_robot_comes_from(self):
        key = (3, "N")
        face = self.book.by_key[key]
        east = self.go_to(key, (5.3, 4.4, math.pi))  # 동쪽에서 온다 → 면의 동쪽에 서서 서쪽(에서 남쪽으로 조금)을 본다
        self.assertGreater(east[1], face.cx)
        self.assertAlmostEqual(abs(math.degrees(east[3])), 180.0 - self.p.corridor_bias_deg, places=3)
        west = self.go_to(key, (3.4, 4.4, 0.0))  # 서쪽에서 온다 → 면의 서쪽에 서서 동쪽(에서 남쪽으로 조금)을 본다
        self.assertLess(west[1], face.cx)
        self.assertAlmostEqual(math.degrees(west[3]), -self.p.corridor_bias_deg, places=3)

    def test_no_corridor_far_from_the_walls(self):
        fmap = A.field_map([(3.06, 2.4)], A.arena_from({"width": 6.12, "height": 4.80}), None)
        scout, _grid, _book = scout_for(fmap, camp="BLUE", start=START)
        scout.plan(START)
        self.assertEqual(scout.corridor, {})


class RetryBeforeLeavingTests(unittest.TestCase):
    """구역을 떠나기 전에, 그 구역에 못 적은 면이 남았으면 바로 다시 들른다 (구역마다 한 번).

    16:01 판: 27초에 팬이 안 닿아 놓친 2N 을 210초에야 다시 가다가 끊겼다 — 그 면 하나가 모자라 R-S1 이 0점이었다.
    """

    def setUp(self):
        self.scout, self.book = planned("best", first=["R-S1"])
        self.sector = lambda stop: {self.book.by_key[k].sector for k in stop[3]}

    def finish_sector_but(self, name, missed):
        """구역 name 의 자리를 차례로 「찍었다」 고 치되, 면 missed 하나는 팬이 안 닿아 못 찍은 것으로 둔다. 다음 틱의 답을 돌려준다."""
        scout, pose = self.scout, START
        while scout.i < len(scout.stops) and name in self.sector(scout.stops[scout.i]):
            x, y, _h, keys = scout.stops[scout.i]
            scout.step(pose, 0.0)  # 그 자리로 떠난다 (구역을 시작한 것으로 적힌다)
            pose = (x, y, 0.0)
            for key in keys:
                if key != missed:
                    scout.book.add(self.exp_of(key), "tank")
            scout.i += 1
            scout.queue, scout.body, scout.any_heading = [], None, False
        return scout.step(pose, 1.0), pose

    def exp_of(self, key):
        from supervisor.faces import Expected
        return Expected(face=self.book.by_key[key], x=0.5, rel=0.0, dist=0.8, incidence_deg=10.0, fraction=1.0, sure=True)

    def test_missed_face_is_revisited_before_the_next_sector(self):
        missed = (2, "N")
        act, _pose = self.finish_sector_but("R-S1", missed)
        self.assertEqual(act[0], "go")
        self.assertEqual(list(self.scout.stops[self.scout.i][3]), [missed])  # 다음 자리는 놓친 면의 자리다
        self.assertTrue(any("떠나기 전에 다시 본다" in line for line in self.scout.log))
        self.assertIn("R-S1", self.scout._sectors_retried)

    def test_only_once_per_sector(self):
        missed = (2, "N")
        _act, pose = self.finish_sector_but("R-S1", missed)
        scout = self.scout
        scout.i += 1  # 다시 들른 자리에서도 못 찍었다
        scout.queue, scout.body, scout.any_heading = [], None, False
        act = scout.step(pose, 2.0)
        self.assertEqual(act[0], "go")
        self.assertNotIn(missed, scout.stops[scout.i][3])  # 이제는 다음 구역으로 간다
        self.assertEqual(sum("떠나기 전에 다시 본다" in line for line in scout.log), 1)

    def test_nothing_extra_when_the_sector_is_complete(self):
        before = len(self.scout.stops)
        self.finish_sector_but("R-S1", None)
        self.assertEqual(len(self.scout.stops), before)

    def test_can_be_switched_off(self):
        self.scout.p.retry_in_sector = False
        before = len(self.scout.stops)
        self.finish_sector_but("R-S1", (2, "N"))
        self.assertEqual(len(self.scout.stops), before)


if __name__ == "__main__":
    unittest.main()


class TargetHintTests(unittest.TestCase):
    """지정표적의 종류가 찍힌 그림이 가리키는 면부터 찍으러 간다 (적지는 않는다).

    2026-11-19 모의 5차: 시작 10.3초의 사진에 지정표적(레이더)이 3 m 밖에서 찍혔는데, 적는 규칙에는 못 미쳐 100초에야 그 면을 찍으러 갔다.
    """

    def setUp(self):
        from supervisor.faces import CameraParams, expected_faces

        self.scout, self.book = planned("tour")
        self.scout.want_kind("comms")
        self.pose = (1.2, 3.2, 0.0)
        self.expected = expected_faces(self.scout.fmap, self.pose, 0.0, CameraParams())
        left = self.book.remaining()
        far = [e for e in self.expected if e.face.key in left and self.book.by_key[e.face.key].sector not in ("CM",) and 2.0 <= e.dist <= 4.5 and e.incidence_deg <= 60.0]
        self.assertTrue(far, "이 자리에서 보이는 상대 진영의 면이 있어야 한다")
        self.e = max(far, key=lambda e: e.dist)  # 먼 면 — 처음 계획에서는 뒤쪽 차례다

    def pic(self, kind="comms", conf=0.7, dx=0.0):
        """그 면이 찍힐 자리의 그림 (dx 만큼 옆으로). 먼 상자 쌍의 두 면은 사진에서 가까이 붙어 있어, 조금만 밀려도 옆 면이 더 가깝다."""
        return (kind, self.e.x + dx, False, self.e.width, self.e.height, conf)

    def place_of(self, key):
        return next(n for n, stop in enumerate(self.scout.stops) if key in stop[3])

    def test_a_weak_sighting_of_the_target_kind_moves_that_face_to_the_front(self):
        scout, key = self.scout, self.e.face.key
        scout.step(START, 0.0)
        before = self.place_of(key)
        self.assertGreater(before, 0)
        recorded = scout.on_pass(("pass", 1), self.pose, 0.0, [self.pic()])
        self.assertEqual(recorded, [])  # 적지는 않는다 (확신 0.7 한 장)
        self.assertEqual(scout._lead, key)
        act = scout.step(self.pose, 1.0)
        self.assertEqual(act[0], "go")
        self.assertEqual(self.place_of(key), scout.i)  # 그 면을 찍을 자리가 지금 갈 자리다
        self.assertEqual((round(act[1], 2), round(act[2], 2)), (round(scout.stops[scout.i][0], 2), round(scout.stops[scout.i][1], 2)))
        self.assertEqual(len(scout.stops), len({(round(s[0], 3), round(s[1], 3), tuple(s[3])) for s in scout.stops}))  # 자리는 하나도 없어지거나 겹치지 않는다
        # 같은 면을 또 가리켜도 순서를 다시 바꾸지 않는다
        scout.on_pass(("pass", 2), self.pose, 0.0, [self.pic()])
        self.assertIsNone(scout._lead)

    def test_when_the_hinted_face_turns_out_to_be_something_else_the_next_face_of_that_box_is_tried(self):
        """연습 판(21:54): 레이더는 상자의 E 면에 있었는데 S 면을 가리켰다. S 면이 전차로 적히면 같은 상자의 다른 면을 이어서 본다."""
        scout, key = self.scout, self.e.face.key
        scout.step(START, 0.0)
        scout.on_pass(("pass", 1), self.pose, 0.0, [self.pic()])
        scout.step(self.pose, 1.0)
        self.assertIsNone(scout._lead)

        class Seen:  # 가리킨 면이 다른 종류로 적혔다
            dist, incidence_deg = 0.9, 20.0
            face = self.book.by_key[key]

        self.book.add(Seen, "tank")
        scout.any_heading = False
        scout.step(self.pose, 2.0)
        nxt = scout.stops[scout.i][3]
        self.assertTrue(any(k[0] == key[0] and k != key for k in nxt), (key, nxt))  # 같은 상자의 다른 면을 찍을 자리가 다음이다
        self.assertTrue(any("이어서 찍으러 간다" in line for line in scout.log[-4:]), scout.log[-4:])
        self.assertEqual(scout._hints_used, 1)  # 이어서 가리킨 것은 횟수에 세지 않는다
        sibling = next(k for k in nxt if k[0] == key[0] and k != key)
        Seen.face = self.book.by_key[sibling]
        self.book.add(Seen, "comms")  # 그 면이 지정표적이었다 → 더 가리키지 않는다
        scout.any_heading = False
        before = list(scout._hinted)
        scout.step(self.pose, 3.0)
        self.assertEqual(sorted(scout._hinted), sorted(before))

    def test_no_hint_for_other_kinds_low_confidence_or_when_switched_off(self):
        scout = self.scout
        scout.step(START, 0.0)
        for pic in (self.pic(kind="tank"), self.pic(conf=0.4), self.pic(dx=0.2)):
            scout.on_pass(("pass", 1), self.pose, 0.0, [pic])
            self.assertIsNone(scout._lead)
        scout.want_kind(None)  # 표적을 이미 찾았다 · 모른다
        scout.on_pass(("pass", 2), self.pose, 0.0, [self.pic()])
        self.assertIsNone(scout._lead)
        scout.want_kind("comms")
        scout.p.hint_on = False
        scout.on_pass(("pass", 3), self.pose, 0.0, [self.pic()])
        self.assertIsNone(scout._lead)

    def test_at_most_hint_max_faces_per_match_and_never_a_centre_face(self):
        from supervisor.faces import CameraParams, expected_faces

        scout = self.scout
        scout.step(START, 0.0)
        scout.p.hint_max = 1
        scout.on_pass(("pass", 1), self.pose, 0.0, [self.pic()])
        first = scout._lead
        self.assertIsNotNone(first)
        scout.step(self.pose, 1.0)
        other = next(e for e in self.expected if e.face.key != first and e.face.key in self.book.remaining() and self.book.by_key[e.face.key].sector != "CM"
                     and e.dist <= 4.5 and e.incidence_deg <= 75.0 and abs(e.x - self.e.x) > 0.25)
        scout.on_pass(("pass", 2), self.pose, 0.0, [("comms", other.x, False, other.width, other.height, 0.9)])
        self.assertIsNone(scout._lead)  # 한 판에 hint_max 번까지
        fresh, book = planned("tour")
        fresh.want_kind("comms")
        fresh.step(START, 0.0)
        pose = (2.0, 3.41, 0.0)  # 가운데 상자(CM) 바로 앞
        cm = [e for e in expected_faces(fresh.fmap, pose, 0.0, CameraParams()) if book.by_key[e.face.key].sector == "CM" and e.dist <= 1.5]
        self.assertTrue(cm)
        only_cm = [("comms", e.x, False, e.width, e.height, 0.9) for e in cm if all(abs(e.x - o.x) > 0.12 for o in expected_faces(fresh.fmap, pose, 0.0, CameraParams()) if book.by_key[o.face.key].sector != "CM")]
        if only_cm:
            fresh.on_pass(("pass", 1), pose, 0.0, only_cm)
            self.assertIsNone(fresh._lead)  # 지정표적은 CM 에 없다
