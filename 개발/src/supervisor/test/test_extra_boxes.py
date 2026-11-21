"""지도에 없는 상자 (2026-11-19): navigation 이 라이다로 찾아 /team/navigation/lidar_boxes 로 알려 주는 상자를 받아들여, 그 면을 정찰 계획의 맨 뒤에 붙인다.

왜: 연습 경기장의 드론 사진은 위 · 아래가 잘려 있어, 위 · 아래 벽에 붙은 상자가 지도에 없다 (실차 한 판에서 셋). 그 면은 한 번도 찍히지 않았다.
navigation 은 그런 상자를 라이다로 찾지만 지도에는 넣지 않는다 (주행 노드가 지도의 상자를 안 움직이는 벽으로 다루므로) — 따로 알려 준다.

여기서 보는 것
  - 받아들이는 조건 (오래 보였는가 · 지도의 상자와 떨어져 있는가 · 기둥 · 경기장 밖 · 채점 구역 · 상한 · 끄기)
  - 만드는 면 (벽 쪽 면은 없다 · 구역이 적힌다), 격자에서 막히는 칸
  - 정찰 자리는 계획의 맨 뒤에 붙고 앞의 순서는 그대로다, 정찰을 끝낸 뒤에 나오면 다시 정찰한다, 상자 없는 임시 지도에서도 된다
  - 목록에서 사라지면 없는 것으로 치고, 다시 나오면 되살린다
  - 가상 경기 한 판: 지도에 없는 상자의 면이 찍혀 M3 개수에 들어간다. 그런 상자가 없는 경기는 전과 똑같다
"""
import math
import unittest

from sim_world import FakePerception, SimMatch
from supervisor import arena as A
from supervisor.faces import CameraParams, FaceBook
from supervisor.grid import Grid
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import FaceScout, ScoutParams

MAP = [(3.06, 2.5), (3.06, 3.5)]  # 지도의 상자 둘 (공통 구역 CM)
WALL_BOX = (4.46, 4.8)  # 위쪽 벽에 붙은, 지도에 없는 상자 (R-S1). 경기장 높이 5.00 m, 상자 반 변 0.20 m
START = (1.2, 2.5, 0.0)


def truth_for(world, **kinds):
    """경기장의 모든 면의 정답. 따로 주지 않은 면은 비객체. kinds 의 이름은 「상자번호_면」 (예 b2_S="tank")."""
    truth = {f.key: A.NON_OBJECT for f in world.faces}
    for name, kind in kinds.items():
        truth[(int(name[1:].split("_")[0]), name.split("_")[1])] = kind
    return truth


class Tiny:
    """작은 가상 세계 (창구를 겸한다): 가라는 자리까지 곧은 거리 / 0.5 m/s 뒤에 가 있고, 찍은 사진의 답은 다음 틱에 온다.

    world = 경기장의 참 모습 (지도에 없는 상자 포함), mapped = mission 에 지도로 주는 상자. lidar = navigation 의 라이다 상자 목록 (None = 온 적 없음).
    """

    def __init__(self, world, truth, mapped, pose=START, slot="B-2", params=None, mode="face", fmap=None):
        self.eyes = FakePerception(world, truth, cam=CameraParams(), x_noise=0.0)
        self.mission = Mission(MissionParams(scout=ScoutParams(mode=mode), **(params or {})))
        self.mapped, self.fmap, self.slot, self.pose = list(mapped), fmap, slot, pose
        self.goal, self.arrive_at = None, 0.0
        self.mail, self.shots, self.goes = [], [], []
        self.now, self.lidar, self.task = 0.0, None, None

    # ---- 창구 (mission.Ports)
    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None):
        goal = (x, y, yaw)
        if goal != self.goal:
            self.goal = goal
            self.arrive_at = self.now + math.hypot(x - self.pose[0], y - self.pose[1]) / 0.5
            self.goes.append((round(x, 2), round(y, 2)))

    def stop(self):
        self.goal = None

    def shoot(self, tag, pan_deg):
        self.shots.append(tag)
        self.mail.append((tag, self.pose, pan_deg, self.eyes.look(self.pose, pan_deg)[0]))
        return True

    def aim(self, *a): return True
    def aim_ready(self): return True
    def fire(self): return True
    def look_ugvs(self): return False
    def analyze_order(self, text): return True
    def set_ready(self): return True
    def report(self, kind, payload): return True

    # ---- 시간
    def run(self, seconds, **more):
        """seconds 동안 0.1초 틱을 돌린다. more 는 Facts 에 그대로 넣는다 (예 gone=…)."""
        for _ in range(int(round(seconds / 0.1))):
            if self.goal is not None and self.now >= self.arrive_at:
                self.pose, self.goal = self.goal, None
            answers, self.mail = self.mail, []
            f = Facts(now=self.now, t=self.now, phase="running", control=True, slot=self.slot, pose=self.pose, arrived=self.goal is None,
                      boxes=self.mapped or None, fmap=self.fmap, answers=answers, extra_boxes=self.lidar, **more)
            self.task = self.mission.tick(f, self)
            self.now += 0.1
        return self.task

    def run_until(self, done, limit=120.0, **more):
        end = self.now + limit
        while not done() and self.now < end:
            self.run(0.1, **more)
        return done()

    def said(self, text):
        """기록에서 text 가 든 줄들."""
        return [line for _t, line in self.mission.log if text in line]


def tiny(extra=WALL_BOX, kinds=None, **kw):
    """지도의 상자 둘(MAP) + 지도에 없는 상자 하나가 있는 세계. 지도에 없는 상자는 경기장의 2번이고, mission 이 받아들이면 그것도 2번이 된다."""
    world = A.field_map(MAP + [extra])
    return Tiny(world, truth_for(world, **(kinds or {"b0_W": "drum", "b1_E": "mortar", "b2_S": "tank"})), MAP, **kw)


class AcceptRulesTests(unittest.TestCase):
    def test_nothing_happens_until_navigation_sends_a_list(self):
        w = tiny()
        w.run(8.0)
        self.assertEqual(len(w.mission.fmap.boxes), 2)
        self.assertFalse(w.said("라이다"))

    def test_accepted_only_after_it_stayed_in_the_list_for_four_seconds(self):
        w = tiny()
        w.run(1.0)
        w.lidar = [WALL_BOX]
        w.run(3.9)
        self.assertEqual(len(w.mission.fmap.boxes), 2)  # 3.9초: 아직
        w.run(0.3)
        self.assertEqual(w.mission.fmap.boxes[2:], [WALL_BOX])  # 4초를 넘겼다 → 맨 뒤 번호(2번)로
        self.assertEqual(w.mission.fmap.boxes[:2], MAP)  # 앞의 번호는 그대로
        self.assertEqual(len(w.said("지도에 없는 상자를 라이다로 찾음")), 1)
        w.run(5.0)
        self.assertEqual(len(w.mission.fmap.boxes), 3)  # 같은 상자를 또 넣지 않는다
        self.assertFalse(w.mission._extra_off)

    def test_a_box_that_drops_out_of_the_list_starts_counting_again(self):
        w = tiny()
        w.lidar = [WALL_BOX]
        w.run(3.0)
        w.lidar = []
        w.run(0.2)  # 한 번 빠졌다
        w.lidar = [WALL_BOX]
        w.run(3.0)
        self.assertEqual(len(w.mission.fmap.boxes), 2)
        w.run(1.3)
        self.assertEqual(len(w.mission.fmap.boxes), 3)

    def test_small_position_changes_are_the_same_candidate(self):
        """목록의 자리가 0.15 m 안에서 조금씩 달라져도 같은 후보로 센다."""
        w = tiny()
        for k in range(45):
            w.lidar = [WALL_BOX if k % 2 else (WALL_BOX[0] + 0.06, WALL_BOX[1] - 0.05)]
            w.run(0.1)
        self.assertEqual(len(w.mission.fmap.boxes), 3)

    def test_refused_places_and_why(self):
        w = tiny()
        near_map, on_pillar, outside, own_half = (3.30, 2.5), (3.16, 4.75), (4.46, 4.95), (1.5, 4.8)
        w.lidar = [near_map, on_pillar, outside, own_half]
        w.run(12.0)
        self.assertEqual(len(w.mission.fmap.boxes), 2)
        why = w.said("는 넣지 않는다")
        self.assertEqual(len(why), 4, why)  # 후보마다 한 번만 적는다 (틱마다 되풀이하지 않는다)
        self.assertEqual(len(w.said("지도에 있는 상자의 자리")), 1)  # 지도의 0번 상자에서 0.24 m
        self.assertEqual(len(w.said("고정 기둥의 자리")), 1)  # 위쪽 벽 가운데 기둥 (3.06, 4.80)
        self.assertEqual(len(w.said("경기장 벽 밖으로 걸쳐 있음")), 1)  # 가운데가 벽에서 5 cm
        self.assertEqual(len(w.said("채점 구역에 찍을 면이 없음")), 1)  # 우리(BLUE) 진영 구역 B-S1 의 상자
        # navigation 이 지도의 0번 상자를 없다고 보면, 그 곁의 라이다 상자는 받아들인다 (드론 사진의 자리가 틀렸던 경우)
        w.run(0.5, gone=frozenset({0}))
        self.assertEqual(w.mission.fmap.boxes[2:], [near_map])

    def test_no_more_than_the_cap(self):
        a, b, c = (4.46, 4.8), (4.46, 0.2), (4.0, 2.5)
        w = tiny(params={"extra_max": 2})
        w.lidar = [a, b, c]
        w.run(10.0)
        self.assertEqual(w.mission.fmap.boxes[2:], [a, b])
        self.assertEqual(len(w.said("상한(2개)에 닿음")), 1)
        self.assertEqual(MissionParams().extra_max, 6)
        # 받아들였던 것이 없어지면 그 자리는 빈다 (없어진 상자가 상한을 차지하지 않는다)
        w.lidar = [b, c]
        w.run(6.5)
        self.assertEqual(w.mission.fmap.boxes[2:], [a, b, c])
        self.assertEqual(w.mission.fmap.gone, frozenset({2}))

    def test_no_more_than_three_times_the_cap_over_a_whole_match(self):
        """받아들였다 없어지기를 되풀이해도(위치가 틀린 동안 생긴 헛것 등) 합쳐서 상한의 3배까지만 받는다."""
        w = tiny(params={"extra_max": 1})
        spots = [(4.46, 4.8), (4.46, 0.2), (4.0, 2.5), (4.4, 1.5)]
        for spot in spots:
            w.lidar = [spot]
            w.run(7.0)
        self.assertEqual(w.mission.fmap.boxes[2:], spots[:3])  # 넷째는 받지 않는다 (앞의 셋은 모두 없어진 뒤다)
        self.assertEqual(w.mission.fmap.gone, frozenset({2, 3, 4}))
        # 둘째 · 셋째도 앞의 상자가 아직 있는 것으로 치던 동안(6초)에는 상한에 걸렸다가, 그 상자가 없어진 뒤에 들어왔다
        self.assertEqual([line.split(" 는 ")[0] for line in w.said("상한(1개)에 닿음")],
                         ["라이다로 찾은 상자 (%.2f, %.2f)" % spot for spot in spots[1:]])

    def test_switched_off_by_the_parameter_and_in_place_mode(self):
        for kw in ({"params": {"extra_boxes": False}}, {"mode": "place"}):
            w = tiny(**kw)
            w.lidar = [WALL_BOX]
            w.run(10.0)
            self.assertEqual(len(w.mission.fmap.boxes), 2, kw)
            self.assertEqual(len(w.mission.book.goal), 8, kw)
            self.assertFalse(w.said("라이다"), kw)


class FacesAndGridTests(unittest.TestCase):
    def test_faces_of_a_box_against_the_top_wall(self):
        fmap = A.field_map(MAP)
        faces = fmap.box_faces(WALL_BOX[0], WALL_BOX[1], 2)
        self.assertEqual([f.side for f in faces], ["E", "W", "S"])  # 벽 쪽(N) 면은 없다. 옆 면은 앞이 트여 있어 남는다
        self.assertEqual({f.sector for f in faces}, {"R-S1"})
        south = faces[-1]
        self.assertEqual((south.box, round(south.cx, 2), round(south.cy, 2), south.nx, south.ny), (2, 4.46, 4.6, 0.0, -1.0))
        self.assertEqual(len(fmap.boxes), 2)  # box_faces 는 지도를 건드리지 않는다
        # 벽에서 0.20 m 떨어진 상자: 벽 쪽 면 앞이 0.30 m 보다 좁아 찍을 수 없다 → 만들지 않는다. 한도를 낮추면 만든다
        self.assertEqual([f.side for f in fmap.box_faces(4.46, 4.6, 2)], ["E", "W", "S"])
        self.assertEqual([f.side for f in fmap.box_faces(4.46, 4.6, 2, wall_gap_m=0.15)], ["E", "N", "W", "S"])

    def test_sides_touching_a_live_box_or_a_pillar_are_left_out(self):
        fmap = A.field_map(MAP)
        self.assertEqual([f.side for f in fmap.box_faces(3.46, 2.5, 2)], ["E", "N", "S"])  # W 는 지도의 0번 상자에 맞닿는다
        self.assertEqual([f.side for f in fmap.box_faces(3.46, 4.8, 2)], ["E", "S"])  # W 는 위쪽 가운데 기둥, N 은 벽
        fmap.set_gone({0})
        self.assertEqual([f.side for f in fmap.box_faces(3.46, 2.5, 2)], ["E", "N", "W", "S"])  # 없다고 본 상자는 가리지 않는다

    def test_map_book_and_grid_after_the_box_is_accepted(self):
        w = tiny()
        w.run(0.3)
        mission = w.mission
        old_faces, old_goal = list(mission.fmap.faces), list(mission.book.goal)
        self.assertTrue(mission.grid.free(4.46, 4.5))
        w.lidar = [WALL_BOX]
        w.run(4.3)
        fmap, book, grid = mission.fmap, mission.book, mission.grid
        self.assertEqual(fmap.faces[:len(old_faces)], old_faces)  # 앞의 면은 그대로
        self.assertEqual([f.key for f in fmap.faces[len(old_faces):]], [(2, "E"), (2, "W"), (2, "S")])
        self.assertEqual(book.goal[:len(old_goal)], old_goal)
        self.assertEqual({f.key for f in book.goal[len(old_goal):]}, {(2, "E"), (2, "W"), (2, "S")})  # 채점 구역(R-S1)의 면 → 목표에 든다
        self.assertEqual(len(fmap.segments()), 4 + 4 * (3 + len(fmap.pillars)))  # 시야 가림에도 들어간다
        # 격자: 상자 둘레(반 변 0.20 + 차체 0.15 + 여유 0.02 = 0.37 m)가 막힌다. 그 바깥 0.10 m 는 「막힌 곳 가까운 칸」
        self.assertIn("extra2", grid.closed)
        self.assertFalse(grid.free(4.46, 4.5))
        self.assertFalse(grid.free(4.80, 4.6))
        self.assertTrue(grid.free(4.46, 4.3))
        self.assertIn(grid.cell_of(4.46, 4.38), grid.tight)
        self.assertNotIn(grid.cell_of(4.46, 4.2), grid.tight)


def scout_on(boxes, camp="BLUE", start=START):
    fmap = A.field_map(boxes)
    grid = Grid(fmap)
    book = FaceBook(fmap, camp)
    return FaceScout(fmap, grid, book, params=ScoutParams(), start=start), fmap, grid, book


def add_box(scout, fmap, grid, book, x, y, pose=START):
    """mission 이 하는 차례대로: 지도 → 격자 → 면 기록 → 정찰."""
    faces = fmap.box_faces(x, y, len(fmap.boxes))
    number = fmap.add_box(x, y, faces)
    half = fmap.box_size / 2.0 + grid.grow
    grid.add_zone("extra%d" % number, (x - half, y - half, x + half, y + half), closed=True, soft=True)
    goal = book.add_faces(faces)
    return scout.add_faces(pose, [f.key for f in goal]), [f.key for f in goal]


class ScoutStopsTests(unittest.TestCase):
    def test_stops_go_to_the_end_and_the_plan_before_them_is_untouched(self):
        scout, fmap, grid, book = scout_on(MAP)
        scout.plan(START)
        before = list(scout.stops)
        self.assertGreater(len(before), 2)
        scout.i = 1  # 계획을 따라가는 중
        added, keys = add_box(scout, fmap, grid, book, *WALL_BOX)
        self.assertGreaterEqual(added, 1)
        self.assertEqual(scout.stops[:len(before)], before)  # 앞의 자리 · 순서는 그대로
        self.assertEqual(scout.i, 1)
        tail = scout.stops[len(before):]
        self.assertEqual(len(tail), added)
        self.assertEqual({k for stop in tail for k in stop[3]}, set(keys))  # 붙은 자리는 새 면만 본다
        self.assertFalse(scout.unreachable & set(keys))
        for x, y, _heading, _keys in tail:  # 새 상자의 막힌 둘레 밖에 선다
            self.assertTrue(grid.free(x, y), (x, y))
        self.assertEqual(scout.add_faces(START, keys), 0)  # 이미 갈 자리가 있는 면은 다시 넣지 않는다
        self.assertEqual(len(scout.stops), len(before) + added)

    def test_faces_that_come_before_the_first_plan_still_go_last(self):
        plain, _fmap, _grid, _book = scout_on(MAP)
        plain.plan(START)
        scout, fmap, grid, book = scout_on(MAP)
        added, keys = add_box(scout, fmap, grid, book, *WALL_BOX)
        self.assertEqual(added, 0)  # 아직 계획 전: 적어만 둔다
        scout.plan(START)
        head = scout.stops[:len(plain.stops)]
        self.assertEqual([(s[0], s[1], s[3]) for s in head], [(s[0], s[1], s[3]) for s in plain.stops])  # 지도의 면은 그 상자가 없을 때와 같은 자리 · 순서
        self.assertEqual({k for stop in scout.stops[len(head):] for k in stop[3]}, set(keys))

    def test_dropped_faces_leave_the_stops_that_are_still_ahead(self):
        scout, fmap, grid, book = scout_on(MAP)
        scout.plan(START)
        n = len(scout.stops)
        before = [(stop[0], stop[1], list(stop[3])) for stop in scout.stops[:n]]
        _added, keys = add_box(scout, fmap, grid, book, *WALL_BOX)
        scout.drop_faces(keys)
        self.assertEqual([k for stop in scout.stops[n:] for k in stop[3]], [])
        self.assertEqual([(stop[0], stop[1], list(stop[3])) for stop in scout.stops[:n]], before)  # 지도의 면을 볼 자리는 그대로
        scout.i = n  # 지도의 면을 다 돌았다 → 빈 자리들은 지나치고 끝난다 (답을 못 얻은 면이 없다고 치려고 모두 적는다)
        for f in book.goal:
            book.votes[f.key] = {A.NON_OBJECT: 1.0}
        self.assertEqual(scout.step(START, 0.0), ("done",))
        self.assertEqual(scout.add_faces(START, keys), 0)  # 이미 적힌 면은 다시 넣지 않는다

    def test_a_plan_with_no_faces_takes_stops_added_later(self):
        """상자 없는 임시 지도 (벽만): 처음 계획은 비어 있고 정찰은 바로 끝난다. 그 뒤에 라이다로 찾은 상자의 면이 붙으면 이어서 간다."""
        scout, fmap, grid, book = scout_on([])
        self.assertEqual(scout.step(START, 0.0), ("done",))
        self.assertEqual(scout.stops, [])
        added, keys = add_box(scout, fmap, grid, book, *WALL_BOX)
        self.assertGreaterEqual(added, 1)
        act = scout.step(START, 1.0)
        self.assertEqual(act[0], "go")
        self.assertEqual((round(act[1], 3), round(act[2], 3)), (round(scout.stops[0][0], 3), round(scout.stops[0][1], 3)))
        self.assertEqual({k for stop in scout.stops for k in stop[3]}, set(keys))


class FlowTests(unittest.TestCase):
    def test_extra_faces_are_photographed_after_the_planned_ones_and_counted(self):
        w = tiny()
        w.lidar = [WALL_BOX]
        self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None), (w.task, w.mission.log, w.mission.scout.log))
        book = w.mission.book
        self.assertEqual(book.counts(), {"R-S1": {"tank": 1}, "R-S2": {}, "R-S3": {}, "CM": {"drum": 1, "mortar": 1}})
        self.assertEqual({side: book.kind((2, side)) for side in "EWS"}, {"E": A.NON_OBJECT, "W": A.NON_OBJECT, "S": "tank"})
        # 지도의 면을 처음 찍은 사진들이 모두, 지도에 없는 상자의 면을 처음 찍은 사진보다 앞이다 (덤은 맨 뒤)
        first = [k for k, tag in enumerate(w.shots) if tag[2] == 0]
        planned = [k for k in first if w.shots[k][1][0] < 2]
        extra = [k for k in first if w.shots[k][1][0] == 2]
        self.assertEqual(len(planned), 8)
        self.assertEqual(len(extra), 3)
        self.assertLess(max(planned), min(extra))
        self.assertFalse(w.mission._extra_off)
        self.assertFalse(w.said("예외"))

    def test_scouting_starts_again_when_the_box_shows_up_after_it_was_done(self):
        w = tiny()
        self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None))
        self.assertEqual(w.mission.book.counts()["R-S1"], {})
        w.run(1.0)
        self.assertIn(w.task, ("rest", "m6 go", "m6 watch"))  # 정찰이 끝나면 M6 순찰로 넘어간다 (2026-11-20, 끄면 rest)
        w.lidar = [WALL_BOX]
        w.run(4.3)
        self.assertIsNone(w.mission.scout_done_at)
        self.assertEqual(len(w.said("다시 정찰한다")), 1)
        self.assertTrue(w.task.startswith("scout"), w.task)
        self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None), (w.task, w.mission.scout.log))
        self.assertEqual(w.mission.book.counts()["R-S1"], {"tank": 1})
        self.assertEqual(len(w.said("정찰 끝")), 2)

    def test_nothing_new_once_the_robot_went_to_wait_at_the_base(self):
        w = tiny()
        self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None))
        done_at = w.mission.scout_done_at
        w.mission._staging = True  # 거점 앞 대기를 시작했다
        w.lidar = [WALL_BOX]
        w.run(6.0)
        self.assertEqual(len(w.mission.fmap.boxes), 2)  # 받아들이지 않는다 (정찰로 돌아가지 않으니 쓸 데가 없다)
        self.assertEqual(w.mission.scout_done_at, done_at)
        self.assertTrue(w.task.startswith("stage"), w.task)
        self.assertFalse(w.said("라이다"))

    def test_gone_flag_survives_when_navigation_reports_its_own_removed_boxes(self):
        """라이다 목록에서 사라진 상자(2번)의 「없음」 표시는, navigation 이 지도에서 뺀 상자 목록(처음 지도의 번호만 담는다)이 와도 지워지지 않는다."""
        w = tiny()
        w.lidar = [WALL_BOX]
        w.run(4.3)
        w.lidar = []
        w.run(6.3)  # Facts.gone 은 아직 온 적이 없다 (None)
        self.assertEqual(w.mission.fmap.gone, frozenset({2}))
        w.run(1.0, gone=frozenset())  # navigation: 뺀 상자 없음
        self.assertEqual(w.mission.fmap.gone, frozenset({2}))
        self.assertFalse(w.said("navigation 이 라이다로 없다고 본 상자"))  # 바뀐 것이 없으니 적지도 않는다
        w.run(1.0, gone=frozenset({1}))
        self.assertEqual(w.mission.fmap.gone, frozenset({1, 2}))
        self.assertEqual(w.said("navigation 이 라이다로 없다고 본 상자")[-1].split(" →")[0], "navigation 이 라이다로 없다고 본 상자: 1번")

    def test_a_real_map_arriving_later_starts_over(self):
        """상자 없는 임시 지도로 시작해 라이다 상자를 하나 받아들인 뒤에 상자가 든 지도가 왔다 (node 가 reset_map 을 부른다):
        앞에서 받아들인 것은 버리고, 새 지도에 그 상자가 있으면 다시 넣지 않고, 없으면 새 지도의 맨 뒤 번호로 다시 받아들인다."""
        for in_new_map in (True, False):
            world = A.field_map(MAP + [WALL_BOX])
            w = Tiny(world, truth_for(world, b2_S="tank"), [], fmap=A.field_map([]))
            w.lidar = [WALL_BOX]
            w.run(4.5)
            self.assertEqual(w.mission.fmap.boxes, [WALL_BOX])
            w.fmap = A.field_map(MAP + [WALL_BOX] if in_new_map else MAP)  # 새 지도 (node 는 새 FieldMap 을 만들어 넘긴다)
            w.mission.reset_map()
            w.run(0.2)
            self.assertEqual(len(w.mission.fmap.boxes), 3 if in_new_map else 2)
            self.assertEqual(w.mission._extras, {})
            w.run(6.0)
            self.assertEqual(w.mission.fmap.boxes, MAP + [WALL_BOX])
            self.assertEqual(sorted(w.mission._extras), [] if in_new_map else [2])
            self.assertEqual(len(w.said("지도에 있는 상자의 자리")), 1 if in_new_map else 0)
            self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None), (in_new_map, w.task))
            self.assertEqual(w.mission.book.counts()["R-S1"], {"tank": 1}, in_new_map)
            self.assertFalse(w.mission._extra_off)

    def test_walls_only_map_and_one_lidar_box(self):
        """perception 이 지도를 못 만들어 상자 없는 임시 지도로 시작한 경기: 라이다로 찾은 상자가 유일한 상자다 (0번)."""
        world = A.field_map([WALL_BOX])
        w = Tiny(world, truth_for(world, b0_S="missile"), [], fmap=A.field_map([]))
        w.run(1.0)
        self.assertEqual(w.mission.fmap.boxes, [])
        self.assertIsNotNone(w.mission.scout_done_at)  # 찍을 면이 없어 바로 끝난다
        w.lidar = [WALL_BOX]
        w.run(4.3)
        self.assertEqual(w.mission.fmap.boxes, [WALL_BOX])
        self.assertIsNone(w.mission.scout_done_at)
        self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None), (w.task, w.mission.scout.log))
        self.assertEqual(w.mission.book.counts()["R-S1"], {"missile": 1})
        self.assertEqual(len(w.shots), 3)
        self.assertFalse(w.mission._extra_off)

    def test_gone_when_it_leaves_the_list_and_back_when_it_returns(self):
        w = tiny()
        w.lidar = [WALL_BOX]
        w.run(4.3)
        mission = w.mission
        scout = mission.scout
        waiting = lambda: {k for stop in scout.stops[scout.i:] for k in stop[3] if k[0] == 2}  # noqa: E731 — 아직 갈 자리에 남은 2번 상자의 면
        self.assertEqual(waiting(), {(2, "E"), (2, "W"), (2, "S")})
        w.lidar = []
        w.run(5.8, gone=frozenset({0}))  # navigation 이 지도의 0번 상자도 뺐다
        self.assertEqual(mission.fmap.gone, frozenset({0}))  # 아직 6초가 안 됐다
        w.run(0.5, gone=frozenset({0}))
        self.assertEqual(mission.fmap.gone, frozenset({0, 2}))
        self.assertNotIn("extra2", mission.grid.closed)  # 격자를 푼다
        self.assertTrue(mission.grid.free(4.46, 4.5))
        self.assertEqual(waiting(), set())  # 그 면은 찍으러 가지 않는다
        self.assertEqual(len(w.said("없는 것으로 치고")), 1)
        self.assertEqual(w.said("navigation 이 라이다로 없다고 본 상자"), ["navigation 이 라이다로 없다고 본 상자: 0번 → 그 면은 맨 뒤로 미루고 사진 짝짓기에서 뺀다"])
        w.run(3.0, gone=frozenset({0}))
        self.assertEqual(mission.fmap.gone, frozenset({0, 2}))  # 틱마다 뒤집히지 않는다
        self.assertEqual(len(w.said("navigation 이 라이다로 없다고 본 상자")), 1)
        # 다시 나온다 (0번도 되살아났다)
        w.lidar = [(WALL_BOX[0] + 0.05, WALL_BOX[1])]
        w.run(0.2, gone=frozenset())
        self.assertEqual(mission.fmap.gone, frozenset())
        self.assertIn("extra2", mission.grid.closed)
        self.assertFalse(mission.grid.free(4.46, 4.5))
        self.assertEqual(len(mission.fmap.boxes), 3)  # 새 상자로 또 넣지 않는다
        self.assertEqual(len(w.said("라이다 목록에 다시 나옴")), 1)
        self.assertTrue(w.run_until(lambda: mission.scout_done_at is not None), (w.task, scout.log))
        self.assertEqual(mission.book.counts()["R-S1"], {"tank": 1})

    def test_an_error_inside_switches_it_off_and_the_match_goes_on(self):
        w = tiny()
        w.run(0.3)

        def broken(pose, want):
            raise RuntimeError("자리 고르기 실패")

        w.mission.scout._append = broken
        w.lidar = [WALL_BOX]
        w.run(4.5)
        self.assertTrue(w.mission._extra_off)
        self.assertEqual(len(w.said("예외")), 1)
        self.assertIn(2, w.mission.fmap.gone)  # 받아들였던 상자는 없는 것으로 친다
        self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None), (w.task, w.mission.scout.log))  # 계획한 정찰은 끝까지 간다
        self.assertEqual(w.mission.book.counts()["CM"], {"drum": 1, "mortar": 1})
        self.assertEqual(len(w.said("예외")), 1)

    def test_restart_ignores_saved_records_of_boxes_that_are_not_on_the_map(self):
        """경기 도중 다시 켜졌을 때: 저장해 둔 면 기록에는 뒤에 넣은 상자(2번)의 것도 있다. 새로 받은 지도에는 그 번호가 없다 → 그 기록은 버리고 지도의 면만 되살린다."""
        w = tiny()
        w.lidar = [WALL_BOX]
        self.assertTrue(w.run_until(lambda: w.mission.scout_done_at is not None))
        saved = w.mission.snapshot()
        self.assertIn("2:S", saved["book"]["votes"])
        again = tiny()
        again.mission.restore(saved)
        again.run(0.3)
        book = again.mission.book
        self.assertEqual(len(again.mission.fmap.boxes), 2)
        self.assertEqual(book.counts(), {"R-S1": {}, "R-S2": {}, "R-S3": {}, "CM": {"drum": 1, "mortar": 1}})
        self.assertFalse([k for k in book.votes if k[0] == 2])
        again.lidar = [WALL_BOX]  # 그 상자는 라이다 목록으로 다시 받아들이고, 면을 다시 찍는다
        again.run(4.3)
        self.assertEqual(len(again.mission.fmap.boxes), 3)
        self.assertIsNone(book.kind((2, "S")))
        self.assertTrue(again.run_until(lambda: again.mission.scout_done_at is not None))
        self.assertEqual(book.counts()["R-S1"], {"tank": 1})


class WholeMatchTests(unittest.TestCase):
    """가상 경기 한 판 (주행 · 인식 · 서버는 가짜). 실제 로봇 성능을 뜻하지 않는다."""

    EXACT = dict(mode="face", pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0)
    BOX = (4.86, 4.8, {"S": "tank", "E": "drum", "W": "mortar"})  # 위쪽 벽에 붙은 상자 (R-S1). 지도에는 없다

    @staticmethod
    def m3(m):
        return [r for r in m.reports if r[1] == 3][-1][2]

    def test_the_box_the_map_missed_is_photographed_and_counted_in_m3(self):
        m = SimMatch(slot="B-1", unmapped=[self.BOX], **self.EXACT).run()
        truth = m.m3_truth()
        self.assertEqual(truth["R-S1"], {"ammo_box": 1, "military_truck": 1, "tank": 1, "drum": 1, "mortar": 1})
        self.assertEqual(self.m3(m), truth, m.mission.log)
        number = m.mapped  # 지도의 상자 17개 뒤 = 17번
        self.assertEqual(m.mission.fmap.boxes[number], self.BOX[:2])
        self.assertEqual({side: m.mission.book.kind((number, side)) for side in "EWS"}, {"E": "drum", "W": "mortar", "S": "tank"})
        self.assertEqual(len([text for _t, text in m.mission.log if "지도에 없는 상자를 라이다로 찾음" in text]), 1)
        self.assertFalse(m.mission._extra_off)
        # 지정표적 사격 · 목표 거점 점령도 한다. 발 수는 따지지 않는다: 이 배치에서는 지정표적(상자 9)이 지도에 없는 상자 곁이라 다시 도는 차례에야
        # 읽히는 판이 있고, 덤의 자리 3곳을 그 앞에 들르는 몇 초 때문에 셋째 발이 거점 앞 대기 시각을 넘긴다 (계획 순서가 다른 판 7개 중 5개에서 2발)
        self.assertGreaterEqual(len(m.fires), 1)
        self.assertIsNotNone(m.captured_at)
        # 끄면 그 구역의 개수가 틀린다 (전의 동작)
        off = SimMatch(slot="B-1", unmapped=[self.BOX], params={"extra_boxes": False}, **self.EXACT).run()
        self.assertEqual(len(off.mission.fmap.boxes), number)
        self.assertEqual(self.m3(off)["R-S1"], {"ammo_box": 1, "military_truck": 1})

    def test_a_match_without_such_boxes_runs_exactly_as_before(self):
        """라이다 목록이 온 적 없을 때(전과 같다) · 빈 목록만 올 때 · 꺼 두고 목록이 올 때: 보고 · 하는 일 · 기록 · 정찰 자리가 모두 같다."""
        def run(lidar=None, **kw):
            m = SimMatch(mode="face", slot="R-2", **kw)
            if lidar is not None:
                m.lidar_boxes = lidar
            m.run()
            return m.reports, m.tasks, m.mission.log, m.mission.scout.log, m.mission.scout.stops, m.mission.scout.shots, m.robot.pose

        plain = run()
        self.assertEqual(run(lidar=lambda t: []), plain)
        self.assertEqual(run(lidar=lambda t: [(1.46, 4.8)], params={"extra_boxes": False}), plain)


if __name__ == "__main__":
    unittest.main()
