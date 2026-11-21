"""실시간 순서 (ScoutParams.order = "live"): 남은 자리의 순서를 경기 도중에 다시 정한다 (FaceScout._live_pick).

기본은 "tour" 그대로다. "live" 는 처음 계획이 "tour" 와 같고, 자리를 마칠 때 · 막혔을 때 · 가는 도중 사정이 바뀔 때
「거점 앞 대기를 시작할 때의 기대 점수」 가 가장 큰 순서의 첫 자리로 간다. 여기서는 그 규칙의 뼈대만 본다 (점수 비교는 가상 경기로 따로 했다).
"""
import itertools
import math
import random
import unittest

from sim_world import SimMatch
from supervisor import arena as A
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import FaceScout, ScoutParams
from test_blocked import Ports, scout_for

BOXES = [(1.43, 4.04), (1.86, 4.0), (4.27, 4.0), (4.67, 4.02), (3.05, 3.41), (1.45, 2.4), (1.85, 2.39), (4.25, 2.39), (4.63, 2.37),
         (3.04, 1.51), (1.47, 0.82), (1.87, 0.83), (4.27, 0.8), (4.64, 0.82)]
START = (0.36, 3.98, 0.0)


def planned(order, **params):
    fmap = A.field_map(BOXES, A.arena_from({"width": 6.12, "height": 4.80}), None)
    scout, _grid, book = scout_for(fmap, camp="BLUE", start=START)
    scout.p = ScoutParams(order=order, **params)
    scout.plan(START)
    return scout, book


def spots(scout):
    return [(round(s[0], 2), round(s[1], 2), tuple(s[3])) for s in scout.stops[scout.i:]]


class LiveOrderTests(unittest.TestCase):
    def test_default_is_tour_and_live_starts_from_the_same_plan(self):
        self.assertEqual(ScoutParams().order, "tour")
        tour, _b = planned("tour")
        live, _b = planned("live")
        self.assertEqual(spots(tour), spots(live))
        self.assertFalse(tour._fields)  # 기본 순서에서는 거리장을 접어 두지 않는다 (하는 일이 전과 같다)
        self.assertEqual(len(live._fields), len({live.grid.cell_of(s[0], s[1]) for s in live.stops}))

    def test_with_time_to_spare_the_planned_order_stays(self):
        """시간이 넉넉하면 어느 순서로 돌아도 기대 점수가 같다 → 계획한 순서(가장 짧은 길)를 바꾸지 않는다
        (어림한 시간이 live_shorter_s 넘게 주는 순서가 있을 때만 바꾼다 — 여기서는 그것을 꺼 두고 본다)."""
        live, _b = planned("live", live_shorter_s=1e9)
        before = spots(live)
        live.clock = (100.0, 5100.0, None)  # 경기 첫머리(한가운데를 피하는 때)는 지났다
        act = live.step(START, 0.0)
        self.assertEqual(spots(live), before)
        self.assertEqual((act[0], round(act[1], 2), round(act[2], 2)), ("go",) + before[0][:2])

    def test_short_of_time_it_finishes_the_sector_that_can_be_finished(self):
        """시간이 모자라면: 면 하나만 남은 구역의 자리부터 간다 (여러 자리가 남아 끝낼 수 없는 구역의 가까운 자리보다 먼저).
        M3 는 구역의 면을 모두 적어야 점수다 — 모든 구역을 한 면씩 모자라게 남기면 0점이다."""
        live, book = planned("live")
        cm = [s for s in live.stops if all(book.by_key[k].sector == "CM" for k in s[3])]
        rs1 = [s for s in live.stops if all(book.by_key[k].sector == "R-S1" for k in s[3])]
        self.assertGreaterEqual(len(cm), 3)
        pose = (cm[0][0], cm[0][1], 0.0)  # CM 의 자리 하나에 서 있다
        last = min(rs1, key=lambda s: math.hypot(s[0] - pose[0], s[1] - pose[1]))  # R-S1 에는 이 자리의 면 하나만 남긴다
        keep = {last[3][0]} | {k for s in cm for k in s[3]}
        for face in book.goal:
            if face.key not in keep:
                book.votes[face.key] = {A.NON_OBJECT: 1.0}
        live.stops = cm + [last]  # 계획한 순서: CM 의 자리들을 먼저, R-S1 의 한 면은 맨 뒤
        far = math.hypot(last[0] - pose[0], last[1] - pose[1])
        live.clock = (200.0, 200.0 + 1.3 * far / live.p.live_speed + 12.0, None)  # 그 한 자리는 갈 수 있고 CM 은 다 돌 수 없는 시간
        self.assertTrue(live._live(pose, 0.0))
        self.assertEqual(live.stops[live.i][3], last[3])
        self.assertIn("실시간 순서", live.log[-1])

    def test_a_spot_next_to_a_fresh_block_is_not_picked_while_clear_spots_remain(self):
        live, _b = planned("live")
        live.clock = (30.0, 232.0, None)
        act = live.step(START, 0.0)
        blocked = (act[1], act[2])
        live.skip_stop(5.0)  # 그 자리에 가지 못했다
        act = live.step(START, 5.1)
        self.assertEqual(act[0], "go")
        self.assertGreater(math.hypot(act[1] - blocked[0], act[2] - blocked[1]), live.p.avoid_m)
        self.assertFalse(live._near_block(act[1], act[2], 5.1))

    def test_no_runaway_when_only_blocked_spots_and_a_late_spot_remain(self):
        """고를 자리가 모두 방금 막힌 곳 가까이이고, 뒤에 생긴 면(지도에 없던 상자)의 자리만 막히지 않았다: 순서를 건드리지 않고 전의 규칙대로
        (막힌 자리는 뒤로 · 막히지 않은 자리부터) 한 번에 넘어간다. 2026-11-20 가상 경기: 막힌 자리를 맨 앞으로 당기기와 맨 뒤에 다시 붙이기가 한 틱에
        수십 번 되풀이돼 자리 목록이 결정 때마다 곱절로 불었다 (한 판에 순서 바꿈 8269번 — 틱이 0.1초를 넘겼다)."""
        live, book = planned("live")
        live.clock = (150.0, 232.0, None)
        a = live.stops[0]
        far = max(live.stops, key=lambda s: math.hypot(s[0] - a[0], s[1] - a[1]))
        for face in book.goal:
            if face.key not in set(a[3]) | set(far[3]):
                book.votes[face.key] = {A.NON_OBJECT: 1.0}
        live.late = set(far[3])  # 먼 자리의 면은 「뒤에 생긴 면」 이다 (실시간 순서가 고르지 않는 자리)
        live.stops = [a, far]
        pose = (a[0] - 0.5, a[1], 0.0)
        self.assertEqual(live.step(pose, 0.0)[:3], ("go", a[0], a[1]))
        live.skip_stop(1.0)  # 그 자리에 가지 못했다
        told = len(live.log)
        act = live.step(pose, 1.1)
        self.assertEqual(act[:3], ("go", far[0], far[1]))  # 막히지 않은 자리부터
        self.assertLessEqual(len(live.stops), 4)  # 막힌 자리는 한 번만 뒤에 붙는다
        self.assertFalse([line for line in live.log[told:] if "실시간 순서" in line])
        for k in range(20):  # 가는 동안에도 불어나지 않는다
            live.step(pose, 1.2 + 0.1 * k)
        self.assertLessEqual(len(live.stops), 4)

    def test_a_robot_standing_on_the_next_spot_sends_it_elsewhere_first(self):
        """다음 자리에 서 있는 로봇(지도에 없는 라이다 상자)이 있으면 그 자리는 뒤로 돌리고 다른 자리부터 간다 — 버리지는 않는다."""
        live, _b = planned("live")
        live.clock = (30.0, 232.0, None)
        first = live.stops[0]
        live.lidar_boxes = [(first[0] + 0.1, first[1])]
        act = live.step(START, 0.0)
        self.assertEqual(act[0], "go")
        self.assertNotEqual((round(act[1], 2), round(act[2], 2)), (round(first[0], 2), round(first[1], 2)))
        self.assertIn(tuple(first[3]), [s[2] for s in spots(live)])

    def test_a_robot_seen_from_above_on_the_next_spot_sends_it_elsewhere_first(self):
        """위에서 본 서 있는 로봇(mission 이 넣어 주는 scout.robots)도 라이다 상자와 같게 본다 — 라이다 목록에 로봇이 없어도 그 곁의 자리를 뒤로 돌린다."""
        live, _b = planned("live")
        live.clock = (30.0, 232.0, None)
        first = live.stops[0]
        live.lidar_boxes = []
        live.robots = [(first[0] + 0.1, first[1], 12.0)]
        act = live.step(START, 0.0)
        self.assertEqual(act[0], "go")
        self.assertNotEqual((round(act[1], 2), round(act[2], 2)), (round(first[0], 2), round(first[1], 2)))
        self.assertIn(tuple(first[3]), [s[2] for s in spots(live)])

    def one_stop_left_in_time(self, stood, **params):
        """면 하나만 남은 구역의 자리 A(곁에 stood 초 서 있던 로봇)와, 그것만으로는 끝나지 않는 구역의 가까운 자리 B. 한 자리만 갈 시간이 있다.
        고른 첫 자리가 A 인지를 돌려준다."""
        live, book = planned("live", live_scenarios=((1.0, 1.0),), **params)
        single = [s for s in live.stops if len(s[3]) == 1]
        a = next(s for s in single if book.by_key[s[3][0]].sector == "R-S1")
        b = next(s for s in single if book.by_key[s[3][0]].sector == "R-S3")
        others = [f.key for f in book.goal if f.sector == "R-S3" and f.key != b[3][0]][:2]  # B 의 구역에는 갈 자리가 없는 면 둘이 더 남는다
        for face in book.goal:
            if face.key not in (a[3][0], b[3][0]) + tuple(others):
                book.votes[face.key] = {A.NON_OBJECT: 1.0}
        live.stops = [a, b]
        gap = math.hypot(a[0] - b[0], a[1] - b[1])
        pose = (b[0] + 0.8 * (a[0] - b[0]) / gap, b[1] + 0.8 * (a[1] - b[1]) / gap, 0.0)  # B 에서 A 쪽으로 0.8 m 인 곳에 서 있다 (B 를 들르면 A 는 늦는다)
        p = live.p
        there = live._dist((pose[0], pose[1]), (a[0], a[1])) / p.live_speed + p.live_stop_s + p.live_face_s
        live.clock = (100.0, 100.0 + there + p.live_block_s + 0.5, None)  # A 로 바로 가면 막혀서 기다리는 시간까지 쳐도 닿는다
        live.robots = [(a[0] + 0.2, a[1], stood)]
        live._live(pose, 0.0)
        return live.stops[live.i][3] == a[3]

    def test_a_robot_that_stood_long_is_not_expected_to_leave_soon(self):
        """「늦게 닿을수록 막힐 가능성이 준다」 를 켜 두면(live_robot_half_s 20초 — 전의 기본값): 곁의 로봇이 방금 선 것이면 닿을 때쯤 떠났을 수 있다
        → 구역을 끝낼 그 자리로 간다. 2분째 서 있는 로봇이면 떠나지 않을 것으로 본다 → 그 자리는 두고 다른 자리로 간다 (00:00 연습 판)."""
        self.assertTrue(self.one_stop_left_in_time(stood=5.0, live_robot_half_s=20.0))
        self.assertFalse(self.one_stop_left_in_time(stood=120.0, live_robot_half_s=20.0))

    def test_by_default_waiting_does_not_make_a_stop_beside_a_robot_better(self):
        """기본값은 그 셈을 끈 것이다 (2026-11-20 03:28 연습 판: 끝까지 서 있던 로봇 곁의 자리가 「나중에 갈수록 좋은」 자리가 되어 3 m 씩 네 번 가로질렀다)
        → 방금 선 로봇이어도 그 곁의 자리를 먼저 고르지 않는다."""
        self.assertGreater(ScoutParams().live_robot_half_s, 1.0e5)
        self.assertFalse(self.one_stop_left_in_time(stood=5.0))
        self.assertFalse(self.one_stop_left_in_time(stood=120.0))

    def test_blockers_are_merged_and_our_own_position_is_ignored(self):
        live, _b = planned("live")
        live.lidar_boxes = [(3.0, 2.0), (4.05, 1.0), (0.40, 3.90)]  # 둘째는 위에서 본 로봇과 같은 것, 셋째는 우리 자리 그 자체 (0.09 m)
        live.robots = [(4.0, 1.0, 30.0), (0.4, 4.0, 9.0), (0.62, 3.98, 60.0)]  # 둘째는 우리 차 자신 (0.05 m), 셋째는 우리 차에 붙어 선 로봇 (0.26 m — 뺀 것이 아니다)
        self.assertEqual(sorted(live._robots_now(START)), [(0.62, 3.98, 60.0), (3.0, 2.0, None), (4.0, 1.0, 30.0)])
        live.lidar_boxes, live.robots = "깨진 값", [("x",)]
        self.assertEqual(live._robots_now(START), [])

    def test_a_change_in_the_robots_seen_calls_for_a_new_choice_on_the_way(self):
        live, _b = planned("live")
        live.clock = (100.0, 232.0, None)
        live.step(START, 0.0)
        self.assertFalse(live._live_due(5.0))
        live.robots = [(3.0, 2.4, 6.0)]
        self.assertTrue(live._live_due(5.0))
        self.assertFalse(live._live_due(0.5))  # 1초에 한 번만 본다

    def test_tour_and_best_do_not_look_at_robots(self):
        for order in ("tour", "best"):
            plain, _b = planned(order)
            with_robots, _b = planned(order)
            first = with_robots.stops[0]
            with_robots.lidar_boxes, with_robots.robots = [(first[0], first[1])], [(first[0] + 0.1, first[1], 90.0)]
            with_robots.clock = (30.0, 232.0, None)
            self.assertEqual(plain.step(START, 0.0), with_robots.step(START, 0.0))
            self.assertEqual(spots(plain), spots(with_robots))

    def test_any_exception_falls_back_to_the_planned_order(self):
        live, _b = planned("live")
        before = spots(live)

        def broken(pose, now, moving=False):
            raise RuntimeError("시험")

        live._live_pick = broken
        act = live.step(START, 0.0)
        self.assertEqual((act[0], round(act[1], 2), round(act[2], 2)), ("go",) + before[0][:2])
        self.assertEqual(spots(live), before)
        self.assertTrue(any("실시간 순서 예외" in line for line in live.log))
        for _ in range(3):
            live._live(START, 1.0)
        self.assertTrue(live._live_off)  # 세 번이면 이 경기에서는 더 쓰지 않는다

    def test_the_spot_of_the_likely_target_is_not_reordered_away(self):
        """지정표적으로 보이는 면의 자리(_take_lead)는 실시간 순서가 건드리지 않는다."""
        live, _b = planned("live")
        live.clock = (30.0, 232.0, None)
        far = live.stops[-1]
        live._lead = far[3][0]
        act = live.step(START, 0.0)
        self.assertEqual((act[0], round(act[1], 2), round(act[2], 2)), ("go", round(far[0], 2), round(far[1], 2)))

    def test_distances_come_from_the_folded_fields(self):
        live, _b = planned("live")
        a, b = live.stops[0], live.stops[-1]
        field_ = live.grid.distances((a[0], a[1]))
        want = field_[live.grid.cell_of(b[0], b[1])]
        self.assertAlmostEqual(live._dist((a[0], a[1]), (b[0], b[1])), want, delta=0.25)  # 어느 쪽에서 재나 거의 같다
        live._fields.clear()
        self.assertAlmostEqual(live._dist((a[0], a[1]), (b[0], b[1])), 1.3 * math.hypot(a[0] - b[0], a[1] - b[1]), places=6)


# 2026-11-20 00:52 연습 판 (연습 2경기장, R-1): 드론 지도의 상자 12개와, 라이다로 뒤에 찾은 상자 둘 (지도 번호 12 · 13 — B-S1 의 상자는 이 둘뿐이다)
RUN14_BOXES = [(4.236, 4.0), (4.649, 3.986), (3.061, 3.155), (1.455, 2.413), (1.851, 2.413), (4.271, 2.387), (4.681, 2.39), (3.071, 1.981),
               (1.479, 0.788), (1.874, 0.794), (4.284, 0.803), (4.701, 0.792)]
RUN14_EXTRA = [(1.889, 4.461), (1.402, 4.429)]
OLD_RULES = dict(live_late=False, live_slack=0.0)  # 그 판에서 돌던 규칙 (견준 순서 넷 · 5초 문턱 · 뒤에 생긴 면은 맨 뒤)


class Run14Tests(unittest.TestCase):
    """그 판의 끝: 자리 셋(+ 뒤에 생긴 상자의 자리 넷)이 남았을 때 서 → 동 → 서로 8 m 를 오갔다 (M3 는 30/30 이었고, 시간만 썼다).

    124.9초 (3.14, 3.73): 1.1 m 앞의 CM 자리 (3.83, 2.83) 를 두고 2.5 m 서쪽의 (0.83, 2.83) 부터 갔다. 기록의 값으로 그 결정을 그대로 다시 만들면:
    남은 세 자리의 여섯 순서는 모두 기대 점수 14.5 였고, 그때 견준 순서 넷(지금 순서 49.1초 · 가까운 것부터 45.1초 · 47.2초 · 47.6초)에
    가장 짧은 순서(41.1초 — CM 자리부터)가 없었다. 가장 짧은 것이 지금 순서보다 4.0초 짧아 「5초 넘게 짧을 때만 바꾼다」 에 걸려 그대로 갔다.
    지금 순서가 서쪽 자리부터였던 것은, 표적 자리를 마친 뒤 남은 자리를 「먼저 돌 구역 B-S1」 의 자리(뒤에 생긴 상자의 것)부터 다시 이었기 때문이다.
    138.7초 (1.34, 3.03): 뒤에 생긴 상자의 자리 둘이 1 m 안에 있는데 「뒤에 생긴 면은 맨 뒤」 라서 동쪽 (3.83, 2.83) 을 다녀온 뒤에야 갔다.
    """

    def scene(self, left, pose, when, unseen=(), **params):
        mission, ports = Mission(MissionParams(scout=ScoutParams(mode="face", order="live", **params))), Ports()
        m1 = {"target": "BLUE_RADAR", "search_sectors": ["B-S1", "CM"], "base": "BLUE_MAINTENANCE"}
        for k in range(60):  # 6초: 지도 → 계획 → 4초 이어서 보인 라이다 상자 둘을 받아들인다 (로봇은 출발 칸에서 첫 자리로 가는 중)
            mission.tick(Facts(now=0.1 * k, t=0.1 * k, phase="running", control=True, slot="R-1", pose=(5.945, 4.017, math.pi), arrived=False,
                               boxes=RUN14_BOXES, extra_boxes=RUN14_EXTRA, parsed=m1), ports)
        scout, book = mission.scout, mission.book
        self.assertEqual({k[0] for k in scout.late}, {12, 13})
        self.assertEqual([self.spot_of(scout, key) for key in ((3, "W"), (2, "W"), (7, "N"))], [(0.83, 2.83), (2.23, 3.62), (3.83, 2.83)])  # 그 판의 자리들
        stops = [s for s in scout.stops if set(s[3]) & set(left)]
        for face in book.goal:
            if face.key not in left and face.box not in unseen:  # unseen = 아직 어느 사진에도 면이 찍힌 적 없는 「뒤에 생긴 상자」
                book.votes[face.key] = {A.NON_OBJECT: 1.0}
        scout.want_kind(None)  # 지정표적은 찾았다 (100.9초)
        scout.stops = scout._order(pose, stops)  # 표적 자리를 마친 뒤 남은 자리를 다시 이은 순서 (먼저 돌 구역 B-S1 의 자리부터)
        scout.i, scout.body, scout.queue = 0, None, []
        scout.clock = (when, 232.0, None)
        scout._live(pose, when)
        return scout

    @staticmethod
    def spot_of(scout, key):
        return next((round(s[0], 2), round(s[1], 2)) for s in scout.stops if key in s[3])

    def test_at_124_9_s_the_near_centre_spot_comes_first_and_the_rest_is_one_sweep_west(self):
        left = {(2, "W"), (7, "N"), (3, "W"), (12, "W"), (12, "S"), (13, "E"), (13, "W"), (13, "S")}
        pose = (3.14, 3.73, 2.5)
        scout = self.scene(left, pose, 124.4)  # 상자 12 는 표적 면을 찍었고, 상자 13 도 사진에 찍힌 적이 있다
        order = [(round(s[0], 2), round(s[1], 2)) for s in scout.stops]
        self.assertEqual(order[0], self.spot_of(scout, (7, "N")))  # 1.1 m 앞의 CM 자리부터
        self.assertEqual(order[0], (3.83, 2.83))
        self.assertGreaterEqual(order.index(self.spot_of(scout, (3, "W"))), len(order) - 3)  # 가장 서쪽의 자리들은 끝에
        xs = [x for x, _y in order]
        self.assertTrue(all(b <= a + 0.3 for a, b in zip(xs, xs[1:])), order)  # 동쪽에서 서쪽으로 한 번에 — 되돌아가지 않는다
        # 상자 13 이 아직 어느 사진에도 안 찍혔으면 그 자리들은 전처럼 맨 뒤에 남는다 (로봇을 상자로 잘못 본 것일 수 있다). 첫 자리는 그래도 CM 자리다
        wary = self.scene(left, pose, 124.4, unseen=(13,))
        self.assertEqual((round(wary.stops[0][0], 2), round(wary.stops[0][1], 2)), (3.83, 2.83))
        tail = [s for s in wary.stops if any(k[0] == 13 for k in s[3])]
        self.assertEqual(wary.stops[-len(tail):], tail)
        # 그 판에서 돌던 규칙으로는 서쪽 끝 자리부터였다 (기록과 같다)
        old = self.scene(left, pose, 124.4, **OLD_RULES)
        self.assertEqual((round(old.stops[0][0], 2), round(old.stops[0][1], 2)), self.spot_of(old, (3, "W")))
        self.assertIn("기대 점수 14.5 → 14.5", old.log[-1])
        self.assertIn("고를 자리 3곳", old.log[-1])

        def length(sc):
            pts = [pose[:2]] + [(s[0], s[1]) for s in sc.stops if set(s[3]) & set(left)]
            return sum(sc._dist(a, b) for a, b in zip(pts, pts[1:]))

        self.assertLess(length(scout), length(old) - 1.5)  # 남은 자리를 다 도는 길이 1.5 m 넘게 짧다

    def test_at_138_7_s_the_late_box_spots_next_to_the_robot_come_before_the_far_centre_spot(self):
        left = {(7, "N"), (12, "W"), (13, "W")}
        pose = (1.34, 3.03, 0.5)
        scout = self.scene(left, pose, 138.7)
        first = (scout.stops[0][0], scout.stops[0][1])
        self.assertLess(math.hypot(first[0] - pose[0], first[1] - pose[1]), 1.3)
        self.assertEqual(self.spot_of(scout, (7, "N")), (round(scout.stops[-1][0], 2), round(scout.stops[-1][1], 2)))
        old = self.scene(left, pose, 138.7, **OLD_RULES)
        self.assertEqual((round(old.stops[0][0], 2), round(old.stops[0][1], 2)), self.spot_of(old, (7, "N")))  # 전에는 2.5 m 동쪽의 자리부터

    def test_shortest_order_is_exact(self):
        rng = random.Random(7)
        for n in (3, 4, 5, 6, 7):
            pts = {k: (rng.uniform(0, 6), rng.uniform(0, 4.8)) for k in range(10, 10 + n)}
            start = (rng.uniform(0, 6), rng.uniform(0, 4.8))

            def d(a, b):
                p, q = (start if a is None else pts[a]), pts[b]
                return math.hypot(p[0] - q[0], p[1] - q[1])

            def length(seq):
                return sum(d(a, b) for a, b in zip([None] + list(seq[:-1]), seq))

            got = FaceScout._shortest(list(pts), d)
            self.assertEqual(sorted(got), sorted(pts))
            self.assertAlmostEqual(length(got), min(length(seq) for seq in itertools.permutations(pts)), places=9)


class LiveMatchTests(unittest.TestCase):
    def test_a_whole_match_with_parked_robots_runs_through(self):
        """가상 경기 한 판 (정찰 자리 곁에 다른 로봇 둘이 서 있다): 예외 없이 끝까지 가고, mission 이 남은 시간을 넘겨 주며, M3 를 보고한다."""
        m = SimMatch(mode="face", slot="R-2", parked=[(3.9, 2.5), (2.3, 1.2)], scout={"order": "live"}).run()
        scout = m.mission.scout
        self.assertEqual(scout._live_errors, 0)
        self.assertFalse(any("실시간 순서 예외" in text for _t, text in m.mission.log))
        self.assertIsNotNone(scout.clock)
        self.assertAlmostEqual(scout.clock[1], m.mission.p.target_at_s - m.mission.p.stage_margin_s)
        self.assertTrue(any(kind == 3 for _t, kind, _payload in m.reports))
        self.assertGreaterEqual(m.mission.book.coverage(), 0.8)


if __name__ == "__main__":
    unittest.main()
