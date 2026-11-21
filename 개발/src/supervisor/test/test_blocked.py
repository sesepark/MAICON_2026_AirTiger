"""막힘 (2026-11-19): 고정 기둥, 다른 UGV 가 선 정찰 자리, 거점 진입. 2026-11-18 팀 테스트 3차에서 본 일을 다시 만든다.

3차의 막힘 세 번 (로그_파일/1118_팀테스트_경기장1_R2/상황.md):
  - 정찰 자리 (4.03, 2.43) · (4.03, 2.83): 다른 팀 로봇이 (3.99, 2.64) 쯤에 서 있었다 → 20 cm · 54 cm 앞에서 21초씩 서 있다 건너뜀
  - 정찰 자리 (2.62, 4.62): 위쪽 벽 가운데의 고정 기둥(지도에 없음)에서 0.24 m → 0.47 m 앞에서 15초 서 있다 건너뜀
"""
import math
import unittest

from sim_world import SimMatch
from supervisor import arena as A
from supervisor.faces import FaceBook
from supervisor.grid import Grid
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import FaceScout, ScoutParams

# 3차(23:39) perception 지도의 상자
BOXES3 = [(1.401, 3.931), (1.844, 3.957), (4.295, 4.169), (4.719, 4.169), (3.319, 3.718), (4.734, 3.756), (3.071, 2.78), (1.825, 2.483),
          (2.803, 1.501), (3.216, 1.528), (1.336, 1.159), (1.719, 1.176), (4.177, 1.218), (4.154, 0.831)]


def pillar_gap(fmap, x, y):
    """(x, y) 에서 가장 가까운 기둥 면까지 (기둥 안이면 0)."""
    half = fmap.box_size / 2.0
    return min(math.hypot(max(abs(x - px) - half, 0.0), max(abs(y - py) - half, 0.0)) for px, py in fmap.pillars)


def scout_for(fmap, camp="RED", start=(5.9, 2.5, math.pi)):
    grid = Grid(fmap)
    for value, (_x0, y0, _x1, y1) in fmap.bases.items():
        rect = (0.0, y0 - 0.15, 0.45, y1 + 0.15) if fmap.base_camp(value) == "BLUE" else (fmap.width - 0.45, y0 - 0.15, fmap.width, y1 + 0.15)
        grid.add_zone(value, rect, closed=True)
    book = FaceBook(fmap, camp)
    return FaceScout(fmap, grid, book, params=ScoutParams(), start=start), grid, book


class Ports:
    """node.py 의 창구 흉내: 같은 목표를 다시 시키면 보내지 않고, 보낸 뒤에는 도착(reach)을 알리거나 이동을 거둘 때까지 「가는 중」 이다."""

    def __init__(self):
        self.goes, self.stops, self.shots = [], 0, []
        self.goal, self.reached = None, False

    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None):
        goal = (round(x, 2), round(y, 2), round(yaw, 2), mode)
        if goal == self.goal:
            return
        self.goal, self.reached = goal, False
        self.goes.append({"x": round(x, 2), "y": round(y, 2), "yaw": yaw, "yaw_free": yaw_free, "mode": mode, "yaw_tol": yaw_tol, "pos_tol": pos_tol})

    def stop(self):
        if self.goal is None:
            return
        self.goal = None
        self.stops += 1

    def reach(self):
        """navigation 이 도착을 알렸다."""
        self.reached = True

    def fail(self):
        """navigation 이 못 갔다고 알렸다 → (x, y, 까닭) 을 돌려준다 (Facts.go_failed 에 넣는다)."""
        goal, self.goal = self.goal, None
        return (goal[0], goal[1], 3)

    @property
    def arrived(self):
        return self.goal is None or self.reached

    def aim(self, *a): return True
    def fire(self): return True

    def shoot(self, tag, pan_deg):
        self.shots.append((tag, pan_deg))
        return True

    def look_ugvs(self): return False
    def analyze_order(self, text): return True
    def set_ready(self): return True
    def report(self, kind, payload): return True
    def aim_ready(self): return True


class PillarTests(unittest.TestCase):
    def test_pillars_block_the_grid_but_make_no_faces(self):
        fmap = A.field_map(BOXES3)
        self.assertEqual(len(fmap.pillars), 10)
        self.assertEqual({f.box for f in fmap.faces}, set(range(len(BOXES3))))  # 면은 임무 상자에만
        grid = Grid(fmap)
        self.assertFalse(grid.free(3.06, 4.50))  # 위쪽 가운데 기둥(아랫면 y = 4.60) + 차체 여유
        self.assertFalse(grid.free(0.50, 5.0 / 3.0))
        self.assertTrue(grid.free(3.06, 4.30))
        self.assertEqual(len(fmap.segments()), 4 + 4 * (len(BOXES3) + 10))  # 시야 가림에는 들어간다
        self.assertEqual(A.field_map(BOXES3, A.arena_from({"pillars": False})).pillars, [])
        self.assertEqual(A.fixed_pillars(3.0, 2.0), [])  # 공식 크기가 아니면 넣지 않는다

    def test_scout_stops_keep_away_from_pillars(self):
        """3차에서는 (2.62, 4.62) — 위쪽 가운데 기둥 면에서 0.24 m — 에 서려고 했다. 이제 기둥에서 차체 여유 + 0.10 m 안에는 서지 않는다."""
        fmap = A.field_map(BOXES3)
        scout, _grid, _book = scout_for(fmap)
        scout.plan((5.9, 2.5, math.pi))
        self.assertGreater(len(scout.stops), 10)
        for x, y, _heading, _keys in scout.stops:
            self.assertGreaterEqual(pillar_gap(fmap, x, y), 0.26, (x, y))
        self.assertFalse(scout.unreachable)


class BodyHeadingTests(unittest.TestCase):
    def test_heading_only_has_to_keep_the_faces_within_pan_reach(self):
        """면 하나를 찍는 자리: 차체는 그 면 쪽 ±56° 안이면 된다 (팬이 겨눈다 — 팬 끝을 조금 넘는 면도 팬을 끝까지 돌려 찍는다).
        3차에서는 ±5.7° 를 맞추느라 자리마다 평균 6.5초를 썼고, 2026-11-19 16:01 실차는 ±40° 에서도 방향 맞추기에 정찰의 27% 를 썼다."""
        fmap = A.field_map([(3.06, 2.5)])
        scout, _grid, book = scout_for(fmap, camp="BLUE", start=(0.2, 2.5, 0.0))
        key = (0, "W")
        scout.stops, scout._planned = [(2.26, 2.5, 0.0, [key])], True
        act = scout.step((1.0, 2.5, 0.0), 0.0)
        self.assertEqual(act[0], "go")
        self.assertAlmostEqual(act[3], 0.0, places=3)
        self.assertAlmostEqual(math.degrees(act[4]), 56.0, places=3)
        # 30° 틀어진 채 자리에 서 있으면 더 돌지 않고 팬으로 겨눠 찍는다
        act = scout.step((2.26, 2.5, math.radians(30.0)), 1.0)
        self.assertEqual(act[0], "shoot")
        self.assertAlmostEqual(act[1], 30.0, places=1)  # 팬은 오른쪽이 +
        # 두 면을 찍는 자리는 그만큼 좁아진다: 가운데 방향으로, 남는 만큼만
        scout2, _g, _b = scout_for(A.field_map([(3.06, 2.5), (3.06, 3.5)]), camp="BLUE", start=(0.2, 2.5, 0.0))
        scout2.stops, scout2._planned = [(2.26, 3.0, 0.0, [(0, "W"), (1, "W")])], True
        act = scout2.step((1.0, 3.0, 0.0), 0.0)
        spread = 2.0 * math.degrees(math.atan2(0.5, 0.6))
        self.assertAlmostEqual(math.degrees(act[4]), max(ScoutParams().min_body_tol_deg, 56.0 - spread / 2.0), places=3)

    def test_mission_passes_the_tolerance_to_navigation(self):
        mission, ports = Mission(MissionParams()), Ports()
        now = 0.0
        for _ in range(8):
            mission.tick(Facts(now=now, t=now, phase="running", control=True, slot="R-2", pose=(5.9, 2.5, math.pi), boxes=BOXES3, arrived=ports.arrived), ports)
            now += 0.1
        self.assertTrue(ports.goes)
        tol = ports.goes[-1]["yaw_tol"]
        self.assertIsNotNone(tol)
        self.assertGreaterEqual(tol, 0.10)
        self.assertLessEqual(tol, math.radians(52.0) + 1e-6)  # 56° 에서 4° 를 좁혀 넘긴다


class BlockedStopTests(unittest.TestCase):
    def setUp(self):
        self.fmap = A.field_map([(3.06, 2.5), (3.06, 3.5)])
        self.scout, self.grid, self.book = scout_for(self.fmap, camp="BLUE", start=(0.2, 2.5, 0.0))
        self.scout.stops = [(2.26, 2.5, 0.0, [(0, "W")]), (2.26, 2.9, 0.0, [(0, "W"), (1, "W")]), (3.06, 1.6, math.pi / 2.0, [(0, "S")])]
        self.scout._planned = True

    def test_shoots_from_where_it_stands_when_the_spot_is_taken(self):
        """자리 (2.26, 2.5) 에 다른 UGV 가 서 있어 0.4 m 앞에서 멈췄다. 볼 면(상자 0 의 W)은 거기서도 보인다 → 그 자리에서 찍는다."""
        scout = self.scout
        self.assertEqual(scout.step((1.2, 2.5, 0.0), 0.0)[0], "go")
        here = (1.86, 2.5, math.radians(10.0))
        self.assertTrue(scout.shoot_here(here, 10.0))
        act = scout.step(here, 10.1)
        self.assertEqual(act[0], "shoot")
        self.assertEqual(act[2][1], (0, "W"))
        scout.shot_taken(act[2], 10.2)
        self.assertEqual(scout.here_shots, 1)

    def test_nearby_stops_are_put_off_after_a_block(self):
        """막힌 자리에서 0.4 m 떨어진 다음 자리는 같은 물체에 또 막힌다 (3차: 21초 + 20초) → 뒤로 미루고 먼 자리부터 간다."""
        scout = self.scout
        scout.step((1.2, 2.5, 0.0), 0.0)
        scout.skip_stop(10.0)  # 선 자리에서 보이는 면이 없었다고 치고 건너뛴다
        act = scout.step((1.2, 2.5, 0.0), 10.1)
        self.assertEqual((act[0], round(act[1], 2), round(act[2], 2)), ("go", 3.06, 1.6))  # (2.26, 2.9) 는 미뤄졌다
        self.assertEqual([(round(s[0], 2), round(s[1], 2)) for s in scout.stops[3:]], [(2.26, 2.5), (2.26, 2.9)])  # 둘 다 맨 뒤에서 한 번 더
        # 한참 뒤(피하는 시간 40초가 지난 뒤)에는 다시 간다. 한 번 미룬 자리는 또 미루지 않는다
        scout.i = 3
        act = scout.step((3.06, 1.6, 0.0), 60.0)
        self.assertEqual((round(act[1], 2), round(act[2], 2)), (2.26, 2.5))

    def test_a_spot_put_off_once_still_waits_while_clear_spots_remain(self):
        """한 번 미뤘던 자리도, 막힌 지 얼마 안 됐고 막히지 않은 자리가 남아 있으면 그 뒤로 보낸다 (모의 6차: 같은 자리에 세 번 가서 세 번 막힘)."""
        scout = self.scout
        scout.step((1.2, 2.5, 0.0), 0.0)
        scout.skip_stop(10.0)  # (2.26, 2.5) 에서 막힘
        act = scout.step((1.2, 2.5, 0.0), 10.1)
        self.assertEqual((round(act[1], 2), round(act[2], 2)), (3.06, 1.6))
        scout.stops.append((3.86, 2.5, math.pi, [(0, "E")]))  # 막힌 곳에서 먼 자리가 하나 더 남아 있다
        scout.i = 3  # (3.06, 1.6) 을 마쳤다 → 다음은 미뤄 둔 (2.26, 2.5)
        scout.queue, scout.body = [], None
        act = scout.step((3.06, 1.6, 0.0), 20.0)  # 막힌 지 10초
        self.assertEqual((act[0], round(act[1], 2), round(act[2], 2)), ("go", 3.86, 2.5))  # 먼 자리부터
        self.assertEqual([(round(s[0], 2), round(s[1], 2)) for s in scout.stops[scout.i + 1:]], [(2.26, 2.5), (2.26, 2.9)])  # 미룬 둘은 그 뒤에 남아 있다
        # 막히지 않은 자리가 더 없으면 기다리지 않고 간다 (전과 같다)
        scout.i += 1
        scout.queue, scout.body = [], None
        act = scout.step((3.86, 2.5, math.pi), 25.0)
        self.assertEqual((act[0], round(act[1], 2), round(act[2], 2)), ("go", 2.26, 2.5))

    def test_mission_uses_both_rules_when_navigation_reports_failure(self):
        mission, ports = Mission(MissionParams()), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5), (3.06, 3.5)])
        pose = (1.2, 2.5, 0.0)
        mission.tick(Facts(now=0.0, t=0.0, pose=pose, arrived=ports.arrived, **base), ports)
        mission.tick(Facts(now=0.1, t=0.1, pose=pose, arrived=ports.arrived, **base), ports)
        goal = ports.goes[-1]
        # 그 자리에서 0.3 m 못 미친 곳에 서서 「못 갔다」 를 받는다
        short = (goal["x"] - 0.3 * math.cos(goal["yaw"]), goal["y"] - 0.3 * math.sin(goal["yaw"]), goal["yaw"])
        mission.tick(Facts(now=9.0, t=9.0, pose=short, arrived=True, go_failed=ports.fail(), **base), ports)
        self.assertEqual(mission.scout.here_shots + mission.scout.skipped, 1)
        if mission.scout.here_shots:  # 면이 보이면 그 자리에서 찍는다
            for k in range(1, 12):
                mission.tick(Facts(now=9.0 + 0.1 * k, t=9.0 + 0.1 * k, pose=short, arrived=ports.arrived, **base), ports)
            self.assertTrue(ports.shots)


class WobbleAndEscapeTests(unittest.TestCase):
    """2026-11-19 실차(연습 경기장 2번) 두 판에서 본 것."""

    def test_no_go_stop_flip_when_the_pose_wobbles_at_the_arrive_limit(self):
        """목표에서 0.13 ~ 0.17 m 인 곳에 서서 위치 값이 도착 한도(0.15 m)를 넘나든다. 실차에서는 같은 목표가 68초 동안 81번 나갔다.
        한 번 「왔다」 고 본 뒤에는 그 자리로 치고, 이동을 한 번 거둔 다음 찍는다."""
        mission, ports = Mission(MissionParams()), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5)])
        mission.tick(Facts(now=0.0, t=0.0, pose=(1.2, 2.5, 0.0), arrived=ports.arrived, **base), ports)
        mission.tick(Facts(now=0.1, t=0.1, pose=(1.2, 2.5, 0.0), arrived=ports.arrived, **base), ports)
        goal = ports.goes[-1]
        now, sent = 0.2, len(ports.goes)
        for k in range(60):  # 6초 동안 한도 안팎을 오간다. navigation 은 「도착」 을 알리지 않는다 (목표에서 0.12 m 밖이라)
            off = 0.13 if k % 2 == 0 else 0.17
            pose = (goal["x"] - off * math.cos(goal["yaw"]), goal["y"] - off * math.sin(goal["yaw"]), goal["yaw"])
            mission.tick(Facts(now=now, t=now, pose=pose, arrived=ports.arrived, **base), ports)
            now += 0.1
            if ports.shots:
                break
        again = [g for g in ports.goes[sent:] if (g["x"], g["y"]) == (goal["x"], goal["y"])]
        self.assertEqual(again, [], "한 번 온 자리로 다시 가라고 했다 (%d번)" % len(again))  # 실차에서는 81번
        self.assertEqual(ports.stops, 1)  # 이동을 거둔 것은 한 번
        self.assertTrue(ports.shots)  # 그 자리에서 찍었다

    def test_an_arrival_a_little_short_counts_as_arrived(self):
        """navigation 이 목표에서 0.19 m 인 곳에서 「도착」 으로 끝냈다 (REACHED [POS_NOT_MET] — 허용을 조금 넘었지만 고치려면 크게 돌아야 해서
        그 자리에서 끝냄). 2026-11-20 03:28 연습 판(2경기장 R-3)의 83초: 이것을 「가지 못함」 으로 쳐서 그 자리를 뒤로 미루고 막힌 곳으로 적었고,
        70초 뒤에 다시 와서 찍었다. 이제 arrive_slack_m 안이면 온 것으로 치고 그 자리에서 찍는다. 그보다 멀거나 스위치를 끄면 전과 같다."""
        def match(off, **params):
            mission, ports = Mission(MissionParams(**params)), Ports()
            base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5)])
            mission.tick(Facts(now=0.0, t=0.0, pose=(1.2, 2.5, 0.0), arrived=ports.arrived, **base), ports)
            mission.tick(Facts(now=0.1, t=0.1, pose=(1.2, 2.5, 0.0), arrived=ports.arrived, **base), ports)
            goal, sent = ports.goes[-1], len(ports.goes)
            pose = (goal["x"] - off * math.cos(goal["yaw"]), goal["y"] - off * math.sin(goal["yaw"]), goal["yaw"])
            ports.reach()  # navigation: 「도착」 (못 갔다는 알림이 아니다)
            now = 8.0
            for _ in range(15):
                mission.tick(Facts(now=now, t=now, pose=pose, arrived=ports.arrived, **base), ports)
                now += 0.1
            again = [g for g in ports.goes[sent:] if (g["x"], g["y"]) == (goal["x"], goal["y"])]
            return mission.scout, ports, again

        scout, ports, again = match(0.19)
        self.assertEqual((scout.skipped, scout.here_shots, list(scout.blocks)), (0, 0, []))  # 못 간 것 · 막힌 곳으로 치지 않는다
        self.assertEqual(again, [])  # 같은 자리로 다시 가라고 하지 않는다
        self.assertTrue(ports.shots)  # 선 자리에서 그 자리의 면을 찍었다
        self.assertEqual(ports.shots[0][0][0], "face")
        # 한도(arrive_m + arrive_slack_m = 0.27 m)보다 멀면 전과 같다: 선 자리에서 보이는 면을 찍거나 그 자리를 뒤로 미룬다
        scout, ports, _again = match(0.30)
        self.assertEqual(scout.skipped + scout.here_shots, 1)
        self.assertEqual(len(scout.blocks), 1)
        # 스위치를 끄면 0.19 m 도 전과 같다
        scout, ports, _again = match(0.19, short_arrival_ok=False)
        self.assertEqual(scout.skipped + scout.here_shots, 1)

    def test_backs_out_along_its_trail_after_failing_twice_in_place(self):
        """다른 로봇과 상자 사이에 들어가 어떤 목표도 못 간다 (실차: 100초). 두 번 잇달아 못 갔고 그 자리 그대로면 지나온 길로 되돌아간다."""
        mission, ports = Mission(MissionParams()), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5)])
        now = 0.0
        for k in range(14):  # (1.0, 1.0) 에서 (2.3, 1.0) 까지 지나온다
            mission.tick(Facts(now=now, t=now, pose=(1.0 + 0.1 * k, 1.0, 0.0), arrived=ports.arrived, **base), ports)
            now += 0.5
        here = (2.3, 1.0, 0.0)
        for _ in range(2):  # 같은 자리에서 두 번 「못 갔다」
            mission.tick(Facts(now=now, t=now, pose=here, arrived=True, go_failed=ports.fail(), **base), ports)
            now += 0.1
            mission.tick(Facts(now=now, t=now, pose=here, arrived=ports.arrived, **base), ports)
            now += 8.0
        self.assertEqual(mission.task, "escape")
        back = ports.goes[-1]
        self.assertTrue(back["yaw_free"])
        self.assertAlmostEqual(back["y"], 1.0, places=2)
        self.assertLessEqual(back["x"], 2.3 - 0.8 + 0.21)  # 지나온 길로 0.8 m 넘게 되돌아간 자리
        self.assertGreaterEqual(back["x"], 1.0)
        # 그 자리에 오면 빠져나오기를 끝내고 하던 일(정찰)로 돌아간다
        ports.reach()
        mission.tick(Facts(now=now, t=now, pose=(back["x"] + 0.1, 1.0, math.pi), arrived=ports.arrived, **base), ports)
        mission.tick(Facts(now=now + 0.1, t=now + 0.1, pose=(back["x"] + 0.1, 1.0, math.pi), arrived=ports.arrived, **base), ports)
        self.assertTrue(mission.task.startswith("scout"), mission.task)
        self.assertEqual(mission.escapes, 1)

    def test_escape_tells_the_scout_which_spot_failed(self):
        """두 번째 실패에서 빠져나오기가 시작되면 정찰은 그 실패를 듣지 못한다 → 빠져나온 뒤 같은 자리를 또 시키지 않게 여기서 미룬다."""
        mission, ports = Mission(MissionParams()), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5), (3.06, 3.5)])
        now = 0.0
        for k in range(14):
            mission.tick(Facts(now=now, t=now, pose=(1.0 + 0.1 * k, 1.0, 0.0), arrived=ports.arrived, **base), ports)
            now += 0.5
        here = (2.3, 1.0, 0.0)
        mission.tick(Facts(now=now, t=now, pose=here, arrived=True, go_failed=ports.fail(), **base), ports)
        now += 0.1
        mission.tick(Facts(now=now, t=now, pose=here, arrived=ports.arrived, **base), ports)
        second = (ports.goes[-1]["x"], ports.goes[-1]["y"])  # 두 번째로 시킨 정찰 자리
        skipped = mission.scout.skipped
        now += 8.0
        mission.tick(Facts(now=now, t=now, pose=here, arrived=True, go_failed=ports.fail(), **base), ports)  # 그것도 못 갔다 → 빠져나오기
        self.assertEqual(mission.task, "escape")
        self.assertEqual(mission.scout.skipped, skipped + 1)  # 정찰도 그 자리를 못 간 것으로 안다
        back = ports.goes[-1]
        ports.reach()
        now += 3.0
        for k in range(3):
            mission.tick(Facts(now=now + 0.1 * k, t=now + 0.1 * k, pose=(back["x"], back["y"], math.pi), arrived=ports.arrived, **base), ports)
        self.assertTrue(mission.task.startswith("scout"), mission.task)
        self.assertNotEqual((ports.goes[-1]["x"], ports.goes[-1]["y"]), second)  # 빠져나온 뒤에는 다른 자리부터

    def _stuck_and_trail_escape_fails(self, **params):
        """(2.3, 1.0) 에서 동쪽을 본 채 두 번 못 갔고, 지나온 길로 되돌아가는 것도 「못 갔다」 로 끝났다 (자리는 그대로)."""
        mission, ports = Mission(MissionParams(**params)), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5)])
        now = 0.0
        for k in range(14):
            mission.tick(Facts(now=now, t=now, pose=(1.0 + 0.1 * k, 1.0, 0.0), arrived=ports.arrived, **base), ports)
            now += 0.5
        here = (2.3, 1.0, 0.0)
        for _ in range(2):
            mission.tick(Facts(now=now, t=now, pose=here, arrived=True, go_failed=ports.fail(), **base), ports)
            now += 0.1
            mission.tick(Facts(now=now, t=now, pose=here, arrived=ports.arrived, **base), ports)
            now += 8.0
        self.assertEqual(mission.task, "escape")
        self.assertEqual(ports.goes[-1]["mode"], "normal")  # 먼저 지나온 길로
        mission.tick(Facts(now=now, t=now, pose=here, arrived=True, go_failed=ports.fail(), **base), ports)  # 그것도 못 갔다
        return mission, ports, base, here, now + 0.1

    def test_backs_off_straight_when_the_trail_escape_also_fails_in_place(self):
        """2026-11-19 모의 경기 3차: 위쪽 통로에서 다른 로봇과 0.25 m 를 두고 마주 선 채 130초 — 지나온 길로 가는 것도 못 갔다.
        그때는 차체 방향 그대로 뒤로 곧게 물러나 사이를 벌린다."""
        mission, ports, base, here, now = self._stuck_and_trail_escape_fails()
        self.assertEqual(mission.task, "escape")
        back = ports.goes[-1]
        self.assertEqual(back["mode"], "straight")
        self.assertAlmostEqual(back["x"], here[0] - 0.40, places=2)  # 동쪽을 보고 있었다 → 서쪽으로 0.40 m
        self.assertAlmostEqual(back["y"], here[1], places=2)
        self.assertTrue(any("뒤로" in text for _t, text in mission.log))
        # 물러난 자리에 오면 하던 일로 돌아간다
        ports.reach()
        there = (back["x"], back["y"], 0.0)
        mission.tick(Facts(now=now, t=now, pose=there, arrived=ports.arrived, **base), ports)
        mission.tick(Facts(now=now + 0.1, t=now + 0.1, pose=there, arrived=ports.arrived, **base), ports)
        self.assertTrue(mission.task.startswith("scout"), mission.task)

    def test_no_back_off_when_switched_off_or_no_room_behind(self):
        mission, ports, _base, _here, _now = self._stuck_and_trail_escape_fails(back_off_m=0.0)
        self.assertNotEqual(ports.goes[-1]["mode"], "straight")
        # 뒤가 벽이면 물러나지 않는다
        mission, ports = Mission(MissionParams()), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5)])
        mission.tick(Facts(now=0.0, t=0.0, pose=(0.3, 1.0, 0.0), arrived=ports.arrived, **base), ports)
        self.assertIsNone(mission._back_off((0.2, 1.0, 0.0)))  # 서쪽 벽에서 0.2 m, 동쪽을 본다 → 뒤는 벽

    def _stuck_after_leaving_our_base(self, t0):
        """B-2 출발 칸(BLUE_MAINTENANCE, x 0 ~ 0.4 · y 1.9 ~ 3.1)에서 나와 (1.3, 2.5) 까지 왔다가 두 번 잇달아 못 갔다. t0 = 그때의 경기 시각."""
        mission, ports = Mission(MissionParams()), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5)])
        now = 0.0
        for k in range(12):  # (0.2, 2.5) 에서 (1.3, 2.5) 까지 — 지나온 길은 출발 칸 안에서 시작한다
            mission.tick(Facts(now=now, t=t0 - 20.0 + now, pose=(0.2 + 0.1 * k, 2.5, 0.0), arrived=ports.arrived, **base), ports)
            now += 0.5
        here = (1.3, 2.5, 0.0)
        for _ in range(2):
            mission.tick(Facts(now=now, t=t0 + now, pose=here, arrived=True, go_failed=ports.fail(), **base), ports)
            now += 0.1
            mission.tick(Facts(now=now, t=t0 + now, pose=here, arrived=ports.arrived, **base), ports)
            now += 8.0
        return mission, ports

    def test_does_not_back_into_our_own_base_after_the_target_is_posted(self):
        """목표거점 게시(4:00) 뒤에는 우리 UGV 가 우리 거점에 들어가도 상대의 점령이 된다 (경기서버연동API.md, 2026-11-19 규정 검토).
        지나온 길로 0.8 m 되돌아간 자리가 우리 출발 칸 곁(빔을 가릴 수 있는 0.20 m 안)이면 그리로 가지 않는다."""
        mission, ports = self._stuck_after_leaving_our_base(t0=235.0)
        own = list(mission.fmap.bases_of("BLUE").values())
        for g in ports.goes:
            self.assertFalse(any(x0 - 0.2 <= g["x"] <= x1 + 0.2 and y0 - 0.2 <= g["y"] <= y1 + 0.2 for x0, y0, x1, y1 in own), g)
        self.assertNotEqual(mission.task, "escape")

    def test_may_back_into_the_start_base_before_the_target_is_posted(self):
        """게시 전(1:00)에는 거점에 들어가도 아무 일 없다 — 전처럼 지나온 길(출발 칸 쪽)로 되돌아간다."""
        mission, ports = self._stuck_after_leaving_our_base(t0=60.0)
        self.assertEqual(mission.task, "escape")
        self.assertLessEqual(ports.goes[-1]["x"], 1.3 - 0.8 + 0.21)

    def test_straight_out_does_not_drive_into_our_own_base(self):
        """돌 자리가 없어 곧게 빠져나갈 때도, 게시 뒤에는 우리 거점 쪽은 고르지 않는다 (앞이 우리 거점 · 뒤는 덜 비어도 뒤로)."""
        mission, ports = Mission(MissionParams()), Ports()
        base = dict(phase="running", control=True, slot="B-2", boxes=[(1.6, 2.5)])
        pose = (0.85, 2.5, math.pi)  # 우리 거점(x 0 ~ 0.4)을 보고 섰다. 뒤(동쪽) 0.55 m 에 상자 면 (x 1.40) — 뒤로는 0.25 m 만 빈다
        mission.tick(Facts(now=0.0, t=235.0, pose=pose, arrived=ports.arrived, **base), ports)
        avoid = mission._own_base_risk(Facts(now=0.1, t=235.0, pose=pose, **base))
        goal = mission._straight_out(pose, avoid)
        self.assertIsNotNone(goal)
        self.assertGreater(goal[0], pose[0])  # 뒤로 (동쪽)
        before = mission._straight_out(pose, mission._own_base_risk(Facts(now=0.1, t=60.0, pose=pose, **base)))
        self.assertLess(before[0], pose[0])  # 게시 전이면 전처럼 더 멀리 빈 앞쪽 (거점 쪽)


class ParkedRobotMatchTests(unittest.TestCase):
    @staticmethod
    def plan(slot="R-2"):
        """서 있는 UGV 가 없을 때의 정찰 자리들 (같은 프로세스에서는 같은 계획이 다시 나온다)."""
        probe = SimMatch(mode="face", slot=slot)
        stops = []
        tick = probe.mission.tick

        def grab(f, ports):
            out = tick(f, ports)
            if not stops and probe.mission.scout is not None and getattr(probe.mission.scout, "_planned", False):
                stops.extend((s[0], s[1]) for s in probe.mission.scout.stops)
            return out

        probe.mission.tick = grab
        probe.run(running_s=30.0)
        return stops

    @staticmethod
    def match(parked, old_rules=False):
        m = SimMatch(mode="face", slot="R-2", parked=parked)
        tick = m.mission.tick
        done = []

        def wrapped(f, ports):
            if old_rules and not done and m.mission.scout is not None:  # 2026-11-18 의 규칙: 못 간 자리는 그냥 건너뛴다
                scout = m.mission.scout
                skip = scout.skip_stop
                scout.shoot_here = lambda pose, now: False
                scout._near_block = lambda x, y, now: False
                scout.skip_stop = lambda now=None, retry=True: skip(now, retry=False)
                done.append(True)
            return tick(f, ports)

        m.mission.tick = wrapped
        return m.run()

    def test_match_goes_on_around_robots_parked_on_scout_spots(self):
        """정찰 계획의 자리 세 곳에 다른 UGV 가 서 있다 (3차처럼). 그 자리마다 8초 뒤 「못 갔다」 가 온다.
        선 자리에서 찍고 · 옆 자리를 미루며 경기를 끝까지 치른다. 못 간 자리를 건너뛰기만 하던 때보다 사진이 줄지 않는다
        (가상 12판의 합: 사진 295 → 357장, 못 적은 면 150 → 100개)."""
        stops = self.plan()
        self.assertGreater(len(stops), 9)
        parked = [stops[2], stops[3], stops[8]]
        old = self.match(parked, old_rules=True)
        m = self.match(parked)
        scout = m.mission.scout
        self.assertGreaterEqual(m.robot.blocked_goals, 2)
        self.assertGreaterEqual(scout.here_shots + scout.skipped, m.robot.blocked_goals - 1)
        self.assertGreaterEqual(scout.here_shots, 1, scout.log[-6:])
        self.assertGreaterEqual(scout.shots, old.mission.scout.shots - 2, (scout.shots, old.mission.scout.shots, scout.log[-6:]))
        self.assertTrue([r for r in m.reports if r[1] == 6])  # 보고 단계까지 갔다


class BaseEntryTests(unittest.TestCase):
    def setUp(self):
        self.mission, self.ports = Mission(MissionParams()), Ports()
        self.boxes = [(3.06, 2.5)]

    def tick(self, now, pose, **more):
        return self.mission.tick(Facts(now=now, t=241.0 + now, phase="running", control=True, slot="R-2", pose=pose, boxes=self.boxes,
                                       objective="BLUE_COMMAND_POST", **more), self.ports)

    def front(self, x=0.6):
        """목표 거점의 바로 앞 (거점의 세로 가운데)."""
        _x0, y0, _x1, y1 = self.mission_bases()["BLUE_COMMAND_POST"]
        return (x, (y0 + y1) / 2.0, math.pi)

    def mission_bases(self):
        if self.mission.fmap is None:  # 지도는 첫 틱에 만들어진다
            self.tick(0.0, (0.6, 2.5, math.pi), arrived=True)
        return self.mission.fmap.bases

    def test_enters_deep_without_asking_for_a_heading(self):
        here = self.front()
        self.tick(0.0, here, arrived=True)
        self.tick(0.1, here, arrived=True)
        go = self.ports.goes[-1]
        x0, y0, x1, y1 = self.mission.fmap.bases["BLUE_COMMAND_POST"]
        self.assertEqual((go["mode"], go["yaw_free"]), ("assault", True))
        self.assertAlmostEqual(go["x"], 0.20, places=2)  # 차체 가운데가 벽에서 0.20 m (앞이 7.5 cm)
        self.assertGreaterEqual(go["y"], y0 + 0.40 - 1e-6)  # 거점 끝의 기둥에서 떨어진 자리
        self.assertLessEqual(go["y"], y1 - 0.40 + 1e-6)
        # 거점 네모에 발만 들인 것(깊이 0.38 m)으로는 서지 않는다 — 센서 빔의 깊이를 모른다
        self.assertEqual(self.tick(1.0, (0.38, go["y"], math.pi), arrived=False), "assault")
        self.assertEqual(self.ports.stops, 0)
        # 깊이 0.30 m 안이면 들어간 것 → 선다
        self.assertEqual(self.tick(2.0, (0.27, go["y"] + 0.1, math.radians(150.0)), arrived=False), "assault in")
        self.assertEqual(self.ports.stops, 1)

    def test_tries_another_spot_inside_when_the_first_is_taken(self):
        here = self.front()
        self.tick(0.0, here, arrived=True)
        self.tick(0.1, here, arrived=True)
        first = self.ports.goes[-1]
        self.assertEqual(first["mode"], "assault")
        self.tick(9.0, (0.8, here[1] + 0.1, math.pi / 2.0), arrived=True, go_failed=(first["x"], first["y"], 3))
        second = self.ports.goes[-1]
        self.assertEqual(second["mode"], "assault")
        self.assertGreater(abs(second["y"] - first["y"]), 0.15)

    def test_dashes_to_the_first_base_but_moves_on_to_the_next_one_normally(self):
        """첫 거점으로는 대기 자리에서 바로 돌진한다 (몇 초 싸움). 목표가 아니어서 다른 거점으로 옮겨 갈 때는 그 거점 앞까지 보통 주행
        (연습 판에서 거점 깊숙한 곳에서 돌진으로 옮겨 가다 상자를 밀었다)."""
        bases = self.mission_bases()
        _x0, y0, _x1, y1 = bases["BLUE_COMMAND_POST"]
        other = next(rect for name, rect in bases.items() if name.startswith("BLUE_") and name != "BLUE_COMMAND_POST")
        waiting = (0.72, (other[1] + other[3]) / 2.0, math.pi)  # 다른 거점 앞의 대기 자리 (목표 거점의 앞은 아니다)
        self.tick(1.0, waiting, arrived=True)
        self.tick(1.1, waiting, arrived=True)
        first = self.ports.goes[-1]
        self.assertEqual((first["mode"], round(first["x"], 2)), ("assault", 0.20))  # 첫 거점: 바로 돌진
        self.assertTrue(y0 <= first["y"] <= y1)
        # 들어갔는데 2초가 지나도 경기가 끝나지 않는다 → 다음 거점으로: 그 앞까지는 보통 주행
        inside = (0.2, first["y"], math.pi)
        self.tick(5.0, inside, arrived=True)
        self.tick(7.2, inside, arrived=True)
        self.tick(7.3, inside, arrived=True)
        move = self.ports.goes[-1]
        self.assertEqual(move["mode"], "normal")
        self.assertAlmostEqual(move["x"], MissionParams().close_from_wall_m, places=2)  # 벽에서 0.72 m
        self.assertFalse(y0 <= move["y"] <= y1)  # 다른 거점의 앞
        # 그 거점의 세로 범위 안 · 벽에서 1.1 m 안에 들어서면 돌진으로 바꾼다
        self.tick(12.0, (0.9, move["y"], math.pi), arrived=False)
        dash = self.ports.goes[-1]
        self.assertEqual((dash["mode"], round(dash["x"], 2), dash["y"]), ("assault", 0.20, move["y"]))

    def test_does_not_flip_between_normal_and_dash_on_the_edge_of_the_front_zone(self):
        """02:23 연습 판: 다음 거점의 「바로 앞」 을 가르는 금 위에서 서서 도는 동안 위치가 몇 mm 흔들려 보통 · 돌진 목표를 1.3초에 여섯 번 바꿔 보냈다."""
        bases = self.mission_bases()
        self.tick(0.5, self.front(), arrived=True)  # 목표 거점으로의 계획이 선다 (첫 거점)
        order = self.mission._assault_plan[1]
        self.mission._assault_plan[2] = 1  # 첫 거점이 목표가 아니었다 → 둘째 거점으로
        self.mission._assault_goal = None
        _x0, y0, _x1, y1 = bases[order[1]]
        edge = y0 - MissionParams().assault_side_m  # 금 (아래쪽에서 올라온다)
        modes = []
        for n, dy in enumerate((-0.006, 0.002, -0.004, 0.002, -0.006, 0.014, -0.003, 0.004)):
            self.tick(10.0 + 0.1 * n, (0.88, edge + dy, math.pi / 2.0), arrived=False)
            modes.append(self.ports.goes[-1]["mode"])
        self.assertEqual(modes[0], "normal")  # 금 아래: 거점 앞까지 보통 주행
        first_dash = modes.index("assault")
        self.assertEqual(set(modes[first_dash:]), {"assault"}, modes)  # 한 번 돌진을 냈으면 되돌리지 않는다
        self.assertLessEqual(len(self.ports.goes) - 1, 3, [g["mode"] for g in self.ports.goes])  # 목표를 번갈아 다시 보내지 않았다

    def test_comes_to_the_front_normally_when_far_from_the_bases_and_retries_another_spot(self):
        far = (2.4, 2.5, math.pi)  # 경기장 안쪽 (늦었다) — 첫 거점이어도 긴 길을 돌진으로 가지 않는다
        self.tick(0.0, far, arrived=True)
        self.tick(0.1, far, arrived=True)
        first = self.ports.goes[-1]
        self.assertEqual(first["mode"], "normal")
        self.assertAlmostEqual(first["x"], MissionParams().close_from_wall_m, places=2)
        self.tick(8.0, far, arrived=True, go_failed=(first["x"], first["y"], 3))  # 거점 앞까지 못 갔다 → 그 거점의 다른 자리 앞으로
        self.assertEqual(self.ports.goes[-1]["mode"], "normal")
        self.assertGreater(abs(self.ports.goes[-1]["y"] - first["y"]), 0.15)


class RobotInTheBaseTests(unittest.TestCase):
    """들어갈 거점 안에 다른 로봇이 서 있을 때 (2026-11-20 01:50 연습 판: 후보 자리가 모두 그 로봇 곁이라 네 번 막히고 33초가 걸렸다)."""

    def setUp(self):
        self.mission, self.ports = Mission(MissionParams()), Ports()

    def tick(self, now, pose, **more):
        return self.mission.tick(Facts(now=now, t=241.0 + now, phase="running", control=True, slot="R-2", pose=pose, boxes=[(3.06, 2.5)],
                                       objective="BLUE_COMMAND_POST", **more), self.ports)

    def first_entry(self, robots):
        self.tick(0.0, (0.8, 2.5, math.pi), arrived=True)  # 지도 · 정찰이 만들어진다
        x0, y0, x1, y1 = self.mission.fmap.bases["BLUE_COMMAND_POST"]
        mid = (y0 + y1) / 2.0
        self.mission._assault_goal = self.mission._assault_plan = None
        self.mission.scout.robots = [(x, mid + dy, 60.0) for x, dy in robots]
        here = (0.8, mid + 0.1, math.pi)
        self.tick(0.1, here, arrived=True)
        self.tick(0.2, here, arrived=True)
        return self.ports.goes[-1], (y0, y1, mid), self.mission._assault_goal[1]

    def test_enters_beside_a_robot_standing_in_the_middle_of_the_base(self):
        go, (y0, y1, mid), spots = self.first_entry([(0.26, -0.05)])
        # 2026-11-20 아침에 바뀐 곳: 전에는 고른 자리에서 0.28 m 옆(그 로봇이 차체 앞 띠에 걸리는 곳)에 선 채 바로 돌진을 냈다.
        # 이제 서 있는 로봇이 있는 거점에는 그 자리의 바로 앞(통로)에 먼저 서고(보통 주행), 선 뒤에 같은 자리로 돌진한다 (LeanBeforeDashTests)
        self.assertEqual((go["mode"], go["yaw_free"]), ("normal", False))
        self.assertAlmostEqual(go["x"], MissionParams().close_from_wall_m, places=2)
        self.assertGreaterEqual(abs(go["y"] - (mid - 0.05)), 0.30)  # 그 로봇에서 세로로 0.30 m 넘게
        self.assertTrue(y0 + 0.21 <= go["y"] <= y1 - 0.21)  # 거점 끝의 기둥에서는 0.22 m 안쪽
        self.assertGreater(len(spots), 2)  # 못 가면 볼 다음 자리들이 남아 있다
        self.assertTrue(any("다른 로봇이 서 있음" in text for _t, text in self.mission.log))
        self.tick(3.0, (0.8, go["y"] - 0.03, go["yaw"]), arrived=False)  # 그 앞에 먼 쪽으로 기울여 섰다 → 돌진
        dash = self.ports.goes[-1]
        self.assertEqual((dash["mode"], dash["yaw_free"], dash["y"]), ("assault", True, go["y"]))
        self.assertAlmostEqual(dash["x"], 0.20, places=2)

    def test_dashes_at_once_as_before_when_the_guard_is_switched_off(self):
        self.mission = Mission(MissionParams(assault_guard=False))
        go, (_y0, _y1, mid), _spots = self.first_entry([(0.26, -0.05)])
        self.assertEqual(go["mode"], "assault")  # 전의 동작: 0.28 m 옆에서 바로 돌진
        self.assertGreaterEqual(abs(go["y"] - (mid - 0.05)), 0.30)
        self.assertIsNone(self.mission._assault_guard)

    def test_order_is_unchanged_without_a_robot_in_or_before_the_base(self):
        plain, _span, spots_plain = self.first_entry([])
        self.setUp()
        far, _span, spots_far = self.first_entry([(2.4, 0.0), (0.26, 1.6)])  # 경기장 안쪽의 로봇 · 옆 거점 안의 로봇
        self.assertEqual((plain["y"], spots_plain), (far["y"], spots_far))

    def test_remembers_a_robot_seen_standing_in_the_base_a_little_while_ago(self):
        """거점 안의 로봇은 드론 화면 끝에 있어 목록에 들어왔다 빠졌다 한다 (02:24 연습 판) → 요즘 60초 안에 본 자리도 본다."""
        self.tick(0.0, (0.8, 2.5, math.pi), arrived=True)
        x0, y0, x1, y1 = self.mission.fmap.bases["BLUE_COMMAND_POST"]
        mid = (y0 + y1) / 2.0
        for age, avoided in ((30.0, True), (100.0, False)):
            self.mission._assault_goal = self.mission._assault_plan = None
            self.mission.scout.robots = []
            self.mission._stood = {(1, 12): (0.26, mid, 10.0 - age)}
            here = (0.8, mid + 0.02, math.pi)
            self.ports.goal = None
            self.tick(10.0, here, arrived=True)
            self.tick(10.1, here, arrived=True)
            first = self.mission._assault_goal[1][0]
            self.assertEqual(abs(first - mid) >= 0.30, avoided, (age, first, mid))

    def test_nothing_breaks_when_the_robot_list_is_odd(self):
        self.tick(0.0, (0.8, 2.5, math.pi), arrived=True)
        self.mission.scout.robots = ["?", (1,)]
        self.tick(0.1, (0.8, 2.5, math.pi), arrived=True)
        self.tick(0.2, (0.8, 2.5, math.pi), arrived=True)
        self.assertEqual(self.ports.goes[-1]["mode"], "assault")


class LeanBeforeDashTests(unittest.TestCase):
    """들어갈 거점에 다른 로봇이 서 있으면 자리의 바로 앞에 「그 로봇의 먼 쪽으로 기울여」 선 뒤에 돌진한다 (2026-11-20 아침, MissionParams.assault_guard).

    가상 주행(고정 기둥이 있는 지도, 60판): 차체가 그 로봇 쪽으로 기운 채 받은 돌진은 1초 만에 BLOCKED — 5° 48판 · 22° 38판 · 60 ~ 90° 32판만 들어갔다.
    벽을 똑바로 보거나 먼 쪽으로 기울면 60판 모두. 거점 사이를 옮겨 갈 때(통로를 따라와 차체가 벽과 나란히 그 로봇 쪽) · 다른 자리로 다시 들어갈 때가 그 장면이다.
    """

    BASE = "BLUE_MAINTENANCE"  # 가운데 거점 — 목표(BLUE_COMMAND_POST, 맨 위)가 아니었을 때 다음으로 들어가 볼 거점

    def setUp(self):
        self.start()

    def start(self, **params):
        self.mission, self.ports = Mission(MissionParams(**params)), Ports()

    def tick(self, now, pose, **more):
        more.setdefault("arrived", self.ports.arrived)
        return self.mission.tick(Facts(now=now, t=241.0 + now, phase="running", control=True, slot="R-2", pose=pose, boxes=[(3.06, 2.5)],
                                       objective="BLUE_COMMAND_POST", **more), self.ports)

    def second_base(self, robot_dy=0.03, robot_x=0.26):
        """첫 거점은 목표가 아니었다 → 가운데 거점으로 옮겨 가는 중이고, 그 거점 한가운데쯤에 다른 로봇이 서 있다. → (거점의 y0, y1, 그 로봇의 y)"""
        self.tick(0.0, (0.8, 4.2, math.pi))  # 지도 · 계획이 선다 (첫 거점으로는 바로 돌진)
        self.assertEqual(self.mission._assault_plan[1][1], self.BASE)
        _x0, y0, _x1, y1 = self.mission.fmap.bases[self.BASE]
        robot_y = (y0 + y1) / 2.0 + robot_dy
        self.mission.scout.robots = [(robot_x, robot_y, 60.0)]
        self.mission._assault_plan[2] = 1
        self.mission._assault_goal = None
        return y0, y1, robot_y

    def modes(self):
        return [g["mode"] for g in self.ports.goes]

    def test_stops_in_front_of_the_spot_leaning_away_before_the_dash(self):
        y0, y1, robot_y = self.second_base()
        down = -math.pi / 2.0  # 위의 거점에서 통로를 따라 내려온다 — 차체는 벽과 나란히, 그 로봇 쪽을 본다
        self.tick(10.0, (0.80, y1 + 0.30, down))
        go = self.ports.goes[-1]
        self.assertEqual((go["mode"], go["yaw_free"]), ("normal", False))
        self.assertAlmostEqual(go["x"], 0.80, places=2)
        self.assertAlmostEqual(go["y"], y1 - 0.22, places=2)  # 우리 쪽(위)의 자리 — 그 로봇에서 0.30 m 넘게
        self.assertAlmostEqual(math.cos(go["yaw"]), -math.cos(math.radians(20.0)), places=6)  # 벽 쪽에서
        self.assertAlmostEqual(math.sin(go["yaw"]), math.sin(math.radians(20.0)), places=6)  # 그 로봇(아래)의 먼 쪽으로 20°
        self.assertAlmostEqual(math.degrees(go["yaw_tol"]), 10.0, places=6)
        sent = len(self.ports.goes)
        # 거점의 세로 범위에 들어섰다 (전에는 여기서 돌진을 냈다 — 차체가 그 로봇 쪽으로 90°) → 아직 내지 않는다
        self.tick(11.0, (0.80, y1 + 0.02, down))
        self.tick(11.5, (0.80, go["y"] + 0.02, down))  # 자리 앞에 왔지만 아직 그 로봇 쪽을 본다
        self.tick(11.8, (0.80, go["y"] + 0.02, math.radians(-150.0)))  # 도는 중: 그 로봇 쪽으로 30°
        self.assertEqual(len(self.ports.goes), sent, self.modes())
        self.tick(12.0, (0.80, go["y"] + 0.02, math.radians(179.0)))  # 벽을 지나 먼 쪽으로 1° → 돌진
        dash = self.ports.goes[-1]
        self.assertEqual((dash["mode"], dash["yaw_free"], dash["y"]), ("assault", True, go["y"]))
        self.assertAlmostEqual(dash["x"], 0.20, places=2)
        # 돌진을 낸 뒤에는 차체가 다시 그 로봇 쪽으로 기울어도(돌며 넘어감) 보통 목표로 되돌리지 않는다
        self.tick(12.1, (0.78, go["y"] + 0.02, math.radians(-172.0)))
        self.tick(12.2, (0.70, go["y"] + 0.14, math.radians(-172.0)))
        self.assertEqual(len(self.ports.goes), sent + 1, self.modes())
        self.assertTrue(any("앞에 먼저 선다" in text for _t, text in self.mission.log))

    def test_a_robot_standing_in_front_of_the_base_not_inside_changes_nothing_but_the_spot_order(self):
        """거점 앞 통로 쪽(벽에서 0.60 m 넘게)에 서 있는 로봇: 가상 주행에서 전의 돌진이 거의 다 한 번에 들어갔고, 그 곁의 앞 자리로 먼저 가는 쪽이 1 ~ 4초 늦었다."""
        y0, y1, robot_y = self.second_base(robot_x=0.70)
        down = -math.pi / 2.0
        self.tick(10.0, (0.80, y1 + 0.30, down))
        go = self.ports.goes[-1]
        self.assertEqual((go["mode"], go["yaw_free"]), ("normal", True))  # 전과 같은 「거점 앞까지」 (방향은 요구하지 않는다)
        self.assertAlmostEqual(go["y"], y1 - 0.22, places=2)  # 자리의 차례는 그 로봇을 피해 고른 것 그대로
        self.assertIsNone(self.mission._assault_guard)
        self.tick(11.0, (0.80, y1 + 0.02, down))  # 거점의 바로 앞에 들어서면 전처럼 바로 돌진
        self.assertEqual((self.ports.goes[-1]["mode"], self.ports.goes[-1]["y"]), ("assault", go["y"]))

    def test_dashes_in_the_same_tick_when_it_already_stands_leaning_away(self):
        y0, y1, robot_y = self.second_base()
        sent = len(self.ports.goes)
        self.tick(10.0, (0.84, y1 - 0.22 - 0.08, math.radians(165.0)))  # 자리 앞 0.08 m 옆 · 먼 쪽으로 15°
        self.assertEqual(len(self.ports.goes), sent + 1, self.modes())
        dash = self.ports.goes[-1]
        self.assertEqual((dash["mode"], dash["yaw_free"]), ("assault", True))
        self.assertAlmostEqual(dash["y"], y1 - 0.22, places=2)

    def test_first_dash_from_the_leaning_waiting_spot_is_not_delayed(self):
        """기다리던 거점이 게시됐다: 대기 자리(_stage — 먼 쪽으로 20° · 허용 10°)에 서 있으면 전과 같이 그 틱에 돌진이 나간다. 두 진영 모두."""
        for slot, start, wall in (("R-2", (5.9, 2.4, math.pi), 0.0), ("B-2", (0.2, 2.4, 0.0), 6.12)):
            fact = lambda now, t, pose, **more: Facts(now=now, t=t, phase="running", control=True, slot=slot, pose=pose, boxes=[(3.06, 2.5)], **more)
            mission, ports = Mission(MissionParams()), Ports()
            mission.tick(fact(0.0, 1.0, start), ports)
            base = mission.stage_base
            _x0, y0, _x1, y1 = mission.fmap.bases[base]
            mission.scout.robots = [(abs(wall - 0.14), (y0 + y1) / 2.0 + 0.03, 200.0)]
            mission._stage(fact(1.0, 236.0, (abs(wall - 2.0), 3.9, start[2])), ports)  # 위에서 왔다 → 위쪽 자리에서 기다린다
            wait = ports.goes[-1]
            ports.reach()
            sent = len(ports.goes)
            turn = math.radians(8.0) * (1.0 if wall == 0.0 else -1.0)  # 허용 안에서 벽 쪽으로 8° 덜 기울어 섰다 (먼 쪽으로 12°)
            here = (abs(wall - 0.86), wait["y"] - 0.09, wait["yaw"] + turn)
            task = mission.tick(fact(5.0, 240.2, here, objective=base, arrived=ports.arrived), ports)
            self.assertEqual(task, "assault")
            self.assertEqual(len(ports.goes), sent + 1, (slot, [g["mode"] for g in ports.goes]))
            dash = ports.goes[-1]
            self.assertEqual((dash["mode"], dash["y"]), ("assault", wait["y"]), slot)
            self.assertAlmostEqual(abs(dash["x"] - wall), 0.20, places=2)
            self.assertTrue(any("자리 앞에 섰다" in text for _t, text in mission.log), slot)
        # 대기 자리에 0.16 m 어긋나 · 그 로봇 쪽으로 25° 기운 채 끝났어도(navigation: 도착 [YAW_NOT_MET]) 그 틱에 돌진한다 —
        # 자리 앞 목표가 대기 목표와 같아 다시 보내도 창구가 보내지 않는다 (기다려도 달라질 것이 없다)
        here = (abs(wall - 0.80), wait["y"] - 0.16, wait["yaw"] + math.radians(45.0) * (1.0 if wall == 0.0 else -1.0))
        mission._assault_plan = mission._assault_goal = mission._dash_base = None
        mission._stage(fact(5.5, 239.9, here), ports)
        sent = len(ports.goes)
        mission.tick(fact(6.0, 240.2, here, objective=base, arrived=True), ports)
        self.assertEqual(len(ports.goes), sent + 1, [g["mode"] for g in ports.goes])
        self.assertEqual((ports.goes[-1]["mode"], ports.goes[-1]["y"]), ("assault", wait["y"]))
        self.assertTrue(any("더 기다리지 않는다" in text for _t, text in mission.log))

    def test_dashes_anyway_when_navigation_ends_the_approach_without_the_lean(self):
        y0, y1, robot_y = self.second_base()
        down = -math.pi / 2.0
        self.tick(10.0, (0.80, y1 + 0.30, down))
        go = self.ports.goes[-1]
        self.assertEqual(go["mode"], "normal")
        self.tick(11.0, (0.80, go["y"] + 0.05, down))  # 가는 중
        self.ports.reach()  # navigation: 도착 (방향은 못 맞췄다 — REACHED [YAW_NOT_MET])
        self.tick(12.0, (0.80, go["y"] + 0.05, math.radians(-150.0)))
        dash = self.ports.goes[-1]
        self.assertEqual((dash["mode"], dash["y"]), ("assault", go["y"]))
        self.assertTrue(any("더 기다리지 않는다" in text for _t, text in self.mission.log))

    def test_dashes_after_a_moment_when_the_approach_goal_never_starts(self):
        """자리 앞 목표가 이미 끝난 목표와 같으면 창구가 다시 보내지 않는다 (대기 자리가 0.15 m 어긋나 섰을 때) → 「가는 중」 을 못 본 채 0.25초면 돌진."""
        y0, y1, robot_y = self.second_base()
        here = (0.80, y1 - 0.22 + 0.15, math.radians(-150.0))  # 세로로 0.15 m · 그 로봇 쪽으로 30°
        for now in (10.0, 10.1, 10.2):
            self.tick(now, here, arrived=True)
            self.assertEqual(self.ports.goes[-1]["mode"], "normal", now)
        self.tick(10.3, here, arrived=True)
        self.assertEqual(self.ports.goes[-1]["mode"], "assault")

    def test_dashes_anyway_when_the_approach_takes_too_long_in_front_of_the_base(self):
        y0, y1, robot_y = self.second_base()
        down = -math.pi / 2.0
        self.tick(10.0, (0.80, y1 + 0.30, down))  # 자리 앞까지 0.52 m → 0.52 / 0.30 + 6.0 = 7.7초까지 기다린다
        front = (0.80, y1 - 0.05, down)  # 거점의 바로 앞인데 더 못 가고 있다 (전에는 여기서 돌진을 냈다)
        self.tick(13.0, front)
        self.tick(17.6, front)
        self.assertEqual(self.ports.goes[-1]["mode"], "normal")
        self.tick(17.9, front)
        self.assertEqual(self.ports.goes[-1]["mode"], "assault")
        # 거점의 바로 앞이 아니면(전에도 보통 주행이던 곳) 오래 걸려도 돌진으로 바꾸지 않는다
        self.start()
        y0, y1, robot_y = self.second_base()
        self.tick(10.0, (0.80, y1 + 0.30, down))
        self.tick(40.0, (0.80, y1 + 0.20, down))
        self.assertEqual(self.ports.goes[-1]["mode"], "normal")

    def test_no_flip_flop_when_the_pose_wobbles_at_the_ready_limits(self):
        y0, y1, robot_y = self.second_base()
        spot = round(y1 - 0.22, 2)
        self.tick(10.0, (0.80, y1 + 0.30, -math.pi / 2.0))
        sent = len(self.ports.goes)
        # (세로로 자리에서, 먼 쪽으로 기운 각) — 준비의 금(0.12 m · 그 로봇 쪽 2°)을 넘나든다
        wobble = ((0.126, 5.0), (0.123, -2.6), (0.118, -2.4), (0.118, -1.8), (0.127, -2.5), (0.119, -3.0), (0.110, 1.0), (0.131, -9.0))
        modes = []
        for n, (dy, tilt) in enumerate(wobble):
            self.tick(11.0 + 0.1 * n, (0.80, spot + dy, math.pi - math.radians(tilt)))
            modes.append(self.ports.goes[-1]["mode"])
        self.assertEqual(modes, ["normal"] * 3 + ["assault"] * 5)
        self.assertEqual(len(self.ports.goes), sent + 1, self.modes())  # 자리 앞 목표 하나 · 돌진 하나뿐 (번갈아 다시 보내지 않았다)

    def test_decides_once_even_if_the_robot_list_flickers(self):
        y0, y1, robot_y = self.second_base()
        down = -math.pi / 2.0
        self.tick(10.0, (0.80, y1 + 0.30, down))
        go, sent = self.ports.goes[-1], len(self.ports.goes)
        for n, robots in enumerate(([], [(0.26, robot_y + 0.5, 3.0)], [], [(0.26, robot_y, 70.0)])):
            self.mission.scout.robots = robots
            self.mission._stood = {}
            self.tick(10.5 + 0.1 * n, (0.80, y1 + 0.02, down))
        self.assertEqual(len(self.ports.goes), sent, self.modes())  # 같은 자리 · 같은 방향의 목표 그대로
        self.assertEqual(self.ports.goes[-1], go)

    def test_another_spot_after_a_failed_dash_goes_through_the_front_again(self):
        y0, y1, robot_y = self.second_base()
        self.tick(10.0, (0.82, y1 - 0.22, math.radians(165.0)))  # 준비된 채 → 첫 자리로 바로 돌진
        first = self.ports.goes[-1]
        self.assertEqual(first["mode"], "assault")
        self.tick(10.5, (0.70, first["y"], math.pi))
        failed = self.ports.fail()  # navigation: 막힘
        self.tick(11.5, (0.66, first["y"], math.pi), go_failed=failed)
        again = self.ports.goes[-1]
        # 전에는 서 있던 자리에서 바로 다음 돌진을 냈다. 이제는 다음 자리(그 로봇의 아래쪽)의 앞에, 아래쪽으로 기울여 먼저 선다
        self.assertEqual((again["mode"], again["yaw_free"]), ("normal", False))
        self.assertAlmostEqual(again["x"], 0.80, places=2)
        self.assertAlmostEqual(again["y"], y0 + 0.22, places=2)
        self.assertAlmostEqual(math.sin(again["yaw"]), -math.sin(math.radians(20.0)), places=6)
        self.assertAlmostEqual(math.cos(again["yaw"]), -math.cos(math.radians(20.0)), places=6)
        self.tick(15.0, (0.80, again["y"] + 0.03, math.radians(-168.0)))  # 그 앞에 먼 쪽(아래)으로 12° 기울여 섰다
        second = self.ports.goes[-1]
        self.assertEqual((second["mode"], second["y"]), ("assault", again["y"]))
        # 도착이라는데 깊이에 못 미쳤을 때도 같다: 그다음 자리의 앞으로 먼저
        self.tick(15.5, (0.60, again["y"], math.pi))
        self.ports.reach()
        self.tick(17.0, (0.42, again["y"], math.pi))
        third = self.ports.goes[-1]
        self.assertEqual(third["mode"], "normal")
        self.assertNotIn(third["y"], (first["y"], second["y"]))
        self.assertEqual(self.mission._assault_goal[3], 2)  # 못 간 횟수는 전과 같이 센다

    def test_a_failed_approach_keeps_the_next_spot_and_next_base_bookkeeping(self):
        y0, y1, robot_y = self.second_base()
        far = lambda n: (2.4 - 0.3 * n, (y0 + y1) / 2.0, math.pi)  # 경기장 안쪽에서 온다 (못 갈 때마다 0.3 m 씩은 움직였다 — 빠져나오기가 끼어들지 않게)
        self.tick(10.0, far(0))
        spots = list(self.mission._assault_goal[1])
        seen = [self.ports.goes[-1]["y"]]
        for n in range(len(spots) - 1):
            failed = self.ports.fail()
            self.tick(12.0 + 2.0 * n, far(n + 1), go_failed=failed)
            self.assertEqual(self.ports.goes[-1]["mode"], "normal")
            self.assertAlmostEqual(self.ports.goes[-1]["x"], 0.80, places=2)
            seen.append(self.ports.goes[-1]["y"])
        self.assertEqual(seen, spots)  # 자리 앞 목표가 못 가면 다음 자리의 앞으로
        failed = self.ports.fail()
        self.tick(30.0, far(len(spots)), go_failed=failed)  # 자리를 다 못 갔다 → 다음 거점
        self.assertEqual(self.mission._assault_plan[2], 2)
        self.assertTrue(any("못 감 → 다음 거점" in text for _t, text in self.mission.log))

    def test_nothing_escapes_when_the_pose_or_the_stored_robots_are_odd(self):
        y0, y1, robot_y = self.second_base()
        self.tick(10.0, (0.80, y1 - 0.30))  # 방향이 빠진 위치 — 기울기를 볼 수 없다 → 전처럼 (거점의 바로 앞이면 돌진)
        self.assertEqual(self.ports.goes[-1]["mode"], "assault")
        self.start()
        y0, y1, robot_y = self.second_base()
        self.tick(10.0, (0.80, y1 + 0.30, -math.pi / 2.0))
        self.mission._assault_guard[1][:] = ["?"]
        self.mission._assault_guard[2] = None
        self.tick(10.1, (0.80, y1 + 0.02, -math.pi / 2.0))  # 서 있던 로봇의 기록이 이상하다 → 이 거점은 전처럼 (바로 앞이니 돌진)
        self.assertEqual(self.ports.goes[-1]["mode"], "assault")
        self.assertIsNone(self.mission._assault_guard)


class WaitingBaseTests(unittest.TestCase):
    """목표거점 발표(4:00)를 기다릴 거점: 기본은 가운데 거점 앞 (팀장 판단 2026-11-19 밤 — 세 거점이 같은 가능성으로 게시된다고 본다).

    MissionParams.stage_kind 에 종류를 주면, 지도가 상대 진영의 거점 종류를 읽어 알려 줬을 때 그 종류의 거점 앞에서 기다린다 (못 읽었으면 가운데).
    기다리던 거점이 게시되면 1.5초, 가운데에서 기다리다 끝 거점이 게시되면 9.2 ~ 9.8초가 걸렸다 (2026-11-19).
    """

    @staticmethod
    def told(kinds):
        """지도 메시지로 받은 지도: 자리 이름 여섯 개 + kinds = {진영: [1번 줄, 2번 줄, 3번 줄]} 의 보고값 이름 (perception 이 읽은 것)."""
        from types import SimpleNamespace
        plan = A.ArenaPlan()
        region = lambda name, r, row: SimpleNamespace(name=name, min_x=r[0], min_y=r[1], max_x=r[2], max_y=r[3], row=row)
        bases = [region("%s-%d" % (camp[0], row), plan.base_rect(camp, row), row) for camp in A.CAMPS for row in (1, 2, 3)]
        bases += [region("%s_%s" % (camp, kind), plan.base_rect(camp, row), row) for camp, order in kinds.items() for row, kind in zip((1, 2, 3), order)]
        return A.field_map_from_msg(SimpleNamespace(width=plan.width, height=plan.height, box_size=plan.box, beam_depth=0.2,
                                                    boxes=[SimpleNamespace(x=3.06, y=2.5)], bases=bases, sectors=[]))

    def waits_at(self, fmap, **params):
        mission, ports = Mission(MissionParams(**params)), Ports()
        mission.tick(Facts(now=0.0, t=1.0, phase="running", control=True, slot="R-2", pose=(5.9, 2.4, math.pi), boxes=[(3.06, 2.5)], fmap=fmap), ports)
        return mission, mission.stage_base, mission.fmap.base_rows[mission.stage_base]

    def test_waits_in_front_of_the_named_kind_when_asked_and_the_kinds_were_read(self):
        kinds = {"BLUE": ["MAINTENANCE", "SUPPLY_DEPOT", "COMMAND_POST"], "RED": ["MAINTENANCE", "SUPPLY_DEPOT", "COMMAND_POST"]}
        mission, base, row = self.waits_at(self.told(kinds), stage_kind="COMMAND_POST")
        self.assertEqual((base, row), ("BLUE_COMMAND_POST", 3))
        goal, heading = mission._stage_point(base, mission.p.stage_from_wall_m)
        x0, y0, x1, y1 = mission.fmap.bases["BLUE_COMMAND_POST"]
        self.assertAlmostEqual(goal[0], 0.80, places=2)  # 벽에서 0.80 m — 0.10 m 더 들어가 서도 앞 모서리가 거점 네모(0.40 m) 밖이다
        self.assertAlmostEqual(goal[1], (y0 + y1) / 2.0, places=2)
        self.assertGreater(goal[0] - 0.10 - 0.17, x1 + 0.10)

    def test_waits_at_the_middle_base_when_the_kinds_were_not_read(self):
        """자리 이름만 온 지도(거점 종류를 못 읽음): 이름은 기본 배치일 뿐이라 믿지 않고 가운데에서 기다린다."""
        _mission, base, row = self.waits_at(self.told({}), stage_kind="COMMAND_POST")
        self.assertEqual(row, 2)
        only_ours = self.told({"RED": ["COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT"]})  # 우리 진영 것만 읽힘 — 상대 진영은 모른다
        self.assertEqual(self.waits_at(only_ours, stage_kind="COMMAND_POST")[2], 2)
        self.assertEqual(self.waits_at(None, stage_kind="COMMAND_POST")[2], 2)  # 지도 메시지 없이 상자 목록만 있을 때

    def test_waits_to_the_side_when_a_robot_stands_in_the_waiting_base(self):
        """03:28 연습 판: 대기 자리 바로 옆 거점 안에 다른 팀 로봇이 서 있어 navigation 이 우회로로 돌아 나갔고 4:00 에 1.6 m 떨어져 있었다."""
        mission, ports = Mission(MissionParams()), Ports()
        fact = lambda now, t, pose: Facts(now=now, t=t, phase="running", control=True, slot="R-2", pose=pose, boxes=[(3.06, 2.5)])
        mission.tick(fact(0.0, 1.0, (5.9, 2.4, math.pi)), ports)
        x0, y0, x1, y1 = mission.fmap.bases[mission.stage_base]
        mid = (y0 + y1) / 2.0
        # 서 있는 로봇이 없으면 거점의 세로 가운데 앞
        mission._stage(fact(1.0, 225.0, (2.0, 1.2, math.pi)), ports)
        self.assertAlmostEqual(ports.goes[-1]["y"], mid, places=2)
        # 그 거점 안에 로봇이 서 있으면 그 로봇에서 0.30 m 넘게 떨어진 쪽 — 우리 자리(아래쪽)에서 가까운 쪽
        mission._stage_y = None
        mission.scout.robots = [(0.27, mid - 0.01, 200.0)]
        mission._stage(fact(2.0, 226.0, (2.0, 1.2, math.pi)), ports)
        go = ports.goes[-1]
        self.assertAlmostEqual(go["x"], 0.80, places=2)
        self.assertGreaterEqual(abs(go["y"] - mid), 0.30)
        self.assertLess(go["y"], mid)  # 아래에서 왔으니 아래쪽
        self.assertTrue(any("떨어진 쪽에서 기다린다" in text for _t, text in mission.log))
        # 한 번 정하면 바꾸지 않는다 (로봇 목록이 흔들려도 대기 자리가 오락가락하지 않게)
        mission.scout.robots = []
        mission._stage(fact(3.0, 236.0, (0.9, go["y"], math.pi)), ports)
        self.assertAlmostEqual(ports.goes[-1]["y"], go["y"], places=2)

    def test_leans_away_from_a_robot_standing_in_the_waiting_base(self):
        """04:48 연습 판 뒤의 가상 주행: 대기 자리에서 차체가 거점 안의 로봇 쪽으로 기울어 서 있으면 첫 돌진이 셋에 한 번 1초 만에 막힌다
        (차체 앞 띠에 그 로봇이 걸린다). 벽을 보되 그 로봇에서 먼 쪽으로 기울여 서면 가상 주행 60판이 모두 들어갔다."""
        fact = lambda now, t, pose, slot: Facts(now=now, t=t, phase="running", control=True, slot=slot, pose=pose, boxes=[(3.06, 2.5)])
        for slot, start, wall_heading in (("R-2", (5.9, 2.4, math.pi), math.pi), ("B-2", (0.2, 2.4, 0.0), 0.0)):
            mission, ports = Mission(MissionParams()), Ports()
            mission.tick(fact(0.0, 1.0, start, slot), ports)
            x0, y0, x1, y1 = mission.fmap.bases[mission.stage_base]
            mid, wall = (y0 + y1) / 2.0, (0.0 if wall_heading else mission.fmap.width)
            lane = 2.0 if wall_heading else mission.fmap.width - 2.0
            # 서 있는 로봇이 없으면 전처럼 벽 쪽 · 허용 30°
            mission._stage(fact(1.0, 225.0, (lane, 1.2, wall_heading), slot), ports)
            self.assertAlmostEqual(math.cos(ports.goes[-1]["yaw"]), math.cos(wall_heading), places=6)
            self.assertAlmostEqual(math.degrees(ports.goes[-1]["yaw_tol"]), 30.0, places=6)
            for ours, side in ((1.2, -1), (3.9, 1)):     # 아래에서 오면 아래쪽에서, 위에서 오면 위쪽에서 기다린다
                mission._stage_y = None
                mission.scout.robots = [(abs(wall - 0.14), mid + 0.03, 200.0)]
                mission._stage(fact(2.0, 226.0, (lane, ours, wall_heading), slot), ports)
                go = ports.goes[-1]
                self.assertEqual(go["y"] > mid, side > 0, (slot, ours, go))
                self.assertAlmostEqual(math.degrees(go["yaw_tol"]), 10.0, places=6)
                # 벽 쪽 성분은 그대로이고(cos 20°), 세로 성분은 그 로봇에서 먼 쪽
                self.assertAlmostEqual(math.cos(go["yaw"]), math.cos(wall_heading) * math.cos(math.radians(20.0)), places=6)
                self.assertAlmostEqual(math.sin(go["yaw"]), side * math.sin(math.radians(20.0)), places=6)
                # 정한 뒤에는 로봇 목록이 비어도 같은 방향이다 (목표가 오락가락하지 않게)
                sent = len(ports.goes)
                mission.scout.robots = []
                mission._stage(fact(3.0, 232.0, (lane, ours, wall_heading), slot), ports)
                self.assertEqual(len(ports.goes), sent)
        # 꺼 두면 전과 같다
        mission, ports = Mission(MissionParams(stage_lean_deg=0.0)), Ports()
        mission.tick(fact(0.0, 1.0, (5.9, 2.4, math.pi), "R-2"), ports)
        mission.scout.robots = [(0.14, 2.43, 200.0)]
        mission._stage(fact(2.0, 226.0, (2.0, 3.9, math.pi), "R-2"), ports)
        self.assertAlmostEqual(abs(ports.goes[-1]["yaw"]), math.pi, places=6)
        self.assertAlmostEqual(math.degrees(ports.goes[-1]["yaw_tol"]), 30.0, places=6)

    def test_the_default_is_the_middle_base_whatever_the_map_says(self):
        kinds = {"BLUE": ["COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT"], "RED": ["COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT"]}
        self.assertEqual(MissionParams().stage_kind, "")
        mission, _base, row = self.waits_at(self.told(kinds))
        self.assertEqual(row, 2)
        goal, _heading = mission._stage_point(mission.stage_base, mission.p.stage_from_wall_m)
        self.assertAlmostEqual(goal[0], 0.80, places=2)  # 벽에서 0.80 m (게시 전에 거점 센서를 가리지 않게)
        self.assertEqual(self.waits_at(self.told(kinds), stage_kind="COMMAND_POST")[2], 1)


class StrikeSpotTests(unittest.TestCase):
    def test_picks_another_firing_spot_when_the_first_is_taken(self):
        """쏠 자리에 다른 UGV 가 서 있으면 다른 자리에서 쏜다. 전에는 같은 자리만 40초 보내다 지정표적을 포기했다."""
        mission, ports = Mission(MissionParams()), Ports()
        boxes = [(4.26, 2.5)]
        pose = (2.0, 2.5, 0.0)
        facts = dict(phase="running", control=True, slot="B-2", boxes=boxes, parsed={"target": "RED_OUTPOST", "search_sectors": ["CM"], "base": "RED_MAINTENANCE"})
        mission.tick(Facts(now=0.0, t=0.0, pose=pose, arrived=True, **facts), ports)
        face = next(f for f in mission.fmap.faces if f.key == (0, "W"))
        mission.target_face = face
        mission.tick(Facts(now=0.1, t=0.1, pose=pose, arrived=True, **facts), ports)
        first = ports.goes[-1]
        self.assertAlmostEqual(first["yaw_tol"], math.radians(25.0))
        mission.tick(Facts(now=9.0, t=9.0, pose=(2.6, 2.5, 0.0), arrived=True, go_failed=(first["x"], first["y"], 3), **facts), ports)
        mission.tick(Facts(now=9.1, t=9.1, pose=(2.6, 2.5, 0.0), arrived=True, **facts), ports)
        second = ports.goes[-1]
        self.assertGreaterEqual(math.hypot(second["x"] - first["x"], second["y"] - first["y"]), 0.35)
        self.assertFalse(mission.target_done)


if __name__ == "__main__":
    unittest.main()


class GoneBoxTests(unittest.TestCase):
    """navigation 이 라이다로 「없다」 고 보아 지도에서 뺀 상자 (지도 고치기). 번호는 처음 지도 그대로 두고 그 면만 뒤로 미룬다."""

    def test_faces_of_a_removed_box_go_last_and_leave_photo_matching(self):
        from supervisor.faces import expected_faces

        fmap = A.field_map(BOXES3)
        scout, _grid, _book = scout_for(fmap)
        scout.plan((5.9, 2.5, math.pi))
        first = scout.stops[0]
        box = first[3][0][0]  # 첫 자리에서 찍을 면의 상자
        pose = (first[0], first[1], first[2])
        self.assertTrue(any(e.face.box == box for e in expected_faces(fmap, pose, 0.0)))
        hidden_before = len(fmap.segments())
        self.assertTrue(fmap.set_gone({box}))
        self.assertFalse(fmap.set_gone({box}))  # 같은 것을 또 받으면 바뀐 것이 없다
        self.assertEqual(len(fmap.segments()), hidden_before - 4)  # 가림 계산에서 빠진다
        self.assertFalse(any(e.face.box == box for e in expected_faces(fmap, pose, 0.0)))  # 사진에 찍힐 것으로 치지 않는다
        self.assertEqual(len(fmap.faces), len(A.field_map(BOXES3).faces))  # 면 목록 · 번호는 그대로

        visited = []
        pose = (5.9, 2.5, math.pi)
        for step in range(4000):
            act = scout.step(pose, step * 0.1)
            if act[0] == "done":
                break
            if act[0] == "go":
                pose = (act[1], act[2], act[3])
            elif act[0] == "shoot":
                visited.append(act[2][1])
                scout.shot_taken(act[2], step * 0.1)
                scout.out.clear()  # 답은 오지 않는다고 친다
        gone_at = [i for i, key in enumerate(visited) if key[0] == box]
        live_first = [i for i, key in enumerate(visited) if key[0] != box and scout.tried.get(key, 0) and i < (gone_at[0] if gone_at else 10 ** 9)]
        self.assertTrue(gone_at, "뺀 상자의 면도 맨 뒤에 한 번은 가 본다 (잘못 뺀 것일 수 있다)")
        others = {key for key in visited if key[0] != box}
        self.assertEqual({key for key in visited[:gone_at[0]] if key[0] != box}, others, "다른 면을 모두 찍은 뒤에야 뺀 상자의 면으로 간다")
        self.assertTrue(live_first)
        self.assertLessEqual(max(scout.tried.get(key, 0) for key in visited if key[0] == box), 1, "뺀 상자의 면은 다시 돌지 않는다")

    def test_mission_takes_the_removed_boxes_from_the_facts(self):
        mission = Mission(MissionParams())
        ports = Ports()
        fmap = A.field_map(BOXES3)
        f = Facts(now=0.0, t=1.0, phase="running", control=True, slot="R-2", pose=(5.9, 2.5, math.pi), arrived=True, fmap=fmap)
        mission.tick(f, ports)
        mission.tick(Facts(now=0.1, t=1.1, phase="running", control=True, slot="R-2", pose=(5.9, 2.5, math.pi), arrived=True, fmap=fmap, gone=frozenset({3})), ports)
        self.assertEqual(mission.fmap.gone, frozenset({3}))
        self.assertTrue(any("3번" in text for _t, text in mission.log))
        mission.tick(Facts(now=0.2, t=1.2, phase="running", control=True, slot="R-2", pose=(5.9, 2.5, math.pi), arrived=True, fmap=fmap, gone=frozenset()), ports)
        self.assertEqual(mission.fmap.gone, frozenset())  # 다시 보이면 되살린다


class NoRoomTests(unittest.TestCase):
    """navigation 이 「돌 자리 없음」 을 알릴 때 (2026-11-19 주행 쪽 변경 3e979ee)."""

    def start(self, pose):
        mission = Mission(MissionParams())
        ports = Ports()
        fmap = A.field_map(BOXES3)
        now = [0.0]

        def tick(pose, **extra):
            now[0] += 0.1
            return mission.tick(Facts(now=now[0], t=1.0 + now[0], phase="running", control=True, slot="R-2", pose=pose, arrived=ports.arrived, fmap=fmap, **extra), ports)

        tick(pose)
        return mission, ports, tick

    def test_backs_straight_out_when_there_is_no_room_to_turn_and_leave(self):
        # 상자 12(4.177, 1.218)의 오른쪽 면 바로 곁에서 위를 보고 서 있다. 앞(위)은 비었다
        pose = (4.52, 1.20, math.pi / 2.0)
        mission, ports, tick = self.start(pose)
        for _ in range(3):
            tick(pose)
        self.assertTrue(ports.goes)
        failed = ports.fail()
        task = tick(pose, go_failed=failed + (True,))
        self.assertEqual(task, "escape")
        last = ports.goes[-1]
        self.assertEqual(last["mode"], "straight")
        self.assertAlmostEqual(last["x"], 4.52, places=2)  # 차체 방향(위아래) 그대로 — 옆으로는 안 간다
        self.assertGreaterEqual(abs(last["y"] - 1.20), 0.15)
        self.assertAlmostEqual(last["yaw"], math.pi / 2.0)
        # 그 자리에 오면 빠져나오기를 끝내고 하던 일(정찰)로 돌아간다
        ports.reach()
        task = tick((last["x"], last["y"], math.pi / 2.0))
        self.assertTrue(task.startswith("scout"), task)

    def test_straight_out_picks_the_free_side(self):
        mission, _ports, _tick = self.start((5.9, 2.5, math.pi))
        # 위쪽 벽 가운데 기둥(3.06, 4.80) 바로 아래에서 위를 보고 있다 → 앞은 기둥, 뒤가 비었다
        goal = mission._straight_out((3.06, 4.40, math.pi / 2.0))
        self.assertIsNotNone(goal)
        self.assertLess(goal[1], 4.40)
        # 오른쪽 벽에 붙어 벽을 보고 있다 → 뒤로
        goal = mission._straight_out((5.95, 2.5, 0.0))
        self.assertLess(goal[0], 5.95)

    def test_does_not_ask_for_another_heading_where_there_is_no_room_to_turn(self):
        fmap = A.field_map(BOXES3)
        mission, ports, tick = self.start((5.9, 2.5, math.pi))
        act = mission.scout.step((5.9, 2.5, math.pi), 0.0)
        self.assertEqual(act[0], "go")
        x, y = act[1], act[2]
        there = (x, y, act[3] + math.radians(60.0))  # 자리에는 왔는데 방향이 60° 모자라다
        tick((5.9, 2.5, math.pi))
        before = len(ports.goes)
        ports.reach()
        tick(there, no_turn=(round(x, 2), round(y, 2)))  # navigation: REACHED + 「돌 자리 없음(도착 방향)」
        for _ in range(30):
            tick(there)
        same_place = [g for g in ports.goes[before:] if abs(g["x"] - round(x, 2)) < 0.02 and abs(g["y"] - round(y, 2)) < 0.02]
        self.assertEqual(same_place, [], "돌 자리가 없는 자리에서 방향만 다른 목표를 다시 시키지 않는다")
        del fmap


class PassShotTests(unittest.TestCase):
    """지나가며 찍기 (기록용, 기본은 끔)."""

    def run_scout(self, gap):
        mission = Mission(MissionParams(pass_gap_s=gap))
        ports = Ports()
        ports.passes = []
        ports.shoot_pass = lambda tag: ports.passes.append(tag) or True
        fmap = A.field_map(BOXES3)
        x, now = 5.9, 0.0
        for step in range(60):  # 곧게 가는 중 (방향 그대로)
            now += 0.1
            x -= 0.03
            mission.tick(Facts(now=now, t=1.0 + now, phase="running", control=True, slot="R-2", pose=(x, 2.5, math.pi), arrived=ports.arrived, fmap=fmap,
                               answers=[(("pass", 0), (x, 2.5, math.pi), 0.0, [("tank", 0.5, False)])] if step == 30 else []), ports)
        return mission, ports

    def test_off_by_default_and_answers_are_not_recorded(self):
        mission, ports = self.run_scout(0.0)
        self.assertEqual(ports.passes, [])
        mission, ports = self.run_scout(1.0)
        self.assertGreaterEqual(len(ports.passes), 4)
        self.assertLessEqual(len(ports.passes), 7)  # 6초 동안 1초에 한 장
        self.assertEqual(mission.book.remaining(), {f.key for f in mission.book.goal})  # 지나가며 찍은 사진의 답은 적지 않는다


class ReachedButNotThereTests(unittest.TestCase):
    """navigation 이 「도착」 이라 했는데 supervisor 기준으로는 자리가 아닐 때 (2026-11-19 주행 통합 뒤 정리).

    navigation 은 돌진으로 벽 곁에 서면 목표에서 0.25 m 안을 도착으로 끝내고(REACHED [POS_NOT_MET]), 막힌 목표 칸은 0.30 m 안의 빈 칸으로 옮기며,
    방향을 못 맞춰도 도착으로 끝낸다 ([YAW_NOT_MET]). 같은 목표는 창구가 다시 보내지 않으므로, 전에는 그대로 서 있거나 지정표적을 버렸다.
    """

    def test_base_entry_goes_to_another_spot_when_reached_short_of_the_depth(self):
        mission, ports = Mission(MissionParams()), Ports()

        def tick(now, pose):
            return mission.tick(Facts(now=now, t=241.0 + now, phase="running", control=True, slot="R-2", pose=pose, boxes=[(3.06, 2.5)],
                                      objective="BLUE_COMMAND_POST", arrived=ports.arrived), ports)

        _x0, y0, _x1, y1 = A.field_map([(3.06, 2.5)]).bases["BLUE_COMMAND_POST"]
        mid = (y0 + y1) / 2.0  # 목표 거점의 바로 앞에서 시작한다 (돌진은 거점 앞에서만 쓴다)
        tick(0.0, (0.6, mid, math.pi))
        first = ports.goes[-1]
        tick(0.1, (0.55, mid, math.pi))
        ports.reach()  # 벽에서 0.42 m 에 서서 도착이라 한다 (들어간 것으로 치는 깊이는 0.30 m)
        for k in range(5):
            tick(3.0 + 0.1 * k, (0.42, first["y"] + 0.05, math.radians(150.0)))
        self.assertGreater(len(ports.goes), 1)
        self.assertGreater(abs(ports.goes[-1]["y"] - first["y"]), 0.15)
        self.assertEqual(ports.goes[-1]["mode"], "assault")

    def strike(self):
        mission, ports = Mission(MissionParams()), Ports()
        facts = dict(phase="running", control=True, slot="B-2", boxes=[(4.26, 2.5)], parsed={"target": "RED_OUTPOST", "search_sectors": ["CM"], "base": "RED_MAINTENANCE"})

        def tick(now, pose):
            return mission.tick(Facts(now=now, t=now, pose=pose, arrived=ports.arrived, **facts), ports)

        tick(0.0, (2.0, 2.5, 0.0))
        mission.target_face = next(f for f in mission.fmap.faces if f.key == (0, "W"))
        tick(0.1, (2.0, 2.5, 0.0))
        tick(0.2, (2.1, 2.5, 0.0))
        return mission, ports, tick, ports.goes[-1]

    def test_strike_picks_another_spot_when_the_heading_leaves_the_target_out_of_pan_reach(self):
        mission, ports, tick, first = self.strike()
        ports.reach()  # 자리에는 왔는데 방향이 90° 틀린 채 도착 (돌 자리 없음 등)
        for k in range(8):
            tick(5.0 + 0.1 * k, (first["x"], first["y"], math.radians(90.0)))
        self.assertFalse(mission.target_done)
        self.assertGreaterEqual(math.hypot(ports.goes[-1]["x"] - first["x"], ports.goes[-1]["y"] - first["y"]), 0.35)

    def test_strike_asks_for_a_turn_first_when_it_already_stands_on_the_spot(self):
        """앞의 일을 하던 곳이 마침 쏠 자리인데 방향만 틀렸다 (모의 5차: 그 자리를 버리고 10.5초를 들여 더 나쁜 자리로 갔다) → 먼저 그 자리에서 돌린다."""
        _mission, _ports, _tick, first = self.strike()  # 쏠 자리가 어디로 정해지는지 본다
        mission, ports = Mission(MissionParams()), Ports()
        facts = dict(phase="running", control=True, slot="B-2", boxes=[(4.26, 2.5)], parsed={"target": "RED_OUTPOST", "search_sectors": ["CM"], "base": "RED_MAINTENANCE"})
        here = (first["x"] + 0.05, first["y"], math.radians(90.0))  # 쏠 자리에서 5 cm, 방향은 90도 틀림

        def tick(now):
            return mission.tick(Facts(now=now, t=now, pose=here, arrived=ports.arrived, **facts), ports)

        tick(0.0)
        mission.target_face = next(f for f in mission.fmap.faces if f.key == (0, "W"))
        ports.stop()
        sent = len(ports.goes)
        for k in range(1, 30):
            tick(0.1 * k)
        new = ports.goes[sent:]
        self.assertTrue(new, "그 자리에서 돌려 달라고 해야 한다")
        self.assertLessEqual(math.hypot(new[0]["x"] - first["x"], new[0]["y"] - first["y"]), 0.02)  # 다른 자리가 아니라 바로 그 자리
        self.assertAlmostEqual(new[0]["yaw"], first["yaw"], places=2)  # 표적 쪽 방향
        self.assertFalse(mission.target_done)
        # 돌려 달라고 했는데도 방향이 그대로면(돌 자리 없음) 전처럼 다른 자리를 고른다
        ports.reach()
        for k in range(30, 45):
            tick(0.1 * k)
        self.assertGreaterEqual(math.hypot(ports.goes[-1]["x"] - first["x"], ports.goes[-1]["y"] - first["y"]), 0.35)

    def test_strike_does_not_call_off_the_move_while_the_target_is_out_of_pan_reach(self):
        """실차는 서서 돌라는 명령 뒤 1.5 ~ 2초 가만히 있다 돈다. 표적이 팬 밖(방향 90° 틀림)인 채 서 있는 동안에는 이동을 거두지 않는다."""
        mission, ports, tick, first = self.strike()
        for k in range(25):  # 2.5초 동안 자리에서 가만히 (navigation 은 아직 「가는 중」)
            tick(5.0 + 0.1 * k, (first["x"], first["y"], math.radians(90.0)))
        self.assertEqual(ports.stops, 0)
        self.assertFalse(mission.target_done)
        for k in range(25):  # 표적이 팬 안으로 들어온 채 서 있으면 전처럼 거두고 팬틸트로 겨눈다
            tick(8.0 + 0.1 * k, (first["x"], first["y"], first["yaw"] + math.radians(30.0)))
        self.assertEqual(ports.stops, 1)

    def test_strike_picks_another_spot_when_reached_far_from_the_spot(self):
        mission, ports, tick, first = self.strike()
        ports.reach()  # navigation 이 목표를 옮겨 0.3 m 떨어진 데 서서 도착이라 한다
        for k in range(3):
            tick(5.0 + 0.1 * k, (first["x"] - 0.3, first["y"], 0.0))
        self.assertFalse(mission.target_done)
        self.assertGreaterEqual(math.hypot(ports.goes[-1]["x"] - first["x"], ports.goes[-1]["y"] - first["y"]), 0.35)
