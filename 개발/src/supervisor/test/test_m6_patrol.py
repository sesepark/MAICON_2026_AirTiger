"""M6 순찰 (mission._m6_patrol): 정찰(M3)이 끝나면 우리 진영을 멈추지 않고 돌며 상대 UGV 의 수광부를 찾고, 보이면 서서 멈춘 상대를 쏜 뒤 다시 돈다. ROS 없이 돈다.

로봇 · navigation · perception 은 아래 Robot 이 흉내 낸다: 시킨 자리로 가는 방향을 보며 곧게 가고(0.3 m/s), 팬틸트는 바로 돌고,
수광부 자리(locate_sensors)는 정해 둔 상대 UGV 가 카메라 앞(팬 ± 60°) 4 m 안에 있을 때만 답한다.
"""
import math
import unittest

from supervisor import arena as A
from supervisor.aim import AimParams, laser_pixel
from supervisor.faces import wrap
from supervisor.mission import Facts, Mission, MissionParams

AIM = AimParams()
DEG_PX = AIM.deg_px(127.4)
BOXES = [(1.40, 3.93), (4.30, 4.17), (3.07, 2.78), (1.83, 2.48), (2.80, 1.50), (4.18, 1.22)]


class Robot:
    def __init__(self, pose, foe=None, foe_camp="RED"):
        self.pose, self.goal, self.now = list(pose), None, 0.0
        self.pan = self.tilt = 0.0
        self.foe, self.foe_camp = foe, foe_camp  # foe(now) -> (x, y) 또는 None
        self.foe_sign, self.foe_lives, self.foe_rear = None, 4, None  # read_ugvs 흉내 (look_ugvs): 호출부호 · 목숨 · 뒤가 향한 쪽
        self.mail, self.fires, self.goes, self.looks, self.aims, self.scans = [], [], [], 0, [], 0

    # ---- 창구
    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None):
        goal = (round(x, 2), round(y, 2), round(yaw, 2))
        if goal != self.goal:
            self.goal = goal
            self.goes.append(goal)

    def stop(self):
        self.goal = None

    def aim(self, pan, tilt):
        self.pan, self.tilt = pan, tilt
        self.aims.append(pan)
        return True

    def aim_ready(self):
        return True

    def look_ugvs(self):
        """read_ugvs 흉내: 상대가 카메라 앞 4 m 안이면 진영을 읽고, read_near m 안이면 호출부호도, 그 상대의 뒤쪽(rear 방향 ± 60°)에 있으면 목숨도 읽는다."""
        self.looks += 1
        seen = self._bearing()
        if seen is not None and abs(seen[0] - self.pan) <= 60.0 and getattr(self, "foe_sign", None):
            fx, fy = self.foe(self.now)
            near = seen[1] <= getattr(self, "read_near", 1.2)
            from_foe = math.atan2(self.pose[1] - fy, self.pose[0] - fx)
            behind = near and self.foe_rear is not None and abs(wrap(from_foe - self.foe_rear)) <= math.radians(60.0)
            self.mail.append((self.now + 0.2, "ugvs", {"camp": self.foe_camp, "callsign": self.foe_sign if near else "", "lives": self.foe_lives if behind else -1, "ammo": -1}))
        return True

    def fire(self):
        seen = self._bearing()
        self.fires.append((self.now, None if seen is None else abs(seen[0] - self.pan)))
        self.mail.append((self.now + 0.1, "fire", ("fired", 10000)))
        return True

    def locate(self, tag):
        self.scans += 1
        seen = self._bearing()
        spots = []
        if seen is not None and abs(seen[0] - self.pan) <= 60.0:
            need_pan, dist = seen
            lu, lv = laser_pixel(dist, AIM)
            u = lu + (need_pan - self.pan) / DEG_PX
            wide = math.degrees(2.0 * math.atan(0.115 / dist)) / DEG_PX / AIM.image_w
            spots = [{"owner": "ugv", "x_center": u / AIM.image_w, "y_center": lv / AIM.image_h, "width": 0.02, "height": 0.03, "confidence": 0.8,
                      "has_host": True, "host_kind": "ugv", "camp": self.foe_camp, "callsign": "", "host_x_center": u / AIM.image_w,
                      "host_y_center": lv / AIM.image_h, "host_width": wide, "host_height": wide}]
        self.mail.append((self.now + 0.2, "sensors", (tag, self.pan, self.tilt, spots)))
        return True

    def report(self, kind, payload): return True
    def set_ready(self): return True
    def analyze_order(self, text): return True

    # ---- 세계
    def _bearing(self):
        """상대 UGV 를 맞히려면 팬이 몇 도여야 하는가 (+ 오른쪽)와 거리. 없거나 4 m 밖이면 None."""
        spot = self.foe(self.now) if self.foe else None
        if spot is None:
            return None
        dx, dy = spot[0] - self.pose[0], spot[1] - self.pose[1]
        dist = math.hypot(dx, dy)
        if dist > 4.0 or dist < 0.3:
            return None
        return -math.degrees(wrap(math.atan2(dy, dx) - self.pose[2])), dist

    def step(self, dt=0.1):
        self.now += dt
        if self.goal is None:
            return
        gx, gy, gyaw = self.goal
        d = math.hypot(gx - self.pose[0], gy - self.pose[1])
        if d > 0.02:
            move = min(d, 0.3 * dt)
            self.pose[0] += (gx - self.pose[0]) / d * move
            self.pose[1] += (gy - self.pose[1]) / d * move
            if d > 0.1:  # 가는 방향을 보고 달린다
                self.pose[2] = wrap(self.pose[2] + max(-0.15, min(0.15, wrap(math.atan2(gy - self.pose[1], gx - self.pose[0]) - self.pose[2]))))
        else:
            turn = wrap(gyaw - self.pose[2])
            self.pose[2] = wrap(self.pose[2] + max(-0.15, min(0.15, turn)))

    @property
    def arrived(self):
        if self.goal is None:
            return True
        gx, gy, gyaw = self.goal
        return math.hypot(gx - self.pose[0], gy - self.pose[1]) <= 0.03 and abs(wrap(gyaw - self.pose[2])) <= 0.05

    def facts(self):
        f = Facts(now=self.now, t=self.now + getattr(self, "t0", 0.0), phase="running", control=True, slot="B-1", pose=tuple(self.pose), arrived=self.arrived, fmap=A.field_map(BOXES))
        for _at, kind, value in [m for m in self.mail if m[0] <= self.now]:
            if kind == "ugvs":
                f.ugv_answers.append(value)
            elif kind == "sensors":
                f.sensor_answers.append(value)
            else:
                f.fire_reply = value
        self.mail = [m for m in self.mail if m[0] > self.now]
        return f


def patrol(robot, seconds, **params):
    """정찰이 이미 끝난 것으로 두고 M6 순찰만 돌린다. -> (mission, [(시각, 할 일)])"""
    mission = Mission(MissionParams(**{"m6_inspect": False, **params}))  # 가까이 돌아가 읽기는 InspectTests 에서만 켠다
    f = robot.facts()
    mission._build(f)
    mission.target_done = True  # 지정표적은 다 쐈다
    tasks = []
    while robot.now < seconds:
        f = robot.facts()
        mission._absorb(f)
        task = mission._m6_patrol(f, robot)
        if not tasks or tasks[-1][1] != task:
            tasks.append((round(robot.now, 1), task))
        robot.step()
    return mission, tasks


def route_of(slot="B-1"):
    mission = Mission(MissionParams())
    mission._build(Facts(now=0.0, t=0.0, slot=slot, pose=(3.0, 2.4, 0.0), fmap=A.field_map(BOXES)))
    return mission, mission._m6_post_list()


def start_on(route):
    """경로 첫 지점에서 둘째 지점을 보고 선 자세."""
    (x0, y0), (x1, y1) = route[0], route[1]
    return (x0, y0, math.atan2(y1 - y0, x1 - x0))


def ahead(route, along, off_deg=0.0, side=0.0):
    """경로 첫 지점에서 둘째 지점 쪽으로 along m 간 곳. off_deg: 그 방향에서 왼쪽으로 돈 각도, side: 왼쪽으로 옆걸음 (m)."""
    (x0, y0), (x1, y1) = route[0], route[1]
    way = math.atan2(y1 - y0, x1 - x0)
    a = way + math.radians(off_deg)
    return (x0 + along * math.cos(a) - side * math.sin(way), y0 + along * math.sin(a) + side * math.cos(way))


class RouteTests(unittest.TestCase):
    def test_route_is_a_loop_inside_our_own_half_and_off_our_bases(self):
        for slot, camp in (("B-1", "BLUE"), ("R-2", "RED")):
            mission, route = route_of(slot)
            self.assertGreaterEqual(len(route), 4, camp)
            self.assertLessEqual(len(route), 6, camp)
            for k, (x, y) in enumerate(route):
                self.assertEqual(A.side_of(mission.fmap, x), camp)
                self.assertTrue(mission.grid.free(x, y))
                wall = x if camp == "BLUE" else mission.fmap.width - x
                self.assertGreater(wall, 1.2, (camp, x, y))  # 우리 거점(벽 쪽)에서 멀다
                self.assertTrue(mission.grid.plan(route[k - 1], (x, y)), (camp, k))  # 앞 지점에서 길이 있다 (마지막 → 첫 지점도)

    def test_no_route_without_a_map(self):
        self.assertEqual(Mission(MissionParams())._m6_post_list(), [])


class PatrolTests(unittest.TestCase):
    def test_keeps_driving_around_our_half_when_nobody_is_there(self):
        robot = Robot((4.8, 2.4, 0.0))  # 상대(빨강) 진영 깊숙이에서 정찰을 마쳤다
        mission, tasks = patrol(robot, 110.0)
        route = mission._m6_post_list()
        self.assertEqual([task for _at, task in tasks], ["m6 go"])  # 서지 않고 계속 돈다
        visited = [(g[0], g[1]) for g in robot.goes]
        for gx, _gy in visited:
            self.assertEqual(A.side_of(mission.fmap, gx), "BLUE", robot.goes)  # 시킨 지점은 모두 우리 진영
        self.assertGreaterEqual(len(set(visited)), 4, robot.goes)
        self.assertEqual(len(robot.goes), len([1 for a, b in zip(visited, visited[1:]) if a != b]) + 1)  # 지점마다 목표를 한 번만 보낸다
        self.assertGreater(robot.looks, 20)  # 도는 중에도 read_ugvs 를 부른다
        self.assertGreater(robot.scans, 60)  # 도는 중에도 locate_sensors 를 부른다
        self.assertEqual(set(robot.aims), {0.0})  # 팬은 가운데
        self.assertEqual(robot.fires, [])
        self.assertGreaterEqual(len(route), 4)

    def test_stops_and_shoots_a_foe_that_stands_still_beside_the_route(self):
        _m, route = route_of()
        robot = Robot(start_on(route), foe=lambda t: ahead(route, 1.8, off_deg=15.0))
        mission, tasks = patrol(robot, 40.0)
        self.assertGreaterEqual(len(robot.fires), 1, tasks)
        self.assertLess(robot.fires[0][1], 1.5, robot.fires)  # 1.5도 안에서 쐈다
        self.assertEqual([task for _at, task in tasks][:2], ["m6 go", "m6 watch"])  # 돌다가 → 보고 선다
        self.assertIn("engage", [task for _at, task in tasks])
        self.assertLess(next(at for at, task in tasks if task == "m6 watch"), 2.0)
        self.assertEqual(mission.ugv_shots, len(robot.fires))
        self.assertEqual([task for _at, task in tasks][-1], "m6 go")  # 다 쏘면 다시 돈다

    def test_does_not_shoot_a_foe_that_keeps_driving_past(self):
        _m, route = route_of()
        robot = Robot(start_on(route), foe=lambda t: ahead(route, 2.2, side=-1.5 + 0.12 * t))  # 앞을 계속 가로질러 간다
        _mission, tasks = patrol(robot, 30.0)
        self.assertEqual(robot.fires, [], tasks)

    def test_never_stops_for_or_shoots_our_own_camp(self):
        _m, route = route_of()
        robot = Robot(start_on(route), foe=lambda t: ahead(route, 1.8, off_deg=15.0), foe_camp="BLUE")  # 우리 편이 서 있다
        _mission, tasks = patrol(robot, 40.0)
        self.assertEqual(robot.fires, [])
        self.assertEqual([task for _at, task in tasks], ["m6 go"])

    def test_keeps_driving_when_it_has_no_shots_left(self):
        _m, route = route_of()
        robot = Robot(start_on(route), foe=lambda t: ahead(route, 1.8, off_deg=15.0))
        _mission, tasks = patrol(robot, 30.0, ugv_shots=0)
        self.assertEqual([task for _at, task in tasks], ["m6 go"])  # 쏠 수 없으면 서지 않는다
        self.assertGreater(robot.looks, 5)  # 상태는 계속 읽는다

    def test_gives_up_a_point_it_cannot_reach(self):
        class Stuck(Robot):
            def step(self, dt=0.1):
                self.now += dt  # 움직이지 않는다

        robot = Stuck((4.8, 2.4, 0.0))
        mission, _tasks = patrol(robot, 60.0, m6_go_give_up_s=10.0)
        self.assertGreaterEqual(len({(g[0], g[1]) for g in robot.goes}), 3)  # 못 가면 다음 지점을 시킨다
        self.assertTrue(any("가지 못함" in text for _t, text in mission.log))

    def test_only_when_scouting_finished_within_three_minutes(self):
        """M6 는 정찰(M3)이 3분 안에 끝났을 때만 — 4:00 목표거점 게시까지 1분 넘게 남았을 때."""
        mission = Mission(MissionParams())
        self.assertEqual(mission.p.m6_scout_done_by_s, 180.0)
        self.assertFalse(mission.m6_allowed(Facts(now=0.0, t=100.0)))  # 정찰(M3)이 아직 안 끝났다
        mission.scout_done_at = 90.0
        self.assertTrue(mission.m6_allowed(Facts(now=0.0, t=100.0)))
        self.assertTrue(mission.m6_allowed(Facts(now=0.0, t=230.0)))  # 일찍 끝냈으면 3분이 지나도 (대기하러 떠날 때까지) 계속한다
        mission.scout_done_at = 180.0
        self.assertTrue(mission.m6_allowed(Facts(now=0.0, t=181.0)))  # 딱 3분에 끝남
        mission.scout_done_at = 180.5
        self.assertFalse(mission.m6_allowed(Facts(now=0.0, t=181.0)))  # 3분을 넘겨 끝남 → 하지 않는다
        mission.scout_done_at = None
        self.assertFalse(mission.m6_allowed(Facts(now=0.0, t=120.0)))  # 다시 정찰 중이다
        anytime = Mission(MissionParams(m6_scout_done_by_s=1000.0))  # 큰 값 = 시간을 따지지 않는다
        self.assertFalse(anytime.m6_allowed(Facts(now=0.0, t=299.0)))  # 그래도 정찰이 안 끝났으면 안 한다
        anytime.scout_done_at = 230.0
        self.assertTrue(anytime.m6_allowed(Facts(now=0.0, t=299.0)))

    def test_tick_stands_still_instead_of_m6_when_scouting_ended_late(self):
        """tick 전체로: 정찰이 3분을 넘겨 끝났으면 순찰 · 살피기 · 발사를 하지 않고 선다."""
        for done_at, expect_m6 in ((150.0, True), (200.0, False)):
            t = done_at + 1.0
            robot = Robot((2.4, 2.4, 0.0), foe=lambda now: (3.4, 2.9))
            mission = Mission(MissionParams(target_at_s=1e9))  # 거점 앞 대기(M5)는 이 시험에서 뺀다
            f = robot.facts()
            f.t = t
            mission._build(f)
            mission.target_done, mission.scout_done_at, mission._pan_homed = True, done_at, True
            tasks = set()
            for _ in range(150):
                f = robot.facts()
                f.t = t + robot.now
                tasks.add(mission.tick(f, robot))
                robot.step()
            if expect_m6:
                self.assertTrue(tasks & {"m6 go", "m6 watch", "engage"}, tasks)
                self.assertGreater(robot.looks, 0)
            else:
                self.assertEqual(tasks, {"rest"})
                self.assertEqual((robot.looks, robot.fires, robot.goes, robot.aims), (0, [], [], []))


    def test_turns_the_body_toward_a_still_foe_at_the_edge_of_the_pan_range(self):
        _m, route = route_of()
        foe = ahead(route, 1.6, off_deg=-55.0)  # 가는 방향에서 오른쪽 55도 (팬 범위 ±50도 밖)
        robot = Robot(start_on(route), foe=lambda t: foe)
        mission, tasks = patrol(robot, 45.0)
        self.assertTrue(any("차체를 그쪽으로 돌린다" in text for _t, text in mission.log), mission.log)
        self.assertGreaterEqual(len(robot.fires), 1, tasks)
        self.assertLess(robot.fires[0][1], 1.5, robot.fires)
        stay = Robot(start_on(route), foe=lambda t: foe)
        patrol(stay, 45.0, m6_turn_to_target=False)
        self.assertEqual(stay.fires, [])  # 돌지 않으면 팬이 닿지 않아 쏘지 못한다

    def test_does_not_wait_on_a_foe_that_moves_a_lot(self):
        _m, route = route_of()
        darting = Robot(start_on(route), foe=lambda t: ahead(route, 2.2, side=1.0 * math.sin(1.2 * t)))  # 앞에서 좌우로 크게 오간다
        mission, tasks = patrol(darting, 40.0)
        self.assertEqual(darting.fires, [], tasks)
        stopped = next(at for at, task in tasks if task in ("m6 watch", "engage"))
        resumed = next((at for at, task in tasks if task == "m6 go" and at > stopped), None)
        self.assertIsNotNone(resumed, tasks)
        self.assertLess(resumed - stopped, mission.p.m6_moving_wait_s + mission.p.m6_hold_s + 2.0, tasks)  # 붙들려 있지 않고 다시 돈다
        again = next((at for at, task in tasks if task in ("m6 watch", "engage") and at > resumed), None)
        self.assertTrue(again is None or again - resumed >= mission.p.m6_resight_s - 0.2, tasks)  # 곧바로 다시 서지 않는다

    def test_a_hit_from_the_patrol_lowers_that_team(self):
        _m, route = route_of()

        class Named(Robot):
            def locate(self, tag):
                Robot.locate(self, tag)
                for spot in self.mail[-1][2][3]:
                    spot["callsign"] = "R-3"
                return True

        robot = Named(start_on(route), foe=lambda t: ahead(route, 1.5, off_deg=10.0))
        mission, tasks = patrol(robot, 30.0)
        self.assertGreaterEqual(len(robot.fires), 1, tasks)
        self.assertGreaterEqual(mission.ugv_hits.get("R-3", 0), 1, mission.log)
        self.assertEqual(mission.ranking()[-1], "R-3")

    def _run_past_four_minutes(self, done_at, **params):
        """정찰이 done_at 에 끝난 것으로 두고 3:20 ~ 4:20 을 tick 으로 돌린다 (4:00 에 목표거점이 게시된다). -> 한 일들"""
        robot = Robot((2.4, 2.4, 0.0))
        mission = Mission(MissionParams(**params))
        f = robot.facts()
        f.t = 200.0
        mission._build(f)
        mission.target_done, mission.scout_done_at, mission._pan_homed = True, done_at, True
        tasks = set()
        for _ in range(800):
            f = robot.facts()
            f.t = 200.0 + robot.now
            if f.t >= 240.0:
                f.objective = "B"
            tasks.add(mission.tick(f, robot))
            robot.step()
        return tasks

    def test_m6_replaces_m5_when_m6_runs(self):
        """M6 를 하는 판에서는 4:00 전 대기도, 게시된 거점 진입도 하지 않고 M6 를 계속한다."""
        self.assertTrue(MissionParams().m6_instead_of_m5)
        tasks = self._run_past_four_minutes(150.0)
        self.assertTrue(tasks <= {"m6 go", "m6 watch", "engage"}, tasks)

    def test_m5_still_runs_when_m6_does_not(self):
        late = self._run_past_four_minutes(200.0)  # 정찰이 3분을 넘겨 끝났다 → M6 를 안 하므로 M5 를 한다
        self.assertFalse(late & {"m6 go", "m6 watch", "engage"}, late)
        self.assertTrue(late - {"rest"}, late)
        both = self._run_past_four_minutes(150.0, m6_instead_of_m5=False)  # 끄면 전처럼 M5 가 먼저다
        self.assertTrue(both - {"m6 go", "m6 watch", "engage", "rest"}, both)

    def test_m6_hunt_off_is_the_old_flow_even_when_scouting_ends_early(self):
        """m6_hunt=False: 정찰이 일찍 끝나도 main 의 기존 흐름 — 서서 read_ugvs 만 부르고, 4:00 전 대기 · 거점 진입(M5)을 한다."""
        self.assertTrue(MissionParams().m6_hunt)
        robot = Robot((2.4, 2.4, 0.0), foe=lambda now: (3.4, 2.9))  # 쏠 수 있는 자리에 멈춘 상대가 있어도
        mission = Mission(MissionParams(m6_hunt=False, target_at_s=1e9))
        f = robot.facts()
        mission._build(f)
        mission.target_done, mission.scout_done_at, mission._pan_homed = True, 100.0, True
        tasks = set()
        for _ in range(150):
            f = robot.facts()
            f.t = 101.0 + robot.now
            tasks.add(mission.tick(f, robot))
            robot.step()
        self.assertEqual(tasks, {"rest"})
        self.assertEqual((robot.fires, robot.goes, robot.aims), ([], [], []))  # 쏘지도, 순찰하지도, 팬을 돌리지도 않는다
        self.assertEqual(robot.looks, 5)  # 3초마다 read_ugvs (15초)
        m5 = self._run_past_four_minutes(150.0, m6_hunt=False)  # M5 는 전처럼 한다
        self.assertFalse(m5 & {"m6 go", "m6 watch", "engage"}, m5)
        self.assertTrue(m5 - {"rest"}, m5)

    def test_m6_hunt_off_ranks_by_the_latest_led_read_only(self):
        mission = Mission(MissionParams(m6_hunt=False))
        mission.camp, mission.foe = "BLUE", "RED"
        for lives in (4, 4, 2):
            mission._absorb(Facts(now=0.0, t=0.0, ugv_answers=[{"camp": "RED", "callsign": "R-2", "lives": lives, "ammo": -1}]))
        self.assertEqual(mission.ugvs["R-2"]["lives"], 2)  # 세 번의 다수(4)가 아니라 가장 최근 값
        mission.ugvs["R-2"]["cap"] = 0
        self.assertEqual(mission.lives_of("R-2"), 2)  # 맞혀서 낮춘 값도 쓰지 않는다

    def test_can_be_switched_off(self):
        self.assertTrue(MissionParams().m6_patrol)
        mission = Mission(MissionParams(m6_patrol=False))
        self.assertFalse(mission.p.m6_patrol)


class InspectTests(unittest.TestCase):
    """쏜 거리에서는 호출부호 · 목숨이 안 읽힌다 → 다 쏜 뒤 그 상대의 곁 · 뒤로 돌아가 가까이에서 읽는다 (mission._m6_inspect)."""

    @staticmethod
    def robot_and_foe(rear_from_us=180.0):
        _m, route = route_of()
        foe = ahead(route, 1.8, off_deg=15.0)
        robot = Robot(start_on(route), foe=lambda t: foe)
        robot.foe_sign, robot.foe_lives = "R-3", 1
        start = start_on(route)
        robot.foe_rear = wrap(math.atan2(start[1] - foe[1], start[0] - foe[0]) + math.radians(rear_from_us))  # 상대의 뒤가 향한 쪽 (우리 쪽에서 본 각도)
        return robot, foe

    def test_goes_around_a_still_foe_to_read_who_it_was_and_its_lives(self):
        robot, foe = self.robot_and_foe()
        mission, tasks = patrol(robot, 90.0, m6_inspect=True)
        names = [task for _at, task in tasks]
        self.assertEqual(len(robot.fires), 3, tasks)
        self.assertLess(names.index("engage"), names.index("m6 around"))  # 먼저 쏘고 → 돌아간다
        self.assertIn("m6 read", names)
        self.assertEqual(mission.ugv_hits, {"R-3": 3}, mission.log)  # 누구인지 몰라 미뤄 둔 세 발이 그 팀의 것이 됐다
        self.assertEqual(mission.ugvs["R-3"]["lives"], 1)  # 뒤에서 읽은 목숨
        self.assertEqual(mission.lives_of("R-3"), 1)
        self.assertEqual(mission.ranking()[-1], "R-3")
        stops = [g for g in robot.goes if 0.5 <= math.hypot(g[0] - foe[0], g[1] - foe[1]) <= 1.1]
        self.assertGreaterEqual(len({(g[0], g[1]) for g in stops}), 2, robot.goes)  # 옆 → 뒤
        self.assertEqual(names[-1], "m6 go")  # 읽고 나면 다시 돈다
        self.assertTrue(any("가까이 읽기 끝" in text for _t, text in mission.log))

    def test_a_close_look_from_the_side_is_enough_for_the_callsign(self):
        robot, _foe = self.robot_and_foe()
        robot.foe_rear = None  # 목숨은 어디서도 안 읽힌다 (LED 가 꺼짐 · 가려짐)
        mission, tasks = patrol(robot, 100.0, m6_inspect=True)
        self.assertEqual(mission.ugv_hits, {"R-3": 3}, mission.log)
        self.assertEqual(mission.lives_of("R-3"), 1)  # 판독이 없어도 맞힌 만큼 낮춰 센다
        self.assertEqual([task for _at, task in tasks][-1], "m6 go")  # 자리를 다 돌면 그만두고 다시 돈다

    def test_does_not_go_around_when_it_already_knows(self):
        robot, _foe = self.robot_and_foe()
        robot.read_near = 9.0  # 멀리서도 호출부호가 읽힌다
        robot.foe_rear = None

        class Named(Robot):
            pass

        mission = Mission(MissionParams(m6_inspect=True))
        f = robot.facts()
        mission._build(f)
        mission.target_done = True
        mission.ugvs["R-3"] = {"lives": 4, "lives_seen": [4, 4], "ammo": None, "t": 0.0}
        names = []
        while robot.now < 50.0:
            f = robot.facts()
            mission._absorb(f)
            names.append(mission._m6_patrol(f, robot))
            robot.step()
        self.assertEqual(len(robot.fires), 3)
        self.assertNotIn("m6 around", names)
        self.assertEqual(mission.ugv_hits, {"R-3": 3})

    def test_can_be_switched_off(self):
        robot, _foe = self.robot_and_foe()
        mission, tasks = patrol(robot, 60.0, m6_inspect=False)
        self.assertNotIn("m6 around", [task for _at, task in tasks])
        self.assertEqual(mission.ugv_hits, {"": 3})


if __name__ == "__main__":
    unittest.main()
