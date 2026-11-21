"""상대 UGV 따라가며 쏘기 (ugv_track.py · mission.engage_ugv) 시험. ROS 없이 돈다. 기본 흐름에서는 꺼져 있다."""
import math
import unittest

from supervisor import ugv_track
from supervisor.aim import AimParams, laser_pixel
from supervisor.mission import Facts, Mission, MissionParams

AIM = AimParams()
DEG_PX = AIM.deg_px(127.4)


def seen_from(need_pan, need_tilt, pan, tilt, z=1.0, owner="ugv", camp="RED", host_width=0.103, conf=0.8, callsign="", px=14.0):
    """맞히려면 팬틸트가 (need_pan, need_tilt) 여야 하는 수광부가, 팬틸트가 (pan, tilt) 일 때 사진에 찍히는 자리 (SensorSpot 의 값).
    host_width 0.103 = 66 픽셀 = 13도 → 차체 폭 0.23 m 로 약 1 m. px = 수광부가 찍힌 크기 (픽셀)."""
    lu, lv = laser_pixel(z, AIM)
    u, v = lu + (need_pan - pan) / DEG_PX, lv - (need_tilt - tilt) / DEG_PX
    return {"owner": owner, "x_center": u / AIM.image_w, "y_center": v / AIM.image_h, "width": px / AIM.image_w, "height": px / AIM.image_h,
            "confidence": conf, "has_host": True, "host_kind": "ugv", "camp": camp, "callsign": callsign,
            "host_x_center": u / AIM.image_w, "host_y_center": v / AIM.image_h, "host_width": host_width, "host_height": host_width}


class ChooseTests(unittest.TestCase):
    def test_only_foe_ugv_sensors_are_chosen(self):
        p = ugv_track.TrackParams()
        foe = seen_from(5, 0, 0, 0)
        friend = seen_from(-5, 0, 0, 0, camp="BLUE")
        unread = seen_from(8, 0, 0, 0, camp="")
        target = seen_from(0, 0, 0, 0, owner="target", camp="")
        self.assertIs(ugv_track.choose_ugv([friend, unread, target, foe], "RED", p), foe)
        self.assertIsNone(ugv_track.choose_ugv([friend, unread, target], "RED", p))
        self.assertIs(ugv_track.choose_ugv([friend, unread, target], "RED", p, allow_unknown_camp=True), unread)

    def test_range_from_the_width_of_the_ugv_box(self):
        self.assertAlmostEqual(ugv_track.ugv_range(seen_from(0, 0, 0, 0), DEG_PX, ugv_track.TrackParams()), 1.0, delta=0.05)


class TrackerTests(unittest.TestCase):
    def test_leads_a_target_moving_at_a_steady_rate(self):
        t = ugv_track.UgvTracker(aim=AIM, deg_per_px=DEG_PX)
        t.observe(0.0, seen_from(0.0, 2.0, 0.0, 0.0), 0.0, 0.0)
        self.assertFalse(t.aligned(0.1, 0.0, 2.0))  # 한 번 본 것으로는 쏘지 않는다
        t.observe(1.0, seen_from(4.0, 2.0, 0.0, 0.0), 0.0, 0.0)  # 1초에 4도
        pan, tilt = t.command(1.0)
        self.assertAlmostEqual(pan, 4.0 + 4.0 * t.p.lead_s, delta=0.3)
        self.assertAlmostEqual(tilt, 2.0, delta=0.3)
        self.assertTrue(t.aligned(1.5, 6.0, 2.0))
        self.assertFalse(t.aligned(1.5, 3.0, 2.0))
        self.assertIsNone(t.command(1.0 + t.p.max_age_s + 0.1))  # 오래 못 봤으면 겨누지 않는다


class FakeTurret:
    """팬틸트 · 발사 · 수광부 자리 답을 흉내 낸다. 표적은 need(now) 각도에 있다."""

    def __init__(self, need, camp="RED", delay=0.2, callsign="", others=()):
        self.need, self.camp, self.delay, self.callsign = need, camp, delay, callsign
        self.others = others  # 같이 보이는 다른 상대 수광부들: [(need(now) 함수, 찍힌 크기 픽셀, 호출부호)]
        self.pan, self.tilt, self.now = 0.0, 0.0, 0.0
        self.mail, self.fires = [], []

    def aim(self, pan, tilt):
        self.pan, self.tilt = pan, tilt
        return True

    def aim_ready(self):
        return True

    def fire(self):
        need = self.need(self.now)
        self.fires.append(math.hypot(need[0] - self.pan, need[1] - self.tilt))
        self.mail.append((self.now + 0.1, "fire", ("fired", 10000)))
        return True

    def locate(self, tag):
        need = self.need(self.now)
        spots = [seen_from(need[0], need[1], self.pan, self.tilt, camp=self.camp, callsign=self.callsign)]
        for other, px, sign in self.others:
            at = other(self.now)
            spots.append(seen_from(at[0], at[1], self.pan, self.tilt, camp=self.camp, callsign=sign, px=px))
        self.mail.append((self.now + self.delay, "sensors", (tag, self.pan, self.tilt, spots)))
        return True

    def run(self, mission, seconds):
        while self.now < seconds:
            f = Facts(now=self.now, t=self.now, phase="running", control=True, arrived=True)
            for _at, kind, value in [m for m in self.mail if m[0] <= self.now]:
                if kind == "sensors":
                    f.sensor_answers.append(value)
                else:
                    f.fire_reply = value
            self.mail = [m for m in self.mail if m[0] > self.now]
            mission._absorb(f)
            mission.engage_ugv(f, self)
            self.now += 0.1


def turret_mission(**params):
    mission = Mission(MissionParams(**params))
    mission.camp, mission.foe = "BLUE", "RED"
    return mission


class StillTests(unittest.TestCase):
    def test_still_after_a_few_looks_in_the_same_place(self):
        t = ugv_track.UgvTracker(aim=AIM, deg_per_px=DEG_PX)
        for i, when in enumerate((0.0, 0.4, 0.8)):
            t.observe(when, seen_from(5.0 + 0.2 * (i % 2), 1.0, 0.0, 0.0), 0.0, 0.0)  # 판독 잡음 0.2도
            self.assertIsNone(t.still_point(when))  # 1초를 채우지 못했다
        t.observe(1.2, seen_from(5.0, 1.0, 0.0, 0.0), 0.0, 0.0)
        pan, tilt = t.still_point(1.2)
        self.assertAlmostEqual(pan, 5.1, delta=0.3)
        self.assertAlmostEqual(tilt, 1.0, delta=0.3)
        self.assertIsNone(t.still_point(1.2 + t.p.max_age_s + 0.1))  # 오래 못 봤으면 모른다

    def test_not_still_while_it_moves(self):
        t = ugv_track.UgvTracker(aim=AIM, deg_per_px=DEG_PX)
        for i in range(8):
            t.observe(0.4 * i, seen_from(-10.0 + 1.2 * i, 1.0, 0.0, 0.0), 0.0, 0.0)  # 초당 3도
            self.assertIsNone(t.still_point(0.4 * i))
        for i in range(8, 12):  # 멈췄다
            t.observe(0.4 * i, seen_from(-0.4, 1.0, 0.0, 0.0), 0.0, 0.0)
        self.assertIsNotNone(t.still_point(0.4 * 11))


class EngageUgvTests(unittest.TestCase):
    def test_fires_at_a_foe_that_stands_still(self):
        turret = FakeTurret(lambda t: (12.0, 2.0))
        mission = turret_mission()
        turret.run(mission, 6.0)
        self.assertEqual(len(turret.fires), 1)  # 발사 간격(10초) 안에는 한 발
        self.assertLess(turret.fires[0], 1.0, turret.fires)
        self.assertEqual(mission.ugv_shots, 1)

    def test_does_not_fire_at_a_foe_that_keeps_moving(self):
        turret = FakeTurret(lambda t: (-20.0 + 3.0 * t, 2.0))  # 초당 3도로 가로지른다
        turret.run(turret_mission(), 10.0)
        self.assertEqual(turret.fires, [])

    def test_fires_once_the_moving_foe_stops(self):
        turret = FakeTurret(lambda t: (-20.0 + 3.0 * min(t, 5.0), 2.0))  # 5초 뒤 멈춘다
        turret.run(turret_mission(), 12.0)
        self.assertEqual(len(turret.fires), 1)
        self.assertLess(turret.fires[0], 1.0, turret.fires)

    def test_tracks_a_moving_foe_and_fires_when_aligned(self):
        turret = FakeTurret(lambda t: (-10.0 + 3.0 * t, 2.0))  # 왼쪽에서 오른쪽으로 초당 3도
        mission = turret_mission(ugv_shots=1, engage_still_only=False)  # 전의 동작: 움직이는 것도 앞을 겨눠 쏜다
        turret.run(mission, 8.0)
        self.assertEqual(len(turret.fires), 1)
        self.assertLess(turret.fires[0], 1.5, turret.fires)
        self.assertEqual(mission.ugv_shots, 1)
        self.assertEqual(mission.target_shots, 0)  # 지정표적 발 수에는 세지 않는다
        self.assertLess(mission._laser_until, 0)  # 지정표적용 「레이저 켜짐」 시간은 건드리지 않는다

    def test_never_fires_at_a_friendly_or_unidentified_ugv(self):
        for camp in ("BLUE", ""):
            turret = FakeTurret(lambda t: (5.0, 0.0), camp=camp)
            turret.run(turret_mission(ugv_shots=1), 5.0)
            self.assertEqual(turret.fires, [], camp)

    def test_does_not_fire_without_the_locate_port(self):
        class NoLocate:
            pass
        self.assertFalse(turret_mission().engage_ugv(Facts(now=0.0, arrived=True), NoLocate()))

    def test_on_by_default_but_only_for_still_targets(self):
        p = MissionParams()  # 2026-11-20 사용자 결정: 정찰이 끝나 서 있는 동안 멈춘 상대 UGV 를 쏜다
        self.assertTrue(p.engage_ugvs)
        self.assertTrue(p.engage_still_only)
        self.assertFalse(p.engage_unknown_camp)


class MotionTests(unittest.TestCase):
    """움직임의 정도: 멈춤 · 조금씩 · 움직임 · 많이 (UgvTracker.motion) 와 그에 따른 조준 · 발사."""

    @staticmethod
    def watch(rate, seconds=3.0, px=14.0):
        t = ugv_track.UgvTracker(aim=AIM, deg_per_px=DEG_PX)
        when = 0.0
        while when <= seconds:
            t.observe(when, seen_from(-5.0 + rate * when, 1.0, 0.0, 0.0, px=px), 0.0, 0.0)
            when += 0.4
        return t, when - 0.4

    def test_kinds_by_how_fast_the_target_moves(self):
        for rate, kind in ((0.0, "still"), (0.1, "still"), (0.8, "creep"), (3.0, "moving"), (9.0, "fast")):
            t, now = self.watch(rate)
            self.assertEqual(t.motion(now), kind, rate)
        self.assertIsNone(ugv_track.UgvTracker(aim=AIM, deg_per_px=DEG_PX).motion(0.0))

    def test_a_creeping_target_is_aimed_ahead_along_its_line(self):
        t, now = self.watch(0.8)
        pan, _tilt, kind, raw_pan, _raw_tilt = t.aim_point(now)
        self.assertEqual(kind, "creep")
        self.assertAlmostEqual(pan, -5.0 + 0.8 * (now + t.p.lead_s), delta=0.15)
        self.assertEqual(pan, raw_pan)

    def test_a_small_far_sensor_must_hold_stiller(self):
        near, _now = self.watch(0.0, px=14.0)
        far, now = self.watch(0.0, px=4.0)
        self.assertAlmostEqual(near.tol(), 1.0)
        self.assertLess(far.tol(), 0.6)
        self.assertEqual(self.watch(0.8, px=4.0)[0].motion(now), "moving")  # 수광부가 작으면 같은 속도도 「조금씩」 이 아니다

    def test_keeps_following_the_same_ugv_when_a_bigger_one_shows_up(self):
        t = ugv_track.UgvTracker(aim=AIM, deg_per_px=DEG_PX)
        mine = seen_from(5.0, 1.0, 0.0, 0.0)
        spot, switched = t.choose([mine], 0.0, 0.0, 0.0)
        self.assertIs(spot, mine)
        t.observe(0.0, mine, 0.0, 0.0)
        bigger = seen_from(-20.0, 1.0, 0.0, 0.0, px=30.0)
        spot, switched = t.choose([bigger, seen_from(5.3, 1.0, 0.0, 0.0)], 0.4, 0.0, 0.0)
        self.assertFalse(switched)
        self.assertIsNot(spot, bigger)
        spot, switched = t.choose([bigger], 0.8, 0.0, 0.0)  # 따라가던 것이 사라졌다 → 다른 것으로 바꾼다
        self.assertIs(spot, bigger)
        self.assertTrue(switched)

    def test_fires_at_a_foe_that_creeps(self):
        turret = FakeTurret(lambda t: (5.0 + 0.8 * t, 2.0))  # 초당 0.8도로 조금씩 움직인다
        turret.run(turret_mission(), 8.0)
        self.assertEqual(len(turret.fires), 1)
        self.assertLess(turret.fires[0], 0.6, turret.fires)

    def test_two_foes_in_view_do_not_stop_it_from_firing(self):
        turret = FakeTurret(lambda t: (12.0, 2.0), others=[(lambda t: (-15.0, 2.0), 13.0, "")])
        turret.run(turret_mission(), 6.0)
        self.assertEqual(len(turret.fires), 1)
        self.assertLess(turret.fires[0], 1.0, turret.fires)


class HitTests(unittest.TestCase):
    """우리가 맞힌 상대는 목숨을 낮춰 세고 M6 순위에서 내려간다 (mission.lives_of · ranking)."""

    def test_a_hit_lowers_that_team_in_the_ranking(self):
        turret = FakeTurret(lambda t: (12.0, 2.0), callsign="R-2")
        mission = turret_mission()
        turret.run(mission, 8.0)
        self.assertEqual(len(turret.fires), 1)
        self.assertEqual(mission.ugv_hits, {"R-2": 1})
        self.assertEqual(mission.lives_of("R-2"), 3)
        self.assertEqual(mission.ranking()[-1], "R-2")  # 나머지 둘은 모르는 값(3.5)
        self.assertTrue(any("맞힌 것으로 본다" in text for _t, text in mission.log))

    def test_the_led_read_cannot_raise_it_back(self):
        turret = FakeTurret(lambda t: (12.0, 2.0), callsign="R-2")
        mission = turret_mission()
        turret.run(mission, 8.0)
        for _ in range(3):  # 판독이 늦거나 틀려 여전히 파랑(4)으로 읽힌다
            mission._absorb(Facts(now=9.0, t=9.0, ugv_answers=[{"camp": "RED", "callsign": "R-2", "lives": 4, "ammo": -1}]))
        self.assertEqual(mission.ugvs["R-2"]["lives"], 4)
        self.assertEqual(mission.lives_of("R-2"), 3)
        mission._absorb(Facts(now=9.5, t=9.5, ugv_answers=[{"camp": "RED", "callsign": "R-2", "lives": 2, "ammo": -1}] * 1))
        mission._absorb(Facts(now=9.6, t=9.6, ugv_answers=[{"camp": "RED", "callsign": "R-2", "lives": 2, "ammo": -1}] * 1))
        self.assertEqual(mission.lives_of("R-2"), 2)  # 판독이 더 낮으면 판독을 따른다
        off = turret_mission(ugv_hits_lower_lives=False)
        off.ugvs["R-2"] = {"lives": 4, "lives_seen": [4], "ammo": None, "t": 0.0, "cap": 3}
        self.assertEqual(off.lives_of("R-2"), 4)

    def test_two_hits_lower_it_twice(self):
        turret = FakeTurret(lambda t: (12.0, 2.0), callsign="R-1")
        mission = turret_mission(fire_interval_s=4.0)
        turret.run(mission, 14.0)
        self.assertGreaterEqual(len(turret.fires), 2)
        self.assertEqual(mission.lives_of("R-1"), 4 - mission.ugv_hits["R-1"])

    def test_a_shot_the_target_walked_out_of_is_not_a_hit(self):
        fired = []
        turret = FakeTurret(lambda t: (12.0, 2.0) if not fired or t < fired[0] + 0.3 else (12.0 + 4.0 * (t - fired[0]), 2.0), callsign="R-2")
        fire = turret.fire
        turret.fire = lambda: (fired.append(turret.now), fire())[1]
        mission = turret_mission()
        turret.run(mission, 8.0)
        self.assertEqual(len(turret.fires), 1)
        self.assertEqual(mission.ugv_hits, {})
        self.assertIsNone(mission.lives_of("R-2"))
        self.assertTrue(any("맞힌 것으로 치지 않는다" in text for _t, text in mission.log))

    def test_whom_we_shot_is_taken_from_the_last_read_when_the_sensor_has_no_callsign(self):
        turret = FakeTurret(lambda t: (12.0, 2.0))
        mission = turret_mission()
        mission._absorb(Facts(now=0.0, t=0.0, ugv_answers=[{"camp": "RED", "callsign": "R-3", "lives": 4, "ammo": -1},
                                                         {"camp": "BLUE", "callsign": "B-2", "lives": 4, "ammo": -1}]))
        turret.run(mission, 8.0)
        self.assertEqual(mission.ugv_hits, {"R-3": 1})
        self.assertEqual(mission.lives_of("R-3"), 3)

    def test_a_team_with_no_lives_left_is_not_shot(self):
        turret = FakeTurret(lambda t: (12.0, 2.0), callsign="R-2")
        mission = turret_mission()
        mission.ugvs["R-2"] = {"lives": 0, "lives_seen": [0, 0], "ammo": None, "t": 0.0}
        turret.run(mission, 6.0)
        self.assertEqual(turret.fires, [])


class WatchTests(unittest.TestCase):
    """정찰이 끝나 서 있는 동안: 팬을 돌려 가며 살핀다 (mission._watch)."""

    class Head:
        def __init__(self):
            self.aims = []

        def aim(self, pan, tilt):
            self.aims.append(pan)
            return True

    def test_pans_around_while_nothing_is_tracked(self):
        mission, head = turret_mission(), self.Head()
        for i in range(100):
            mission._watch(Facts(now=0.1 * i, t=0.1 * i, arrived=True), head)
        self.assertEqual(head.aims[:4], [0.0, -40.0, 40.0, 0.0])
        self.assertEqual(len(head.aims), 4)  # 3초에 한 번

    def test_holds_still_while_the_laser_is_on(self):
        mission, head = turret_mission(), self.Head()
        mission._last_fire = 0.0
        for i in range(40):
            mission._watch(Facts(now=0.1 * i, t=0.1 * i, arrived=True), head)
        self.assertEqual(head.aims, [])

    def test_no_panning_when_the_list_is_empty(self):
        mission, head = turret_mission(watch_pans_deg=()), self.Head()
        mission._watch(Facts(now=0.0, t=0.0, arrived=True), head)
        self.assertEqual(head.aims, [])


class LivesTests(unittest.TestCase):
    def test_lives_follow_the_majority_of_the_last_three_reads(self):
        mission = turret_mission()

        def read(lives, callsign="R-2"):
            mission._absorb(Facts(now=0.0, t=0.0, ugv_answers=[{"camp": "RED", "callsign": callsign, "lives": lives, "ammo": -1}]))
            return mission.ugvs[callsign]["lives"]

        self.assertEqual(read(4), 4)
        self.assertEqual(read(2), 2)  # 둘이 갈리면 최근 값
        self.assertEqual(read(4), 4)  # 한 번 튄 값은 버린다
        self.assertEqual(read(-1), 4)  # 못 읽은 장은 세지 않는다
        self.assertEqual(read(3), 3)  # 4 · 4 · 3 이 아니라 2 · 4 · 3 → 다 다르면 최근 값
        self.assertEqual(read(3), 3)
        self.assertEqual(mission.ranking()[-1], "R-2")  # 목숨이 줄어든 것으로 읽힌 팀이 덜 위험하다 (나머지는 모르는 값 3.5)


if __name__ == "__main__":
    unittest.main()
