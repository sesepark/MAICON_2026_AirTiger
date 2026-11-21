"""위에서 본 UGV 자리 → 「서 있는 다른 로봇」 (supervisor/standing.py) 과, 그것이 mission 을 거쳐 정찰에 닿는 길.

2026-11-20 00:00 연습 판: 가운데에 1분 넘게 서 있던 다른 팀 로봇 곁의 정찰 자리로 가다가 19초 뒤에야 막힘을 알았다 — navigation 이 서 있는 로봇의
라이다 상자를 목록에서 빼게 된 뒤로 정찰은 다른 로봇을 볼 길이 없었다. perception 이 드론 사진에서 찾은 자리(/team/perception/ugvs)를 받아 쓴다.
"""
import math
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from supervisor.mission import Facts, Mission, MissionParams
from supervisor.standing import Standing, add_sight
from test_blocked import Ports

US = [(1.0, 1.0)]  # 그 무렵 우리가 있던 자리


def feed(tracker, points, start=0.0, count=6, gap=1.0, recent=US):
    """같은 관측을 gap 초 간격으로 count 번 넣는다. 마지막 관측의 때를 돌려준다."""
    at = start
    for k in range(count):
        at = start + gap * k
        tracker.add(at, points, recent)
    return at


class StandingTests(unittest.TestCase):
    def test_a_robot_in_one_place_counts_after_the_minimum_time(self):
        tracker = Standing()
        at = feed(tracker, [(3.0, 2.4, 0.9)], count=3)  # 2초째
        self.assertEqual(tracker.robots(at, min_s=4.0), [])
        at = feed(tracker, [(3.02, 2.41, 0.9)], start=3.0, count=4)  # 6초째 (몇 cm 흔들려도 같은 자리다)
        got = tracker.robots(at, min_s=4.0)
        self.assertEqual(len(got), 1)
        self.assertEqual((got[0][0], got[0][1]), (3.0, 2.4))  # 처음 선 곳을 답한다 (값이 흔들려도 자리가 바뀌지 않게)
        self.assertAlmostEqual(got[0][2], 6.0)

    def test_a_moving_robot_is_not_standing_and_the_count_restarts_where_it_stops(self):
        tracker = Standing()
        for k in range(8):  # 1초에 0.3 m 씩 간다
            tracker.add(float(k), [(2.0 + 0.3 * k, 2.4)], US)
        self.assertEqual(tracker.robots(7.0), [])
        at = feed(tracker, [(4.1, 2.4)], start=8.0, count=6)  # 멈췄다
        got = tracker.robots(at)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0][2], 6.0, delta=1.01)  # 멈춘 곳에서부터 센다 (떠나온 시간은 치지 않는다)
        self.assertEqual(len(tracker.tracks), 1)  # 한 대로 이어 봤다

    def test_our_own_robot_is_left_out(self):
        tracker = Standing()
        at = feed(tracker, [(1.05, 0.95, 0.9), (3.0, 2.4, 0.9)], count=8)  # 앞의 것은 우리 차다 (우리 자리에서 7 cm)
        self.assertEqual([(x, y) for x, y, _s in tracker.robots(at)], [(3.0, 2.4)])
        # 우리가 움직이는 중이어도, 사진을 찍을 무렵 지나온 자리 가운데 하나에 맞으면 우리 차다
        tracker = Standing()
        for k in range(8):
            ours = [(1.0 + 0.25 * k - 0.05 * j, 1.0) for j in range(4)]
            tracker.add(float(k), [(ours[2][0], 1.0), (3.0, 2.4)], ours)
        self.assertEqual([(x, y) for x, y, _s in tracker.robots(7.0)], [(3.0, 2.4)])

    def test_a_second_robot_right_next_to_us_is_kept(self):
        tracker = Standing()
        at = feed(tracker, [(1.02, 1.0), (1.36, 1.0)], count=8)  # 둘 다 0.40 m 안 — 가장 가까운 하나만 우리 차다
        self.assertEqual([(x, y) for x, y, _s in tracker.robots(at)], [(1.36, 1.0)])

    def test_without_our_position_the_observation_is_dropped(self):
        tracker = Standing()
        self.assertFalse(tracker.add(0.0, [(1.0, 1.0), (3.0, 2.4)], []))
        self.assertEqual((tracker.tracks, tracker.dropped), ([], 1))

    def test_when_we_are_not_found_nothing_near_us_is_trusted(self):
        """우리 차를 못 골랐다(자리가 0.4 m 넘게 밀려 찍혔다): 우리 곁 0.8 m 안의 것은 우리 차일 수 있으니 그 관측에서는 쓰지 않는다."""
        tracker = Standing()
        at = feed(tracker, [(1.55, 1.0), (3.0, 2.4)], count=8)
        self.assertEqual([(x, y) for x, y, _s in tracker.robots(at)], [(3.0, 2.4)])

    def test_stale_data_is_not_used(self):
        tracker = Standing()
        at = feed(tracker, [(3.0, 2.4)], count=8)
        self.assertEqual(len(tracker.robots(at + 2.0)), 1)
        self.assertEqual(tracker.robots(at + 4.0), [])  # 관측이 끊겼다 (3.2초 넘게 새 것이 없다) → 모른다
        self.assertFalse(tracker.alive(at + 4.0))
        # 다시 오기 시작했는데 그 자리에 더는 안 보이면 잊는다 (3.5초 안 보임)
        feed(tracker, [], start=at + 5.0, count=3)
        self.assertEqual(tracker.tracks, [])
        # 앞의 것보다 새롭지 않은 관측은 쓰지 않는다 (같은 목록이 틱마다 다시 와도 된다)
        self.assertFalse(tracker.add(at, [(3.0, 2.4)], US))

    def test_slow_feed_of_one_picture_per_five_seconds_still_works(self):
        """우리 시설물이 파괴되면 사진이 2초 · 5초에 한 장으로 준다: 그만큼 넓게 잇고 오래 기억한다."""
        tracker = Standing()
        at = feed(tracker, [(3.0, 2.4), (5.0, 1.0)], count=3, gap=5.0)  # 10초째
        self.assertAlmostEqual(tracker.interval, 5.0)
        got = sorted(tracker.robots(at + 4.0))
        self.assertEqual([(x, y) for x, y, _s in got], [(3.0, 2.4), (5.0, 1.0)])
        self.assertAlmostEqual(got[0][2], 10.0)
        self.assertEqual(tracker.robots(at + 13.0), [])  # 다음 사진이 두 번 넘게 안 왔다

    def test_a_ghost_seen_once_never_counts(self):
        tracker = Standing()
        for k in range(10):
            tracker.add(float(k), [(3.0, 2.4)] + ([(5.0, 4.0)] if k == 4 else []), US)
        self.assertEqual([(x, y) for x, y, _s in tracker.robots(9.0)], [(3.0, 2.4)])

    def test_add_sight_keeps_the_newest_and_skips_bad_numbers(self):
        def pose(x, y, z=0.9):
            return SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z))

        sights = None
        for k in range(15):
            sights = add_sight(sights, float(k), [pose(1.0, 2.0), pose(float("nan"), 2.0)])
        self.assertEqual(len(sights), 12)
        self.assertEqual(sights[0][0], 3.0)
        self.assertEqual(sights[-1], (14.0, [(1.0, 2.0, 0.9)]))
        self.assertEqual(add_sight(None, 1.0, []), [(1.0, [])])  # 「봤는데 없다」 도 관측이다


class MissionRobotsTests(unittest.TestCase):
    BASE = dict(phase="running", control=True, slot="B-2", boxes=[(3.06, 2.5), (3.06, 3.5)])

    def run_ticks(self, sights_at, params=None, seconds=8.0):
        """8초 동안 0.1초마다 틱. sights_at(now) = 그 틱에 넘길 Facts.ugv_sights."""
        mission, ports = Mission(params or MissionParams()), Ports()
        pose = (1.2, 2.5, 0.0)
        now = 0.0
        while now <= seconds:
            mission.tick(Facts(now=now, t=now, pose=pose, arrived=True, ugv_sights=sights_at(now), **self.BASE), ports)
            now = round(now + 0.1, 1)
        return mission

    @staticmethod
    def seen(now):
        """1초마다 한 번, 0.6초 늦게 오는 관측들 (가장 새 12개): 우리 차 (1.2, 2.5) 와 (4.0, 2.5) 에 서 있는 다른 로봇."""
        stamps = [k + 0.4 for k in range(int(now) + 1) if k + 0.6 <= now]
        return [(at, [(1.22, 2.48, 0.9), (4.0, 2.5, 0.9)]) for at in stamps][-12:] if stamps else []

    def test_standing_robots_reach_the_scout_without_our_own(self):
        mission = self.run_ticks(self.seen)
        robots = mission.scout.robots
        self.assertEqual([(round(x, 2), round(y, 2)) for x, y, _s in robots], [(4.0, 2.5)])
        self.assertGreaterEqual(robots[0][2], 4.0)
        self.assertTrue(any("위에서 본 서 있는 다른 로봇 1대" in text for _t, text in mission.log))
        self.assertEqual(sum("위에서 본 서 있는 다른 로봇" in text for _t, text in mission.log), 1)  # 바뀐 때에만 적는다

    def test_nothing_changes_when_the_topic_never_comes_or_is_switched_off(self):
        self.assertIsNone(self.run_ticks(lambda now: None).scout.robots)
        self.assertIsNone(self.run_ticks(self.seen, MissionParams(overhead_robots=False)).scout.robots)

    def test_the_feed_stopping_empties_the_list(self):
        mission = self.run_ticks(lambda now: self.seen(min(now, 6.0)), seconds=12.0)  # 6초 뒤로는 새 관측이 없다
        self.assertEqual(mission.scout.robots, [])

    def test_a_broken_observation_does_not_stop_the_match(self):
        mission = self.run_ticks(lambda now: [(now, [("x", None)])] if now > 1.0 else None, seconds=3.0)
        self.assertIsNone(mission.scout.robots)
        self.assertTrue(any("위에서 본 UGV 자리를 쓰다 예외" in text for _t, text in mission.log))
        self.assertTrue(math.isfinite(mission.scout.i))  # 정찰은 그대로 돈다


class NodeCallbackTests(unittest.TestCase):
    """node.py 의 on_ugvs (토픽을 받는 곳)를 ROS 없이 돌려 본다 — ROS 모듈은 빈 것으로 바꿔 끼우고, 끝나면 되돌린다."""

    ROS = ["rclpy", "rclpy.action", "rclpy.node", "rclpy.qos", "rclpy.time", "geometry_msgs", "geometry_msgs.msg", "sensor_msgs", "sensor_msgs.msg",
           "team_interfaces", "team_interfaces.action", "team_interfaces.msg", "team_interfaces.srv", "mil_msgs", "mil_msgs.msg", "mil_msgs.srv"]

    def setUp(self):
        self.saved = {name: sys.modules.get(name) for name in self.ROS + ["supervisor.node"]}
        for name in self.ROS:
            sys.modules[name] = mock.MagicMock()
        sys.modules["rclpy.node"].Node = type("Node", (), {})
        sys.modules.pop("supervisor.node", None)
        from supervisor import node

        self.on_ugvs = node.SupervisorNode.on_ugvs
        self.notes = []
        self.me = SimpleNamespace(
            ugv_sights=None, get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(1000.0 * 1e9))),
            note=lambda name, make: self.notes.append((name, make())),
            get_logger=lambda: SimpleNamespace(warning=lambda *a, **k: self.notes.append(("warning", a))))

    def tearDown(self):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    @staticmethod
    def message(stamp, points, frame="arena"):
        sec = int(stamp)
        return SimpleNamespace(header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))),
                               poses=[SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z)) for x, y, z in points])

    def test_observations_are_kept_with_their_time_on_our_clock(self):
        began = time.monotonic()
        self.on_ugvs(self.me, self.message(999.7, [(1.0, 2.0, 0.9), (3.0, 2.5, 0.8)]))  # 사진은 0.3초 전에 받은 것이다 (ROS 시계 1000.0)
        at, points = self.me.ugv_sights[-1]
        self.assertEqual(points, [(1.0, 2.0, 0.9), (3.0, 2.5, 0.8)])
        self.assertAlmostEqual(at - began, -0.3, delta=0.05)
        self.assertEqual(self.notes, [("ugvs_seen", {"count": 2, "late_ms": 300})])
        self.on_ugvs(self.me, self.message(0.0, []))  # 시각이 안 적힌 메시지 → 받은 때를 쓴다. 빈 목록도 관측이다
        self.assertEqual(len(self.me.ugv_sights), 2)
        self.assertAlmostEqual(self.me.ugv_sights[-1][0] - began, 0.0, delta=0.05)

    def test_other_frames_and_broken_messages_change_nothing(self):
        self.on_ugvs(self.me, self.message(999.9, [(1.0, 2.0, 0.9)], frame="map"))
        self.assertIsNone(self.me.ugv_sights)
        self.on_ugvs(self.me, SimpleNamespace())  # 깨진 메시지: 예외를 밖으로 내지 않는다 (받는 스레드가 죽지 않게)
        self.assertIsNone(self.me.ugv_sights)
        self.assertEqual([name for name, _v in self.notes], ["warning"])


if __name__ == "__main__":
    unittest.main()
