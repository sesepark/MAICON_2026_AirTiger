"""자리에 서자마자 찍기: 자리 · 방향이 되면 navigation 을 기다리지 않고 이동을 거두고, 팬은 차체가 멈추기를 기다리지 않고 보낸다.

2026-11-19 16:01 실차의 시간을 쪼갠 결과: 정찰 207초 가운데 자리에 서서 쓴 것이 111초 — 방향 맞추기 56초, 팬이 돌기를 기다린 것 30초,
「차체가 멈춘 뒤에야 팬을 보내는」 기다림 12초. 면이 팬에 닿게 된 때부터 사진까지 가운데값 4.3초였다.
"""
import math
import unittest

from supervisor import arena as A
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import PAN_LIMIT_DEG


class Ports:
    """node.py 의 창구 흉내 (이동 · 팬 · 사진만)."""

    def __init__(self):
        self.goes, self.stops, self.shots, self.aims = [], 0, [], []
        self.goal, self.reached, self.pan = None, False, 0.0

    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None):
        goal = (round(x, 2), round(y, 2), round(yaw, 2), mode)
        if goal == self.goal:
            return
        self.goal, self.reached = goal, False
        self.goes.append(goal)

    def stop(self):
        if self.goal is not None:
            self.goal = None
            self.stops += 1

    @property
    def arrived(self):
        return self.goal is None or self.reached

    def aim(self, pan, tilt):
        self.aims.append((round(pan, 1), tilt))
        self.pan = pan
        return True

    def pan_now(self):
        return self.pan

    def shoot(self, tag, pan_deg):
        self.shots.append((tag, round(pan_deg, 1)))
        self.pan = pan_deg
        return True

    def fire(self): return True
    def look_ugvs(self): return False
    def analyze_order(self, text): return True
    def set_ready(self): return True
    def report(self, kind, payload): return True
    def aim_ready(self): return True


class ShootEarlyTests(unittest.TestCase):
    BOXES = [(3.06, 2.5)]  # 가운데 상자 하나 (벽에서 멀다 — 통로가 아니다)

    def start(self, **params):
        self.fmap = A.field_map(self.BOXES)
        self.mission = Mission(MissionParams(pass_gap_s=0.0, **params))
        self.ports = Ports()
        self.now = 0.0
        self.pose = (0.3, 2.5, 0.0)
        for _ in range(5):
            self.tick()
        self.assertTrue(self.ports.goes)
        return self.ports.goes[-1]

    def tick(self, pose=None):
        self.now += 0.1
        if pose is not None:
            self.pose = pose
        self.mission.tick(Facts(now=self.now, t=1.0 + self.now, phase="running", control=True, slot="B-2", pose=self.pose, arrived=self.ports.arrived,
                                fmap=self.fmap), self.ports)

    def face_pan(self, pose):
        face = self.mission.scout.next_face()
        return -math.degrees(math.atan2(math.sin(math.atan2(face.cy - pose[1], face.cx - pose[0]) - pose[2]),
                                        math.cos(math.atan2(face.cy - pose[1], face.cx - pose[0]) - pose[2])))

    def test_stops_and_aims_at_once_then_shoots_after_the_settle(self):
        x, y, yaw, _mode = self.start()
        there = (x, y, yaw + math.radians(30.0))  # 자리에 왔고 방향은 허용(±56°) 안 — navigation 은 아직 「도착」 을 알리지 않았다
        self.tick(there)
        self.assertEqual(self.ports.stops, 1)  # 서 있기를 1초 기다리지 않고 바로 거둔다
        self.assertTrue(self.ports.aims)  # 팬도 바로 보낸다
        self.assertAlmostEqual(self.ports.aims[-1][0], 30.0, delta=1.0)
        self.assertEqual(self.ports.shots, [])  # 차체가 가라앉기 전에는 찍지 않는다
        for _ in range(8):
            self.tick(there)
        self.assertEqual(len(self.ports.shots), 1)
        self.assertAlmostEqual(self.ports.shots[0][1], self.ports.aims[-1][0], delta=0.5)  # 미리 돌려 둔 각도 그대로 찍는다

    def test_old_behaviour_when_switched_off(self):
        x, y, yaw, _mode = self.start(shoot_early=False)
        there = (x, y, yaw + math.radians(30.0))
        self.tick(there)
        self.assertEqual(self.ports.stops, 0)  # 1초 서 있는 것을 본 뒤에야 거둔다

    def test_pan_follows_the_face_on_the_last_stretch(self):
        x, y, yaw, _mode = self.start()
        before = len(self.ports.aims)
        near = (x - 0.4 * math.cos(yaw + 0.5), y - 0.4 * math.sin(yaw + 0.5), yaw + 0.5)  # 자리 0.4 m 앞, 비스듬히 다가온다
        for _ in range(6):
            self.tick(near)
        self.assertEqual(self.ports.stops, 0)  # 아직 자리가 아니다
        self.assertGreater(len(self.ports.aims), before)
        want = max(-(PAN_LIMIT_DEG - 2.0), min(PAN_LIMIT_DEG - 2.0, self.face_pan(near)))
        self.assertAlmostEqual(self.ports.aims[-1][0], want, delta=1.0)
        self.assertLessEqual(len(self.ports.aims) - before, 2)  # 같은 자리에서 되풀이해 보내지 않는다

    def test_small_difference_keeps_the_pan_already_set(self):
        x, y, yaw, _mode = self.start()
        there = (x, y, yaw + math.radians(30.0))
        self.tick(there)
        sent = self.ports.aims[-1][0]
        nudged = (x, y, yaw + math.radians(33.0))  # 멈추면서 방향이 3° 더 돌았다
        for _ in range(8):
            self.tick(nudged)
        self.assertEqual(len(self.ports.shots), 1)
        self.assertAlmostEqual(self.ports.shots[0][1], sent, delta=0.5)  # 다시 돌리지 않는다 (다시 돌리면 멈추기를 또 기다린다)


class CorridorIsNotCutShortTests(unittest.TestCase):
    """벽 쪽 통로의 자리는 어긋남의 여유가 작다 → navigation 이 끝까지 가게 두고, 일찍 거두지 않는다."""

    def test_no_early_stop_at_a_corridor_stop(self):
        fmap = A.field_map([(4.3, 4.0), (4.7, 4.0)], A.arena_from({"width": 6.12, "height": 4.80}), None)
        mission, ports = Mission(MissionParams(pass_gap_s=0.0)), Ports()
        now, pose = 0.0, (0.3, 4.0, 0.0)
        for _ in range(5):
            now += 0.1
            mission.tick(Facts(now=now, t=1.0 + now, phase="running", control=True, slot="B-1", pose=pose, arrived=ports.arrived, fmap=fmap), ports)
        scout = mission.scout
        self.assertTrue(scout.corridor)
        key = sorted(scout.corridor)[0]
        scout.i = next(n for n, stop in enumerate(scout.stops) if list(stop[3]) == [key])
        scout.queue, scout.body, scout.any_heading = [], None, False
        for _ in range(3):
            now += 0.1
            mission.tick(Facts(now=now, t=1.0 + now, phase="running", control=True, slot="B-1", pose=pose, arrived=ports.arrived, fmap=fmap), ports)
        x, y, yaw, _mode = ports.goes[-1]
        stops = ports.stops
        now += 0.1
        mission.tick(Facts(now=now, t=1.0 + now, phase="running", control=True, slot="B-1", pose=(x, y, yaw), arrived=ports.arrived, fmap=fmap), ports)
        self.assertEqual(ports.stops, stops)  # 바로 거두지 않는다
        self.assertTrue(ports.aims)  # 팬은 미리 보낸다


if __name__ == "__main__":
    unittest.main()
