"""정찰 자리로 가는 도중에 사정이 바뀔 때 (2026-11-19 18:08 모의 경기 6차의 기록에서 본 두 가지).

  1. 가는 동안 navigation 이 그 상자를 지도에서 뺐다 → 끝까지 가서 찍지 않는다 (찍어도 적을 수 없다 — 사진 짝짓기는 뺀 상자의 면을 치지 않는다).
     그 판: 151.7초에 떠난 아래 벽 통로의 자리 (3.96, 0.39) 는 상자 9 의 S 면 하나를 보는 자리였다. 156.2초에 상자 9 가 빠졌는데도
     끝까지 가서 찍었고, 로봇은 그 통로에서 경기 끝까지 나오지 못했다.
  2. 가는 동안 지정표적 사격이 끼어들었다 → 사격을 마치면 가던 자리로 다시 간다 (가 보지도 않은 자리를 「가지 못함」 으로 미루지 않는다).
     그 판: 13.2초에 시킨 (5.23, 3.23) 을 28.0초에 사격이 끝나자마자 맨 뒤로 미뤘다.
  3. 벽 쪽 통로의 자리를 정한 뒤에, navigation 이 그 통로 안으로 나와 있는 상자(지도에는 없는 것)를 라이다로 찾았다 → 그 통로에는 들어가지 않는다.
     그 판: 지도는 붙은 상자 둘을 (4.45, 0.99) 의 상자 하나로 그렸고(아래 벽까지 0.79 m), navigation 은 44.9초부터 (4.77, 0.77) 의 상자를 알렸다.
"""
import unittest

from supervisor import arena as A
from supervisor.faces import FaceBook
from supervisor.grid import Grid
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import FaceScout, ScoutParams

# 떨어져 선 상자 셋: 공통 구역 하나, 상대(RED) 진영 둘 — 자리 하나가 상자 하나의 면만 보게
BOXES = [(3.06, 2.5), (4.46, 3.9), (4.46, 1.1)]
START = (1.2, 2.5, 0.0)


def scout_on(boxes=BOXES, start=START):
    fmap = A.field_map(boxes)
    book = FaceBook(fmap, "BLUE")
    return FaceScout(fmap, Grid(fmap), book, params=ScoutParams(), start=start), fmap, book


def walk(scout, pose, now=0.0, limit=4000):
    """정찰을 끝까지 흉내 낸다 (가라는 자리에 바로 가 있고, 답은 오지 않는다). 돌려주는 값: 찍은 면의 key 를 차례대로."""
    shot = []
    for _ in range(limit):
        act = scout.step(pose, now)
        if act[0] == "done":
            break
        if act[0] == "go":
            pose = (act[1], act[2], act[3])
        elif act[0] == "shoot":
            shot.append(act[2][1])
            scout.shot_taken(act[2], now)
            scout.out.clear()
        now += 0.1
    return shot


class Ports:
    """node.py 의 창구 흉내 (test_blocked.py 의 것과 같은 모양): 같은 목표는 다시 보내지 않고, 이동을 거두면 「서 있음」 이다."""

    def __init__(self):
        self.goes, self.goal, self.shots = [], None, []

    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None):
        goal = (round(x, 2), round(y, 2), round(yaw, 2), mode)
        if goal != self.goal:
            self.goal = goal
            self.goes.append((round(x, 2), round(y, 2)))

    def stop(self):
        self.goal = None

    @property
    def arrived(self):
        return self.goal is None

    def aim(self, *a): return True
    def fire(self): return True

    def shoot(self, tag, pan_deg):
        self.shots.append(tag)
        return True

    def look_ugvs(self): return False
    def analyze_order(self, text): return True
    def set_ready(self): return True
    def report(self, kind, payload): return True


class RemovedOnTheWayTests(unittest.TestCase):
    def test_the_stop_is_left_when_its_box_is_removed_on_the_way(self):
        scout, fmap, book = scout_on()
        act = scout.step(START, 0.0)
        self.assertEqual(act[0], "go")
        stop = scout.stops[scout.i]
        box = {k[0] for k in stop[3]}
        self.assertEqual(len(box), 1, stop)  # 이 배치에서는 자리 하나가 상자 하나의 면만 본다
        halfway = ((START[0] + act[1]) / 2.0, (START[1] + act[2]) / 2.0, START[2])
        self.assertEqual(scout.step(halfway, 1.0)[:3], act[:3])  # 아직은 그 자리로 간다
        fmap.set_gone(box)  # 가는 동안 navigation 이 그 상자를 뺐다
        after = scout.step(halfway, 1.1)
        self.assertEqual(after[0], "go")
        now_stop = scout.stops[scout.i]
        self.assertFalse({k[0] for k in now_stop[3]} & box, now_stop)  # 다른 상자의 면을 보는 자리로 간다
        self.assertIn((round(stop[0], 2), round(stop[1], 2)), scout.deferred)  # 그 자리는 맨 뒤로 미뤘다
        self.assertEqual(scout.queue and scout.queue[0][0] in box, False)
        self.assertEqual(len([line for line in scout.log if "가는 동안 navigation 이 상자를 뺀 면" in line]), 1)
        # 끝까지 돌면: 뺀 상자의 면은 다른 면을 모두 찍은 뒤에야 한 번 가 본다 (전부터의 규칙 — 잘못 뺀 것일 수 있다)
        shot = walk(scout, halfway, 1.2)
        gone_at = [n for n, key in enumerate(shot) if key[0] in box]
        live = {key for key in shot if key[0] not in box}
        self.assertTrue(gone_at and live)
        self.assertEqual({key for key in shot[:gone_at[0]]}, live)

    def test_only_the_removed_boxs_faces_leave_a_stop_that_sees_two_boxes(self):
        """자리 하나가 두 상자의 면을 볼 때: 빠진 상자의 면만 미루고 남은 면은 그 자리에서 찍는다."""
        scout, fmap, book = scout_on()
        scout.step(START, 0.0)
        x, y, heading, keys = scout.stops[scout.i]
        other = next(f.key for f in book.goal if f.key[0] not in {k[0] for k in keys})
        scout.stops[scout.i] = (x, y, heading, list(keys) + [other])  # 시험용으로 다른 상자의 면 하나를 이 자리에 더 맡긴다
        scout.queue = list(keys) + [other]
        fmap.set_gone({other[0]})
        act = scout.step(START, 0.5)
        self.assertEqual((act[0], act[1], act[2]), ("go", x, y))  # 자리는 그대로 간다
        self.assertEqual(scout.queue, list(keys))
        self.assertEqual(scout.stops[-1][3], [other])  # 빠진 상자의 면은 맨 뒤의 자리로

    def test_a_stop_started_for_a_box_already_removed_is_not_interrupted(self):
        """미뤘다가 맨 뒤에 한 번 가 보는 자리(빠진 줄 알고 가는 자리)는 가는 도중에 그만두지 않는다."""
        scout, fmap, book = scout_on()
        scout.plan(START)
        box = scout.stops[0][3][0][0]
        fmap.set_gone({box})
        shot = walk(scout, START)
        self.assertTrue([key for key in shot if key[0] == box])  # 맨 뒤에 한 번은 찍으러 갔다
        self.assertFalse([line for line in scout.log if "가는 동안 navigation 이 상자를 뺀 면" in line])

    def test_a_face_whose_box_is_removed_while_standing_at_the_stop_is_not_shot(self):
        scout, fmap, book = scout_on()
        act = scout.step(START, 0.0)
        stop = scout.stops[scout.i]
        there = (act[1], act[2], act[3])
        self.assertEqual(scout.step(there, 1.0)[0], "shoot")  # 자리에 왔다 → 찍으려 한다
        fmap.set_gone({stop[3][0][0]})
        act = scout.step(there, 1.1)
        self.assertNotEqual(act[0], "shoot")
        self.assertEqual(scout.shots, 0)


class MissionTests(unittest.TestCase):
    def facts(self, now, pose, ports, **more):
        return Facts(now=now, t=now, phase="running", control=True, slot="B-2", pose=pose, arrived=ports.arrived, boxes=BOXES, **more)

    def test_mission_sends_the_next_stop_when_the_box_is_removed_on_the_way(self):
        mission, ports = Mission(MissionParams()), Ports()
        mission.tick(self.facts(0.0, START, ports), ports)
        mission.tick(self.facts(0.1, START, ports), ports)
        first = ports.goes[-1]
        box = {k[0] for k in mission.scout.stops[mission.scout.i][3]}
        mission.tick(self.facts(0.2, START, ports, gone=frozenset(box)), ports)
        self.assertEqual(mission.fmap.gone, frozenset(box))
        mission.tick(self.facts(0.3, START, ports, gone=frozenset(box)), ports)
        self.assertNotEqual(ports.goes[-1], first)  # 다음 자리를 시켰다
        self.assertFalse({k[0] for k in mission.scout.stops[mission.scout.i][3]} & box)
        self.assertFalse(ports.shots)
        # 정찰이 정한 것은 경기 기록(mission.log → events.jsonl 의 note)에도 남는다: 계획, 그리고 그 자리를 지나친 까닭
        told = [text for _t, text in mission.log if text.startswith("정찰: ")]
        self.assertEqual(len([text for text in told if "정찰 계획: 설 자리" in text]), 1, told)
        self.assertEqual(len([text for text in told if "가는 동안 navigation 이 상자를 뺀 면" in text]), 1, told)
        self.assertEqual(len(told), len(mission.scout.log))  # step 이 적은 줄은 모두, 한 번씩

    def test_the_stop_asked_before_a_strike_is_visited_after_it(self):
        mission, ports = Mission(MissionParams()), Ports()
        order = {"target": "RED_OUTPOST", "search_sectors": ["CM"], "base": "RED_MAINTENANCE"}
        mission.tick(self.facts(0.0, START, ports, parsed=order), ports)
        mission.tick(self.facts(0.1, START, ports, parsed=order), ports)
        asked = ports.goes[-1]
        # 그 자리로 가는 중에 지정표적 면을 알게 됐다 (지나가며 찍은 사진으로 적힌 경우) → 사격이 끼어든다
        mission.target_face = next(f for f in mission.fmap.faces if f.key == (1, "W"))
        mission.tick(self.facts(0.2, START, ports, parsed=order), ports)
        self.assertEqual(mission.task, "strike")
        self.assertNotEqual(ports.goes[-1], asked)  # 쏠 자리로 가는 목표로 바뀌었다
        # 사격이 끝났다: 레이저가 켜진 동안 서 있다가(이동을 거둔다) 정찰로 돌아온다
        mission.target_done, mission._laser_until = True, 0.6
        mission.tick(self.facts(0.3, START, ports, parsed=order), ports)
        self.assertTrue(ports.arrived)
        mission.tick(self.facts(1.0, START, ports, parsed=order), ports)
        self.assertTrue(mission.task.startswith("scout"), mission.task)
        self.assertEqual(ports.goes[-1], asked)  # 가던 자리로 다시 간다
        self.assertEqual(mission.scout.skipped, 0)
        self.assertFalse([text for _t, text in mission.log if "가지 못함" in text], mission.log)


WALL_BOX = (4.449, 0.986)  # 그 판의 상자 9: 아래 벽까지 0.79 m → S 면은 벽 쪽 통로의 면이다
LIDAR_BOX = (4.766, 0.774)  # 그 판에서 navigation 이 알린 상자: 벽 쪽 변이 상자 9 의 S 면보다 0.21 m 더 벽 쪽


class NarrowedCorridorTests(unittest.TestCase):
    def scout(self, lidar):
        scout, fmap, book = scout_on([(3.06, 2.5), WALL_BOX])
        scout.plan(START)
        self.assertIn((1, "S"), scout.corridor)  # 계획은 그 면을 통로의 자리에서 찍기로 했다
        scout.lidar_boxes = lidar
        return scout

    def test_which_lidar_boxes_narrow_the_corridor(self):
        self.assertTrue(self.scout([LIDAR_BOX])._corridor_narrowed((1, "S")))
        self.assertTrue(self.scout([(0.3, 2.0), LIDAR_BOX])._corridor_narrowed((1, "S")))
        self.assertTrue(self.scout([(3.9, 0.3)])._corridor_narrowed((1, "S")))  # 통로 어귀에 선 것 (서 있는 로봇이 상자로 잡힌 경우도 같다)
        for name, lidar in (("목록을 받은 적 없음", None), ("빈 목록", []),
                            ("같은 줄에 나란히 붙은 상자 (지도에 빠진 짝)", [(WALL_BOX[0] + 0.40, WALL_BOX[1])]),
                            ("5 cm 더 벽 쪽 — 잰 값의 차이로 본다", [(WALL_BOX[0] + 0.40, WALL_BOX[1] - 0.05)]),
                            ("상자의 반대쪽(N 면 쪽)", [(WALL_BOX[0], WALL_BOX[1] + 0.45)]),
                            ("벽 밖에 걸친 것 (벽의 점으로 생긴 헛것)", [(4.52, -0.12)]),
                            ("통로를 따라 멀리", [(2.6, 0.5)]), ("모양이 다른 목록", [None])):
            self.assertFalse(self.scout(lidar)._corridor_narrowed((1, "S")), name)
        self.assertFalse(self.scout([LIDAR_BOX])._corridor_narrowed((1, "N")))  # 통로의 면이 아니다
        off = self.scout([LIDAR_BOX])
        off.p.corridor_intrude_m = 0.0
        self.assertFalse(off._corridor_narrowed((1, "S")))

    def test_the_corridor_stop_is_put_off_and_then_skipped_while_narrowed(self):
        plain = walk(self.scout(None), START)
        self.assertIn((1, "S"), plain)  # 라이다 상자가 없으면 전처럼 통로에 들어가 찍는다
        scout = self.scout([LIDAR_BOX])
        shot = walk(scout, START)
        self.assertNotIn((1, "S"), shot)
        self.assertEqual({key for key in plain if key != (1, "S")}, set(shot))  # 다른 면은 모두 그대로 찍는다
        said = [line for line in scout.log if "통로 안으로 라이다 상자가 나와 있다" in line]
        self.assertEqual([line.split("→ ")[1] for line in said], ["뒤로 미룬다", "가지 않는다"])
        self.assertIn((1, "S"), scout.book.remaining())

    def test_the_put_off_stop_is_visited_when_the_lidar_box_has_gone_by_then(self):
        """미룬 뒤에 그 상자가 목록에서 사라졌으면(지나가던 로봇 등) 맨 뒤에서 그 통로 자리에 간다."""
        scout = self.scout([LIDAR_BOX])
        pose, now, shot = START, 0.0, []
        for _ in range(4000):
            act = scout.step(pose, now)
            if act[0] == "done":
                break
            if any("통로 안으로 라이다 상자가 나와 있다" in line for line in scout.log):
                scout.lidar_boxes = []  # 한 번 미룬 뒤에 목록에서 사라졌다
            if act[0] == "go":
                pose = (act[1], act[2], act[3])
            elif act[0] == "shoot":
                shot.append(act[2][1])
                scout.shot_taken(act[2], now)
                scout.out.clear()
            now += 0.1
        self.assertEqual(shot[-1], (1, "S"))

    def test_mission_hands_the_lidar_list_to_the_scout(self):
        mission, ports = Mission(MissionParams()), Ports()
        boxes = [(3.06, 2.5), WALL_BOX]
        for now in (0.0, 0.1):
            mission.tick(Facts(now=now, t=now, phase="running", control=True, slot="B-2", pose=START, arrived=ports.arrived, boxes=boxes, extra_boxes=[LIDAR_BOX]), ports)
        self.assertEqual(mission.scout.lidar_boxes, [LIDAR_BOX])
        self.assertTrue(mission.scout._corridor_narrowed((1, "S")))
        off, ports = Mission(MissionParams(extra_boxes=False)), Ports()  # 라이다 상자를 통째로 끄면 넘기지도 않는다
        for now in (0.0, 0.1):
            off.tick(Facts(now=now, t=now, phase="running", control=True, slot="B-2", pose=START, arrived=ports.arrived, boxes=boxes, extra_boxes=[LIDAR_BOX]), ports)
        self.assertIsNone(off.scout.lidar_boxes)


if __name__ == "__main__":
    unittest.main()
