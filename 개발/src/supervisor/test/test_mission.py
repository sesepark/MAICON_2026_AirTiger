import math
import unittest

from sim_world import SimMatch
from supervisor import arena as A


class MissionFlowTests(unittest.TestCase):
    """가상 경기 한 판을 끝까지 돌린다 (주행 · 인식 · 서버는 가짜). 실제 로봇 성능을 뜻하지 않는다."""

    @classmethod
    def setUpClass(cls):
        cls.m = SimMatch(mode="place", slot="B-1", pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0).run()

    def last(self, kind):
        return [r for r in self.m.reports if r[1] == kind][-1]

    def test_ready_and_m1_are_sent(self):
        self.assertTrue(self.m.mission.ready_ok)
        self.assertEqual(self.last(1)[2], self.m.order_truth)

    def test_m3_final_report_matches_truth(self):
        self.assertEqual(self.last(3)[2], self.m.m3_truth())
        self.assertLessEqual(self.last(3)[0], 300.0)

    def test_designated_target_is_shot_from_in_front(self):
        mission = self.m.mission
        self.assertIsNotNone(mission.target_face)
        self.assertEqual(self.m.truth[mission.target_face.key], A.target_kind(self.m.order_truth["target"])[1])
        self.assertEqual(len(self.m.fires), 1)  # 첫 발이 수광부에 닿은 것을 사진으로 보고 그만 쏜다 (2026-11-19 — 전에는 늘 target_shots 번)
        self.assertEqual(mission.target_hits, 1)
        face = mission.target_face
        for _now, pose, (pan, _tilt) in self.m.fires:  # 쏠 때 카메라(레이저)가 그 면을 향해 있었는가
            bearing = math.atan2(face.cy - pose[1], face.cx - pose[0])
            aim = pose[2] - math.radians(pan)
            self.assertLess(abs(math.atan2(math.sin(bearing - aim), math.cos(bearing - aim))), math.radians(3.0))

    def test_enters_the_objective_base_after_it_is_posted(self):
        x0, y0, x1, y1 = self.m.fmap.bases[self.m.objective]
        x, y, _yaw = self.m.robot.pose
        self.assertTrue(x0 - 0.05 <= x <= x1 + 0.05 and y0 <= y <= y1, (self.m.robot.pose, self.m.objective))
        order = [name for _t, name in self.m.tasks]
        self.assertLess(order.index("stage"), order.index("assault"))
        self.assertGreaterEqual(next(t for t, name in self.m.tasks if name == "assault"), 240.0)

    def test_m6_is_sent_in_the_report_phase(self):
        t, _kind, ranking = self.last(6)
        self.assertEqual(sorted(ranking), ["R-1", "R-2", "R-3"])

    def test_m6_order_is_3_2_1_when_nothing_tells_the_three_apart(self):
        """2026-11-20 팀장 결정: 상대 셋을 가를 것이 없으면 호출부호 3 · 2 · 1 순 (전에는 1 · 2 · 3). 읽은 것이 있으면 그것이 먼저다."""
        from supervisor.mission import Mission, MissionParams
        mission = Mission(MissionParams())
        mission.camp = "BLUE"
        self.assertEqual(mission.ranking(), ["R-3", "R-2", "R-1"])
        mission.camp = "RED"
        self.assertEqual(mission.ranking(), ["B-3", "B-2", "B-1"])
        mission.ugvs["B-2"] = {"lives": 4, "ammo": 8, "t": 0.0}   # 목숨 · 탄이 가득 찬 것으로 읽힌 상대가 가장 위험하다
        self.assertEqual(mission.ranking(), ["B-2", "B-3", "B-1"])
        mission.ugvs["B-3"] = {"lives": 1, "ammo": 2, "t": 0.0}
        self.assertEqual(mission.ranking(), ["B-2", "B-1", "B-3"])

    def test_face_mode_also_runs_a_whole_match(self):
        m = SimMatch(mode="face", slot="R-2", pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0).run()
        self.assertEqual([r for r in m.reports if r[1] == 3][-1][2], m.m3_truth())
        self.assertEqual(len(m.fires), 1)  # 명중을 확인하면 한 발로 끝낸다

    def test_robot_that_cannot_finish_its_final_turn_still_scouts_and_shoots(self):
        """실제 로봇에서 본 일: 정찰 자리에 와서 방향을 16도쯤 남기고 더 돌지 못하며 「도착」 도 알리지 않는다.
        그래도 자리에 서 있으면 지금 방향에서 팬으로 찍고, 지정표적도 쏜다 (전에는 그 자리에서 경기가 끝날 때까지 서 있었다)."""
        for mode in ("face", "place"):
            m = SimMatch(mode=mode, slot="R-2", pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0, stall_yaw_deg=16.0).run()
            got = [r for r in m.reports if r[1] == 3][-1][2]
            truth = m.m3_truth()
            hit = sum(1 for sector in truth if got.get(sector) == truth[sector])
            # 면 기준(기본)은 네 구역 중 셋 이상. 자리 기준은 팬을 쓰지 않아 자리마다 방향을 다 맞춰야 하므로(자리마다 3초씩 더 쓴다) 둘 이상으로 본다 —
            # 2026-11-19 에 설 자리를 벽 · 상자 · 기둥에서 떨어진 칸으로 옮기면서 자리 수가 늘어 이 방식은 한 구역을 더 놓친다 (경기에서는 쓰지 않는 방식)
            self.assertGreaterEqual(hit, len(truth) - (1 if mode == "face" else 2), (mode, got, truth, m.mission.log[-8:]))
            self.assertGreaterEqual(len(m.fires), 1, (mode, m.mission.log[-8:]))


    def test_base_entry_picks_its_spot_once(self):
        """위치 값이 흔들려도(기본 잡음 3 cm) 거점에 들어갈 자리를 틱마다 다시 정하지 않는다 — 실제 3차 판에서는 50초에 60번 새 목표가 나갔다."""
        m = SimMatch(mode="face", slot="R-2")
        sent, go = [], m.ports.go

        def spy(x, y, yaw, yaw_free=False, mode="normal", **more):
            if mode == "assault":
                sent.append((round(x, 2), round(y, 2)))
            return go(x, y, yaw, yaw_free, mode, **more)

        m.ports.go = spy
        m.run()
        self.assertTrue(sent)
        self.assertLessEqual(len(set(sent)), 3, sorted(set(sent)))
        self.assertIsNotNone(m.captured_at)  # 목표 거점에 들어간 순간 경기가 끝났다

    def test_tries_the_other_bases_when_the_layout_of_base_kinds_differs(self):
        """2026-11-19 공지: 작전지휘소 · 정비소 · 군수보급소가 놓이는 자리는 바뀔 수 있다. 서버가 말한 목표(지도의 이름으로는 2번 줄)의 센서가
        실제로는 3번 줄 거점에 있는 경기: 지도의 이름대로인 거점에 먼저 들어가고, 경기가 안 끝나면 나머지 거점을 차례로 들어가 5:00 전에 점령한다."""
        for sensor_row in (1, 3):
            m = SimMatch(mode="face", slot="R-2", sensor_row=sensor_row).run()
            self.assertIsNotNone(m.captured_at, (sensor_row, m.entered, m.mission.log[-6:]))
            self.assertLess(m.captured_at, 300.0)
            self.assertEqual(m.entered[0][1], m.objective)  # 먼저 지도의 이름대로
            self.assertEqual(m.entered[-1][1], m.sensor_base)
            own = set(m.fmap.bases_of(m.camp))
            self.assertFalse(own & {value for _t, value in m.entered})  # 우리 진영 거점에는 들어가지 않는다

    def test_enters_the_right_base_first_when_the_map_says_where_each_kind_is(self):
        """perception 이 벽 그림으로 거점 종류의 배치를 읽어 지도에 넣어 준 경기 (perception/base_panels.py): 자리 이름 여섯 개 뒤에 상대 진영의
        보고값 이름 세 개가 온다. 서버가 말한 목표의 센서가 실제로 있는 줄(1번 · 3번)을 지도가 알려 주므로, 다른 거점을 거치지 않고 바로 그 거점에 들어간다."""
        from types import SimpleNamespace

        for sensor_row in (1, 3):
            # 거점 진입(M5)을 보는 시험이다 — 정찰이 3분 안에 끝나도 M6 가 M5 를 대신하지 않게 둔다 (m6_instead_of_m5)
            m = SimMatch(mode="face", slot="R-2", sensor_row=sensor_row, params={"m6_instead_of_m5": False})
            foe = A.opponent(m.camp)
            kind = m.objective.split("_", 1)[1]  # 서버가 말할 목표의 종류 (가상 경기장의 기본 배치로는 2번 줄에 있다)
            order = [k for k in A.BASE_KINDS if k != kind]
            order.insert(sensor_row - 1, kind)  # 벽 그림으로 읽은 배치: 그 종류는 sensor_row 번 줄에 있다

            def region(name, rect, row):
                return SimpleNamespace(name=name, min_x=rect[0], min_y=rect[1], max_x=rect[2], max_y=rect[3], row=row)

            def rect(camp, row):
                return m.fmap.bases[m.fmap.base_in_row(camp, row)]

            bases = [region("%s-%d" % (camp[0], row), rect(camp, row), row) for camp in A.CAMPS for row in (1, 2, 3)]
            bases += [region("%s_%s" % (foe, k), rect(foe, row), row) for row, k in zip((1, 2, 3), order)]
            message = SimpleNamespace(width=m.fmap.width, height=m.fmap.height, box_size=m.fmap.box_size, beam_depth=m.fmap.beam_depth,
                                      boxes=[SimpleNamespace(x=x, y=y) for x, y in m.fmap.boxes], bases=bases, sectors=[])
            told = A.field_map_from_msg(message, m.mission.arena, m.mission.base_kinds)
            self.assertEqual(told.base_rows[m.objective], sensor_row)
            tick = m.mission.tick

            def tick_with_map(f, ports, tick=tick, told=told):
                f.fmap = told  # 노드가 /team/navigation/map 으로 받은 지도를 넘기듯이
                return tick(f, ports)

            m.mission.tick = tick_with_map
            m.run()
            self.assertIsNotNone(m.captured_at, (sensor_row, m.entered, m.mission.log[-6:]))
            self.assertEqual([value for _t, value in m.entered], [m.sensor_base], (sensor_row, m.entered))  # 첫 번에 맞는 거점
            self.assertLess(m.captured_at, 255.0)  # 차례로 들어가 볼 때(둘째 256초 · 셋째 267초)보다 이르다



if __name__ == "__main__":
    for mode in ("place", "face"):
        m = SimMatch(mode=mode, slot="B-1").run()
        print(mode, "흐름:", m.tasks)
        print("  기록:", m.mission.log)
        print("  M3 맞음:", [r for r in m.reports if r[1] == 3][-1][2] == m.m3_truth(), "발사", len(m.fires), "보고 횟수", len(m.reports))
    unittest.main()
