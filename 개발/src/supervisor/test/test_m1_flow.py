"""명령문 해석(M1)을 묻고, 늦게 온 LLM 답으로 바꾸는 흐름. 주행 · 사진은 쓰지 않는다 (지도 없이 틱만 돌린다)."""
import unittest

from supervisor.mission import Facts, Mission, MissionParams


class Ports:
    def __init__(self):
        self.asks, self.reports = [], []
        self.answer_ask = True

    def go(self, *a, **k): ...
    def stop(self): ...
    def aim(self, *a): return True
    def fire(self): return True
    def shoot(self, *a): return False
    def look_ugvs(self): return False

    def analyze_order(self, text):
        if self.answer_ask:
            self.asks.append(text)
        return self.answer_ask

    def set_ready(self): return True

    def report(self, kind, payload):
        self.reports.append((kind, dict(payload) if isinstance(payload, dict) else payload))
        return True


QUICK = {"target": "RED_RADAR", "search_sectors": ["CM"], "base": "RED_MAINTENANCE"}
FINAL = {"target": "RED_OUTPOST", "search_sectors": ["CM", "R-S1"], "base": "RED_MAINTENANCE"}


def run(mission, ports, seconds, start=0.0, **facts):
    """0.1초 간격으로 돌린다. 보고는 보낸 다음 틱에 받아들여진 것으로 친다."""
    now = start
    results = []
    while now < start + seconds:
        f = Facts(now=now, t=now, phase="running", control=True, slot="B-1", order_text="명령문", results=results, **facts)
        before = len(ports.reports)
        mission.tick(f, ports)
        results = [(kind, True) for kind, _payload in ports.reports[before:]]
        now += 0.1
    return now


class M1FlowTests(unittest.TestCase):
    def test_asks_once_when_the_answer_is_confirmed(self):
        mission, ports = Mission(MissionParams()), Ports()
        now = run(mission, ports, 1.0)
        self.assertEqual(len(ports.asks), 1)
        run(mission, ports, 30.0, start=now, parsed=QUICK, parsed_pending=False)
        self.assertEqual(len(ports.asks), 1)
        self.assertEqual([r for r in ports.reports if r[0] == 1], [(1, QUICK)])

    def test_keeps_asking_while_only_the_word_rules_answered_and_resends_changed_m1(self):
        p = MissionParams()
        mission, ports = Mission(p), Ports()
        now = run(mission, ports, 20.0, parsed=QUICK, parsed_pending=True)
        self.assertEqual(len(ports.asks), 3)  # 0초 · 9초 · 18초
        self.assertEqual(mission.designated, "RED_RADAR")
        now = run(mission, ports, 20.0, start=now, parsed=FINAL, parsed_pending=False)
        self.assertEqual(len(ports.asks), 3)  # 확인됐으니 더 묻지 않는다
        self.assertEqual(mission.designated, "RED_OUTPOST")
        self.assertEqual([r[1] for r in ports.reports if r[0] == 1], [QUICK, FINAL])
        self.assertTrue(any("바뀜" in text for _t, text in mission.log))

    def test_stops_asking_after_the_limit(self):
        p = MissionParams(m1_reask_s=1.0, m1_max_asks=4)
        mission, ports = Mission(p), Ports()
        run(mission, ports, 30.0, parsed=QUICK, parsed_pending=True)
        self.assertEqual(len(ports.asks), 4)

    def test_retries_at_once_while_the_request_cannot_be_sent(self):
        mission, ports = Mission(MissionParams()), Ports()
        ports.answer_ask = False  # llm_worker 가 아직 안 떴다
        now = run(mission, ports, 3.0)
        ports.answer_ask = True
        run(mission, ports, 0.3, start=now)
        self.assertEqual(len(ports.asks), 1)

    def test_changed_target_drops_the_old_strike(self):
        mission, ports = Mission(MissionParams()), Ports()
        now = run(mission, ports, 1.0, parsed=QUICK, parsed_pending=True)
        mission.target_face, mission.target_shots, mission.target_done = object(), 2, True
        run(mission, ports, 1.0, start=now, parsed=FINAL)
        self.assertIsNone(mission.target_face)
        self.assertEqual((mission.target_shots, mission.target_done), (0, False))


if __name__ == "__main__":
    unittest.main()
