import os
import tempfile
import time
import unittest

from sim_world import SimMatch
from supervisor import checkpoint
from supervisor.mission import Facts, Mission, MissionParams
from supervisor.scout import ScoutParams


class CheckpointTests(unittest.TestCase):
    def test_save_load_and_age_limit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            self.assertIsNone(checkpoint.load(path))
            self.assertTrue(checkpoint.save(path, {"clock": 12.5}))
            self.assertEqual(checkpoint.load(path)["clock"], 12.5)
            self.assertIsNone(checkpoint.load(path, max_age_s=-1.0))
            checkpoint.clear(path)
            self.assertIsNone(checkpoint.load(path))

    def test_restarted_mission_keeps_faces_and_does_not_report_empty_m3(self):
        first = SimMatch(mode="place", slot="B-1", pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0).run(running_s=120.0, report_s=0.0)
        saved = first.mission.snapshot()
        before = first.mission.book.counts()
        self.assertTrue(any(before.values()))

        again = Mission(MissionParams(scout=ScoutParams(mode="place")), arena=first.mission.arena, base_kinds=first.mission.base_kinds, cam=first.mission.cam)
        again.restore(saved)
        self.assertEqual(again.designated, first.mission.designated)

        class Quiet:
            def __init__(self):
                self.reports = []

            def report(self, kind, payload):
                self.reports.append((kind, payload))
                return True

            def __getattr__(self, name):
                return lambda *a, **k: True

        ports = Quiet()
        now = time.monotonic()
        # 지도가 오기 전: M3 를 보내지 않는다 (빈 것으로 덮어쓰지 않는다)
        again.tick(Facts(now=now, t=121.0, phase="running", control=True, slot="B-1", pose=first.said()), ports)
        self.assertFalse([r for r in ports.reports if r[0] == 3])
        # 지도가 오면 면 기록이 되살아나고, 그 뒤의 M3 는 앞에서 적은 것을 담고 있다
        again.tick(Facts(now=now + 0.1, t=121.1, phase="running", control=True, slot="B-1", pose=first.said(), boxes=list(first.fmap.boxes)), ports)
        self.assertEqual(again.book.counts(), before)
        again.tick(Facts(now=now + 30.0, t=151.0, phase="running", control=True, slot="B-1", pose=first.said(), boxes=list(first.fmap.boxes)), ports)
        sent = [r for r in ports.reports if r[0] == 3]
        self.assertTrue(sent)
        for sector, objs in before.items():
            for kind, n in objs.items():
                self.assertGreaterEqual(sent[-1][1][sector].get(kind, 0), n)


if __name__ == "__main__":
    unittest.main()
