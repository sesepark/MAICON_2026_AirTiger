import math
import unittest

from sim_world import run_scout


class ScoutModesTests(unittest.TestCase):
    """정찰 두 방식을 가상 경기장에서 끝까지 돌린다 (주행 · 인식은 가짜). 실제 로봇 성능을 뜻하지 않는다."""

    def test_place_mode_reads_all_sectors_without_error(self):
        r = run_scout("place", pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0)
        self.assertTrue(r["finished"], r)
        self.assertEqual((r["sectors_ok"], r["faces_wrong"]), (4, 0), r)

    def test_face_mode_reads_all_sectors_without_error(self):
        r = run_scout("face", pose_xy=0.0, pose_yaw_deg=0.0, assumed_hfov=128.0)
        self.assertTrue(r["finished"], r)
        self.assertEqual((r["sectors_ok"], r["faces_wrong"]), (4, 0), r)
        self.assertTrue(any("정찰 계획" in line for line in r["log"][:2]), r["log"][:2])  # 벽 쪽 통로의 면이 있으면 그 줄이 먼저 온다

    def test_both_modes_finish_from_the_red_side_with_noise(self):
        for mode in ("place", "face"):
            r = run_scout(mode, camp="RED", start=(5.9, 2.5, math.pi), seed=3)
            self.assertTrue(r["finished"], (mode, r))
            self.assertGreaterEqual(r["sectors_ok"], 3, (mode, r))


if __name__ == "__main__":
    import collections

    for mode in ("place", "face"):
        for label, kw in (("위치 3 cm · 2°", {}), ("위치 6 cm · 5°", {"pose_xy": 0.06, "pose_yaw_deg": 5.0})):
            res = [run_scout(mode, camp=camp, start=start, seed=s, **kw) for s in range(5) for camp, start in (("BLUE", (0.2, 2.5, 0.0)), ("RED", (5.9, 2.5, math.pi)))]
            ok = collections.Counter(r["sectors_ok"] for r in res)
            print(mode, label, "넷 다 맞음 %d/%d" % (ok[4], len(res)), "시간 %.0f초" % (sum(r["secs"] for r in res) / len(res)), "길 %.1f m" % (sum(r["path_m"] for r in res) / len(res)))
    unittest.main()
