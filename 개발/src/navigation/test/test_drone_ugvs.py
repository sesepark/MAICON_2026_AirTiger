"""드론이 본 UGV 자리(/team/perception/ugvs)를 주행에 쓰는 것 시험 (drive/ugvs.py · obstacles · executor). ROS 없이 돈다.

  1) 막은 것이 서 있는 UGV 인지   3) 헛돎 · 곁에 선 UGV   6) 기억 칸 정리   8) 지도 고치기에서 UGV 점 빼기
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_drive_core import DT, SimRobot, goal, lidar_scan, make_world, scan_of_discs  # noqa: E402

from navigation.drive import executor as ex  # noqa: E402
from navigation.drive import pose_tracker as loc  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker, scan_to_body  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.ugvs import UgvTracks, split_self, without_points_near  # noqa: E402


def feed(tracks, t0, t1, others, me=(1.0, 1.0), step=1.0, local=None):
    """t0 ~ t1 동안 step 마다 사진 한 장: 우리 차 me 와 others(시각 → [(x, y)] 함수 또는 목록)."""
    t = t0
    while t <= t1 + 1e-9:
        pts = others(t) if callable(others) else list(others)
        tracks.on_frame(t, [me] + pts, [me], local=local(t) if local else me, pose=me)
        t += step


class SplitSelfTest(unittest.TestCase):
    def test_picks_the_one_on_our_recent_path(self):
        path = [(1.0 + 0.1 * k, 2.0) for k in range(6)]   # 1.5 s 동안 0.5 m 왔다
        me, others = split_self([(3.0, 3.0), (1.05, 2.03), (1.9, 2.0)], path, 0.40)
        self.assertEqual(me, (1.05, 2.03))                 # 지금 자리(1.5, 2.0)에서는 0.45 m — 지나온 길로 견줘야 잡힌다
        self.assertEqual(others, [(3.0, 3.0), (1.9, 2.0)])

    def test_none_when_nothing_is_near(self):
        me, others = split_self([(3.0, 3.0)], [(1.0, 2.0)], 0.40)
        self.assertIsNone(me)
        self.assertEqual(others, [(3.0, 3.0)])


class TracksTest(unittest.TestCase):
    def test_parked_and_moving(self):
        p = DriveParams()
        tracks = UgvTracks(p)
        feed(tracks, 0.0, 6.0, lambda t: [(3.0, 2.0), (4.0 + 0.3 * t, 3.5)])   # 하나는 서 있고 하나는 0.3 m/s 로 간다
        parked = tracks.parked(6.2)
        self.assertEqual(len(parked), 1)
        self.assertAlmostEqual(parked[0].x, 3.0)
        self.assertGreaterEqual(parked[0].still_s(), p.ugv_parked_s)
        self.assertEqual(len(tracks.fresh(6.2)), 2)
        self.assertIsNotNone(tracks.near(6.2, 3.1, 2.0, 0.35))
        self.assertIsNone(tracks.near(6.2, 3.6, 2.0, 0.35))

    def test_jitter_does_not_break_parked(self):
        p = DriveParams()
        tracks = UgvTracks(p)
        feed(tracks, 0.0, 6.0, lambda t: [(3.0 + 0.06 * math.sin(7 * t), 2.0 + 0.06 * math.cos(5 * t))])   # 드론 오차만큼 흔들린다
        self.assertEqual(len(tracks.parked(6.1)), 1)

    def test_forgets_and_needs_two_sightings_and_a_live_feed(self):
        p = DriveParams()
        tracks = UgvTracks(p)
        tracks.on_frame(0.0, [(1.0, 1.0), (3.0, 2.0)], [(1.0, 1.0)])
        self.assertEqual(tracks.fresh(0.1), [])                       # 한 번 본 것은 아직 믿지 않는다 (헛것)
        feed(tracks, 1.0, 3.0, [(3.0, 2.0)])
        self.assertEqual(len(tracks.fresh(3.1)), 1)
        feed(tracks, 4.0, 8.0, [])                                     # 그 뒤로 안 보인다
        self.assertEqual(tracks.fresh(8.1), [])
        self.assertFalse(tracks.alive(8.0 + p.ugv_feed_s + 0.5))       # 사진이 끊겼다
        self.assertEqual(tracks.fresh(8.0 + p.ugv_feed_s + 0.5), [])

    def test_slow_feed_after_our_facilities_are_destroyed(self):
        """우리 진영 시설물이 파괴되면 사진이 2 s · 5 s 간격으로 온다 — 서 있는 UGV 는 그래도 이어서 알고, 끊기면 서 있던 것만 붙든다."""
        p = DriveParams()
        tracks = UgvTracks(p)
        feed(tracks, 0.0, 5.0, [(3.0, 2.0)])                 # 1 s 간격
        feed(tracks, 10.0, 30.0, [(3.02, 2.01)], step=5.0)   # 둘 파괴 → 5 s 간격
        self.assertAlmostEqual(tracks.interval, 5.0)
        self.assertTrue(tracks.alive(34.0))                  # 다음 사진을 기다리는 동안에도 쓴다
        parked = tracks.parked(34.0)
        self.assertEqual(len(parked), 1)
        self.assertGreater(parked[0].still_s(), 25.0)
        self.assertIsNone(tracks.slip(34.0))                 # 뜸한 사진으로는 헛돎을 보지 않는다
        self.assertFalse(tracks.alive(30.0 + 12.5))          # 셋 파괴(403) → 더 안 온다
        self.assertEqual(len(tracks.fresh(30.0 + 12.5)), 1)  # 서 있던 UGV 는 붙든다 (ugv_hold_s)
        self.assertTrue(tracks.holding(30.0 + 12.5))
        self.assertIsNone(tracks.slip(30.0 + 12.5))
        feed(tracks, 50.0, 53.0, [(3.0, 2.0)])               # 다시 1 s 로 오면 간격도 따라 준다
        self.assertFalse(tracks.holding(53.5))
        self.assertGreater(tracks.parked(53.5)[0].still_s(), 45.0)   # 같은 UGV 로 이어진다
        self.assertAlmostEqual(tracks.interval, 1.0)

    def test_frame_without_our_pose_is_dropped(self):
        tracks = UgvTracks(DriveParams())
        self.assertFalse(tracks.on_frame(1.0, [(3.0, 2.0)], []))
        self.assertEqual(tracks.skipped, 1)
        self.assertFalse(tracks.alive(1.1))

    def test_parked_ugv_stays_trusted_when_we_are_not_found(self):
        """우리 차를 못 고른 사진(자세 오차 · 우리가 화면 밖)에서도, 이미 믿던 · 서 있는 UGV 가 그 자리에서 다시 보이면 그대로 믿는다 (2026-11-20).
        전에는 우리 곁(0.80 m)이면 한 장 동안 뺐다 → 그 곁의 기억 칸을 지우고 길 고르기 자리도 빠졌다."""
        p = DriveParams()
        tracks = UgvTracks(p)
        me, ugv = (2.0, 2.5), (2.6, 2.5)
        feed(tracks, 0.0, 11.0, [ugv], me=me)
        tracks.on_frame(12.0, [ugv], [me])                      # 우리 차가 안 잡힌 사진
        self.assertEqual([round(u.still_s()) for u in tracks.fresh(12.0)], [12])
        tracks.on_frame(13.0, [(2.42, 2.5), ugv], [me])         # 우리 차가 0.42 m 밀려 잡혔다 (우리 차로 못 고른다 · 그 UGV 에서 0.18 m)
        self.assertEqual([(u.x, u.y, round(u.still_s())) for u in tracks.fresh(13.0)], [(2.6, 2.5, 13)])
        tracks.on_frame(14.0, [(2.42, 2.5)], [me])              # 그 UGV 는 안 잡히고 밀린 우리 차만 — 그것이 그 UGV 를 옮기지 못한다
        self.assertEqual([(u.x, u.y) for u in tracks.fresh(14.0)], [(2.6, 2.5)])
        tracks.on_frame(15.0, [me, ugv], [me])                  # 다시 제대로 — 서 있던 시간이 이어진다
        self.assertEqual([(u.x, u.y, round(u.still_s())) for u in tracks.fresh(15.0)], [(2.6, 2.5, 15)])

    def test_parked_ugv_missed_by_the_drone_keeps_its_parked_time(self):
        """2026-11-20 00:00 실차: 76 s 서 있던 UGV 를 드론이 50 s 놓쳤다가 그 자리에서 다시 잡았다 — 전에는 0 s 부터 다시 세었다."""
        p = DriveParams()
        tracks = UgvTracks(p)
        me, ugv = (1.0, 1.0), (3.0, 2.6)
        feed(tracks, 0.0, 60.0, [ugv], me=me)
        feed(tracks, 61.0, 110.0, [], me=me)                    # 50 s 동안 안 잡혔다
        self.assertEqual(tracks.fresh(110.0), [])               # 그동안은 지금 보이는 궤적이 아니다 (길 고르기 · 막은 것 판단에 안 쓴다)
        self.assertEqual(tracks.lost_parked(110.0), [ugv])
        tracks.on_frame(111.0, [me, (3.04, 2.63)], [me])        # 그 자리(0.05 m)에 다시 보였다
        back = tracks.fresh(111.0)
        self.assertEqual(len(back), 1)
        self.assertGreater(back[0].still_s(), 100.0)
        self.assertEqual(tracks.lost_parked(111.0), [])

    def test_lost_parked_ugv_is_not_revived_elsewhere_or_too_late(self):
        p = DriveParams()
        tracks = UgvTracks(p)
        me, ugv = (1.0, 1.0), (3.0, 2.6)
        feed(tracks, 0.0, 20.0, [ugv], me=me)
        feed(tracks, 21.0, 30.0, [], me=me)
        tracks.on_frame(31.0, [me, (3.5, 2.6)], [me])           # 0.5 m 떨어진 데 보인 것은 다른 궤적이다 (0 s 부터)
        tracks.on_frame(32.0, [me, (3.5, 2.6)], [me])
        self.assertEqual([round(u.still_s()) for u in tracks.fresh(32.0)], [1])
        self.assertEqual(tracks.lost_parked(32.0), [ugv])
        feed(tracks, 33.0, 33.0 + p.ugv_lost_keep_s + 10.0, [], me=me)
        self.assertEqual(tracks.lost_parked(tracks.stamp), [])  # ugv_lost_keep_s 가 지나면 잊는다

    def test_ugv_seen_moving_away_is_not_kept_as_lost(self):
        p = DriveParams()
        tracks = UgvTracks(p)
        me = (1.0, 1.0)
        feed(tracks, 0.0, 10.0, [(3.0, 2.6)], me=me)
        feed(tracks, 11.0, 13.0, lambda t: [(3.0 + 0.4 * (t - 10.0), 2.6)], me=me)   # 떠나는 것이 보였다
        feed(tracks, 14.0, 25.0, [], me=me)
        self.assertEqual(tracks.lost_parked(25.0), [])

    def test_parked_position_is_the_mean_of_the_frames(self):
        """서 있는 UGV 의 자리는 그 자리에서 본 사진들의 평균이다 (2026-11-20 — 전에는 마지막 한 장, 길 고르기에는 첫 장의 값)."""
        p = DriveParams()
        tracks = UgvTracks(p)
        me = (1.0, 1.0)
        seen = [(3.06, 2.55), (3.00, 2.60), (2.97, 2.62), (3.03, 2.58), (2.94, 2.65)]   # 첫 장이 가장 치우쳤다
        for k, q in enumerate(seen):
            tracks.on_frame(float(k), [me, q], [me])
        u = tracks.fresh(4.0)[0]
        self.assertAlmostEqual(u.x, 3.0, places=6)
        self.assertAlmostEqual(u.y, 2.6, places=6)
        self.assertEqual(u.still_s(), 4.0)
        tracks.on_frame(5.0, [me, (3.5, 2.6)], [me])        # 옮겨 갔다 → 그 자리부터 다시
        u = tracks.fresh(5.0)[0]
        self.assertEqual((u.x, u.y, u.still_s()), (3.5, 2.6, 0.0))

    def test_a_follower_on_our_path_is_not_taken_for_us(self):
        """바로 뒤를 따라오는 UGV 는 우리가 지나온 길 위에 있다 — 길까지의 거리만으로는 우리 차와 구별이 안 된다. 이미 궤적이 있는 UGV 의 검출은 뺀 뒤 고른다."""
        p = DriveParams()
        tracks = UgvTracks(p)
        for k in range(9):                                   # 0.4 m/s 로 가는 우리 차, 사진은 0.5 s 늦다 (0.2 m 뒤로 보인다)
            x = 1.0 + 0.4 * k
            trail = [(x - 0.4 * j / 8.0, 2.0) for j in range(13)]
            gap = max(0.35, 0.85 - 0.10 * k)                 # 따라오는 UGV: 0.85 m 뒤에서 사진마다 0.10 m 씩 붙어 0.35 m 뒤까지
            mine = (x - 0.2, 2.06)                           # 우리 차의 드론 자리는 길에서 6 cm 벗어나 보인다
            follower = (x - 0.2 - gap, 2.0)                  # 따라오는 차는 길 한가운데
            if gap <= 0.36:                                  # 붙은 뒤에는 길까지의 거리만 보면 따라오는 차가 더 가깝다 (전의 규칙은 틀린다)
                self.assertIs(split_self([mine, follower], trail, p.ugv_self_match_m)[0], follower)
            tracks.on_frame(float(k), [mine, follower], trail)
            others = [(round(u.x, 2), round(u.y, 2)) for u in tracks.tracks if u.seen == float(k)]
            self.assertEqual(others, [(round(follower[0], 2), 2.0)], k)   # 궤적으로 들어간 것은 늘 따라오는 차다

    def test_unknown_thing_beside_us_is_still_not_trusted(self):
        """우리 차를 못 고른 사진에서 우리 곁에 새로 보인 것 · 움직이던 것은 전처럼 믿지 않는다 (우리 차일 수 있다)."""
        p = DriveParams()
        tracks = UgvTracks(p)
        me = (2.0, 2.5)
        for k in range(4):
            tracks.on_frame(float(k), [(2.5, 2.5)], [me])       # 우리 차는 한 번도 안 잡혔고 0.5 m 곁에 무엇이 있다
        self.assertEqual(tracks.fresh(3.0), [])
        tracks.on_frame(4.0, [me, (2.5, 2.5)], [me])            # 우리 차를 골랐다 → 그것은 다른 UGV 다
        self.assertEqual(len(tracks.fresh(4.0)), 1)

    def test_our_own_car_never_becomes_a_track(self):
        tracks = UgvTracks(DriveParams())
        for k in range(8):   # 0.4 m/s 로 가는 우리 차 — 드론 자리는 0.3 m 뒤처져 보인다
            x = 1.0 + 0.4 * k
            tracks.on_frame(float(k), [(x - 0.3, 2.0)], [(x - 0.4 * j / 4.0, 2.0) for j in range(5)])
        self.assertEqual(tracks.tracks, [])

    def test_slip(self):
        p = DriveParams()
        stuck = UgvTracks(p)
        feed(stuck, 0.0, 5.0, [], me=(2.0, 2.0), local=lambda t: (0.15 * t, 0.0))   # 바퀴는 0.15 m/s 로 간다는데 드론이 본 우리는 그 자리
        slip = stuck.slip(5.1)
        self.assertIsNotNone(slip)
        self.assertGreaterEqual(slip["asked"], p.ugv_slip_asked_m)
        self.assertIsNone(stuck.slip(5.2))                                           # 같은 구간은 한 번만
        moving = UgvTracks(p)
        t = 0.0
        while t <= 5.0:
            me = (2.0 + 0.15 * t, 2.0)
            moving.on_frame(t, [me], [me], local=(0.15 * t, 0.0), pose=me)
            t += 1.0
        self.assertIsNone(moving.slip(5.1))
        started = UgvTracks(p)                                                       # 서 있다가 마지막 사진 직전에 출발: 드론은 아직 못 봤다
        feed(started, 0.0, 5.0, [], me=(2.0, 2.0), local=lambda t: (0.3 if t >= 5.0 else 0.0, 0.0))
        self.assertIsNone(started.slip(5.1))


class ObstaclesWithDroneTest(unittest.TestCase):
    def tracker(self, p):
        tr = ObstacleTracker(p)
        tr.ugvs = UgvTracks(p)
        return tr

    def test_blocker_kind(self):
        p, spec, grid = make_world(boxes=())
        pose = Pose2D(2.0, 2.5, 0.0)
        tr = self.tracker(p)
        tr.on_scan(6.0, pose, scan_of_discs(pose, [(2.8, 2.5, 0.12)], p), spec, grid)
        self.assertIsNone(tr.blocker_ugv(6.0))                                       # 드론 자리가 없다
        feed(tr.ugvs, 0.0, 6.0, [(2.85, 2.52)], me=(2.0, 2.5))
        kind = tr.blocker_ugv(6.0)
        self.assertEqual(kind[0], "parked")
        self.assertGreaterEqual(kind[1], p.ugv_parked_s)
        other = self.tracker(p)
        other.on_scan(6.0, pose, scan_of_discs(pose, [(2.8, 2.5, 0.12)], p), spec, grid)
        feed(other.ugvs, 0.0, 6.0, lambda t: [(4.0 - 0.2 * t, 2.5)], me=(2.0, 2.5))  # 다가와서 방금 선 UGV
        self.assertEqual(other.blocker_ugv(6.0)[0], "moving")
        empty = self.tracker(p)
        empty.on_scan(6.0, pose, scan_of_discs(pose, [(2.8, 2.5, 0.12)], p), spec, grid)
        feed(empty.ugvs, 0.0, 6.0, [(5.0, 4.0)], me=(2.0, 2.5))                      # UGV 는 딴 데 있다
        self.assertEqual(empty.blocker_ugv(6.0)[0], "none")

    def test_tight_path_opens_at_once_for_a_parked_ugv(self):
        p, spec, grid = make_world(boxes=())
        pose = Pose2D(2.0, 2.5, 0.0)
        plain = ObstacleTracker(p)
        plain.on_scan(6.0, pose, scan_of_discs(pose, [(2.8, 2.5, 0.12)], p), spec, grid)
        self.assertIsNone(plain.tight_cells(6.05, grid))                             # 라이다로는 방금 본 것이라 아직 줄이지 않는다
        tr = self.tracker(p)
        feed(tr.ugvs, 0.0, 6.0, [(2.8, 2.5)], me=(2.0, 2.5))
        tr.on_scan(6.0, pose, scan_of_discs(pose, [(2.8, 2.5, 0.12)], p), spec, grid)
        tight = tr.tight_cells(6.05, grid)
        self.assertIsNotNone(tight)
        self.assertLess(len(tight), len(tr.extra_cells(6.05, grid)))

    def remembered(self, p, spec, grid, after):
        """UGV 를 라이다로 보다가(0 ~ 3 s) 못 보게 된 뒤(기억으로만 남음) 드론 자리가 after(t) 로 이어질 때 6 s 뒤에 남은 칸 수 · 지운 칸 수."""
        pose = Pose2D(2.0, 2.5, 0.0)
        tr = self.tracker(p)
        t = 0.0
        while t < 9.0:
            if abs(t - round(t)) < 1e-6:
                tr.ugvs.on_frame(t, [(2.0, 2.5)] + ([(2.8, 2.5)] if t <= 3.0 else after(t)), [(2.0, 2.5)])
            seen = scan_of_discs(pose, [(2.8, 2.5, 0.12)], p) if t <= 3.0 else []
            tr.on_scan(t, pose, seen, spec, grid)
            if 3.0 < t:
                for seed in tr._seeds.values():   # 라이다가 그 자리를 못 본다 (가림 · 뒤쪽) — _keep_unseen 이 하는 일을 흉내
                    seed[1] = max(seed[1], min(t + p.obstacle_ttl_s, seed[2] + p.obstacle_memory_s))
            tr.extra_cells(t, grid)
            t += 0.1
        return len(tr._seeds), tr.dropped_ugv_cells

    def test_memory_is_dropped_when_the_ugv_left_and_kept_while_it_stays(self):
        p, spec, grid = make_world(boxes=())
        left, dropped = self.remembered(p, spec, grid, lambda t: [(2.8 + 0.5 * (t - 3.0), 2.5)])   # 0.5 m/s 로 떠났다
        self.assertEqual(left, 0)
        self.assertGreater(dropped, 0)
        stays, dropped = self.remembered(p, spec, grid, lambda t: [(2.8, 2.5)])
        self.assertGreater(stays, 0)
        self.assertEqual(dropped, 0)
        # 서 있다가 드론에서 사라졌다 (떠나는 것을 본 적이 없다 — 드론이 놓쳤을 수 있다) → 지우지 않는다 (2026-11-20. 전: ugv_ttl_s 뒤 지웠다)
        unseen, dropped = self.remembered(p, spec, grid, lambda t: [])
        self.assertGreater(unseen, 0)
        self.assertEqual(dropped, 0)
        old = DriveParams()
        old.ugv_lost_keep_s = 0.0   # 끄면 전과 같다
        unseen, dropped = self.remembered(old, spec, grid, lambda t: [])
        self.assertEqual(unseen, 0)
        self.assertGreater(dropped, 0)
        # 떠나다가(움직이는 것이 보인 뒤) 사라졌다 → 떠난 것이다, 지운다
        left, dropped = self.remembered(p, spec, grid, lambda t: [(2.8 + 0.4 * (t - 3.0), 2.5)] if t < 4.5 else [])
        self.assertEqual(left, 0)
        self.assertGreater(dropped, 0)

    def test_old_cells_beside_a_long_parked_ugv_are_not_held_for_ever(self):
        """오래 서 있는 UGV(ugv_plan_parked_s 넘음 — 드론 자리가 이미 길 고르기에 들어 있다) 곁에 한 번 찍힌 점의 칸은 라이다 규칙대로 사라진다.
        2026-11-20 03:28 실차: 거점 안에 선 UGV 곁 0.35 m 에 찍힌 칸이 169 s 넘게 남아(58칸) 그 앞 0.75 m 통로를 0.3 m 로 좁혔다."""
        p, spec, grid = make_world(boxes=())
        pose, ugv = Pose2D(2.0, 2.5, 0.0), (2.8, 2.5)

        def run(until, parked_from=0.0):
            tr = self.tracker(p)
            t = 0.0
            while t < until:
                if abs(t - round(t)) < 1e-6 and t >= parked_from:
                    tr.ugvs.on_frame(t, [(2.0, 2.5), ugv], [(2.0, 2.5)])
                # 0 ~ 3 s: 그 UGV 와, 그 곁 0.30 m 에 잠깐 찍힌 것 (지나가던 것 · 방향이 틀린 스캔의 헛점). 그 뒤로는 아무것도 안 찍힌다
                seen = scan_of_discs(pose, [ugv + (0.12,), (2.8, 2.2, 0.03)], p) if t <= 3.0 else []
                tr.on_scan(t, pose, seen, spec, grid)
                tr.extra_cells(t, grid)
                t += 0.1
            return tr, t

        tr, t = run(9.0)                       # 서 있은 지 10 s 가 안 됐다: 전처럼 그 곁의 칸을 붙든다 (드론 자리가 아직 길 고르기에 없다)
        self.assertGreater(len(tr._seeds), 0)
        self.assertEqual(tr._ugv_plan, ())
        tr, t = run(20.0)                      # 10 s 넘게 서 있다: 드론 자리가 막고, 묵은 칸은 사라진다
        self.assertEqual(tr._ugv_plan, (ugv,))
        self.assertEqual(len(tr._seeds), 0)
        cells = tr.extra_cells(t, grid)
        self.assertIn(grid.cell_of(*ugv), cells)                    # 그 UGV 자리는 여전히 막혀 있다
        self.assertNotIn(grid.cell_of(2.8, 2.5 - 0.55), cells)      # 그 곁 0.55 m 는 비었다 (전: 묵은 칸의 0.35 m 원이 덮었다)

    def test_a_thing_seen_only_briefly_is_not_a_standing_object_later(self):
        """잠깐 찍히고 기억으로만 남은 칸은 시간이 지나도 「서 있는 물체」 가 아니다 — 처음 찍힌 때부터 마지막으로 찍힌 때까지를 본다 (2026-11-20.
        전: 처음 찍힌 지 tight_still_s 면 서 있는 물체 — 지나간 UGV 가 남긴 칸이 목표를 옮기고 좁은 틈 길에서 작게 보였다). 드론 없이 라이다만."""
        p, spec, grid = make_world(boxes=())
        pose, thing = Pose2D(2.0, 2.5, 0.0), (2.8, 2.5)
        cell = grid.cell_of(2.8 - 0.12, 2.5)   # 그 물체의 우리 쪽 면 — 점이 찍히는 칸

        def run(seen_s, until=6.0):
            tr = ObstacleTracker(p)
            t = 0.0
            while t < until:
                tr.on_scan(t, pose, scan_of_discs(pose, [thing + (0.12,)], p) if t <= seen_s else [], spec, grid)
                for seed in tr._seeds.values():   # 그 뒤로는 라이다가 그 자리를 못 본다 (가림 · 뒤쪽) — _keep_unseen 이 하는 일을 흉내
                    seed[1] = max(seed[1], min(t + p.obstacle_ttl_s, seed[2] + p.obstacle_memory_s))
                tr.extra_cells(t, grid)
                t += 0.1
            return tr, t

        tr, t = run(seen_s=0.8)                # 0.8 s 찍히고 사라졌다 (지나갔다) — 기억으로는 남아 있다
        self.assertGreater(len(tr._seeds), 0)
        self.assertIn(cell, tr.extra_cells(t, grid))
        self.assertFalse(tr.standing_cover(t, grid, cell))
        self.assertIsNone(tr.tight_cells(t, grid))
        tr, t = run(seen_s=3.0)                # 3 s 동안 찍혔다 (서 있었다) — 안 보이게 된 뒤에도 서 있는 물체다
        self.assertTrue(tr.standing_cover(t, grid, cell))
        self.assertIsNotNone(tr.tight_cells(t, grid))
        tr, t = run(seen_s=9.0, until=2.5)     # 지금 찍히고 있는 것은 전과 같다: 처음 본 지 2 s 면 서 있는 물체
        self.assertTrue(tr.standing_cover(t, grid, cell))
        tr, t = run(seen_s=9.0, until=1.5)
        self.assertFalse(tr.standing_cover(t, grid, cell))

    def test_memory_survives_a_frame_where_we_are_not_found(self):
        """뒤 0.6 m 에 선 UGV 의 기억 칸: 우리 차를 못 고른 사진 한 장으로 지우지 않는다 (2026-11-20 — 전에는 그 한 장에 모두 지웠다)."""
        p, spec, grid = make_world(boxes=())
        tr = self.tracker(p)
        me, ugv = (2.0, 2.5), (2.6, 2.5)
        ahead, behind = Pose2D(2.0, 2.5, 0.0), Pose2D(2.0, 2.5, math.pi)
        t = 0.0
        while t < 16.0:
            if abs(t - round(t)) < 1e-6:
                tr.ugvs.on_frame(t, [ugv] if round(t) == 12 else [me, ugv], [me])
            pose = ahead if t <= 5.0 else behind                # 5 s 뒤 등을 돌린다 (라이다 뒤쪽은 못 본다)
            tr.on_scan(t, pose, scan_of_discs(pose, [ugv + (0.12,)], p) if t <= 5.0 else [], spec, grid)
            for seed in tr._seeds.values():
                seed[1] = max(seed[1], seed[2] + p.obstacle_memory_s)
            tr.extra_cells(t, grid)
            if t >= 11.5:
                self.assertEqual(tr._ugv_plan, (ugv,), t)       # 길 고르기에 넣은 자리도 그대로다
            t += 0.1
        self.assertGreater(len(tr._seeds), 0)
        self.assertEqual(tr.dropped_ugv_cells, 0)

    def test_map_vehicles_do_not_touch_the_version_unless_they_change(self):
        """FieldMap.vehicles 는 지도 메시지마다(약 10 Hz) 오지만 바뀐 때만 임시 장애물 칸을 다시 만든다 (2026-11-20)."""
        p, spec, grid = make_world(boxes=())
        tr = ObstacleTracker(p)
        tr.on_vehicles(0.0, [])
        version = tr.version
        cells = tr.extra_cells(0.0, grid)
        tr.on_vehicles(0.1, [])
        self.assertEqual(tr.version, version)
        self.assertIs(tr.extra_cells(0.1, grid), cells)
        tr.on_vehicles(0.2, [(3.0, 2.0)])
        self.assertGreater(tr.version, version)
        self.assertIn(grid.cell_of(3.0, 2.0), tr.extra_cells(0.2, grid))
        self.assertNotIn(grid.cell_of(3.0, 2.0), tr.extra_cells(0.2 + p.vehicle_ttl_s + 0.1, grid))   # 오래되면 전처럼 뺀다

    def test_ugv_beside_a_box_is_not_taken_for_a_shifted_box_face(self):
        """상자 윗면에서 0.30 m 안에 선 UGV 의 점은 「어긋난 상자 면」 으로 0.03 m 만 막혔다 — 드론이 그 자리에서 UGV 를 봤으면 UGV 로 (0.18 m) 막는다 (2026-11-20)."""
        p, spec, grid = make_world(boxes=((3.0, 2.0),))        # 윗면 y = 2.2
        pose = Pose2D(2.2, 2.45, 0.0)
        ugv = (3.0, 2.48)                                        # 아랫면이 상자 윗면에서 0.16 m
        small, big = p.displaced_radius + grid.grow, p.obstacle_radius + grid.grow
        plain = self.tracker(p)
        plain.on_scan(0.0, pose, scan_of_discs(pose, [ugv + (0.12,)], p), spec, grid)
        self.assertIn(small, {seed[0] for seed in plain._seeds.values()})          # 드론 자리가 없으면 전과 같다: 상자 쪽 점은 작게
        tr = self.tracker(p)
        feed(tr.ugvs, 0.0, 2.0, [ugv], me=(pose.x, pose.y))
        tr.on_scan(2.0, pose, scan_of_discs(pose, [ugv + (0.12,)], p), spec, grid)
        self.assertEqual({seed[0] for seed in tr._seeds.values()}, {big})           # 그 UGV 의 점은 모두 크게
        self.assertGreater(len(tr.extra_cells(2.0, grid)), len(plain.extra_cells(0.0, grid)))
        # 그 UGV 에서 먼, 진짜로 어긋난 상자 면의 점은 그대로 작게 막는다
        far = self.tracker(p)
        feed(far.ugvs, 0.0, 2.0, [(5.0, 4.0)], me=(pose.x, pose.y))
        far.on_scan(2.0, pose, scan_of_discs(pose, [ugv + (0.12,)], p), spec, grid)
        self.assertIn(small, {seed[0] for seed in far._seeds.values()})

    def test_ugv_beside_a_box_is_not_counted_as_a_corridor_wall(self):
        p, spec, grid = make_world(boxes=((3.0, 2.0), (3.0, 3.3)))   # 아래 상자 윗면 y 2.2 · 위 상자 아랫면 y 3.1 — 그 사이 0.9 m (통로가 아니다)
        pose = Pose2D(3.0, 2.75, 0.0)
        points = [(0.0, -0.30), (0.05, -0.30), (-0.05, -0.31)]       # 오른쪽 0.30 m (y 2.45) 의 점들 — 아래 상자 윗면에서 0.25 m
        job = ex.GoalExecutor(goal(5.0, 2.75), p, 0.0)
        plain = self.tracker(p)
        plain.on_scan(0.0, pose, points, spec, grid)
        self.assertTrue(job._corridor(pose, grid, plain, 0.0))      # 그 점들을 벽(어긋난 상자 면)으로 세면 양옆 사이가 0.65 m — 통로
        tr = self.tracker(p)
        feed(tr.ugvs, 0.0, 2.0, [(3.0, 2.38)], me=(pose.x, pose.y))  # 그 점들은 드론이 본 UGV 의 것이다
        tr.on_scan(2.0, pose, points, spec, grid)
        self.assertFalse(job._corridor(pose, grid, tr, 2.0))

    def test_memory_without_a_drone_ugv_is_untouched(self):
        """드론이 그 자리에서 UGV 를 본 적이 없는 칸(지도에 빠진 상자 등)은 건드리지 않는다."""
        p, spec, grid = make_world(boxes=())
        pose = Pose2D(2.0, 2.5, 0.0)
        tr = self.tracker(p)
        t = 0.0
        while t < 9.0:
            if abs(t - round(t)) < 1e-6:
                tr.ugvs.on_frame(t, [(2.0, 2.5), (5.0, 4.0)], [(2.0, 2.5)])
            tr.on_scan(t, pose, scan_of_discs(pose, [(2.8, 2.5, 0.12)], p) if t <= 3.0 else [], spec, grid)
            for seed in tr._seeds.values():
                seed[1] = max(seed[1], seed[2] + p.obstacle_memory_s)
            tr.extra_cells(t, grid)
            t += 0.1
        self.assertGreater(len(tr._seeds), 0)
        self.assertEqual(tr.dropped_ugv_cells, 0)

    def test_parked_ugv_beside_us_becomes_points_for_the_move_off_plan(self):
        p, spec, grid = make_world(boxes=())
        tr = self.tracker(p)
        pose = Pose2D(2.0, 2.5, 0.0)
        feed(tr.ugvs, 0.0, 4.0, [(1.65, 2.5), (4.0, 4.0)], me=(2.0, 2.5))   # 바로 뒤 0.35 m (라이다가 못 보는 쪽) · 먼 것
        pts = tr.ugv_body_points(4.1, pose)
        self.assertEqual(len(pts), 9)
        self.assertTrue(all(x < -0.2 for x, _y in pts))
        self.assertEqual(ObstacleTracker(p).ugv_body_points(4.1, pose), [])


class StallWithDroneTest(unittest.TestCase):
    """서 있는 UGV 가 유일한 길을 막았다 (test_drive_core.StandingBlockerTest 와 같은 장면) — 드론으로 알면 더 일찍 끝낸다."""

    WALL = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]
    DISC = (3.0, 2.6, 0.12)

    def drive(self, drone):
        p, spec, grid = make_world(boxes=self.WALL)
        job = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=40.0), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.6, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        obstacles = ObstacleTracker(p)
        if drone is not None:
            obstacles.ugvs = UgvTracks(p)
        t, stopped = -8.0, None
        while t < 0.0:   # 그 UGV 는 우리가 출발하기 전부터 서 있었다
            if drone:
                obstacles.ugvs.on_frame(t, [(robot.pose.x, robot.pose.y), self.DISC[:2]], [(robot.pose.x, robot.pose.y)])
            t += 1.0
        t, k = 0.0, 0
        while not job.done and t < 45.0:
            if k % 2 == 0:
                obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [self.DISC], p), spec, grid)
            if drone is not None and k % 20 == 0:
                me = (robot.pose.x, robot.pose.y)
                obstacles.ugvs.on_frame(t, [me] + ([self.DISC[:2]] if drone else []), [me])
            cmd = job.step(t, robot.pose, loc.OK, grid, obstacles)
            if stopped is None and job.phase == "waiting_obstacle":
                stopped = t
            robot.apply(cmd)
            t += DT
            k += 1
        return job, t - stopped

    def test_gives_up_sooner_and_says_so(self):
        plain, waited_plain = self.drive(None)
        job, waited = self.drive(True)
        self.assertEqual(plain.outcome, ex.BLOCKED, plain.detail)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("[OBSTACLE]", job.detail)
        self.assertIn("서 있는 UGV", job.detail)
        self.assertNotIn("드론", plain.detail)
        self.assertLess(waited, waited_plain - 0.5)

    def test_drone_seeing_no_ugv_changes_nothing_but_the_note(self):
        plain, waited_plain = self.drive(None)
        job, waited = self.drive(False)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertAlmostEqual(waited, waited_plain, delta=0.11)
        self.assertIn("UGV 안 보임", job.detail)


class PlanAroundParkedUgvTest(unittest.TestCase):
    """오래 서 있는 UGV 는 라이다로 보기 전부터 길 고르기에 넣는다 (ugv_plan_parked_s) — 사용자 결정 2026-11-19."""

    WALL = [(3.0, y) for y in (0.2, 0.6, 1.8)] + [(3.0, y) for y in (3.4, 3.8, 4.2, 4.6)]   # 틈 둘: y 2.0 ~ 3.2 (곧은 길), y 0.8 ~ 1.6 (돌아가는 길)
    UGV = (3.0, 2.6)

    def drive(self, parked_s, lidar=False, params=None):
        p, spec, grid = make_world(boxes=self.WALL, params=params)
        job = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=60.0), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.6, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        obstacles = ObstacleTracker(p)
        obstacles.ugvs = UgvTracks(p)
        t = -float(parked_s)
        while t < 0.0:
            obstacles.ugvs.on_frame(t, [(1.0, 2.6), self.UGV], [(1.0, 2.6)])
            t += 1.0
        t, k, crossing = 0.0, 0, None
        while not job.done and t < 60.0:
            if k % 2 == 0:   # 라이다는 그 UGV 를 못 본다고 치면(lidar=False) 빈 스캔 — 드론만으로 길을 고르는지 본다
                obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [self.UGV + (0.12,)], p) if lidar else [], spec, grid)
            if k % 20 == 0:
                me = (robot.pose.x, robot.pose.y)
                obstacles.ugvs.on_frame(t, [me, self.UGV], [me])
            before = robot.pose.x
            robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
            if crossing is None and before < 3.0 <= robot.pose.x:
                crossing = robot.pose.y
            t += DT
            k += 1
        return job, crossing, obstacles

    def test_long_parked_ugv_is_planned_around_before_lidar_sees_it(self):
        job, crossing, obstacles = self.drive(parked_s=30)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLess(crossing, 1.7)                      # 아래 틈으로 돌아갔다
        self.assertEqual(obstacles._ugv_plan, (self.UGV,))

    def test_briefly_stopped_ugv_is_not_planned_around(self):
        params = DriveParams()
        params.ugv_plan_parked_s = 1000.0                   # 그만큼 오래 서 있지 않았다
        job, crossing, obstacles = self.drive(parked_s=30, params=params)
        self.assertEqual(obstacles._ugv_plan, ())
        self.assertGreater(crossing, 2.3)                   # 곧은 길 그대로 (라이다에 아무것도 없다)

    def test_plan_disc_goes_away_when_the_feed_stops_or_the_ugv_leaves(self):
        p, spec, grid = make_world(boxes=())
        tr = ObstacleTracker(p)
        tr.ugvs = UgvTracks(p)
        pose = Pose2D(1.0, 1.0, 0.0)
        feed(tr.ugvs, 0.0, 12.0, [(3.0, 2.6)])
        tr.on_scan(12.0, pose, [], spec, grid)
        self.assertIn(grid.cell_of(3.0, 2.6), tr.extra_cells(12.0, grid))
        tight = tr.tight_cells(12.0, grid)                   # 좁은 틈 길에서는 더 작게 본다
        self.assertLess(len(tight), len(tr.extra_cells(12.0, grid)))
        feed(tr.ugvs, 13.0, 16.0, lambda t: [(3.0 + 0.4 * (t - 12.0), 2.6)])   # 떠났다
        tr.on_scan(16.0, pose, [], spec, grid)
        self.assertNotIn(grid.cell_of(3.0, 2.6), tr.extra_cells(16.0, grid))
        feed(tr.ugvs, 17.0, 40.0, [(5.0, 4.0)])
        tr.on_scan(40.0, pose, [], spec, grid)
        self.assertIn(grid.cell_of(5.0, 4.0), tr.extra_cells(40.0, grid))
        tr.on_scan(60.0, pose, [], spec, grid)               # 사진이 끊겼다 (시설물 셋 파괴 · 통신) — 서 있던 자리는 붙든다
        self.assertIn(grid.cell_of(5.0, 4.0), tr.extra_cells(60.0, grid))
        gone = 40.0 + p.ugv_hold_s + 1.0                     # 라이다로도 확인하지 못한 채 ugv_hold_s 가 지나면 잊는다
        tr.on_scan(gone, pose, [], spec, grid)
        self.assertEqual(len(tr.extra_cells(gone, grid)), 0)


class HoldAfterFeedLossTest(unittest.TestCase):
    """드론 사진이 끊긴 뒤 (시설물 셋 파괴 → 403): 서 있던 UGV 는 붙들고, 그 뒤로는 라이다가 확인 · 반박한다 (2026-11-20)."""

    UGV = (2.8, 2.5)
    LOST = 12.0   # 마지막 사진

    def world(self, others=None, params=None):
        p, spec, grid = make_world(boxes=(), params=params)
        tr = ObstacleTracker(p)
        tr.ugvs = UgvTracks(p)
        feed(tr.ugvs, 0.0, self.LOST, others or [self.UGV], me=(1.0, 2.5))
        return p, spec, grid, tr

    def scan(self, tr, t, pose, discs, p, spec, grid, turn_rate=0.0):
        ranges, angle_min, inc = lidar_scan(pose, discs, p, spec)
        tr.on_scan(t, pose, scan_to_body(ranges, angle_min, inc, p), spec, grid, (angle_min, inc, np.asarray(ranges)), turn_rate=turn_rate)

    def test_only_parked_ugvs_are_held_and_only_for_hold_s(self):
        p, spec, grid, tr = self.world(lambda t: [self.UGV, (4.0, 1.0 + 0.3 * t)])   # 하나는 서 있고 하나는 간다
        t = self.LOST + 20.0
        self.assertFalse(tr.ugvs.alive(t))
        held = tr.ugvs.fresh(t)
        self.assertEqual([(u.x, u.y) for u in held], [self.UGV])
        self.assertEqual(tr.ugvs.summary(t)["held"], 1)
        self.assertEqual(tr.ugvs.fresh(self.LOST + p.ugv_hold_s + 1.0), [])
        off = DriveParams()
        off.ugv_hold_s = 0.0
        _p, _spec, _grid, plain = self.world(params=off)
        self.assertEqual(plain.ugvs.fresh(t), [])                                     # 끄면 전처럼 모두 잊는다

    def test_lidar_seeing_through_the_spot_drops_it(self):
        p, spec, grid, tr = self.world()
        pose = Pose2D(1.0, 2.5, 0.0)
        t = self.LOST + 20.0
        tr.on_scan(t, pose, [], spec, grid)
        self.assertIn(grid.cell_of(*self.UGV), tr.extra_cells(t, grid))               # 길 고르기에 그대로 있다
        for k in range(1, p.ugv_hold_clear_scans):
            self.scan(tr, t + 0.1 * k, pose, [], p, spec, grid)                       # 그 자리가 비어 광선이 벽까지 닿는다
        self.assertEqual(len(tr.ugvs.fresh(t + 0.3)), 1)                              # 아직 세는 중
        self.scan(tr, t + 0.1 * p.ugv_hold_clear_scans, pose, [], p, spec, grid)
        self.assertEqual(tr.ugvs.fresh(t + 0.5), [])
        self.assertEqual(tr.ugvs.refuted, 1)
        self.assertNotIn(grid.cell_of(*self.UGV), tr.extra_cells(t + 0.5, grid))

    def test_lidar_seeing_it_there_keeps_it_past_hold_s(self):
        p, spec, grid, tr = self.world()
        pose = Pose2D(1.0, 2.5, 0.0)
        disc = [self.UGV + (0.12,)]
        t = self.LOST + 5.0
        end = self.LOST + p.ugv_hold_s + 30.0
        while t < end:
            self.scan(tr, t, pose, disc, p, spec, grid)
            t += 0.5
        held = tr.ugvs.fresh(end)
        self.assertEqual(len(held), 1)
        self.assertGreater(held[0].still_s(), p.ugv_hold_s)                           # 서 있던 시간도 이어 센다
        pose = Pose2D(2.0, 2.5, 0.0)
        self.scan(tr, end, pose, disc, p, spec, grid)
        self.assertEqual(tr.blocker_ugv(end)[0], "parked")                            # 막은 것 판단도 그대로 듣는다
        self.assertEqual(tr.ugvs.refuted, 0)

    def test_unseen_spot_is_neither_confirmed_nor_dropped(self):
        p, spec, grid, tr = self.world()
        t = self.LOST + 20.0
        behind = Pose2D(3.6, 2.5, 0.0)                                                # 그 UGV 는 바로 뒤 — 라이다가 못 보는 90°
        hidden = Pose2D(1.0, 2.5, 0.0)
        for k in range(10):
            self.scan(tr, t + 0.1 * k, behind, [], p, spec, grid)
        for k in range(10, 20):
            self.scan(tr, t + 0.1 * k, hidden, [(1.9, 2.5, 0.20)], p, spec, grid)     # 더 가까운 것에 가렸다
        self.assertEqual(len(tr.ugvs.fresh(t + 2.0)), 1)
        self.assertEqual(tr.ugvs.refuted, 0)
        self.assertIsNone(tr.ugvs.fresh(t + 2.0)[0].lidar)
        self.assertEqual(len(tr.ugv_body_points(t + 2.0, Pose2D(3.2, 2.5, 0.0))), 9)  # 곁에 선 UGV 로도 그대로 쓴다

    def test_scans_while_turning_fast_do_not_count(self):
        p, spec, grid, tr = self.world()
        pose = Pose2D(1.0, 2.5, 0.0)
        t = self.LOST + 20.0
        for k in range(10):
            self.scan(tr, t + 0.1 * k, pose, [], p, spec, grid, turn_rate=1.0)
        self.assertEqual(len(tr.ugvs.fresh(t + 1.0)), 1)

    def test_nothing_changes_while_the_feed_is_alive(self):
        p, spec, grid, tr = self.world()
        pose = Pose2D(1.0, 2.5, 0.0)
        for k in range(10):                                                           # 사진이 오는 동안에는 라이다가 비었다 해도 건드리지 않는다
            self.scan(tr, self.LOST + 0.1 * k, pose, [], p, spec, grid)
        self.assertEqual(len(tr.ugvs.fresh(self.LOST + 1.0)), 1)
        self.assertEqual(tr.ugvs.refuted, 0)
        self.assertEqual(tr.blocker_ugv(self.LOST + 1.0), None)                       # 막는 점이 없다

    def test_drives_around_a_ugv_that_was_parked_when_the_feed_died(self):
        """PlanAroundParkedUgvTest 와 같은 장면인데 출발 전에 사진이 끊겼다 — 라이다로 보기 전에도 아래 틈으로 돌아간다."""
        wall = PlanAroundParkedUgvTest.WALL
        ugv = PlanAroundParkedUgvTest.UGV

        def drive(params=None):
            p, spec, grid = make_world(boxes=wall, params=params)
            job = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=60.0), p, 0.0)
            robot = SimRobot(Pose2D(1.0, 2.6, 0.0))
            self.assertTrue(job.validate(spec, grid, robot.pose))
            obstacles = ObstacleTracker(p)
            obstacles.ugvs = UgvTracks(p)
            feed(obstacles.ugvs, -50.0, -20.0, [ugv], me=(1.0, 2.6))                  # 20 s 전에 끊겼다
            t, k, crossing = 0.0, 0, None
            while not job.done and t < 60.0:
                if k % 2 == 0:
                    obstacles.on_scan(t, robot.pose, [], spec, grid)                  # 라이다는 그 UGV 를 못 본다고 친다
                before = robot.pose.x
                robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
                if crossing is None and before < 3.0 <= robot.pose.x:
                    crossing = robot.pose.y
                t += DT
                k += 1
            return job, crossing

        job, crossing = drive()
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLess(crossing, 1.7)
        off = DriveParams()
        off.ugv_hold_s = 0.0
        job, crossing = drive(off)
        self.assertGreater(crossing, 2.3)                                             # 끄면 전처럼 곧은 길 (그 UGV 를 잊었다)


class MaskTest(unittest.TestCase):
    def test_points_on_a_ugv_are_removed(self):
        pose = (2.0, 2.0, math.pi / 2)   # 북쪽을 본다
        body = np.array([(1.0, 0.0), (1.0, 0.1), (0.0, -1.5), (2.0, 0.0)])   # 경기장: (2, 3) · (1.9, 3) · (3.5, 2) · (2, 4)
        kept, dropped = without_points_near(body, pose, [(2.0, 3.05)], 0.35)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(kept), 2)
        kept, dropped = without_points_near(body, pose, [], 0.35)
        self.assertEqual((len(kept), dropped), (4, 0))


if __name__ == "__main__":
    unittest.main()
