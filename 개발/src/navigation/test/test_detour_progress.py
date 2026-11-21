"""돌아가는 길을 가는 동안은 NO_PROGRESS 로 끝내지 않는다 (params.no_progress_detour_m, 2026-11-20). ROS 없이 돈다.

2026-11-20 00:00 실차 21번 목표: 2.0 m 길이 다른 UGV 에 막혀 4.6 m 로 다시 짰고, 그 길을 3.1 m 간 8.0 s 에 BLOCKED [NO_PROGRESS] 로 끝났다
(남은 길이 1.96 m — 0.01 m 만 더 갔으면 진행이었다). 돌아가는 길의 평균 빠르기는 0.37 m/s 였다 (서서 도는 시간 포함).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_drive_core import DT, SimRobot, goal, make_world, scan_of_discs  # noqa: E402

from navigation.drive import executor as ex  # noqa: E402
from navigation.drive import pose_tracker as loc  # noqa: E402
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402

# x = 3.0 의 상자 벽에 틈 둘: y 2.4 ~ 2.8 (곧은 길) 과 아래 벽 쪽 y 0.2 ~ 0.6 (돌아가는 길).
# 곧은 길의 틈 앞은 양옆이 상자 줄로 막힌 골목이다 (x 1.8 ~ 3.0) — 돌아가려면 먼저 목표에서 멀어지는 쪽(서쪽)으로 나와야 한다
WALL = [(3.0, y) for y in (1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]
ALLEY = [(x, y) for x in (2.0, 2.4, 2.8) for y in (1.9, 3.3)]
BLOCKER = (3.0, 2.6, 0.14)   # 곧은 길의 틈을 막고 선 것 — 골목 안 x 2.3 까지 가서야 보인다


def drive(params):
    p, spec, grid = make_world(boxes=WALL + ALLEY, params=params)
    job = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=60.0), p, 0.0)
    robot = SimRobot(Pose2D(0.6, 2.6, 0.0))
    assert job.validate(spec, grid, robot.pose)
    obstacles = ObstacleTracker(p)
    t, k, seen, longest = 0.0, 0, False, 0.0
    while not job.done and t < 60.0:
        seen = seen or robot.pose.x > 2.3
        if k % 2 == 0:
            obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [BLOCKER], p) if seen else [], spec, grid)
        robot.apply(job.step(t, robot.pose, loc.OK, grid, obstacles))
        longest = max(longest, job.remaining or 0.0)
        t += DT
        k += 1
    return job, t, longest


class DetourProgressTest(unittest.TestCase):
    def test_long_detour_is_driven_to_the_end(self):
        job, t, longest = drive(DriveParams(max_v=0.30))
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertGreater(longest, 6.0)        # 돌아가는 길은 6 m 가 넘었다 (막히기 전 남은 길이 2.7 m)

    def test_without_it_the_detour_is_cut_short(self):
        """끄면(no_progress_detour_m 0) 전과 같다: 돌아가는 길을 가는 중에 NO_PROGRESS."""
        job, t, longest = drive(DriveParams(max_v=0.30, no_progress_detour_m=0.0))
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("[%s]" % ex.CODE_NO_PROGRESS, job.detail)

    def test_routes_flipping_take_the_extra_time_back(self):
        """길이 이쪽 저쪽으로 번갈아 바뀌는 때(11:19 판): 두 번째로 길어지면 돌아가는 길로 쳐 준 시간을 거두고 더 주지 않는다."""
        p, spec, grid = make_world()
        job = ex.GoalExecutor(goal(4.5, 2.5, timeout_s=60.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, Pose2D(1.0, 2.5, 0.0)))
        job._state = "drive"
        t, x = 0.0, 1.0
        credits = []
        # 0.5 s 마다: 3.6 m 길을 따라 가다가 → 5.0 m 길로 바뀜(한 번째) → 다시 3.6 m 길 → 다시 5.0 m 길(두 번째). 차체는 내내 0.3 m/s 로 움직인다
        for k in range(40):
            route = (3.6, 5.0, 3.6, 5.0)[min(3, k // 10)]
            remaining = route - 0.05 * (k % 10) if k >= 10 else 3.6 - 0.06 * k   # 처음 길에서 3.06 m 까지 줄여 놓았다 — 그 뒤로는 그보다 짧아지지 않는다
            job.remaining = remaining
            job._watch_progress(t, remaining, 3.5)
            job._credit_motion(t, Pose2D(x, 2.5, 0.0))
            credits.append((job._detour_jumps, round(job._detour_credit, 1)))
            t += 0.5
            x += 0.15
        self.assertEqual(credits[9], (0, 0.0))            # 처음 길: 돌아가는 길이 아니다
        self.assertEqual(credits[19][0], 1)
        self.assertGreater(credits[19][1], 3.0)           # 한 번 길어진 뒤 그 길을 가는 동안은 쳐 준다
        self.assertEqual(credits[29][0], 1)               # 짧은 쪽 길로 돌아와도 (전의 최소보다는 길다) 아직 한 번이다
        self.assertEqual(credits[-1], (2, 0.0))           # 두 번째로 길어진 뒤에는 거두고 더 주지 않는다


if __name__ == "__main__":
    unittest.main()
