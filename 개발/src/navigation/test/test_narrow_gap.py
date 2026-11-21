"""좁은 틈 3단계 (2026-11-19 밤, M2): 지도에 0.40 m(규정의 최소 이격)보다 좁게 그려진 틈. ROS 없이 돈다.

규정은 상자 ↔ 상자 · 상자 ↔ 벽 사이 40 cm 이상을 보장한다. 지도의 틈이 0.40 m 면 전부터 지나갔지만(test_drive_core GridTest),
드론 지도의 상자가 틈 쪽으로 2 cm 만 더 그려져도 「길 없음」 → 8 s 뒤 BLOCKED 였다. 보통 격자로 길이 전혀 없을 때만 구조물을
narrow_grow_m 만 부풀린 격자로 다시 짠다. 비교 벤치: 임시_테스트_파일/navigation_좁은틈/gap_bench.py
"""
import math
import unittest

import test_drive_core as D
from navigation.drive import executor as ex
from navigation.drive.fieldmap import MapSpec
from navigation.drive.geometry import Pose2D
from navigation.drive.params import DriveParams

X, YC = 3.0, 2.6


def wall(gap, shift=0.0):
    """x = 3.0 줄의 상자 벽, y = 2.6 (+ shift) 둘레에 폭 gap 의 틈 하나."""
    lo, hi = YC + shift - gap / 2.0, YC + shift + gap / 2.0
    boxes = []
    y = lo - 0.2
    while y > -0.2:
        boxes.append((X, round(y, 3)))
        y -= 0.4
    y = hi + 0.2
    while y < 5.2:
        boxes.append((X, round(y, 3)))
        y += 0.4
    return boxes


def real_gap(trace, p, gap=0.40):
    """궤적 내내 차체 사각형과 실제 상자(틈 0.40 m) 사이의 가장 좁은 틈."""
    spec = MapSpec(6.12, 5.0, 0.40, tuple(wall(gap)), "arena")
    return min(ex.body_clearance(spec, pose.x, pose.y, pose.yaw, p.robot_half_length, p.robot_half_width) for _t, pose, _c, _ph, _b in trace)


class NarrowGridTest(unittest.TestCase):
    def test_gap_drawn_narrower_than_40cm_opens_only_in_the_narrowed_grid(self):
        for gap in (0.36, 0.32, 0.30):
            p, spec, grid = D.make_world(boxes=wall(gap))
            self.assertIsNone(grid.plan((1.0, 2.6), (5.0, 2.6)), gap)
            narrow = grid.narrowed(p.narrow_grow_m)
            path = narrow.plan((1.0, 2.6), (5.0, 2.6))
            self.assertIsNotNone(path, gap)
            inside = [q for q in path if abs(q[0] - X) < 0.2]
            self.assertTrue(inside and all(abs(q[1] - YC) < 0.06 for q in inside), "틈 속은 가운데로 곧게")

    def test_closed_wall_stays_closed(self):
        p, spec, grid = D.make_world(boxes=wall(0.20))   # 차체(0.23 m)가 못 지나가는 틈
        self.assertIsNone(grid.narrowed(p.narrow_grow_m).plan((1.0, 2.6), (5.0, 2.6)))

    def test_outside_the_gap_the_path_keeps_the_normal_clearance(self):
        """틈을 나온 뒤 목표가 비스듬히 있어도 상자 모서리를 질러가지 않는다 (보통 격자로 막힌 칸은 건너뛰지 않는다)."""
        p, spec, grid = D.make_world(boxes=wall(0.36))
        narrow = grid.narrowed(p.narrow_grow_m)
        path = [(1.5, 1.2)] + narrow.plan((1.5, 1.2), (4.5, 4.0))
        for a, b in zip(path, path[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 0.11:      # 여러 칸을 편 선분
                self.assertFalse(narrow._segment_costly(a, b), (a, b))

    def test_same_grid_when_not_narrower(self):
        p, spec, grid = D.make_world()
        self.assertIs(grid.narrowed(grid.grow), grid)
        self.assertIs(grid.narrowed(p.narrow_grow_m), grid.narrowed(p.narrow_grow_m))   # 한 번 만들어 둔다


class NarrowDriveTest(unittest.TestCase):
    def drive(self, gap, start=(1.0, 2.6, 0.0), target=(5.0, 2.6), params=None, shift=0.0):
        p, spec, grid = D.make_world(boxes=wall(gap, shift), params=params)
        job = ex.GoalExecutor(D.goal(*target, timeout_s=60.0), p, 0.0)
        robot = D.SimRobot(Pose2D(*start))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        flags = []
        trace = D.run(job, robot, spec, grid, p, max_t=70.0, on_tick=lambda t, r, c: flags.append((job.narrow_path, r.pose.x, c.v)))
        return job, trace, p, flags

    def test_passes_a_gap_the_map_draws_4cm_too_narrow(self):
        job, trace, p, flags = self.drive(0.36)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertGreater(real_gap(trace, p), 0.03)
        self.assertTrue(any(f[0] for f in flags))
        in_gap = [v for narrow, x, v in flags if narrow and abs(x - X) < 0.25]
        self.assertTrue(in_gap and max(in_gap) <= p.narrow_v + 1e-6, "틈 속은 narrow_v 로")
        far = [v for narrow, x, v in flags if x < 2.0]
        self.assertGreater(max(far), 0.3, "틈에서 먼 곳은 보통 속도로")

    def test_passes_diagonally_without_clipping_the_real_corner(self):
        for start, target in (((1.5, 1.2, 0.0), (4.5, 4.0)), ((1.5, 4.0, 0.0), (4.5, 1.0))):
            job, trace, p, _flags = self.drive(0.32, start=start, target=target)
            self.assertEqual(job.outcome, ex.REACHED, job.detail)
            self.assertGreater(real_gap(trace, p), 0.0)

    def test_one_side_drawn_10cm_into_the_gap(self):
        job, trace, p, _flags = self.drive(0.30, shift=0.05)   # 아래 상자만 10 cm 올려 그림: 지도 틈 0.30, 가운데가 5 cm 위
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertGreater(real_gap(trace, p), 0.0)

    def test_not_used_when_the_normal_grid_has_a_path(self):
        job, _trace, _p, flags = self.drive(0.40)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertFalse(any(f[0] for f in flags))

    def test_switched_off_is_the_old_behaviour(self):
        params = DriveParams()
        params.narrow_grow_m = 0.0
        job, _trace, _p, _flags = self.drive(0.36, params=params)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)

    def test_unmapped_object_in_the_narrow_gap_still_blocks(self):
        """틈 속에 지도에 없는 물체(다른 UGV)가 있으면 밀고 들어가지 않는다 — 전처럼 앞에서 서고 BLOCKED."""
        p, spec, grid = D.make_world(boxes=wall(0.36))
        job = ex.GoalExecutor(D.goal(5.0, 2.6, timeout_s=40.0), p, 0.0)
        robot = D.SimRobot(Pose2D(1.0, 2.6, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        disc = (3.0, 2.6, 0.10)
        trace = D.run(job, robot, spec, grid, p, discs=[disc], max_t=45.0)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertGreater(min((disc[0] - disc[2]) - (pose.x + p.robot_half_length) for _t, pose, _c, _ph, _b in trace), 0.05)


if __name__ == '__main__':
    unittest.main()
