"""주행 핵심 로직(navigation/drive) 시험. ROS 없이 돈다.

간단한 차체 시뮬(명령 → 1차 지연 → 단륜 모델)로 목표 실행 전체를 돌려 본다.
실물의 미끄러짐 · 데드밴드는 흉내 내지 않는다 — 로직(상태 전이 · 결과 코드 · 경로)을 확인하는 시험이다.
"""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navigation.drive import executor as ex  # noqa: E402
from navigation.drive import pose_tracker as loc  # noqa: E402
from navigation.drive.fieldmap import MapSpec, fixed_pillars, from_field_map  # noqa: E402
from navigation.drive.geometry import Pose2D, advance, wrap  # noqa: E402
from navigation.drive.grid import Grid  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker, scan_to_body  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.trail import Trail  # noqa: E402

DT = 0.05


def make_world(boxes=((3.06, 2.5),), width=6.12, height=5.0, params=None):
    p = params or DriveParams()
    spec = MapSpec(width, height, 0.40, tuple(boxes), "arena")
    grid = Grid(spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
    return p, spec, grid


def goal(x, y, yaw=None, mode=ex.MODE_NORMAL, pos_tol=0.10, yaw_tol=0.10, timeout_s=60.0, frame="arena"):
    return ex.GoalSpec(x, y, yaw, mode, pos_tol, yaw_tol, timeout_s, frame, "run-1")


class SimRobot:
    """명령을 tau 로 따라가는 단륜 차체. 실제 자세를 그대로 위치로 준다 (SLAM 이 완벽하다고 본다)."""

    def __init__(self, pose, tau=0.1):
        self.pose = pose
        self.v = 0.0
        self.w = 0.0
        self.tau = tau
        self.stuck = False

    def apply(self, cmd, dt=DT):
        if self.stuck:  # 바퀴가 헛돈다
            self.v = self.w = 0.0
            return
        k = min(1.0, dt / self.tau)
        self.v += (cmd.v - self.v) * k
        self.w += (cmd.w - self.w) * k
        self.pose = advance(self.pose, self.v, self.w, dt)


def scan_of_discs(pose, discs, params, n=360):
    """discs = [(x, y, r)] 지도에 없는 원기둥 → 차체 기준 점 (LiDAR 모양 흉내 없이 원 둘레 점)."""
    body = []
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    for cx, cy, r in discs:
        for k in range(24):
            a = 2 * math.pi * k / 24
            wx, wy = cx + r * math.cos(a), cy + r * math.sin(a)
            dx, dy = wx - pose.x, wy - pose.y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            if params.scan_self_mask_m < math.hypot(bx, by) < params.scan_max_range:
                body.append((bx, by))
    return body


def run(job, robot, spec, grid, params, discs=(), max_t=60.0, level_fn=None, on_tick=None):
    obstacles = ObstacleTracker(params)
    t = 0.0
    trace = []
    while not job.done and t < max_t:
        if discs and int(round(t / DT)) % 2 == 0:  # 10 Hz 스캔
            obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, discs, params), spec, grid)
        level = level_fn(t) if level_fn else loc.OK
        cmd = job.step(t, robot.pose, level, grid, obstacles)
        trace.append((t, robot.pose, cmd, job.phase, job.blocked))
        if on_tick:
            on_tick(t, robot, cmd)
        robot.apply(cmd)
        t += DT
    return trace


def old_blocking(**more):
    """막혔을 때의 전 흐름으로 돌린 파라미터: 물체 0.30 m 앞까지 가서(through_standoff_m 끔) 5 s 기다린다(obstacle_give_up_static_s 끔).
    그 흐름 위의 장치(막힘 알림 · 길이 열리면 이어 감 · 되짚기)를 따로 볼 때 쓴다."""
    params = DriveParams()
    params.through_standoff_m = 0.0
    params.obstacle_give_up_static_s = 0.0
    for name, value in more.items():
        setattr(params, name, value)
    return params


class GridTest(unittest.TestCase):
    def test_plans_around_box_and_keeps_clearance(self):
        p, spec, grid = make_world()
        path = grid.plan((1.0, 2.5), (5.0, 2.5))
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (5.0, 2.5))
        prev = (1.0, 2.5)
        for pt in path:  # 모든 선분이 부풀린 박스를 피한다
            self.assertTrue(grid.segment_free(prev, pt) or prev == (1.0, 2.5))
            prev = pt

    def test_40cm_gap_is_passable_but_blocked_by_extra(self):
        # 박스 두 개 사이 40 cm 틈 (가운데 간격 0.8 m) 만 열린 벽
        boxes = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]
        p, spec, grid = make_world(boxes=boxes)
        path = grid.plan((1.0, 2.6), (5.0, 2.6))
        self.assertIsNotNone(path, "40 cm 틈으로 지나가야 한다")
        extra = set(grid.cells_in_disc(3.0, 2.6, 0.30))
        self.assertIsNone(grid.plan((1.0, 2.6), (5.0, 2.6), extra))

    def test_nearest_free_and_bounds(self):
        p, spec, grid = make_world()
        self.assertFalse(grid.free(3.06, 2.5))
        snapped = grid.nearest_free(3.06, 2.5, max_dist=0.5)
        self.assertIsNotNone(snapped)
        self.assertTrue(grid.free(*snapped))
        self.assertIsNone(grid.nearest_free(3.06, 2.5, max_dist=0.05))


class FieldMapTest(unittest.TestCase):
    def test_from_message_like_object(self):
        point = lambda x, y: SimpleNamespace(x=x, y=y, z=0.0)  # noqa: E731
        pose = lambda x, y, yaw: SimpleNamespace(position=point(x, y), orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))  # noqa: E731
        msg = SimpleNamespace(
            header=SimpleNamespace(frame_id="arena"), width=6.12, height=5.0, box_size=0.4,
            boxes=[point(1.0, 1.0)],
            vehicles=[SimpleNamespace(role=1, pose=pose(0.2, 2.5, 0.0)), SimpleNamespace(role=2, pose=pose(5.9, 2.5, math.pi))],
        )
        update = from_field_map(msg)
        self.assertEqual(update.spec.boxes, ((1.0, 1.0),))
        self.assertEqual(update.others, [(5.9, 2.5)])
        self.assertAlmostEqual(update.self_pose.x, 0.2)
        self.assertAlmostEqual(update.spec.distance_to_structure(1.0, 1.5), 0.3, places=6)
        self.assertEqual(update.spec.distance_to_structure(1.0, 1.0), 0.0)
        self.assertAlmostEqual(update.spec.distance_to_structure(3.0, 0.05), 0.05)


class FixedPillarsTest(unittest.TestCase):
    """벽에 붙은 고정 기둥 10개 (2026-11-19): 지도 메시지에는 없고, 주행이 규칙의 기본 배치를 더해 쓴다."""

    @staticmethod
    def message(boxes=((1.0, 1.0),), width=6.12, height=5.0):
        return SimpleNamespace(header=SimpleNamespace(frame_id="arena"), width=width, height=height, box_size=0.4,
                               boxes=[SimpleNamespace(x=x, y=y, z=0.0) for x, y in boxes], vehicles=[])

    def test_default_layout_only_for_the_official_arena_size(self):
        pillars = fixed_pillars(6.12, 5.0)
        self.assertEqual(len(pillars), 10)
        self.assertIn((0.2, 5.0 / 3.0), [(round(x, 6), y) for x, y in pillars])  # BLUE 벽, 2 · 3번 줄 사이
        self.assertEqual(fixed_pillars(3.0, 2.0), ())  # 시험용 작은 방에는 넣지 않는다

    def test_pillars_join_the_structure_but_not_the_box_list(self):
        update = from_field_map(self.message(), pillars=True)
        spec = update.spec
        self.assertEqual(spec.boxes, ((1.0, 1.0),))  # 임무 상자 목록은 그대로
        self.assertEqual(len(spec.fixed), 10)
        self.assertTrue(spec.inside_box(0.2, 3.33))  # 기둥 안은 목표로 받지 않는다
        self.assertAlmostEqual(spec.distance_to_structure(0.75, 5.0 * 2.0 / 3.0), 0.35, places=6)  # 기둥의 경기장 쪽 면(x = 0.40)까지
        self.assertAlmostEqual(spec.distance_to_structure(3.06, 4.3), 0.3, places=6)  # 위쪽 가운데 기둥의 아랫면(y = 4.60)까지
        grid = Grid(spec)
        self.assertFalse(grid.free(0.50, 3.33))  # 기둥 + 차체 여유(0.17 m) 안
        self.assertTrue(grid.free(0.65, 3.33))
        self.assertEqual(from_field_map(self.message()).spec.fixed, ())  # 따로 켜지 않으면 넣지 않는다

    def test_a_map_box_on_a_pillar_spot_is_not_added_twice(self):
        update = from_field_map(self.message(boxes=((0.2, 5.0 / 3.0), (1.0, 1.0))), pillars=True)
        self.assertEqual(len(update.spec.fixed), 9)

    def test_lidar_points_on_a_pillar_are_explained_by_the_map(self):
        p = DriveParams()
        for pillars, expect_unexplained in ((False, True), (True, False)):
            spec = from_field_map(self.message(), pillars=pillars).spec
            grid = Grid(spec)
            tracker = ObstacleTracker(p)
            pose = Pose2D(1.0, 5.0 * 2.0 / 3.0, math.pi)  # 기둥의 경기장 쪽 면을 0.6 m 앞에서 본다
            body = [(0.6, y) for y in (-0.15, -0.05, 0.05, 0.15)]
            tracker.on_scan(0.0, pose, body, spec, grid)
            self.assertEqual(bool(tracker.unexplained_body), expect_unexplained, pillars)
            self.assertEqual(bool(tracker.extra_cells(0.1, grid)), expect_unexplained, pillars)


class StuckRulesTest(unittest.TestCase):
    """2026-11-19 실차 두 판(연습 경기장 2번)에서 본 「그 자리에 서 있기」 를 줄이는 규칙들."""

    def test_points_beside_a_mapped_box_block_less_than_free_standing_points(self):
        """지도보다 15 cm 어긋난 상자의 면(지도 상자 곁의 점)은 0.20 m 만 막고, 떨어져 있는 점(다른 UGV)은 전처럼 0.35 m 를 막는다."""
        p, spec, grid = make_world(boxes=((3.0, 2.5),))
        tracker = ObstacleTracker(p)
        pose = Pose2D(3.0, 3.35, -math.pi / 2.0)  # 상자의 윗면(y = 2.70)을 0.5 m 앞에서 본다. 실제 면은 y = 2.85 에 있다
        tracker.on_scan(0.0, pose, [(0.5, dx) for dx in (-0.1, 0.0, 0.1)], spec, grid)
        self.assertEqual(len(tracker.unexplained_body), 3)
        cells = tracker.extra_cells(0.1, grid)
        self.assertIn(grid.cell_of(3.0, 3.02), cells)  # 실제 면에서 0.17 m — 차체 여유 안
        self.assertNotIn(grid.cell_of(3.0, 3.12), cells)  # 0.27 m — 예전(0.35 m)이면 막혔다
        far = ObstacleTracker(p)
        far.on_scan(0.0, Pose2D(1.0, 1.0, 0.0), [(0.6, 0.0)], spec, grid)  # 상자 · 벽에서 먼 점
        self.assertIn(grid.cell_of(1.6, 1.3), far.extra_cells(0.1, grid))  # 0.30 m 옆도 막는다

    def test_gives_up_before_the_timeout_when_it_makes_no_progress(self):
        """목표로 가는 길이 없다 (지도의 상자로 둘러싸임): 전에는 시한(60 s)까지 서 있었다. 이제 no_progress_give_up_s 뒤 BLOCKED."""
        ring = [(3.0 + dx, 2.5 + dy) for dx in (-0.4, 0.0, 0.4) for dy in (-0.4, 0.0, 0.4) if (dx, dy) != (0.0, 0.0)]
        p, spec, grid = make_world(boxes=ring)
        job = ex.GoalExecutor(goal(3.0, 2.5, timeout_s=60.0), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertLess(trace[-1][0], p.blocked_report_s + p.no_progress_give_up_s + 1.0)
        self.assertIn("가까워지지 못해", job.detail)

    def test_routes_flipping_back_and_forth_is_not_progress(self):
        """2026-11-19 실차 11:19 판 첫 목표: 지도에 없는 물체 때문에 북쪽(3.6 m) · 남쪽(5.0 m) 우회로가 3 ~ 4 s 마다 번갈아 바뀌었다.
        바뀐 길을 따라 달리면 그 길의 남은 길이는 줄어 전에는 「진행」 으로 셌고, 12 s 포기가 켜지지 않아 41 s 시한을 다 썼다.
        남은 길이의 최소는 다시 짜도 이어 가므로, 그 최소보다 짧아지지 않으면 진행이 아니다 → no_progress_give_up_s 뒤 BLOCKED [NO_PROGRESS]."""
        p, spec, grid = make_world()
        job = ex.GoalExecutor(goal(4.5, 2.5, timeout_s=60.0), p, 0.0)
        pose = Pose2D(1.0, 2.5, 0.0)
        self.assertTrue(job.validate(spec, grid, pose))
        t, direct = 0.0, 3.5
        for k in range(40):  # 1 s 마다: 짝수 초부터 4 s 동안 한 길을 따라 1.2 m 줄었다가, 다른 길로 다시 짬 (곧은 거리는 그대로)
            route = (3.6, 5.0)[(k // 4) % 2]
            remaining = route - 0.3 * (k % 4)
            if k % 4 == 0:
                job._plan(t, pose, grid, set())  # 다른 길로 다시 짬
            job._watch_progress(t, remaining, direct)
            if job.blocked:
                job.step(t, pose, loc.OK, grid, None)  # 포기는 step() 맨 앞에서 본다
            if job.done:
                break
            t += 1.0
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("[%s]" % ex.CODE_NO_PROGRESS, job.detail)
        self.assertLessEqual(t, 4.0 + p.blocked_report_s + p.no_progress_give_up_s)  # 첫 길의 최소(2.7 m)를 찍은 뒤로 진행 없음

    def test_moves_off_when_there_is_no_room_to_turn(self):
        """옆 8 cm 에 지도에 없는 것이 붙어 있고 뒤는 지도의 상자다. 가야 할 곳은 뒤쪽이라 서서 돌아야 하는데 돌면 닿는다.
        전에는 「붙었는데 뒤도 막힘」 으로 8 s 뒤 BLOCKED (실차 2026-11-19 10:50 판: 그 자리에서 85 s).
        이제 비키기 계획(_room_move_plan "obstacle")이 그 물체에서 멀어지는 앞쪽(곧게 · 호)으로 비킨 뒤 돈다 (phase moving_off).
        cfb804c 의 「곧게 조금 나아가기」(creeping)는 이 계획과 겹쳐 통합 때 뺐다."""
        p, spec, grid = make_world(boxes=((2.58, 2.82),))  # 차체 왼쪽 뒤 모서리 바로 뒤의 지도 상자 (뒤로 못 뺀다)
        disc = (3.02, 2.795, 0.10)  # 왼쪽 옆의 지도에 없는 것 (가장 가까운 면이 차체 옆에서 8 cm)
        job = ex.GoalExecutor(goal(2.2, 1.9, timeout_s=40.0), p, 0.0)
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, discs=[disc])
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        phases = {phase for _t, _pose, _cmd, phase, _b in trace}
        self.assertIn("moving_off", phases)
        self.assertNotIn("creeping", phases)
        for _t, pose, *_ in trace:  # 그 물체에 닿지 않는다 (차체 모서리까지 0.17 m)
            self.assertGreater(math.hypot(pose.x - disc[0], pose.y - disc[1]), disc[2] + 0.17 - 0.01)


class PoseTrackerTest(unittest.TestCase):
    def test_wheel_heading_is_default_and_ignores_imu(self):
        p = DriveParams()
        self.assertEqual(p.yaw_source, "wheel")
        tr = loc.PoseTracker(p)
        tr.on_pose(0.0, 0.0, Pose2D(1.0, 1.0, 0.0))
        tr.on_imu(0.0, 0.0)
        tr.on_imu(0.1, 1.0)  # yaw_source=wheel 이면 IMU 는 방향에 안 쓰인다
        self.assertAlmostEqual(tr.estimate().yaw, 0.0, places=9)

    def test_wheel_heading_follows_wheel_difference(self):
        p = DriveParams()
        tr = loc.PoseTracker(p)
        tr.on_pose(0.0, 0.0, Pose2D(1.0, 1.0, 0.0))
        tr.on_wheels(0.0, 5.0, 5.0)              # 누적 거리 원점은 아무 값
        tr.on_wheels(0.1, 5.0 - 0.02, 5.0 + 0.02)  # 오른쪽이 더 감 = 좌회전
        self.assertAlmostEqual(tr.estimate().yaw, 0.04 / p.wheel_base_m, places=9)
        tr.on_wheels(0.2, 5.0 - 0.02 + 0.03, 5.0 + 0.02 + 0.03)  # 같이 직진: 방향 변화 없음
        self.assertAlmostEqual(tr.estimate().yaw, 0.04 / p.wheel_base_m, places=9)
        tr.on_wheels(0.3, 5.0 + 0.01, 5.0 + 0.05 - 0.04)  # 왼쪽이 더 감 = 우회전
        self.assertLess(tr.estimate().yaw, 0.04 / p.wheel_base_m)

    def test_wheel_reset_and_bad_values_are_dropped(self):
        tr = loc.PoseTracker(DriveParams())
        tr.on_pose(0.0, 0.0, Pose2D(0.0, 0.0, 0.0))
        tr.on_wheels(0.0, 1.0, 1.0)
        tr.on_wheels(0.1, 0.0, 0.0)               # 엔코더 초기화가 아닌 1 m 이하 후진: 회전 아님
        tr.on_wheels(0.2, 50.0, 0.0)              # 1 m 넘게 튐: 버림
        tr.on_wheels(0.3, float('nan'), 0.0)
        self.assertAlmostEqual(tr.estimate().yaw, 0.0, places=9)

    def test_imu_heading_can_still_be_selected(self):
        p = DriveParams()
        p.yaw_source = "imu"
        tr = loc.PoseTracker(p)
        tr.on_pose(0.0, 0.0, Pose2D(0.0, 0.0, 0.0))
        tr.on_wheels(0.0, 0.0, 0.0)
        tr.on_wheels(0.1, -0.1, 0.1)               # imu 모드에서는 바퀴를 방향에 쓰지 않는다
        self.assertAlmostEqual(tr.estimate().yaw, 0.0, places=9)
        tr.on_imu(0.2, 0.0)
        tr.on_imu(0.3, 0.4)
        self.assertAlmostEqual(tr.estimate().yaw, 0.4, places=9)

    def test_fills_between_slam_updates_with_odom_and_imu(self):
        p = DriveParams()
        p.yaw_source = "imu"
        tr = loc.PoseTracker(p)
        tr.on_imu(0.0, 0.0)
        tr.on_odom(0.0, 10.0, 10.0)  # /odom 원점은 아무 데나
        tr.on_pose(0.0, 0.0, Pose2D(1.0, 1.0, math.pi / 2))
        tr.on_command(0.2)
        tr.on_odom(0.1, 10.0, 10.1)  # 0.1 m 움직임 (odom 방향과 무관)
        est = tr.estimate()
        self.assertAlmostEqual(est.x, 1.0, places=6)
        self.assertAlmostEqual(est.y, 1.1, places=6)
        tr.on_imu(0.2, 0.5)
        self.assertAlmostEqual(tr.estimate().yaw, math.pi / 2 + 0.5, places=6)

    def test_late_pose_keeps_motion_since_its_stamp(self):
        p = DriveParams()
        tr = loc.PoseTracker(p)
        tr.on_wheels(0.0, 0.0, 0.0)
        tr.on_odom(0.0, 0.0, 0.0)
        tr.on_command(0.3)
        for k in range(1, 11):
            tr.on_odom(k * 0.05, 0.03 * k, 0.0)
        # 0.25 초에 찍힌 위치가 0.5 초에 도착: 그 뒤 0.15 m 움직임이 더해져야 한다
        tr.on_pose(0.5, 0.25, Pose2D(2.0, 0.0, 0.0))
        self.assertAlmostEqual(tr.estimate().x, 2.15, places=6)

    def test_levels_have_hysteresis(self):
        p = DriveParams()
        tr = loc.PoseTracker(p)
        self.assertEqual(tr.update_level(0.0), loc.LOST)
        t = 0.0
        for _ in range(5):
            tr.on_pose(t, t, Pose2D(0, 0, 0))
            tr.update_level(t)
            t += 0.1
        self.assertEqual(tr.level, loc.OK)
        self.assertEqual(tr.update_level(t + 0.35), loc.DEGRADED)
        self.assertEqual(tr.update_level(t + 0.7), loc.HOLD)
        self.assertEqual(tr.update_level(t + 2.5), loc.HOLD)
        self.assertEqual(tr.update_level(t + 2.75), loc.LOST)
        # 다시 오기 시작: 한 번으로는 정상이 아니고 연속 3번이어야 한다
        t += 3.0
        tr.on_pose(t, t, Pose2D(0, 0, 0))
        self.assertEqual(tr.update_level(t), loc.HOLD)
        tr.on_pose(t + 0.1, t + 0.1, Pose2D(0, 0, 0))
        self.assertNotEqual(tr.update_level(t + 0.1), loc.OK)
        tr.on_pose(t + 0.2, t + 0.2, Pose2D(0, 0, 0))
        self.assertEqual(tr.update_level(t + 0.2), loc.OK)

    def test_single_late_pose_only_degrades(self):
        p = DriveParams()
        tr = loc.PoseTracker(p)
        t = 0.0
        for _ in range(5):
            tr.on_pose(t, t, Pose2D(0, 0, 0))
            tr.update_level(t)
            t += 0.1
        t += 0.35  # 한 번 늦음
        self.assertEqual(tr.update_level(t), loc.DEGRADED)
        for _ in range(3):
            tr.on_pose(t, t, Pose2D(0, 0, 0))
            tr.update_level(t)
            t += 0.1
        self.assertEqual(tr.level, loc.OK)


class ObstacleTest(unittest.TestCase):
    def test_map_points_are_explained_and_others_block(self):
        p, spec, grid = make_world()
        obs = ObstacleTracker(p)
        pose = Pose2D(1.0, 2.5, 0.0)
        wall_pt = (-0.95, 0.0)  # 차체 뒤 0.95 m = 세계 x 0.05 (벽에서 5 cm) → 지도로 설명됨
        other = (0.30, 0.0)  # 앞 30 cm 의 물체 (차체 앞에서 17.5 cm)
        obs.on_scan(0.0, pose, [wall_pt, other], spec, grid)
        self.assertEqual(obs.unexplained_body, [other])
        self.assertTrue(obs.blocking_ahead(0.1, 1.0))
        self.assertFalse(obs.blocking_ahead(0.1, -1.0))
        self.assertTrue(obs.extra_cells(0.1, grid))
        self.assertFalse(obs.extra_cells(5.0, grid))  # obstacle_ttl_s 뒤 사라짐

    def test_scan_conversion_uses_lidar_mount(self):
        p = DriveParams()
        # LiDAR 0° 가 차체 왼쪽(+90°) → 각도 -90° 의 점은 차체 앞
        pts = scan_to_body([1.0], -math.pi / 2, 0.01, p)
        self.assertAlmostEqual(pts[0][0], p.lidar_x + 1.0, places=6)
        self.assertAlmostEqual(pts[0][1], 0.0, places=6)
        self.assertEqual(scan_to_body([0.05, float("inf"), float("nan")], 0.0, 0.1, p), [])

    def test_scan_drops_points_on_our_own_body(self):
        """2026-11-19 16:55 실차: 라이다 45° 0.128 m · 135° 0.155 m (차체 뒤쪽 양옆, 차체 사각형 안)이 매 스캔 찍혀 서서 돌지 못했다."""
        p = DriveParams()
        a45, a135 = math.radians(45.0), math.radians(135.0)
        self.assertEqual(scan_to_body([0.128], a45, 0.01, p), [])
        self.assertEqual(scan_to_body([0.155], a135, 0.01, p), [])
        self.assertEqual(len(scan_to_body([0.30], a45, 0.01, p)), 1)  # 몸 밖의 진짜 물체는 남는다
        self.assertEqual(len(scan_to_body([0.20], -math.pi / 2, 0.01, p)), 1)  # 바로 앞 0.24 m
        off = DriveParams(scan_self_margin_m=0.0)
        self.assertEqual(len(scan_to_body([0.128], a45, 0.01, off)), 1)  # 끄면 전과 같다


def lidar_scan(pose, discs, p, spec, n=504, blind=True):
    """실차 라이다 모양의 스캔 (ranges, angle_min, angle_increment): 경기장 벽 · 원기둥(discs = [(x, y, r)])까지의 거리,
    차체 뒤 ±135 ~ 180° 는 NaN (실차 bag 1118 두 개: 드라이버가 늘 가린다)."""
    inc = 2 * math.pi / n
    lx, ly = pose.x + p.lidar_x * math.cos(pose.yaw), pose.y + p.lidar_x * math.sin(pose.yaw)
    ranges = []
    for k in range(n):
        body = math.radians(p.lidar_yaw_deg) + k * inc
        if blind and abs(wrap(body)) > math.radians(135.0):
            ranges.append(float("nan"))
            continue
        dx, dy = math.cos(pose.yaw + body), math.sin(pose.yaw + body)
        hit = min(t for t in ((-lx / dx if dx < 0 else (spec.width - lx) / dx) if abs(dx) > 1e-9 else math.inf,
                              (-ly / dy if dy < 0 else (spec.height - ly) / dy) if abs(dy) > 1e-9 else math.inf))
        for cx, cy, r in discs:  # 광선과 원
            fx, fy = lx - cx, ly - cy
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - r * r)
            if disc >= 0.0 and -b - math.sqrt(disc) > 0.0:
                hit = min(hit, -b - math.sqrt(disc))
        ranges.append(hit)
    return ranges, 0.0, inc


class ObstacleMemoryTest(unittest.TestCase):
    """라이다가 그 자리를 볼 수 없으면 임시 장애물을 잊지 않는다 (2026-11-19 오전 실차: 출발 앞 물체를 돌아가면 뒤로 가 2.5 s 뒤 잊히고,
    그쪽 길이 짧아 보여 되돌아 돌기를 되풀이했다)."""

    def setUp(self):
        self.p, self.spec, self.grid = make_world(boxes=())

    def feed(self, tracker, t, pose, discs, rays=True):
        ranges, angle_min, inc = lidar_scan(pose, discs, self.p, self.spec)
        body = scan_to_body(ranges, angle_min, inc, self.p)
        tracker.on_scan(t, pose, body, self.spec, self.grid, (angle_min, inc, np.asarray(ranges)) if rays else None)

    def blocked(self, tracker, t, x, y):
        return self.grid.cell_of(x, y) in tracker.extra_cells(t, self.grid)

    def test_remembers_an_obstacle_that_went_behind(self):
        tracker = ObstacleTracker(self.p)
        thing = [(2.8, 2.5, 0.10)]
        for k in range(-7, 1):   # 앞 0.7 m 에 0.7 s 동안 본다 (잠깐만 찍힌 칸은 기억하지 않는다 — memory_min_seen_s)
            self.feed(tracker, 0.1 * k, Pose2D(2.0, 2.5, 0.0), thing)
        self.assertTrue(self.blocked(tracker, 0.1, 2.7, 2.5))
        behind = Pose2D(2.0, 2.5, math.pi)  # 돌아서면 그 자리는 차체 뒤 (라이다가 못 보는 쪽) — 물체는 그대로 있다
        for k in range(1, 60):
            self.feed(tracker, 0.1 * k, behind, thing)
        self.assertTrue(self.blocked(tracker, 6.0, 2.7, 2.5))  # 전에는 2.5 s 뒤 잊었다
        summary = tracker.memory_summary(6.0)  # 경기 기록(debug.jsonl state 의 memory)
        self.assertGreater(summary['remembered'], 0)
        self.assertGreater(summary['oldest_s'], self.p.obstacle_ttl_s)
        t = 6.0
        while t < self.p.obstacle_memory_s + self.p.obstacle_ttl_s + 0.5:
            t += 0.1
            self.feed(tracker, t, behind, thing)
        self.assertFalse(self.blocked(tracker, t, 2.7, 2.5))  # 마지막으로 본 뒤 obstacle_memory_s 까지만

    def test_forgets_an_obstacle_that_is_seen_to_be_gone(self):
        tracker = ObstacleTracker(self.p)
        pose = Pose2D(2.0, 2.5, 0.0)
        self.feed(tracker, 0.0, pose, [(2.8, 2.5, 0.10)])
        for k in range(1, 30):
            self.feed(tracker, 0.1 * k, pose, [])  # 떠났다 — 광선이 그 자리를 지나 벽까지 닿는다
        self.assertFalse(self.blocked(tracker, 3.0, 2.7, 2.5))

    def test_keeps_an_obstacle_hidden_behind_a_nearer_one(self):
        tracker = ObstacleTracker(self.p)
        pose = Pose2D(2.0, 2.5, 0.0)
        for k in range(-7, 1):
            self.feed(tracker, 0.1 * k, pose, [(3.0, 2.5, 0.10)])
        for k in range(1, 50):
            self.feed(tracker, 0.1 * k, pose, [(2.45, 2.5, 0.10)])  # 더 가까운 것이 앞을 가렸다 — 뒤의 것이 있는지 알 수 없다
        self.assertTrue(self.blocked(tracker, 5.0, 2.9, 2.5))

    def test_a_cell_seen_only_briefly_is_not_remembered(self):
        """한두 스캔에만 찍힌 칸(빨리 돌 때 방향이 틀려 찍힌 벽 등)은 뒤로 가도 기억하지 않는다 — obstacle_ttl_s 뒤 사라진다."""
        tracker = ObstacleTracker(self.p)
        thing = [(2.8, 2.5, 0.10)]
        self.feed(tracker, 0.0, Pose2D(2.0, 2.5, 0.0), thing)
        self.feed(tracker, 0.1, Pose2D(2.0, 2.5, 0.0), thing)
        behind = Pose2D(2.0, 2.5, math.pi)
        for k in range(2, 60):
            self.feed(tracker, 0.1 * k, behind, thing)
        self.assertFalse(self.blocked(tracker, 6.0, 2.7, 2.5))

    def test_scans_taken_while_turning_fast_do_not_add_obstacles(self):
        """빨리 도는 동안의 스캔은 새 칸을 만들지 않는다 (이미 있는 칸은 이어 간다). 천천히 돌 때는 전처럼 만든다."""
        pose = Pose2D(2.0, 2.5, 0.0)
        ranges, angle_min, inc = lidar_scan(pose, [(2.8, 2.5, 0.10)], self.p, self.spec)
        body = scan_to_body(ranges, angle_min, inc, self.p)
        rays = (angle_min, inc, np.asarray(ranges))
        fast = ObstacleTracker(self.p)
        fast.on_scan(0.0, pose, body, self.spec, self.grid, rays, turn_rate=1.0)
        self.assertFalse(self.blocked(fast, 0.1, 2.7, 2.5))
        self.assertTrue(fast.unexplained_body)                      # 정지 · 회전 확인은 그 점을 그대로 본다
        self.assertTrue(fast.blocking_ahead(0.0, 1.0, speed=0.45, extra=0.3))
        self.assertGreater(fast.skipped_turning, 0)
        slow = ObstacleTracker(self.p)
        slow.on_scan(0.0, pose, body, self.spec, self.grid, rays, turn_rate=0.3)
        self.assertTrue(self.blocked(slow, 0.1, 2.7, 2.5))
        slow.on_scan(0.1, pose, body, self.spec, self.grid, rays, turn_rate=1.0)   # 있던 칸은 도는 동안에도 이어진다
        self.assertTrue(self.blocked(slow, 2.5, 2.7, 2.5))

    def test_cells_under_the_robot_are_dropped(self):
        """라이다는 몸 바로 곁을 못 봐서 그 자리가 비었는지 알 수 없다 — 몸이 올라선 자리는 비어 있으니 지운다
        (실차 bag 1118_201710: 이것이 없을 때 기억으로만 남은 칸 위를 실제로 29번 지나갔다)."""
        tracker = ObstacleTracker(self.p)
        self.feed(tracker, 0.0, Pose2D(2.0, 2.5, 0.0), [(2.6, 2.5, 0.03)])
        self.assertIn(self.grid.cell_of(2.57, 2.5), tracker._seeds)
        self.feed(tracker, 0.2, Pose2D(2.6, 2.5, math.pi), [])  # 그 자리에 와서 섰다 (돌아서서 뒤쪽은 못 본다)
        self.assertNotIn(self.grid.cell_of(2.57, 2.5), tracker._seeds)

    def test_without_rays_it_forgets_after_the_ttl_as_before(self):
        tracker = ObstacleTracker(self.p)
        self.feed(tracker, 0.0, Pose2D(2.0, 2.5, 0.0), [(2.8, 2.5, 0.10)], rays=False)
        for k in range(1, 30):
            self.feed(tracker, 0.1 * k, Pose2D(2.0, 2.5, math.pi), [(2.8, 2.5, 0.10)], rays=False)
        self.assertFalse(self.blocked(tracker, 3.0, 2.7, 2.5))


class ExecutorTest(unittest.TestCase):
    def test_normal_goal_around_box_with_heading(self):
        p, spec, grid = make_world()
        job = ex.GoalExecutor(goal(5.0, 2.5, yaw=math.pi / 2), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLess(math.hypot(robot.pose.x - 5.0, robot.pose.y - 2.5), 0.10)
        self.assertLess(abs(wrap(robot.pose.yaw - math.pi / 2)), 0.10 + 0.05)
        self.assertIn("aligning", {ph for *_, ph, _b in trace})
        for _t, pose, _c, _ph, _b in trace:  # 박스에 닿지 않았다
            self.assertGreater(spec.distance_to_structure(pose.x, pose.y), p.robot_radius - 0.03)

    def test_yaw_ignored_with_supervisor_convention(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(2.0, 2.5, yaw=math.pi, yaw_tol=3.15), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p)
        self.assertEqual(job.outcome, ex.REACHED)
        self.assertNotIn("aligning", {ph for *_, ph, _b in trace})
        self.assertLess(abs(robot.pose.yaw), 0.2)

    def test_rotate_in_place_only(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(1.0, 2.5, yaw=-math.pi / 2), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        run(job, robot, spec, grid, p)
        self.assertEqual(job.outcome, ex.REACHED)
        self.assertLess(abs(wrap(robot.pose.yaw + math.pi / 2)), 0.15)
        self.assertLess(math.hypot(robot.pose.x - 1.0, robot.pose.y - 2.5), 0.05)

    def test_invalid_goals(self):
        p, spec, grid = make_world()
        here = Pose2D(1.0, 2.5, 0.0)
        cases = {
            "frame": goal(2.0, 2.5, frame="map"),
            "outside": goal(7.0, 2.5),
            "in box": goal(3.06, 2.5),
            "tolerance": goal(2.0, 2.5, pos_tol=0.0),
            "straight sideways": goal(1.0, 3.5, mode=ex.MODE_STRAIGHT),
        }
        for name, g in cases.items():
            with self.subTest(name):
                job = ex.GoalExecutor(g, p, 0.0)
                self.assertFalse(job.validate(spec, grid, here))
                self.assertEqual(job.outcome, ex.INVALID_GOAL)
                self.assertTrue(job.detail)
        job = ex.GoalExecutor(goal(2.0, 2.5), p, 0.0)
        self.assertFalse(job.validate(None, None, here))

    def test_goal_too_close_to_wall_is_moved(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(0.05, 2.5), p, 0.0)
        self.assertTrue(job.validate(spec, grid, Pose2D(1.0, 2.5, 0.0)))
        self.assertTrue(job.adjusted)
        self.assertTrue(grid.free(*job.target))

    def test_straight_mode_does_not_turn(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(1.6, 2.52, mode=ex.MODE_STRAIGHT, pos_tol=0.015), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLess(abs(robot.pose.x - 1.6), 0.02)
        self.assertLess(max(abs(c.w) for _t, _p, c, _ph, _b in trace), 0.51)
        self.assertLess(abs(robot.pose.yaw), 0.02)

    def test_assault_keeps_speed_and_stops_near_wall(self):
        p, spec, grid = make_world(boxes=())
        target_x = 6.12 - (p.robot_half_length + 0.05)  # 차체 앞이 RED 벽에서 5 cm
        job = ex.GoalExecutor(goal(target_x, 2.5, mode=ex.MODE_ASSAULT, pos_tol=0.03), p, 0.0)
        robot = SimRobot(Pose2D(5.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertGreater(max(c.v for _t, _p, c, _ph, _b in trace), p.assault_v - 1e-6)
        self.assertLess(abs(robot.pose.x - target_x), 0.03)
        self.assertLess(robot.pose.x + p.robot_half_length, 6.12)

    def test_waits_then_replans_around_unmapped_obstacle(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(3.5, 2.5), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, discs=[(2.0, 2.5, 0.12)])
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        for _t, pose, *_ in trace:
            self.assertGreater(math.hypot(pose.x - 2.0, pose.y - 2.5), 0.12 + p.robot_half_length - 0.02)

    def test_blocked_is_reported_and_gives_up_early_when_only_path_is_through_an_unmapped_obstacle(self):
        """벽의 유일한 틈을 지도에 없는 물체가 막았다: blocked 를 알리며 기다리다 obstacle_give_up_s 뒤 시한 전에 BLOCKED (2026-11-18 팀 테스트 뒤)."""
        boxes = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]
        p, spec, grid = make_world(boxes=boxes, params=old_blocking())   # 전 흐름 (지금 기본은 StandingBlockerTest)
        job = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=40.0), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.6, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, discs=[(3.0, 2.6, 0.12)], max_t=45.0)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("지도에 없는 물체", job.detail)
        first_blocked = next(t for t, _p, _c, _ph, b in trace if b)
        ended = trace[-1][0]
        # 진전이 멈춘 뒤 give_up 만큼 + 지나온 길로 물러나는 시간 (되짚기, 2026-11-19 밤) — 시한(40 s)보다 훨씬 전
        self.assertLess(ended, first_blocked + p.obstacle_give_up_s + p.retrace_m / p.retrace_v + 1.0)
        self.assertTrue(all(b for t, _p, _c, _ph, b in trace if t >= first_blocked))  # 막힌 동안 계속 알린다
        for _t, pose, *_ in trace:  # 막힌 물체에 박지 않았다
            self.assertGreater(math.hypot(pose.x - 3.0, pose.y - 2.6), 0.12 + p.robot_half_length - 0.02)

    def test_slows_down_and_stops_farther_from_an_unmapped_obstacle_when_fast(self):
        """돌진(0.6 m/s) 중 앞에 지도에 없는 물체: 미리 줄이고, 서 있을 때의 stop_dist 보다 앞에서 선다 (2026-11-18 팀 테스트 2차).
        벽의 유일한 틈을 물체가 막아 돌아갈 길이 없다 → 곧게 다가가 선다."""
        boxes = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]
        p, spec, grid = make_world(boxes=boxes, params=old_blocking())   # 0.30 m 앞까지 가는 전 흐름에서 속도에 따른 정지 거리를 본다
        job = ex.GoalExecutor(goal(5.0, 2.6, mode=ex.MODE_ASSAULT, timeout_s=6.0), p, 0.0)
        robot = SimRobot(Pose2D(0.8, 2.6, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        disc = (3.0, 2.6, 0.12)
        trace = run(job, robot, spec, grid, p, discs=[disc], max_t=6.0)
        approach = [(pose, c) for _t, pose, c, _ph, _b in trace]
        gaps = [(disc[0] - disc[2]) - (pose.x + p.robot_half_length) for pose, _c in approach]
        self.assertGreater(min(gaps), p.stop_dist - 0.05)  # 그대로 0.6 m/s 로 오다 0.30 m 에서 서면 그 안으로 밀려 들어온다
        near = [c.v for (pose, c), gap in zip(approach, gaps) if 0.35 < gap < 0.5]
        self.assertTrue(near and max(near) < p.assault_v - 0.15, near)  # 가까워지며 줄였다

    def test_backs_off_from_something_touching_the_front(self):
        """지도에 없는 물체가 차체 앞에 붙어 있으면(닿음 · 상대가 다가옴) 기다리기만 하지 않고 뒤로 뺀다."""
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(4.0, 2.5, timeout_s=6.0), p, 0.0)
        robot = SimRobot(Pose2D(2.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        disc = (2.0 + p.robot_half_length + 0.06 + 0.10, 2.5, 0.10)  # 표면이 차체 앞 6 cm
        trace = run(job, robot, spec, grid, p, discs=[disc], max_t=6.0)
        phases = [ph for *_r, ph, _b in trace]
        self.assertIn("backing_up", phases)
        self.assertLess(min(pose.x for _t, pose, *_ in trace), 2.0 - 0.05)

    def test_does_not_back_into_a_wall(self):
        """뒤가 벽이면 앞에 무엇이 붙어 있어도 뒤로 빼지 않는다."""
        p, spec, grid = make_world(boxes=())
        start_x = p.robot_half_length + 0.05  # 차체 뒤가 벽에서 5 cm
        job = ex.GoalExecutor(goal(3.0, 2.5, timeout_s=5.0), p, 0.0)
        robot = SimRobot(Pose2D(start_x, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        disc = (start_x + p.robot_half_length + 0.06 + 0.10, 2.5, 0.10)
        trace = run(job, robot, spec, grid, p, discs=[disc], max_t=5.0)
        self.assertNotIn("backing_up", [ph for *_r, ph, _b in trace])
        self.assertGreater(min(pose.x for _t, pose, *_ in trace), start_x - 0.02)

    def test_does_not_turn_in_place_into_a_touching_obstacle(self):
        """목표가 뒤쪽이라 서서 돌아야 하는데 차체 모서리 바로 앞에 지도에 없는 물체가 있으면 돌기 전에 물러난다."""
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(1.0, 2.5, timeout_s=10.0), p, 0.0)
        robot = SimRobot(Pose2D(2.5, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        disc = (2.5 + 0.14, 2.5 + 0.16, 0.05)  # 왼쪽 앞 모서리 바로 바깥
        trace = run(job, robot, spec, grid, p, discs=[disc], max_t=10.0)
        phases = [ph for *_r, ph, _b in trace]
        self.assertIn("backing_up", phases)
        self.assertLess(phases.index("backing_up"), phases.index("rotating") if "rotating" in phases else len(phases))
        self.assertEqual(job.outcome, ex.REACHED, job.detail)

    def test_blocked_clears_when_path_opens(self):
        boxes = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]
        p, spec, grid = make_world(boxes=boxes, params=old_blocking())   # 서 있는 물체 앞에서 2 s 에 끝내지 않는 전 흐름 (상대가 8 s 뒤 비킨다)
        job = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=40.0), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.6, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        obstacles = ObstacleTracker(p)
        t, saw_blocked = 0.0, False
        while not job.done and t < 40.0:
            if t < 8.0 and int(round(t / DT)) % 2 == 0:  # 8 초 뒤 상대가 비킨다
                obstacles.on_scan(t, robot.pose, scan_of_discs(robot.pose, [(3.0, 2.6, 0.12)], p), spec, grid)
            elif int(round(t / DT)) % 2 == 0:
                obstacles.on_scan(t, robot.pose, [], spec, grid)
            cmd = job.step(t, robot.pose, loc.OK, grid, obstacles)
            saw_blocked = saw_blocked or job.blocked
            robot.apply(cmd)
            t += DT
        self.assertTrue(saw_blocked)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)

    def test_localization_hold_then_lost(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(4.0, 2.5), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        level = lambda t: loc.OK if t < 1.0 else (loc.HOLD if t < 3.0 else loc.LOST)  # noqa: E731
        trace = run(job, robot, spec, grid, p, level_fn=level)
        self.assertEqual(job.outcome, ex.LOCALIZATION_LOST)
        held = [c for t, _p, c, ph, _b in trace if 1.0 <= t < 3.0]
        self.assertTrue(all(c.v == 0.0 and c.w == 0.0 for c in held))
        self.assertFalse(any(b for *_r, b in trace))  # 위치를 기다린 시간은 막힘이 아니다

    def test_waits_briefly_for_first_pose(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(4.0, 2.5), p, 0.0)
        self.assertTrue(job.validate(spec, grid, None))
        self.assertEqual(job.step(0.5, None, loc.LOST, grid, None), ex.STOP)
        self.assertIsNone(job.outcome)
        self.assertEqual(job.phase, "holding_localization")
        job.step(p.pose_lost_s + 0.1, None, loc.LOST, grid, None)
        self.assertEqual(job.outcome, ex.LOCALIZATION_LOST)

    def test_degraded_caps_speed(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(4.0, 2.5), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, level_fn=lambda t: loc.DEGRADED, max_t=3.0)
        self.assertLessEqual(max(c.v for _t, _p, c, _ph, _b in trace), p.degraded_v + 1e-9)
        self.assertIn("degraded_localization", trace[-1][3])

    def test_timeout_and_cancel(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(5.5, 2.5, timeout_s=2.0), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        run(job, robot, spec, grid, p)
        self.assertEqual(job.outcome, ex.TIMED_OUT)
        job = ex.GoalExecutor(goal(5.5, 2.5), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        job.finish(ex.CANCELED, "취소 요청")
        self.assertEqual(job.step(0.0, robot.pose, loc.OK, grid, None), ex.STOP)

    def test_stuck_backs_up(self):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(4.0, 2.5), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        stuck_until = 3.0

        def hold_robot(t, r, cmd):  # 3 초 동안 바퀴가 헛돈다
            r.stuck = t < stuck_until

        trace = run(job, robot, spec, grid, p, on_tick=hold_robot, max_t=30.0)
        self.assertIn("backing_up", {ph for *_r, ph, _b in trace})
        self.assertEqual(job.outcome, ex.REACHED)


def min_body_gap(trace, spec, p):
    """궤적 내내 차체 사각형과 지도 구조물 사이의 가장 좁은 틈 (간단 시뮬은 부딪힘을 흉내 내지 않으므로 이것으로 본다)."""
    return min(ex.body_clearance(spec, pose.x, pose.y, pose.yaw, p.robot_half_length, p.robot_half_width) for _t, pose, *_ in trace)


class TurnRoomTest(unittest.TestCase):
    """서서 돌기와 지도 구조물(벽 · 박스 · 고정 기둥) — 2026-11-19. 차체 반 대각선(0.170)이 격자 부풀림(0.17)과 같아,
    구조물 곁에서 서서 돌면 모서리가 걸려 시한까지 돌라는 명령만 냈다 (실차 흉내 30판 중 7판). 이제 돌기 전에 차체를 돌려 보고,
    막히면 비켜 보고, 그래도 안 되면 BLOCKED + detail 「돌 자리 없음(…)」 으로 supervisor 에 알린다."""

    def test_body_clearance_is_exact(self):
        spec = MapSpec(6.12, 5.0, 0.40, ((3.0, 2.5),), "arena", fixed=((0.2, 1.667),))
        hl, hw = 0.125, 0.115
        self.assertAlmostEqual(ex.body_clearance(spec, 1.5, 0.13, 0.0, hl, hw), 0.015, places=6)  # 아래 벽과 나란히
        self.assertAlmostEqual(ex.body_clearance(spec, 3.0 - 0.2 - hl - 0.05, 2.5, 0.0, hl, hw), 0.05, places=6)  # 박스를 마주 봄
        self.assertLessEqual(ex.body_clearance(spec, 3.0 - 0.2 - 0.05, 2.5, 0.3, hl, hw), 0.0)  # 박스에 파고듦
        self.assertAlmostEqual(ex.body_clearance(spec, 0.4 + hl + 0.03, 1.667, 0.0, hl, hw), 0.03, places=6)  # 고정 기둥도 구조물
        # 무작위 자세: 차체 둘레를 1 mm 간격으로 훑은 값과 같다 (겹치지 않을 때)
        import random
        rng = random.Random(3)
        checked = 0
        for _ in range(300):
            x, y, yaw = 3.0 + rng.uniform(-0.6, 0.6), 2.5 + rng.uniform(-0.6, 0.6), rng.uniform(-math.pi, math.pi)
            exact = ex.body_clearance(spec, x, y, yaw, hl, hw)
            if exact <= 0.0:
                continue
            c, s_ = math.cos(yaw), math.sin(yaw)
            pts = [(u, -hw) for u in [-hl + k * 0.001 for k in range(251)]] + [(u, hw) for u in [-hl + k * 0.001 for k in range(251)]]
            pts += [(-hl, v) for v in [-hw + k * 0.001 for k in range(231)]] + [(hl, v) for v in [-hw + k * 0.001 for k in range(231)]]
            brute = min(spec.distance_to_structure(x + c * u - s_ * v, y + s_ * u + c * v) for u, v in pts)
            self.assertAlmostEqual(exact, brute, delta=0.0015)
            checked += 1
        self.assertGreater(checked, 100)

    def test_reports_no_room_at_goal_instead_of_turning_into_a_wall(self):
        """벽과 나란히 2.5 cm 옆에 와서 벽 쪽 90° 로 서라는 목표: 비킬 수도 없게(no_room_moves 0) 하면 벽에 대고 돌지 않고 바로
        도착으로 끝낸다 (자리는 맞다 — 미션을 이어 가게). detail 에 「돌 자리 없음(도착 방향)」."""
        p, spec, grid = make_world(boxes=())
        p.no_room_moves = 0
        robot = SimRobot(Pose2D(3.0, 0.14, 0.0))
        job = ex.GoalExecutor(goal(3.0, 0.14, yaw=math.pi / 2, timeout_s=20.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=20.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn(ex.NO_ROOM_AT_GOAL, job.detail)
        self.assertIn("벽 · 상자 · 기둥", job.detail)
        self.assertTrue(all(abs(cmd.w) < 1e-9 for _t, _p, cmd, *_ in trace))  # 벽에 대고 돌지 않았다
        self.assertLess(trace[-1][0], 3.0)

    def test_arcs_off_a_wall_it_is_parallel_to_then_turns(self):
        """같은 자리에서 비키게 두면: 곧게는 벽에서 안 멀어지므로 완만한 호로 비켜 나와 돌고, 벽에 닿지 않는다."""
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.0, 0.14, 0.0))
        job = ex.GoalExecutor(goal(3.0, 0.40, yaw=math.pi / 2, timeout_s=30.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=30.0)
        phases = [ph for *_r, ph, _b in trace]
        self.assertTrue({"moving_off", "backing_up"} & set(phases), set(phases))
        self.assertGreater(min_body_gap(trace, spec, p), 0.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)

    def test_backs_off_a_box_it_faces_then_turns_toward_a_goal_behind(self):
        """박스를 마주 보고 3.5 cm 앞에 섰는데 목표가 뒤에 있다: 뒤로 비켜 가운데를 박스에서 떼고 돈다 (전에는 모서리가 박스에 걸렸다)."""
        p, spec, grid = make_world(boxes=((3.0, 2.5),))
        robot = SimRobot(Pose2D(3.0 - 0.2 - 0.16, 2.5, 0.0))
        job = ex.GoalExecutor(goal(1.5, 2.5, timeout_s=30.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=30.0)
        phases = [ph for *_r, ph, _b in trace]
        self.assertIn("backing_up", phases)
        self.assertLess(phases.index("backing_up"), phases.index("rotating"))
        self.assertGreater(min_body_gap(trace, spec, p), 0.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)

    def test_reports_no_room_to_leave_when_it_cannot_turn_toward_the_path(self):
        p, spec, grid = make_world(boxes=())
        p.no_room_moves = 0
        robot = SimRobot(Pose2D(3.0, 0.14, 0.0))
        job = ex.GoalExecutor(goal(3.0, 2.0, timeout_s=20.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=20.0)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn(ex.NO_ROOM_TO_LEAVE, job.detail)
        self.assertLess(trace[-1][0], 3.0)

    def test_a_turn_that_does_not_happen_is_reported(self):
        """지도로는 돌 자리가 있는데 실제로는 안 돈다(지도와 실제가 어긋나 모서리가 걸림): rot_stuck_s 뒤, 자리는 맞으니 바로 도착으로 끝낸다
        (시한까지 돌라는 명령만 내지 않는다. 자리에서는 비키지 않는다 — 비키면 자리를 벗어나 오가기만 한다)."""
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        robot.stuck = True
        job = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.pi / 2, timeout_s=30.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=30.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn(ex.NO_ROOM_AT_GOAL, job.detail)
        self.assertIn("방향이 안 바뀜", job.detail)
        self.assertNotIn("backing_up", [ph for *_r, ph, _b in trace])
        self.assertLess(trace[-1][0], 5.0)

    def test_heading_that_never_settles_ends_as_reached(self):
        """도착 방향이 align_max_s 넘게 허용 안에 안 들면 남은 각도와 상관없이 도착으로 끝낸다 (시한까지 돌지 않는다)."""
        p, spec, grid = make_world(boxes=())
        p.align_max_s = 3.0
        p.rot_stuck_s = 100.0  # 안 도는 회전 감지는 끄고 시간 상한만 본다
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        robot.stuck = True  # 도는 명령을 내도 차체는 그대로 → 방향이 늘 30° 어긋나 있다
        job = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.radians(30.0), timeout_s=30.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=30.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn("방향 맞추기", job.detail)
        self.assertLess(trace[-1][0], 5.0)

    def test_open_floor_turns_are_unchanged(self):
        """구조물에서 먼 곳의 서서 돌기는 전과 같다 (검사를 건너뛴다)."""
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        job = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.pi, timeout_s=20.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=20.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertFalse({"moving_off", "backing_up"} & {ph for *_r, ph, _b in trace})


class ResultCodeTest(unittest.TestCase):
    """결과 코드 (2026-11-19): detail 끝의 「[코드]」. REACHED 인데 끝낸 자세가 요청한 허용 밖이면 어느 길로 끝났든 코드가 붙는다."""

    def test_plain_reached_has_no_code(self):
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        job = ex.GoalExecutor(goal(4.0, 2.5, yaw=0.0, timeout_s=20.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        run(job, robot, spec, grid, p, max_t=20.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertEqual((job.codes, job.detail), ([], ""))

    def test_reached_with_heading_off_always_says_so(self):
        """방향을 못 맞춘 채 도착으로 끝나는 두 길(돌 자리 없음 · 시간 상한) 모두 [YAW_NOT_MET] 가 붙는다."""
        for setup, also in ((lambda p: None, ex.CODE_NO_ROOM_AT_GOAL), (lambda p: (setattr(p, "align_max_s", 3.0), setattr(p, "rot_stuck_s", 100.0)), None)):
            p, spec, grid = make_world(boxes=())
            setup(p)
            robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
            robot.stuck = True
            job = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.pi / 2, timeout_s=30.0), p, 0.0)
            self.assertTrue(job.validate(spec, grid, robot.pose))
            run(job, robot, spec, grid, p, max_t=30.0)
            self.assertEqual(job.outcome, ex.REACHED, job.detail)
            self.assertIn("[%s]" % ex.CODE_YAW_NOT_MET, job.detail)
            self.assertNotIn(ex.CODE_POS_NOT_MET, job.codes)
            if also:
                self.assertIn("[%s]" % also, job.detail)

    def test_blocked_codes(self):
        """못 움직이는 차체: 시한이 짧으면 [BLOCKED_AT_TIMEOUT], 길면 그 전에 [NO_PROGRESS]."""
        for timeout_s, code in ((6.0, ex.CODE_BLOCKED_AT_TIMEOUT), (40.0, ex.CODE_NO_PROGRESS)):
            p, spec, grid = make_world(boxes=())
            robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
            robot.stuck = True
            job = ex.GoalExecutor(goal(5.0, 2.5, timeout_s=timeout_s), p, 0.0)
            self.assertTrue(job.validate(spec, grid, robot.pose))
            run(job, robot, spec, grid, p, max_t=45.0)
            self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
            self.assertEqual(job.codes, [code], job.detail)
            self.assertTrue(job.detail.endswith("[%s]" % code), job.detail)


class EscapeTest(unittest.TestCase):
    """비키는 동작 하나로 (2026-11-19): 지도와 라이다 점을 함께 보고 고른다. 뒤(라이다가 못 보는 쪽)로 빼기는 한도가 있다."""

    def _job(self, p, spec, grid, pts):
        obstacles = ObstacleTracker(p)
        pose = Pose2D(3.0, 2.5, 0.0)
        obstacles.on_scan(0.0, pose, pts, spec, grid)
        job = ex.GoalExecutor(goal(5.0, 2.5), p, 0.0)
        self.assertTrue(job.validate(spec, grid, pose))
        return job, pose, obstacles

    def test_backs_away_from_something_right_ahead_but_only_so_many_times(self):
        p, spec, grid = make_world(boxes=())
        ahead = [(0.125 + 0.08, y) for y in (-0.08, -0.04, 0.0, 0.04, 0.08)]  # 차체 앞 8 cm
        job, pose, obstacles = self._job(p, spec, grid, ahead)
        move = job._room_move_plan(0.0, pose, grid, obstacles, "obstacle")
        self.assertIsNotNone(move)
        self.assertLess(move[0], 0.0)  # 앞은 막혔으니 뒤로
        for _ in range(p.reverse_max):
            job._note_reverse(pose)
        self.assertIsNone(job._room_move_plan(0.0, pose, grid, obstacles, "obstacle"))  # 가까워지지 못한 채 한도만큼 뺐다 → 더 빼지 않는다
        closer = Pose2D(pose.x + p.reverse_refill_m + 0.05, pose.y, 0.0)
        self.assertTrue(job._reverse_ok(closer))  # 목표에 가까워졌으면 다시 센다

    def test_prefers_the_side_the_lidar_can_see(self):
        """왼쪽 옆에 붙은 물체: 뒤로도 앞으로도 멀어질 수 있으면 보이는 앞쪽(오른쪽으로 도는 호)으로 비킨다. 그 물체에 더 붙지 않는다."""
        p, spec, grid = make_world(boxes=())
        beside = [(x, 0.115 + 0.05) for x in (-0.08, -0.04, 0.0, 0.04, 0.08)]  # 왼쪽 옆 5 cm
        job, pose, obstacles = self._job(p, spec, grid, beside)
        v, w, _duration = job._room_move_plan(0.0, pose, grid, obstacles, "obstacle")
        self.assertGreater(v, 0.0)
        self.assertLess(w, 0.0)

    def test_a_stuck_robot_is_not_backed_up_again_and_again(self):
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        robot.stuck = True
        job = ex.GoalExecutor(goal(5.0, 2.5, timeout_s=40.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=45.0)
        phases = [ph for *_r, ph, _b in trace]
        backups = sum(1 for a, b in zip(phases, phases[1:]) if b == "backing_up" and a != "backing_up")
        # 뒤로 빼기 한 번(0.8 s)은 뒤로 갈 때의 진행 확인(시작 1.2 s 에 첫 판정)보다 짧다 → 전처럼 reverse_max 번까지
        self.assertEqual(backups, p.reverse_max)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)



def trace_tail(trace, x_from):
    """(자세, 명령) — 차체가 x_from 을 넘어선 뒤의 것 (동쪽으로 가는 주행)."""
    return [(pose, cmd) for _t, pose, cmd, _ph, _b in trace if pose.x >= x_from]


class StandingBlockerTest(unittest.TestCase):
    """서 있는 물체가 유일한 길을 막았다 — 다가가지도 오래 기다리지도 않는다 (2026-11-19 밤).
    본선 경기장 4판: 떨어져 선 물체 앞에서 2 s 넘게 기다린 15번 모두 그 물체는 움직이지 않았다. 사용자: 교착에서 기다려도 상대는 움직이지 않는다."""

    WALL = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]   # 틈은 y 2.6 하나
    DISC = (3.0, 2.6, 0.12)

    def drive(self, params=None, target=(5.0, 2.6), discs=None):
        p, spec, grid = make_world(boxes=self.WALL, params=params)
        job = ex.GoalExecutor(goal(*target, timeout_s=40.0), p, 0.0)
        robot = SimRobot(Pose2D(1.0, 2.6, 0.0))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, discs=[self.DISC] if discs is None else discs, max_t=45.0)
        return job, robot, trace, p

    def gap(self, pose, p):
        return (self.DISC[0] - self.DISC[2]) - (pose.x + p.robot_half_length)

    def test_stops_well_short_and_gives_up_after_two_seconds(self):
        job, robot, trace, p = self.drive()
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("[OBSTACLE]", job.detail)
        self.assertGreater(min(self.gap(pose, p) for _t, pose, *_ in trace), p.through_standoff_m - 0.10)   # 0.70 m 앞에서 섰다 (전 0.30)
        stopped = next(t for t, _pose, _c, ph, _b in trace if ph == "waiting_obstacle")
        self.assertLess(trace[-1][0], stopped + p.obstacle_give_up_static_s + 1.0)                          # 선 뒤 2 s 남짓 (전 5 s)
        self.assertFalse(any(ph == "retracing" for *_r, ph, _b in trace))                                   # 돌 자리가 있어 물러날 일이 없다
        self.assertNotIn("지나온 길로", job.detail)

    def test_old_flow_goes_up_to_the_obstacle_and_waits_five_seconds(self):
        job, robot, trace, p = self.drive(old_blocking(retrace_m=0.0))
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertLess(min(self.gap(pose, p) for _t, pose, *_ in trace), p.stop_dist + 0.12)
        stopped = next(t for t, _pose, _c, ph, _b in trace if ph == "waiting_obstacle")
        self.assertGreater(trace[-1][0], stopped + p.obstacle_give_up_s - 0.5)

    def test_goal_in_front_of_the_obstacle_is_still_reached(self):
        """목표가 그 물체보다 앞이면(물체 곁의 자리) 멀리 서지 않고 목표까지 간다."""
        job, robot, _trace, p = self.drive(target=(2.42, 2.6))   # 물체 면(2.88)에서 차체 앞까지 0.33 m 인 자리 — 임시 장애물이 막는 칸이다
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLess(abs(robot.pose.x - 2.42), 0.10)

    def test_blocker_static_tells_standing_from_moving(self):
        p, spec, grid = make_world(boxes=())
        pose = Pose2D(2.0, 2.5, 0.0)
        standing = ObstacleTracker(p)
        moving = ObstacleTracker(p)
        t = 0.0
        while t < 2.6:
            standing.on_scan(t, pose, scan_of_discs(pose, [(2.8, 2.5, 0.12)], p), spec, grid)
            moving.on_scan(t, pose, scan_of_discs(pose, [(3.3 - 0.25 * t, 2.5, 0.12)], p), spec, grid)   # 0.25 m/s 로 다가온다
            if t < p.tight_still_s - 0.2:
                self.assertFalse(standing.blocker_static(t, grid, p.tight_still_s))   # 아직 오래 보지 않았다
            t += 0.1
        self.assertTrue(standing.blocker_static(t - 0.1, grid, p.tight_still_s))
        self.assertFalse(moving.blocker_static(t - 0.1, grid, p.tight_still_s))
        self.assertFalse(ObstacleTracker(p).blocker_static(t, grid, p.tight_still_s))   # 막는 점이 없다


class RetraceTest(unittest.TestCase):
    """시한 전에 BLOCKED 로 끝내기 전에, 빠듯한 자리면 지나온 길을 돌지 않고 뒤로 되짚어 물러난다 (2026-11-19 밤).
    본선 경기장 8판: supervisor 가 지나온 점을 보통 목표로 보낸 「되돌아가기」 는 28번 중 2번만 닿았다 — 막힌 자리에서 먼저 서서 돌아야 했다."""

    WALL = [(3.0, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(3.0, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]   # 틈은 y 2.6 하나

    def blocked_in_the_gap(self, params=None, on_tick=None, start=(1.0, 2.6, 0.0), trail=None):
        # 물체 0.30 m 앞(상자 사이 틈 안)까지 가서 막히는 전 흐름으로 본다 — 지금 기본은 0.70 m 앞에 서서 물러날 일이 없다 (StandingBlockerTest)
        p, spec, grid = make_world(boxes=self.WALL, params=params or old_blocking())
        job = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=40.0), p, 0.0, trail=trail)
        robot = SimRobot(Pose2D(*start))
        self.assertTrue(job.validate(spec, grid, robot.pose))
        hook = (lambda t, r, c: on_tick(t, r, c, job)) if on_tick else None
        trace = run(job, robot, spec, grid, p, discs=[(3.0, 2.6, 0.12)], max_t=45.0, on_tick=hook)
        return job, robot, trace, p

    def test_backs_out_along_its_own_path_without_turning_before_giving_up(self):
        job, robot, trace, p = self.blocked_in_the_gap()
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("지나온 길로", job.detail)
        self.assertIn("[OBSTACLE]", job.detail)
        back = [(pose, cmd) for _t, pose, cmd, ph, _b in trace if ph == "retracing"]
        self.assertTrue(back)
        self.assertTrue(all(cmd.v < 0.0 for _pose, cmd in back))
        self.assertGreater(back[0][0].x - robot.pose.x, p.retrace_m - 0.10)    # 그만큼 물러났다
        self.assertLess(abs(robot.pose.y - 2.6), 0.08)                         # 지나온 길(곧은 길) 위로
        self.assertLess(abs(wrap(robot.pose.yaw)), math.radians(15.0))         # 돌지 않았다

    def test_follows_a_curved_path_backwards(self):
        """비스듬히 와서 틈 앞에서 꺾었다 — 뒤로 가면서 그 굽은 길을 따라간다."""
        job, _robot, trace, _p = self.blocked_in_the_gap(start=(1.6, 1.9, 0.6))
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("지나온 길로", job.detail)
        came = [pose for _t, pose, _c, ph, _b in trace if ph != "retracing"]
        for _t, pose, _c, ph, _b in trace:
            if ph == "retracing":   # 되짚는 동안 차체는 올 때 지나간 자리에서 0.10 m 안에 있다
                self.assertLess(min(math.hypot(pose.x - q.x, pose.y - q.y) for q in came), 0.10)

    def test_retrace_can_be_turned_off(self):
        job, _robot, trace, p = self.blocked_in_the_gap(old_blocking(retrace_m=0.0))
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertNotIn("지나온 길로", job.detail)
        self.assertFalse(any(ph == "retracing" for *_r, ph, _b in trace))
        first_blocked = next(t for t, _p, _c, _ph, b in trace if b)
        self.assertLess(trace[-1][0], first_blocked + p.obstacle_give_up_s)

    def test_stops_retracing_when_the_rear_is_blocked_and_remembers_it(self):
        """되짚는 중에 뒤에 선 것(라이다가 못 본다)에 닿아 더 못 간다 → 밀지 않고 끝내고, 그 뒤 얼마 동안은 뒤로 가지 않는다."""
        seen = {}

        def jam(t, robot, _cmd, job):
            if job.phase == "retracing":
                seen.setdefault("since", t)
            if "since" in seen and t - seen["since"] > 1.0:
                robot.stuck = True

        trail = Trail()
        job, robot, trace, p = self.blocked_in_the_gap(on_tick=jam, trail=trail)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        ended = trace[-1][0]
        self.assertLess(ended, seen["since"] + 1.0 + 2 * p.reverse_check_s + 0.3)   # 닿은 뒤 1.7 s 안에 그만둔다 (물러나는 시간 한도 7 s 까지 밀지 않는다)
        self.assertGreater(ended, seen["since"] + 1.0)                                # 반응 지연 동안(시작 0.5 s)에는 판정하지 않는다
        self.assertTrue(trail.rear_blocked(ended))
        self.assertFalse(trail.rear_blocked(ended + p.rear_blocked_s + 0.1))
        again = ex.GoalExecutor(goal(5.0, 2.6, timeout_s=40.0), p, ended, trail=trail)   # 다음 목표도 그 기억을 이어받는다
        self.assertFalse(again._reverse_ok(robot.pose))

    def test_straight_goal_backwards_ends_early_when_the_rear_is_blocked(self):
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        robot.stuck = True
        job = ex.GoalExecutor(goal(2.6, 2.5, mode=ex.MODE_STRAIGHT, pos_tol=0.08, timeout_s=12.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=15.0)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertIn("뒤가 막혀", job.detail)
        self.assertLess(trace[-1][0], 2.5)   # 전: 8 s 뒤 NO_PROGRESS

    def test_no_retrace_in_the_open(self):
        """넓은 곳에서 가까워지지 못해 끝낼 때는 물러나지 않는다 (그 자리에서 돌아 다른 데로 가면 된다)."""
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(1.0, 2.5, 0.0))
        job = ex.GoalExecutor(goal(5.0, 2.5, timeout_s=40.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))

        def jam(t, robot, _cmd):
            robot.stuck = t > 3.0

        trace = run(job, robot, spec, grid, p, max_t=45.0, on_tick=jam)
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)
        self.assertFalse(any(ph == "retracing" for *_r, ph, _b in trace))


class TrailTest(unittest.TestCase):
    def test_gives_the_point_behind_in_body_coordinates_and_rewinds(self):
        trail = Trail()
        for k in range(21):   # (0, 0) 에서 +y 쪽으로 1 m, 북쪽을 보고
            trail.update(Pose2D(0.0, 0.05 * k, math.pi / 2))
        bx, by = trail.back_target(0.25)
        self.assertAlmostEqual(bx, -0.275, delta=0.03)   # 차체 바로 뒤
        self.assertAlmostEqual(by, 0.0, delta=0.01)
        self.assertAlmostEqual(trail.back_room(), 1.0, delta=0.01)
        n = len(trail)
        trail.update(Pose2D(0.0, 0.80, math.pi / 2), rewinding=True)   # 0.2 m 되짚었다: 지나친 끝 점들은 지워진다
        self.assertLess(len(trail), n)
        bx, by = trail.back_target(0.25)
        self.assertAlmostEqual(bx, -0.275, delta=0.03)

    def test_turned_in_place_after_arriving_the_path_is_no_longer_behind(self):
        trail = Trail()
        for k in range(21):
            trail.update(Pose2D(0.05 * k, 0.0, 0.0))
        trail.update(Pose2D(1.0, 0.0, math.pi / 2))   # 와서 왼쪽으로 90° 돌았다 → 지나온 길은 차체 오른쪽
        err = ex.GoalExecutor._rear_err(trail.back_target(0.25))
        self.assertGreater(abs(err), math.radians(50.0))

    def test_empty_trail_has_no_target(self):
        trail = Trail()
        self.assertIsNone(trail.back_target(0.25))
        trail.update(Pose2D(1.0, 1.0, 0.0))
        self.assertIsNone(trail.back_target(0.25))


class CorridorTest(unittest.TestCase):
    """통로(양옆이 corridor_width_m 보다 좁은 자리)에서는 진행 띠의 옆 여유를 줄이고 천천히 간다 (2026-11-19 18:06 본선 경기장 실차:
    지도로는 0.73 m, 실제로는 0.45 m 인 벽 쪽 통로에서 양옆이 띠에 걸려 끝까지 못 나왔다)."""

    BOXES = tuple((x, 0.90) for x in (2.0, 2.4, 2.8, 3.2, 3.6, 4.0))   # 아래 면 y = 0.70 → 아래 벽과의 통로가 지도로는 0.70 m

    def drive(self, params, real_face=0.45, y0=0.30):
        p, spec, grid = make_world(boxes=self.BOXES, params=params)
        robot = SimRobot(Pose2D(1.2, y0, 0.0))
        job = ex.GoalExecutor(goal(3.0, y0, timeout_s=40.0), p, 0.0)      # 통로 안의 자리
        self.assertTrue(job.validate(spec, grid, robot.pose))
        face = [(x / 100.0, real_face, 0.01) for x in range(180, 425, 5)]  # 진짜 상자 면: 지도보다 0.25 m 아래
        trace = run(job, robot, spec, grid, p, discs=face, max_t=45.0)
        inside = [(pose, cmd) for _t, pose, cmd, _ph, _b in trace if 1.8 <= pose.x <= 4.2]
        return job, inside

    def test_narrow_corridor_is_driven_slowly_with_a_small_side_margin(self):
        job, inside = self.drive(DriveParams())
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLessEqual(max(abs(cmd.v) for _pose, cmd in inside), DriveParams().corridor_v + 1e-6)
        clear = min(0.45 - pose.y - (0.115 * abs(math.cos(pose.yaw)) + 0.125 * abs(math.sin(pose.yaw))) for pose, _cmd in inside)
        self.assertGreater(clear, 0.0, '진짜 상자 면에 닿음')

    def test_straight_run_in_a_corridor_that_matches_the_map_is_faster(self):
        """통로가 지도대로이고(곁에 지도에 없는 점이 없다) 곧게 가는 동안은 corridor_fast_v 까지 낸다. 자리에 다가가면 corridor_v 로 돌아온다."""
        p, spec, grid = make_world(boxes=self.BOXES, params=DriveParams())
        robot = SimRobot(Pose2D(1.2, 0.35, 0.0))
        job = ex.GoalExecutor(goal(4.6, 0.35, timeout_s=40.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=45.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        inside = [(pose, cmd) for _t, pose, cmd, _ph, _b in trace if 1.8 <= pose.x <= 4.2]
        top = max(abs(cmd.v) for _pose, cmd in inside)
        self.assertGreater(top, p.corridor_v + 0.02)
        self.assertLessEqual(top, p.corridor_fast_v + 1e-6)
        near = [abs(cmd.v) for pose, cmd in trace_tail(trace, 4.6 - p.corridor_fast_remaining_m + 0.05)]
        self.assertLessEqual(max(near), p.corridor_v + 1e-6)

    def test_corridor_fast_speed_can_be_turned_off(self):
        params = DriveParams()
        params.corridor_fast_v = 0.0
        p, spec, grid = make_world(boxes=self.BOXES, params=params)
        robot = SimRobot(Pose2D(1.2, 0.35, 0.0))
        job = ex.GoalExecutor(goal(4.6, 0.35, timeout_s=40.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=45.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLessEqual(max(abs(cmd.v) for _t, _pose, cmd, _ph, _b in trace), p.corridor_v + 1e-6)

    def test_without_the_corridor_rule_the_same_corridor_blocks(self):
        # goal_shift_m 0: 목표 칸을 덮은 서 있는 점 곁의 빈 칸으로 목표를 옮기는 것(2026-11-20)도 끈다 — 켜면 0.08 m 옆으로 옮겨 닿는다
        job, inside = self.drive(DriveParams(corridor_width_m=0.0, goal_shift_m=0.0))
        self.assertEqual(job.outcome, ex.BLOCKED, job.detail)

    def test_open_floor_and_the_aisle_between_box_rows_are_not_corridors(self):
        p, spec, grid = make_world(boxes=((2.0, 2.5), (3.2, 2.5)))        # 상자 사이 0.80 m 길
        job = ex.GoalExecutor(goal(5.0, 4.0), p, 0.0)
        self.assertFalse(job._corridor(Pose2D(2.6, 2.5, math.pi / 2), grid))
        self.assertFalse(job._corridor(Pose2D(4.5, 3.5, 0.0), grid))
        self.assertTrue(job._corridor(Pose2D(2.0, 2.0, 0.0), make_world(boxes=((2.0, 2.5), (2.0, 1.5)))[2]))   # 상자 사이 0.60 m

    def test_the_band_along_the_top_and_bottom_walls_is_a_corridor_by_position(self):
        """위 · 아래 벽에서 corridor_wall_band_m 안이면 폭을 재지 않고 통로다 (지도상의 자리만으로). 0 이면 자리로는 보지 않는다."""
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(5.0, 2.5), p, 0.0)
        self.assertTrue(job._corridor(Pose2D(3.0, 0.45, 0.0), grid))
        self.assertTrue(job._corridor(Pose2D(3.0, spec.height - 0.45, 0.0), grid))
        self.assertFalse(job._corridor(Pose2D(3.0, 1.0, 0.0), grid))
        self.assertFalse(job._corridor(Pose2D(0.45, 2.5, 0.0), grid))          # 왼쪽 · 오른쪽 벽(거점 쪽)은 아니다
        off = ex.GoalExecutor(goal(5.0, 2.5), DriveParams(corridor_wall_band_m=0.0), 0.0)
        self.assertFalse(off._corridor(Pose2D(3.0, 0.45, 0.0), grid))


class NearMemoryAndTimeoutTest(unittest.TestCase):
    def test_points_just_inside_the_outer_wall_are_the_wall(self):
        """외벽에서 wall_explain_m 안의 점은 벽이다 — 자세 · 경기장 크기가 10 cm 쯤 어긋나도 벽이 「지도에 없는 물체」 가 되지 않는다
        (2026-11-19 18:06 본선 경기장: 아래 벽이 0.10 ~ 0.13 m 안쪽에 보여 벽 쪽 통로의 양옆이 다 장애물이 됐다). 벽에서 더 나온 물체는 그대로 잡힌다."""
        p, spec, grid = make_world(boxes=())
        pose = Pose2D(3.0, 0.45, 0.0)
        nan = (0.0, 0.0125, np.full(502, np.nan))
        wall = [(dx, 0.13 - pose.y) for dx in np.arange(-0.4, 0.45, 0.05)]      # y = 0.13 에 보이는 아래 벽
        ugv = [(dx, 0.26 - pose.y) for dx in np.arange(0.5, 0.75, 0.05)]        # 벽에 붙어 선 UGV 의 이쪽 면 (y = 0.26)
        tracker = ObstacleTracker(p)
        tracker.on_scan(0.0, pose, wall + ugv, spec, grid, nan)
        self.assertEqual(len(tracker.unexplained_body), len(ugv))
        off = ObstacleTracker(DriveParams(wall_explain_m=0.0))
        off.on_scan(0.0, pose, wall + ugv, spec, grid, nan)
        self.assertEqual(len(off.unexplained_body), len(wall) + len(ugv))       # 끄면 전처럼 벽도 지도에 없는 점

    def test_when_memory_alone_closes_the_way_only_the_memory_is_dropped(self):
        """기억으로만 남은 칸 때문에 길이 없으면, 임시 장애물을 통째로 빼지 않고 기억만 뺀다 — 지금 찍히는 물체는 피해서 짠다
        (2026-11-19 본선 경기장 17:57 · 16:01 스캔 재생: 출발 순간 기억 때문에 길이 없던 목표 2 · 3개)."""
        p, spec, grid = make_world(boxes=())
        tracker = ObstacleTracker(p)
        pose = Pose2D(2.0, 2.5, 0.0)
        nan = (0.0, 0.0125, np.full(502, np.nan))                     # 광선이 다 값 없음 = 「비었다」 를 볼 수 없다
        wall = [(1.0, y - pose.y) for y in np.arange(0.2, 4.85, 0.05)]   # x = 3.0 에 위아래로 이어진 점들 (한 번 찍힘)
        for k in range(-7, 1):                                        # 0.7 s 동안 찍혔다
            tracker.on_scan(0.1 * k, pose, wall, spec, grid, nan)
        seen = (0.9, 0.0)                                             # 지금도 찍히는 물체: 바로 앞 (2.9, 2.5)
        for k in range(1, 8):
            tracker.on_scan(0.5 * k, pose, [seen], spec, grid, nan)
        t = 3.5
        extra, fresh = tracker.extra_cells(t, grid), tracker.fresh_cells(t, grid)
        self.assertLess(len(fresh), len(extra))
        self.assertIsNone(grid.plan((pose.x, pose.y), (5.0, 2.5), extra))     # 기억까지 넣으면 길이 없다
        job = ex.GoalExecutor(goal(5.0, 2.5, timeout_s=40.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, pose))
        job._plan(t, pose, grid, extra, tracker)
        self.assertTrue(job._path)
        self.assertTrue(job._fresh_path)
        self.assertFalse(job._through_obstacle)
        self.assertFalse(grid.path_blocked((pose.x, pose.y), job._path, fresh))   # 지금 보이는 물체는 비켜 간다

    def test_memory_only_cells_right_beside_the_robot_are_dropped(self):
        """몸 둘레 memory_near_m 안의 칸은 라이다가 「비었다」 를 확인하지 못한다 — 기억으로만 남은 것은 지운다 (지금 찍히는 것은 남긴다)."""
        p, spec, grid = make_world(boxes=())
        tracker = ObstacleTracker(p)
        side = (3.0, 2.5 - p.robot_half_width - 0.05)  # 오른쪽 옆 5 cm
        ranges, a0, inc = lidar_scan(Pose2D(3.0, 3.2, -math.pi / 2), [(side[0], side[1], 0.03)], p, spec)
        tracker.on_scan(0.0, Pose2D(3.0, 3.2, -math.pi / 2), scan_to_body(ranges, a0, inc, p), spec, grid, (a0, inc, np.asarray(ranges)))
        cell = grid.cell_of(side[0], side[1] + 0.03)
        self.assertIn(cell, tracker._seeds)
        pose = Pose2D(3.0, 2.5, math.pi)
        for k in range(1, 8):  # 그 곁에 와 섰다 — 그 칸은 이제 몸 바로 곁이라 「비었다」 를 볼 수 없다
            ranges, a0, inc = lidar_scan(pose, [], p, spec)
            tracker.on_scan(0.5 * k, pose, scan_to_body(ranges, a0, inc, p), spec, grid, (a0, inc, np.full(len(ranges), np.nan)))
        self.assertNotIn(cell, tracker._seeds)
        fresh = ObstacleTracker(p)
        for k in range(8):  # 지금도 찍히는 물체는 곁에 있어도 남는다
            body = [(0.0, -(p.robot_half_width + 0.05))]
            fresh.on_scan(0.5 * k, pose, body, spec, grid, (a0, inc, np.full(len(ranges), np.nan)))
        self.assertTrue(fresh._seeds)

    def test_timeout_while_turning_just_outside_the_position_tolerance_is_reached(self):
        """허용 안에서 돌기 시작해 도는 동안 몇 mm 밀린 채 시한이 되면 도착 (14:13 실차: 0.128 m · 허용 0.12 m 에서 TIMED_OUT)."""
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.pi / 2, pos_tol=0.12, timeout_s=5.0), p, 0.0)
        pose = Pose2D(3.0 - 0.128, 2.5, 0.0)
        self.assertTrue(job.validate(spec, grid, pose))
        job._state = "align"
        job.step(5.1, pose, loc.OK, grid, None)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn("[POS_NOT_MET]", job.detail)
        far = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.pi / 2, pos_tol=0.12, timeout_s=5.0), p, 0.0)
        far.validate(spec, grid, Pose2D(2.7, 2.5, 0.0))
        far._state = "align"
        far.step(5.1, Pose2D(2.7, 2.5, 0.0), loc.OK, grid, None)
        self.assertEqual(far.outcome, ex.TIMED_OUT, far.detail)



class DeadZoneRobot(SimRobot):
    """2026-11-19 실차 기록으로 잰 회전: 서서 도는 명령이 dead(1.4 rad/s)보다 작으면 안 돌고, 돌면 명령의 scale(0.35)배.
    명령은 delay(0.3 s, 실측 반응 0.13 ~ 0.5 s) 뒤에 듣는다."""

    def __init__(self, pose, dead=1.4, scale=0.35, delay=0.3):
        super().__init__(pose)
        self.dead, self.scale = dead, scale
        self.queue = [ex.Command()] * int(round(delay / DT))

    def apply(self, cmd, dt=DT):
        self.queue.append(cmd)
        cmd = self.queue.pop(0)
        w = 0.0 if abs(cmd.v) < 0.03 and abs(cmd.w) < self.dead else cmd.w * self.scale
        super().apply(ex.Command(cmd.v, w), dt)


class RealTurnTest(unittest.TestCase):
    """서서 돌기를 실차처럼 (작은 명령은 안 돎 · 명령보다 느림) — 2026-11-19 연습 경기 다섯 판의 기록: 0.8 ~ 1.4 rad/s 는 대부분 안 돌았다.
    전에는 남은 각이 작아져 명령이 1.4 rad/s 밑으로 내려가면 다시 서, 남은 10 ~ 20° 에서 방향 맞추기 상한까지 0.8 rad/s 를 냈다."""

    def _align(self, robot, yaw_deg, model=None, tol_deg=10.0, max_t=20.0):
        p, spec, grid = make_world(boxes=())
        job = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.radians(yaw_deg), yaw_tol=math.radians(tol_deg), timeout_s=max_t), p, 0.0, model)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=max_t)
        return job, trace

    def test_a_small_turn_is_not_left_stalled_below_the_dead_zone(self):
        robot = DeadZoneRobot(Pose2D(3.0, 2.5, 0.0))
        job, trace = self._align(robot, 25.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLessEqual(abs(wrap(robot.pose.yaw - math.radians(25.0))), math.radians(10.0) + 1e-6)
        self.assertLess(trace[-1][0], 6.0)  # 전: 방향 맞추기 상한(10 s)까지 0.8 rad/s 로 섰다
        self.assertGreaterEqual(job.turn.start_w, 1.3)  # 돌기 시작한 각속도를 쟀다

    def test_what_it_learned_carries_over_to_the_next_goal(self):
        model = ex.TurnModel()
        first, t1 = self._align(DeadZoneRobot(Pose2D(3.0, 2.5, 0.0)), 25.0, model)
        second, t2 = self._align(DeadZoneRobot(Pose2D(3.0, 2.5, 0.0)), 25.0, model)
        self.assertEqual(second.outcome, ex.REACHED, second.detail)
        self.assertIs(second.turn, model)
        self.assertLess(t2[-1][0], t1[-1][0])  # 처음부터 돌 만큼 낸다

    def test_a_large_turn_does_not_teach_a_high_start_speed(self):
        """남은 각이 커서 원래 1.6 rad/s 를 낸 회전에서는 「돌기 시작한 각속도」 를 배우지 않는다 (배우면 지연이 큰 차체에서 작은 각을 넘겨 돈다)."""
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        job, _trace = self._align(robot, 120.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertTrue(job.turn.start_w is None or job.turn.start_w <= job.p.min_turn_w + 0.1, job.turn.summary())

    def test_open_floor_turns_with_an_ideal_robot_are_unchanged(self):
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        job, trace = self._align(robot, 30.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertLess(trace[-1][0], 4.0)


class CornerAndNearGoalTest(unittest.TestCase):
    """두 상자가 만나는 오목한 모서리에 비스듬히 닿은 자세 · 허용을 몇 cm 넘어 선 자세 (2026-11-19 가상에서 나온 막다른 상태)."""

    def test_finds_a_way_out_of_a_concave_corner_it_touches(self):
        p, spec, grid = make_world(boxes=((4.305, 4.177), (4.693, 3.72)))
        for pose in (Pose2D(4.744, 4.09, math.radians(-51.6)), Pose2D(4.677, 4.079, math.radians(-63.4))):  # 틈 0.6 mm · 지도상 4 mm 겹침
            job = ex.GoalExecutor(goal(2.0, 2.0, timeout_s=30.0), p, 0.0)
            self.assertIsNotNone(job._room_move_plan(0.0, pose, grid, None, "structure"))  # 전: 비킬 곳이 없어 목표마다 곧바로 BLOCKED
            cmd = job._no_room(0.0, pose, grid, None, ex.NO_ROOM_TO_LEAVE, "stuck", 1.0)
            self.assertIsNone(job.outcome, job.detail)
            self.assertLess(cmd.v, 0.0)

    def test_does_not_turn_around_to_fix_a_few_centimetres(self):
        """허용 0.10 m 를 2 cm 넘어 섰고 목표가 등 뒤: 목표 쪽으로 180° 돌지 않고 그 자리에서 방향만 맞춘다 ([POS_NOT_MET])."""
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.12, 2.5, 0.0))
        job = ex.GoalExecutor(goal(3.0, 2.5, yaw=math.pi / 2, pos_tol=0.10, yaw_tol=0.3, timeout_s=20.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        job._state = "settle"
        trace = run(job, robot, spec, grid, p, max_t=20.0)
        self.assertEqual(job.outcome, ex.REACHED, job.detail)
        self.assertIn("[POS_NOT_MET]", job.detail)
        self.assertNotIn("driving", [ph for *_r, ph, _b in trace])
        self.assertLess(trace[-1][0], 6.0)


class TurningScanTest(unittest.TestCase):
    """빨리 도는 동안 스캔을 지도에 놓는 방향이 틀리던 것 (2026-11-19 밤, 21:53 실차): 찍힌 때의 자세 · 한 바퀴 동안의 회전 · 바퀴 방향의 배율."""

    def turning_tracker(self, wheel_gain=1.0, rate=1.0, seconds=4.0, learn=True):
        """rate(rad/s)로 도는 차체. 바퀴는 실제의 wheel_gain 배로 돌았다고 알린다. SLAM 위치는 0.1 s 마다 실제 방향으로 온다."""
        p = DriveParams()
        p.yaw_scale_learn = learn
        tr = loc.PoseTracker(p)
        left = right = 0.0
        t, yaw = 0.0, 0.0
        tr.on_wheels(t, left, right)
        tr.on_pose(t, t, Pose2D(1.0, 1.0, yaw))
        while t < seconds - 1e-9:
            t += 0.05
            yaw = wrap(yaw + rate * 0.05)
            half = wheel_gain * rate * 0.05 * p.wheel_base_m / 2.0
            left, right = left - half, right + half
            tr.on_wheels(t, left, right)
            if int(round(t / 0.05)) % 2 == 0:
                tr.on_pose(t, t, Pose2D(1.0, 1.0, yaw))
        return tr, t, yaw

    def test_pose_at_an_earlier_time_and_turn_rate(self):
        tr, t, yaw = self.turning_tracker(rate=1.0, seconds=2.0, learn=False)
        then = tr.estimate_at(t - 0.2)
        self.assertAlmostEqual(wrap(yaw - then.yaw), 0.2, delta=0.02)        # 0.2 s 앞의 방향은 0.2 rad 덜 돈 것
        self.assertAlmostEqual(tr.yaw_rate(t), 1.0, delta=0.05)
        self.assertAlmostEqual(tr.estimate_at(t + 1.0).yaw, tr.estimate().yaw, places=6)   # 앞날은 지금 자세

    def test_wheel_yaw_scale_is_learned_from_slam(self):
        """바퀴가 실제의 2배로 돌았다고 알리면(스키드 조향) SLAM 위치와 견줘 곱하는 값이 0.5 쪽으로 내려간다. 맞게 알리면 1.0 그대로."""
        tr, _t, _yaw = self.turning_tracker(wheel_gain=2.0, seconds=12.0)
        self.assertGreater(tr.yaw_scale_updates, 3)
        self.assertLess(tr.yaw_scale, 0.62)
        self.assertGreaterEqual(tr.yaw_scale, DriveParams().yaw_scale_min)
        tr, _t, _yaw = self.turning_tracker(wheel_gain=1.0, seconds=12.0)
        self.assertAlmostEqual(tr.yaw_scale, 1.0, delta=0.03)
        tr, _t, _yaw = self.turning_tracker(wheel_gain=2.0, seconds=12.0, learn=False)
        self.assertEqual(tr.yaw_scale, 1.0)

    def test_scale_is_not_learned_when_slam_disagrees_wildly(self):
        """SLAM 이 그동안 거의 안 돌았다고 하면(정합이 틀린 때) 그 창은 버린다."""
        p = DriveParams()
        tr = loc.PoseTracker(p)
        left = right = 0.0
        tr.on_wheels(0.0, left, right)
        tr.on_pose(0.0, 0.0, Pose2D(1.0, 1.0, 0.0))
        t = 0.0
        while t < 6.0:
            t += 0.05
            left, right = left - 0.05 * p.wheel_base_m / 2.0, right + 0.05 * p.wheel_base_m / 2.0
            tr.on_wheels(t, left, right)
            if int(round(t / 0.05)) % 2 == 0:
                tr.on_pose(t, t, Pose2D(1.0, 1.0, 0.0))
        self.assertEqual(tr.yaw_scale, 1.0)

    def test_scan_is_unwound_over_its_sweep(self):
        """한 바퀴 동안 차체가 왼쪽으로 0.1 rad 돌았다: 첫 광선의 점은 끝 광선 때의 차체에서 보면 오른쪽으로 0.1 rad 돌아가 있다. 끝 광선의 점은 그대로."""
        p = DriveParams()
        p.lidar_yaw_deg, p.lidar_x = 0.0, 0.0
        n = 101
        ranges = [float('nan')] * n
        ranges[0] = ranges[n - 1] = 2.0
        inc = 0.01
        plain = scan_to_body(ranges, 0.0, inc, p)
        fixed = scan_to_body(ranges, 0.0, inc, p, sweep=(0.0, 0.0, -0.1))
        self.assertAlmostEqual(math.atan2(fixed[0][1], fixed[0][0]), math.atan2(plain[0][1], plain[0][0]) - 0.1, places=6)
        self.assertAlmostEqual(fixed[1][0], plain[1][0], places=9)
        self.assertAlmostEqual(fixed[1][1], plain[1][1], places=9)
        self.assertAlmostEqual(math.hypot(*fixed[0]), 2.0, places=6)

    def test_turn_command_is_raised_when_the_body_does_not_start_turning(self):
        """가는 도중 서서 돌라고 했는데 안 돈다(정지 마찰): breakaway_hold_s 뒤부터 rotate_kick_w 까지 올린다. 도는 차체에는 올리지 않는다."""
        p, spec, grid = make_world(boxes=())
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        robot.stuck = True
        job = ex.GoalExecutor(goal(1.0, 2.5, timeout_s=20.0), p, 0.0)     # 목표가 바로 뒤
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=2.4)
        turning = [(t, abs(cmd.w)) for t, _pose, cmd, ph, _b in trace if ph == "rotating"]
        self.assertTrue(all(w <= p.rotate_max_w + 1e-6 for t, w in turning if t < p.breakaway_hold_s))
        self.assertGreater(max(w for _t, w in turning), p.rotate_max_w + 0.3)
        self.assertLessEqual(max(w for _t, w in turning), p.rotate_kick_w + 1e-6)
        robot = SimRobot(Pose2D(3.0, 2.5, 0.0))
        job = ex.GoalExecutor(goal(1.0, 2.5, timeout_s=20.0), p, 0.0)
        self.assertTrue(job.validate(spec, grid, robot.pose))
        trace = run(job, robot, spec, grid, p, max_t=6.0)
        self.assertLessEqual(max(abs(cmd.w) for _t, _pose, cmd, ph, _b in trace if ph == "rotating"), p.rotate_max_w + 1e-6)


if __name__ == "__main__":
    unittest.main()
