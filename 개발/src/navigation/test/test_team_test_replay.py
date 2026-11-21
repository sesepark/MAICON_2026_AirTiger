"""2026-11-18 포털 팀 테스트(연습 경기장 1번, R-2)에서 본 주행 문제를 가상 경기장에서 재현한다.

기록: 로그_파일/1118_팀테스트_경기장1_R2/ (상황.md · supervisor/events.jsonl · navigation.log), 정리: 시험기록/2026-11-18_팀테스트_재현/README.md
방향 맞추기 멈춤 둘은 6fa5606(executor._align)으로 고쳐 2026-11-19 expectedFailure 를 지웠다. 상자 스침은 4a5e3d0(grid.py)으로 고쳤다.
거점 진입과 다른 UGV 세 가지는 2026-11-19 고침(속도에 맞춘 정지 · 감속, 닿으면 물러나기, 그 물체를 지나야만 갈 수 있으면 시한 전에 BLOCKED)의 회귀 시험이다.

가상 경기장
- 지도 = 그 판(2차) perception 지도의 상자 13개. 고정 기둥 10개는 perception 지도에 없다 (벽 쪽 끝 띠 안을 내지 않는다).
  2026-11-19 부터 주행 · 위치 추정 노드가 규칙의 기본 배치를 스스로 더한다 (drive/fieldmap.py fixed_pillars, 파라미터 fixed_pillars)
- 실차 흉내 sim_world.REAL_ROBOT [추정]: 명령 반응 지연 0.30 s · 1차 지연 0.20 s, 서서 도는 명령은 0.45 s 이어져야 돈다
- 차체는 250 × 230 mm 사각형으로 부딪힘을 본다
- 목표의 시한은 supervisor(4d14876)와 같게 min(설정, 15 s + 1 m 에 10 s)
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '시험기록' / 'web_drive'))
import sim_world as W  # noqa: E402

REACHED, BLOCKED, TIMED_OUT = 1, 3, 4
# 2차(22:23) perception 지도의 상자 (supervisor/events.jsonl 의 map)
BOXES_1118_2 = ((1.406, 3.923), (1.834, 3.894), (4.305, 4.177), (3.068, 3.745), (4.693, 3.72), (3.301, 2.691), (1.827, 2.479),
                (2.792, 1.62), (3.228, 1.62), (1.712, 1.217), (4.194, 1.192), (1.737, 0.861), (4.2, 0.804))
B1_ASSAULT = (0.175, 3.817)  # 두 판 모두 M5 로 보낸 BLUE_COMMAND_POST 안의 돌진 목표
# 3차(23:39) perception 지도의 상자 (로그_파일/1118_팀테스트_경기장1_R2/3차_2339/supervisor/events.jsonl 의 map)
BOXES_1118_3 = ((1.401, 3.931), (1.844, 3.957), (4.295, 4.169), (4.719, 4.169), (3.319, 3.718), (4.734, 3.756), (3.071, 2.78), (1.825, 2.483),
                (2.803, 1.501), (3.216, 1.528), (1.336, 1.159), (1.719, 1.176), (4.177, 1.218), (4.154, 0.831))


def start(slot='R-2', vehicles=(), seed=1, vehicle_seen=W.UGV_SEEN, boxes=BOXES_1118_2):
    world = W.World(boxes=boxes, hidden=W.PILLARS, vehicles=vehicles, vehicle_seen=vehicle_seen, **W.ARENA)
    robot = W.SimRobot(world, W.slot_start_pose(slot), odom_offset=1.0, slip=0.85, seed=seed, **W.REAL_ROBOT)
    loop = W.ClosedLoop(world, robot)
    loop.start_from_slot(slot, W.base_region(slot))
    loop.run(1.5)
    return loop


def go(loop, x, y, yaw_deg=None, yaw_tol=0.10, cap_s=60.0, mode=0):
    r = loop.robot
    timeout = min(cap_s, 15.0 + math.hypot(x - r.x, y - r.y) / 0.10)
    return loop.drive_to(x, y, limit_s=timeout + 5.0, yaw=None if yaw_deg is None else math.radians(yaw_deg), yaw_tol=yaw_tol,
                         timeout_s=timeout, mode=mode)


def drive_with(loop, goal, mode, timeout_s, move=None, speed=0.3, each=None):
    """목표를 보내고 끝날 때까지 돌린다. move(robot) 가 자리를 주면 그때부터 다른 UGV(vehicles[0])를 speed 로 그리로 옮긴다.
    each(robot) 는 주기마다 부른다."""
    loop.allowed = True
    handle, thread, box = loop.send_goal(goal[0], goal[1], yaw=math.pi, mode=mode, timeout_s=timeout_s)
    job = loop.job_of(handle)
    loop.drive_started = loop.t
    target = None
    while job is not None and not job.done and loop.t < loop.drive_started + timeout_s + 5.0:
        r = loop.robot
        if move is not None and target is None:
            target = move(r)
        if target is not None:
            x, y = loop.world.vehicles[0]
            dx, dy = target[0] - x, target[1] - y
            d = math.hypot(dx, dy)
            step = min(d, speed * loop.DT)
            if d > 1e-9:
                loop.world.vehicles[0] = (x + dx / d * step, y + dy / d * step)
        if each is not None:
            each(r)
        loop.step()
    thread.join(1.0)
    return box.get('result')


def heading_error_deg(loop, yaw_deg):
    e = loop.robot.yaw - math.radians(yaw_deg)
    return abs(math.degrees(math.atan2(math.sin(e), math.cos(e))))


class TeamTestReplay(unittest.TestCase):
    def _turn_in_place(self, left_deg, yaw_tol):
        """출발 칸에 선 채로 left_deg 만큼만 돌라는 목표 (자리는 그대로)."""
        loop = start()
        r = loop.robot
        target = math.degrees(r.yaw) + left_deg
        result = go(loop, r.x, r.y, target, yaw_tol=yaw_tol, cap_s=20.0)
        return loop, result, target

    def test_turns_the_last_few_degrees_at_a_scout_spot(self):
        """1차 첫 정찰 자리: 3.6 cm 앞까지 와서 방향 11° 를 남기고 3분을 서 있었다 (/cmd_vel 회전 0.8 과 0 을 짧게 되풀이).
        허용 0.10 rad 보다 조금 더(12°) 돌아야 할 때: 가상에서도 0.3 s 회전 · 정지를 되풀이하며 그대로 TIMED_OUT."""
        loop, result, target = self._turn_in_place(12.0, 0.10)
        self.assertEqual(result.outcome, REACHED, result.detail)
        self.assertLess(heading_error_deg(loop, target), math.degrees(0.10) + 3.0)

    def test_turns_with_the_wide_tolerance_of_the_second_run(self):
        """2차는 방향 허용을 0.5 rad(29°)로 넓혔는데도 33 ~ 40° 를 남기고 30 s 시한을 썼다.
        허용보다 조금 더(38°) 돌아야 할 때: 가상에서도 그대로 TIMED_OUT."""
        loop, result, target = self._turn_in_place(38.0, 0.5)
        self.assertEqual(result.outcome, REACHED, result.detail)

    # ---- 거점 진입(M5) 과 다른 UGV — 드론 사진(로봇의 server_bridge/drone/, 10 s 간격)으로 본 것:
    #   1차: 다른 UGV 가 B-1 안 (0.40, 4.13) 에 끝까지 서 있었고, 우리는 그 앞 약 0.3 m(정지 범위와 같다)에서 닿지 않고 멈춘 채 50 s
    #   2차: 그 UGV 가 우리가 들어가던 10 s 사이에 입구 쪽 (0.54, 3.88) 으로 나왔고, 우리는 0.6 m/s 로 들어가다 닿아(2.5 cm 튕김) 30 s × 2
    #   → 라이다는 서 있는 UGV 를 봤다. 문제는 ① 정지 범위가 속도와 상관없이 0.30 m ② 막힌 채 시한까지 기다리기만 함 (2026-11-19 고침)

    def test_assault_into_an_occupied_base_does_not_touch_and_gives_up_early(self):
        """1차처럼 목표 자리에 다른 UGV 가 서 있다. 고치기 전: 돌진 0.6 m/s 로 지도에 없는 벽 기둥을 끼고 돌다 옆이 닿고 시한(28 s)까지 기다림.
        고친 뒤: 닿지 않고, 그 UGV 를 지나야만 갈 수 있으니 obstacle_give_up_s 뒤 시한 전에 BLOCKED."""
        loop = start(slot='B-2', vehicles=[(0.30, 3.90)])
        began = loop.t
        result = go(loop, *B1_ASSAULT, 180.0, mode=2)
        self.assertEqual(loop.robot.hidden_contacts, 0, '지도에 없는 기둥 · UGV 에 닿음')
        self.assertEqual(result.outcome, BLOCKED, result.detail)
        self.assertLess(loop.t - began, 15.0)

    def test_passes_a_standing_ugv_in_a_doorway_slowly(self):
        """상자 벽의 0.9 m 문에 다른 UGV 가 비켜 서 있다 (남은 틈: 라이다에 보이는 면에서 0.49 ~ 0.54 m). 2026-11-19 실차 16:01 · 16:55 처럼
        서 있는 UGV 곁에서 「지도에 없는 물체가 길을 막음」 으로 포기하던 것 — 서 있는 물체만 tight_radius 로 줄여 다시 짜고 천천히 지나간다.
        끄면(tight_radius 0) 전처럼 BLOCKED 다."""
        wall = tuple((4.5, y) for y in (0.25, 0.65, 1.05, 1.45, 1.85, 3.15, 3.55, 3.95, 4.35, 4.75))

        def run(ugv_y, seen, tight):
            world = W.World(boxes=wall, hidden=W.PILLARS, vehicles=[(4.5, ugv_y)], vehicle_seen=seen, **W.ARENA)
            robot = W.SimRobot(world, W.slot_start_pose('R-2'), odom_offset=1.0, slip=0.85, seed=1, **W.REAL_ROBOT)
            loop = W.ClosedLoop(world, robot, drive_params={'tight_radius': tight})
            loop.start_from_slot('R-2', W.base_region('R-2'))
            loop.run(1.5)
            return go(loop, 3.0, 2.5), robot

        for ugv_y, seen in ((2.40, W.UGV_SEEN), (2.35, 0.23)):      # 가운데 12 cm 만 보이는 UGV · 몸 전체가 보이는 UGV
            with self.subTest(ugv_y=ugv_y, seen=seen):
                result, robot = run(ugv_y, seen, 0.0)
                self.assertEqual(result.outcome, BLOCKED, result.detail)
                result, robot = run(ugv_y, seen, 0.10)
                self.assertEqual(result.outcome, REACHED, result.detail)
                self.assertEqual(robot.hidden_contacts, 0, '서 있는 UGV · 기둥에 닿음')

    def test_does_not_hit_a_ugv_coming_out_to_the_base_entrance(self):
        """2차처럼 우리가 들어가는 중에 다른 UGV 가 입구 쪽으로 0.3 m/s 로 나온다. 고치기 전: 0.59 m/s 로 닿고 시한까지.
        고친 뒤: 다가오는 것을 보고 줄여 서고, 시한 전에 BLOCKED.

        2026-11-19: 기둥이 지도에 들어가 길이 달라졌다 — 기둥을 크게 돌지 않고 곧게 올라가 입구 바로 앞에 먼저 와 선다. 이 시험의 상대 UGV 는
        우리가 서 있어도 정해진 자리 (0.50, 3.80) 까지 밀고 들어오므로(피할 수 없다) 닿는 것 자체는 막을 수 없다. 그래서 보는 것은
        「우리가 앞으로 가는 중에 닿지 않는다」 — 닿는 순간 우리는 서 있었거나 물러나는 중이어야 한다."""
        loop = start(slot='B-2', vehicles=[(0.85, 4.45)], vehicle_seen=0.23)
        state = {'seen': 0, 'while_driving': 0}

        def each(r):
            if r.hidden_contacts > state['seen'] and loop.cmd[0] > 0.01:
                state['while_driving'] += 1
            state['seen'] = r.hidden_contacts

        result = drive_with(loop, B1_ASSAULT, mode=2, timeout_s=28.0,
                            move=lambda r: (0.50, 3.80) if r.y > 2.75 else None, speed=0.3, each=each)
        self.assertEqual(state['while_driving'], 0, '앞으로 가는 중에 나오던 UGV 에 닿음')
        self.assertEqual(result.outcome, BLOCKED, result.detail)

    # ---- 고정 기둥 · 거점 진입 (2026-11-19): 지도에 없어 「모르는 물체」 로 보던 기둥을 주행 · 위치 추정이 스스로 지도에 더한다

    def _third_run(self, pillars, seed=1, **drive_params):
        world = W.World(boxes=BOXES_1118_3, hidden=W.PILLARS, **W.ARENA)
        robot = W.SimRobot(world, W.slot_start_pose('R-2'), odom_offset=1.0, slip=0.85, seed=seed, **W.REAL_ROBOT)
        loop = W.ClosedLoop(world, robot, drive_params=dict(drive_params, fixed_pillars=pillars))
        loop.start_from_slot('R-2', W.base_region('R-2'))
        loop.run(1.5)
        return loop

    def test_goal_beside_the_top_wall_pillar(self):
        """3차의 세 번째 막힘: (3.83, 3.62) 에서 위쪽 가운데 기둥 옆의 정찰 자리 (2.62, 4.62) 로. 실차는 0.47 m 앞에서 15 s 를 서 있다 BLOCKED.
        기둥이 지도에 없으면 가상에서도 「지도에 없는 물체가 길을 막음」 으로 0.7 m 앞에서 끝난다 (기둥의 라이다 점마다 0.35 m 를 막아 목표 칸이 그 안).
        기둥을 지도에 넣으면 도착하고, 기둥 둘레에 임시 장애물이 생기지 않는다.

        2026-11-19 밤: 그 물체를 지나야만 갈 수 있는 길에서는 0.70 m 앞에 선다(through_standoff_m) — 그러면 기둥이 지도에 없어도 닿지 않고 도착한다
        (기둥에 0.30 m 까지 붙기 전에 서서, 오래 서 있던 점을 작게 보고 다시 짠 길이 열린다). 0.30 m 까지 가던 전 흐름에서는 BLOCKED 그대로다."""
        loop = self._third_run(pillars=False, through_standoff_m=0.0, obstacle_give_up_static_s=0.0, goal_shift_m=0.0)
        self.assertEqual(go(loop, 3.83, 3.62, 180.0, yaw_tol=0.6).outcome, REACHED)
        result = go(loop, 2.62, 4.62)
        self.assertEqual(result.outcome, BLOCKED, result.detail)

        # 2026-11-20: 서 있는 물체가 목표 칸을 덮으면 곁의 빈 칸으로 옮겨 간다 (goal_shift_m) — 그 전 흐름에서도 기둥 곁 0.15 m 옆 자리에 닿는다
        loop = self._third_run(pillars=False, through_standoff_m=0.0, obstacle_give_up_static_s=0.0)
        self.assertEqual(go(loop, 3.83, 3.62, 180.0, yaw_tol=0.6).outcome, REACHED)
        result = go(loop, 2.62, 4.62)
        self.assertEqual(result.outcome, REACHED, result.detail)
        self.assertIn("옮겨 감", result.detail)

        # 지금 기본(0.70 m 앞에 선다): 닿지 않는다. 도착하느냐는 그 자리에 온 때에 달렸다 — 오래 서 있던 점을 작게 보고 짠 길이 열린 뒤면 도착,
        # 그 전이면 「서 있는 물체가 길을 막음」 으로 2 s 에 끝난다 (회전 상한 1.6 에서는 도착, 2.2 에서는 먼저 와서 BLOCKED)
        loop = self._third_run(pillars=False)
        self.assertEqual(go(loop, 3.83, 3.62, 180.0, yaw_tol=0.6).outcome, REACHED)
        result = go(loop, 2.62, 4.62)
        self.assertIn(result.outcome, (REACHED, BLOCKED), result.detail)
        self.assertEqual(loop.robot.hidden_contacts, 0, '지도에 없는 기둥에 닿음')

        loop = self._third_run(pillars=True)
        self.assertEqual(go(loop, 3.83, 3.62, 180.0, yaw_tol=0.6).outcome, REACHED)
        result = go(loop, 2.62, 4.62)
        self.assertEqual(result.outcome, REACHED, result.detail)
        self.assertEqual((loop.robot.collisions, loop.robot.hidden_contacts), (0, 0))
        self.assertEqual(loop.drive.obstacles.snapshot(loop.drive._now(), loop.drive.grid), [])

    def test_assault_into_an_empty_base_does_not_ram_or_turn_against_the_wall(self):
        """빈 거점으로 비스듬히 돌진. 고치기 전(2026-11-19 가상): 목표 0.15 m 앞까지 0.6 m/s 로 가다 벽을 들이받거나, 목표 7 cm 옆에 와서
        3 cm 안으로 가려고 벽에 붙은 채 서서 돌다 모서리가 걸려 시한(27 s)까지 갔다 (22번 중 7번).
        고친 뒤: 0.5 m 앞부터 줄이고, 도착 허용 안이면 서서 돌지 않고 선다 → 몇 초 안에 REACHED, 벽에 오래 닿아 있지 않는다."""
        for stage, entry in (((1.2, 3.2), (0.20, 3.97)), ((0.60, 2.50), (0.20, 3.97)), ((0.60, 2.50), (0.20, 1.03))):
            world = W.World(boxes=BOXES_1118_2, hidden=W.PILLARS, **W.ARENA)
            robot = W.SimRobot(world, W.slot_start_pose('R-2'), odom_offset=1.0, slip=0.85, **W.REAL_ROBOT)
            loop = W.ClosedLoop(world, robot)
            loop.start_from_slot('R-2', W.base_region('R-2'))
            loop.run(1.5)
            self.assertEqual(go(loop, stage[0], stage[1], 180.0, yaw_tol=0.5).outcome, REACHED)
            began = loop.t
            result = loop.drive_to(entry[0], entry[1], limit_s=35.0, yaw=None, yaw_tol=3.15, timeout_s=30.0, mode=2)
            self.assertEqual(result.outcome, REACHED, (stage, entry, result.detail))
            self.assertLess(loop.t - began, 9.0, (stage, entry))
            self.assertLess(robot.x, 0.30, (stage, entry))  # 거점 깊숙이 (차체 앞이 벽에서 0.18 m 안)
            self.assertLess(robot.collisions, 20, (stage, entry))  # 벽에 닿은 채 버티지 않는다 (20 Hz 로 1 s 미만)
            self.assertEqual(robot.hidden_contacts, 0, (stage, entry))  # 거점 옆 기둥에는 닿지 않는다
            pose = loop.poses[-1]
            self.assertLess(math.hypot(pose[1] - robot.x, pose[2] - robot.y), 0.06, '거점 안에서 위치 추정이 벗어남')

    def test_final_turn_does_not_stall_just_outside_the_tolerance(self):
        """도착 방향 맞추기(6fa5606): 남은 각도가 허용보다 조금 큰 곳에서 돌기도 전에 멈추기를 되풀이하던 것.
        허용 6° · 20° · 34° 각각 그 구간의 각도에서 모두 도착해야 한다 (고치기 전에는 13개 중 3개씩 시한까지).
        회전 시도 3번 뒤 20° 안이면 허용 밖이어도 REACHED 로 끝내므로, 끝의 오차는 「허용 + 3°」 와 20° 가운데 큰 쪽까지 본다."""
        for tol, angles in ((0.10, (8.0, 12.0, -12.0)), (0.35, (30.0, 38.0, -38.0)), (0.60, (38.0, 50.0, -38.0))):
            for left in angles:
                loop, result, target = self._turn_in_place(left, tol)
                self.assertEqual(result.outcome, REACHED, (tol, left, result.detail))
                self.assertLess(heading_error_deg(loop, target), max(math.degrees(tol) + 3.0, 20.5), (tol, left))

    def test_backs_off_after_another_ugv_pushes_in(self):
        """앞에서 기다리는데 다른 UGV 가 밀고 들어와 닿았다 (피할 수 없다). 고치기 전: 닿은 채 시한까지.
        고친 뒤: 상대가 멈추면 물러나 떨어지고, 시한 전에 BLOCKED."""
        wall = [(1.8, y) for y in (0.2, 0.6, 1.0, 1.4, 1.8, 2.2)] + [(1.8, y) for y in (3.0, 3.4, 3.8, 4.2, 4.6)]  # 틈 y 2.4 ~ 2.8
        world = W.World(boxes=wall, hidden=W.PILLARS, vehicles=[(1.8, 2.6)], vehicle_seen=0.23, **W.ARENA)
        robot = W.SimRobot(world, W.slot_start_pose('B-2'), odom_offset=1.0, slip=0.85, **W.REAL_ROBOT)
        loop = W.ClosedLoop(world, robot)
        loop.start_from_slot('B-2', W.base_region('B-2'))
        loop.run(1.5)
        state = {}

        def push(r):  # 우리가 서서 기다리기 시작하면 4 s 동안 0.3 m/s 로 닿을 때까지 밀고 들어온다
            job = loop.drive._job
            if 'since' not in state and job is not None and job.phase == 'waiting_obstacle' and r.v < 0.02:
                state['since'] = loop.t
            if 'since' in state and loop.t - state['since'] < 4.0:
                if world.body_clearance(r.x, r.y, r.yaw, walls=False, items=world.unmapped()) > 0.004:
                    x, y = world.vehicles[0]
                    world.vehicles[0] = (x - 0.3 * loop.DT, y)
                else:
                    state['touched'] = True

        result = drive_with(loop, (3.0, 2.6), mode=0, timeout_s=20.0, each=push)
        self.assertTrue(state.get('touched'), '재현이 안 됨: 상대가 닿지 않았다')
        r = loop.robot
        self.assertGreater(world.body_clearance(r.x, r.y, r.yaw, walls=False, items=world.unmapped()), 0.05, '닿은 채로 끝남')
        self.assertEqual(result.outcome, BLOCKED, result.detail)
        self.assertLess(loop.t - loop.drive_started, 15.0)

    def test_does_not_graze_a_box_on_the_way_to_the_scout_spots(self):
        """2차의 정찰 자리 순서대로 간다. 전에는 가상(seed 2)에서 (2.62, 3.42) 로 가며 0.42 m/s 로 도는 중에 차체 옆이 상자 (3.07, 3.75) 모서리를 스쳤다.
        2026-11-19: 격자의 줄 당기기가 넓은 길을 다시 모서리에 붙이지 않게 고친 뒤로 닿지 않는다 (expectedFailure 를 지움 — drive/grid.py 의 plan)."""
        loop = start(seed=2)
        for x, y in ((4.225, 2.225), (4.025, 1.825), (3.625, 1.025), (2.625, 2.625), (2.825, 2.825), (4.025, 3.625), (2.625, 3.425)):
            go(loop, x, y, cap_s=40.0)
        self.assertEqual(loop.robot.collisions, 0, '상자에 닿음 (차체 최소 거리 %.3f m)' % loop.robot.min_body_clearance)


    # ---- 서서 돌다 지도 상자 · 벽 · 기둥에 박힘 (2026-11-19, executor 의 _structure_turn_hits · _turn_stuck · _no_room)
    #   차체 반 대각선 0.170 = 격자 부풀림 0.17 이라, 상자 곁에서 다음 목표로 가려고 서서 돌면 모서리가 걸린다. 돌기 전 확인은 지도에 없는 점만 봤고
    #   끼임 감지는 앞뒤로 갈 때만 켜져, 그 자리에서 1.6 rad/s 회전 명령을 시한까지 냈다. 그 판의 뒤 목표도 같은 자리에서 줄줄이 시한이었다

    def test_does_not_get_pinned_turning_beside_a_box(self):
        """3차 지도 · seed 3 (임시_테스트_파일/navigation_줄당기기_검토/loop_compare.py 의 무작위 자리). 고치기 전: 둘째 자리 (1.843, 2.906) 에
        도착한 뒤 셋째로 가려고 서서 돌다 0.35 s 만에 상자 (1.825, 2.483) 에 모서리가 걸려, 셋째부터 다섯 목표가 모두 시한 (닿은 주기 3,771).
        고친 뒤: 돌기 전에 차체를 돌려 보아 걸리면 비켜 돈다 → 모두 도착."""
        loop = start(seed=3, boxes=BOXES_1118_3)
        results = [go(loop, x, y, cap_s=40.0) for x, y in ((1.814, 3.449), (1.843, 2.906), (2.429, 1.214), (3.799, 1.556))]
        r = loop.robot
        self.assertEqual([res.outcome for res in results], [REACHED] * 4, [res.detail for res in results])
        self.assertLess(r.collisions - r.hidden_contacts, 20, '지도 상자 · 벽에 닿은 주기')


if __name__ == '__main__':
    unittest.main()
