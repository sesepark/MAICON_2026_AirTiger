"""가상 경기장 + 실제 노드 코드(navigation_slam · navigation)를 ROS 없이 묶어 돌린다.

쓰는 곳
- test/test_closed_loop.py : SLAM 과 주행 노드를 같이 돌리는 닫힌 루프 시험
- web_drive.py --sim       : 로봇 없이 PC 에서 웹 화면을 써 보기

로봇 흉내 (2026-11-18 실측에 맞춤)
- /odom 10 Hz: orientation 은 항상 (0,0,0,1), x·y 는 바퀴 기준 내부 방향으로 적분 (1 cm 단위)
- /odom/odom_raw 20 Hz: [왼쪽, 오른쪽] 누적 거리 (1 cm 단위)
- /scan 10 Hz: base_lidar_link (x 0.0398 m, +90°), 505 점, 뒤쪽 90° 는 가려짐, 잡음 1 cm
- 명령 반응: 지연 + 1차 지연. slip < 1 이면 바퀴가 돈 것보다 차체가 덜 돈다 (스키드 조향)
- turn_breakaway_s > 0: 서서 도는 명령이 이만큼 이어져야 돌기 시작한다 (2026-11-18 팀 테스트: 0.3 초씩 끊긴 회전 명령으로는 돌지 않았다)
- 부딪힘은 차체 사각형(250 × 230 mm)으로 본다. 벽 · 상자 · 숨긴 물체에 닿으면 그쪽으로 더 가지 않고 collisions 를 센다
"""
import math
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE / 'test') not in sys.path:
    sys.path.insert(0, str(PACKAGE / 'test'))
import test_node_split as fake  # noqa: E402  가짜 rclpy · 메시지

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena_default import ARENA, PILLARS, base_region, slot_start_pose  # noqa: E402,F401

__all__ = ['ARENA', 'PILLARS', 'SAMPLE_BOXES', 'UGV_SIZE', 'UGV_SEEN', 'REAL_ROBOT', 'REAL_ROBOT_1119', 'World', 'SimRobot', 'ClosedLoop', 'base_region', 'slot_start_pose']

WHEEL_BASE = 0.175
LIDAR_X = 0.0398
SCAN_POINTS = 505
ROBOT_RADIUS = 0.13   # 차체 가운데가 벽·상자에서 이만큼은 떨어져 있어야 한다 (차체 250 × 230 mm 의 옆면 0.115 m + 여유)
HALF_LENGTH, HALF_WIDTH = 0.125, 0.115   # 차체 250 × 230 mm
UGV_SIZE = (0.25, 0.23)   # 다른 팀 UGV 의 차체 (부딪히는 몸)
# 다른 UGV 중 라이다 높이에 걸리는 부분 (가운데 정사각형 한 변) [추정]. 라이다는 차체 위에 있어 같은 기종의 차체 옆면은 거의 안 보이고
# 위에 얹힌 라이다 · 팬틸트만 보일 수 있다 (우리 스캔도 0.12 m 안의 점은 우리 몸으로 지운다 — scan_self_mask_m).
UGV_SEEN = 0.12
# 실차 흉내 [추정] — 2026-11-18 실측(명령 반응 0.13~0.5 s, 정지 후 0.14~0.26 s 더 감)과 팀 테스트(0.3 초씩 끊긴 회전 명령으로는 안 돎)에 맞춘 값.
# 이것을 넣어야 팀 테스트의 「정찰 자리에서 방향을 못 맞춤」 · 「돌진 중 옆이 닿음」이 재현된다 (test/test_team_test_replay.py)
REAL_ROBOT = dict(delay_s=0.30, lag_s=0.20, turn_breakaway_s=0.45)
# 2026-11-19 실차 기록(연습 경기 다섯 판, navigation/debug.jsonl 0.5 s 위치)으로 잰 회전 [실측]: 서서 도는 명령 0.8 · 1.2 · 1.4 rad/s 는
# 표본의 52 ~ 100 % 가 0.1 rad/s 도 못 돌았고(중앙 0.01 ~ 0.06), 1.6 rad/s 는 실제 중앙 0.39 · 위 사분위 0.63 rad/s (683 표본),
# 주행 중 1.5 ~ 2.0 rad/s 는 0.55 ~ 0.87 rad/s. 멈춘 뒤 더 돈 각도는 중앙 −1° · 최대 10° (128번).
# → 서서 돌 때 1.4 rad/s 밑은 안 돌고(turn_dead_w), 바퀴가 내는 각속도는 명령의 0.35 배(turn_scale). REAL_ROBOT 은 그대로 두고 따로 둔다
REAL_ROBOT_1119 = dict(REAL_ROBOT, turn_dead_w=1.4, turn_scale=0.35)


def body_outline(x, y, yaw, step=0.02):
    """차체 사각형의 둘레 점 (step 간격) → (xs, ys)."""
    corners = ((HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH))
    pts = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        n = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / step)))
        pts.extend((ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(n))
    body = np.array(pts)
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * body[:, 0] - s * body[:, 1], y + s * body[:, 0] + c * body[:, 1]


def stamp(t):
    sec = int(t)
    return NS(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def quat(yaw):
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


class World:
    """벽(0..width, 0..height)과 상자. 지도에는 없지만 실제로는 있는 것:
    hidden   = 숨긴 상자 (x, y) — 상자 크기의 정사각형, 또는 (x, y, 가로, 세로)
    vehicles = 다른 팀 UGV (x, y) — 몸은 UGV_SIZE, 라이다에는 가운데 vehicle_seen 정사각형만 보인다."""

    def __init__(self, width, height, box_size, boxes, hidden=(), vehicles=(), vehicle_seen=UGV_SEEN):
        self.width, self.height, self.box_size = float(width), float(height), float(box_size)
        self.boxes = [tuple(map(float, b)) for b in boxes]
        self.hidden = [tuple(map(float, b)) for b in hidden]
        self.vehicles = [tuple(map(float, v[:2])) for v in vehicles]
        self.vehicle_seen = float(vehicle_seen)

    def _rows(self, items):
        half = self.box_size / 2
        rows = [(b[0], b[1], b[2] / 2, b[3] / 2) if len(b) >= 4 else (b[0], b[1], half, half) for b in items]
        return np.array(rows, dtype=float).reshape(-1, 4)

    def unmapped(self):
        """지도에 없는 물체의 몸 [(x, y, 가로, 세로)] (숨긴 상자 + 다른 UGV)."""
        return self.hidden + [(x, y, *UGV_SIZE) for x, y in self.vehicles]

    def _solid(self):
        """부딪히는 몸 [(가운데 x, y, 반폭 x, 반폭 y)]."""
        return self._rows(self.boxes + self.unmapped())

    def _seen(self):
        """라이다에 걸리는 것."""
        return self._rows(self.boxes + self.hidden + [(x, y, self.vehicle_seen, self.vehicle_seen) for x, y in self.vehicles])

    def rays(self, x, y, angles):
        """(x, y) 에서 angles 방향으로 쏜 광선이 처음 닿는 거리."""
        dx, dy = np.cos(angles), np.sin(angles)
        best = np.full(len(angles), 25.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            for value, direction, limit in ((x, dx, self.width), (y, dy, self.height)):
                t = np.where(direction > 0, (limit - value) / direction, np.where(direction < 0, -value / direction, np.inf))
                best = np.minimum(best, np.where(t > 0, t, np.inf))
            solid = self._seen()
            if len(solid):
                near = np.full((len(angles), len(solid)), -np.inf)
                far = np.full((len(angles), len(solid)), np.inf)
                for value, direction, centre, half in ((x, dx, solid[:, 0], solid[:, 2]), (y, dy, solid[:, 1], solid[:, 3])):
                    a = (centre[None, :] - half[None, :] - value) / direction[:, None]
                    b = (centre[None, :] + half[None, :] - value) / direction[:, None]
                    near = np.maximum(near, np.minimum(a, b))
                    far = np.minimum(far, np.maximum(a, b))
                hit = (near <= far) & (near > 0)
                best = np.minimum(best, np.where(hit, near, np.inf).min(axis=1))
        return best

    def clearance(self, x, y):
        """(x, y) 에서 가장 가까운 벽·상자까지의 거리 (상자 안이면 0)."""
        return float(self.clearances(np.array([x], dtype=float), np.array([y], dtype=float))[0])

    def clearances(self, xs, ys, walls=True, items=None):
        """점마다 가장 가까운 벽·상자·숨긴 물체까지의 거리. 안에 들어가 있으면 음수(들어간 깊이). items 를 주면 그 물체만."""
        d = np.minimum.reduce([xs, self.width - xs, ys, self.height - ys]) if walls else np.full(len(xs), np.inf)
        solid = self._solid() if items is None else self._rows(items)
        if len(solid):
            ax = np.abs(xs[:, None] - solid[None, :, 0]) - solid[None, :, 2]
            ay = np.abs(ys[:, None] - solid[None, :, 1]) - solid[None, :, 3]
            outside = (ax > 0) | (ay > 0)
            box = np.where(outside, np.hypot(np.maximum(ax, 0.0), np.maximum(ay, 0.0)), np.maximum(ax, ay))
            d = np.minimum(d, box.min(axis=1))
        return d

    def body_clearance(self, x, y, yaw, walls=True, items=None):
        """차체 사각형에서 가장 가까운 벽·상자까지의 거리 (닿으면 0, 겹치면 음수 — 겹침이 줄어드는 쪽으로는 움직일 수 있게)."""
        return float(self.clearances(*body_outline(x, y, yaw), walls=walls, items=items).min())


class SimRobot:
    def __init__(self, world, pose, odom_offset=1.0, slip=1.0, delay_s=0.10, lag_s=0.12, seed=1,
                 encoder_m=0.01, noise_m=0.01, turn_breakaway_s=0.0, turn_dead_w=0.0, turn_scale=1.0):
        self.world = world
        self.turn_dead_w, self.turn_scale = turn_dead_w, turn_scale   # REAL_ROBOT_1119: 서서 돌 때 이 밑의 명령은 안 돎 · 각속도 배율
        self.x, self.y, self.yaw = pose
        self.internal = self.yaw - odom_offset   # 로봇 odom 노드의 내부 방향 (경기장 방향과 odom_offset 만큼 다르다)
        self.ox, self.oy = 2.0, -1.0
        self.left, self.right = 3.0, -1.0
        self.slip, self.delay_s, self.lag_s = slip, delay_s, lag_s
        self.turn_breakaway_s = turn_breakaway_s
        self._turn_since = None        # 서서 도는 명령이 이어지기 시작한 시각
        self.encoder_m, self.noise_m = encoder_m, noise_m
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.v = self.w = 0.0          # 지금 바퀴가 내는 속도
        self._queue = []               # (적용 시각, v, w)
        self.t = 0.0
        self.collisions = 0            # 차체가 벽·상자·지도에 없는 물체에 닿아 더 못 간 주기 수
        self.hidden_contacts = 0       # 그중 지도에 없는 물체(숨긴 상자 · 다른 UGV)에 닿은 주기 수
        self.min_clearance = float('inf')        # 차체 가운데에서 벽·상자까지의 최소 거리
        self.min_body_clearance = float('inf')   # 차체 사각형에서 벽·상자까지의 최소 거리

    def command(self, v, w):
        self._queue.append((self.t + self.delay_s, v, w))

    def step(self, dt):
        self.t += dt
        target = None
        while self._queue and self._queue[0][0] <= self.t:
            target = self._queue.pop(0)
        if target is not None:
            self._target = (target[1], target[2])
        tv, tw = getattr(self, '_target', (0.0, 0.0))
        if self.turn_breakaway_s > 0 and abs(tv) < 0.03 and abs(tw) > 1e-6:
            if self._turn_since is None:
                self._turn_since = self.t
            if self.t - self._turn_since < self.turn_breakaway_s and abs(self.w) < 0.05:
                tw = 0.0               # 아직 안 돈다 (서서 도는 마찰)
        else:
            self._turn_since = None
        if self.turn_dead_w > 0 and abs(tv) < 0.03 and abs(tw) < self.turn_dead_w:
            tw = 0.0                   # 서서 도는 명령이 작으면 안 돈다 (모터 · 바닥 마찰)
        tw *= self.turn_scale
        k = 1.0 if self.lag_s <= 0 else min(1.0, dt / self.lag_s)
        self.v += (tv - self.v) * k
        self.w += (tw - self.w) * k
        v, w = self.v, self.w
        # 바퀴는 명령대로 돈다. 차체는 slip 만큼만 돈다
        self.left += (v - w * WHEEL_BASE / 2) * dt
        self.right += (v + w * WHEEL_BASE / 2) * dt
        self.internal += w * dt
        self.ox += v * dt * math.cos(self.internal)
        self.oy += v * dt * math.sin(self.internal)
        nyaw = self.yaw + w * self.slip * dt
        nx = self.x + v * dt * math.cos(nyaw)
        ny = self.y + v * dt * math.sin(nyaw)
        after = self.world.body_clearance(nx, ny, nyaw)
        if after <= 0.0 and after < self.world.body_clearance(self.x, self.y, self.yaw) + 1e-9:
            self.collisions += 1       # 차체가 닿음: 그쪽으로 더 가지 않는다 (바퀴는 헛돈다)
            unmapped = self.world.unmapped()
            if unmapped and self.world.body_clearance(nx, ny, nyaw, walls=False, items=unmapped) <= 0.0:
                self.hidden_contacts += 1
        else:
            self.x, self.y, self.yaw = nx, ny, nyaw
        self.min_clearance = min(self.min_clearance, self.world.clearance(self.x, self.y))
        self.min_body_clearance = min(self.min_body_clearance, self.world.body_clearance(self.x, self.y, self.yaw))

    def _quantize(self, value):
        return round(value / self.encoder_m) * self.encoder_m if self.encoder_m else value

    def odom(self, t):
        return NS(header=NS(frame_id='odom', stamp=stamp(t)), child_frame_id='base_footprint',
                  pose=NS(pose=NS(position=NS(x=self._quantize(self.ox), y=self._quantize(self.oy)), orientation=quat(0.0))))

    def wheels(self):
        return NS(data=[self._quantize(self.left), self._quantize(self.right)])

    def scan(self, t):
        increment = 2 * math.pi / SCAN_POINTS
        angles = np.arange(SCAN_POINTS) * increment
        lx, ly = self.x + LIDAR_X * math.cos(self.yaw), self.y + LIDAR_X * math.sin(self.yaw)
        ranges = self.world.rays(lx, ly, self.yaw + math.pi / 2 + angles)
        ranges = ranges + self.np_rng.normal(0.0, self.noise_m, SCAN_POINTS)
        ranges[(angles > math.radians(45)) & (angles < math.radians(135))] = 0.0   # 드라이버가 가린 구간 (차체 뒤쪽)
        return NS(header=NS(frame_id='base_lidar_link', stamp=stamp(t)), ranges=ranges.tolist(), angle_min=0.0,
                  angle_increment=increment, range_min=0.02, range_max=25.0)


class ClosedLoop:
    """가상 로봇 ↔ SLAM 노드 ↔ 주행 노드. step() 한 번이 50 ms (20 Hz)."""

    DT = 0.05

    def __init__(self, world, robot, drive_params=None, lidar=True, slam_params=None):
        self.world, self.robot = world, robot
        self.t = 1000.0
        self.clock = fake.FakeClock()
        self.clock.t = self.t
        previous = (fake.FakeNode.clock, fake.FakeNode.distinct_publishers, fake.FakeNode.parameters)
        fake.FakeNode.clock, fake.FakeNode.distinct_publishers = self.clock, True
        try:
            slam_module, _ = fake.load_slam()
            drive_module, _ = fake.load_driving()
            fake.FakeNode.parameters = dict(slam_params or {})   # SLAM 노드 ROS 파라미터 (예: map_fusion=False)
            self.slam = slam_module['SlamNode']()
            fake.FakeNode.parameters = previous[2]
            self.drive = drive_module.DrivingNode()
        finally:
            fake.FakeNode.clock, fake.FakeNode.distinct_publishers, fake.FakeNode.parameters = previous
        self.drive_module = drive_module
        for name, value in (drive_params or {}).items():
            setattr(self.drive.params, name, value)
        self.lidar = lidar
        self.run_id = 'sim-run'
        self.phase, self.camp, self.slot = 'waiting', '', ''
        self.allowed = False
        self.tick = 0
        self.poses = []          # (t, SLAM x, y, yaw, 실제 x, y, yaw)
        self.cmd = (0.0, 0.0)
        self.listeners = {}      # 'pose' · 'map' · 'cmd' · 'status' · 'scan' · 'debug' → 함수
        self.slam.pose_publisher.publish = self._slam_pose
        self.slam.map_publisher.publish = self._slam_map
        self.drive.cmd_publisher.publish = self._drive_cmd
        self.drive.status_publisher.publish = lambda message: self._emit('status', message)
        for node in (self.slam, self.drive):
            publisher = getattr(node, 'debug_publisher', None)
            if publisher is not None:
                publisher.publish = lambda message: self._emit('debug', message)
        self._goal = None        # (handle, thread, 결과 담을 dict)
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ 노드 출력
    def _emit(self, kind, message):
        listener = self.listeners.get(kind)
        if listener is not None:
            listener(message)

    def _slam_pose(self, message):
        p, q = message.pose.position, message.pose.orientation
        yaw = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)
        self.poses.append((self.t, p.x, p.y, yaw, self.robot.x, self.robot.y, self.robot.yaw))
        self.drive._receive_pose(message)
        self._emit('pose', message)

    def _slam_map(self, message):
        self.drive._receive_map(message)
        self._emit('map', message)

    def _drive_cmd(self, message):
        self.cmd = (float(message.linear.x), float(message.angular.z))
        self.robot.command(*self.cmd)
        self._emit('cmd', message)

    # ------------------------------------------------------------------ 시험기가 넣는 것 (supervisor · server_bridge · perception 대신)
    def publish_server(self):
        self.slam._receive_server_state(NS(
            header=NS(stamp=stamp(self.t)), run_id=self.run_id, connected=True, team_assigned=bool(self.slot),
            slot=self.slot, camp=self.camp, phase=self.phase, ready=True, detail=''))

    def publish_map(self, bases=(), self_pose=None, others=(), boxes=None):
        """perception 대신 지도를 넣는다. boxes 를 주면 그 상자 목록을 (실제 세계와 다른 지도를 시험할 때)."""
        field = fake.FieldMap()
        field.header = NS(frame_id='arena', stamp=stamp(self.t))
        field.width, field.height, field.box_size = self.world.width, self.world.height, self.world.box_size
        field.boxes = [NS(x=x, y=y, z=0.0) for x, y in (self.world.boxes if boxes is None else boxes)]
        field.bases = [NS(name=name, min_x=x0, min_y=y0, max_x=x1, max_y=y1, row=0) for name, x0, y0, x1, y1 in bases]
        field.sectors, field.faces = [], []
        field.vehicles = []
        if self_pose is not None:
            field.vehicles.append(fake.FieldVehicle(role=1, pose=NS(position=NS(x=self_pose[0], y=self_pose[1], z=0.0), orientation=quat(self_pose[2]))))
        for x, y in others:
            field.vehicles.append(fake.FieldVehicle(role=2, pose=NS(position=NS(x=x, y=y, z=0.0), orientation=quat(0.0))))
        self.slam._receive_perception_map(field)

    def send_goal(self, x, y, yaw=None, mode=0, pos_tol=0.10, yaw_tol=0.10, timeout_s=60.0, on_feedback=None, on_result=None):
        request = NS(run_id=self.run_id, mode=mode, ignore_yaw=yaw is None, position_tolerance_m=pos_tol,
                     yaw_tolerance_rad=yaw_tol, execution_timeout_ms=int(timeout_s * 1000),
                     goal=NS(header=NS(frame_id='arena'), pose=NS(position=NS(x=x, y=y, z=0.0), orientation=quat(yaw or 0.0))))
        handle = fake.FakeGoalHandle(request)
        if on_feedback is not None:
            handle.publish_feedback = on_feedback
        box = {}

        def run():
            box['result'] = self.drive._execute(handle)
            if on_result is not None:
                on_result(box['result'])

        thread = threading.Thread(target=run, daemon=True)
        self._goal = (handle, thread, box)
        thread.start()
        # 목표가 주행 노드에 등록될 때까지 가상 시계를 멈춘 채 기다린다 → 같은 입력이면 같은 결과 (재현 시험)
        deadline = time.monotonic() + 2.0
        while thread.is_alive() and self.job_of(handle) is None and time.monotonic() < deadline:
            thread.join(0.001)
        return handle, thread, box

    def job_of(self, handle):
        """그 목표의 GoalExecutor (등록 전 · 거절이면 None)."""
        job = self.drive._job
        return job if job is not None and getattr(job, 'handle', None) is handle else None

    def cancel(self):
        if self._goal is not None:
            handle = self._goal[0]
            handle.is_cancel_requested = True
            self.drive._on_cancel(handle)

    # ------------------------------------------------------------------ 한 주기
    def step(self):
        with self.lock:
            self.robot.step(self.DT)
            self.t += self.DT
            self.clock.t = self.t
            self.tick += 1
            wheels = self.robot.wheels()
            self.slam._receive_wheels(wheels)
            self.drive._receive_wheels(wheels)
            if self.tick % 2 == 0:
                odom = self.robot.odom(self.t)
                self.slam._receive_odometry(odom)
                self.drive._receive_odom(odom)
                if self.lidar:
                    scan = self.robot.scan(self.t)
                    self.slam._receive_scan(scan)
                    self.drive._receive_scan(scan)
                    self._emit('scan', scan)
            if self.tick % 10 == 0:
                self.publish_server()
                self.drive._publish_status()
            self.drive._receive_control(NS(motion_allowed=self.allowed, lease_ms=500, run_id=self.run_id,
                                           reason='' if self.allowed else '시험기: 주행 허용 꺼짐'))
            self.drive._on_control_tick()
            for node in (self.slam, self.drive):
                publish = getattr(node, '_publish_debug', None)
                if publish is not None and self.tick % 4 == 0:
                    publish()

    def run(self, seconds):
        for _ in range(int(round(seconds / self.DT))):
            self.step()

    def error(self):
        """마지막 SLAM 위치의 (거리 오차 m, 방향 오차 rad). 위치가 없으면 None."""
        if not self.poses:
            return None
        _, x, y, yaw, rx, ry, ryaw = self.poses[-1]
        return math.hypot(x - rx, y - ry), abs(math.atan2(math.sin(yaw - ryaw), math.cos(yaw - ryaw)))

    def start_from_slot(self, slot, region, ready_s=1.0, map_boxes=None):
        """경기와 같은 순서: ready (잠김) → running → 첫 지도. region = (min_x, min_y, max_x, max_y).

        map_boxes: 첫 지도의 상자 목록 (드론 지도가 실제와 다를 때). None 이면 실제 세계의 boxes.
        """
        self.camp, self.slot, self.phase = ('BLUE' if slot.startswith('B') else 'RED'), slot, 'ready'
        self.publish_server()
        self.run(ready_s)
        self.phase = 'running'
        self.publish_server()
        self.run(0.5)
        self.publish_map(bases=[(slot, *region)], boxes=map_boxes)

    def drive_to(self, x, y, limit_s=90.0, **goal):
        """목표를 보내고 끝날 때까지 돌린다. 결과 메시지(없으면 None)를 돌려준다."""
        self.allowed = True
        handle, thread, box = self.send_goal(x, y, **goal)
        job = self.job_of(handle)
        end = self.t + limit_s
        while job is not None and not job.done and self.t < end:
            self.step()              # 목표가 끝난 주기에서 멈춘다 (실행 스레드가 끝나기를 기다리는 동안 가상 시계가 흐르지 않게)
        thread.join(1.0)
        if thread.is_alive():
            self.cancel()
            thread.join(1.0)
        return box.get('result')


SAMPLE_BOXES = ((1.4, 1.0), (1.4, 3.8), (2.2, 2.4), (3.0, 1.4), (3.0, 3.4), (3.8, 2.4), (4.6, 1.0), (4.6, 3.8))   # 시험용 임무 상자 배치
