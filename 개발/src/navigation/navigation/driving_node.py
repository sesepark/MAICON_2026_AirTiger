"""주행 노드 (/navigation). 팀 코드에서 /cmd_vel 을 발행하는 유일한 노드.

입력
  /team/navigation/map      FieldMap (SLAM, transient_local)  — 벽 · 박스 · 다른 UGV
  /team/navigation/pose     PoseStamped (SLAM, 10 Hz 이상)     — 자기 위치 (arena)
  /scan                     LaserScan                         — 지도에 없는 물체 · 충돌 정지
  /odom, /odom/odom_raw     SLAM 위치 사이를 메움 (/odom 은 거리만, 방향은 바퀴 거리 차이. yaw_source=imu 면 /imu/data)
  /team/perception/ugvs     PoseArray (perception, 약 1 Hz)   — 드론이 본 UGV 자리 (우리 차 포함). 막은 것이 서 있는 UGV 인지 · 헛돎 · 기억 칸 정리 (drive/ugvs.py)
  /team/control/state       동작 허용 임대 (끊기면 정지 + CONTROL_LOST)
  /team/mission/state       run_id 확인
출력
  /cmd_vel (control_hz, 목표 실행 중에만), /team/node_status (1 Hz + 바뀔 때)
  /team/navigation/debug (std_msgs/String JSON, debug_hz > 0 일 때만) — 경로 · 임시 장애물 · 단계 (시험 화면용)
기록
  TEAM_LOG_DIR 이 있으면(team_node.py 가 켰을 때) <기록 폴더>/navigation/debug.jsonl — 같은 내부 상태(log_hz)와 목표 · 결과 · 단계 · 지도 (run_log.py)
  /team/navigation/navigate (Navigate 액션 서버) — 실행 중 새 목표가 오면 이전 목표를 CANCELED 로 끝내고 멈추지 않고 바꾼다

로직은 navigation/drive/ 에 있고 이 파일은 ROS 연결만 한다.
"""
import json
import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32MultiArray, String
from team_interfaces.action import Navigate
from team_interfaces.msg import ControlState, FieldMap, MissionState, NodeStatus

from navigation.drive import executor as ex
from navigation.drive import pose_tracker as loc
from navigation.drive.fieldmap import InvalidMap, from_field_map
from navigation.drive.trail import Trail
from navigation.drive.ugvs import UgvTracks
from navigation.drive.geometry import Pose2D, quaternion_from_yaw, yaw_from_quaternion
from navigation.drive.grid import Grid
from navigation.drive.obstacles import ObstacleTracker, scan_to_body
from navigation.drive.params import DriveParams, param_items
from navigation.run_log import RunLog

MAP_TOPIC = '/team/navigation/map'
POSE_TOPIC = '/team/navigation/pose'
SCAN_TOPIC = '/scan'
ODOM_TOPIC = '/odom'
IMU_TOPIC = '/imu/data'
WHEEL_TOPIC = '/odom/odom_raw'  # Float32MultiArray [왼쪽, 오른쪽] 바퀴 누적 거리(m) — base_node.cpp handle_odom
UGVS_TOPIC = '/team/perception/ugvs'  # 드론이 본 UGV 자리 (perception/ugv_watch.py). frame arena, position.z = 점수
CONTROL_TOPIC = '/team/control/state'
MISSION_TOPIC = '/team/mission/state'
STATUS_TOPIC = '/team/node_status'
CMD_TOPIC = '/cmd_vel'
DEBUG_TOPIC = '/team/navigation/debug'  # 시험 화면(시험기록/web_drive)용. debug_hz > 0 일 때만 낸다
ACTION_NAME = '/team/navigation/navigate'

MAX_LINEAR = 0.6  # mil_base_node 상한 [실측]
MAX_ANGULAR = 3.0
ZERO_REPEAT = 5  # 멈출 때 0 을 이만큼 이어서 보낸다

MAP_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
POSE_QOS = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
ROBOT_QOS = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
CONTROL_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
MISSION_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
STATUS_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class DrivingNode(Node):
    """FieldMap · SLAM 위치 · /scan 으로 Navigate 목표까지 주행한다."""

    def __init__(self):
        super().__init__('navigation')
        self.params = self._declare_params()
        self._lock = threading.RLock()
        self.latest_map = None  # 마지막으로 받은 FieldMap 원본
        self.map_update = None
        self.grid = None
        self._map_problem = ''
        self.tracker = loc.PoseTracker(self.params)
        self.obstacles = ObstacleTracker(self.params)
        # 드론이 본 UGV (drive/ugvs.py). 우리 차를 골라내려고 지난 몇 초의 자세를 들고 있는다 (제어 주기마다 한 줄)
        self.ugvs = UgvTracks(self.params)
        self._pose_hist = deque(maxlen=128)  # (시각, x, y, 로컬 x, 로컬 y)
        if self.params.drone_ugvs:
            self.obstacles.ugvs = self.ugvs
        self.trail = Trail()  # 지나온 길(로컬 궤적) · 「뒤가 막혔다」 는 기억 — 목표가 바뀌어도 이어 쓴다 (되짚기, 2026-11-19 밤)
        self.turn_model = ex.TurnModel()  # 서서 돌기에서 잰 정지 여유 · 돌기 시작한 각속도 — 목표가 바뀌어도 이어 쓴다 (2026-11-19 실차 회전 실측)
        self._control = None  # (ControlState, 받은 시각)
        self._mission = None
        self._job = None  # 지금 실행 중인 GoalExecutor
        self._zero_left = 0
        self._last_scan_t = None
        self._last_error = ''
        self._last_level = None
        self._last_cmd = (0.0, 0.0)
        self._warned = {}
        # 경기 기록 (TEAM_LOG_DIR 이 있을 때만). 경기 중에는 ros2 로 /team/navigation/debug 를 볼 수 없어서 (2026-11-19)
        self.run_log = RunLog('navigation')
        self._logged_phase = None  # 마지막으로 기록한 (목표 번호, 단계, 막힘)
        self._log_ticks = 0
        self._goal_count = 0

        sensors = ReentrantCallbackGroup()
        self._action_group = ReentrantCallbackGroup()
        control_group = MutuallyExclusiveCallbackGroup()
        map_group = MutuallyExclusiveCallbackGroup()

        self.map_subscription = self.create_subscription(FieldMap, MAP_TOPIC, self._receive_map, MAP_QOS, callback_group=map_group)
        self.pose_subscription = self.create_subscription(PoseStamped, POSE_TOPIC, self._receive_pose, POSE_QOS, callback_group=sensors)
        self.scan_subscription = self.create_subscription(LaserScan, SCAN_TOPIC, self._receive_scan, qos_profile_sensor_data, callback_group=sensors)
        self.odom_subscription = self.create_subscription(Odometry, ODOM_TOPIC, self._receive_odom, ROBOT_QOS, callback_group=sensors)
        self.imu_subscription = self.create_subscription(Imu, IMU_TOPIC, self._receive_imu, ROBOT_QOS, callback_group=sensors)
        self.wheel_subscription = self.create_subscription(Float32MultiArray, WHEEL_TOPIC, self._receive_wheels, ROBOT_QOS, callback_group=sensors)
        if self.params.drone_ugvs:
            self.ugvs_subscription = self.create_subscription(PoseArray, UGVS_TOPIC, self._receive_ugvs, POSE_QOS, callback_group=sensors)
        self.control_subscription = self.create_subscription(ControlState, CONTROL_TOPIC, self._receive_control, CONTROL_QOS, callback_group=sensors)
        self.mission_subscription = self.create_subscription(MissionState, MISSION_TOPIC, self._receive_mission, MISSION_QOS, callback_group=sensors)
        self.cmd_publisher = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.status_publisher = self.create_publisher(NodeStatus, STATUS_TOPIC, STATUS_QOS)
        self.debug_publisher = self.create_publisher(String, DEBUG_TOPIC, 5)
        self.action_server = ActionServer(
            self, Navigate, ACTION_NAME,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._action_group,
        )
        self.control_timer = self.create_timer(1.0 / self.params.control_hz, self._on_control_tick, callback_group=control_group)
        self.status_timer = self.create_timer(1.0, self._publish_status, callback_group=sensors)
        if self.params.debug_hz > 0.0:
            self.debug_timer = self.create_timer(1.0 / self.params.debug_hz, self._publish_debug, callback_group=sensors)
        if self.run_log.enabled:
            self.run_log.write({'event': 'start', 't': round(self._now(), 3), 'params': dict(param_items(self.params))})
            if self.params.log_hz > 0.0:
                self.log_timer = self.create_timer(1.0 / self.params.log_hz, self._write_log, callback_group=sensors)
            self.get_logger().info('내부 상태 기록: %s' % self.run_log.path)

    # ------------------------------------------------------------------ 준비
    def _declare_params(self):
        params = DriveParams()
        for name, default in param_items(params):
            value = self.declare_parameter(name, default).value
            if value is not None:
                setattr(params, name, type(default)(value))
        return params

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_once(self, key, text, every_s=5.0):
        now = time.monotonic()
        if now - self._warned.get(key, -1e9) >= every_s:
            self._warned[key] = now
            self.get_logger().warning(text)

    # ------------------------------------------------------------------ 입력
    def _receive_map(self, message):
        self.latest_map = message
        try:
            update = from_field_map(message, pillars=self.params.fixed_pillars)
        except InvalidMap as error:
            self._map_problem = str(error)
            self._warn_once('map', '지도를 쓰지 않음: %s' % error)
            return
        if update.spec.frame_id != self.params.expected_frame:
            self._map_problem = '지도 frame_id %r (기대 %r)' % (update.spec.frame_id, self.params.expected_frame)
            self._warn_once('map-frame', '지도를 쓰지 않음: ' + self._map_problem)
            return
        grid = self.grid
        if self.map_update is None or grid is None or self.map_update.spec.geometry_key() != update.spec.geometry_key():
            p = self.params
            grid = Grid(update.spec, radius=p.robot_radius, margin=p.inflate_margin, cell=p.cell, soft_m=p.soft_m, soft_cost=p.soft_cost)
            self.get_logger().info('지도 갱신: %.2f × %.2f m, 박스 %d개, 고정 기둥 %d개' % (
                update.spec.width, update.spec.height, len(update.spec.boxes), len(update.spec.fixed)))
            self._log_event('map', size=[update.spec.width, update.spec.height], box_size=update.spec.box_size,
                            boxes=[[round(b[0], 3), round(b[1], 3)] for b in update.spec.boxes])
        now = self._now()
        with self._lock:
            changed = grid is not self.grid
            # 경기장 치수가 그대로이고 상자만 바뀌었으면(map_fusion 이 빼고 · 되살리고 · 옮김) 임시 장애물 기억을 지우지 않는다 (2026-11-19).
            # 본선 경기장 8판의 그런 바뀜 21번마다 기억이 통째로 지워졌다 (18:06 판 147초: 기억 481칸 → 0, 벽 쪽 통로에 들어서던 때).
            # 기억은 라이다가 본 것이라 지도의 상자가 바뀌어도 그대로 맞다: 새 상자 자리의 기억은 그 상자와 겹칠 뿐이고,
            # 뺀 상자 자리에는 기억이 없었다 (지도로 설명되던 점). 치수가 바뀌면(임시 지도 → 진짜 지도) 전처럼 지운다
            same_arena = (self.map_update is not None and abs(self.map_update.spec.width - update.spec.width) < 1e-6
                          and abs(self.map_update.spec.height - update.spec.height) < 1e-6)
            self.map_update, self.grid, self._map_problem = update, grid, ''
            if changed:
                if not same_arena:
                    self.obstacles.clear()
                if self._job is not None:
                    self._job.on_grid_changed(grid)
            self.obstacles.on_vehicles(now, update.others)

    def _receive_pose(self, message):
        if message.header.frame_id != self.params.expected_frame:
            self._warn_once('pose-frame', '위치를 쓰지 않음: frame_id %r (기대 %r)' % (message.header.frame_id, self.params.expected_frame))
            return
        p, q = message.pose.position, message.pose.orientation
        pose = Pose2D(float(p.x), float(p.y), yaw_from_quaternion(q.x, q.y, q.z, q.w))
        with self._lock:
            now = self._now()
            self.tracker.on_pose(now, _stamp_seconds(message.header.stamp), pose)
            self.turn_model.rate.on_slam_yaw(now, pose.yaw)  # 서서 돌 때 실제로 도는 빠르기를 잰다 (drive/turn_rate.py)

    def _receive_scan(self, message):
        now = self._now()
        self._last_scan_t = now
        p = self.params
        # 스캔이 찍힌 때: header.stamp = 첫 광선, 한 바퀴는 scan_time (없으면 time_increment × 점 수). 받은 때는 그보다 한 바퀴 + 전송만큼 늦다
        stamp = _stamp_seconds(message.header.stamp)
        took = float(getattr(message, 'scan_time', 0.0) or 0.0)
        if not (0.0 < took <= 0.3):
            took = float(getattr(message, 'time_increment', 0.0) or 0.0) * len(message.ranges)
        if not (0.0 < took <= 0.3):
            took = 0.0
        lag = now - stamp if stamp is not None else None
        self._scan_lag = (round(lag, 3) if lag is not None else None, round(took, 3))
        with self._lock:
            at_stamp = p.scan_at_stamp and lag is not None and -0.05 <= lag <= 0.5
            # 스캔을 지도에 놓을 자세: 끝 광선 때의 것 (전: 받은 때의 것 — 빨리 도는 동안 그 차이만큼 벽이 「지도에 없는 점」 으로 찍혔다)
            when = min(now, stamp + took) if at_stamp else now
            pose = self.tracker.estimate_at(when) if at_stamp else self.tracker.estimate()
            rate = self.tracker.yaw_rate(when)
            sweep = None
            if at_stamp and p.scan_deskew and took > 0.0:
                dx, dy = self.tracker.sweep(stamp, when)
                sweep = (dx, dy, -rate * (when - stamp))   # 한 바퀴 동안 돈 각은 고르게 본 빠르기로 (바퀴 방향은 3.3° 씩 뛴다)
            pts = scan_to_body(message.ranges, message.angle_min, message.angle_increment, p, message.range_min, message.range_max, sweep=sweep)
            if pose is None or self.grid is None or self.map_update is None:
                return
            # 광선 그대로도 넘긴다 — 라이다가 그 자리를 볼 수 있었는지(뒤쪽 가린 90° · 값 없는 광선)로 임시 장애물을 잊을지 정한다
            rays = (float(message.angle_min), float(message.angle_increment), np.asarray(message.ranges, dtype=float))
            self.obstacles.on_scan(now, pose, pts, self.map_update.spec, self.grid, rays, turn_rate=rate)

    def _receive_odom(self, message):
        p = message.pose.pose.position
        with self._lock:
            self.tracker.on_odom(self._now(), float(p.x), float(p.y))

    def _receive_wheels(self, message):
        data = message.data
        if len(data) < 2:
            return
        with self._lock:
            self.tracker.on_wheels(self._now(), float(data[0]), float(data[1]))

    def _receive_imu(self, message):
        rate = getattr(getattr(message, 'angular_velocity', None), 'z', None)
        if rate is not None:  # 각속도는 방향 소스와 상관없이 본다 — 실제 회전 빠르기 (SLAM 방향과 맞는 것이 확인된 뒤에만 쓰인다, drive/turn_rate.py)
            with self._lock:
                self.turn_model.rate.on_gyro(self._now(), float(rate))
        if self.params.yaw_source != 'imu' or message.orientation_covariance[0] < 0.0:  # -1 = orientation 없음
            return
        q = message.orientation
        with self._lock:
            self.tracker.on_imu(self._now(), yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _receive_ugvs(self, message):
        """드론이 본 UGV 자리. 우리 차는 그 사진 무렵 우리가 있던 자리로 골라낸다 — 그때의 자세를 모르면 그 사진은 버린다 (drive/ugvs.py)."""
        if message.header.frame_id != self.params.expected_frame:
            self._warn_once('ugvs-frame', '드론 UGV 자리를 쓰지 않음: frame_id %r (기대 %r)' % (message.header.frame_id, self.params.expected_frame))
            return
        stamp = _stamp_seconds(message.header.stamp)
        points = [(float(p.position.x), float(p.position.y)) for p in message.poses]
        now = self._now()
        with self._lock:
            if not 0.0 <= now - stamp + 0.2 <= self.params.ugv_feed_s:
                return  # 너무 오래됐거나 시계가 다르다
            window = [h for h in self._pose_hist if stamp - self.params.ugv_self_window_s <= h[0] <= stamp + 0.1]
            at = min(window, key=lambda h: abs(h[0] - stamp)) if window else None
            used = self.ugvs.on_frame(stamp, points, [(h[1], h[2]) for h in window],
                                      local=(at[3], at[4]) if at else None, pose=(at[1], at[2]) if at else None)
            summary = self.ugvs.summary(now)
        if used:
            self._log_event('ugvs', stamp=round(stamp, 3), seen=[[round(x, 3), round(y, 3)] for x, y in points], **summary)

    def _receive_control(self, message):
        with self._lock:
            self._control = (message, self._now())

    def _receive_mission(self, message):
        with self._lock:
            self._mission = message

    # ------------------------------------------------------------------ 제어 허용
    def _motion_allowed(self, now, run_id):
        if self._control is None:
            return False, '제어 허용(/team/control/state)을 받지 못함'
        msg, recv = self._control
        age = now - recv
        if not msg.motion_allowed:
            return False, '제어 허용 꺼짐: %s' % (msg.reason or '사유 없음')
        if age > self.params.control_timeout_s:
            return False, '제어 허용이 %.2f s 동안 오지 않음' % age
        if msg.lease_ms and age > msg.lease_ms / 1000.0:
            return False, '제어 허용 임대 %d ms 만료' % msg.lease_ms
        if msg.run_id and run_id and msg.run_id != run_id:
            return False, '제어 허용의 run_id %r ≠ 목표 %r' % (msg.run_id, run_id)
        return True, ''

    # ------------------------------------------------------------------ 주기 제어
    def _on_control_tick(self):
        now = self._now()
        cmd = ex.STOP
        active = False
        try:
            with self._lock:
                level = self.tracker.update_level(now)
                level_changed = level != self._last_level
                self._last_level = level
                pose = self.tracker.estimate()
                if pose is not None and self.params.drone_ugvs:
                    local = self.tracker.local()
                    self._pose_hist.append((now, pose.x, pose.y, local.x, local.y))
                job = self._job
                if job is not None and not job.done:
                    allowed, why = self._motion_allowed(now, job.goal.run_id)
                    if not allowed:
                        if now - job.start_t < self.params.control_timeout_s:
                            job.phase = 'waiting_control'
                        else:
                            job.finish(ex.CONTROL_LOST, why, pose)
                    else:
                        if self.tracker.jumped:
                            job.replan()
                            self.tracker.jumped = False
                        cmd = job.step(now, pose, level, self.grid, self.obstacles, local=self.tracker.local())
                    active = not job.done
                    self._log_phase(now, job, pose, cmd)
            self._last_error = ''
            if level_changed:
                self._publish_status()  # 상태가 바뀌면 바로 알린다 (interfaces.yaml: 변경 즉시 + 1 Hz)
                self._log_event('level', level=loc.LEVEL_NAMES[level], pose=self._pose_list(pose))
        except Exception as error:  # 예외에서는 반드시 멈춘다
            self._last_error = repr(error)
            self.get_logger().error('제어 주기 예외 → 정지: %r' % (error,))
            with self._lock:
                if self._job is not None:
                    self._job.finish(ex.BLOCKED, '주행 노드 내부 오류: %r' % (error,), code=ex.CODE_INTERNAL)
            cmd, active = ex.STOP, False
        self._send(cmd, active)

    def _send(self, cmd, active):
        v = max(-MAX_LINEAR, min(MAX_LINEAR, cmd.v))
        w = max(-MAX_ANGULAR, min(MAX_ANGULAR, cmd.w))
        moving = abs(v) > 1e-6 or abs(w) > 1e-6
        if active or moving:
            self._zero_left = ZERO_REPEAT  # 멈춘 뒤 0 을 이만큼 더 보낸다
        elif self._zero_left > 0:
            self._zero_left -= 1
        else:
            return
        if not active and moving:  # 목표가 없는데 움직이는 명령은 보내지 않는다
            v = w = 0.0
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_publisher.publish(msg)
        self._last_cmd = (float(v), float(w))
        with self._lock:
            self.tracker.on_command(v)

    def stop_robot(self):
        try:
            for _ in range(ZERO_REPEAT):
                self.cmd_publisher.publish(Twist())
        except Exception:  # 종료 중이라 context 가 이미 닫혔을 수 있다
            pass

    # ------------------------------------------------------------------ 액션
    def _on_goal(self, goal_request):
        return GoalResponse.ACCEPT  # 검사는 실행 쪽에서 하고 INVALID_GOAL 로 알린다 (이유를 결과에 담기 위해)

    def _on_cancel(self, goal_handle):
        """취소 요청은 받는 즉시 목표를 끝낸다 → 다음 제어 주기(≤ 50 ms)에 정지."""
        with self._lock:
            job = self._job
            if job is not None and getattr(job, 'handle', None) is goal_handle and not job.done:
                job.finish(ex.CANCELED, '취소 요청', self.tracker.estimate())
        return CancelResponse.ACCEPT

    def _goal_spec(self, request):
        g = request.goal
        q = g.pose.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        ignore = bool(request.ignore_yaw) or request.yaw_tolerance_rad >= ex.YAW_IGNORE_TOL
        yaw = None if ignore or norm < 1e-6 else yaw_from_quaternion(q.x, q.y, q.z, q.w)
        spec = ex.GoalSpec(
            x=float(g.pose.position.x), y=float(g.pose.position.y), yaw=yaw, mode=int(request.mode),
            pos_tol=float(request.position_tolerance_m), yaw_tol=float(request.yaw_tolerance_rad),
            timeout_s=float(request.execution_timeout_ms) / 1000.0, frame_id=g.header.frame_id, run_id=request.run_id,
        )
        return spec, (not ignore and norm < 1e-6)

    def _execute(self, goal_handle):
        spec, bad_quat = self._goal_spec(goal_handle.request)
        job = ex.GoalExecutor(spec, self.params, self._now(), self.turn_model, self.trail)
        job.handle = goal_handle
        with self._lock:
            pose = self.tracker.estimate()
            mission_run = self._mission.run_id if self._mission is not None else ''
            if not spec.run_id:
                job.finish(ex.INVALID_GOAL, 'run_id 가 비어 있음')
            elif mission_run and spec.run_id != mission_run:
                job.finish(ex.INVALID_GOAL, 'run_id %r ≠ 현재 실행 %r' % (spec.run_id, mission_run))
            elif bad_quat:
                job.finish(ex.INVALID_GOAL, '도착 방향(orientation)이 비어 있음 — 방향이 상관없으면 ignore_yaw=true')
            elif job.validate(self.map_update.spec if self.map_update else None, self.grid, pose):
                previous = self._job
                if previous is not None and not previous.done:
                    previous.finish(ex.CANCELED, '새 목표로 교체됨', pose)
                self._job = job
        with self._lock:
            self._goal_count += 1
            job.log_no = self._goal_count
        self._log_event('goal', no=job.log_no, mode=ex.MODE_NAMES.get(spec.mode, spec.mode), goal=[spec.x, spec.y, spec.yaw],
                        pos_tol=spec.pos_tol, yaw_tol=spec.yaw_tol, timeout=spec.timeout_s, run_id=spec.run_id,
                        pose=self._pose_list(pose), accepted=job.outcome != ex.INVALID_GOAL, detail=job.detail)
        if job.outcome == ex.INVALID_GOAL:
            self.get_logger().warning('목표 거절: %s' % job.detail)
        else:
            self.get_logger().info('목표 %s (%.2f, %.2f) %s' % (ex.MODE_NAMES[spec.mode], spec.x, spec.y, job.detail))
        period = 1.0 / max(0.5, self.params.feedback_hz)
        last_feedback = -1e9
        while not job.done:
            if goal_handle.is_cancel_requested:
                with self._lock:
                    job.finish(ex.CANCELED, '취소 요청', self.tracker.estimate())
                break
            if not rclpy.ok():
                with self._lock:
                    job.finish(ex.CANCELED, '노드 종료')
                break
            now = time.monotonic()
            if now - last_feedback >= period:
                last_feedback = now
                self._publish_feedback(goal_handle, job)
            time.sleep(0.02)
        with self._lock:
            if self._job is job:
                self._job = None
            final_pose = job.final_pose or self.tracker.estimate()
        result = Navigate.Result()
        result.outcome = int(job.outcome)
        result.detail = job.detail
        result.final_pose = self._pose_msg(final_pose)
        if job.outcome == ex.REACHED:
            goal_handle.succeed()
        elif job.outcome == ex.CANCELED and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        self.get_logger().info('목표 끝: %s %s' % (ex.OUTCOME_NAMES.get(job.outcome, job.outcome), job.detail))
        self._log_event('result', no=job.log_no, outcome=ex.OUTCOME_NAMES.get(job.outcome, job.outcome), detail=job.detail,
                        pose=self._pose_list(final_pose), elapsed=round(self._now() - job.start_t, 2))
        return result

    def _publish_feedback(self, goal_handle, job):
        with self._lock:
            pose = self.tracker.estimate()
            fb = Navigate.Feedback()
            fb.phase = job.phase
            fb.current_pose = self._pose_msg(pose)
            fb.distance_valid = job.remaining is not None and pose is not None
            fb.remaining_distance_m = float(job.remaining) if fb.distance_valid else 0.0
            fb.blocked = bool(job.blocked)
            fb.blocked_duration_s = float(self._now() - job.blocked_since) if job.blocked and job.blocked_since is not None else 0.0
        goal_handle.publish_feedback(fb)

    def _pose_msg(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.params.expected_frame
        if pose is not None:
            msg.pose.position.x, msg.pose.position.y = float(pose.x), float(pose.y)
            qx, qy, qz, qw = quaternion_from_yaw(pose.yaw)
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qx, qy, qz, qw
        return msg

    # ------------------------------------------------------------------ 상태
    def _publish_status(self):
        now = self._now()
        with self._lock:
            level = self.tracker.level
            has_pose = self.tracker.last_pose_recv is not None
            run_id = self._mission.run_id if self._mission is not None else ''
        problems = []
        if self._last_error:
            state = NodeStatus.ERROR
            problems.append(self._last_error)
        elif self.grid is None or not has_pose:
            state = NodeStatus.INITIALIZING
            problems.append('지도 없음' if self.grid is None else '자기 위치 없음')
            if self._map_problem:
                problems.append(self._map_problem)
        else:
            state = NodeStatus.READY
            if level != loc.OK:
                state = NodeStatus.DEGRADED
                problems.append('위치 상태 %s' % loc.LEVEL_NAMES[level])
            if self._last_scan_t is None or now - self._last_scan_t > 1.0:
                state = NodeStatus.DEGRADED
                problems.append('/scan 없음 (지도에 없는 물체를 못 피함)')
            if self._map_problem:
                state = NodeStatus.DEGRADED
                problems.append(self._map_problem)
        msg = NodeStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.run_id = run_id
        msg.node_name = 'navigation'
        msg.state = state
        msg.detail = '; '.join(problems)
        self.status_publisher.publish(msg)

    def _publish_debug(self):
        """시험 화면용: 지금 자세 · 목표 · 경로 · 임시 장애물을 JSON 으로. 주행 판단에는 쓰지 않는다."""
        out = self._debug_state(self._now())
        self.debug_publisher.publish(String(data=json.dumps(out, ensure_ascii=False, separators=(',', ':'))))

    def _debug_state(self, now, cells=True):
        """/team/navigation/debug 와 기록 파일에 쓰는 내부 상태. cells=False 면 임시 장애물 칸(큼) 대신 칸 수만."""
        with self._lock:
            pose = self.tracker.estimate()
            job = self._job
            out = {
                'node': 'navigation', 't': round(now, 3),
                'pose': [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 4)] if pose is not None else None,
                'level': loc.LEVEL_NAMES[self.tracker.level],
                'pose_age': round(now - self.tracker.last_pose_recv, 3) if self.tracker.last_pose_recv is not None else None,
                'scan_age': round(now - self._last_scan_t, 3) if self._last_scan_t is not None else None,
                'cmd': [round(self._last_cmd[0], 3), round(self._last_cmd[1], 3)],
                'map': self.grid is not None, 'map_problem': self._map_problem, 'error': self._last_error,
            }
            allowed, why = self._motion_allowed(now, job.goal.run_id if job is not None else '')
            out['allowed'], out['why'] = allowed, why
            if job is not None:
                g = job.goal
                out['job'] = {
                    'mode': ex.MODE_NAMES.get(g.mode, g.mode), 'phase': job.phase, 'goal': [g.x, g.y, g.yaw],
                    'target': [round(job.target[0], 3), round(job.target[1], 3)],
                    'path': [[round(x, 3), round(y, 3)] for x, y in (job._path or ())],
                    'remaining': round(job.remaining, 3) if job.remaining is not None else None,
                    # 지금 경로의 종류: through = 임시 장애물을 빼고 짠 것(그 물체를 지나야만 간다) · tight = 서 있는 물체를 작게 보고 짠 것 (2026-11-19 밤)
                    'through': bool(getattr(job, '_through_obstacle', False)), 'tight': bool(getattr(job, '_tight_path', False)),
                    # 서 있는 물체가 목표 칸을 덮어 목표를 곁의 빈 칸으로 옮겨 가는 중인가 (2026-11-20, executor._standing_shift — target 이 그 자리)
                    'shifted': getattr(job, '_shift_home', None) is not None,
                    'corridor': bool(getattr(job, 'in_corridor', False)),
                    'blocked': bool(job.blocked), 'elapsed': round(now - job.start_t, 2), 'timeout': g.timeout_s,
                    # 드론으로 본 것 (2026-11-19): 막은 것의 종류 [parked · moving · none, 서 있던 s] · 헛돈 것으로 본 횟수
                    'blocker': list(job.blocker_kind) if getattr(job, 'blocker_kind', None) else None, 'slips': getattr(job, 'drone_slips', 0),
                }
            if self.params.drone_ugvs:
                out['ugvs'] = dict(self.ugvs.summary(now), dropped_cells=self.obstacles.dropped_ugv_cells)
            if self.grid is not None:
                taken = self.obstacles.snapshot(now, self.grid)
                if cells:
                    out['obstacles'] = {'cell': self.grid.cell, 'cells': [v for c in taken for v in c]}
                else:
                    out['obstacle_cells'] = len(taken)
                out['unexplained'] = [[round(x, 2), round(y, 2)] for x, y in self.obstacles.unexplained_world[:300]]
                # 스캔이 찍힌 때와 받은 때의 차 · 한 바퀴 시간 (s), 바퀴 방향에 곱하는 값과 그것을 고친 횟수 (2026-11-19 밤 — 방향 오차의 몫을 재려고)
                out['scan_lag'] = getattr(self, '_scan_lag', None)
                out['yaw_scale'] = [round(self.tracker.yaw_scale, 3), self.tracker.yaw_scale_updates]
                out['trail'] = self.trail.summary(now)  # 지나온 길의 점 수 · 「뒤가 막혔다」 가 남은 시간 (2026-11-19 밤)
                out['turn'] = self.turn_model.summary()  # 서서 돌기에서 잰 것: 정지 여유(°) · 돌기 시작한 각속도 (2026-11-19)
                out['memory'] = self.obstacles.memory_summary(now)  # 라이다가 못 봐서 기억으로만 남은 칸 (obstacle_memory_s 맞추기용, 2026-11-19)
        return out

    # ------------------------------------------------------------------ 경기 기록 (run_log)
    def _write_log(self):
        """log_hz 타이머: 목표 실행 중에는 매번, 쉬는 동안은 log_idle_s 마다 내부 상태 한 줄."""
        try:
            self._log_ticks += 1
            job = self._job
            busy = job is not None and not job.done
            every = max(1, int(round(self.params.log_idle_s * self.params.log_hz)))
            if not busy and self._log_ticks % every:
                return
            out = self._debug_state(self._now(), cells=False)
            out['event'] = 'state'
            self.run_log.write(out)
        except Exception as error:  # 기록 때문에 노드가 죽으면 안 된다
            self._warn_once('run-log', '내부 상태 기록 실패: %r' % (error,), every_s=30.0)

    def _log_event(self, event, **fields):
        if self.run_log.enabled:
            self.run_log.write(dict({'event': event, 't': round(self._now(), 3)}, **fields))

    def _log_phase(self, now, job, pose, cmd):
        """제어 주기에서 목표의 단계 · 막힘이 바뀌면 그때 한 줄 (2 Hz 상태 사이에 지나가는 단계도 남게)."""
        if not self.run_log.enabled:
            return
        key = (getattr(job, 'log_no', None), job.phase, bool(job.blocked))
        if key == self._logged_phase:
            return
        self._logged_phase = key
        self.run_log.write({'event': 'phase', 't': round(now, 3), 'no': key[0], 'phase': job.phase, 'blocked': key[2],
                            'pose': self._pose_list(pose), 'cmd': [round(cmd.v, 3), round(cmd.w, 3)],
                            'remaining': round(job.remaining, 3) if job.remaining is not None else None, 'detail': job.detail})

    @staticmethod
    def _pose_list(pose):
        return [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 4)] if pose is not None else None


def main(args=None):
    rclpy.init(args=args)
    node = DrivingNode()
    executor = MultiThreadedExecutor(num_threads=6)  # 제어 타이머 · 센서 · 목표 실행 스레드(교체 때 둘)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_robot()
        node.run_log.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
