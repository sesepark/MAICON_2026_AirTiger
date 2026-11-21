"""supervisor 의 ROS 연결. 판단은 mission.py 가 하고, 여기는 메시지를 받아 넣어 주고 시키는 것을 메시지로 내보낸다.

구조는 주최측 공식 team_node.py 뼈대(2026-11-18)와 같다: 메시지 받기는 배경 스레드의 rclpy.spin, 판단은 메인 스레드에서 0.1초마다.
서비스 · 액션은 모두 요청만 보내고 바로 돌아온다 (답은 콜백이 받아 다음 틱에 넣는다).

받는 것   /mil/match · /mil/state · /image_raw (대회 제공), /team/navigation/pose, /team/navigation/map, /team/navigation/lidar_boxes,
          /team/perception/ugvs (드론 사진에서 찾은 UGV 자리 — 없어도 된다), /team/server/state · /team/server/command, /team/node_status
내보내는 것 /team/control/state (0.1초마다), /team/mission/state · /team/node_status (1초마다)
부르는 것  /mil/gimbal · /mil/fire, /team/perception/classify_image · read_ugvs, /team/server/set_ready · submit_report,
          액션 /team/navigation/navigate · /team/llm/analyze
이름과 값은 설정 파일(JSON)로 바꾼다 — 기본값은 아래 DEFAULTS. 형식의 뜻은 docs/요청사항.md.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback
import uuid
from collections import deque

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from team_interfaces.action import Analyze, Navigate
from team_interfaces.msg import ControlState, FieldMap, NodeStatus, ObjectCount, ParsedCommand, Report, SectorCount, ServerCommand, ServerState, ThreatAssessment
from team_interfaces.msg import MissionState as TeamMissionState
from team_interfaces.srv import ClassifyImage, LocateSensors, ReadUgvs, SetReady, SubmitReport

from . import arena as A
from . import checkpoint
from .aim import AimParams, parse_sensors
from .faces import CameraParams
from .mission import Facts, Mission, MissionParams
from .standing import add_sight
from .recorder import Recorder, photos_allowed, session_folder
from .scout import ScoutParams

PERIOD = 0.1
ANSWER_TIMEOUT = 3.0  # 대회 제공 서비스(조준 · 발사)의 답을 기다리는 최대 시간 (공식 뼈대)

DEFAULTS = {
    "scout_mode": "face",  # "face" 면 기준 · "place" 자리 기준 (docs 와 README 참고)
    "slot": "",  # 서버가 자리를 알려 주지 않을 때만 쓴다 (시험용). 예 "B-1"
    "force_start": False,  # True 면 경기 신호 없이 켜지자마자 시작한 것으로 친다 (연습 때 시험용)
    "record_dir": "log/supervisor",  # 경기 기록 폴더 (켜질 때마다 그 아래에 폴더 하나: events.jsonl · photos.jsonl · photos/). 빈 문자열이면 남기지 않는다
    "save_photos": True,  # 판독에 보낸 사진을 파일로 남긴다
    "m1_fallback_s": 12.0,  # 명령문을 받고 이만큼 지나도 llm_worker 의 답이 없으면 supervisor 가 같은 낱말 해석기를 직접 부른다
    "state_file": "supervisor_state.json",  # 경기 중 상태를 적어 두는 파일 (다시 켜졌을 때 되살린다). 빈 문자열이면 쓰지 않는다
    # 준비완료(POST /ready)를 보내는 조건. 이름은 앞의 "/" 가 있든 없든 같게 본다
    "require_nodes": ["perception"],  # /team/node_status 가 READY 나 DEGRADED 여야 하는 프로그램 (모델을 다 읽었다는 뜻)
    "alive_nodes": ["navigation", "server_bridge"],  # 상태를 내보내고 있기만 하면 되는 프로그램 (navigation 은 지도가 오기 전에는 INITIALIZING 인데, 지도는 시작 뒤에야 온다)
    "ready_deadline_s": 12.0,  # 시작 대기가 이만큼 지나면 위 조건이 안 돼도 준비완료를 보낸다 — 안 보내면 그 팀을 빼고 시작할 수 있다 (API 문서 1단계)
    "camera": {},  # CameraParams 의 값 (hfov_deg 등)
    "arena": {},  # ArenaPlan 의 값
    "base_kinds": None,  # {"BLUE": [1줄, 2줄, 3줄 거점], "RED": [...]}
    # ScoutParams 의 값. order 의 기본을 여기서 "live" 로 둔다 (팀장 결정 2026-11-20 00:15: 평가 때 실시간 순서를 켠다) — 정찰 순서를 경기 도중에
    # 다시 정한다 (scout.FaceScout._live_pick). 2026-11-20 00:00 연습 판에서 실차로 한 판 돌았다 (M3 30/30 · 180초 · 예외 없음).
    # 끄려면 설정 파일에 "scout": {"order": "tour"}. ScoutParams 자체의 기본값은 "tour" 그대로다 (시험들이 그 순서를 본다)
    "scout": {"order": "live"},
    # MissionParams 의 값.
    # m6_hunt 를 여기서 False 로 둔다 (2026-11-20 06:00 통합 쪽 — **팀장이 정하기 전까지**). 사격 담당의 「M6 사냥」 이 05:25 ~ 05:40 사이 main 에 들어왔다:
    # 정찰(M3)이 3분 안에 끝나면 우리 진영을 돌며 상대 UGV 를 살피고 멈춘 것을 쏘고, **그 판에서는 M5(거점 앞 대기 · 거점 진입)를 하지 않는다**
    # (MissionParams.m6_hunt · m6_instead_of_m5 의 기본이 켬). 실차로는 한 판도 돌지 않았고(임시_테스트_파일/m6_평가/README.md 3.3),
    # 연습 판의 정찰은 대부분 3분 안에 끝나므로(138 ~ 204초) 그대로 두면 거의 모든 판에서 M5 를 하지 않게 된다. 팀장의 M6 결정은 「보류」 였다 (03:20).
    # False 면 그 README 대로 전의 흐름이다: 정찰이 끝난 자리에 서서 3초마다 read_ugvs 만 부르고, 4:00 전 거점 앞 대기 → 거점 진입을 한다.
    # 켜려면 설정 파일에 "mission": {"m6_hunt": true} 를 넣거나 이 값을 지운다. MissionParams 자체의 기본값은 그대로 켬이다 (그 시험들이 그 값을 본다)
    "mission": {"m6_hunt": False},
    # AimParams 의 값 — 지정표적을 사진으로 조준하기 (aim.py). 레이저 점 식 laser_u0 · laser_ku · laser_v0 · laser_kv (기본값은 2026-11-19 에 잰 것),
    # 허용 tol_px, 고치는 횟수 max_corrections, 끄려면 "enabled": false (그러면 계산한 각도로만 쏜다)
    "aim": {},
    # rate_deg_s: 팬틸트가 실제로 도는 빠르기 어림 — 2026-11-18 팀 테스트에서 57도를 돌리라 한 뒤 0.8초에 찍은 사진이 32도쯤만 돈 채였다 (약 40도/초).
    #             /mil/gimbal 의 ok 는 접수일 뿐이라 도착을 알 수 없다 → settle_s + (돌릴 각도 / rate_deg_s) 만큼 기다린 뒤에 찍거나 쏜다.
    # reaim_deg:  정찰 사진에서 겨눌 각도가 이만큼 넘게 달라져야 다시 돌린다 (위치 값이 흔들릴 때 팬틸트가 왔다 갔다 하지 않게. 화각이 넓어 몇 도는 상관없다)
    "gimbal": {"pan_deg_per_unit": 0.5, "tilt_deg_per_unit": 0.3, "speed": 120.0, "acc": 20.0, "settle_s": 0.6, "rest_tilt_deg": 0.0, "rate_deg_s": 30.0, "reaim_deg": 4.0},
    # yaw_tolerance_rad 0.10 → 0.17 (5.7° → 10°, 2026-11-19 사용자: 방향 허용을 조금 더 푼다). 목표마다 허용을 따로 넣지 않는 이동의 기본값이다
    # (정찰 · 사격 · 거점 앞 대기는 따로 넣는다). 카메라 · 레이저는 팬으로 겨누므로 차체 방향 5° 차이는 판독 · 조준에 쓰이지 않는다
    "nav": {"position_tolerance_m": 0.10, "yaw_tolerance_rad": 0.17, "timeout_ms": 60000},
    "names": {
        "pose": "/team/navigation/pose", "map": "/team/navigation/map", "boxes": "/team/perception/boxes", "navigate": "/team/navigation/navigate",
        "lidar_boxes": "/team/navigation/lidar_boxes",  # navigation 이 라이다로 찾았으나 지도에는 넣지 않은 상자 (mission 의 지도에 없는 상자)
        "ugvs": "/team/perception/ugvs",  # perception 이 드론 사진에서 찾은 UGV 자리 (우리 차 포함) — 정찰의 실시간 순서가 서 있는 로봇 곁의 자리를 뒤로 돌린다
        "classify": "/team/perception/classify_image", "read_ugvs": "/team/perception/read_ugvs", "locate_sensors": "/team/perception/locate_sensors",
        "set_ready": "/team/server/set_ready", "submit_report": "/team/server/submit_report",
        "server_state": "/team/server/state", "server_command": "/team/server/command", "analyze": "/team/llm/analyze",
    },
}


def load_config(path):
    """기본값에 설정 파일을 덮는다. 파일이 깨져 있으면 기본값으로 돈다 (설정 실수로 경기를 못 치르면 안 된다) — cfg["_config_error"] 에 까닭을 남긴다."""
    cfg = json.loads(json.dumps(DEFAULTS))
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError("맨 바깥이 { } 가 아님")
            for key, value in loaded.items():
                if isinstance(cfg.get(key), dict) and isinstance(value, dict):
                    cfg[key].update(value)
                else:
                    cfg[key] = value
        except (OSError, ValueError) as error:
            cfg = json.loads(json.dumps(DEFAULTS))
            cfg["_config_error"] = "%s: %s" % (path, error)
    return cfg


# navigation 이 Navigate 결과의 detail 에 넣는 표시 (개발/src/supervisor/docs/주행_결과_코드_확인_2026-11-19.md · 돌_자리_없음_보고_2026-11-19.md).
# 다른 프로그램의 모듈은 import 하지 않고 글자로 견준다 — 「들어 있는지」 로 본다. detail 끝의 [코드] 가 기준이고, 코드가 붙기 전 판(3e979ee)의
# 한글 문장도 같이 본다. 바뀌면 노드_관리/interfaces.yaml 의 navigate 와 함께 고친다
NO_TURN_AT_GOAL = ("[NO_ROOM_AT_GOAL]", "돌 자리 없음(도착 방향)")  # REACHED: 자리에는 왔는데 도착 방향으로 돌 자리가 없다
NO_ROOM_TO_LEAVE = ("[NO_ROOM_TO_LEAVE]", "돌 자리 없음(출발)")  # BLOCKED: 목표 쪽으로 가려면 서서 돌아야 하는데 돌 자리가 없다


def says(detail, marks):
    return any(mark in detail for mark in marks)

# 서버 단계(GET /me 의 phase) → supervisor 의 단계. 연습 모드에서 로봇의 경기 신호가 없을 때만 쓴다
SERVER_PHASES = {"ready": "standby", "running": "running", "paused": "paused", "reporting": "report"}


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def latched(depth=1):
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class SupervisorNode(Node):
    def __init__(self, cfg):
        super().__init__("supervisor")
        self.cfg = cfg
        self.run_id = uuid.uuid4().hex[:12]
        self.started = self.get_clock().now()
        self.lock = threading.Lock()
        names = cfg["names"]
        scout = ScoutParams(**{"mode": cfg["scout_mode"], **cfg["scout"]})
        aim = cfg.get("aim") or {}
        unknown = sorted(k for k in aim if k not in AimParams.__dataclass_fields__)  # 모르는 이름 때문에 켜지지 못하면 안 된다 → 빼고 돈다
        if unknown:
            self.get_logger().error("설정 aim 의 모르는 이름을 빼고 돕니다: %s" % ", ".join(unknown))
        aim = AimParams(**{k: v for k, v in aim.items() if k in AimParams.__dataclass_fields__})
        self.mission = Mission(MissionParams(scout=scout, **{**cfg["mission"], "aim": aim}), A.arena_from(cfg["arena"]), cfg["base_kinds"], CameraParams(**cfg["camera"]))
        # ---- 최신값 (콜백이 채운다)
        self.cmd, self.mode = "idle", ""
        self.hp = None
        self.battery = None
        self.frame = None  # 최신 사진 (sensor_msgs/Image 그대로)
        self._frame_at = -1e9  # 그 사진이 들어온 때 (조준 확인 사진은 팬틸트가 멈춘 뒤에 들어온 것만 쓴다)
        self.poses = deque(maxlen=200)  # (시각, x, y, yaw)
        self.boxes = None
        self.field = None  # /team/navigation/map 으로 받은 지도
        self.map_upgraded = False  # 임시 지도(상자 없음)에서 진짜 지도로 바뀌었다 → mission 에 알릴 것
        self.slot = cfg["slot"] or None
        self.order_text = self.objective = self.parsed = None
        self.status = {}  # 프로그램 이름 → (상태, 받은 때)
        self.events = {"answers": [], "ugvs": [], "sensors": [], "spots": [], "results": [], "fire": None, "go_failed": None, "no_turn": None}
        self.gone = None  # navigation 이 라이다로 없다고 보아 지도에서 뺀 상자의 번호 (처음 받은 지도의 번호). 아직 견줘 보지 않았으면 None
        self._map_first = []  # 처음 받은 지도의 상자 [(x, y)] — 뒤에 오는 지도와 견준다
        self._map_last = None  # 마지막으로 견줘 본 지도의 상자들
        self.extra_boxes = None  # navigation 이 라이다로 찾았으나 지도에는 넣지 않은 상자 [(x, y)] — 가장 새 목록. 받은 적이 없으면 None
        self.ugv_sights = None  # 위에서 본 UGV 자리의 요즘 관측들 [(때 — time.monotonic 의 시계, [(x, y, 점수)])]. 받은 적이 없으면 None (on_ugvs)
        self.done = False
        # ---- 경기 시계 (시작 신호부터, 일시정지 동안은 멈춘다)
        self._run_since = time.monotonic() if cfg["force_start"] else None
        self._ran = 0.0
        self._started = bool(cfg["force_start"])
        # ---- 받기
        try:
            from mil_msgs.msg import MatchCommand, MissionState
            from mil_msgs.srv import Fire, Gimbal
            self.create_subscription(MatchCommand, "/mil/match", self.on_match, latched())
            self.create_subscription(MissionState, "/mil/state", self.on_state, latched())
            self.cli_gimbal = self.create_client(Gimbal, "/mil/gimbal")
            self.cli_fire = self.create_client(Fire, "/mil/fire")
            self._Gimbal, self._Fire = Gimbal, Fire
        except ImportError:  # 로봇이 아닌 곳 (대회 제공 메시지가 없음) — 조준 · 발사 없이 돈다
            self.get_logger().warning("mil_msgs 가 없습니다: 경기 신호 · 조준 · 발사 없이 돕니다")
            self.cli_gimbal = self.cli_fire = None
        self.create_subscription(Image, "/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, names["pose"], self.on_pose, 10)
        self.create_subscription(FieldMap, names["map"], self.on_map, latched())
        self.create_subscription(PoseArray, names["boxes"], self.on_boxes, latched())  # 지도 대신 상자 목록만 오는 경우 (시험)
        self.create_subscription(PoseArray, names["lidar_boxes"], self.on_lidar_boxes, 1)  # 1초마다 온다 (가장 새 것 하나만 쓴다)
        self.create_subscription(PoseArray, names.get("ugvs") or "/team/perception/ugvs", self.on_ugvs, 1)  # 약 1초마다 (시설물이 파괴되면 2 · 5초, 끊기기도 한다)
        self.create_subscription(ServerState, names["server_state"], self.on_server_state, latched())
        self.create_subscription(ServerCommand, names["server_command"], self.on_server_command, latched(2))
        self.create_subscription(NodeStatus, "/team/node_status", self.on_status, 10)
        # ---- 내보내기 · 부르기
        self.pub_control = self.create_publisher(ControlState, "/team/control/state", 1)
        self.pub_mission = self.create_publisher(TeamMissionState, "/team/mission/state", latched())
        self.pub_status = self.create_publisher(NodeStatus, "/team/node_status", 1)
        self.cli_classify = self.create_client(ClassifyImage, names["classify"])
        self.cli_ugvs = self.create_client(ReadUgvs, names["read_ugvs"])
        self.cli_sensors = self.create_client(LocateSensors, names["locate_sensors"])  # 수광부의 자리와 임자 (상대 UGV 따라가기 — 선택 사항)
        self.cli_ready = self.create_client(SetReady, names["set_ready"])
        self.cli_report = self.create_client(SubmitReport, names["submit_report"])
        self.act_nav = ActionClient(self, Navigate, names["navigate"])
        self.act_llm = ActionClient(self, Analyze, names["analyze"])
        # ---- 시킨 것의 상태
        self.goal = None  # (x, y, yaw) 지금 목표
        self.goal_no = 0
        self.goal_handle = None
        self.reached_no = 0  # 도착 알림을 받은 목표 번호
        self.pan_tilt = (0.0, cfg["gimbal"]["rest_tilt_deg"])  # 마지막으로 보낸 팬틸트 (도). 위치를 읽을 수 없으므로 이것이 곧 위치다
        self.gimbal_at = -1e9
        self.gimbal_ready_at = -1e9  # 이 시각부터 팬틸트가 보낸 각도에 가 있다고 본다
        self.gimbal_busy = None
        self.fire_busy = None
        self.photo_queue = deque()  # 판독에 보낼 사진 (tag, 사진, 위치, 팬)
        self.classify_busy = None
        self.ugv_busy = None
        self.sensor_busy = None  # (tag, 보낸 시각) — 수광부 자리를 물어 놓은 것
        self._watch = None  # 레이저 점 확인: {"tag", "off": 쏘기 전 사진, "on": 켜진 동안의 사진들, "from", "until"}
        self.sequence = 0
        self.revision = 0
        self._last_slow = -1e9
        self._last_task = ""
        self.server_phase = None  # 서버가 알려 준 단계 (/team/server/state)
        self._server_live = False  # 이 프로그램이 도는 동안 서버 단계가 ready · running · reporting 인 적이 있다
        self._recovered = False  # 단계를 처음 알게 된 때의 처리(저장 파일 지우기 · 읽기)를 했는가
        self._last_phase = None
        self.rec = Recorder(session_folder(cfg["record_dir"], self.run_id), dict(cfg, run_id=self.run_id))
        self.rec.clock = self.clock
        self.mission.photo_trace = self.rec.photo_record
        self._rec_at = -1e9
        self._saved_at = -1e9
        self._logged = 0
        self.parsed_pending = False
        self._standby_since = None
        self._deadline_said = False
        self._order_since = None
        self._fallback_tried = False
        if cfg.get("_config_error"):
            self.get_logger().error("설정 파일을 읽지 못해 기본값으로 돕니다 — %s" % cfg["_config_error"])
        self.get_logger().info("supervisor 켜짐 (run_id %s, 정찰 방식 %s, 기록 %s)" % (self.run_id, cfg["scout_mode"], self.rec.dir or "남기지 않음"))

    def note(self, name, make):
        """기록 파일에 한 줄 (recorder.py). make() 가 내용(dict)을 만든다 — 만들다가 예외가 나도 삼킨다. 기록 때문에 경기가 멈추면 안 된다."""
        try:
            self.rec.event(name, **make())
        except Exception:  # noqa: BLE001
            pass

    # ================================================================== 받기 (배경 스레드) — 빨리 끝낸다
    def on_match(self, m):
        self.mode = m.mode
        if m.cmd == "stop" and Time.from_msg(m.header.stamp) >= self.started:
            self.done = True  # 이 프로그램이 켜진 뒤에 온 stop 만 (공식 뼈대)
        self.cmd = m.cmd  # 경기 시계와 단계는 step() 이 derive_phase() 로 정한다
        self.note("match", lambda: dict(cmd=m.cmd, mode=m.mode))
        self.get_logger().info("경기 신호: %s (%s)" % (m.cmd, m.mode))

    def on_state(self, m):
        if self.hp is not None and int(m.hp) != self.hp:
            self.note("hp", lambda: dict(hp=int(m.hp), before=self.hp))
        self.hp = int(m.hp)
        self.battery = round(float(getattr(m, "battery_pct", -1.0)), 1)

    def on_image(self, m):
        w = self._watch  # 레이저 점 확인 중이면 켜진 동안의 사진을 모은다 (watch_laser)
        if w is not None and w["from"] is not None and w["from"] <= time.monotonic() <= w["until"] and len(w["on"]) < 8:
            w["on"].append(m)
        self.frame = m
        self._frame_at = time.monotonic()

    def on_pose(self, m):
        self.poses.append((stamp_s(m.header.stamp), m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation), time.monotonic()))

    def on_map(self, m):
        """첫 지도를 쓴다. 상자가 없는 지도(perception 의 임시 지도 — 벽만)도 받는다: 정찰은 못 해도 거점 진입은 한다.
        임시 지도로 시작했는데 상자가 든 지도가 오면 그것으로 바꾼다 (경기 150초 전까지만 — 그 뒤에는 하던 일을 흔들지 않는다)."""
        if m.header.frame_id != "arena" or not (m.width > 0.0 and m.height > 0.0):
            return
        t = self.clock()
        # 「임시 지도였는가」 는 처음 받은 지도의 상자 수로 본다 — self.field.boxes 에는 mission 이 라이다로 찾은 상자를 뒤에 넣을 수 있다
        upgrade = self.field is not None and not self._map_first and bool(m.boxes) and (t is None or t < 150.0)
        if self.field is None or upgrade:
            self.field = A.field_map_from_msg(m, A.arena_from(self.cfg["arena"]), self.cfg["base_kinds"])
            if upgrade:
                self.map_upgraded = True  # 다음 틱에 mission 이 지도 · 정찰 계획을 새로 만든다
            self._map_first = [(float(p.x), float(p.y)) for p in m.boxes]
            self._map_last, self.gone = tuple(self._map_first), None
            self.note("map", lambda: dict(boxes=[[round(x, 3), round(y, 3)] for x, y in self.field.boxes], bases={k: [list(v), self.field.base_rows.get(k)] for k, v in self.field.bases.items()},
                           size=[self.field.width, self.field.height, self.field.box_size]))
            return
        # 그 뒤의 지도: navigation 은 라이다로 「없다」 고 본 상자를 빼고 낸다 (남은 상자의 좌표 · 순서는 그대로, 번호만 당겨진다 — 다시 보이면 되살린다).
        # 지도를 다시 만들지 않고, 처음 지도의 상자 가운데 지금 지도에 없는 것의 번호만 가려낸다 (면 기록 · 정찰 자리가 처음 번호로 묶여 있다)
        now_boxes = tuple((float(p.x), float(p.y)) for p in m.boxes)
        if now_boxes == self._map_last or not self._map_first:
            return
        self._map_last = now_boxes
        gone = frozenset(i for i, (x, y) in enumerate(self._map_first) if not any(abs(x - a) < 0.02 and abs(y - b) < 0.02 for a, b in now_boxes))
        if len(gone) > max(3, len(self._map_first) // 2):
            return  # 절반 넘게 사라졌다면 지도가 통째로 달라진 것이다 (다른 사진으로 다시 만든 지도 등) → 따르지 않는다
        if gone != (self.gone or frozenset()):
            self.note("map_gone", lambda: dict(gone=sorted(gone), boxes=len(now_boxes), first=len(self._map_first)))
        self.gone = gone

    def on_lidar_boxes(self, m):
        """navigation 이 라이다로 찾았으나 지도에는 넣지 않은 상자들의 가운데. 가장 새 목록을 그대로 mission 에 넘긴다 (Facts.extra_boxes) —
        얼마나 오래 보였는지 · 받아들일지는 mission 이 정한다. 목록이 바뀐 때에만 기록에 남긴다."""
        if m.header.frame_id != "arena":
            return
        boxes = [(float(p.position.x), float(p.position.y)) for p in m.poses if math.isfinite(p.position.x) and math.isfinite(p.position.y)]
        if boxes != self.extra_boxes:
            self.note("lidar_boxes", lambda: dict(boxes=[[round(x, 3), round(y, 3)] for x, y in boxes]))
        self.extra_boxes = boxes

    def on_ugvs(self, m):
        """perception 이 드론 사진에서 찾은 UGV 자리들 (우리 차 포함). 요즘 관측 몇 개를 그대로 mission 에 넘긴다 (Facts.ugv_sights) —
        우리 차를 골라내고 「서 있는 로봇」 을 가리는 것은 mission 이 한다 (standing.Standing). 이 토픽이 없거나 끊겨도 아무 일도 없다
        (받은 적이 없으면 None, 오래된 관측은 mission 이 쓰지 않는다).

        관측의 때: header.stamp 는 그 사진을 server_bridge 가 받은 시각이다. 거기서 지금까지(판독에 걸린 시간)를 빼서 이 프로그램의 시계로 옮긴다.
        그 차이가 이상하면(음수 · 5초 넘음 — 시계가 다르다) 받은 때를 그대로 쓴다.
        """
        try:  # 덤으로 받는 것이다 — 여기서 난 예외로 메시지를 받는 스레드가 죽으면 안 된다
            if m.header.frame_id != "arena":
                return
            now = time.monotonic()
            late = self.get_clock().now().nanoseconds * 1e-9 - stamp_s(m.header.stamp)
            self.ugv_sights = add_sight(self.ugv_sights, now - late if 0.0 <= late <= 5.0 else now, m.poses)
            count = len(self.ugv_sights[-1][1])
            if count != getattr(self, "_ugv_count", None):  # 대수가 바뀐 때에만 기록에 남긴다 (자리 하나하나는 perception/ugvs.jsonl 에 있다)
                self._ugv_count = count
                self.note("ugvs_seen", lambda: dict(count=count, late_ms=round(1000.0 * late)))
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning("위에서 본 UGV 자리를 받다 예외: %s: %s" % (type(error).__name__, error), throttle_duration_sec=10.0)

    def on_boxes(self, m):
        boxes = [(p.position.x, p.position.y) for p in m.poses]
        if boxes:
            self.boxes = boxes
            self.note("boxes", lambda: dict(boxes=boxes))

    def on_server_state(self, m):
        if m.connected and m.phase:
            if m.phase != self.server_phase:
                self.note("server_phase", lambda: dict(phase=m.phase, slot=m.slot, ready=bool(m.ready)))
            self.server_phase = m.phase
        if m.team_assigned and m.slot:
            if m.slot != self.slot:
                self.note("slot", lambda: dict(slot=m.slot, server_phase=m.phase))
            self.slot = m.slot

    def on_server_command(self, m):
        if m.kind == ServerCommand.TACTICAL_ORDER and m.text.strip():
            if m.text != self.order_text:
                self.note("order", lambda: dict(text=m.text))
            self.order_text = m.text
        elif m.kind == ServerCommand.BASE_OBJECTIVE and m.objective_code.strip():
            if m.objective_code.strip() != self.objective:
                self.note("objective", lambda: dict(code=m.objective_code.strip()))
            self.objective = m.objective_code.strip()

    def on_status(self, m):
        name = m.node_name.strip("/")  # "/navigation" 으로 보내는 프로그램도, "navigation" 으로 보내는 프로그램도 있다
        if name != self.get_name():
            self.status[name] = (m.state, time.monotonic())

    def others_ready(self, now, phase):
        """준비완료를 보내도 되는가: 다른 프로그램과 카메라가 준비됐거나, 시작 대기가 ready_deadline_s 를 넘겼다."""
        def heard(name, usable):
            state, at = self.status.get(name.strip("/"), (None, 0.0))
            return state is not None and now - at <= 3.0 and (not usable or state in (NodeStatus.READY, NodeStatus.DEGRADED))

        missing = [n for n in self.cfg["require_nodes"] if not heard(n, True)] + [n for n in self.cfg["alive_nodes"] if not heard(n, False)]
        if self.frame is None:
            missing.append("카메라 사진")
        if phase != "standby":
            self._standby_since = None
            return not missing
        if self._standby_since is None:
            self._standby_since = now
        if missing and now - self._standby_since >= float(self.cfg["ready_deadline_s"]):
            if not self._deadline_said:
                self._deadline_said = True
                self.note("ready_deadline", lambda: dict(missing=missing, states=dict(self.status)))
                self.get_logger().warning("준비되지 않은 것이 있지만 시작 대기 %.0f초가 지나 준비완료를 보냅니다: %s" % (self.cfg["ready_deadline_s"], ", ".join(missing)))
            return True
        return not missing

    def recover(self, first_phase):
        """단계를 처음 알게 됐을 때: 새 경기면 저장 파일을 지우고, 경기 도중(진행 · 일시정지)에 다시 켜진 것이면 읽어 온다."""
        path = self.cfg["state_file"]
        if not path:
            return
        if first_phase not in ("running", "paused"):
            checkpoint.clear(path)
            return
        data = checkpoint.load(path)
        if data is None:
            return
        self.mission.restore(data.get("mission") or {})
        self.parsed = self.mission.m1
        self.slot = self.slot or data.get("slot")
        self._ran = float(data.get("clock") or 0.0) + (time.time() - float(data["saved_at"]))  # 꺼져 있던 동안에도 경기 시간은 흘렀다
        self._started = True
        self.get_logger().info("경기 도중에 다시 켜짐: 저장해 둔 상태를 읽음 (경기 시각 %.0f초)" % self._ran)

    # ================================================================== 한 틱 (메인 스레드)
    def clock(self):
        if not self._started:
            return None
        return self._ran + (time.monotonic() - self._run_since if self._run_since is not None else 0.0)

    def derive_phase(self):
        """지금 단계(idle · standby · running · paused · report · stop)와 그것을 정한 쪽("force" · "robot" · "server" · None = 아직 모름).

        경기 모드(mode = mission)에서는 로봇의 경기 신호(/mil/match)로만 정한다.
        연습 모드에서 로봇의 경기 신호가 없으면(idle) 서버가 알려 주는 단계를 따른다 — 참가팀 포털의 팀 테스트가 이렇다:
        로봇은 잠기지 않고 시작 신호도 주지 않으므로, 서버 단계가 ready 면 서서 준비완료를 보내고, running 이 되면 시작하고,
        reporting 이면 멈춰 M6 를 보내고, finished 면 끝낸다 (포털 안내: 「시작 전 · 보고 중에는 팀 코드가 스스로 멈춰 있어야 합니다」).
        """
        if self.cfg["force_start"]:
            return "running", "force"
        robot = {"start": "running", "resume": "running", "pause": "paused"}.get(self.cmd, self.cmd)
        if self.mode == "mission" or robot != "idle":
            return robot, "robot"
        server = self.server_phase
        if server in SERVER_PHASES:
            self._server_live = True
            return SERVER_PHASES[server], "server"
        if server == "finished" and self._server_live:
            return "stop", "server"  # 이 프로그램이 지켜본 시험이 끝났다 (켜질 때 이미 finished 였으면 지난 시험의 것 → idle)
        return "idle", ("server" if server is not None else ("robot" if self.mode else None))

    def phase(self):
        return self.derive_phase()[0]

    def follow_phase(self, now, phase, source):
        """단계가 바뀐 것을 경기 시계 · 저장 파일 · 종료에 반영한다."""
        if source is not None and not self._recovered:
            self._recovered = True
            self.recover("idle" if source == "force" else phase)  # force_start 로 돌릴 때는 지난 실행의 상태를 잇지 않는다
        if phase == "running":
            if self._run_since is None:
                self._run_since = now
            self._started = True
        elif self._run_since is not None:  # 일시정지 · 보고 · 종료 동안 시계는 멈춘다
            self._ran += now - self._run_since
            self._run_since = None
        if phase == "stop" and source == "server":
            self.done = True
        if (phase, source) != self._last_phase:
            self._last_phase = (phase, source)
            self.note("phase", lambda: dict(phase=phase, source=source, cmd=self.cmd, mode=self.mode, server_phase=self.server_phase))
            if source == "server":
                self.get_logger().info("단계: %s (서버 단계 %s 를 따름 — 로봇은 연습 모드)" % (phase, self.server_phase))

    def pose_now(self):
        if not self.poses or time.monotonic() - self.poses[-1][4] > 0.5:
            return None
        return self.poses[-1][1:4]

    def pose_at(self, t):
        """시각 t 에 가장 가까운 위치 (사진을 찍은 순간의 위치를 찾을 때)."""
        if not self.poses:
            return None
        return min(self.poses, key=lambda q: abs(q[0] - t))[1:4]

    def pose_between(self, t):
        """받은 때(이 프로그램의 시계) t 의 위치 — 앞뒤 두 값 사이를 곧게 이어서 (움직이는 중에 찍은 사진용). 앞뒤 값이 0.3초 넘게 떨어져 있으면 None.

        사진 머리의 시각은 쓰지 않는다: 로봇의 카메라는 켜진 뒤 흐른 시간(예 20293초)을, 위치는 벽시계를 적어 서로 견줄 수 없다
        (2026-11-19 실차: 그래서 지나가며 찍기가 한 장도 안 찍혔다). 사진도 위치도 「이 프로그램이 받은 때」 로 맞춘다 — 둘 다 0.1초쯤 늦게 온다.
        """
        older = before = after = None
        for q in self.poses:
            if q[4] <= t:
                older, before = before, q
            elif after is None:
                after = q
                break
        if before is None or t - before[4] > 0.3:
            return None
        if after is None:  # 사진이 마지막 위치 값보다 뒤에 왔다 → 마지막 두 값이 가던 대로 조금 더 간 것으로 친다
            if older is None or not 0.0 < before[4] - older[4] <= 0.3:
                return None
            before, after = older, before
        if after[4] - before[4] > 0.3 or after[4] <= before[4]:
            return None
        k = (t - before[4]) / (after[4] - before[4])
        turn = math.atan2(math.sin(after[3] - before[3]), math.cos(after[3] - before[3]))
        return (before[1] + k * (after[1] - before[1]), before[2] + k * (after[2] - before[2]), before[3] + k * turn)

    def step(self):
        now = time.monotonic()
        phase, source = self.derive_phase()
        self.follow_phase(now, phase, source)
        control = self.cfg["force_start"] or self.mode == "practice" or self.cmd in ("start", "resume")
        with self.lock:
            ev, self.events = self.events, {"answers": [], "ugvs": [], "sensors": [], "spots": [], "results": [], "fire": None, "go_failed": None, "no_turn": None}
        ready = self.others_ready(now, phase)
        if self.map_upgraded:
            self.map_upgraded = False
            self.mission.reset_map()
        f = Facts(now=now, t=self.clock(), phase=phase, control=control, slot=self.slot, pose=self.pose_now(), arrived=self.arrived(),
                  boxes=self.boxes, fmap=self.field, order_text=self.order_text, parsed=self.parsed, parsed_pending=self.parsed_pending, objective=self.objective, hp=self.hp,
                  nodes_ready=ready,
                  answers=ev["answers"], ugv_answers=ev["ugvs"], sensor_answers=ev["sensors"], laser_spots=ev["spots"], fire_reply=ev["fire"], results=ev["results"], go_failed=ev.get("go_failed"),
                  no_turn=ev.get("no_turn"), gone=self.gone, extra_boxes=self.extra_boxes, ugv_sights=self.ugv_sights)
        began = time.monotonic()
        task = self.mission.tick(f, self)
        # 판단 한 번(Mission.tick)에 걸린 시간 — 0.5초마다 적는 틱 줄에 그 사이의 가장 긴 값을 남긴다 (step 은 0.1초 안에 끝나야 한다)
        self._tick_ms = max(getattr(self, "_tick_ms", 0.0), 1000.0 * (time.monotonic() - began))
        self.pump_photos()
        self.pump_watch()
        if self.order_text and self.parsed is None and not self._fallback_tried:
            if self._order_since is None:
                self._order_since = now
            elif now - self._order_since >= self.cfg["m1_fallback_s"]:
                self.parse_locally()
        if self.cfg["state_file"] and phase == "running" and f.t is not None and now - self._saved_at >= 1.0:
            self._saved_at = now
            checkpoint.save(self.cfg["state_file"], {"clock": f.t, "slot": self.slot, "mission": self.mission.snapshot()})
        self.publish(now, phase, control, task)
        if now - self._rec_at >= 0.5:
            self._rec_at = now
            tick_ms, self._tick_ms = round(getattr(self, "_tick_ms", 0.0), 1), 0.0
            self.note("tick", lambda: dict(phase=phase, control=control, task=task, tick_ms=tick_ms, pose=None if f.pose is None else [round(f.pose[0], 3), round(f.pose[1], 3), round(math.degrees(f.pose[2]), 1)],
                           goal=self.goal, arrived=f.arrived, pan_tilt=[round(v, 1) for v in self.pan_tilt], hp=self.hp, battery=self.battery, waiting_photos=len(self.photo_queue), nodes_ready=ready,
                                           node_states={name: state for name, (state, _at) in self.status.items()}))
        for when, text in self.mission.log[self._logged:]:
            self.note("note", lambda text=text: dict(text=text))
            self.get_logger().info("[%s] %s" % ("%.1f" % when if when is not None else "-", text))
        self._logged = len(self.mission.log)

    def publish(self, now, phase, control, task):
        act = control and phase in ("running", "idle")
        c = ControlState()
        c.header.stamp = self.get_clock().now().to_msg()
        c.run_id, c.sequence = self.run_id, self.sequence
        c.motion_allowed = c.aiming_allowed = c.firing_allowed = bool(act)
        c.lease_ms, c.reason = 500, task
        self.sequence += 1
        self.pub_control.publish(c)
        if task != self._last_task or now - self._last_slow >= 1.0:
            if task != self._last_task:
                self.revision += 1
                if task.split(" ")[0] != self._last_task.split(" ")[0]:
                    t = self.clock()
                    self.get_logger().info("[%s] 하는 일: %s" % ("%.1f" % t if t is not None else "-", task.split(" ")[0]))
            self._last_task, self._last_slow = task, now
            m = TeamMissionState()
            m.header.stamp = c.header.stamp
            m.run_id, m.revision = self.run_id, self.revision
            m.phase = {"report": "reporting", "stop": "finished", "idle": "standby"}.get(phase, phase)
            m.slot = self.slot or ""
            m.camp = A.camp_of(self.slot) if self.slot else ""
            m.active_mission = {"scout": 3, "strike": 2, "stage": 5, "assault": 5, "report": 6}.get(task.split(" ")[0], 0)
            m.target_id = self.mission.designated or ""
            m.objective_id = self.objective or ""
            m.detail = task
            self.pub_mission.publish(m)
            s = NodeStatus()
            s.header.stamp = c.header.stamp
            s.run_id, s.node_name, s.state, s.detail = self.run_id, self.get_fully_qualified_name(), NodeStatus.READY, task
            self.pub_status.publish(s)

    # ================================================================== 시키는 창구 (mission.Ports)
    def arrived(self):
        """지금 목표에 도착해 멈췄는가. 목표를 준 적이 없으면 서 있는 것으로 본다."""
        return self.goal is None or self.reached_no == self.goal_no

    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None):
        goal = (round(x, 2), round(y, 2), round(yaw, 2), mode)
        if goal == self.goal:
            return
        if not self.act_nav.server_is_ready():
            return
        nav = self.cfg["nav"]
        g = Navigate.Goal()
        g.run_id = self.run_id
        g.goal.header.frame_id = "arena"
        g.goal.header.stamp = self.get_clock().now().to_msg()
        g.goal.pose.position.x, g.goal.pose.position.y = float(x), float(y)
        g.goal.pose.orientation.z, g.goal.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        g.mode = {"assault": Navigate.Goal.MODE_ASSAULT, "straight": Navigate.Goal.MODE_STRAIGHT}.get(mode, Navigate.Goal.MODE_NORMAL)
        g.ignore_yaw = bool(yaw_free)
        g.position_tolerance_m = float(pos_tol if pos_tol is not None else nav["position_tolerance_m"])
        g.yaw_tolerance_rad = 3.15 if yaw_free else float(yaw_tol if yaw_tol is not None else nav["yaw_tolerance_rad"])
        g.execution_timeout_ms = int(nav["timeout_ms"])
        here = self.pose_now()
        if here is not None:  # 가까운 목표에 긴 시한을 주면 못 가는 자리에서 그만큼 서 있게 된다 → 거리에 맞춘다 (설정값이 위쪽 끝)
            # 10초 + 1 m 에 5초 (2026-11-19 주행 통합 4번. 전: 15초 + 1 m 에 10초 = 0.10 m/s 가정, 실제 최고 0.45 m/s 라 2 m 목표에 35초).
            # 오전 실차 6판에서 자리에 와서 방향까지 맞은 31번이 모두 이 안이다 (8 + 거리/0.2 이면 1번이 잘림: 0.41 m 에 10.8초).
            # 시한 때 자리에서 방향만 맞추던 중이면 navigation 이 REACHED [YAW_NOT_MET] 로 끝낸다 (6583152)
            # 2026-11-19 저녁: 10초 + 1 m 에 10초로 늘림. navigation 이 막히면 스스로 돌아가게 됐고(0984297 · 43050e6 — 돌아가는 길은 곧은 길의
            # 2 ~ 3배, 좁은 틈은 0.15 m/s), 못 가는 자리에서는 시한을 기다리지 않고 5 ~ 8초 안에 BLOCKED 로 끝낸다. 오늘 실차 7판의 이동 175번 가운데
            # 시한으로 끝난 것은 2번뿐이고, 그 하나(모의 5차 140초)는 1.0 m 앞 목표를 2.9 m 돌아 0.84 m 남기고 움직이던 중에 15초 시한이 됐다
            g.execution_timeout_ms = min(g.execution_timeout_ms, int(1000 * (10.0 + math.hypot(x - here[0], y - here[1]) / 0.10)))
        self.goal = goal
        self.goal_no += 1
        number = self.goal_no
        tol_deg = round(math.degrees(g.yaw_tolerance_rad), 1)
        self.note("go", lambda: dict(no=number, x=round(x, 3), y=round(y, 3), yaw_deg=round(math.degrees(yaw), 1), mode=mode, yaw_free=bool(yaw_free), yaw_tol_deg=tol_deg))
        self.act_nav.send_goal_async(g).add_done_callback(lambda fut: self._nav_accepted(fut, number))

    def _nav_accepted(self, fut, number):
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.note("go_rejected", lambda: dict(no=number))
            if number == self.goal_no:
                self.goal = None  # 거절됨 → 다음 틱에 같은 목표를 다시 보낼 수 있게
            return
        if number == self.goal_no:
            self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f2: self._nav_result(f2, number))

    def _nav_result(self, fut, number):
        res = fut.result()
        if res is not None:
            fp = res.result.final_pose.pose
            self.note("go_result", lambda: dict(no=number, current=number == self.goal_no, outcome=int(res.result.outcome), detail=res.result.detail, final=[round(fp.position.x, 3), round(fp.position.y, 3)]))
        if res is None or number != self.goal_no:
            return  # 옛 목표의 알림은 버린다
        detail = res.result.detail or ""
        if res.result.outcome == Navigate.Result.REACHED:
            self.reached_no = number
            if says(detail, NO_TURN_AT_GOAL) and self.goal is not None:
                with self.lock:  # 자리에는 왔는데 도착 방향으로 돌 자리가 없다 → mission 이 같은 자리의 다른 방향을 또 시키지 않게
                    self.events["no_turn"] = (self.goal[0], self.goal[1])
        else:
            failed, self.goal = self.goal, None  # 못 갔다 → mission 이 다시 시키면 새로 보낸다
            if failed is not None and res.result.outcome not in (Navigate.Result.CANCELED, Navigate.Result.CONTROL_LOST):
                with self.lock:  # 시한 넘김 · 막힘 · 위치 잃음: mission 이 같은 자리를 되풀이해 시키지 않게 알려 준다
                    self.events["go_failed"] = (failed[0], failed[1], int(res.result.outcome), says(detail, NO_ROOM_TO_LEAVE))

    def stop(self):
        if self.goal is None:
            return
        handle, self.goal, self.goal_handle = self.goal_handle, None, None
        self.note("stop", lambda: dict(no=self.goal_no))
        if handle is not None:
            handle.cancel_goal_async()

    def _gimbal(self, pan_deg, tilt_deg):
        g = self.cfg["gimbal"]
        if self.cli_gimbal is None or not self.cli_gimbal.service_is_ready():
            return False
        if self.gimbal_busy is not None and not self.gimbal_busy.done():
            if time.monotonic() - self.gimbal_at < ANSWER_TIMEOUT:
                return False
        req = self._Gimbal.Request()
        req.pan = max(-100.0, min(100.0, pan_deg / g["pan_deg_per_unit"]))
        req.tilt = max(-100.0, min(100.0, tilt_deg / g["tilt_deg_per_unit"]))
        req.speed, req.acc = float(g["speed"]), float(g["acc"])
        target = (req.pan * g["pan_deg_per_unit"], req.tilt * g["tilt_deg_per_unit"])
        travel = max(abs(target[0] - self.pan_tilt[0]), abs(target[1] - self.pan_tilt[1]))
        self.pan_tilt = target
        self.gimbal_at = time.monotonic()
        self.gimbal_ready_at = self.gimbal_at + float(g["settle_s"]) + travel / max(1.0, float(g.get("rate_deg_s", 30.0)))
        self.gimbal_busy = self.cli_gimbal.call_async(req)
        self.gimbal_busy.add_done_callback(lambda fut, pan=req.pan, tilt=req.tilt: self._gimbal_done(fut, pan, tilt))
        return True

    def _gimbal_done(self, fut, pan, tilt):
        res = fut.result()
        self.note("gimbal", lambda: dict(pan_units=round(pan, 1), tilt_units=round(tilt, 1), pan_deg=round(self.pan_tilt[0], 1), tilt_deg=round(self.pan_tilt[1], 1),
                       reply=None if res is None else res.message, left_ms=None if res is None else int(res.left_ms)))

    def aim(self, pan_deg, tilt_deg):
        return self._gimbal(pan_deg, tilt_deg)

    def pan_now(self):
        """마지막으로 보낸 팬 각도 (°). mission 이 미리 돌려 둔 팬을 그대로 써도 되는지 볼 때 쓴다."""
        return self.pan_tilt[0]

    def aim_ready(self):
        """마지막으로 보낸 각도에 팬틸트가 가 있을 만큼 시간이 지났는가."""
        return time.monotonic() >= self.gimbal_ready_at

    def fire(self):
        if self.cli_fire is None or not self.cli_fire.service_is_ready():
            return False
        if self.fire_busy is not None and not self.fire_busy.done() and time.monotonic() - self._fire_at < ANSWER_TIMEOUT:
            return False
        self._fire_at = time.monotonic()
        pose = self.pose_now()
        self.note("fire", lambda: dict(pose=None if pose is None else [round(pose[0], 3), round(pose[1], 3), round(math.degrees(pose[2]), 1)], pan_tilt=[round(v, 1) for v in self.pan_tilt]))
        self.fire_busy = self.cli_fire.call_async(self._Fire.Request())
        self.fire_busy.add_done_callback(self._fire_done)
        return True

    def _fire_done(self, fut):
        res = fut.result()
        self.note("fire_reply", lambda: dict(reply=None if res is None else res.message, left_ms=None if res is None else int(res.left_ms)))
        w = self._watch
        if w is not None and w["from"] is None:
            if res is not None and res.message == "fired":
                now = time.monotonic()  # 켜지고 0.4초 뒤부터 1.8초 동안의 사진을 모은다 (레이저는 5초 켜진다)
                w["from"], w["until"] = now + 0.4, now + 2.2
            else:
                self._watch = None  # 쏘지 못했다
        if res is not None:
            with self.lock:
                self.events["fire"] = (res.message, int(res.left_ms))

    def watch_laser(self, tag, near=None):
        """다음 발사의 레이저 점이 사진 어디에 찍히는지 본다 (맞았는지 확인). 발사 직전에 부른다 — 지금 사진을 「쏘기 전」 으로 잡아 둔다.
        near = 식으로 계산한 레이저 점 (u, v) — 주면 그 둘레만 본다 (laser_spot.WINDOW_PX: 뒤에서 움직이는 사람을 레이저 점으로 잡지 않게).
        답은 Facts.laser_spots 로 간다: (tag, {"u", "v", "peak", "area"} 또는 None)."""
        if self.frame is None or time.monotonic() - self._frame_at > 0.5:
            return False
        self._watch = {"tag": tag, "off": self.frame, "on": [], "from": None, "until": None, "asked": time.monotonic(),
                       "near": None if near is None else (float(near[0]), float(near[1]))}
        return True

    def pump_watch(self):
        """모은 사진으로 레이저 점을 찾아 mission 에 알린다 (step 에서 부른다)."""
        w = self._watch
        if w is None:
            return
        now = time.monotonic()
        if w["from"] is None:
            if now - w["asked"] > 6.0:
                self._watch = None  # 발사 답이 오지 않았다
            return
        if now <= w["until"]:
            return
        self._watch = None
        spot = None
        try:
            from . import laser_spot
            off = laser_spot.image_rgb(w["off"])
            on = [img for img in (laser_spot.image_rgb(m) for m in w["on"]) if img is not None]
            if off is not None and on:
                spot = laser_spot.find_spot([off], on, near=w.get("near"))
        except Exception as error:  # 사진 계산이 실패해도 경기는 계속한다
            self.get_logger().warning("레이저 점 확인 실패: %s" % error)
        tag, frames = w["tag"], len(w["on"])
        near = w.get("near")
        self.note("laser_spot", lambda: dict(tag=[str(v) for v in tag], frames=frames, spot=spot, near=near))
        with self.lock:
            self.events["spots"].append((tag, spot))

    def shoot(self, tag, pan_deg):
        """팬을 맞추고(다르면 먼저 돌리고 기다린다) 지금 사진 한 장을 판독 줄에 넣는다."""
        if len(self.photo_queue) >= 6 or self.frame is None:
            return False
        g = self.cfg["gimbal"]
        if abs(pan_deg - self.pan_tilt[0]) > float(g.get("reaim_deg", 4.0)) or abs(g["rest_tilt_deg"] - self.pan_tilt[1]) > 0.6:
            if self.cli_gimbal is not None:
                self._gimbal(pan_deg, g["rest_tilt_deg"])
                return False
            self.pan_tilt = (pan_deg, g["rest_tilt_deg"])  # 팬틸트가 없는 곳(시험): 돌린 것으로 친다
        if time.monotonic() < self.gimbal_ready_at:
            return False
        frame = self.frame
        shot_at = stamp_s(frame.header.stamp)
        # 사진의 시각과 위치의 시각이 같은 시계일 때만 「찍은 순간의 위치」 를 찾는다. 아니면(2초 넘게 다르면) 지금 위치를 쓴다 — 서서 찍으므로 거의 같다
        pose = self.pose_at(shot_at) if self.poses and abs(self.poses[-1][0] - shot_at) <= 2.0 else self.pose_now()
        if pose is None:
            return False
        name = self.rec.save_photo(tag, frame) if self.cfg["save_photos"] and photos_allowed() else None
        self.note("photo", lambda: dict(tag=[str(v) for v in tag], file=name, pose=[round(pose[0], 3), round(pose[1], 3), round(math.degrees(pose[2]), 1)], pan_deg=round(self.pan_tilt[0], 1),
                       stamp=round(shot_at, 3)))
        self.photo_queue.append((tag, frame, pose, self.pan_tilt[0]))
        return True

    def shoot_aim(self, tag):
        """조준 확인 사진: 팬틸트를 건드리지 않고 지금 사진 한 장을 판독 줄에 넣는다 (mission 의 사진 조준 — aim.py).

        답은 다른 사진과 같은 길로 오되, 그림의 자리 · 크기 전부와 수광부 탐지를 담는다 (_classified).
        팬틸트가 다 돈 뒤에 들어온 사진만 쓴다 — 도는 중의 사진으로 고치면 그만큼을 또 돌리게 된다.
        """
        now = time.monotonic()
        if len(self.photo_queue) >= 6 or self.frame is None or now < self.gimbal_ready_at:
            return False
        if self._frame_at < self.gimbal_ready_at or now - self._frame_at > 0.5:
            return False  # 팬틸트가 멈춘 뒤의 새 사진이 아직 없다 (카메라가 끊겼으면 mission 이 시한 뒤 계산한 각도로 쏜다)
        frame = self.frame
        pose = self.pose_now()
        name = self.rec.save_photo(tag, frame) if self.cfg["save_photos"] and photos_allowed() else None
        self.note("photo", lambda: dict(tag=[str(v) for v in tag], file=name, pose=None if pose is None else [round(pose[0], 3), round(pose[1], 3), round(math.degrees(pose[2]), 1)],
                       pan_deg=round(self.pan_tilt[0], 1), tilt_deg=round(self.pan_tilt[1], 1), stamp=round(stamp_s(frame.header.stamp), 3)))
        self.photo_queue.append((tag, frame, pose, self.pan_tilt[0]))
        return True

    def shoot_pass(self, tag):
        """지나가며 찍기: 팬틸트를 건드리지 않고 지금 사진 한 장을 판독 줄에 넣는다. 판독 줄이 비어 있을 때만 (정찰 · 조준 사진을 밀어내지 않는다).

        mission 은 이 답을 면 기록의 「본 것」 으로 쓴다 (mission._pass_votes → scout.on_pass: 다른 자리의 사진과 같은 답일 때만 적는다).
        기록의 photo 줄(사진 시각의 위치 · 팬)과 classify 줄(그림의 자리 · 크기)로 나중에 견줄 수 있다.
        """
        now = time.monotonic()
        if self.photo_queue or self.classify_busy is not None or self.frame is None or now < self.gimbal_ready_at or now - self._frame_at > 0.3:
            return False
        frame = self.frame
        shot_at = stamp_s(frame.header.stamp)
        pose = self.pose_between(self._frame_at)
        if pose is None:
            return False
        name = self.rec.save_photo(tag, frame) if self.cfg["save_photos"] and photos_allowed() else None
        self.note("photo", lambda: dict(tag=[str(v) for v in tag], file=name, pose=[round(pose[0], 3), round(pose[1], 3), round(math.degrees(pose[2]), 1)],
                       pan_deg=round(self.pan_tilt[0], 1), tilt_deg=round(self.pan_tilt[1], 1), stamp=round(shot_at, 3)))
        self.photo_queue.append((tag, frame, pose, self.pan_tilt[0]))
        return True

    def pump_photos(self):
        """판독은 한 번에 하나씩 보낸다 (줄 세워 두고 답이 오면 다음 것)."""
        if self.classify_busy is not None:
            if time.monotonic() - self.classify_busy[1] < 6.0:
                return
            tag, _at, pose, pan = self.classify_busy  # 답이 없다 → 버린다
            self.classify_busy = None
            self.note("classify_timeout", lambda: dict(tag=[str(v) for v in tag]))
            with self.lock:
                self.events["answers"].append((tag, pose, pan, None))
        if not self.photo_queue or not self.cli_classify.service_is_ready():
            return
        tag, frame, pose, pan = self.photo_queue.popleft()
        req = ClassifyImage.Request()
        req.run_id, req.image = self.run_id, frame
        self.classify_busy = (tag, time.monotonic(), pose, pan)
        self.cli_classify.call_async(req).add_done_callback(lambda fut: self._classified(fut, tag, pose, pan))

    def _classified(self, fut, tag, pose, pan):
        res = fut.result()
        pictures = None
        if res is not None and res.ok:
            # 정찰 사진: 그림마다 (종류, 가로 자리, 잘림, 가로 크기, 세로 크기, 확신). 크기는 그 그림이 어느 면의 것인지 가릴 때 쓴다 (scout.pick_each)
            pictures = [(p.kind, float(p.x_center), bool(p.cut_off), float(p.width), float(p.height), float(p.confidence)) for p in res.pictures]
            if tag and tag[0] == "aim":  # 조준 확인 사진: 그림의 자리 · 크기 전부와, perception 이 detail 에 실어 준 수광부 탐지를 넘긴다
                pictures = {"pictures": [{"kind": p.kind, "x": float(p.x_center), "y": float(p.y_center), "w": float(p.width), "h": float(p.height),
                                          "conf": float(p.confidence), "cut": bool(p.cut_off)} for p in res.pictures],
                            "sensors": parse_sensors(res.detail)}
        took = time.monotonic() - self.classify_busy[1] if self.classify_busy is not None and self.classify_busy[0] == tag else None
        self.note("classify", lambda: dict(tag=[str(v) for v in tag], ok=bool(res is not None and res.ok), detail=None if res is None else res.detail, took_s=None if took is None else round(took, 2),
                       pictures=None if res is None else [{"kind": p.kind, "x": round(p.x_center, 3), "y": round(p.y_center, 3), "w": round(p.width, 3), "h": round(p.height, 3),
                                                           "conf": round(p.confidence, 2), "cut": bool(p.cut_off)} for p in res.pictures]))
        with self.lock:
            self.events["answers"].append((tag, pose, pan, pictures))
        if self.classify_busy is not None and self.classify_busy[0] == tag:
            self.classify_busy = None

    def look_ugvs(self):
        if self.frame is None or self.classify_busy is not None or self.photo_queue or not self.cli_ugvs.service_is_ready():
            return False
        if self.ugv_busy is not None and not self.ugv_busy.done() and time.monotonic() - self._ugv_at < 6.0:
            return False
        req = ReadUgvs.Request()
        req.run_id, req.image = self.run_id, self.frame
        # 판독에 보낸 사진을 남긴다 — 호출부호 · 목숨이 왜 안 읽혔는지 나중에 사진으로 볼 수 있게 (2026-11-20 07시 판은 사진이 없어 볼 수 없었다)
        self._ugv_photo_no = getattr(self, "_ugv_photo_no", 0) + 1
        tag = ("ugvs", self._ugv_photo_no)
        name = self.rec.save_photo(tag, self.frame) if self.cfg["save_photos"] and photos_allowed() else None
        if name:
            self.note("ugv_photo", lambda: dict(tag=[str(v) for v in tag], file=name, pan_deg=round(self.pan_tilt[0], 1)))
        self._ugv_at = time.monotonic()
        self.ugv_busy = self.cli_ugvs.call_async(req)
        self.ugv_busy.add_done_callback(self._ugvs_read)
        return True

    def _ugvs_read(self, fut):
        res = fut.result()
        if res is not None and res.ok:
            seen = [{"camp": u.camp, "callsign": u.callsign, "lives": int(u.lives), "ammo": int(u.ammo)} for u in res.ugvs]
            if seen:
                self.note("ugvs", lambda: dict(ugvs=seen))
            with self.lock:
                self.events["ugvs"] += seen

    def locate(self, tag):
        """지금 사진에서 수광센서 자리를 perception 에 묻는다 (상대 UGV 따라가기 — mission.engage_ugv). 답은 Facts.sensor_answers 로 간다: (tag, 찍을 때의 팬, 틸트, 센서 목록 또는 None).

        팬틸트를 돌린 뒤에 받은 사진만 보낸다 — 도는 중의 사진으로 계산하면 「찍을 때의 팬틸트」 가 틀려 엉뚱한 쪽으로 고친다.
        """
        now = time.monotonic()
        if self.frame is None or not self.cli_sensors.service_is_ready():
            return False
        if self.sensor_busy is not None and now - self.sensor_busy[1] < 3.0:
            return False
        if now < self.gimbal_ready_at or self._frame_at < self.gimbal_ready_at or now - self._frame_at > 0.5:
            return False  # 팬틸트가 아직 돌고 있거나, 돈 뒤의 사진이 아직 안 왔거나, 사진이 끊겼다
        req = LocateSensors.Request()
        req.run_id, req.image = self.run_id, self.frame
        pan, tilt = self.pan_tilt
        self.sensor_busy = (tag, now)
        self.cli_sensors.call_async(req).add_done_callback(lambda fut: self._sensors_located(fut, tag, pan, tilt))
        return True

    def _sensors_located(self, fut, tag, pan, tilt):
        res = fut.result()
        spots = None
        if res is not None and res.ok:
            spots = [{"owner": s.owner, "x_center": float(s.x_center), "y_center": float(s.y_center), "width": float(s.width), "height": float(s.height),
                      "confidence": float(s.confidence), "has_host": bool(s.has_host), "host_kind": s.host_kind, "camp": s.camp, "callsign": s.callsign,
                      "host_x_center": float(s.host_x_center), "host_y_center": float(s.host_y_center), "host_width": float(s.host_width), "host_height": float(s.host_height)}
                     for s in res.sensors]
        took = time.monotonic() - self.sensor_busy[1] if self.sensor_busy is not None and self.sensor_busy[0] == tag else None
        self.note("sensors", lambda: dict(tag=[str(v) for v in tag], ok=spots is not None, detail=None if res is None else res.detail, took_s=None if took is None else round(took, 2),
                       pan_tilt=[round(pan, 1), round(tilt, 1)],
                       sensors=None if spots is None else [{"owner": s["owner"], "x": round(s["x_center"], 4), "y": round(s["y_center"], 4), "w": round(s["width"], 4), "h": round(s["height"], 4),
                                                            "conf": round(s["confidence"], 2), "host": s["host_kind"], "camp": s["camp"]} for s in spots]))
        with self.lock:
            self.events["sensors"].append((tag, pan, tilt, spots))
        if self.sensor_busy is not None and self.sensor_busy[0] == tag:
            self.sensor_busy = None

    def analyze_order(self, text):
        if not self.act_llm.server_is_ready():
            return False
        g = Analyze.Goal()
        g.run_id, g.task_id, g.kind = self.run_id, "m1-%d" % int(time.monotonic() * 10), Analyze.Goal.PARSE_ORDER
        g.tactical_command, g.execution_timeout_ms = text, 8000
        self.act_llm.send_goal_async(g).add_done_callback(self._llm_accepted)
        return True

    def _llm_accepted(self, fut):
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.note("m1_ask_rejected", dict)
            return  # mission 이 m1_reask_s 뒤에 다시 묻는다
        handle.get_result_async().add_done_callback(self._llm_result)

    def _llm_result(self, fut):
        res = fut.result()
        if res is None or not res.result.success:
            self.note("m1_failed", lambda: dict(code=None if res is None else res.result.error_code, detail=None if res is None else res.result.detail))
            return
        c = res.result.command
        self.parsed = {"target": c.target, "search_sectors": list(c.search_sectors), "base": c.base}
        self.parsed_pending = res.result.error_code in ("LLM_PENDING", "LLM_FAILED")  # 낱말 해석만으로 낸 답이다 → mission 이 다시 물어본다
        self.note("m1_parsed", lambda: dict(parsed=self.parsed, code=res.result.error_code, detail=res.result.detail))

    def parse_locally(self):
        """llm_worker 가 답하지 않을 때의 마지막 길: 같은 작업공간에 설치된 llm_worker 의 낱말 해석기를 직접 부른다 (LLM 은 쓰지 않는다)."""
        self._fallback_tried = True
        try:
            from llm_worker import interpret, parser

            body = interpret.payload(parser.parse(self.order_text, A.camp_of(self.slot)))
        except Exception as error:  # noqa: BLE001 — 설치돼 있지 않거나 해석기가 실패해도 경기는 계속한다
            why = type(error).__name__
            self.note("m1_fallback_failed", lambda: dict(error=why))
            self.get_logger().error("llm_worker 의 답이 없고, 낱말 해석기도 직접 부르지 못함 (%s)" % why)
            return
        if body and self.parsed is None:
            self.parsed = {"target": body.get("target", ""), "search_sectors": list(body.get("search_sectors", [])), "base": body.get("base", "")}
            self.parsed_pending = True  # llm_worker 가 뒤늦게 답하면 그것으로 바꾼다
            self.note("m1_parsed", lambda: dict(parsed=self.parsed, code="LOCAL_FALLBACK", detail="llm_worker 의 답이 없어 직접 해석"))
            self.get_logger().warning("llm_worker 의 답이 없어 낱말 해석기를 직접 불렀습니다")

    def set_ready(self):
        if not self.cli_ready.service_is_ready():
            return False
        req = SetReady.Request()
        req.run_id = self.run_id
        self.note("report", lambda: dict(kind="ready"))
        self.cli_ready.call_async(req).add_done_callback(lambda fut: self._reported(fut, "ready"))
        return True

    def report(self, kind, payload):
        if not self.cli_report.service_is_ready():
            return False
        req = SubmitReport.Request()
        req.run_id = self.run_id
        r = Report()
        r.report_id, r.revision, r.kind = "m%d" % kind, self.revision, kind
        if kind == 1:
            r.command = ParsedCommand(target=payload.get("target") or "", search_sectors=list(payload.get("search_sectors") or []), base=payload.get("base") or "")
        elif kind == 3:
            r.sectors = [SectorCount(sector=name, complete=True, objects=[ObjectCount(type=k, count=int(n)) for k, n in sorted(objs.items())]) for name, objs in sorted(payload.items())]
        elif kind == 6:
            r.assessment = ThreatAssessment(ranking=list(payload), rationale="")
        req.report = r
        self.note("report", lambda: dict(kind=kind, payload=payload))
        self.cli_report.call_async(req).add_done_callback(lambda fut: self._reported(fut, kind))
        return True

    def _reported(self, fut, kind):
        res = fut.result()
        ok = res is not None and res.outcome == 1  # ACCEPTED
        self.note("report_result", lambda: dict(kind=kind, outcome=None if res is None else int(res.outcome), http=None if res is None else int(res.http_status), detail=None if res is None else res.detail))
        with self.lock:
            self.events["results"].append((kind, ok))


def spin_forever(node):
    """배경 스레드: 메시지를 받아 콜백을 실행한다. 콜백에서 예외가 나도 받기가 멈추지 않게 다시 돈다 (공식 뼈대)."""
    while rclpy.ok():
        try:
            rclpy.spin(node)
        except Exception:  # noqa: BLE001
            if not rclpy.ok():
                break
            node.get_logger().error("콜백 예외 (받기는 계속):\n" + traceback.format_exc(), throttle_duration_sec=2.0)


def main():
    rclpy.init()
    cfg = load_config(os.environ.get("SUPERVISOR_CONFIG", os.path.join(os.getcwd(), "supervisor.json")))
    node = SupervisorNode(cfg)
    thread = threading.Thread(target=spin_forever, args=(node,), daemon=True)
    thread.start()
    try:
        while rclpy.ok() and not node.done:
            t0 = time.time()
            try:
                node.step()
            except Exception:  # noqa: BLE001 — 한 틱의 예외로 프로그램이 끝나지 않게
                node.get_logger().error("step() 예외 (계속 실행):\n" + traceback.format_exc(), throttle_duration_sec=2.0)
            time.sleep(max(0.0, PERIOD - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
