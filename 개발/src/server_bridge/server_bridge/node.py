"""server_bridge 의 ROS 연결: 경기 서버(HTTP)와 팀 프로그램 사이를 잇는다. 판단은 하지 않는다.

  상태      GET /me 를 계속 봐서 /team/server/state 로 내보낸다 (연결 여부 · 우리 자리 · 진영 · 단계)
  준비완료  서비스 /team/server/set_ready → POST /ready 한 번
  임무      GET /order (1차 전술명령) · GET /target (2차 목표거점)을 서버가 줄 때까지 걸어 두고, 받는 즉시 /team/server/command 로
  보고      서비스 /team/server/submit_report → M1 · M3 · M6 를 한 번 보내고 서버의 답 그대로 (ACCEPTED · REJECTED · UNKNOWN).
            다시 보낼지는 supervisor 가 정한다
  드론      진행 중 GET /drone 을 0.5초마다. 새 장만 /team/server/uav_image 로

서버 주소와 임무코드는 비밀 파일(secrets.json) 또는 환경변수에서 읽는다. 임무코드는 기록 어디에도 찍지 않는다.
본 경기는 서버의 포트 8000, 참가팀 포털의 팀 테스트는 같은 주소의 시험 포트 8100 이다 (경로 · 형식은 같다).
port_mode 가 "auto" 면 두 포트의 GET /me 를 보고 우리 팀의 경기가 열린 쪽을 스스로 고른다 (둘 다 열려 있으면 본 경기 쪽).
설정을 바꾸지 않고 같은 프로그램으로 팀 테스트와 본 경기를 모두 치르기 위해서다.
HTTP 는 전부 따로 도는 스레드와 여러 스레드 실행기에서 한다 (한 요청이 다른 것을 막지 않게).

기록: 서버와 주고받은 것을 <기록 폴더>/http.jsonl 에 한 줄씩, 드론 화면은 drone_save_s 초마다 한 장을 <기록 폴더>/drone/ 에 남긴다.
기록 폴더는 team_node.py 가 켠 경우 ~/team/log/<날짜_시각>/server_bridge/, 따로 켠 경우 log/server_bridge/<날짜_시각>/.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.parse

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from team_interfaces.msg import MissionState, NodeStatus, Report, ServerCommand, ServerState, UavFrame
from team_interfaces.srv import SetReady, SubmitReport

from . import constants as C
from . import reports as R
from .client import ServerClient, detail_text

DEFAULTS = {
    "url": "",  # 예 "http://192.168.0.10:8000" — secrets.json 의 server_url 이나 환경변수 MAICON_SERVER_URL 로도 준다
    "port_mode": "auto",  # auto(경기가 열린 쪽) · match(url 그대로) · test(시험 포트만)
    "test_port": C.TEST_PORT,  # 팀 테스트용 시험 포트
    "wait_s": 60,  # 임무를 걸어 두는 시간 (서버 상한 120)
    "me_period_s": 1.0,
    "timeout_s": 4.0,
    "drone": True,
    "drone_poll_s": 0.5,
    "drone_camera": "",  # 연습 기간의 팀 테스트에서 쓸 카메라: "test1" · "test2" (로봇이 있는 연습 경기장). 시험 포트를 쓰는 동안에만 붙이고 본 경기 포트에는 붙이지 않는다
    "record": True,  # 서버와 주고받은 것을 파일로 남긴다
    "drone_save_s": 10.0,  # 드론 화면을 이 간격으로 한 장씩 파일로 남긴다 (0 이면 새 화면마다, 음수면 남기지 않는다)
}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_config(folder):
    cfg = dict(DEFAULTS)
    cfg.update(read_json(os.environ.get("SERVER_BRIDGE_CONFIG", os.path.join(folder, "server_bridge.json"))))
    secrets = read_json(os.environ.get("TEAM_SECRETS", os.path.join(folder, "secrets.json")))
    cfg["url"] = os.environ.get("MAICON_SERVER_URL") or secrets.get("server_url") or cfg["url"]
    cfg["code"] = os.environ.get("MAICON_MISSION_CODE") or secrets.get("mission_code") or ""
    return cfg


class HttpLog:
    """서버와 주고받은 것을 한 줄씩 남긴다. 임무코드가 들어 있으면 지운다."""

    def __init__(self, folder, code):
        self.dir = None
        self.code = code or ""
        self.lock = threading.Lock()
        self.fh = None
        if folder:
            try:
                os.makedirs(os.path.join(folder, "drone"), exist_ok=True)
                self.dir = folder
            except OSError:
                pass

    def write(self, **record):
        if self.dir is None:
            return
        try:
            line = json.dumps(dict(record, wall=round(time.time(), 3)), ensure_ascii=False, default=str)
            if len(self.code) >= 4:
                line = line.replace(self.code, "***")
            with self.lock:
                if self.fh is None:
                    self.fh = open(os.path.join(self.dir, "http.jsonl"), "a", encoding="utf-8")
                self.fh.write(line + "\n")
                self.fh.flush()
        except (OSError, TypeError, ValueError):
            pass

    def save_drone(self, number, jpeg):
        if self.dir is None:
            return None
        try:
            name = "%04d.jpg" % number
            with open(os.path.join(self.dir, "drone", name), "wb") as fh:
                fh.write(jpeg)
            return name
        except OSError:
            return None


def record_folder(enabled, name="server_bridge"):
    if not enabled or os.environ.get("TEAM_LOG_OFF") == "1":
        return None
    shared = os.environ.get("TEAM_LOG_DIR")
    return os.path.join(shared, name) if shared else os.path.join("log", name, time.strftime("%Y%m%d_%H%M%S"))


def test_base(url, port):
    """서버 주소에서 포트만 바꾼 주소 (http://192.168.0.10:8000 → http://192.168.0.10:8100)."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:  # IPv6
        host = "[%s]" % host
    return urllib.parse.urlunsplit((parts.scheme or "http", "%s:%d" % (host, port), parts.path.rstrip("/"), "", ""))


def latched(depth=1):
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ServerBridge(Node):
    def __init__(self):
        super().__init__("server_bridge")
        self.cfg = cfg = load_config(os.getcwd())
        self.client = self.other = None  # 지금 쓰는 서버와, auto 일 때 같이 지켜보는 다른 포트
        self.which = "match"
        if cfg["url"] and cfg["code"]:
            main = ServerClient(cfg["url"], C.PREFIX_REPORT, cfg["code"], timeout=float(cfg["timeout_s"]))
            test = main.with_base(test_base(cfg["url"], int(cfg["test_port"])))
            if cfg["port_mode"] == "test":
                self.client, self.which = test, "test"
            else:
                self.client = main
                self.other = test if cfg["port_mode"] == "auto" and test.base != main.base else None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.run_id = ""
        self.phase = self.slot = self.camp = None
        self.ready = False
        self.connected = False
        self.match_seen = False
        self.order_text = None
        self.objective = None
        self._why = {}
        self.log = HttpLog(record_folder(cfg["record"]), cfg["code"])
        self.began = time.monotonic()
        self.light = os.environ.get("TEAM_LOG_LIGHT") == "1"  # 디스크가 모자라면 team_node.py 가 켠다 → 화면은 남기지 않는다
        self._me_logged = (None, -1e9)
        self._drone_n = 0
        self._drone_saved_at = -1e9
        group = ReentrantCallbackGroup()
        self.pub_state = self.create_publisher(ServerState, "/team/server/state", latched())
        self.pub_command = self.create_publisher(ServerCommand, "/team/server/command", latched(2))
        self.pub_uav = self.create_publisher(UavFrame, "/team/server/uav_image", QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.pub_status = self.create_publisher(NodeStatus, "/team/node_status", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(MissionState, "/team/mission/state", self.on_mission, latched(), callback_group=group)
        self.create_service(SetReady, "/team/server/set_ready", self.on_set_ready, callback_group=group)
        self.create_service(SubmitReport, "/team/server/submit_report", self.on_submit_report, callback_group=group)
        self.create_timer(1.0, self.publish_slow, callback_group=group)
        if self.client is None:
            self.get_logger().error("서버 주소나 임무코드가 없습니다 (secrets.json 의 server_url · mission_code). 서버와 통신하지 않습니다")
        else:
            self.get_logger().info("server_bridge 켜짐 (서버 %s, 포트 고르기 %s%s)" % (self.client.base, cfg["port_mode"], " · 시험 포트 %s" % cfg["test_port"] if self.other else ""))
            for name, target in (("status", self.status_loop), ("order", self.order_loop), ("target", self.target_loop), ("drone", self.drone_loop)):
                threading.Thread(target=self.guard, args=(name, target), name="bridge-" + name, daemon=True).start()

    # ------------------------------------------------------------------ 공통
    def say(self, text):
        self.get_logger().info(text)

    def guard(self, name, target):
        """스레드 하나가 예외로 죽어도 다시 돈다 (경기 중에는 사람이 고쳐 줄 수 없다)."""
        while not self.stop_event.is_set():
            try:
                target()
                return
            except Exception as error:  # noqa: BLE001
                self.get_logger().error("%s 스레드 예외 %s — 1초 뒤 다시" % (name, type(error).__name__))
                self.stop_event.wait(1.0)

    def http(self, method, endpoint, params=None, body=None, timeout=None):
        """서버에 한 번 묻고 기록한다. GET /me 는 답이 바뀌었을 때와 5초마다만 적는다 (1초에 여러 번 묻는다)."""
        began = time.monotonic()
        status, out = self.client.call(method, endpoint, params, body, timeout)
        took = time.monotonic() - began
        if endpoint == C.EP_ME:
            key = (status, json.dumps(out, sort_keys=True, ensure_ascii=False, default=str) if out is not None else None)
            if key == self._me_logged[0] and began - self._me_logged[1] < 5.0:
                return status, out
            self._me_logged = (key, began)
        self.log.write(e="http", method=method, endpoint=endpoint, server=self.which, status=status, took_s=round(took, 3), params=params, body=body, out=out)
        return status, out

    def on_mission(self, m):
        self.run_id = m.run_id

    def stamp(self):
        return self.get_clock().now().to_msg()

    def publish_state(self):
        # run_id 는 supervisor 의 /team/mission/state 에서 받는다. 켜진 직후 그것을 받기 전에 빈 run_id 로 내보내면,
        # 곧 값이 바뀌어 보여 받는 쪽(navigation_slam)이 실행이 바뀐 줄 알고 위치를 초기화한다 → 2초까지는 기다린다
        if not self.run_id and time.monotonic() - self.began < 2.0:
            return
        with self.lock:
            phase, slot, camp, ready, connected = self.phase, self.slot, self.camp, self.ready, self.connected
        s = ServerState()
        s.header.stamp = self.stamp()
        s.run_id, s.connected, s.team_assigned = self.run_id, bool(connected), slot is not None
        s.slot, s.camp, s.phase, s.ready = slot or "", camp or "", phase or "", bool(ready)
        s.detail = self.client.note() if self.client is not None else "서버 주소 · 임무코드 없음"
        self.pub_state.publish(s)

    def publish_slow(self):
        self.publish_state()
        n = NodeStatus()
        n.header.stamp = self.stamp()
        n.run_id, n.node_name = self.run_id, self.get_fully_qualified_name()
        n.state = NodeStatus.READY if self.client is not None else NodeStatus.ERROR
        n.detail = self.client.note() if self.client is not None else "서버 주소 · 임무코드 없음"
        self.pub_status.publish(n)

    def running(self):
        with self.lock:
            return self.phase == C.PHASE_RUNNING

    # ------------------------------------------------------------------ 상태 (GET /me)
    def status_loop(self):
        last, fails = None, 0
        while not self.stop_event.is_set():
            status, me = self.http("GET", C.EP_ME)
            if self.other is not None and not self.match_seen:
                status, me = self.pick_server(status, me)
            if status != 200 or not isinstance(me, dict):
                fails += 1
                with self.lock:
                    self.connected = fails < 3
                if fails in (3, 10) or fails % 30 == 0:
                    self.say("GET /me 실패가 이어짐 (%d번째 · %s %s) — 계속 다시 시도 · %s" % (fails, status, detail_text(me), self.client.note()))
                self.stop_event.wait(0.2 if fails <= 2 else 1.0)
                continue
            fails = 0
            phase = me.get("phase")
            slot = R.clean_slot(me.get("slot"))
            camp = C.camp_of(slot) if slot else (me.get("camp") if me.get("camp") in C.CAMPS else None)
            with self.lock:
                self.connected = True
                if phase in (C.PHASE_READY, C.PHASE_RUNNING, C.PHASE_REPORTING):
                    self.match_seen = True
                if phase in (C.PHASE_IDLE, C.PHASE_WAITING) and last not in (None, C.PHASE_IDLE, C.PHASE_WAITING):
                    self.order_text = self.objective = None  # 새 경기를 기다리는 단계로 돌아갔다 → 지난 경기의 임무를 잊는다
                self.phase, self.slot, self.camp, self.ready = phase, slot, camp, bool(me.get("ready"))
            if phase != last:
                self.say("서버 단계 %s · 자리 %s" % (phase, slot))
                last = phase
                self.publish_state()
            self.stop_event.wait(0.3 if phase in (C.PHASE_READY, C.PHASE_REPORTING) else float(self.cfg["me_period_s"]))

    def pick_server(self, status, me):
        """본 경기 포트와 시험 포트 중 우리 팀의 경기가 열린 쪽을 고른다. 경기를 한 번 보면(ready 이후) 더 바꾸지 않는다.

        지금 쓰는 쪽이 열려 있으면 그대로. 아니면 다른 쪽을 물어 보고, 그쪽이 열려 있으면 바꾼다. 둘 다 열려 있으면 본 경기 쪽.
        돌려주는 값: 고른 쪽의 (상태 코드, GET /me 의 답).
        """
        def live(st, out):
            return st == 200 and isinstance(out, dict) and out.get("phase") in C.LIVE_PHASES

        if live(status, me) and self.which == "match":
            return status, me
        began = time.monotonic()
        st2, me2 = self.other.call("GET", C.EP_ME)
        if live(st2, me2) and (self.which == "test" or not live(status, me)):
            self.client, self.other = self.other, self.client
            self.which = "test" if self.which == "match" else "match"
            self.log.write(e="server_picked", server=self.which, base=self.client.base, phase=me2.get("phase"), took_s=round(time.monotonic() - began, 3))
            self.say("서버를 %s 로 정함 (%s — 그쪽에서 우리 팀 단계가 %s)" % (self.client.base, "시험 포트" if self.which == "test" else "본 경기 포트", me2.get("phase")))
            return st2, me2
        return status, me

    # ------------------------------------------------------------------ 임무 (걸어 두고 기다린다)
    def poll_timeout(self):
        return float(self.cfg["wait_s"]) + max(5.0, self.client.patience)

    def pause(self, name, status, out, spent, gap):
        """임무 없이 돌아왔다 → 한 번 묻는 데 적어도 gap 초는 쓰게 쉰다 (쉬지 않으면 요청이 쉴 새 없이 나간다)."""
        why = ("서버 답: " + (detail_text(out) or "아직 없음")) if status == 200 else "요청 실패: " + ("응답 없음" if status is None else "HTTP %s %s" % (status, detail_text(out)))
        if self._why.get(name) != why:
            self._why[name] = why
            self.say("%s 아직 없음 (%s) — 계속 묻는다" % (name, why))
        if spent < gap:
            self.stop_event.wait(gap - spent)

    def command(self, kind, text="", code=""):
        c = ServerCommand()
        c.header.stamp = self.stamp()
        c.run_id, c.kind, c.text, c.objective_code = self.run_id, kind, text, code
        c.command_id = "%d-%s" % (kind, hashlib.sha1((text or code).encode("utf-8")).hexdigest()[:10])  # 다시 받아도 같은 명령이면 같은 값
        self.pub_command.publish(c)

    def order_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                phase, camp, ready, have = self.phase, self.camp, self.ready, self.order_text
            if have is not None or not camp or phase not in (C.PHASE_READY, C.PHASE_RUNNING) or (phase == C.PHASE_READY and not ready):
                self.stop_event.wait(0.2)
                continue
            began = time.monotonic()
            status, out = self.http("GET", C.EP_ORDER, {"wait": int(self.cfg["wait_s"])}, timeout=self.poll_timeout())
            if status == 200 and isinstance(out, dict) and out.get("accepted") and isinstance(out.get("order"), dict):
                text = str(out["order"].get("tactical_command") or "")
                with self.lock:
                    self.order_text = text
                self.say("전술명령 받음 (%d자)" % len(text))
                self.command(ServerCommand.TACTICAL_ORDER, text=text)
                continue
            self.pause("전술명령", status, out, time.monotonic() - began, 1.0)

    def target_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                phase, camp, have = self.phase, self.camp, self.objective
            if have is not None or phase != C.PHASE_RUNNING:
                self.stop_event.wait(0.1)
                continue
            began = time.monotonic()
            status, out = self.http("GET", C.EP_TARGET, {"wait": int(self.cfg["wait_s"])}, timeout=self.poll_timeout())
            if status == 200 and isinstance(out, dict) and out.get("accepted") and isinstance(out.get("order"), dict):
                objective = R.clean_objective(out["order"], camp)
                if objective is not None:
                    with self.lock:
                        self.objective = objective
                    self.command(ServerCommand.BASE_OBJECTIVE, code=objective)  # 먼저 내보내고 적는다 (4:00 진입은 빠를수록 좋다)
                    self.say("목표거점 받음: %s" % objective)
                    continue
                self.pause("목표거점", 200, {"reason": "모르는 목표거점 값 %r" % (out["order"].get("objective"),)}, 0.0, 5.0)
                continue
            self.pause("목표거점", status, out, time.monotonic() - began, 0.5)

    # ------------------------------------------------------------------ 드론 화면
    def drone_loop(self):
        last = None
        failed = (None, -1e9)  # 마지막으로 적은 실패 (상태 코드, 시각) — 같은 실패는 5초에 한 번만 적는다
        while not self.stop_event.is_set():
            # 경기 진행 중에만 받는다. 연습용 카메라 이름(drone_camera)은 시험 포트(팀 테스트)를 쓰고 있을 때만 붙인다 — 그때는 단계와 상관없이 받는다
            # (API 문서 4단계). 본 경기 포트에는 절대 붙이지 않는다: 설정에 이름이 남아 있어도 본 경기에서 연습 경기장 사진을 받는 일이 없게.
            camera = str(self.cfg["drone_camera"] or "") if self.which == "test" else ""
            if not self.cfg["drone"] or not (self.running() or camera):
                self.stop_event.wait(0.2)
                continue
            began = time.monotonic()
            status, jpeg, headers = self.client.drone(camera or None)
            took = time.monotonic() - began
            captured = next((v for k, v in headers.items() if str(k).lower() == "x-captured-at"), "") if status == 200 and jpeg else ""
            key = (captured or hash(jpeg)) if status == 200 and jpeg else None
            fresh = key is not None and key != last  # 같은 화면을 또 받았으면 내보내지 않는다
            saved = None
            if fresh:
                last = key
                self._drone_n += 1
                gap = float(self.cfg["drone_save_s"])
                if gap >= 0 and not self.light and began - self._drone_saved_at >= gap:
                    self._drone_saved_at = began
                    saved = self.log.save_drone(self._drone_n, jpeg)
                f = UavFrame()
                f.header.stamp = self.stamp()
                f.run_id, f.captured_at_utc, f.capture_time_known = self.run_id, str(captured), bool(captured)
                # image.header.stamp 는 0 으로 둔다 (촬영 시각은 captured_at_utc. 시계 변환을 검증하지 않았다).
                # frame_id 로 어느 카메라의 사진인지 알린다: 본 경기 포트면 "uav", 팀 테스트(시험 포트)면 "uav:test:<카메라 이름>".
                # perception 이 지도의 경기장 높이를 이것으로 고른다 (연습 경기장 4.80 m · 본선 경기장 5.00 m — 2026-11-19)
                # 서버가 답의 X-Camera 로 받은 카메라를 알려 준다 (main · test1 · test2 — API 문서 4단계) → 그것을 먼저 보고, 없으면 쓰고 있는 포트로 정한다
                seen = next((str(v).strip().lower() for k, v in headers.items() if str(k).lower() == "x-camera"), "")
                if seen:
                    f.image.header.frame_id = "uav" if seen == "main" else "uav:test:%s" % seen
                else:
                    f.image.header.frame_id = "uav:test:%s" % camera if self.which == "test" else "uav"
                f.image.format, f.image.data = "jpeg", jpeg
                self.pub_uav.publish(f)
            quiet = status != 200 and failed[0] == status and began - failed[1] < 5.0
            if status != 200 and not quiet:
                failed = (status, began)
            if fresh or (status != 200 and not quiet):
                self.log.write(e="drone", status=status, camera=camera or None, took_s=round(took, 3), bytes=len(jpeg) if jpeg else 0, captured=str(captured), n=self._drone_n, file=saved,
                               headers={str(k): str(v) for k, v in headers.items() if str(k).lower().startswith("x-")})
            self.stop_event.wait(max(0.05, float(self.cfg["drone_poll_s"]) - (time.monotonic() - began)))

    # ------------------------------------------------------------------ 서비스 (한 번 부르면 HTTP 한 번)
    @staticmethod
    def outcome(status, out, res):
        if status == 200 and isinstance(out, dict) and out.get("accepted"):
            res.outcome = 1  # ACCEPTED
        elif status is None or status >= 500:
            res.outcome = 3  # UNKNOWN — 서버가 받았는지 모른다
        else:
            res.outcome = 2  # REJECTED
        res.http_status = int(status or 0)
        res.detail = detail_text(out)
        return res

    def on_set_ready(self, req, res):
        if self.client is None:
            res.outcome, res.detail = 2, "서버 설정 없음"
            return res
        with self.lock:
            already = self.connected and self.ready and self.phase == C.PHASE_READY
        if already:  # 서버가 이미 준비완료로 알고 있다 (GET /me) → 다시 보내지 않는다
            res.outcome, res.http_status, res.detail = 1, 200, "이미 준비완료 (GET /me 로 확인)"
            return res
        status, out = self.http("POST", C.EP_READY)
        if status == 200 and isinstance(out, dict) and out.get("accepted"):
            with self.lock:
                self.ready = True
            self.say("POST /ready 접수")
        else:
            self.say("POST /ready → %s %s" % (status, detail_text(out)))
        return self.outcome(status, out, res)

    def on_submit_report(self, req, res):
        if self.client is None:
            res.outcome, res.detail = 2, "서버 설정 없음"
            return res
        r = req.report
        with self.lock:
            camp = self.camp
        if req.run_id and self.run_id and req.run_id != self.run_id:  # 막지는 않는다 (보고를 잃는 쪽이 더 나쁘다) — 기록에만 남긴다
            self.log.write(e="run_id_mismatch", request=req.run_id, known=self.run_id, kind=int(r.kind))
        if r.kind == Report.M1:
            endpoint, body = C.EP_M1, R.m1_body(r.command.target, list(r.command.search_sectors), r.command.base)
        elif r.kind == Report.M3:
            endpoint, body = C.EP_M3, R.m3_body([(s.sector, {o.type: o.count for o in s.objects}) for s in r.sectors])
        elif r.kind == Report.M6:
            endpoint, body = C.EP_M6, R.m6_body(list(r.assessment.ranking), camp)
        else:
            res.outcome, res.detail = 2, "모르는 보고 종류 %d" % r.kind
            return res
        if not body or (r.kind == Report.M3 and not body[C.M3_SECTORS]) or (r.kind == Report.M6 and len(body[C.M6_RANKING]) != 3):
            res.outcome, res.detail = 2, "보낼 내용이 없거나 형식에 맞지 않음 (서버에 보내지 않음)"
            self.log.write(e="report_not_sent", kind=int(r.kind), body=body)
            return res
        status, out = self.http("POST", endpoint, body=body)
        if not (status == 200 and isinstance(out, dict) and out.get("accepted")):
            self.say("보고 %s → %s %s" % (endpoint, status, detail_text(out)))
        return self.outcome(status, out, res)


def main():
    rclpy.init()
    node = ServerBridge()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
