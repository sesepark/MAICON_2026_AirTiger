#!/usr/bin/env python3
"""supervisor 를 혼자 시험할 때 쓰는 가짜 상대: navigation · perception · server_bridge · llm_worker 와 로봇(/mil/*)을 한 프로그램이 흉내 낸다.

가상 경기장(개발/src/supervisor/test/fixtures/layout13.json) 안에서 가짜 로봇이 좌표를 받으면 가고, 사진 판독은 정답대로 답한다.
실제 프로그램이 아니다 — supervisor 의 메시지 주고받기가 맞는지만 본다.

  python3 fake_world.py --slot B-1 --standby 3 --running 300      (ROS 2 환경, PYTHONPATH 에 supervisor/test 가 있어야 한다)
끝나면 받은 보고와 정답을 견준 결과를 찍는다.
"""
import argparse
import json
import math
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from mil_msgs.msg import MatchCommand, MissionState
from mil_msgs.srv import Fire, Gimbal
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from sim_world import FakePerception, SimRobot, load_layout, scout_grid
from supervisor import arena as A
from supervisor.faces import CameraParams
from team_interfaces.action import Analyze, Navigate
from team_interfaces.msg import FieldMap, FieldRegion, NodeStatus, ParsedCommand, Picture, ServerCommand, ServerState
from team_interfaces.srv import ClassifyImage, ReadUgvs, SetReady, SubmitReport


def latched(depth=1):
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class FakeWorld(Node):
    def __init__(self, args):
        super().__init__("fake_world")
        self.args = args
        self.fmap, self.truth = load_layout()
        self.camp = A.camp_of(args.slot)
        self.eyes = FakePerception(self.fmap, self.truth, CameraParams(hfov_deg=128.0), seed=1)
        grid = scout_grid(self.fmap)
        for name in list(grid.zones):
            grid.set_zone(name, False)
        x0, y0, x1, y1 = self.fmap.bases[self.fmap.base_in_row(self.camp, A.row_of(args.slot))]
        self.robot = SimRobot(grid, ((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0 if self.camp == "BLUE" else math.pi), seed=1)
        foe = A.opponent(self.camp)
        kind = next(k for key, k in sorted(self.truth.items()) if k in A.TARGET_KINDS and A.side_of(self.fmap, next(f for f in self.fmap.faces if f.key == key).cx) == foe)
        self.order = {"target": "%s_%s" % (foe, kind.upper()), "search_sectors": ["CM"], "base": self.fmap.base_in_row(foe, 2)}
        self.objective = self.fmap.base_in_row(foe, A.row_of(args.slot))
        self.lock = threading.Lock()
        self.pan = 0.0
        self.t0 = time.monotonic()
        self.phase = "standby"
        self.reports, self.fires, self.goals = [], [], 0
        self.views = {}  # 사진의 시각 → (그때 로봇 위치, 그때 팬) — 판독은 사진을 찍은 순간의 것으로 답해야 한다
        self.ready = False
        self.run_id = ""
        group = ReentrantCallbackGroup()
        # 로봇
        self.pub_match = self.create_publisher(MatchCommand, "/mil/match", latched())
        self.pub_state = self.create_publisher(MissionState, "/mil/state", latched())
        self.pub_image = self.create_publisher(Image, "/image_raw", qos_profile_sensor_data)
        self.create_service(Gimbal, "/mil/gimbal", self.on_gimbal, callback_group=group)
        self.create_service(Fire, "/mil/fire", self.on_fire, callback_group=group)
        # navigation
        self.pub_pose = self.create_publisher(PoseStamped, "/team/navigation/pose", 10)
        self.current = None
        if not args.no_drive:
            ActionServer(self, Navigate, "/team/navigation/navigate", self.navigate, goal_callback=lambda g: GoalResponse.ACCEPT,
                         cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=group)
        # perception
        self.pub_map = self.create_publisher(FieldMap, "/team/navigation/map", latched())
        if not args.no_perception:
            self.create_service(ClassifyImage, "/team/perception/classify_image", self.on_classify, callback_group=group)
            self.create_service(ReadUgvs, "/team/perception/read_ugvs", self.on_ugvs, callback_group=group)
        self.drone_jpeg = b"\xff\xd8\xff\xd9"
        if args.drone_jpeg:
            with open(args.drone_jpeg, "rb") as fh:
                self.drone_jpeg = fh.read()
        # server_bridge · llm_worker — --http 를 주면 흉내 내지 않고, 대신 경기 서버(HTTP)를 흉내 낸다 (진짜 server_bridge · llm_worker 를 붙여 볼 때)
        self.http = args.http > 0
        if not self.http:
            self.pub_server = self.create_publisher(ServerState, "/team/server/state", latched())
            self.pub_command = self.create_publisher(ServerCommand, "/team/server/command", latched(2))
            self.create_service(SetReady, "/team/server/set_ready", self.on_ready, callback_group=group)
            self.create_service(SubmitReport, "/team/server/submit_report", self.on_report, callback_group=group)
            ActionServer(self, Analyze, "/team/llm/analyze", self.analyze, callback_group=group)
        else:
            if args.test_port:  # 본 경기 포트에는 경기가 없고(idle), 시험 포트에서 경기가 열린다
                serve_http(self, args.http, live=False)
                serve_http(self, args.test_port)
            else:
                serve_http(self, args.http)
        self.pub_status = self.create_publisher(NodeStatus, "/team/node_status", 10)
        self.create_subscription(NodeStatus, "/team/node_status", self.on_status, 10)
        self.create_timer(0.05, self.tick, callback_group=group)
        self.create_timer(1.0, self.slow, callback_group=group)
        self._sent = set()
        self.match("standby")

    # ---- 시간표
    def elapsed(self):
        return time.monotonic() - self.t0 - self.args.standby

    def match(self, cmd):
        m = MatchCommand()
        m.header.stamp = self.get_clock().now().to_msg()
        # --practice: 참가팀 포털의 팀 테스트처럼 로봇은 연습 모드로 남고 경기 신호를 주지 않는다 (단계는 서버 흉내만 알려 준다)
        m.mode, m.cmd = ("practice", "idle") if self.args.practice else ("mission", cmd)
        self.pub_match.publish(m)
        self.phase = cmd
        self.get_logger().info("경기 신호 → %s%s" % (cmd, " (로봇에는 주지 않음 — 연습 모드)" if self.args.practice else ""))

    def tick(self):
        t = self.elapsed()
        if self.phase == "standby" and t >= 0:
            self.match("start")
        elif self.phase == "start" and t >= self.args.running:
            self.match("report")
        elif self.phase == "report" and t >= self.args.running + self.args.report:
            self.match("stop")
        with self.lock:
            if self.phase == "start":
                self.robot.tick(0.05)
            x, y, yaw = self.robot.pose
        now = self.get_clock().now().to_msg()
        p = PoseStamped()
        p.header.stamp, p.header.frame_id = now, "arena"
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        if not self.args.no_slam:
            self.pub_pose.publish(p)
        if int(time.monotonic() * 20) % 2 == 0:  # 사진 10 Hz (내용은 비어 있다 — 가짜 perception 은 로봇의 참 위치로 답한다)
            img = Image()
            img.header.stamp, img.header.frame_id = now, "camera"
            img.height, img.width, img.encoding, img.step = 4, 4, "rgb8", 12
            img.data = bytes(48)
            self.views[(now.sec, now.nanosec)] = ((x, y, yaw), self.pan)
            for key in list(self.views)[:-200]:
                del self.views[key]
            self.pub_image.publish(img)
        if self.phase == "start" and t >= 0.5 and "boxes" not in self._sent and not self.args.no_slam:
            self._sent.add("boxes")
            fm = FieldMap()
            fm.header.stamp, fm.header.frame_id = now, "arena"
            fm.width, fm.height, fm.box_size, fm.beam_depth = self.fmap.width, self.fmap.height, self.fmap.box_size, self.fmap.beam_depth
            fm.boxes = [Point(x=bx, y=by, z=0.0) for bx, by in self.fmap.boxes]
            fm.bases = [FieldRegion(name=n, min_x=r[0], min_y=r[1], max_x=r[2], max_y=r[3], row=self.fmap.base_rows[n]) for n, r in self.fmap.bases.items()]
            fm.sectors = [FieldRegion(name=n, min_x=r[0], min_y=r[1], max_x=r[2], max_y=r[3], row=0) for n, r in self.fmap.sectors.items()]
            self.pub_map.publish(fm)
        if not self.http and self.phase == "start" and "order" not in self._sent:
            self._sent.add("order")
            self.command(ServerCommand.TACTICAL_ORDER, text=self.order_text())
        if not self.http and self.phase == "start" and t >= self.args.target_at and "objective" not in self._sent:
            self._sent.add("objective")
            self.command(ServerCommand.BASE_OBJECTIVE, code=self.objective)

    def order_text(self):
        """정답(self.order)을 담은 명령문 — 참가팀 포털의 팀 테스트 기본 명령문과 같은 문체 (시설을 기능으로 돌려 말한다)."""
        target = {"COMMS": "적 부대 간 교신을 중계하는 시설", "OUTPOST": "적 병력이 상주하며 전방을 눈으로 경계하는 시설",
                  "RADAR": "적 감시체계의 핵심 시설"}[self.order["target"].split("_", 1)[1]]
        base = {"COMMAND_POST": "작전을 지휘·통제하는 거점", "MAINTENANCE": "장비를 수리·정비할 수 있는 거점",
                "SUPPLY_DEPOT": "군수지원이 가능한 거점"}[self.order["base"].split("_", 1)[1]]
        return "중앙 구역을 정찰하고 전장 객체를 확인하라. 탐색 완료 후, %s을 타격하고, 최종적으로 %s으로 이동하라." % (target, base)

    def command(self, kind, text="", code=""):
        c = ServerCommand()
        c.header.stamp = self.get_clock().now().to_msg()
        c.run_id, c.command_id, c.kind, c.text, c.objective_code = self.run_id, "c%d" % kind, kind, text, code
        self.pub_command.publish(c)

    def slow(self):
        now = self.get_clock().now().to_msg()
        if not self.http:
            s = ServerState()
            s.header.stamp = now
            s.run_id, s.connected, s.team_assigned, s.slot, s.camp = self.run_id, True, True, self.args.slot, self.camp
            s.phase = {"standby": "waiting", "start": "running", "report": "reporting", "stop": "finished"}[self.phase]
            s.ready = self.ready
            self.pub_server.publish(s)
        st = MissionState()
        st.hp, st.hp_max, st.battery_pct = 4, 4, 90.0
        self.pub_state.publish(st)
        # navigation 은 실제 주행 프로그램처럼 "/" 없는 이름으로, 지도를 내보내기 전(시작 전)에는 INITIALIZING 으로 알린다
        for name in ("navigation", "/perception") + (() if self.http else ("/server_bridge", "/llm_worker")):
            if name.strip("/") in self.args.quiet_nodes.split(",") or (name == "/perception" and self.args.no_perception):
                continue
            n = NodeStatus()
            n.header.stamp, n.run_id, n.node_name, n.state = now, self.run_id, name, NodeStatus.READY
            if name == "navigation" and self.phase == "standby":
                n.state, n.detail = NodeStatus.INITIALIZING, "지도 없음"
            self.pub_status.publish(n)

    def on_status(self, m):
        if m.node_name == "/supervisor":
            self.run_id = m.run_id

    # ---- 로봇
    def on_gimbal(self, req, res):
        self.pan = req.pan * 0.5
        res.message, res.left_ms = ("ok", 0) if self.phase == "start" or self.args.practice else ("locked", 0)
        return res

    def on_fire(self, req, res):
        if self.phase != "start" and not self.args.practice:
            res.message, res.left_ms = "locked", 0
            return res
        if self.phase != "start":
            self.early_fires = getattr(self, "early_fires", 0) + 1  # 연습 모드에서는 잠기지 않는다 → 시작 전 · 보고 중에 쏘면 팀 코드의 잘못
        self.fires.append((round(self.elapsed(), 1), self.robot.pose, self.pan))
        res.message, res.left_ms = "fired", 10000
        return res

    # ---- navigation: 좌표를 받으면 간다. 새 목표가 오면 이전 것은 CANCELED 로 끝낸다
    def navigate(self, handle):
        goal = handle.request.goal.pose
        yaw = math.atan2(2.0 * goal.orientation.w * goal.orientation.z, 1.0 - 2.0 * goal.orientation.z ** 2)
        with self.lock:
            self.goals += 1
            self.current = handle
            self.robot.go(goal.position.x, goal.position.y, yaw)
        result = Navigate.Result()
        while rclpy.ok():
            time.sleep(0.05)
            with self.lock:
                moving, mine = self.robot.moving, self.current is handle
            if not mine:  # 새 목표가 왔다 → 이 목표는 끝낸다 (취소 요청이 없었으므로 canceled 가 아니라 abort)
                handle.abort()
                result.outcome = Navigate.Result.CANCELED
                return result
            if handle.is_cancel_requested:  # 멈춤
                with self.lock:
                    self.robot.goal, self.robot.moving, self.robot.route = None, False, []
                handle.canceled()
                result.outcome = Navigate.Result.CANCELED
                return result
            if not moving:
                handle.succeed()
                result.outcome = Navigate.Result.REACHED
                return result
        return result

    # ---- perception
    def on_classify(self, req, res):
        time.sleep(self.args.latency)
        stamp = req.image.header.stamp
        with self.lock:
            pose, pan = self.views.get((stamp.sec, stamp.nanosec), (self.robot.pose, self.pan))
            pics, _keys = self.eyes.look(pose, pan)
        res.ok = True
        for kind, x, cut in pics:
            res.pictures.append(Picture(kind=kind, x_center=float(min(1.0, max(0.0, x))), y_center=0.5, width=0.1, height=0.2, confidence=0.9, cut_off=bool(cut)))
        return res

    def on_ugvs(self, req, res):
        res.ok = True
        return res

    # ---- server_bridge · llm_worker
    def on_ready(self, req, res):
        self.ready = True
        res.outcome, res.http_status = SetReady.Response.ACCEPTED, 200
        return res

    def on_report(self, req, res):
        r = req.report
        if r.kind == Report_M1:
            body = {"target": r.command.target, "search_sectors": list(r.command.search_sectors), "base": r.command.base}
        elif r.kind == Report_M3:
            body = {s.sector: {o.type: o.count for o in s.objects} for s in r.sectors}
        else:
            body = list(r.assessment.ranking)
        ok = (r.kind in (Report_M1, Report_M3) and self.phase == "start") or (r.kind == Report_M6 and self.phase == "report")
        self.reports.append((round(self.elapsed(), 1), r.kind, body, ok))
        res.outcome, res.http_status = (SubmitReport.Response.ACCEPTED, 200) if ok else (SubmitReport.Response.REJECTED, 409)
        return res

    def analyze(self, handle):
        time.sleep(1.5)
        result = Analyze.Result()
        result.success = True
        result.command = ParsedCommand(target=self.order["target"], search_sectors=self.order["search_sectors"], base=self.order["base"])
        handle.succeed()
        return result

    # ---- 끝난 뒤
    def summary(self):
        want = {s: {} for s in A.scored_sectors(self.camp)}
        for f in self.fmap.faces:
            kind = self.truth[f.key]
            if f.sector in want and kind in A.OBJECT_KINDS:
                want[f.sector][kind] = want[f.sector].get(kind, 0) + 1
        last = {}
        for t, kind, body, ok in self.reports:
            if ok:
                last[kind] = (t, body)
        x0, y0, x1, y1 = self.fmap.bases[self.objective]
        x, y, _yaw = self.robot.pose
        out = {
            "준비완료": self.ready,
            "M1 맞음": last.get(1, (None, None))[1] == self.order,
            "M3 맞은 구역": sum(1 for s in want if (last.get(3, (None, {}))[1] or {}).get(s) == want[s]),
            "M3 보고 횟수": sum(1 for r in self.reports if r[1] == 3),
            "M3 마지막 보고 시각": last.get(3, (None, None))[0],
            "지정표적": self.order["target"], "발사 횟수": len(self.fires),
            "목표거점": self.objective, "거점 안에서 끝남": bool(x0 - 0.05 <= x <= x1 + 0.05 and y0 <= y <= y1),
            "M6 보고": last.get(6, (None, None))[1],
            "목표거점을 서버가 준 시각": getattr(self, "objective_sent", None),
            "시작 전·보고 중에 쏜 횟수": getattr(self, "early_fires", 0), "경기가 없는 포트에 보낸 보고·요청": getattr(self, "idle_port_hits", 0),
            "받은 이동 목표 수": self.goals, "간 거리 m": round(self.robot.path_m, 1), "돈 각도": round(math.degrees(self.robot.turn_rad)),
        }
        return out


Report_M1, Report_M3, Report_M6 = 1, 3, 6


def serve_http(world, port, live=True):
    """경기 서버를 흉내 낸다 (경기서버연동API 2026-11-04 의 창구 일부). 단계는 가짜 세계의 경기 신호를 따른다.

    live=False 면 우리 팀의 경기가 없는 서버를 흉내 낸다 (GET /me 는 idle, 나머지는 409).
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, headers=None, raw=None):
            data = raw if raw is not None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "image/jpeg" if raw is not None else "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def phase(self):
            return {"standby": "ready", "start": "running", "report": "reporting", "stop": "finished"}[world.phase]

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            name = url.path.rstrip("/").split("/")[-1]
            if not live:
                world.idle_port_hits = getattr(world, "idle_port_hits", 0) + (name != "me")
                return self.reply(200, {"slot": None, "camp": None, "phase": "idle", "ready": False}) if name == "me" else self.reply(409, {"detail": "편성되지 않음"})
            wait = float(urllib.parse.parse_qs(url.query).get("wait", ["0"])[0])
            if name == "me":
                return self.reply(200, {"phase": self.phase(), "slot": world.args.slot, "camp": world.camp, "ready": world.ready})
            if name == "order":
                end = time.monotonic() + wait
                while time.monotonic() < end and world.phase == "standby":
                    time.sleep(0.05)
                if world.phase == "start":
                    return self.reply(200, {"accepted": True, "order": {"tactical_command": world.order_text(), "issued_at": "2026-11-18 00:00:00"}})
                return self.reply(200, {"accepted": False, "reason": "아직 게시 전"})
            if name == "target":
                end = time.monotonic() + wait
                while time.monotonic() < end and world.phase == "start" and world.elapsed() < world.args.target_at:
                    time.sleep(0.02)
                if world.phase == "start" and world.elapsed() >= world.args.target_at:
                    world.objective_sent = world.objective_sent or round(world.elapsed(), 2)
                    return self.reply(200, {"accepted": True, "order": {"objective": world.objective, "objective_label": ""}})
                return self.reply(200, {"accepted": False, "reason": "아직 게시 전"})
            if name == "drone":
                if world.phase != "start":
                    return self.reply(409, {"detail": "진행 중이 아님"})
                return self.reply(200, None, {"X-Captured-At": "2026-11-18T00:00:%06.3f+09:00" % (int(world.elapsed()) % 60)}, raw=world.drone_jpeg)
            return self.reply(404, {"detail": "없는 창구"})

        def do_POST(self):
            name = urllib.parse.urlparse(self.path).path.rstrip("/").split("/")[-1]
            if not live:
                world.idle_port_hits = getattr(world, "idle_port_hits", 0) + 1
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                return self.reply(409, {"detail": "편성되지 않음"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
            body.pop("mission_code", None)
            if name == "ready":
                world.ready = True
                return self.reply(200, {"accepted": True})
            kind = {"interpretation": Report_M1, "detection": Report_M3, "situation": Report_M6}.get(name)
            if kind is None:
                return self.reply(404, {"detail": "없는 창구"})
            ok = (kind in (Report_M1, Report_M3) and world.phase == "start") or (kind == Report_M6 and world.phase == "report")
            if kind == Report_M3:
                content = {s["sector"]: {o["type"]: o["count"] for o in s["objects"]} for s in body.get("sectors", [])}
            elif kind == Report_M6:
                content = body.get("ranking")
            else:
                content = body
            world.reports.append((round(world.elapsed(), 1), kind, content, ok))
            return self.reply(200, {"accepted": True, "received": body}) if ok else self.reply(409, {"detail": "지금 단계에서는 받지 않음 (%s)" % self.phase()})

    if live:
        world.objective_sent = None
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="B-1")
    ap.add_argument("--standby", type=float, default=4.0)
    ap.add_argument("--running", type=float, default=300.0)
    ap.add_argument("--report", type=float, default=6.0)
    ap.add_argument("--target-at", type=float, default=240.0)
    ap.add_argument("--latency", type=float, default=0.5)
    ap.add_argument("--drone-jpeg", default="", help="서버 흉내가 드론 사진으로 줄 JPEG 파일 (없으면 4바이트짜리 빈 사진)")
    ap.add_argument("--no-drive", action="store_true", help="이동 목표(/team/navigation/navigate)를 받지 않는다 (진짜 navigation_drive 를 붙여 볼 때)")
    ap.add_argument("--no-slam", action="store_true", help="위치 · 지도(/team/navigation/pose · map)를 내보내지 않는다 (진짜 navigation_slam 을 붙여 볼 때)")
    ap.add_argument("--no-perception", action="store_true", help="perception 을 흉내 내지 않는다 (진짜 perception 을 붙여 볼 때): 사진 판독 · UGV 판독 · 상태 알림을 만들지 않는다")
    ap.add_argument("--practice", action="store_true", help="로봇은 연습 모드로 남고 경기 신호를 주지 않는다 (팀 테스트 흉내). 단계는 서버 흉내만 알려 준다")
    ap.add_argument("--test-port", type=int, default=0, help="이 포트에서 경기가 열리고, --http 포트에는 경기가 없다 (시험 포트 흉내)")
    ap.add_argument("--quiet-nodes", default="", help="상태(/team/node_status)를 내보내지 않을 프로그램 이름 (쉼표로). 준비완료 시한을 볼 때")
    ap.add_argument("--http", type=int, default=0, help="이 포트에서 경기 서버(HTTP)를 흉내 낸다. 주면 server_bridge · llm_worker 는 흉내 내지 않는다")
    args = ap.parse_args()
    rclpy.init()
    node = FakeWorld(args)
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        while rclpy.ok() and node.phase != "stop":
            executor.spin_once(timeout_sec=0.05)
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    print("FAKE_WORLD_SUMMARY " + json.dumps(node.summary(), ensure_ascii=False))
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
