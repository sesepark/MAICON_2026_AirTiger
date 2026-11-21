"""드론 사진(/team/server/uav_image) → 경기장의 UGV 자리(/team/perception/ugvs). perception 노드에 붙는 연결 부분 (2026-11-19, navigation 담당이 붙임).

계산은 drone_ugvs.UgvView 가 한다. 여기서는 받아서 넘기고 내보내기만 한다.

  - 지도(map_maker)가 나온 뒤부터 돈다: 그 지도를 만든 사진의 경계 · 카메라 높이 · 지도 크기를 그대로 쓴다 (map_maker.view_info).
    드론 카메라는 천장에 붙어 있어 경기 동안 같다. 벽만 있는 임시 지도뿐이면(경계를 못 찾음) 돌지 않는다
  - 드론 사진이 올 때마다(약 1 Hz) 가장 새 것 한 장만 본다. 계산은 따로 도는 스레드가 한다 (사진 판독 서비스를 막지 않게).
    모델은 그 스레드가 처음에 한 번 읽는다 — 못 읽으면(파일 · ultralytics 없음) 그 뒤로 끈다 (detail 에 남는다)
  - 내보내는 것: geometry_msgs/PoseArray, frame_id "arena", header.stamp = 그 드론 사진의 header.stamp (server_bridge 가 받은 시각).
    UGV 마다 position.x · y = 가운데 (m), position.z = 점수 (0 ~ 1). 방향은 모른다 (orientation.w = 1).
    **우리 차도 들어 있다** (모델은 진영 · 호출부호를 가리지 못한다) — 받는 쪽(navigation)이 자기 위치로 골라낸다.
    한 대도 없으면 빈 목록을 낸다 (「봤는데 없다」 와 「안 왔다」 를 받는 쪽이 가릴 수 있게)
  - 화면 위아래가 잘리는 카메라에서는 벽 쪽의 UGV 가 안 나온다. 보이는 세로 범위는 기록(ugvs.jsonl 의 첫 줄 seen_y)에 남긴다

기록: TEAM_LOG_DIR 이 있으면 그 아래 perception/ugvs.jsonl 에 사진마다 한 줄 (자리 · 걸린 시간).
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

from perception import drone_ugvs
from perception.drone_boundary import Boundary


class UgvWatch:
    def __init__(self, node, group, map_maker, device=None):
        from geometry_msgs.msg import PoseArray
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from team_interfaces.msg import UavFrame

        self.node, self.map_maker = node, map_maker
        declare = node.declare_parameter
        self.enabled = bool(declare("ugv_watch", True).value)
        self.model_path = os.path.expanduser(str(declare("ugv_watch_model", drone_ugvs.DEFAULT_MODEL).value))
        self.conf = float(declare("ugv_watch_conf", 0.5).value)  # 학습 기록의 검증 기준 (찾음 98.4 % · 헛것 28 / 208장)
        self.ugv_height = float(declare("ugv_watch_height", drone_ugvs.UGV_HEIGHT).value)
        self.min_gap_s = float(declare("ugv_watch_min_s", 0.5).value)  # 사진이 이보다 자주 와도 이 간격으로만 본다
        # 한 장을 보는 데 이보다 오래 걸리는 일이 세 번 이어지면 스스로 끈다 (0 이하 = 끄지 않는다). 로봇에서 아직 돌려 보지 않았다 —
        # 판독이 밀리면 같은 프로그램의 그림 판독(M3)이 느려질 수 있으므로, 그럴 바에는 이 기능을 버린다 (navigation 은 이 토픽이 없으면 전처럼 돈다)
        self.max_took_s = float(declare("ugv_watch_max_s", 1.5).value)
        own = str(declare("ugv_watch_device", "").value)  # 빈 문자열 = 그림 모델과 같은 장치
        self.device = own or device
        self.detail = "UGV 찾기: 지도를 기다림" if self.enabled else "UGV 찾기: 끔"
        self.frame = None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.view, self.view_key = None, None
        self.sent = 0
        shared = os.environ.get("TEAM_LOG_DIR")
        self.log_dir = os.path.join(shared, "perception") if shared and os.environ.get("TEAM_LOG_OFF") != "1" else None
        if not self.enabled:
            return
        self.PoseArray = PoseArray
        out = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        frames = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.publisher = node.create_publisher(PoseArray, "/team/perception/ugvs", out)
        node.create_subscription(UavFrame, "/team/server/uav_image", self.on_frame, frames, callback_group=group)
        threading.Thread(target=self.work, name="ugv-watch", daemon=True).start()

    def on_frame(self, message):
        with self.lock:
            self.frame = message
        self.wake.set()

    def view_for(self):
        """지도를 만든 경계로 UgvView 를 만든다 (지도가 바뀌면 다시). 아직 없으면 None."""
        info = getattr(self.map_maker, "view_info", None)
        if not info or not info.get("boundary") or not info.get("camera_height"):
            return None
        key = id(info)
        if key != self.view_key:
            boundary = Boundary.from_dict(info["boundary"])
            if boundary is None:
                return None
            self.view = drone_ugvs.UgvView(boundary, float(info["camera_height"]), info.get("arena"), self.ugv_height, conf=self.conf)
            self.view.mv.maps()  # 표를 미리 만든다
            self.view_key = key
            lens = info["boundary"].get("lens") or {}
            # 렌즈 · 끝 띠: 지도를 만든 경계가 어떤 것이었나 (픽셀 비 1.0 · 중심 = 화면 가운데면 단순 렌즈 — 본선 카메라에서는 그 경계가 오른쪽을 0.2 ~ 0.3 m 틀리게 옮겼다)
            self.write_log(e="view", camera_height=round(float(info["camera_height"]), 3), arena=info.get("arena"),
                           ugv_height=self.ugv_height, seen_y=self.view.seen_y(), conf=self.conf,
                           lens={k: round(float(lens[k]), 3) for k in ("k1", "k2", "aspect", "cx", "cy") if k in lens},
                           strip=info["boundary"].get("strip"), rms_m=info["boundary"].get("rms_m"))
        return self.view

    def work(self):
        import cv2

        model, last, done, slow = None, -1e9, None, 0
        while True:
            self.wake.wait(2.0)
            self.wake.clear()
            with self.lock:
                frame = self.frame
            if frame is None or frame is done or time.monotonic() - last < self.min_gap_s:
                continue
            try:
                view = self.view_for()
                if view is None:
                    continue
                if model is None:
                    self.detail = "UGV 찾기: 모델을 읽는 중"
                    try:
                        model = drone_ugvs.load_model(self.model_path, device=self.device)
                    except Exception as error:  # noqa: BLE001 — 모델을 못 쓰면 끈다 (다른 일은 그대로)
                        self.detail = "UGV 찾기: 끔 (모델을 못 읽음 — %s: %s)" % (type(error).__name__, str(error)[:80])
                        self.node.get_logger().error(self.detail)
                        self.write_log(e="off", error="%s: %s" % (type(error).__name__, error))
                        return
                    self.node.get_logger().info("UGV 찾기 모델: %s" % model.path)
                    self.write_log(e="model", path=model.path)
                done, last = frame, time.monotonic()
                image = cv2.imdecode(np.frombuffer(bytes(frame.image.data), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                ugvs = view.detect(image, model.predict)
                self.publish(frame, ugvs)
                took = time.monotonic() - last
                self.sent += 1
                slow = slow + 1 if self.max_took_s > 0.0 and took > self.max_took_s and self.sent > 1 else 0
                if slow >= 3:
                    self.detail = "UGV 찾기: 끔 (한 장에 %.1f s — 너무 느림)" % took
                    self.node.get_logger().error(self.detail)
                    self.write_log(e="off", error="slow", took_ms=round(took * 1000, 1))
                    return
                self.detail = "UGV 찾기: %d대 (%d번째 사진, %.0f ms)" % (len(ugvs), self.sent, took * 1000)
                stamp = frame.header.stamp
                self.write_log(e="ugvs", stamp=round(stamp.sec + stamp.nanosec * 1e-9, 3), captured=str(frame.captured_at_utc),
                               ugvs=[[round(x, 3), round(y, 3), round(s, 2)] for x, y, s in ugvs], took_ms=round(took * 1000, 1))
            except Exception as error:  # noqa: BLE001 — 한 장에서 난 오류로 멈추지 않는다
                self.detail = "UGV 찾기: 오류 %s" % type(error).__name__
                self.node.get_logger().warning("UGV 찾기 오류: %s: %s" % (type(error).__name__, error), throttle_duration_sec=5.0)

    def publish(self, frame, ugvs):
        from geometry_msgs.msg import Pose

        message = self.PoseArray()
        message.header.stamp = frame.header.stamp
        message.header.frame_id = "arena"
        for x, y, score in ugvs:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(score)
            pose.orientation.w = 1.0
            message.poses.append(pose)
        self.publisher.publish(message)

    def write_log(self, **record):
        if self.log_dir is None:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "ugvs.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(record, wall=round(time.time(), 3)), ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass
