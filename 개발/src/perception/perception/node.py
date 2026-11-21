"""perception 노드: /team/perception/classify_image 서비스 제공.

supervisor 가 사진 한 장을 보내면 그 안의 그림(상자 옆면)들을 왼쪽부터 답한다.
이 노드는 사진만 보고 답하며 로봇 위치 · 지도 · 팬틸트는 다루지 않는다.

실행:  ros2 run perception perception_node --ros-args -p model_path:=/home/ws/team/models/pictures.engine
"""
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from team_interfaces.msg import MissionState, NodeStatus, Picture, SensorSpot, UgvState
from team_interfaces.srv import ClassifyImage, LocateSensors, ReadUgvs

from perception import pictures, sensors, ugvs, zoom
from perception.detector import PictureDetector
from perception.map_maker import MapMaker
from perception.ugv_watch import UgvWatch


def image_to_rgb(image):
    """sensor_msgs/Image -> HxWx3 uint8 RGB. /image_raw 는 실측 640x480 rgb8."""
    encoding = image.encoding.lower()
    if encoding not in ('rgb8', 'bgr8'):
        raise ValueError('unsupported image encoding: %s' % image.encoding)
    rows = np.frombuffer(image.data, np.uint8).reshape(image.height, -1)
    pixels = rows[:, :image.width * 3].reshape(image.height, image.width, 3)
    return pixels if encoding == 'rgb8' else pixels[:, :, ::-1]


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        self.min_confidence = float(self.declare_parameter('min_confidence', 0.5).value)
        # 수광부(target_sensor) 탐지는 이 점수부터 답에 싣는다 (detail 끝의 sensors=[...]). 고르는 것은 supervisor 가 한다
        self.min_sensor_confidence = float(self.declare_parameter('min_sensor_confidence', 0.15).value)
        self.detector = PictureDetector(
            model_path=self.declare_parameter('model_path', '/home/ws/team/models/pictures.engine').value,
            imgsz=int(self.declare_parameter('imgsz', 640).value),
            detect_conf=float(self.declare_parameter('detect_conf', 0.15).value),
            device=str(self.declare_parameter('device', '0').value),
        )
        # 작은(먼) 그림은 둘레를 잘라 키워 한 번 더 판독한다 (perception/zoom.py). zoom_max_height 0 이면 끔.
        # 로봇에서 한 번에 약 35 ms 라 사진 한 장에 최대 zoom_limit 번 더 든다. classify_image 에만 쓴다.
        self.zoom_max_height = int(self.declare_parameter('zoom_max_height', 96).value)
        self.zoom_fill = float(self.declare_parameter('zoom_fill', 0.25).value)
        self.zoom_limit = int(self.declare_parameter('zoom_limit', 4).value)
        self.state = NodeStatus.INITIALIZING
        self.detail = 'loading model'
        self.run_id = ''
        self.infer_lock = threading.Lock()

        # Picture.msg 의 kind 글자와 계산 모듈의 글자가 어긋나면 supervisor 가 답을 못 읽는다.
        expected = {Picture.MILITARY_TRUCK, Picture.MORTAR, Picture.TANK, Picture.MISSILE, Picture.AMMO_BOX,
                    Picture.DRUM, Picture.COMMS, Picture.OUTPOST, Picture.RADAR, Picture.NON_OBJECT, Picture.UNKNOWN}
        if expected != set(pictures.ALL_KINDS):
            raise RuntimeError('Picture.msg kinds differ from perception.pictures')

        group = ReentrantCallbackGroup()
        self.status_publisher = self.create_publisher(
            NodeStatus, '/team/node_status',
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE))
        self.create_subscription(
            MissionState, '/team/mission/state', self._receive_mission,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL), callback_group=group)
        self.create_service(ClassifyImage, '/team/perception/classify_image', self._classify, callback_group=group)
        self.create_service(ReadUgvs, '/team/perception/read_ugvs', self._read_ugvs, callback_group=group)
        # 수광부의 사진 속 자리와 그것이 어디에 붙은 것인지 (supervisor 가 상대 UGV 를 따라가며 쏠 때 쓴다 — 선택 사항)
        self.create_service(LocateSensors, '/team/perception/locate_sensors', self._locate_sensors, callback_group=group)
        self.min_ugv_confidence = float(self.declare_parameter('min_ugv_confidence', 0.35).value)
        self.can_see_ugvs = False      # 모델에 UGV 클래스가 있을 때만 True (모델을 읽은 뒤 정한다)
        # UGV 판독(read_ugvs)용 모델. 안테나 · LED 막대까지 찾는 별도 가중치가 있으면 그것을 쓰고, 없으면 그림 모델을 같이 쓴다.
        # 부품 클래스를 그림 모델에 합치면 그림 인식(M3)이 나빠져서 따로 둔다 (2026-11-19 비교).
        self.ugv_detector = PictureDetector(
            model_path=self.declare_parameter('ugv_model_path', '/home/ws/team/models/ugv_parts.engine').value,
            imgsz=self.detector.imgsz, detect_conf=self.detector.detect_conf, device=self.detector.device)
        self.ugv_reader = None         # read_ugvs 가 실제로 쓰는 모델 (모델을 읽은 뒤 정한다)
        # locate_sensors 를 부품 모델 하나로만 답한다 (그 모델에 ugv_sensor 클래스가 있을 때). false 면 전처럼 그림 모델 + 부품 모델
        # 기본값을 False 로 둔다 (2026-11-20 06:00 통합 쪽 — 팀장 · 사격 담당이 정하기 전까지): locate_sensors 는 지정표적 조준(M2)에도 쓰인다.
        # 그림 모델 + 부품 모델로 답하던 전의 흐름으로 지정표적은 17판을 치렀고(조준 사진으로 확인한 명중), 부품 모델 하나로 지정표적의 수광부를
        # 찾는 것은 실차로 한 판도 돌지 않았다. 켜려면 programs.json 의 perception 줄에 "args": ["--ros-args", "-p", "locate_with_ugv_model_only:=true"]
        self.locate_with_ugv_model_only = bool(self.declare_parameter('locate_with_ugv_model_only', False).value)
        self.map_maker = MapMaker(self, group)   # 드론 사진 → 지도(/team/perception/field_map). 연결은 map_maker.py
        # 경기 중 드론 사진 → UGV 자리(/team/perception/ugvs). 지도가 나온 뒤부터 돈다. 연결은 ugv_watch.py
        self.ugv_watch = UgvWatch(self, group, self.map_maker, device=self.detector.device)
        self.create_timer(1.0, self._publish_status, callback_group=group)
        self._publish_status()
        threading.Thread(target=self._load_model, daemon=True).start()

    # ---- 상태 ----
    def _receive_mission(self, message):
        self.run_id = message.run_id
        self.map_maker.on_mission(message)   # 경기가 시작되면 드론 사진으로 지도를 만든다

    def _set_state(self, state, detail):
        self.state, self.detail = state, detail
        self.get_logger().info('state=%d %s' % (state, detail))
        self._publish_status()

    def _publish_status(self):
        status = NodeStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.run_id = self.run_id
        status.node_name = 'perception'
        status.state = self.state
        status.detail = '%s | %s | %s' % (self.detail, self.map_maker.detail, self.ugv_watch.detail)
        self.status_publisher.publish(status)

    def _load_model(self):
        try:
            self.detector.load()
        except Exception as error:    # 모델 파일 없음 · torch/ultralytics 없음 · 엔진 불일치 등
            self._set_state(NodeStatus.ERROR, '%s: %s' % (type(error).__name__, error))
            return
        ugv_note = ''
        try:
            self.ugv_detector.load()
            self.ugv_reader = self.ugv_detector
            ugv_note = ' ugv_model=%s' % self.ugv_detector.model_path
        except FileNotFoundError:
            self.ugv_reader = self.detector          # 별도 가중치가 없으면 그림 모델로 UGV 위치만
        except Exception as error:
            self.ugv_reader = self.detector
            ugv_note = ' ugv_model_failed(%s: %s)' % (type(error).__name__, str(error)[:80])
        self.can_see_ugvs = any(ugvs.is_ugv(n) for n in self.ugv_reader.names.values())
        unknown = sorted(n for n in self.detector.names.values()
                         if pictures.kind_of(n) is None and pictures.is_picture(n))
        missing = sorted(k for k in pictures.MODEL_KINDS
                         if k not in {pictures.kind_of(n) for n in self.detector.names.values()})
        detail = 'model=%s%s' % (self.detector.model_path, ugv_note)
        if missing:
            detail += ' missing_kinds=%s' % missing
        if unknown:
            detail += ' unmapped_classes=%s' % unknown
        if not self.can_see_ugvs:
            detail += ' no_ugv_class(read_ugvs 는 ok=false 로 답함)'
        # 10가지 중 못 내는 종류가 있으면 답이 틀릴 수 있으므로 DEGRADED 로 알린다.
        self._set_state(NodeStatus.DEGRADED if missing else NodeStatus.READY, detail)

    # ---- 서비스 ----
    def _classify(self, request, response):
        started = time.monotonic()
        response.ok = False
        response.pictures = []
        if self.detector.model is None:
            response.detail = 'model not ready: %s' % self.detail
            return response
        try:
            rgb = image_to_rgb(request.image)
            with self.infer_lock:
                detections = self.detector.detect(rgb)
                # 그림의 종류 · 점수만 다시 정한다. 수광부 · UGV 는 처음 탐지(detections) 그대로 쓴다.
                picture_detections = zoom.refine(rgb, detections, self.detector.detect, pictures.is_picture,
                                                 max_height=self.zoom_max_height, fill=self.zoom_fill,
                                                 limit=self.zoom_limit)
            found = pictures.build_pictures(picture_detections, request.image.width, request.image.height,
                                            min_confidence=self.min_confidence)
            # UGV 상자 안의 수광부는 UGV 에 달린 것이다 → 지정표적 조준에 쓰는 목록에서 뺀다 (분명한 UGV 만: 점수 0.5 이상)
            ugv_boxes = [d[2:6] for d in detections if ugvs.is_ugv(d[0]) and d[1] >= max(0.5, self.min_ugv_confidence)]
            sensors = pictures.build_sensors(detections, request.image.width, request.image.height, exclude_boxes=ugv_boxes,
                                             min_confidence=self.min_sensor_confidence)
        except Exception as error:
            response.detail = '%s: %s' % (type(error).__name__, error)
            self.get_logger().error('classify failed: %s' % response.detail)
            return response
        for item in found:
            picture = Picture()
            picture.kind = item['kind']
            picture.x_center = float(item['x_center'])
            picture.y_center = float(item['y_center'])
            picture.width = float(item['width'])
            picture.height = float(item['height'])
            picture.confidence = float(item['confidence'])
            picture.cut_off = bool(item['cut_off'])
            response.pictures.append(picture)
        response.ok = True
        # 수광부는 그림이 아니라 pictures 에 넣지 않는다 (supervisor 의 면 짝짓기가 밀린다). 지정표적 조준에 쓰도록 detail 끝에 싣는다.
        response.detail = '%d pictures in %d ms | %s' % (len(found), int((time.monotonic() - started) * 1000),
                                                         pictures.sensors_text(sensors))
        return response

    def _read_ugvs(self, request, response):
        started = time.monotonic()
        response.ok = False
        response.ugvs = []
        if self.ugv_reader is None or self.ugv_reader.model is None:
            response.detail = 'model not ready: %s' % self.detail
            return response
        if not self.can_see_ugvs:
            # 빈 목록 + ok=true 로 답하면 "UGV 가 없다"는 뜻이 되므로, 볼 수 없을 때는 실패로 알린다.
            response.detail = 'model has no ugv class'
            return response
        try:
            rgb = image_to_rgb(request.image)
            with self.infer_lock:
                detections = self.ugv_reader.detect(rgb)
            found = ugvs.build_ugvs(detections, request.image.width, request.image.height, rgb=rgb,
                                    min_confidence=self.min_ugv_confidence)
        except Exception as error:
            response.detail = '%s: %s' % (type(error).__name__, error)
            self.get_logger().error('read_ugvs failed: %s' % response.detail)
            return response
        for item in found:
            state = UgvState()
            state.camp = item['camp']
            state.callsign = item['callsign']
            state.lives = int(item['lives'])
            state.ammo = int(item['ammo'])
            state.x_center = float(item['x_center'])
            state.y_center = float(item['y_center'])
            state.width = float(item['width'])
            state.height = float(item['height'])
            state.confidence = float(item['confidence'])
            response.ugvs.append(state)
        response.ok = True
        response.detail = '%d ugvs in %d ms (ammo is never read)' % (len(found), int((time.monotonic() - started) * 1000))
        return response


    def _locate_sensors(self, request, response):
        started = time.monotonic()
        response.ok = False
        response.sensors = []
        if self.detector.model is None:
            response.detail = 'model not ready: %s' % self.detail
            return response
        try:
            rgb = image_to_rgb(request.image)
            # 부품 모델이 따로 있고 UGV 의 수광센서까지 찾으면 그것 하나만 돌린다 (2026-11-20): 정찰이 끝난 뒤 상대 UGV 를 살피고 쏘는 동안에는
            # 그림 모델(M3)을 부르지 않는다. 부품 모델도 그림 · target_sensor 를 함께 찾으므로 owner(target / ugv) 판정은 그대로다.
            parts_only = (self.locate_with_ugv_model_only and self.ugv_reader is not None and self.ugv_reader is not self.detector
                          and any(sensors.is_ugv_sensor(n) for n in self.ugv_reader.names.values()))
            with self.infer_lock:
                if parts_only:
                    detections, parts = self.ugv_reader.detect(rgb), None
                else:
                    detections = self.detector.detect(rgb)
                    # 부품 모델이 따로 있으면 그것도 돌린다 (ugv_sensor · 안테나 색 → UGV 의 센서와 진영)
                    parts = self.ugv_reader.detect(rgb) if self.ugv_reader is not None and self.ugv_reader is not self.detector else None
            found = sensors.build_sensors(detections, request.image.width, request.image.height, rgb=rgb, part_detections=parts,
                                          min_confidence=max(0.25, self.min_sensor_confidence))
        except Exception as error:
            response.detail = '%s: %s' % (type(error).__name__, error)
            self.get_logger().error('locate_sensors failed: %s' % response.detail)
            return response
        for item in found:
            spot = SensorSpot()
            spot.owner = item['owner']
            spot.x_center, spot.y_center = float(item['x_center']), float(item['y_center'])
            spot.width, spot.height = float(item['width']), float(item['height'])
            spot.confidence = float(item['confidence'])
            spot.has_host = bool(item['has_host'])
            spot.host_kind, spot.camp, spot.callsign = item['host_kind'], item['camp'], item['callsign']
            spot.host_x_center, spot.host_y_center = float(item['host_x_center']), float(item['host_y_center'])
            spot.host_width, spot.host_height = float(item['host_width']), float(item['host_height'])
            response.sensors.append(spot)
        response.ok = True
        response.detail = '%d sensors in %d ms%s' % (len(found), int((time.monotonic() - started) * 1000), ' (ugv model only)' if parts_only else '')
        return response


def main():
    rclpy.init()
    node = PerceptionNode()
    executor = MultiThreadedExecutor(num_threads=3)   # 추론 중에도 상태 발행 · 드론 사진 받기가 멈추지 않게
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
