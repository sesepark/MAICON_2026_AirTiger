"""탐지 결과를 LocateSensors 응답의 수광센서 목록으로 바꾸는 순수 계산 (ROS · 모델 없이 시험 가능).

supervisor 가 레이저를 겨눌 자리를 알려 주는 것이 목적이다 (M2 지정표적, 선택 사항으로 상대 UGV).
- 센서마다 하나, 왼쪽부터.
- 센서가 어디에 붙은 것인지(owner)를 「무엇 안에 있는가」 로 정한다. 모델의 클래스 이름(target_sensor / ugv_sensor)만으로는 둘이 섞인다.
    UGV 박스 안에 있다                        → ugv   (모델이 target_sensor 라 했어도)
    그림 박스(조금 넓힌 것) 안 · 바로 곁에 있다  → target
    어디에도 안 붙어 있다                      → 모델이 target_sensor 라 했으면 target, 아니면 unknown
  지정표적인지(통신시설 · 감시초소 · 레이더시설)는 host_kind 로 알려 주고, 어느 것을 쏠지는 supervisor 가 정한다.
- 못 정한 값은 지어내지 않는다.
"""
from perception import pictures, ugvs

TARGET = 'target'
UGV = 'ugv'
UNKNOWN = 'unknown'

_TARGET_SENSOR_NAMES = ('target sensor', 'sensor', '수광센서')
_UGV_SENSOR_NAMES = ('ugv sensor',)


def _norm(name):
    return str(name).strip().lower().replace('-', ' ').replace('_', ' ')


def is_sensor(class_name):
    return _norm(class_name) in _TARGET_SENSOR_NAMES + _UGV_SENSOR_NAMES


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def is_ugv_sensor(class_name):
    """UGV 에 달린 수광센서 클래스인가 (이 클래스가 있는 모델이면 locate_sensors 를 그 모델 하나로 답할 수 있다)."""
    return _norm(class_name) in _UGV_SENSOR_NAMES


def _gap(point, box):
    """점이 박스 밖으로 벗어난 거리를 박스 크기에 대한 비율로 (안이면 0). 가로 · 세로 중 큰 쪽."""
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2]) / max(box[2] - box[0], 1e-6)
    dy = max(box[1] - y, 0.0, y - box[3]) / max(box[3] - box[1], 1e-6)
    return max(dx, dy)


def build_sensors(detections, image_width, image_height, rgb=None, part_detections=None,
                  min_confidence=0.25, min_host_confidence=0.35, same_sensor_iou=0.5,
                  ugv_margin=0.10, picture_margin=0.35):
    """탐지 목록 -> 센서 목록(dict), 왼쪽부터.

    detections: 그림 모델의 탐지 (class_name, confidence, x1, y1, x2, y2), 픽셀. 그림 · target_sensor · ugv 가 들어 있다.
    part_detections: UGV 부품 모델의 탐지 (ugv · antenna · led_bar · ugv_sensor). 없으면 detections 만 쓴다.
    ugv_margin: 센서 가운데가 UGV 박스에서 이 비율 안으로 벗어나 있어도 그 UGV 의 것으로 본다.
    picture_margin: 센서 가운데가 그림 박스에서 이 비율 안으로 벗어나 있어도 그 그림(상자 면)의 것으로 본다 (센서는 그림 가장자리 · 바로 곁에 붙는다).
    """
    w, h = float(image_width), float(image_height)
    parts = list(part_detections) if part_detections is not None else []
    everything = list(detections) + parts

    def ratio(x1, y1, x2, y2):
        return (max(0.0, min(w, x1)) / w, max(0.0, min(h, y1)) / h, max(0.0, min(w, x2)) / w, max(0.0, min(h, y2)) / h)

    # 붙어 있을 수 있는 것: 그림(그림 모델) · UGV (부품 모델이 있으면 그쪽이 진영까지 읽는다)
    hosts_pictures = []
    for name, conf, x1, y1, x2, y2 in detections:
        kind = pictures.kind_of(name)
        if kind is not None and conf >= min_host_confidence:
            hosts_pictures.append((kind, ratio(x1, y1, x2, y2)))
    robots = ugvs.build_ugvs(parts if parts else detections, w, h, rgb=rgb, min_confidence=min_host_confidence)
    if parts:   # 그림 모델만 본 UGV 도 「UGV 안」 판정에는 쓴다 (진영은 모른다)
        seen = [(u['x_center'] - u['width'] / 2, u['y_center'] - u['height'] / 2, u['x_center'] + u['width'] / 2, u['y_center'] + u['height'] / 2) for u in robots]
        for u in ugvs.build_ugvs(detections, w, h, rgb=None, min_confidence=min_host_confidence):
            box = (u['x_center'] - u['width'] / 2, u['y_center'] - u['height'] / 2, u['x_center'] + u['width'] / 2, u['y_center'] + u['height'] / 2)
            if all(_iou(box, other) <= 0.3 for other in seen):
                robots.append(u)

    found = []
    for name, conf, x1, y1, x2, y2 in sorted((d for d in everything if is_sensor(d[0])), key=lambda d: d[1], reverse=True):
        box = ratio(x1, y1, x2, y2)
        if conf < min_confidence or box[2] - box[0] <= 0.0 or box[3] - box[1] <= 0.0:
            continue
        if any(_iou(box, other['_box']) > same_sensor_iou for other in found):
            continue   # 두 모델이 같은 센서를 각각 찾았다 → 점수 높은 하나만
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        item = {'owner': UNKNOWN, 'x_center': centre[0], 'y_center': centre[1], 'width': box[2] - box[0], 'height': box[3] - box[1],
                'confidence': max(0.0, min(1.0, float(conf))), 'has_host': False, 'host_kind': '', 'camp': '', 'callsign': '',
                'host_x_center': 0.0, 'host_y_center': 0.0, 'host_width': 0.0, 'host_height': 0.0, '_box': box}

        def attach(kind, hx, hy, hw, hh):
            item.update(has_host=True, host_kind=kind, host_x_center=hx, host_y_center=hy, host_width=hw, host_height=hh)

        robot = None
        for u in robots:
            ubox = (u['x_center'] - u['width'] / 2, u['y_center'] - u['height'] / 2, u['x_center'] + u['width'] / 2, u['y_center'] + u['height'] / 2)
            if _gap(centre, ubox) <= ugv_margin and (robot is None or u['width'] * u['height'] < robot['width'] * robot['height']):
                robot = u   # 여러 대에 걸치면 작은(더 꼭 맞는) 박스의 것
        near = None
        for kind, pbox in hosts_pictures:
            gap = _gap(centre, pbox)
            if gap <= picture_margin and (near is None or gap < near[0]):
                near = (gap, kind, pbox)
        # UGV 박스와 그림 박스 양쪽에 걸리면: 센서 가운데가 UGV 박스 안이면 UGV 것, 가장자리 여유에만 걸렸으면 그림 것
        inside_robot = robot is not None and _gap(centre, (robot['x_center'] - robot['width'] / 2, robot['y_center'] - robot['height'] / 2,
                                                             robot['x_center'] + robot['width'] / 2, robot['y_center'] + robot['height'] / 2)) == 0.0
        if robot is not None and (inside_robot or near is None):
            item['owner'] = UGV
            item['camp'], item['callsign'] = robot.get('camp', ''), robot.get('callsign', '')
            attach('ugv', robot['x_center'], robot['y_center'], robot['width'], robot['height'])
        elif near is not None:
            _g, kind, pbox = near
            item['owner'] = TARGET
            attach(kind, (pbox[0] + pbox[2]) / 2.0, (pbox[1] + pbox[3]) / 2.0, pbox[2] - pbox[0], pbox[3] - pbox[1])
        elif _norm(name) in _TARGET_SENSOR_NAMES:
            item['owner'] = TARGET
        found.append(item)
    for item in found:
        del item['_box']
    found.sort(key=lambda s: s['x_center'])
    return found
