"""탐지 결과를 ClassifyImage 응답의 그림 목록으로 바꾸는 순수 계산 (ROS · 모델 없이 시험 가능).

규칙은 개발/src/supervisor/docs/요청사항.md 2절을 따른다.
- 그림 하나에 답 하나, 왼쪽부터 (x_center 가 작은 것부터).
- 못 알아본 그림은 빼지 않고 unknown 으로 한 칸을 채운다.
- 가장자리에서 잘린 그림은 cut_off = true.
"""

# Picture.msg 의 kind 값. 모델이 내는 것은 앞의 10가지이고, unknown 은 점수가 낮을 때 여기서 붙인다.
MILITARY_TRUCK = 'military_truck'
MORTAR = 'mortar'
TANK = 'tank'
MISSILE = 'missile'
AMMO_BOX = 'ammo_box'
DRUM = 'drum'
COMMS = 'comms'
OUTPOST = 'outpost'
RADAR = 'radar'
NON_OBJECT = 'non_object'
UNKNOWN = 'unknown'

MODEL_KINDS = (MILITARY_TRUCK, MORTAR, TANK, MISSILE, AMMO_BOX, DRUM, COMMS, OUTPOST, RADAR, NON_OBJECT)
ALL_KINDS = MODEL_KINDS + (UNKNOWN,)

# 모델(학습 데이터)의 클래스 이름이 조금 달라도 같은 kind 로 읽는다.
_ALIASES = {
    MILITARY_TRUCK: ('military truck', 'truck', 'military vehicle', '군용트럭'),
    MORTAR: ('박격포',),
    TANK: ('전차', '탱크'),
    MISSILE: ('rocket', '미사일'),
    AMMO_BOX: ('ammo box', 'ammunition box', '탄약상자'),
    DRUM: ('oil drum', 'fuel barrel', 'barrel', '드럼통'),
    COMMS: ('communication facility', 'comms facility', '통신시설'),
    OUTPOST: ('guard post', 'guard tower', 'watchtower', '감시초소'),
    RADAR: ('radar facility', '레이더시설'),
    NON_OBJECT: ('non object', 'other', 'background', '기타'),
}


# 모델이 내더라도 "그림"이 아닌 클래스. 그림 목록에 넣으면 supervisor 의 면 짝짓기가 한 칸씩 밀리므로 뺀다.
NOT_PICTURES = ('target sensor', 'sensor', 'ugv', 'robot', 'enemy ugv', 'friendly ugv', '로봇',
                'antenna', '안테나', 'led bar', 'led', 'ugv sensor')

# 그중 지정표적의 수광부(그림 위쪽의 둥근 돔). 그림 목록에는 넣지 않고 따로 알려 준다 (build_sensors) — supervisor 가 레이저를 겨누는 데 쓴다.
SENSORS = ('target sensor', 'sensor')


def _norm(name):
    return str(name).strip().lower().replace('-', ' ').replace('_', ' ')


def is_picture(class_name):
    return _norm(class_name) not in NOT_PICTURES


_LOOKUP = {}
for _kind, _names in _ALIASES.items():
    for _name in (_kind,) + _names:
        _LOOKUP[_norm(_name)] = _kind


def is_sensor(class_name):
    return _norm(class_name) in SENSORS


def kind_of(class_name):
    """모델 클래스 이름 -> kind. 10가지에 없으면 None."""
    return _LOOKUP.get(_norm(class_name))


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def build_pictures(detections, image_width, image_height,
                   min_confidence=0.5, edge_margin=0.015, same_picture_iou=0.6):
    """탐지 목록 -> 그림 목록(dict), 왼쪽부터.

    detections: (class_name, confidence, x1, y1, x2, y2) 의 목록. 좌표는 픽셀.
    min_confidence: 이 값보다 낮거나 10가지에 없는 클래스면 kind 를 unknown 으로 낸다 (항목은 빼지 않는다).
    edge_margin: 박스가 사진 가장자리에서 이 비율 안에 닿으면 잘린 것으로 본다.
    same_picture_iou: 두 박스가 이보다 많이 겹치면 같은 그림으로 보고 점수가 높은 하나만 남긴다.
    """
    w, h = float(image_width), float(image_height)
    boxes = []
    for name, conf, x1, y1, x2, y2 in detections:
        if not is_picture(name):
            continue
        box = (max(0.0, min(w, x1)) / w, max(0.0, min(h, y1)) / h,
               max(0.0, min(w, x2)) / w, max(0.0, min(h, y2)) / h)
        if box[2] - box[0] <= 0.0 or box[3] - box[1] <= 0.0:
            continue
        boxes.append((float(conf), name, box))

    kept = []
    for conf, name, box in sorted(boxes, key=lambda b: b[0], reverse=True):
        if all(_iou(box, other) <= same_picture_iou for _, _, other in kept):
            kept.append((conf, name, box))

    pictures = []
    for conf, name, box in kept:
        kind = kind_of(name)
        if kind is None or conf < min_confidence:
            kind = UNKNOWN
        pictures.append({
            'kind': kind,
            'x_center': (box[0] + box[2]) / 2.0,
            'y_center': (box[1] + box[3]) / 2.0,
            'width': box[2] - box[0],
            'height': box[3] - box[1],
            'confidence': max(0.0, min(1.0, conf)),
            'cut_off': (box[0] <= edge_margin or box[1] <= edge_margin
                        or box[2] >= 1.0 - edge_margin or box[3] >= 1.0 - edge_margin),
        })
    pictures.sort(key=lambda p: p['x_center'])
    return pictures


def build_sensors(detections, image_width, image_height, min_confidence=0.15, same_sensor_iou=0.5, limit=12, exclude_boxes=()):
    """탐지 목록 -> 수광부 목록(dict), 왼쪽부터. 수광부가 아닌 클래스는 무시한다.

    detections: (class_name, confidence, x1, y1, x2, y2) 의 목록. 좌표는 픽셀.
    min_confidence: 이보다 낮은 탐지는 뺀다 (어느 것을 쓸지는 supervisor 가 그림과의 자리로 고른다 — 여기서는 넉넉히 넘긴다).
    same_sensor_iou: 두 박스가 이보다 많이 겹치면 같은 수광부로 보고 점수가 높은 하나만 남긴다.
    limit: 점수가 높은 것부터 이 개수까지만.
    exclude_boxes: (x1, y1, x2, y2) 픽셀 상자들 — 가운데가 이 안에 든 수광부는 뺀다. 사진 속 UGV 의 상자를 넣는다:
        UGV 에 달린 수광부는 지정표적의 것과 생김새가 같아, 표적 앞에 UGV 가 서 있으면 그 돔을 겨눌 수 있다.
    """
    w, h = float(image_width), float(image_height)
    boxes = []
    for name, conf, x1, y1, x2, y2 in detections:
        if not is_sensor(name) or conf < min_confidence:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2 for ex1, ey1, ex2, ey2 in exclude_boxes):
            continue
        box = (max(0.0, min(w, x1)) / w, max(0.0, min(h, y1)) / h,
               max(0.0, min(w, x2)) / w, max(0.0, min(h, y2)) / h)
        if box[2] - box[0] <= 0.0 or box[3] - box[1] <= 0.0:
            continue
        boxes.append((float(conf), box))

    kept = []
    for conf, box in sorted(boxes, key=lambda b: b[0], reverse=True):
        if len(kept) < limit and all(_iou(box, other) <= same_sensor_iou for _, other in kept):
            kept.append((conf, box))

    sensors = [{
        'x_center': (box[0] + box[2]) / 2.0,
        'y_center': (box[1] + box[3]) / 2.0,
        'width': box[2] - box[0],
        'height': box[3] - box[1],
        'confidence': max(0.0, min(1.0, conf)),
    } for conf, box in kept]
    sensors.sort(key=lambda s: s['x_center'])
    return sensors


def sensors_text(sensors):
    """수광부 목록 -> ClassifyImage 답의 detail 끝에 붙이는 글. 메시지 정의를 바꾸지 않고 넘기려고 글로 싣는다.

    모양: sensors=[[x_center, y_center, width, height, confidence], ...]  (사진 크기에 대한 비율, 왼쪽부터. 없으면 sensors=[])
    읽는 쪽은 supervisor 의 aim.parse_sensors.
    """
    rows = ','.join('[%.4f,%.4f,%.4f,%.4f,%.2f]' % (s['x_center'], s['y_center'], s['width'], s['height'], s['confidence'])
                    for s in sensors)
    return 'sensors=[%s]' % rows
