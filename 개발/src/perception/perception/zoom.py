"""작은 그림을 사진에서 잘라 키워 한 번 더 판독하는 계산 (ROS · 모델 없이 시험 가능).

로봇 카메라는 640x480 이고 모델도 640 으로 보므로, 먼 그림은 줄여서 망가진 것이 아니라 원래 작게 찍힌 것이다.
모델은 대부분 가까이서 찍힌(큰) 그림으로 배웠다 → 작은 그림 둘레를 잘라 모델 입력 크기로 키우면
학습 때와 비슷한 크기로 보게 된다.

- 박스 자리는 처음 탐지 그대로 두고 종류 · 점수만 다시 정한다 (자리는 작아도 잘 맞는다 — 틀리는 것은 종류다).
- 다시 본 답의 점수가 처음 답보다 높을 때만 바꾼다. 같은 자리의 그림이 없거나 점수가 낮으면 처음 답을 그대로 둔다
  (확대해서 답이 사라지거나, 가까이서 이미 맞힌 답이 낮은 점수의 답으로 덮이지 않게).
- 그림이 아닌 탐지(ugv · target_sensor 등)는 건드리지 않는다.

모델 추론은 infer(rgb) -> [(class_name, confidence, x1, y1, x2, y2)] 로 받는다 (PictureDetector.detect).
"""


def crop_window(box, image_width, image_height, fill=0.4, min_side=64):
    """box 를 가운데에 두고, box 의 긴 변이 창의 fill 배가 되는 정사각형 창 (x0, y0, x1, y1) 정수 픽셀.

    창은 사진 안으로 밀어 넣는다. 사진보다 크게는 만들지 않는다.
    """
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) / float(fill)
    side = int(round(min(max(side, min_side), image_width, image_height)))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0 = int(round(min(max(cx - side / 2.0, 0), image_width - side)))
    y0 = int(round(min(max(cy - side / 2.0, 0), image_height - side)))
    return x0, y0, x0 + side, y0 + side


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def best_match(box, candidates, min_iou=0.3):
    """candidates (사진 좌표의 탐지 목록) 중 box 와 가장 많이 겹치는 것. min_iou 미만이면 None."""
    best, best_iou = None, min_iou
    for cand in candidates:
        u = _iou(box, cand[2:6])
        if u >= best_iou:
            best, best_iou = cand, u
    return best


def refine(rgb, detections, infer, is_picture, max_height=96, fill=0.25, limit=4, min_iou=0.3):
    """detections 중 작은 그림을 잘라 키워 다시 판독한 목록 (순서 · 길이 · 박스 자리는 그대로, 바뀌는 것은 종류 · 점수).

    rgb: HxWx3 uint8. detections: infer(rgb) 의 결과. is_picture(class_name) -> bool.
    max_height: 박스 높이가 이 픽셀 이하인 그림만 다시 본다. 0 이하면 아무것도 하지 않는다.
    fill: 잘라 낸 창에서 그림이 차지할 비율 (긴 변 기준). 작을수록 둘레가 많이 들어간다.
    limit: 한 사진에서 다시 보는 그림 수의 상한 (추론 한 번에 로봇에서 약 35 ms). 작은 것부터.
    """
    if max_height <= 0 or limit <= 0:
        return list(detections)
    height, width = rgb.shape[:2]
    small = [i for i, d in enumerate(detections)
             if is_picture(d[0]) and 0 < d[5] - d[3] <= max_height]
    small.sort(key=lambda i: detections[i][5] - detections[i][3])
    out = list(detections)
    for i in small[:limit]:
        name, conf, x1, y1, x2, y2 = detections[i]
        x0, y0, cx1, cy1 = crop_window((x1, y1, x2, y2), width, height, fill=fill)
        crop = rgb[y0:cy1, x0:cx1]
        found = [(n, p, a + x0, b + y0, c + x0, d + y0) for n, p, a, b, c, d in infer(crop) if is_picture(n)]
        match = best_match((x1, y1, x2, y2), found, min_iou=min_iou)
        if match is not None and match[1] > conf:
            out[i] = (match[0], match[1], x1, y1, x2, y2)
    return out
