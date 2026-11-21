"""드론 화면 + 경기장 경계 → 상자 가운데 좌표 — 학습한 YOLO OBB 모델로 (drone_boxes 의 규칙 대신).

  bv, mv = detect_boxes_model(bgr, boundary, predict, camera_height, height_source)
    predict: map 그림(BGR) → [(종류 이름, 점수, 꼭짓점 픽셀 4개)]. 보통 BoxModel(path).predict
    돌려주는 값: (drone_boxes.BoxView — 지도(drone_field_map)가 그대로 쓰는 형식, drone_map_view.MapView)

순서
  1) MapView 로 map 그림을 만든다 (렌즈 왜곡을 펴고 경기장으로 잘라 상자 윗면 높이 평면 1 cm/px — 학습 그림과 같은 코드)
  2) 모델이 상자 윗면 · 줄무늬 블록을 회전 사각형으로 찾는다
  3) 꼭짓점을 경기장 좌표로 → 가운데 = 상자 바닥 자리 (map 그림이라 높이 보정이 이미 들어 있다).
     상자들이 바닥 20 cm 격자에 놓였으면 drone_boxes 와 같은 _grid_divisors 로 축마다 나눗값을 격자에 맞춰 다시 낸다
  4) 거르기 — drone_boxes 와 같은 기준: 종류가 상자가 아님(줄무늬 블록은 지도에 넣지 않는다) · 점수 낮음 · 크기 이상 ·
     끝 띠 안 · 위아래 벽에 붙음 · 다른 상자와 가운데가 가까움(점수 높은 쪽) · 경계 치수(끝 띠 · CM 반폭)가 범위 밖이면 ok 아님
  5) 순서는 drone_boxes 와 같다 (위 줄부터 0.4 m 줄, 같은 줄은 왼쪽부터) — 지도의 faces[].box 가 이 번호를 쓴다
  view.ok = 상자가 하나 이상 · 카메라 높이를 기본값으로 짐작하지 않았음 (drone_boxes 와 같은 기준)

모델 파일: 학습은 training/drone_boxes (README). 로봇에서는 .engine (TensorRT, 로봇에서 만든다) — 없으면 같은 이름의 .pt,
그것도 없으면 git 에 넣은 모델 (training/drone_boxes/trained/BUNDLED_NAME — colcon 설치 뒤에는 share/perception/models/).
ultralytics · torch 는 BoxModel.load() 안에서만 읽는다 (없으면 예외 → map_maker 가 규칙으로 돌아간다).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from perception.drone_boxes import Box, BoxParams, BoxView
from perception.drone_map_view import PLANE_H, MapView

DEFAULT_MODEL = "/home/ws/team/models/drone_boxes.engine"  # perception 그림 모델(pictures.engine)과 같은 곳
BUNDLED_NAME = "drone_boxes_v2_edgecut.pt"  # git 에 넣은 모델 — DEFAULT_MODEL 자리에 파일이 없을 때 쓴다


def bundled_model():
    """git 에 넣은 모델 파일 경로 (colcon 설치본의 share → 소스 폴더 순). 없으면 None."""
    places = []
    try:
        from ament_index_python.packages import get_package_share_directory

        places.append(os.path.join(get_package_share_directory("perception"), "models", BUNDLED_NAME))
    except Exception:  # noqa: BLE001 — ROS 가 없는 PC · 설치하지 않은 소스
        pass
    places.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "drone_boxes", "trained", BUNDLED_NAME))
    return next((p for p in places if os.path.isfile(p)), None)


def limit_threads():
    """환경변수 OMP_NUM_THREADS 가 정해져 있으면 torch 의 스레드 수를 그 값으로 다시 맞춘다. 돌려주는 값: 맞춘 수 (하지 않았으면 None).

    ultralytics 는 CPU 장치를 고를 때(모델을 처음 돌릴 때) torch.set_num_threads(min(8, 코어 − 1)) 을 불러 환경변수를 덮는다.
    계산 프로세스(map_maker.start_child · drone_ugvs)는 OMP_NUM_THREADS=1 로 띄우는데, 그것이 numpy 에는 듣고 torch 에는 듣지 않았다 →
    지도 계산 프로세스 넷이 torch 스레드를 여덟(로봇은 다섯)씩 잡고 서로 다툰다. 개발 PC(16코어)에서 넷을 함께 돌려 잰 값 2026-11-19:
    그대로면 모델 한 번에 2 ~ 5초 (모델 읽기 3 ~ 7초), 하나로 묶으면 0.08 ~ 0.12초 (읽기 2초). 로봇에서는 재지 않았다.
    환경변수가 없는 곳(perception 노드 자체 · 학습 PC)에서는 아무것도 하지 않는다.
    """
    try:
        count = int(os.environ.get("OMP_NUM_THREADS") or 0)
        if count <= 0:
            return None
        import torch

        torch.set_num_threads(count)
        return count
    except Exception:  # noqa: BLE001 — 못 묶어도 계산은 한다 (느릴 뿐이다)
        return None


@dataclass
class ModelBoxParams:
    conf: float = 0.40  # 이 점수 아래는 버린다
    box_class: str = "box"  # 지도에 넣는 종류 (모델의 다른 종류 — stripe 등 — 는 rejected 로)
    imgsz: int = 640
    min_side_m: float = 0.28  # 윗면 한 변 허용 범위 (m) — 공식 0.40, 경계 축척 오차 ±10 %
    max_side_m: float = 0.56
    dedup_m: float = 0.25  # 두 상자 가운데가 이보다 가까우면 같은 상자로 본다 (상자끼리는 0.40 m 떨어진다)
    strip_margin: float = field(default_factory=lambda: BoxParams().strip_margin)
    wall_margin: float = field(default_factory=lambda: BoxParams().wall_margin)


class BoxModel:
    """Ultralytics OBB 가중치 감싸개. predict(map 그림 BGR) → [(이름, 점수, 꼭짓점 픽셀 4개)]."""

    def __init__(self, path=DEFAULT_MODEL, imgsz=640, device=None):
        self.path, self.imgsz, self.device = path, imgsz, device
        self.model = None
        self.names = {}

    def load(self):
        path = os.path.expanduser(self.path)
        if not os.path.isfile(path):  # .engine 이 없으면 같은 이름의 .pt
            alt = os.path.splitext(path)[0] + ".pt"
            if not os.path.isfile(alt):
                alt = bundled_model()  # 그것도 없으면 git 에 넣은 모델
            if alt is None:
                raise FileNotFoundError("상자 모델 파일이 없습니다: %s (.pt · git 에 넣은 %s 도 없음)" % (path, BUNDLED_NAME))
            path = alt
        from ultralytics import YOLO

        model = YOLO(path, task="obb")
        names = model.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        self.path, self.model = path, model
        self.predict(np.zeros((500, 612, 3), np.uint8))  # 한 번 돌려 둔다 (첫 지도가 느려지지 않게)
        limit_threads()  # 위에서 ultralytics 가 되돌려 놓은 torch 스레드 수를 다시 묶는다
        return self

    def predict(self, bgr, conf=0.05):
        kw = {"imgsz": self.imgsz, "conf": conf, "verbose": False}
        if self.device is not None:
            kw["device"] = self.device
        r = self.model.predict(bgr, **kw)[0]
        if r.obb is None or len(r.obb) == 0:
            return []
        pts = r.obb.xyxyxyxy.cpu().numpy()
        return [(self.names.get(int(c), str(int(c))), float(s), [tuple(map(float, q)) for q in quad])
                for c, s, quad in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), pts)]


_LOADED = {}


def get_model(path=DEFAULT_MODEL, imgsz=640, device=None):
    """한 프로세스에서 한 번만 읽는다."""
    key = (os.path.expanduser(path), imgsz, device)
    if key not in _LOADED:
        _LOADED[key] = BoxModel(path, imgsz, device).load()
    return _LOADED[key]


def _yaw_deg(pts):
    """정사각형 방향 (−45 ~ 45°). 네 변 각을 90° 주기로 접어 평균."""
    c = s = 0.0
    for i in range(4):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % 4]
        a = math.atan2(y1 - y0, x1 - x0) * 4.0
        c, s = c + math.cos(a), s + math.sin(a)
    return math.degrees(math.atan2(s, c)) / 4.0


def detect_boxes_model(image, boundary, predict, camera_height, height_source="given", params=None, box_params=None):
    """box_params: drone_boxes.BoxParams — 경계 치수 범위 · 바닥 격자 맞추기 설정을 규칙과 같게 쓴다."""
    p = params or ModelBoxParams()
    bp = box_params or BoxParams()
    mv = MapView(boundary, camera_height, PLANE_H)
    view = BoxView(False, box_size=PLANE_H)
    view.camera, view.scale, view.height_source = mv.camera, mv.s, height_source
    view.divisor, view.divisor_source = (mv.s, mv.s), ("s", "s")
    img = mv.render(image)
    dets = predict(img)
    W, H = mv.field_w, mv.field_h
    N = np.array(mv.camera[:2])
    cands, rejected, low = [], [], 0
    for name, score, quad in dets:
        if score < p.conf:
            low += 1
            continue
        corners = mv.pixel_to_field(quad)
        P = corners.mean(axis=0)
        side = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
        why = None
        if name != p.box_class:
            why = "종류 %s (지도에 넣지 않음)" % name
        elif not p.min_side_m <= side <= p.max_side_m:
            why = "크기 %.2f m (%.2f ~ %.2f)" % (side, p.min_side_m, p.max_side_m)
        if why is not None:
            rejected.append({"x": round(float(P[0]), 3), "y": round(float(P[1]), 3), "score": round(score, 3), "reason": why})
            continue
        floor = mv.floor_of(corners)  # 바닥 지도에서 본 윗면 (drone_boxes.Box.top 과 같은 뜻)
        top = (float(floor[:, 0].min()), float(floor[:, 0].max()), float(floor[:, 1].min()), float(floor[:, 1].max()))
        cands.append({"score": float(score), "top": top, "T": mv.floor_of([P])[0], "yaw": _yaw_deg(corners), "side": side})

    # 윗면 가운데 → 상자 가운데: map 그림은 s 로 이미 되돌렸지만, 규칙(drone_boxes)처럼 상자가 바닥 20 cm 격자에 놓였으면
    # 축마다 격자에 가장 잘 맞는 나눗값으로 다시 낸다 (높이 측정 · 경계 축척 오차를 없앤다). 높이를 정해 준 때(given)는 하지 않는다 — 규칙과 같다
    div, div_source = (mv.s, mv.s), ("s", "s")
    if bp.grid_cell and height_source != "given" and cands:
        from perception.drone_boxes import _grid_divisors

        tops = [c["top"] for c in cands]
        div, div_source, grid_info = _grid_divisors(tops, tuple(N), (W / 2.0, H / 2.0), mv.s, bp)
        # 한 축만 격자에 맞으면, 다른 축은 맞은 축의 나눗값을 가운데로 다시 찾는다. 격자 값은 옆면으로 잰 s 둘레 ±7 % 에서 찾는데,
        # 한 장으로 잰 s 가 그만큼 틀리면 맞는 값이 범위 끝에 걸려 버려진다 (2026-11-19 16:14 B 경기장: s 1.132, x 축 격자 값 ≈ 1.21 이
        # 범위 끝 → s 로 내서 가장자리 상자가 15 cm 어긋남). 두 축의 나눗값은 2 ~ 3 % 만 다르다 (경계 지도의 가로 · 세로 축척 차이)
        if list(div_source).count("grid") == 1:
            good = list(div_source).index("grid")
            again, again_source, again_info = _grid_divisors(tops, tuple(N), (W / 2.0, H / 2.0), div[good], bp)
            bad = 1 - good
            if again_source[bad] == "grid":
                div = tuple(again[bad] if ax == bad else div[ax] for ax in (0, 1))
                div_source = tuple("grid" if ax == bad else div_source[ax] for ax in (0, 1))
                grid_info = list(grid_info) + [dict(again_info[bad], centre=round(float(div[good]), 4), retry=True)]
        view.debug["grid"] = grid_info
        if "grid" in div_source:
            view.notes.append("상자 자리를 바닥 격자(%.0f cm)에 맞춰 냄: 나눗값 x %.3f (%s) · y %.3f (%s), s %.3f" % (
                bp.grid_cell * 100, div[0], div_source[0], div[1], div_source[1], mv.s))
    view.divisor, view.divisor_source = tuple(float(v) for v in div), tuple(div_source)

    kept = []
    for c in cands:
        T = c["T"]
        P = (float(N[0] + (T[0] - N[0]) / div[0]), float(N[1] + (T[1] - N[1]) / div[1]))
        why = None
        if boundary.strip is not None and not (boundary.strip + p.strip_margin <= P[0] <= W - boundary.strip - p.strip_margin):
            why = "끝 띠 안 (거점 · 줄무늬 벽 블록 자리)"
        elif not (p.wall_margin <= P[1] <= H - p.wall_margin):
            why = "위아래 벽에 붙음"
        if why is not None:
            rejected.append({"x": round(P[0], 3), "y": round(P[1], 3), "score": round(c["score"], 3), "reason": why})
            continue
        kept.append((Box(P[0], P[1], c["score"], c["top"]), c["yaw"], c["side"]))
    kept.sort(key=lambda t: -t[0].score)
    final = []
    for b, yaw, side in kept:
        clash = next((o for o, _y, _s in final if math.hypot(b.x - o.x, b.y - o.y) < p.dedup_m), None)
        if clash is None:
            final.append((b, yaw, side))
        else:
            rejected.append({"x": round(b.x, 3), "y": round(b.y, 3), "score": round(b.score, 3),
                             "reason": "(%.2f, %.2f) 상자와 겹침" % (clash.x, clash.y)})
    final.sort(key=lambda t: (-round(t[0].y / 0.4), t[0].x))  # drone_boxes 와 같은 순서
    view.boxes = [b for b, _y, _s in final]
    view.rejected = rejected
    view.debug["yaw_deg"] = [round(y, 2) for _b, y, _s in final]
    view.debug["side_m"] = [round(s, 3) for _b, _y, s in final]
    # 경계 치수가 상자 찾기가 받는 범위 밖이면(구역선을 잘못 고른 경계) 규칙과 같이 지도로 내지 않는다
    sane = boundary.strip is None or (bp.strip_range[0] <= boundary.strip <= bp.strip_range[1] and bp.cm_range[0] <= boundary.cm_half <= bp.cm_range[1])
    if not sane:
        view.notes.append("경계 치수가 범위 밖 (끝 띠 %.2f · CM 반폭 %.2f m) — 지도로 내지 않음" % (boundary.strip, boundary.cm_half))
    view.notes.append("모델: 후보 %d · 점수 %.2f 미만 %d · 버림 %d · 상자 %d (카메라 높이 %.2f m, %s)" % (
        len(dets), p.conf, low, len(rejected), len(view.boxes), camera_height, height_source))
    view.ok = bool(view.boxes) and height_source != "default" and sane
    return view, mv
