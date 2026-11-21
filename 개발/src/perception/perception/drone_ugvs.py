"""드론 화면 → 경기장의 UGV 자리들 — 학습한 YOLO OBB 모델(training/drone_boxes 의 ugv_v1, 클래스 ugv 하나)로.

  view = UgvView(boundary, camera_height, arena=(6.12, 4.80))    # 경기마다 한 번 (지도를 만든 경계 · 카메라 높이 그대로)
  ugvs = view.detect(bgr, predict)                               # [(x, y, 점수)] m, frame arena — 우리 차 · 다른 차 모두

순서
  1) MapView 로 map 그림을 만든다 (상자 찾기 모델과 같은 그림 — 학습 그림이 그렇게 만들어졌다)
  2) 모델이 UGV 를 회전 사각형으로 찾는다 → 가운데
  3) 높이 고치기: map 그림은 높이 0.40 m(상자 윗면) 평면으로 편 것이라 그보다 낮은 UGV 는 카메라 바로 아래 점 쪽으로 끌려 찍힌다.
     그 픽셀의 바닥 점 F = N + (P − N)·s 에서, 높이 hu 인 점의 참 자리 = N + (F − N)·(Hc − hu)/Hc.
     hu(ugv_height, 기본 0.10 m)는 공식 테스트 경기의 사진으로 맞춘 값이다 (아래 UGV_HEIGHT 의 설명 — 차체 전체 높이 0.32 m 보다 낮다: 라벨 네모의
     가운데가 바닥 쪽으로 치우친다). 카메라 2.55 m 에서 hu 가 5 cm 틀리면 가운데에서 2.5 m 떨어진 자리가 5 cm 어긋난다.
     navigation 이 우리 차의 드론 자리와 라이다 위치의 차이(ugvs.self_off)를 기록하므로 그것으로 다시 맞춘다
  4) 지도와 같은 좌표로: 지도 크기(arena)가 경계의 크기(6.12 × 5.00)와 다르면 정중앙을 그대로 두고 옮긴다 (drone_field_map.field_map 과 같은 셈)
  5) 거르기: 점수 낮음 · 경기장 밖 · 가운데가 가까운 두 개는 점수 높은 쪽만

모델 파일: 로봇의 DEFAULT_MODEL(.engine — 로봇에서 만든다) → 같은 이름의 .pt → git 에 넣은 BUNDLED_NAME. ultralytics · torch 는 load_model 안에서만 읽는다.
numpy · OpenCV 만 쓴다. ROS 에 묶이지 않는다 (연결은 ugv_watch.py).
"""
from __future__ import annotations

import math
import os

import numpy as np

from perception.drone_map_view import PLANE_H, MapView

DEFAULT_MODEL = "/home/ws/team/models/drone_ugvs.engine"
BUNDLED_NAME = "drone_ugv_v1.pt"
# 위의 3). 공식 테스트 경기 6판(2026-11-19 16:55 ~ 18:06, 본선 카메라 · 카메라 높이 2.55 m)의 드론 사진에 이 모델을 돌려, 우리 차의 드론 자리를
# 그때의 라이다 위치와 견준 값 (441장, 가운데에서 0.8 m 넘게 떨어진 것): 바깥쪽 치우침 가운데값 0.00 → +0.05 m · 0.10 → −0.01 · 0.20 → −0.07 · 0.30 → −0.12.
# 연습 카메라(test1 · test2)로는 재지 않았다
UGV_HEIGHT = 0.10


def bundled_model():
    """git 에 넣은 UGV 모델 파일 경로 (colcon 설치본의 share → 소스 폴더 순). 없으면 None."""
    places = []
    try:
        from ament_index_python.packages import get_package_share_directory

        places.append(os.path.join(get_package_share_directory("perception"), "models", BUNDLED_NAME))
    except Exception:  # noqa: BLE001 — ROS 가 없는 PC · 설치하지 않은 소스
        pass
    places.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "drone_boxes", "trained", BUNDLED_NAME))
    return next((p for p in places if os.path.isfile(p)), None)


def find_model(path=DEFAULT_MODEL):
    """쓸 모델 파일: path → 같은 이름의 .pt → git 에 넣은 것. 없으면 None."""
    path = os.path.expanduser(path or DEFAULT_MODEL)
    if os.path.isfile(path):
        return path
    alt = os.path.splitext(path)[0] + ".pt"
    return alt if os.path.isfile(alt) else bundled_model()


def load_model(path=DEFAULT_MODEL, imgsz=640, device=None):
    """모델을 읽어 predict(map 그림 BGR) → [(이름, 점수, 꼭짓점 픽셀 4개)] 를 돌려준다. 못 읽으면 예외."""
    from perception.drone_box_model import BoxModel

    found = find_model(path)
    if found is None:
        raise FileNotFoundError("UGV 모델 파일이 없습니다: %s (.pt · git 에 넣은 %s 도 없음)" % (path, BUNDLED_NAME))
    return BoxModel(found, imgsz, device).load()


class UgvView:
    def __init__(self, boundary, camera_height, arena=None, ugv_height=UGV_HEIGHT, ugv_class="ugv", conf=0.5, dedup_m=0.15, outside_m=0.10):
        self.mv = MapView(boundary, camera_height, PLANE_H)
        w0, h0 = self.mv.field_w, self.mv.field_h
        self.width, self.height = (float(arena[0]), float(arena[1])) if arena else (w0, h0)
        self.shift = np.array([(self.width - w0) / 2.0, (self.height - h0) / 2.0])
        hc = float(camera_height)
        self.ugv_height = min(max(float(ugv_height), 0.0), hc - 0.1)
        self.lift = (hc - self.ugv_height) / hc  # 바닥 점 → 높이 hu 평면의 점
        self.ugv_class, self.conf, self.dedup_m, self.outside_m = ugv_class, float(conf), float(dedup_m), float(outside_m)

    def to_arena(self, plane_xy):
        """map 그림 평면(0.40 m)의 점들 → UGV 높이로 고친 지도 좌표."""
        n = np.array(self.mv.camera[:2])
        floor = self.mv.floor_of(plane_xy)
        return n + (floor - n) * self.lift + self.shift

    def detect(self, bgr, predict):
        """→ [(x, y, 점수)] 점수 높은 것부터. predict 는 load_model(...).predict 또는 같은 꼴의 함수."""
        found = []
        for name, score, quad in predict(self.mv.render(bgr)):
            if score < self.conf or (self.ugv_class and name != self.ugv_class):
                continue
            x, y = self.to_arena([self.mv.pixel_to_field(quad).mean(axis=0)])[0]
            if not (-self.outside_m <= x <= self.width + self.outside_m and -self.outside_m <= y <= self.height + self.outside_m):
                continue
            found.append((float(x), float(y), float(score)))
        found.sort(key=lambda u: -u[2])
        kept = []
        for u in found:
            if all(math.hypot(u[0] - k[0], u[1] - k[1]) >= self.dedup_m for k in kept):
                kept.append(u)
        return kept

    def seen_y(self, share=0.5):
        """카메라에 찍히는 세로 범위 (y_min, y_max) m — UGV 높이 · 지도 좌표. 가로줄의 share 넘게 화면 안이면 보이는 줄로 친다.
        화면 위아래가 잘려 벽 쪽이 안 보이는 카메라에서 navigation 의 「드론이 못 보는 띠」(drone_blind_wall_m)를 정하는 데 쓴다 (기록용)."""
        mu, mv = self.mv.maps()
        iw, ih = self.mv.boundary.image_size
        inside = (mu >= 0) & (mu <= iw - 1) & (mv >= 0) & (mv <= ih - 1)
        rows = np.nonzero(inside.mean(axis=1) >= share)[0]
        if rows.size == 0:
            return None
        cx = self.mv.size[0] / 2.0
        top, bottom = self.to_arena(self.mv.pixel_to_field([(cx, float(rows[0])), (cx, float(rows[-1]))]))[:, 1]
        return (round(float(min(top, bottom)), 3), round(float(max(top, bottom)), 3))
