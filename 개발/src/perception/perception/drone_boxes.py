"""드론 화면 한 장 + 경기장 경계(drone_boundary) → 상자(40 cm 정육면체) 가운데 좌표.

좌표계는 drone_boundary · sim-strategy team/navigation/fieldmap.py 와 같다: x 는 BLUE 쪽 벽(0) → RED 쪽 벽, y 는 아래(0) → 위. 단위 m.

높이 때문에 생기는 어긋남 (시차)
  드론 카메라는 경기장 바로 위 한 점에 있다 (현장 확인 2026-11-18: 연습 · 본선 모두 카메라 x, y = 경기장 정중앙, 건물 천장에 붙어 있음).
  경계 대응(Boundary.pixel_to_field)은 바닥(높이 0) 전용이라, 높이 h 인 점을 그대로 옮기면 카메라 바로 아래 점 N 에서
  s = Hc / (Hc − h) 배 먼 곳으로 간다 (Hc = 카메라 높이). 그래서 바닥 지도에서 상자 윗면은 0.4·s 크기의 정사각형으로,
  N 에서 바깥으로 밀려 보이고, 그 사이에 N 쪽 옆면(그림)이 보인다.
      상자 가운데 P = N + (T − N) / s          T = 바닥 지도에서 본 윗면 가운데
  s 는 옆면 폭으로 잰다: 옆면의 바닥 모서리 f 와 윗면 모서리 t 사이에 f = N + (t − N)/s 가 성립한다.
  면마다 s 를 재고 중앙값을 쓴다. 알고 있으면 camera_height 로 줄 수 있다.
  연습 화면에서 잰 값 (2026-11-18): test1 · test2 모두 s ≈ 1.17 ~ 1.19 → 카메라 높이 약 2.5 ~ 2.7 m.
  합성 장면 시험에서 잰 높이는 실제보다 약 2 % 낮다. 화면마다 ±5 % 흔들리므로 경기 중에는 처음 몇 장의 중앙값으로 고정해 쓰는 것이 낫다.

순서
  1) 화면을 바닥 지도(위에서 본 그림, 1 cm = 1 px)로 편다.
  2) 바닥에 칠한 선(구역선 · 줄 경계 · 중앙 원 — 경계 모델이 위치를 안다)을 지운 뒤, 네 변이 곧은 정사각형(윗면)을 찾는다.
     선을 지우지 않으면 선과 상자 모서리가 이루는 빈 바닥 칸이 상자로 잡힌다. 한 변이 화면 끝에 걸린 윗면은 세 변으로 찾는다.
     가운데 구역(CM) 안에는 가로 줄 경계가 칠해져 있지 않다 → 그 높이에 놓인 윗면 변도 「안 보이는 변」 으로 치고 세 변으로 찾는다.
  3) 윗면마다 네 변을 ±1.5 cm 안에서 다듬는다. 크기가 이 화면 윗면들의 중앙값과 크게 다르면 카메라에서 먼 변을 기준으로 맞춘다.
     카메라 쪽 옆면의 바닥 모서리로 s 를 잰다.
  3-b) 붙은 상자 줄 (2026-11-19): 맞닿은 두 윗면 사이의 모서리가 안 보이면(둘 다 검은 판 · 또는 그 모서리가 칠한 선 위) 2) 는 둘 다 놓친다.
     줄 전체(1 × 2 · 1 × 3 직사각형)의 바깥 네 변으로 찾아 같은 크기로 나눈다 (_joined_candidates). 나눈 윗면마다 테두리 글자가 보여야 한다.
  4) 상자 가운데를 구하고 거른다: 크기가 다름 · 화면 밖으로 잘림 · 노란색(거점 테두리 · 줄무늬 블록) · 끝 띠 안 ·
     카메라 쪽 옆면이 보여야 하는데 바닥만 보임 · 다른 상자와 겹침.
  5) 화면 끝에 걸린 상자 (2026-11-19): 카메라는 경기장을 다 비추지 않는다 (연습 경기장은 위아래, 본선 경기장은 좌우도 잘릴 수 있다).
     윗면의 먼 변이 화면 밖이면 2) 가 못 찾는다 → 카메라 쪽 옆면의 바닥선(바닥과 옆면이 만나는 곧은 선)으로 찾는다 (_border_boxes).
     바닥선은 높이 0 이라 시차가 없다. 벽에서 0.45 m 안(상자 가운데 기준)의 것은 내지 않는다 (border_wall_gap — 주최측: 객체 상자는 벽에서 40 cm 넘게 띄운다).
     상자 가운데 P = N + (T − N) / κ 의 나눗값 κ 는 축마다 따로: 상자들이 바닥의 20 cm 격자에 맞춰 놓여 있으면 격자에 맞는 값,
     아니면 s 그대로 (_grid_divisors). 그렇게 낸 좌표는 「경기장 정중앙에서 몇 m」 가 맞는 값이다.

한계 [확인 필요]
  - 경계 지도 자체의 축척 오차를 그대로 물려받는다 (A4 용지로 잰 test2 지도: x 비율 0.97 ~ 1.07, y 비율 1.02 ~ 1.15).
    test1 은 렌즈값이 화면마다 흔들려 지도 축척이 화면마다 ±10 % 달라진다.
    상자들이 20 cm 격자에 놓인 배치에서는 κ 를 격자에 맞춰 이 오차를 없앤다 (연습 화면: 격자점과의 차이 RMS 1 ~ 2.5 cm).
    격자와 무관하게 놓인 배치(2026-11-18 1경기장)에서는 전과 같다.
  - 벽의 자리는 여기서 재지 않는다 (화면 위아래가 잘려 위 · 아래 벽이 안 보인다). 2026-11-19 2경기장 실차: 라이다가 본 벽은 지도의
    6.12 × 5.00 보다 가로 6 cm · 세로 10 cm 씩 안쪽이었다 (포털 모형 600 × 480 과 같음). 지도에 적는 벽 크기는 drone_field_map.field_map 의
    arena 로 바꿀 수 있다 (기본은 전과 같은 6.12 × 5.00).
  - 화면 끝에 걸린 상자: 카메라 쪽 옆면이 다른 상자에 가려 있으면 못 찾는다. 옆면 그림이 같은 두 상자가 붙어 있으면 둘로 나누지 못한다.
    벽 가까이(0.45 m 안)는 내지 않는다 — 그 자리에서는 UGV · 사람이 뒤의 벽 그림 · 블록과 함께 옆면처럼 보인다 (연습 사진에서 바닥선의 곧은 정도로
    걸러지는 것을 확인했지만 본선 경기장의 벽은 본 적이 없다). 바닥 글자 「COMMON」 자리는 맞춤 좌표로 적어 둔 것이라 2 cm 쯤 어긋날 수 있다.
  - 붙은 상자 줄은 바깥 네 변이 모두 반 넘게 보여야 한다 — 바깥 변이 칠한 선을 따라 놓인 줄 · 2 × 2 로 붙은 덩어리는 못 찾는다.
  - 윗면 변은 화면 1 px (약 1 ~ 2 cm) 정도로만 잡힌다 → 합성 장면에서 가운데 오차 최대 약 5 cm.
  - 노란 줄무늬 벽 블록 · 끝 띠 안의 물체는 내지 않는다 (상자 목록은 구역 안의 그림 상자용).
  - 상자는 경기장 축에 나란하다고 본다. 돌아간 상자 · 사람이 가린 상자는 놓칠 수 있다.

numpy · OpenCV 만 쓴다. ROS 에 묶이지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SIDES = ("E", "W", "N", "S")
_FILL = (60, 0, 60)  # 화면 밖을 채우는 색 (바닥 지도)


@dataclass
class BoxParams:
    box_size: float = 0.40  # 정육면체 한 변 · 높이 (공식 40 cm)
    ppm: int = 100  # 바닥 지도 해상도 (px / m)
    margin: float = 0.3  # 경기장 밖으로 더 펴는 폭 (m)
    # 윗면 정사각형 찾기
    grad_min: float = 60.0  # 이보다 센 밝기 변화만 모서리로 본다 (Sobel, 0~255 영상)
    top_px: tuple = (38, 72, 2)  # 바닥 지도에서 본 윗면 한 변 후보 (px): 0.4·s 에 지도 축척 오차(화면마다 ±10 %)가 곱해진다
    edge_frac: float = 0.8  # 변의 가운데 이 비율만 본다 (모서리 근처는 이웃 상자와 섞인다)
    tex_weight: float = 1.5  # 윗면 안쪽이 무늬로 가득하면 (옆면 그림 · 벽) 감점
    min_score: float = 0.5  # 정사각형 점수 (네 변 중 가장 약한 변의 모서리 비율 − 감점)
    nms_px: int = 31
    refine_m: float = 0.015  # 변 다듬기 범위 (±m)
    size_tol: float = 0.06  # 윗면 크기가 이 화면 윗면들의 중앙값과 이보다 다르면 카메라 쪽 변을 맞춘다 (m). 경계 지도 축척이 곳에 따라 ±5 % 넘게 달라 (A4: y 비율 1.02~1.15) 더 좁히면 맞는 변도 옮긴다
    line_mask_m: float = 0.035  # 칠한 선 가운데에서 이 거리 안은 지운다
    row_lines_cross_cm: bool = False  # 가로 줄 경계가 가운데 구역(CM)을 가로질러 칠해진 경기장이면 True (연습 경기장 1 · 2번은 아니다).
    #                                   False 면 CM 안의 그 높이에 놓인 윗면 변을 「안 보이는 변」 으로 치고 나머지 세 변으로 찾는다
    # 카메라
    camera_xy: tuple | None = None  # None 이면 경기장 정중앙 (현장 확인 2026-11-18)
    camera_height: float | None = None  # 알면 준다 (m). None 이면 옆면 폭으로 잰다
    height_range: tuple = (1.5, 10.0)  # 잴 때 허용하는 카메라 높이
    default_height: float = 2.65  # 옆면을 잴 수 없을 때 (연습 화면에서 잰 값)
    face_min_width: float = 0.08  # 이보다 넓게 보일 옆면만 s 를 재는 데 쓴다 (m, 바닥 지도)
    face_min_score: float = 0.35
    face_min_lever: float = 0.7  # 카메라 바로 아래에서 윗면 모서리까지 이보다 먼 면만 (가까우면 1 cm 오차가 s 를 크게 흔든다)
    min_faces: int = 3
    # 거르기
    min_top_m: float = 0.36  # 바닥 지도에서 본 윗면 한 변 허용 범위 (m) — 넓게 두고, 이 화면 윗면들의 중앙값 ±top_rel_tol 로 다시 거른다
    max_top_m: float = 0.72
    top_rel_tol: float = 0.2
    check_width: float = 0.06  # 카메라 쪽 옆면이 이보다 넓게 보여야 할 때만 확인한다
    min_face_nonfloor: float = 0.3  # 그 옆면 자리에서 바닥이 아닌 픽셀 비율이 이 이상이어야 한다
    max_yellow: float = 0.015  # 윗면 안 노란 픽셀 비율 상한 — 상자 윗면은 검정 · 흰색이고, 거점 테두리 · 줄무늬 블록은 노랗다
    yellow_h: tuple = (15, 40)  # HSV (OpenCV 범위)
    yellow_s: int = 60  # test1 조명에서 거점 노란 선이 채도 60~90 으로 흐리게 찍힌다
    yellow_v: int = 110
    strip_margin: float = 0.1  # 상자 가운데가 끝 띠 선보다 이만큼 안쪽이어야 한다 (m). 연습 경기장 상자는 0.5 m 넘게 안쪽
    wall_margin: float = 0.2  # 위아래 벽에서 상자 가운데까지 최소 (m)
    # 경계 결과 확인 — 세로선 5개 조합을 잘못 고른 화면은 b.ok 가 True 여도 끝 띠 · CM 반폭이 엉뚱하다 (경계 세션 일괄 시험 2026-11-18).
    # 연습 경기장에서 잰 값 끝 띠 0.82~0.92 · CM 반폭 0.59~0.63 에 여유를 둔 범위. 벗어나면 상자는 찾되 ok=False
    strip_range: tuple = (0.75, 1.0)
    cm_range: tuple = (0.55, 0.70)
    max_overlap: float = 0.2  # 두 상자 바닥 면적이 이 비율 넘게 겹치면 점수가 낮은 쪽을 버린다
    # 붙은 상자 줄 — 맞닿은 두 윗면 사이의 모서리가 안 보일 때 (둘 다 검은 판이라 밝기 차이가 없거나, 그 모서리가 칠한 선 위에 있을 때)
    joined: bool = True
    joined_counts: tuple = (2, 3)  # 한 줄로 붙은 상자 수
    joined_size_tol: float = 0.12  # 윗면 한 변을 이 화면 윗면들의 중앙값 ±이 비율 안에서 찾는다 (중앙값은 글자 띠를 변으로 짚어 9 % 까지 작게 나온다)
    joined_min_score: float = 0.6  # 줄 전체의 바깥 네 변 중 가장 약한 변의 모서리 비율 − 감점. 연습 화면의 진짜 줄은 0.75 ~ 1.0
    joined_min_seen: float = 0.5  # 바깥 변마다 칠한 선 · 화면 밖이 아닌 부분이 이 비율 이상 보여야 한다
    joined_line_margin: float = 0.02  # 줄 찾기 · 글자 띠 확인은 칠한 선 둘레를 line_mask_m 보다 이만큼 더 넓게 뺀다 (선 모서리가 띠 밖으로 새는 것)
    joined_refine_m: float = 0.03  # 줄의 바깥 변 다듬기 범위 (±m). 줄은 반으로 줄인 지도에서 찾아 자리가 2 cm 까지 어긋난다 → refine_m 보다 넓게
    joined_max_cover: float = 0.3  # 나눈 윗면이 이미 찾은 윗면과 이 비율 넘게 겹치면 그 줄은 버린다 (따로 찾은 것이 낫다)
    joined_min_frame: float = 0.2  # 나눈 윗면마다 테두리 글자 띠(변에서 1.5 ~ 6.5 cm 안쪽)가 적어도 두 변에서 이 비율 넘게 보여야 한다.
    #                                상자들 사이의 빈 바닥도 바깥 변이 곧은 직사각형이지만 글자 띠가 없다 (test1 의 빈 바닥: 네 변 모두 0.00,
    #                                2경기장의 진짜 줄 네 곳: 둘째로 잘 보이는 변이 0.24 ~ 0.53)
    # 화면 끝에 걸린 상자 — 윗면의 먼 변이 화면 밖이라 정사각형으로는 못 찾는 것. 카메라 쪽 옆면의 바닥선으로 찾는다 (_border_boxes)
    border: bool = True
    border_wall_gap: float = 0.45  # 이렇게 찾은 상자의 가운데는 위아래 벽(맞춤 좌표의 0 · H)에서 이만큼은 떨어져야 한다 (m).
    #   주최측 답(2026-11-19): 객체 상자는 벽에 붙여 놓지 않는다 — 벽과 상자 사이에 40 cm 넘게 비운다 (가운데는 벽에서 0.6 m 넘게).
    #   본선 경기장 세로가 5.00 m 라면 벽은 맞춤 좌표로 0.1 m 더 바깥이라, 0.6 이 아니라 0.45 로 둔다 (두 크기 모두에서 맞게).
    #   벽 가까이에서는 서 있는 UGV 와 그 뒤의 벽 그림 · 블록이 함께 「옆면」 처럼 보여 없는 상자가 나온다 (13:06 1경기장 사진).
    #   연습 경기장처럼 벽에 붙여 놓은 상자(11:47 판의 위쪽 셋)까지 찾으려면 0.2 로 낮춘다 — 그때는 벽 가까이의 UGV 가 상자로 나올 수 있다
    border_min_straight: float = 0.75  # 바닥선을 따라 밝기 변화가 뚜렷한 점의 비율 하한. 연습 사진: 벽에 붙은 진짜 상자 0.78 ~ 1.00,
    #                                    벽 가까이의 UGV · 걸어가는 사람에서 나온 것 0.25 ~ 0.69
    border_min_edge: float = 0.7  # 바닥선(상자 한 변 길이의 가운데 80 %)에서 모서리 픽셀 비율
    border_min_lever: float = 0.8  # 카메라 바로 아래에서 바닥선까지 이보다 멀어야 한다 (m) — 가까우면 옆면이 거의 안 보인다
    border_min_depth: float = 0.05  # 바닥선에서 화면 끝까지 옆면이 이만큼은 보여야 한다 (m)
    border_min_face: float = 0.6  # 옆면 자리(가운데 · 양쪽 가장자리 따로)에서 바닥이 아닌 픽셀 비율 하한 — 옆면은 그림이라 무늬 · 색이 있다
    border_labels: tuple = ((2.45, 2.97, 4.385, 4.49), (3.15, 3.67, 0.48, 0.615))  # 바닥 글자 「COMMON」 두 곳 (x0, x1, y0, y1) m — 맞춤 좌표.
    #   글자의 아랫선이 곧은 바닥선처럼 보이고, 그 위에 줄무늬 블록이나 상자가 있으면 「옆면」 조건까지 채운다 → 이 안의 모서리는 바닥선으로 치지 않는다
    border_max_floor: float = 0.55  # 바닥선의 카메라 쪽 띠에서 바닥이 아닌 픽셀 비율 상한 (바닥 글자 「COMMON」 이 바로 앞에 있어도 받게 0.4 보다 넉넉히)
    # 격자에 맞춘 시차 — 상자는 바닥의 20 cm 격자에 맞춰 놓인다 (포털의 경기장 모형: 격자 20 cm, 상자 2 × 2칸. 격자점 하나가 경기장 정중앙)
    grid_cell: float = 0.20  # 격자 한 칸 (m). 0 이면 쓰지 않고 옆면으로 잰 s 만 쓴다
    grid_range: float = 0.07  # 옆면으로 잰 s 의 ±이 비율 안에서 찾는다 (더 넓히면 한 칸 옆 격자점에 맞는 값이 섞인다)
    grid_min_lever: float = 0.6  # 카메라 바로 아래에서 이보다 먼 윗면만 쓴다 (m, 축마다). 가까운 것은 어느 값으로 나눠도 격자점에 맞는다
    grid_min_boxes: int = 5  # 그런 윗면이 이보다 적으면 맞추지 않는다
    grid_max_rms: float = 0.025  # 가장 잘 맞는 값에서도 격자점과의 차이(RMS, m)가 이보다 크면 격자에 놓인 것이 아니라고 보고 맞추지 않는다.
    #                              가장 안 맞는 1/6 은 빼고 센다 (삐뚤게 놓인 상자 · 변을 잘못 짚은 윗면 하나 때문에 버리지 않게).
    #                              연습 화면: 격자에 놓인 배치 0.007 ~ 0.026, 격자와 무관하게 놓인 배치(1경기장 2026-11-18) 0.029 ~ 0.041


@dataclass
class Box:
    x: float  # 가운데 (m, 경기장 좌표)
    y: float
    score: float  # 윗면 정사각형 점수 0~1
    top: tuple  # 바닥 지도에서 본 윗면 (x0, x1, y0, y1) m — 시차로 커지고 밀린 것
    face_check: float | None = None  # 카메라 쪽 옆면 자리의 바닥 아닌 비율 (None: 확인할 옆면 없음)
    cut: str | None = None  # 윗면 한 변이 화면 밖이면 그 변 (W · E · S · N) — 그 방향 좌표는 다른 상자들의 윗면 크기로 채운 값
    joined: bool = False  # 붙은 상자 줄을 나눠서 얻은 상자 (윗면 사이 모서리가 안 보여 줄 전체의 바깥 변으로 찾음)
    row: int | None = None  # 그 줄의 번호 (같은 줄에서 나온 상자끼리 같다. 따로 찾은 상자는 None)
    border: bool = False  # 화면 끝에 걸려 윗면이 다 보이지 않아, 카메라 쪽 옆면의 바닥선으로 찾은 상자 (cut = 화면 밖으로 나간 쪽)

    def to_dict(self):
        return {"x": round(self.x, 3), "y": round(self.y, 3), "score": round(self.score, 3), "top": [round(float(v), 3) for v in self.top],
                "face_check": None if self.face_check is None else round(self.face_check, 3), "cut": self.cut, "joined": self.joined,
                "border": self.border}


@dataclass
class BoxView:
    ok: bool
    boxes: list = field(default_factory=list)  # [Box] 위 줄부터, 같은 줄은 왼쪽부터
    camera: tuple = (0.0, 0.0, 0.0)  # (x, y, 높이) m
    scale: float = 1.0  # s = Hc / (Hc − 상자 높이)
    height_source: str = ""  # given · measured · default
    faces_used: int = 0  # s 를 잰 옆면 수
    divisor: tuple = (1.0, 1.0)  # 윗면 가운데를 상자 가운데로 옮길 때 실제로 쓴 나눗값 (x, y). 격자에 맞추지 못한 축은 s 그대로
    divisor_source: tuple = ("", "")  # 축마다 grid (격자에 맞춤) · s (옆면으로 잰 값 그대로)
    box_size: float = 0.40
    rejected: list = field(default_factory=list)  # [{"x", "y", "reason"}] 버린 후보 (P 기준)
    notes: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)

    def positions(self):
        """[(x, y)] 상자 가운데."""
        return [(b.x, b.y) for b in self.boxes]

    def field_map(self, boundary, stamp=None, base_naming=None, base_kinds=None, arena=None):
        """team_interfaces/FieldMap 형식 dict (/team/perception/field_map 으로 내보낼 내용). 자세한 것은 perception/drone_field_map.py.

        stamp: 드론 사진을 찍은 시각 (초). ROS 메시지는 drone_field_map.field_map_msg(이 dict) 로 만든다.
        arena: 지도에 적을 벽에서 벽까지의 크기 (가로, 세로) m. None 이면 경계를 맞출 때 쓴 크기 그대로.
        """
        from .drone_field_map import field_map

        return field_map(self, boundary, stamp, base_naming, base_kinds, self.box_size, arena=arena)

    def to_dict(self):
        return {"ok": self.ok, "boxes": [b.to_dict() for b in self.boxes], "camera": [round(v, 3) for v in self.camera],
                "scale": round(self.scale, 4), "height_source": self.height_source, "faces_used": self.faces_used,
                "divisor": [round(float(v), 4) for v in self.divisor], "divisor_source": list(self.divisor_source),
                "rejected": self.rejected, "notes": list(self.notes)}


# ------------------------------------------------------------------ 바닥 지도
class _Raster:
    """화면을 위에서 본 바닥 지도로 편 것과 그 위의 기울기 · 무늬 지도. 좌표 변환 도우미."""

    def __init__(self, bgr, b, p):
        import cv2

        self.b, self.p = b, p
        self.W, self.H = b.field_w, b.field_h
        self.ppm, self.m = p.ppm, p.margin
        w, h = int(round((self.W + 2 * self.m) * self.ppm)), int(round((self.H + 2 * self.m) * self.ppm))
        gx, gy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        X, Y = gx / self.ppm - self.m, self.H + self.m - gy / self.ppm
        pix = b.field_to_pixel(np.stack([X.ravel(), Y.ravel()], 1)).reshape(h, w, 2)
        self.valid = np.isfinite(pix).all(2)
        pix = np.nan_to_num(pix, nan=-1e4).astype(np.float32)
        self.img = cv2.remap(bgr, pix[..., 0], pix[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=_FILL)
        hh, ww = bgr.shape[:2]
        self.valid &= (pix[..., 0] >= 0) & (pix[..., 0] <= ww - 1) & (pix[..., 1] >= 0) & (pix[..., 1] <= hh - 1)
        self.X, self.Y = X, Y
        f = cv2.GaussianBlur(self.img, (3, 3), 0).astype(np.float32)
        sx, sy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
        k = np.argmax(sx * sx + sy * sy, axis=2)[..., None]
        self.gx = np.take_along_axis(sx, k, 2)[..., 0]  # +x 방향 밝기 변화 (색 채널 중 가장 센 것)
        self.gy = -np.take_along_axis(sy, k, 2)[..., 0]  # +y(위) 방향
        self.core = cv2.erode(self.valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # 화면 테두리 자체는 모서리가 아니다
        self.gx[~self.core] = 0
        self.gy[~self.core] = 0
        self.tex = cv2.blur(np.hypot(self.gx, self.gy), (3, 3))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        sat = cv2.blur(hsv[..., 1].astype(np.float32), (3, 3))
        self.nonfloor = (self.tex > 40) | (sat > 45)  # 바닥: 무늬 없고 채도 낮음 (밝기는 보지 않는다 — 그림자 · 반사)
        y0, y1 = p.yellow_h
        self.yellow = (hsv[..., 0] >= y0) & (hsv[..., 0] <= y1) & (hsv[..., 1] >= p.yellow_s) & (hsv[..., 2] >= p.yellow_v)
        self.lines = self._painted_lines()
        self.known = self.valid & ~self.lines
        self.blind = self._blind_rows()

    def _painted_lines(self):
        """경계 모델이 아는 바닥 선 (세로 구역선 5 · 가로 줄 경계 2 · 중앙 원) 둘레."""
        b, d = self.b, self.p.line_mask_m
        X, Y, W, H = self.X, self.Y, self.W, self.H
        out = np.zeros(X.shape, bool)
        if b.strip is None:
            return out
        for x0 in (b.strip, W / 2 - b.cm_half, W / 2, W / 2 + b.cm_half, W - b.strip):
            out |= (np.abs(X - x0) < d) & (Y > -0.05) & (Y < H + 0.05)
        for y0 in (H / 3, 2 * H / 3):
            out |= (np.abs(Y - y0) < d) & (X > b.strip - d) & (X < W - b.strip + d)
        out |= np.abs(np.hypot(X - W / 2, Y - H / 2) - b.circle_r) < d
        return out

    def _blind_rows(self):
        """가운데 구역(CM) 안에서 가로 줄 경계의 높이(y = H/3 · 2H/3)에 걸친 띠. 여기는 선이 칠해져 있지 않은데도 _painted_lines 가 지운다.

        줄 경계는 양쪽 구역(끝 띠 선 ~ CM 선)에만 칠해져 있고, CM 은 위아래로 나뉘지 않는다 (연습 경기장 1 · 2번 화면, 포털의 경기장 모형).
        그래서 이 띠 위에 놓인 윗면 변은 「모서리가 없다」 가 아니라 「안 보인다」 로 쳐야 한다 — 화면 밖에 걸린 변과 같게 (_square_candidates).
        그러지 않으면 윗면의 한 변이 이 높이에 놓인 CM 안 상자를 통째로 놓친다 (2026-11-19 2경기장 12:01 화면의 가운데 선 위 상자).
        """
        b, d = self.b, self.p.line_mask_m
        out = np.zeros(self.X.shape, bool)
        if b.strip is None or self.p.row_lines_cross_cm:
            return out
        inside_cm = (self.X > self.W / 2 - b.cm_half + d) & (self.X < self.W / 2 + b.cm_half - d)
        for y0 in (self.H / 3, 2 * self.H / 3):
            out |= (np.abs(self.Y - y0) < d) & inside_cm
        return out

    # 좌표
    def col(self, x):
        return (np.asarray(x, float) + self.m) * self.ppm - 0.5

    def row(self, y):
        return (self.H + self.m - np.asarray(y, float)) * self.ppm - 0.5

    def sample(self, a, x, y):
        import cv2

        c = self.col(x).astype(np.float32)
        r = self.row(y).astype(np.float32)
        return cv2.remap(a, c.reshape(1, -1), r.reshape(1, -1), cv2.INTER_LINEAR, borderValue=0).reshape(c.shape)

    def region(self, a, x0, x1, y0, y1):
        c0, c1 = int(round(float(self.col(x0)))), int(round(float(self.col(x1))))
        r0, r1 = int(round(float(self.row(y1)))), int(round(float(self.row(y0))))
        h, w = a.shape[:2]
        return a[max(r0, 0):max(min(r1, h), 0), max(c0, 0):max(min(c1, w), 0)]


# ------------------------------------------------------------------ 1) 윗면 정사각형
def _shift(a, dr, dc):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(-dr, 0):h + min(-dr, 0), max(-dc, 0):w + min(-dc, 0)] = a[max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)]
    return out


_EDGE_SIDE = ("W", "E", "S", "N")  # _refine_top 의 (x0, x1, y0, y1) 순서


def _edge_maps(R, p):
    """(세로 모서리, 가로 모서리, 화면 안, 무늬) — 0/1 float32. 칠한 선 위의 모서리는 뺐고, ±2 px 어긋남을 받게 넓혔다. 한 번만 만들어 둔다."""
    import cv2

    if getattr(R, "_edges", None) is None:
        ev = ((np.abs(R.gx) > p.grad_min) & ~R.lines).astype(np.float32)
        eh = ((np.abs(R.gy) > p.grad_min) & ~R.lines).astype(np.float32)
        ev = cv2.dilate(ev, np.ones((1, 5), np.uint8))  # ±2 px 어긋남 허용
        eh = cv2.dilate(eh, np.ones((5, 1), np.uint8))
        core = (R.core & ~R.blind).astype(np.float32)  # 「볼 수 있는 곳」 — 화면 안이고, CM 안의 선 없는 가로 띠가 아닌 곳
        tex = (((np.abs(R.gx) > p.grad_min) | (np.abs(R.gy) > p.grad_min)) & R.valid).astype(np.float32)
        R._edges = (ev, eh, core, tex)
    return R._edges


def _wide_lines(R, p):
    """칠한 선 둘레를 joined_line_margin 만큼 더 넓힌 것. 선을 지운 띠의 가장자리에는 선의 모서리가 조금 남는다 (선 굵기 · 경계 모델의 1 ~ 2 cm 오차).

    정사각형 찾기는 네 변이 모두 그런 자리일 때만 속지만, 붙은 줄 찾기 · 글자 띠 확인은 한 변만 그래도 속는다 → 그 둘은 이것을 쓴다.
    """
    import cv2

    if getattr(R, "_wide", None) is None:
        n = 2 * int(round(p.joined_line_margin * p.ppm)) + 1
        R._wide = cv2.dilate(R.lines.astype(np.uint8), np.ones((n, n), np.uint8)) > 0
    return R._wide


def _joined_candidates(R, p, size_x, size_y):
    """붙은 상자 줄(2 · 3개)을 통째로 찾는다 → [(점수, (x0, x1, y0, y1), 상자 수, 줄 방향 "x" · "y")] 점수 높은 순.

    붙은 두 윗면은 둘 다 검은 판이라 사이 모서리가 화면에 안 보일 수 있다 (2026-11-19 2경기장: 가운데 선 위에 세로로 붙은 쌍).
    또 그 모서리가 칠한 선 위에 놓이면 선을 지울 때 함께 지워진다 (같은 날: 가운데 선을 사이에 두고 가로로 붙은 쌍).
    어느 쪽이든 정사각형 찾기(_square_candidates)는 「네 변 중 가장 약한 변」 이 낮아 두 상자를 모두 놓친다.
    → 줄 전체(한 변 d, 긴 변 k·d 인 직사각형)의 바깥 네 변만 보고 찾는다. 사이 모서리는 보지 않는다.
    변의 일부가 칠한 선에 가려도 깎이지 않게, 모서리 비율은 「선 · 화면 밖이 아닌 부분」 에 대한 비율로 센다.
    윗면 크기는 이 화면에서 따로 찾은 윗면들의 중앙값 둘레에서만 찾는다 (크기를 모르면 옆면 + 윗면도 직사각형으로 보인다).
    계산을 줄이려고 바닥 지도를 반으로 줄여서(2 cm = 1 px) 찾는다 — 자리는 2 cm 까지만 맞으므로 부르는 쪽이 변을 다시 다듬는다.
    """
    import cv2

    _ev, _eh, _core, tex = _edge_maps(R, p)
    lines = _wide_lines(R, p)
    ev = cv2.dilate(((np.abs(R.gx) > p.grad_min) & ~lines).astype(np.float32), np.ones((1, 5), np.uint8))  # _edge_maps 와 같되 선 둘레를 더 넓게 뺀다
    eh = cv2.dilate(((np.abs(R.gy) > p.grad_min) & ~lines).astype(np.float32), np.ones((5, 1), np.uint8))
    rows, cols = ev.shape[0] // 2, ev.shape[1] // 2

    def half(a):  # 2 × 2 칸 평균
        return cv2.resize(np.ascontiguousarray(a[:rows * 2, :cols * 2], np.float32), (cols, rows), interpolation=cv2.INTER_AREA)

    ev, eh, tex = half(ev), half(eh), half(tex)
    free = half(R.core & ~lines)
    d0 = 0.5 * (size_x + size_y) * p.ppm
    sizes = [d for d in range(4 * (p.top_px[0] // 4), p.top_px[1], 4) if abs(d - d0) <= p.joined_size_tol * d0 + 2]  # 원래 지도의 px, 4 의 배수
    if not sizes:
        return []
    inside = half((R.X >= 0) & (R.X <= R.W) & (R.Y >= 0) & (R.Y <= R.H)) > 0.99
    pad = (max(p.joined_counts) * max(sizes)) // 4 + 2  # 가장 멀리 옮겨 보는 px (줄인 지도)
    memo = {}

    def blur(name, a, ksize):
        """평균을 낸 뒤 둘레를 0 으로 넓혀 둔다 — 같은 평균을 줄 수 · 방향마다 다시 내지 않고, 옮겨 보기는 복사 없이 잘라 쓴다."""
        key = (name, ksize)
        if key not in memo:
            memo[key] = cv2.copyMakeBorder(cv2.blur(a, ksize), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        return memo[key]

    def at(a, dr, dc):  # _shift(a, dr, dc) 와 같은 값 (밖은 0)
        return a[pad + dr:pad + dr + rows, pad + dc:pad + dc + cols]

    out = []
    for k in p.joined_counts:
        for axis in ("x", "y"):  # 줄이 뻗은 방향
            best = np.full((rows, cols), -1.0, np.float32)
            size = np.zeros((rows, cols), np.int32)
            for d in sizes:
                e = d // 2  # 줄인 지도에서 윗면 한 변 (px, 짝수)
                long_px = k * e
                n_s, n_l = int(e * p.edge_frac) | 1, int(long_px * p.edge_frac) | 1
                hs, hl = e // 2, long_px // 2
                inner = blur("tex", tex, (int(e * 0.6) | 1,) * 2)
                offs = [int(round((i - (k - 1) / 2.0) * e)) for i in range(k)]  # 상자마다 가운데가 줄 가운데에서 떨어진 px
                if axis == "y":  # 세로로 붙은 줄: 긴 변이 세로(W · E), 짧은 변이 가로(S · N)
                    sv, fv = blur("ev", ev, (1, n_l)), blur("free", free, (1, n_l))
                    sh, fh = blur("eh", eh, (n_s, 1)), blur("free", free, (n_s, 1))
                    moves = [(0, -hs), (0, hs), (hl, 0), (-hl, 0)]  # W E S N
                    busy = [at(inner, o, 0) for o in offs]
                else:
                    sv, fv = blur("ev", ev, (1, n_s)), blur("free", free, (1, n_s))
                    sh, fh = blur("eh", eh, (n_l, 1)), blur("free", free, (n_l, 1))
                    moves = [(0, -hl), (0, hl), (hs, 0), (-hs, 0)]
                    busy = [at(inner, 0, o) for o in offs]
                edges, known = None, None
                for i, (dr, dc) in enumerate(moves):
                    seen = at(fv if i < 2 else fh, dr, dc)
                    cover = at(sv if i < 2 else sh, dr, dc) / np.maximum(seen, 1e-3)
                    edges = cover if edges is None else np.minimum(edges, cover, out=edges)
                    ok = seen >= p.joined_min_seen
                    known = ok if known is None else known & ok
                worst = busy[0]
                for b_ in busy[1:]:
                    worst = np.maximum(worst, b_)
                sc = np.where(known, np.minimum(edges, 1.0), -1.0) - p.tex_weight * np.maximum(worst - 0.08, 0)
                upd = sc > best
                best[upd], size[upd] = sc[upd], d
            best[~inside] = -1
            tie = best + 1e-6 * ((np.arange(best.size).reshape(best.shape) * 7919) % 1000) / 1000.0
            near = max(p.nms_px // 2, 3) | 1
            peak = (tie >= cv2.dilate(tie, np.ones((near, near), np.uint8))) & (best >= p.joined_min_score)
            for r, c in np.argwhere(peak):
                d = size[r, c] / p.ppm
                x, y = (2 * c + 1.0) / p.ppm - R.m, R.H + R.m - (2 * r + 1.0) / p.ppm  # 줄인 칸의 가운데
                wx, wy = (k * d, d) if axis == "x" else (d, k * d)
                out.append((float(best[r, c]), (x - wx / 2, x + wx / 2, y - wy / 2, y + wy / 2), k, axis))
    out.sort(key=lambda t: -t[0])
    return out


def _square_candidates(R, p):
    """네 변이 곧은 모서리인 정사각형 → [(점수, x, y, 한 변 m, 잘린 변)]. 칠한 선 위의 모서리는 세지 않는다.

    한 변이 화면 밖(또는 화면 끝)에 걸리면 그 변은 모르는 것으로 두고 나머지 세 변으로 본다 (잘린 변 = W · E · S · N).
    네 변이 다 보이는 것보다 0.1 낮게 친다.
    """
    import cv2

    ev, eh, core, tex = _edge_maps(R, p)
    best = np.full(ev.shape, -1.0, np.float32)
    size = np.zeros(ev.shape, np.int32)
    cut = np.full(ev.shape, -1, np.int8)
    for d in range(*p.top_px):
        n = int(d * p.edge_frac) | 1
        h = d // 2
        sv, sh = cv2.blur(ev, (1, n)), cv2.blur(eh, (n, 1))
        cv_, ch = cv2.blur(core, (1, n)), cv2.blur(core, (n, 1))
        cover = np.stack([_shift(sv, 0, -h), _shift(sv, 0, h), _shift(sh, h, 0), _shift(sh, -h, 0)])  # W E S N
        seen = np.stack([_shift(cv_, 0, -h), _shift(cv_, 0, h), _shift(ch, h, 0), _shift(ch, -h, 0)])
        known = seen >= 0.5
        cover = np.where(known, cover / np.maximum(seen, 1e-3), 2.0)
        n_known = known.sum(0)
        edges = np.where(n_known >= 3, cover.min(0), -1.0) - 0.1 * (n_known == 3)
        inner = cv2.blur(tex, (int(d * 0.6) | 1,) * 2)
        sc = edges - p.tex_weight * np.maximum(inner - 0.08, 0)
        upd = sc > best
        best[upd], size[upd] = sc[upd], d
        cut[upd] = np.where(n_known == 3, np.argmin(known, axis=0), -1)[upd]
    inside = (R.X >= 0) & (R.X <= R.W) & (R.Y >= 0) & (R.Y <= R.H)
    best[~inside] = -1
    tie = best + 1e-6 * ((np.arange(best.size).reshape(best.shape) * 7919) % 1000) / 1000.0  # 평평한 봉우리에서 하나만
    peak = (tie >= cv2.dilate(tie, np.ones((p.nms_px, p.nms_px), np.uint8))) & (best >= p.min_score * 0.6)
    out = [(float(best[r, c]), float(R.X[r, c]), float(R.Y[r, c]), size[r, c] / p.ppm, _EDGE_SIDE[cut[r, c]] if cut[r, c] >= 0 else None)
           for r, c in np.argwhere(peak)]
    out.sort(key=lambda t: -t[0])
    return out


def _refine_top(R, x, y, d, p, skip=None):
    """정사각형 네 변을 ±refine_m 안에서 모서리가 가장 센 곳으로 옮긴다 → ((x0, x1, y0, y1), 변마다 세기). skip 변은 그대로 둔다."""
    return _refine_rect(R, (x - d / 2, x + d / 2, y - d / 2, y + d / 2), p, skip)


def _refine_rect(R, rect, p, skip=None, reach=None, outer=False):
    """직사각형 (x0, x1, y0, y1) 의 네 변을 ±reach(기본 refine_m) 안에서 모서리가 가장 센 곳으로 옮긴다 → (직사각형, 변마다 세기).

    outer: 가장 센 곳의 절반 넘게 센 봉우리가 여럿이면 가장 바깥 것을 고른다. 윗면은 변에서 3 cm 쯤 안쪽에 흰 글자가 찍혀 있어
    그 글자 줄이 진짜 변(검은 판과 바닥 사이)보다 세게 나올 때가 많다 — 넓게 찾을 때는 글자 줄로 끌려가지 않게 한다.
    """
    e = list(rect)
    dx, dy = e[1] - e[0], e[3] - e[2]
    reach = p.refine_m if reach is None else reach
    offs = np.arange(-reach, reach + 1e-9, 0.0025)
    out, strength = [], []
    for i in range(4):
        if _EDGE_SIDE[i] == skip:
            out.append(e[i])
            strength.append(0.0)
            continue
        if i < 2:
            along = np.linspace(e[2] + 0.15 * dy, e[3] - 0.15 * dy, 30)
            sup = [np.abs(R.sample(R.gx, np.full_like(along, e[i] + o), along)).mean() for o in offs]
        else:
            along = np.linspace(e[0] + 0.15 * dx, e[1] - 0.15 * dx, 30)
            sup = [np.abs(R.sample(R.gy, along, np.full_like(along, e[i] + o))).mean() for o in offs]
        k = int(np.argmax(sup))
        if outer:
            sup = np.asarray(sup)
            order = range(len(offs) - 1, -1, -1) if i % 2 else range(len(offs))  # E · N 은 + 쪽이 바깥, W · S 는 − 쪽이 바깥
            for j in order:
                if sup[j] >= 0.5 * sup[k] and sup[j] >= sup[max(j - 1, 0)] and sup[j] >= sup[min(j + 1, len(offs) - 1)]:
                    k = j
                    break
        out.append(e[i] + offs[k])
        strength.append(float(sup[k]))
    return tuple(out), tuple(strength)


def _even_top(top, strength, sx, sy, tol, N):
    """윗면 크기가 중앙값(sx · sy)과 tol 넘게 다르면 한 변을 남기고 다른 변을 중앙값만큼 떨어진 자리로 옮긴다.

    상자 윗면은 모두 같은 크기다 (지도 축척 오차로 몇 % 다를 뿐). 틀리는 쪽은 대개 카메라 쪽 변이다 — 그 변 바깥에 옆면 그림과
    옆면 바닥 모서리가 있어 엉뚱한 모서리를 잡는다. 카메라에서 먼 변은 바닥이나 붙은 상자와 맞닿아 깨끗하다.
    → 카메라에서 먼 변을 남긴다. 상자가 카메라 바로 아래에 걸쳐 있으면 (양쪽 다 옆면이 거의 없다) 모서리가 센 변을 남긴다.
    """
    x0, x1, y0, y1 = top

    def keep_low(lo, hi, n, s_lo, s_hi):
        if hi < n:  # 카메라가 큰 쪽 → 작은 쪽 변이 멀다
            return True
        if lo > n:
            return False
        return s_lo >= s_hi

    if abs((x1 - x0) - sx) > tol:
        x0, x1 = (x0, x0 + sx) if keep_low(x0, x1, N[0], strength[0], strength[1]) else (x1 - sx, x1)
    if abs((y1 - y0) - sy) > tol:
        y0, y1 = (y0, y0 + sy) if keep_low(y0, y1, N[1], strength[2], strength[3]) else (y1 - sy, y1)
    return x0, x1, y0, y1


def _fit_cut_top(R, x, y, sx, sy, cut, reach=0.07, step=0.005):
    """한 변이 화면 밖인 윗면: 크기를 다른 윗면들의 중앙값(sx × sy)으로 고정하고, 보이는 세 변의 모서리가 가장 센 자리로 옮긴다.

    세 변만으로는 크기가 정해지지 않아 정사각형 찾기가 엉뚱한 크기를 고를 수 있다 (붙은 상자 사이 · 옆면 그림 모서리).
    """
    best = (-1.0, None)
    t = np.linspace(0.15, 0.85, 24)
    for cx in np.arange(x - reach, x + reach + 1e-9, step):
        for cy in np.arange(y - reach, y + reach + 1e-9, step):
            x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
            sup = []
            for side in ("W", "E", "S", "N"):
                if side == cut:
                    continue
                if side in "WE":
                    along = y0 + (y1 - y0) * t
                    v = R.sample(R.gx, np.full_like(along, x0 if side == "W" else x1), along)
                else:
                    along = x0 + (x1 - x0) * t
                    v = R.sample(R.gy, along, np.full_like(along, y0 if side == "S" else y1))
                sup.append(float(np.minimum(np.abs(v) / 60.0, 1.0).mean()))
            sc = min(sup)
            if sc > best[0]:
                best = (sc, (x0, x1, y0, y1))
    return best[1]


def _yellow_edges(R, top, band=(0.005, 0.03), frac=0.2):
    """윗면 네 변의 안쪽 띠 중 노란 픽셀이 frac 넘는 변의 수 — 거점 테두리 안에 생긴 사각형은 두 변 이상이 노랗다."""
    x0, x1, y0, y1 = top
    a, b = band
    bands = [(x0 + a, x0 + b, y0, y1), (x1 - b, x1 - a, y0, y1), (x0, x1, y0 + a, y0 + b), (x0, x1, y1 - b, y1 - a)]
    n = 0
    for bb in bands:
        r = R.region(R.yellow, *bb)
        n += int(r.size > 0 and r.mean() > frac)
    return n


# ------------------------------------------------------------------ 2) 옆면 · 시차
def _face(top, side, N):
    """면 하나의 (축 0/1, 윗면 모서리 t, 바깥 방향 ±1, 면을 따라가는 범위 a0~a1). 카메라 쪽이 아니면 None."""
    x0, x1, y0, y1 = top
    ax = 0 if side in "EW" else 1
    t = {"E": x1, "W": x0, "N": y1, "S": y0}[side]
    d = 1 if side in "EN" else -1
    if d * (N[ax] - t) <= 0:
        return None
    a0, a1 = (y0, y1) if ax == 0 else (x0, x1)
    return ax, t, d, a0, a1


def _face_toward_neighbor(top, side, N, others, c=0.2):
    """카메라 쪽 옆면 앞(c = 0.2 까지)에 다른 윗면이 있으면 True — 붙은 상자 사이 면은 안 보인다."""
    f = _face(top, side, N)
    if f is None:
        return False
    ax, t, d, a0, a1 = f
    lo, hi = sorted((t, t + (N[ax] - t) * c))
    for o in others:
        if ax == 0 and o[0] < hi and o[1] > lo and o[2] < a1 and o[3] > a0:
            return True
        if ax == 1 and o[2] < hi and o[3] > lo and o[0] < a1 and o[1] > a0:
            return True
    return False


def _face_scan(R, top, side, N, cs):
    """카메라 쪽 옆면의 바닥 모서리를 c = 1 − 1/s 마다 맞춰 본다 → (가장 맞는 c, 점수, 옆면 폭 m). 안 보이는 면은 None.

    점수 = 예상 바닥 모서리를 따라 모서리 세기 × 그 바깥 3 cm 가 바닥처럼 매끈한 정도.
    옆면 그림 속 가로줄도 모서리처럼 보이지만 그 바깥이 그림이라 점수가 낮다.
    """
    f = _face(top, side, N)
    if f is None:
        return None
    ax, t, d, a0, a1 = f
    Na, Np = N[ax], N[1 - ax]
    g = R.gx if ax == 0 else R.gy
    tex = np.where(R.lines, 0.0, R.tex).astype(np.float32)
    best = (-1.0, 0.0)
    for c in cs:
        fpos = t + (Na - t) * c
        p0, p1 = Np + (a0 - Np) * (1 - c), Np + (a1 - Np) * (1 - c)
        u = np.linspace(p0 + 0.1 * (p1 - p0), p1 - 0.1 * (p1 - p0), 20)
        pos = np.full_like(u, fpos)
        xy = (pos, u) if ax == 0 else (u, pos)
        out = (pos + d * 0.03, u) if ax == 0 else (u, pos + d * 0.03)
        if R.sample(R.valid.astype(np.float32), *out).min() < 0.99:
            continue
        sc = float(np.minimum(np.abs(R.sample(g, *xy)) / 60.0, 1.0).mean() * math.exp(-float(np.median(R.sample(tex, *out))) / 25.0))
        if sc > best[0]:
            best = (sc, c, u)
    if best[0] < 0:
        return 0.0, -1.0, 0.0
    # 바깥 매끈함 항이 모서리를 바깥으로 조금 끌어낸다 → 고른 자리 ±1.5 cm 에서 모서리가 가장 센 곳으로 다시 짚는다
    sc, c, u = best
    f0 = t + (Na - t) * c
    offs = np.arange(-0.015, 0.0151, 0.0025)
    sup = [np.abs(R.sample(g, *((np.full_like(u, f0 + o), u) if ax == 0 else (u, np.full_like(u, f0 + o))))).mean() for o in offs]
    c = (f0 + offs[int(np.argmax(sup))] - t) / (Na - t)
    return c, sc, abs(Na - t) * c


def _face_nonfloor(R, top, side, N, c):
    """카메라 쪽 옆면이 보여야 할 자리에서 바닥이 아닌 픽셀 비율 (칠한 선 · 화면 밖은 빼고). 확인할 수 없으면 None."""
    f = _face(top, side, N)
    if f is None:
        return None
    ax, t, d, a0, a1 = f
    w = abs(N[ax] - t) * c
    if w < R.p.check_width:
        return None
    lo, hi = sorted((t + d * 0.015, t + d * (w - 0.01)))
    a0, a1 = a0 + 0.1, a1 - 0.1
    box = (lo, hi, a0, a1) if ax == 0 else (a0, a1, lo, hi)
    nf, kn = R.region(R.nonfloor, *box), R.region(R.known, *box)
    if kn.sum() < 20:
        return None
    return float(nf[kn].mean())


# ------------------------------------------------------------------ 본체
def detect_boxes(bgr, boundary, params=None):
    """드론 화면(BGR) + drone_boundary.detect_boundary 결과 → BoxView. 경계를 못 찾았으면 ok=False."""
    p = params or BoxParams()
    view = BoxView(False, box_size=p.box_size)
    if boundary is None or boundary.hom is None:
        view.notes.append("경계가 없어 상자를 찾지 않음")
        return view
    sane = boundary.ok
    if not boundary.ok:
        view.notes.append("경계 검사가 실패한 화면 — 상자 좌표를 믿기 어렵다")
    elif boundary.strip is not None and not (p.strip_range[0] <= boundary.strip <= p.strip_range[1] and p.cm_range[0] <= boundary.cm_half <= p.cm_range[1]):
        sane = False
        view.notes.append("경계의 끝 띠 %.2f m · CM 반폭 %.2f m 가 범위 밖 — 구역선을 잘못 골랐을 수 있다" % (boundary.strip, boundary.cm_half))
    R = _Raster(bgr, boundary, p)
    W, H, h = R.W, R.H, p.box_size
    N = tuple(p.camera_xy) if p.camera_xy is not None else (W / 2.0, H / 2.0)

    # 1) 윗면 후보. 잘린 윗면은 다 보이는 윗면들의 크기(중앙값)로 잘린 변을 채운다
    cands = [c for c in _square_candidates(R, p) if c[0] >= p.min_score]
    refined = [_refine_top(R, x, y, d, p, skip=cut) for _s, x, y, d, cut in cands]
    tops = [t for t, _st in refined]

    # 윗면만 보고 거를 수 있는 것 (s 를 몰라도 된다)
    def colour_check(top):
        # 연습 화면 상자 28개는 이 값이 모두 0.000, 거점 · 줄무늬 블록 자리는 0.03 ~ 0.13 (2026-11-18)
        if R.region(R.yellow, top[0] + 0.02, top[1] - 0.02, top[2] + 0.02, top[3] - 0.02).mean() > p.max_yellow:
            return "노란색 (줄무늬 블록 등)"
        if _yellow_edges(R, top) >= 2:
            return "노란 테두리 (거점)"
        return None

    def first_check(top, cut):
        if not _plausible_top(top, p):
            return "윗면 크기 %.2f × %.2f m" % (top[1] - top[0], top[3] - top[2])
        if cut is None and R.region(R.valid, top[0] - 0.01, top[1] + 0.01, top[2] - 0.01, top[3] + 0.01).mean() < 0.98:
            return "윗면이 화면 밖으로 잘림"
        return None

    why0 = [first_check(t, c[4]) for t, c in zip(tops, cands)]
    # 상자 윗면은 모두 크기가 같다. 바닥 지도 축척이 화면마다 달라(test1 은 ±10 %) 크기 한도는 이 화면 윗면들의 중앙값 기준으로.
    # 중앙값은 노란 것(거점 · 줄무늬 블록)을 뺀 다 보이는 윗면으로
    whole = [t for t, c, w in zip(tops, cands, why0) if c[4] is None and w is None and colour_check(t) is None]
    size_x = float(np.median([t[1] - t[0] for t in whole])) if whole else 0.4 * 1.18
    size_y = float(np.median([t[3] - t[2] for t in whole])) if whole else 0.4 * 1.18
    for i, (_s, x, y, _d, cut) in enumerate(cands):
        if why0[i] is not None:
            continue
        if cut is not None:
            tops[i] = _fit_cut_top(R, x, y, size_x, size_y, cut)
        elif len(whole) >= 3:
            tops[i] = _even_top(tops[i], refined[i][1], size_x, size_y, p.size_tol, N)
        sx, sy = tops[i][1] - tops[i][0], tops[i][3] - tops[i][2]
        if len(whole) >= 3 and (abs(sx / size_x - 1) > p.top_rel_tol or abs(sy / size_y - 1) > p.top_rel_tol):
            why0[i] = "윗면 크기 %.2f × %.2f m (이 화면 상자들 %.2f × %.2f)" % (sx, sy, size_x, size_y)
        else:
            why0[i] = colour_check(tops[i])
    # 1-b) 붙은 상자 줄: 윗면 사이 모서리가 안 보여 정사각형으로는 못 찾은 것. 줄 전체의 바깥 변으로 찾아 같은 크기로 나눈다
    joined_from = len(cands)
    row_of = {}
    if p.joined and len(whole) >= 3:
        taken = [tops[i] for i, w in enumerate(why0) if w is None]
        tried = []
        for score, rect, k, axis in _joined_candidates(R, p, size_x, size_y):
            why = None
            if any(_top_cover(t, o) > p.joined_max_cover for t in _split_rect(rect, k, axis) for o in taken):
                why = "이미 찾은 윗면과 겹침"  # 대부분 여기서 끝난다 (사이 모서리가 보이는 줄은 상자마다 따로 찾았다) — 변 다듬기 전에 거른다
            else:
                rect, _strength = _refine_rect(R, rect, p, reach=p.joined_refine_m, outer=True)
            parts = _split_rect(rect, k, axis)
            for t in parts:
                if why is not None:
                    break
                sx, sy = t[1] - t[0], t[3] - t[2]
                if abs(sx / size_x - 1) > p.top_rel_tol or abs(sy / size_y - 1) > p.top_rel_tol:
                    why = "윗면 크기 %.2f × %.2f m (이 화면 상자들 %.2f × %.2f)" % (sx, sy, size_x, size_y)
                why = why or first_check(t, None) or colour_check(t)
                if why is None and any(_top_cover(t, o) > p.joined_max_cover for o in taken):
                    why = "이미 찾은 윗면과 겹침"
                if why is None and _frame_text(R, p, t) < p.joined_min_frame:
                    why = "윗면 테두리 글자가 안 보임 (상자 사이의 빈 바닥일 수 있다)"
            tried.append({"rect": [round(float(v), 3) for v in rect], "n": k, "axis": axis, "score": round(score, 3), "reason": why})
            if why is not None:
                continue
            for t in parts:
                row_of[len(cands)] = len(tried)  # 후보 번호 → 줄 번호
                cands.append((score - 0.05, (t[0] + t[1]) / 2.0, (t[2] + t[3]) / 2.0, t[1] - t[0], None))  # 따로 찾은 윗면보다 조금 낮게 친다
                tops.append(t)
                why0.append(None)
            taken += parts
        view.debug["joined"] = tried
        if len(cands) > joined_from:
            view.notes.append("붙은 상자 줄에서 상자 %d개를 더 찾음 (윗면 사이 모서리가 안 보임)" % (len(cands) - joined_from))
    view.debug["candidates"] = len(cands)
    view.debug["top_size"] = (round(size_x, 3), round(size_y, 3))
    good = [i for i, w in enumerate(why0) if w is None]  # 이것만 s 를 재고 서로 이웃으로 친다

    def others(i):
        return [tops[j] for j in good if j != i]

    # 2) 시차 s
    c_lo, c_hi = h / p.height_range[1], h / p.height_range[0]
    if p.camera_height is not None:
        s, source, used = p.camera_height / (p.camera_height - h), "given", 0
    else:
        cs = np.arange(c_lo, c_hi + 1e-9, 0.0025)
        meas = []
        for i in good:
            if i >= joined_from:  # 줄에서 나눈 윗면은 s 를 재는 데 쓰지 않는다 (따로 찾은 윗면만 — 이웃으로는 친다)
                continue
            for side in SIDES:
                if _face_toward_neighbor(tops[i], side, N, others(i)):
                    continue
                f = _face(tops[i], side, N)
                if f is None or abs(N[f[0]] - f[1]) < p.face_min_lever:
                    continue
                r = _face_scan(R, tops[i], side, N, cs)
                if r[1] >= p.face_min_score and r[2] >= p.face_min_width:
                    meas.append(r[0])
        used = len(meas)
        if used >= p.min_faces:
            c = float(np.median(meas))
            s, source = 1.0 / (1.0 - c), "measured"
            view.debug["face_c"] = [round(v, 4) for v in sorted(meas)]
        else:
            s, source = p.default_height / (p.default_height - h), "default"
            view.notes.append("옆면을 %d개밖에 못 재서 카메라 높이를 기본값 %.2f m 로 둠" % (used, p.default_height))
    hc = h * s / (s - 1.0)
    view.camera, view.scale, view.height_source, view.faces_used = (N[0], N[1], hc), s, source, used
    c = 1.0 - 1.0 / s
    # 윗면 가운데 → 상자 가운데의 나눗값. 상자가 바닥 격자에 맞춰 놓여 있으면 축마다 격자에 맞는 값을 쓰고, 아니면 s 그대로
    div, div_source = (s, s), ("s", "s")
    if p.grid_cell and p.camera_height is None:
        div, div_source, grid_info = _grid_divisors([tops[i] for i in good if cands[i][4] is None], N, (W / 2.0, H / 2.0), s, p)
        view.debug["grid"] = grid_info
        if "grid" in div_source:
            view.notes.append("상자 자리를 바닥 격자(%.0f cm)에 맞춰 냄: 나눗값 x %.3f (%s) · y %.3f (%s), 옆면으로 잰 s %.3f" % (
                p.grid_cell * 100, div[0], div_source[0], div[1], div_source[1], s))
    view.divisor, view.divisor_source = div, div_source

    # 3) 상자 가운데와 거르기
    keep, rejected, bad_rows = [], [], set()
    for i, ((score, _x, _y, _d, cut), top) in enumerate(zip(cands, tops)):
        T = ((top[0] + top[1]) / 2.0, (top[2] + top[3]) / 2.0)
        P = (N[0] + (T[0] - N[0]) / div[0], N[1] + (T[1] - N[1]) / div[1])
        why = why0[i]
        if why is None and boundary.strip is not None and not (boundary.strip + p.strip_margin <= P[0] <= W - boundary.strip - p.strip_margin):
            why = "끝 띠 안 (거점 · 줄무늬 벽 블록 자리)"
        elif why is None and not (p.wall_margin <= P[1] <= H - p.wall_margin):
            why = "위아래 벽에 붙음"
        check = None
        if why is None:
            fc = [v for side in SIDES if not _face_toward_neighbor(top, side, N, others(i))
                  for v in [_face_nonfloor(R, top, side, N, c)] if v is not None]
            check = float(np.mean(fc)) if fc else None
            if check is not None and check < p.min_face_nonfloor:
                why = "카메라 쪽 옆면 자리가 바닥 (%.0f%%)" % (check * 100)
        if why is None:
            keep.append(Box(P[0], P[1], score, top, check, cut, joined=i >= joined_from))
            keep[-1].row = row_of.get(i)
        else:
            rejected.append({"x": round(P[0], 3), "y": round(P[1], 3), "score": round(score, 3), "reason": why})
            bad_rows.add(row_of.get(i))
    # 붙은 줄은 통째로 받거나 통째로 버린다 — 한 상자라도 걸러진 줄은 상자 줄이 아니라 상자 · 벽 · 선 사이의 빈 바닥이기 쉽다
    for b in [b for b in keep if b.row is not None and b.row in bad_rows]:
        keep.remove(b)
        rejected.append({"x": round(b.x, 3), "y": round(b.y, 3), "score": round(b.score, 3), "reason": "붙은 줄의 다른 상자가 걸러짐"})

    # 겹침: 바닥 면적이 겹치면 점수 높은 쪽
    keep.sort(key=lambda b: -b.score)
    final = []
    for b in keep:
        clash = next((o for o in final if _overlap(b, o, h) > p.max_overlap), None)
        if clash is None:
            final.append(b)
        else:
            rejected.append({"x": round(b.x, 3), "y": round(b.y, 3), "score": round(b.score, 3),
                             "reason": "(%.2f, %.2f) 상자와 겹침" % (clash.x, clash.y)})
    lost_rows = {b.row for b in keep if b.row is not None and not any(b is o for o in final)}
    for b in [b for b in final if b.row is not None and b.row in lost_rows]:  # 겹쳐서 한 상자가 빠진 줄도 통째로 버린다
        final.remove(b)
        rejected.append({"x": round(b.x, 3), "y": round(b.y, 3), "score": round(b.score, 3), "reason": "붙은 줄의 다른 상자가 걸러짐"})
    # 4) 화면 끝에 걸린 상자: 윗면의 먼 변이 화면 밖이라 위에서 못 찾은 것을 카메라 쪽 옆면의 바닥선으로 찾는다
    if p.border:
        tried = []
        for b in _border_boxes(R, p, N, s, final, tried):
            if boundary.strip is not None and not (boundary.strip + p.strip_margin <= b.x <= W - boundary.strip - p.strip_margin):
                rejected.append({"x": round(b.x, 3), "y": round(b.y, 3), "score": round(b.score, 3), "reason": "끝 띠 안 (화면 끝에 걸린 것)"})
            elif not (max(p.wall_margin, p.border_wall_gap) <= b.y <= H - max(p.wall_margin, p.border_wall_gap)):
                rejected.append({"x": round(b.x, 3), "y": round(b.y, 3), "score": round(b.score, 3),
                                 "reason": "벽에서 %.2f m 안 (화면 끝에 걸린 것 — 벽에 붙은 상자이거나 벽 가까이의 UGV)" % max(p.wall_margin, p.border_wall_gap)})
            elif not any(_overlap(b, o, h) > p.max_overlap for o in final):
                final.append(b)
        view.debug["border"] = tried
        found = sum(1 for b in final if b.border)
        if found:
            view.notes.append("화면 끝에 걸린 상자 %d개를 옆면의 바닥선으로 찾음 (윗면의 먼 변이 화면 밖)" % found)
    final.sort(key=lambda b: (-round(b.y / 0.4), b.x))  # 위 줄부터 (0.4 m 줄), 같은 줄은 왼쪽부터
    view.boxes, view.rejected = final, rejected
    view.ok = sane and bool(final) and source != "default"
    view.debug["raster"] = R
    return view


_BORDER_SIDES = {"N": (1, 1), "S": (1, -1), "E": (0, 1), "W": (0, -1)}  # 화면 밖으로 나간 쪽 → (바깥 방향의 축 0 = x · 1 = y, 부호)


def _border_boxes(R, p, N, s, taken, tried=None):
    """화면 끝에 걸려 윗면이 다 보이지 않는 상자 → [Box]. 카메라 쪽 옆면의 바닥선(바닥과 옆면이 만나는 곧은 선)으로 찾는다.

    드론 카메라는 경기장을 다 비추지 않는다 (연습 경기장은 위아래가, 본선 경기장은 좌우도 잘릴 수 있다 — 2026-11-19 주최측 답).
    화면 끝 가까이의 상자는 윗면의 먼 변이 화면 밖이거나, 윗면이 아예 안 보이고 카메라 쪽 옆면의 아래쪽만 보인다
    (2026-11-19 2경기장 11:47 판: 위쪽 벽 가까이의 상자 셋이 옆면 아래 12 ~ 20 cm 만 보였다). 정사각형 찾기는 변이 셋은 보여야 한다.
    옆면의 바닥선은 바닥(높이 0)에 있어 시차가 없다 → 그 선의 가운데에서 바깥으로 상자 반 변만큼 간 곳이 상자 가운데다.

    받는 조건 (모두)
      - 바닥선: 상자 한 변 길이의 곧은 모서리 (가운데 80 % 에서 모서리 비율 border_min_edge 이상), 카메라 바로 아래에서 border_min_lever 넘게 멀다
      - 그 바깥(화면 끝 쪽)이 옆면이다: 화면 끝까지, 또는 옆면이 보일 폭까지, 가운데와 양쪽 가장자리가 모두 바닥이 아니다 (그림의 무늬 · 색).
        양쪽 가장자리를 따로 보는 것은 UGV(25 × 23 cm) 처럼 좁은 것을 거르려는 것. 노란 것(줄무늬 블록 · 거점 테두리)은 버린다
      - 그 안쪽(카메라 쪽)은 바닥이다
      - 윗면의 먼 변이 화면 밖이다 — 다 보이는 상자는 정사각형으로 찾는 쪽이 맡는다
      - 이미 찾은 상자와 겹치지 않는다
    tried: 목록을 주면 살펴본 후보와 버린 까닭을 넣어 준다 (기록 · 그림용).
    """
    import cv2

    ev, eh, _core, _tex = _edge_maps(R, p)
    free = (R.core & ~R.lines).astype(np.float32)
    valid = R.valid
    labels = np.zeros(valid.shape, bool)
    for x0, x1, y0, y1 in p.border_labels:
        labels |= (R.X >= x0) & (R.X <= x1) & (R.Y >= y0) & (R.Y <= y1)
    ev, eh = np.where(labels, 0.0, ev), np.where(labels, 0.0, eh)
    nonfloor = R.nonfloor & ~labels  # 바닥 글자는 바닥이다
    seen_face = R.known & ~labels  # 옆면인지 볼 때는 글자 자리를 빼고 센다
    size, half = p.box_size, p.box_size / 2.0
    n = int(size * p.ppm * p.edge_frac) | 1
    out = []
    # 이미 찾은 상자가 지도에서 차지하는 자리 (바닥 네모 ~ 윗면 네모를 감싼 네모, 3 cm 넉넉히). 그 안의 곧은 선은 바닥선이 아니다
    hulls = [(min(o.x - half, o.top[0]) - 0.03, max(o.x + half, o.top[1]) + 0.03, min(o.y - half, o.top[2]) - 0.03, max(o.y + half, o.top[3]) + 0.03)
             for o in taken]
    for side, (axis, sign) in _BORDER_SIDES.items():
        edge = eh if axis == 1 else ev
        ksize = (n, 1) if axis == 1 else (1, n)
        seen = cv2.blur(free, ksize)
        cover = np.where(seen >= 0.6, cv2.blur(edge, ksize) / np.maximum(seen, 1e-3), 0.0)
        count = np.cumsum(valid, axis=0 if axis == 1 else 1)
        total = count[-1:, :] if axis == 1 else count[:, -1:]
        # 그 픽셀에서 바깥 방향으로 화면 끝까지의 px 수. 지도의 행은 위가 +y, 열은 오른쪽이 +x
        depth = (count - 1) if (side in "NW") else (total - count)
        coord = R.Y if axis == 1 else R.X
        lever = sign * (coord - N[axis])
        mask = (cover >= p.border_min_edge) & valid & (depth >= p.border_min_depth * p.ppm) & (depth <= 0.9 * p.ppm) & (lever >= p.border_min_lever)
        mask &= (R.X >= 0) & (R.X <= R.W) & (R.Y >= 0) & (R.Y <= R.H)
        # 상자 가운데가 놓일 수 없는 자리(끝 띠 안 · 벽 가까이)는 처음부터 뺀다 — 벽 · 거점 · 줄무늬 블록 둘레에 후보가 가장 많다 (계산 시간의 대부분)
        cx, cy = (R.X, R.Y + sign * half) if axis == 1 else (R.X + sign * half, R.Y)
        slack_x, slack_y = (0.25, 0.0) if axis == 1 else (0.0, 0.25)  # 선을 따라가는 쪽은 뒤에서 ±0.24 m 까지 옮겨 본다
        if R.b.strip is not None:
            mask &= (cx >= R.b.strip + p.strip_margin - slack_x) & (cx <= R.W - R.b.strip - p.strip_margin + slack_x)
        gap = max(p.wall_margin, p.border_wall_gap)
        mask &= (cy >= gap - slack_y) & (cy <= R.H - gap + slack_y)
        seen_m = depth / float(p.ppm)  # 바닥선에서 화면 끝까지 (m)
        face_m = lever * (s - 1.0)  # 옆면이 보일 폭 (m)
        mask &= seen_m < face_m + size * s - 0.03  # 윗면의 먼 변이 화면 밖인 것만 — 다 보이는 상자는 정사각형으로 찾는 쪽이 맡는다
        mask &= np.minimum(face_m, seen_m) - 0.01 >= p.border_min_depth  # 옆면이 너무 조금 보이는 것은 뺀다
        # 모서리가 뚜렷한 것부터, 같으면 화면 끝에서 먼 것(카메라 쪽)부터 — 옆면 그림 속의 가로줄보다 진짜 바닥선이 먼저 오게
        cand = [(-round(float(cover[r, c]), 2), -int(depth[r, c]), int(r), int(c)) for r, c in np.argwhere(mask)]
        cand.sort()
        cand = [(-neg, r, c) for neg, _d, r, c in cand]
        picked, dropped, looked = [], [], 0  # 받은 자리 · 버린 자리 (선을 따라가는 좌표, 바깥 방향 좌표) · 자세히 본 수
        for sc, r, c in cand:
            if looked >= 16:
                break
            f, a = float(coord[r, c]), float((R.X if axis == 1 else R.Y)[r, c])  # 바닥선의 바깥 방향 좌표 · 선을 따라가는 좌표
            if any(abs(a - a2) < 0.3 and abs(f - f2) < 0.3 for a2, f2 in picked) or any(abs(a - a2) < 0.2 and abs(f - f2) < 0.04 for a2, f2 in dropped):
                continue
            spot = (a, f) if axis == 1 else (f, a)
            if any(o[0] <= spot[0] <= o[1] and o[2] <= spot[1] <= o[3] for o in hulls):  # 이미 찾은 상자의 옆면 · 윗면 안 — 자세히 볼 것도 없다
                dropped.append((a, f))
                continue
            looked += 1

            def box(a0, a1, t0, t1):  # 선을 따라 a0 ~ a1, 바닥선에서 바깥으로 t0 ~ t1 (m) 인 네모 → (x0, x1, y0, y1)
                lo, hi = sorted((f + sign * t0, f + sign * t1))
                return (a0, a1, lo, hi) if axis == 1 else (lo, hi, a0, a1)

            def frac(a0, a1, t0, t1, what, where=None):
                m, k = R.region(what, *box(a0, a1, t0, t1)), R.region(R.known if where is None else where, *box(a0, a1, t0, t1))
                return float(m[k].mean()) if k.sum() >= 6 else None

            quality = None
            vis = float(depth[r, c]) / p.ppm  # 바닥선에서 화면 끝까지 (m)
            face = abs(f - N[axis]) * (s - 1.0)  # 옆면이 보일 폭 (m)
            top_far = face + size * s  # 바닥선에서 윗면의 먼 변까지 (m)
            band = min(face, vis) - 0.01
            why = None
            if vis >= top_far - 0.03:
                why = "윗면의 먼 변이 화면 안 (정사각형으로 찾을 상자)"
            elif band < p.border_min_depth:
                why = "옆면이 너무 조금 보임"
            elif (frac(a - 0.08, a + 0.08, 0.015, band, nonfloor, seen_face) or 0.0) < p.border_min_face:
                why = "바닥선 바깥이 옆면이 아님"
            else:
                # 바닥선은 옆 상자 · 줄무늬 블록의 바닥선과 한 줄로 이어질 수 있다 → 선을 따라 조금씩 옮겨 가며, 옆면 조건을 모두 채우는 자리 중
                # 옆면의 양 끝(세로 모서리)이 상자 한 변만큼 떨어져 가장 뚜렷한 자리를 고른다
                t = np.linspace(0.02, max(band, 0.03), 10)
                grad = R.gx if axis == 1 else R.gy

                def end(u):
                    xy = (np.full_like(t, u), f + sign * t) if axis == 1 else (f + sign * t, np.full_like(t, u))
                    return float(np.abs(R.sample(grad, *xy)).mean())

                def fails(u):
                    parts = [frac(u - 0.19, u - 0.08, 0.015, band, nonfloor, seen_face), frac(u - 0.08, u + 0.08, 0.015, band, nonfloor, seen_face),
                             frac(u + 0.08, u + 0.19, 0.015, band, nonfloor, seen_face)]
                    if any(v is None for v in parts) or min(parts) < p.border_min_face:
                        return "바닥선 바깥이 옆면이 아님 (%s)" % " · ".join("모름" if v is None else "%.0f%%" % (v * 100) for v in parts)
                    yellow = frac(u - 0.19, u + 0.19, 0.015, band, R.yellow)
                    if yellow is not None and yellow > 0.08:
                        return "노란색 (줄무늬 블록 등)"
                    steps = max(int(round((band - 0.015) / 0.025)), 1)  # 옆면은 바닥선에서 화면 끝까지 빈틈없이 이어진다 — 바닥 글자 위에 떨어져 있는 옆면을 거른다
                    for i in range(steps):
                        t0, t1 = 0.015 + (band - 0.015) * i / steps, 0.015 + (band - 0.015) * (i + 1) / steps
                        v = frac(u - 0.15, u + 0.15, t0, t1, nonfloor, seen_face)
                        if v is not None and v < 0.4:
                            return "바닥선과 옆면 사이에 바닥이 끼어 있음 (바닥 글자 등)"
                    inside = frac(u - 0.17, u + 0.17, -0.07, -0.015, nonfloor)
                    if inside is None or inside > p.border_max_floor:
                        return "바닥선의 카메라 쪽이 바닥이 아님"
                    return None

                best, first = None, fails(a)  # 버릴 때 적을 까닭은 처음 자리의 것
                for o in np.arange(-0.24, 0.2401, 0.02):
                    u = a + float(o)
                    bad = first if abs(o) < 1e-9 else fails(u)
                    if bad is not None:
                        continue
                    lo, hi = end(u - half * 1.03), end(u + half * 1.03)
                    if min(lo, hi) >= 0.25 * p.grad_min and (best is None or lo + hi > best[0]):
                        best = (lo + hi, u)
                if best is None:
                    why = first or "옆면의 양 끝이 안 보임"
                else:
                    a = best[1]
                    # 바닥선이 곧은가: 상자 한 변을 따라 점마다, 바닥선 ±1 cm 안의 가장 센 밝기 변화가 문턱을 넘는 비율. UGV · 사람의 가장자리는 들쭉날쭉하다
                    al = np.linspace(a - half * 0.9, a + half * 0.9, 36)
                    gout = R.gy if axis == 1 else R.gx
                    rows = [np.abs(R.sample(gout, *((al, np.full_like(al, f + d)) if axis == 1 else (np.full_like(al, f + d), al)))) for d in (-0.01, 0.0, 0.01)]
                    straight = float((np.max(rows, axis=0) >= 0.6 * p.grad_min).mean())
                    ends_min = min(end(a - half * 1.03), end(a + half * 1.03)) / p.grad_min
                    quality = (round(straight, 2), round(ends_min, 2))
                    if straight < p.border_min_straight:
                        why = "바닥선이 곧지 않음 (%.0f%%)" % (straight * 100)
            centre = (a, f + sign * half) if axis == 1 else (f + sign * half, a)
            foot = (a, f) if axis == 1 else (f, a)  # 바닥선의 가운데
            if why is None and any(abs(centre[0] - o.x) < size * 0.8 and abs(centre[1] - o.y) < size * 0.8 for o in list(taken) + out):
                why = "이미 찾은 상자와 겹침"
            if why is None and any(o[0] <= foot[0] <= o[1] and o[2] <= foot[1] <= o[3] for o in hulls):
                why = "이미 찾은 상자의 옆면 · 윗면 안의 선 (그림 속 가로줄 · 윗면의 먼 변)"
            if tried is not None:
                tried.append({"side": side, "x": round(centre[0], 3), "y": round(centre[1], 3), "edge": round(sc, 3), "seen_m": round(vis, 3), "reason": why,
                              "quality": quality})
            (dropped if why is not None else picked).append((a, f))
            if why is not None:
                continue
            tc = (N[0] + (centre[0] - N[0]) * s, N[1] + (centre[1] - N[1]) * s)  # 윗면이 보였다면 있을 자리 (그림용)
            top = (tc[0] - half * s, tc[0] + half * s, tc[1] - half * s, tc[1] + half * s)
            out.append(Box(centre[0], centre[1], round(0.5 * sc, 3), top, None, side, border=True))
    return out


def _split_rect(rect, k, axis):
    """직사각형을 줄 방향(axis)으로 k 등분 → [(x0, x1, y0, y1)]."""
    x0, x1, y0, y1 = rect
    if axis == "x":
        e = np.linspace(x0, x1, k + 1)
        return [(float(e[i]), float(e[i + 1]), y0, y1) for i in range(k)]
    e = np.linspace(y0, y1, k + 1)
    return [(x0, x1, float(e[i]), float(e[i + 1])) for i in range(k)]


def _top_cover(a, b):
    """윗면 a 의 넓이 중 b 와 겹치는 비율."""
    ox = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return ox * oy / max((a[1] - a[0]) * (a[3] - a[2]), 1e-9)


def _frame_text(R, p, top, band=(0.015, 0.065), trim=0.15):
    """윗면 테두리 글자 띠가 보이는 정도: 변마다 안쪽 band 띠에서 밝기 변화가 센 픽셀 비율 → 네 변 중 둘째로 큰 값.

    상자 윗면은 변에서 3 cm 쯤 안쪽에 흰 글자가 변을 따라 찍혀 있다. 빛이 번진 윗면은 두세 변만 보이므로 둘째로 큰 값을 본다
    (한 변만이면 옆 상자의 변 · 칠한 선의 모서리가 띠에 들어와도 그렇게 보인다). 띠의 반 넘게가 칠한 선 둘레이거나 화면 밖인 변은 세지 않는다.
    """
    _ev, _eh, _core, tex = _edge_maps(R, p)
    seen = R.valid & ~_wide_lines(R, p)
    busy = (tex > 0) & seen
    x0, x1, y0, y1 = top
    a, b = band
    dx, dy = (x1 - x0) * trim, (y1 - y0) * trim
    found = [0.0, 0.0]
    for box in ((x0 + a, x0 + b, y0 + dy, y1 - dy), (x1 - b, x1 - a, y0 + dy, y1 - dy), (x0 + dx, x1 - dx, y0 + a, y0 + b), (x0 + dx, x1 - dx, y1 - b, y1 - a)):
        m, k = R.region(busy, *box), R.region(seen, *box)
        if k.size and k.mean() > 0.5:
            found.append(float(m[k].mean()))
    return sorted(found)[-2]


def _grid_divisors(tops, N, centre, s, p):
    """축마다 「윗면 가운데 → 상자 가운데」 의 나눗값 κ → ((κx, κy), (축마다 "grid" · "s"), 축마다 잰 것).

    상자는 바닥의 격자(한 칸 grid_cell, 격자점 하나가 경기장 정중앙)에 맞춰 놓인다 (포털의 경기장 모형. 2026-11-19 2경기장 지도 세 장:
    상자 가운데가 격자점에서 RMS 3 cm 안). 그래서 N + (T − N)/κ 가 격자점에 가장 잘 맞는 κ 를 s 둘레에서 축마다 따로 찾는다.
    s 하나로 나누지 않는 까닭
      - 옆면으로 잰 s 는 화면마다 ±2 % 흔들린다 (같은 사진을 로봇은 1.191, PC 는 1.149 로 쟀다 → 2 m 떨어진 상자가 6 cm 달라진다).
      - 바닥 지도의 축척 오차가 가로 · 세로가 다르다. 세로는 줄 경계 두 개(가운데에서 ±0.83 m)만 기준이고 화면 위아래가 잘려 그 바깥은
        렌즈 식을 늘려 쓴 값이다. s 는 축척과 무관한 비율이라 이 오차를 못 없애고, κ 는 「지도 축척 × s」 를 통째로 재므로 없앤다.
    믿지 못하면 s 그대로 둔다: 카메라 바로 아래에서 grid_min_lever 넘게 떨어진 윗면이 grid_min_boxes 개 미만이거나,
    가장 잘 맞는 값이 찾는 범위의 끝이거나, 그때도 격자점과의 차이(가장 안 맞는 1/6 을 뺀 RMS)가 grid_max_rms 보다 클 때
    (상자가 격자에 놓이지 않은 배치). 낸 좌표는 「경기장 정중앙에서 몇 m」 가 맞는 값이다 — 벽까지의 거리는 FieldMap 의 width · height 가 정한다.
    """
    cell = float(p.grid_cell)
    ks = s * np.arange(1.0 - p.grid_range, 1.0 + p.grid_range + 1e-9, 0.0025)
    T = np.array([((t[0] + t[1]) / 2.0, (t[2] + t[3]) / 2.0) for t in tops], float).reshape(-1, 2)
    div, source, info = [], [], []
    for ax in (0, 1):
        v = T[:, ax] - N[ax]
        v = v[np.abs(v) >= p.grid_min_lever]
        k_ax, how, rms_best = s, "s", None
        if len(v) >= p.grid_min_boxes:
            q = (N[ax] - centre[ax] + v[None, :] / ks[:, None]) / cell
            err = np.sort(np.abs(q - np.round(q)) * cell, axis=1)[:, :len(v) - len(v) // 6]  # 가장 안 맞는 1/6 은 뺀다
            rms = np.sqrt((err ** 2).mean(1))
            i = int(np.argmin(rms))
            rms_best = float(rms[i])
            if 0 < i < len(ks) - 1 and rms_best <= p.grid_max_rms:
                k_ax, how = float(ks[i]), "grid"
        div.append(k_ax)
        source.append(how)
        info.append({"axis": "xy"[ax], "boxes": int(len(v)), "best": None if rms_best is None else round(float(ks[i]), 4),
                     "rms": None if rms_best is None else round(rms_best, 4), "used": how})
    return tuple(div), tuple(source), info


def _plausible_top(top, p):
    return p.min_top_m <= top[1] - top[0] <= p.max_top_m and p.min_top_m <= top[3] - top[2] <= p.max_top_m


def _overlap(a, b, size):
    ox = max(0.0, size - abs(a.x - b.x))
    oy = max(0.0, size - abs(a.y - b.y))
    return ox * oy / (size * size)


# ------------------------------------------------------------------ 그림
def draw_boxes(bgr, boundary, view, px_per_m=100):
    """(원본 위에 상자 바닥 · 윗면을 그린 그림, 위에서 본 지도) — 둘 다 BGR.

    초록 = 상자 바닥(높이 0, 경기장 좌표), 노랑 = 윗면(높이 0.4 — 화면에 실제로 보이는 자리). 빨간 X = 버린 후보.
    """
    import cv2

    raw = bgr.copy()
    hgt = view.camera[2]
    N = view.camera[:2]
    size = 0.4

    kx, ky = view.divisor if view.divisor_source != ("", "") else (hgt / (hgt - size),) * 2

    def lift(pts, z):
        if not z:
            return list(pts)
        return [(N[0] + (x - N[0]) * kx, N[1] + (y - N[1]) * ky) for x, y in pts]

    def poly(img, pts, color, th, to_px):
        q = np.concatenate([np.linspace(pts[i], pts[(i + 1) % len(pts)], 20) for i in range(len(pts))])
        uv = to_px(q)
        uv = uv[np.isfinite(uv).all(1)]
        if len(uv) > 1:
            cv2.polylines(img, [np.round(uv).astype(np.int32)], True, color, th, cv2.LINE_AA)

    for i, b in enumerate(view.boxes):
        sq = [(b.x - size / 2, b.y - size / 2), (b.x + size / 2, b.y - size / 2), (b.x + size / 2, b.y + size / 2), (b.x - size / 2, b.y + size / 2)]
        poly(raw, sq, (0, 255, 0), 2, boundary.field_to_pixel)
        poly(raw, lift(sq, size), (0, 220, 255), 1, boundary.field_to_pixel)
        uv = boundary.field_to_pixel(lift([(b.x, b.y)], size))[0]
        if np.isfinite(uv).all():
            cv2.putText(raw, str(i), (int(uv[0]) - 5, int(uv[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for r in view.rejected:
        uv = boundary.field_to_pixel([(r["x"], r["y"])])[0]
        if np.isfinite(uv).all():
            cv2.drawMarker(raw, (int(uv[0]), int(uv[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
    uv = boundary.field_to_pixel([N])[0]
    if np.isfinite(uv).all():
        cv2.drawMarker(raw, (int(uv[0]), int(uv[1])), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)

    w, h = boundary.field_w, boundary.field_h
    gw, gh = int(round(w * px_per_m)), int(round(h * px_per_m))
    gx, gy = np.meshgrid(np.arange(gw) + 0.5, np.arange(gh) + 0.5)
    pix = boundary.field_to_pixel(np.stack([gx.ravel() / px_per_m, h - gy.ravel() / px_per_m], 1)).reshape(gh, gw, 2)
    pix = np.nan_to_num(pix, nan=-1e4).astype(np.float32)
    top = cv2.remap(bgr, pix[..., 0], pix[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=_FILL)

    def to_top(q):
        q = np.asarray(q, float)
        return np.stack([q[:, 0] * px_per_m, (h - q[:, 1]) * px_per_m], 1)

    for i, b in enumerate(view.boxes):
        sq = [(b.x - size / 2, b.y - size / 2), (b.x + size / 2, b.y - size / 2), (b.x + size / 2, b.y + size / 2), (b.x - size / 2, b.y + size / 2)]
        poly(top, sq, (0, 255, 0), 2, to_top)
        poly(top, lift(sq, size), (0, 220, 255), 1, to_top)
        cv2.putText(top, "%d (%.2f, %.2f)" % (i, b.x, b.y), (int(b.x * px_per_m) - 40, int((h - b.y) * px_per_m) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(top, (int(N[0] * px_per_m), int((h - N[1]) * px_per_m)), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
    return raw, top
