"""드론 화면 한 장 + 경기장 경계(drone_boundary) → 줄마다 어느 거점인지 (작전지휘소 · 정비소 · 군수보급소).

왜 필요한가 (2026-11-19 주최측 공지)
  「M5 목표거점 — 위치: 고정, 종류/배치: 작전지휘소 · 정비소 · 군수보급소의 배치 위치는 변경 가능, 이미지 · 형태: 사전 제공 기준과 동일」.
  서버는 목표를 종류로만 알려 준다(예 RED_MAINTENANCE). 그 종류가 1 · 2 · 3번 줄 중 어디에 있는지는 어느 API 에도 없다.
  거점의 종류를 알려 주는 것은 거점 바로 뒤 벽에 붙은 그림 한 장뿐이다 (가로 1.2 m × 벽 높이 0.4 m, 줄무늬 블록 사이).
  실제로 연습 경기장 두 곳은 위부터 정비소 · 작전지휘소 · 군수보급소, 본선 경기장은 (11-19 사진) 작전지휘소 · 정비소 · 군수보급소였다.

하는 일
  1) 벽 그림이 있을 띠를 화면에서 편다 (wall_band). 벽은 높이가 있어 바닥 대응(Boundary.field_to_pixel)으로 바로 옮길 수 없다 →
     drone_boxes 와 같은 식을 쓴다: 높이가 있는 점은 카메라 바로 아래 점 N 에서 바깥으로 밀려 보인다 (벽 위쪽 끝은 s = Hc / (Hc − 0.4) 배 먼 자리).
     편 띠는 「경기장 안에서 그 벽을 똑바로 본 모양」 이다 (위가 벽 위쪽. BLUE 벽은 왼쪽이 y 가 작은 쪽, RED 벽은 왼쪽이 y 가 큰 쪽).
     경계 모델이 아는 벽 자리는 정확하지 않다 — 연습 화면 넷에서 벽 바닥선은 모델보다 9 ~ 22 px 안쪽, 벽 위쪽 끝은 9 ~ 41 px 안쪽에 있었다
     (경계는 바닥 선으로만 맞춘 것이라 벽 바깥까지는 잘 맞지 않는다). 그래서 띠를 넉넉히 편다: 모델의 벽 바닥선 안쪽 0.65배 ~ 바깥 1.3배
     (1.0 = 모델이 계산한 벽 위쪽 끝). 연습 화면에서 그림은 이 띠의 −0.39 ~ +0.80 사이에, 높이 0.63 ~ 1.05 로 들어 있었다.
  2) 띠 석 장(한 벽의 1 · 2 · 3번 줄)에서 기준 그림 석 장을 찾는다. 밝기가 아니라 윤곽(밝기 변화의 세기)으로 견주고,
     기준 그림의 가운데에서 조금 위쪽 부분(거점 안에 선 로봇에 덜 가려지는 곳)을 크기 · 자리를 바꿔 가며 댄다.
     한 벽의 세 그림은 같은 벽에 붙어 있으므로 띠 안에서 같은 높이 · 같은 세로 크기에 있어야 한다
     → 세 장이 같은 높이(±row_tol) · 같은 세로 배율에서 낸 점수만 합친다 (그림이 아닌 것에 우연히 맞는 일이 줄어든다).
  3) 한 벽에는 세 종류가 하나씩 있다 → 여섯 가지 배치마다 가장 좋은 점수 합을 구해, 가장 큰 배치를 고른다.
     둘째 배치와의 차이(gap)가 min_gap 보다 작으면 「모름」 (None) — 틀린 답을 내느니 내지 않는다.
     모르면 supervisor 는 상대 진영의 거점을 차례로 들어가 본다 (mission._assault_order — 목표가 아닌 거점에 들어가도 아무 일도 없다).

모름(None)이 되는 경우
  경계가 없음 · 띠가 화면 밖이거나 렌즈 모델로 옮길 수 없음 · 화면에서 벽이 너무 얇게 보임(카메라가 벽을 거의 위에서만 봄) ·
  그림이 가려지거나 흐려서, 또는 벽에 그 그림이 없어서 배치 사이의 차이가 작음.
  띠의 절반 넘게가 화면 밖이거나, 화면 안이어도 무늬가 전혀 없는 자리(한 가지 색으로 칠해졌거나 가려져서 온 자리 — blank_mask)일 때도 모름이다.
  본선 카메라는 경기장의 좌우가 잘려 벽이 화면에 없을 수 있다 (2026-11-19 주최측 안내) — 그 진영은 모름이 되고, 보이는 진영만 읽는다.

기준 그림
  주최측 1차 사전교육 자료(공식_자료/원본/PDF/prestudy(1)_2026MAICON.pdf) 15쪽 「목표거점 형태」 의 사진 석 장을
  144 × 48 회색으로 줄인 것. 로봇에 따로 파일을 깔지 않아도 되게 이 파일 끝에 PNG 를 글자(base64)로 넣었다.

지도에 넣는 법 (add_to_map)
  지도의 bases 에는 자리 이름(B-1 ~ R-3) 여섯 개가 먼저 있고 그대로 둔다 (navigation 이 그 이름으로 출발 칸을 찾는다).
  알아낸 진영마다 보고값 이름(예 RED_MAINTENANCE) 세 개를 그 뒤에 덧붙인다 — 네모와 줄 번호는 그 그림이 있는 줄의 것.
  supervisor(arena.field_map_from_msg)는 자리 이름을 기본 배치로 먼저 읽고, 뒤에 온 보고값 이름으로 덮어쓴다. 메시지 형식은 그대로다.

numpy · OpenCV 만 쓴다. ROS 에 묶이지 않는다.
"""
from __future__ import annotations

import base64
import itertools
import time
from dataclasses import dataclass

import numpy as np

KINDS = ("COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT")
CAMPS = ("BLUE", "RED")
ROWS = (1, 2, 3)
_LAYOUTS = tuple(itertools.permutations(range(3)))  # 배치 여섯 가지: (1번 줄 종류 번호, 2번 줄, 3번 줄)


@dataclass
class PanelParams:
    wall_height: float = 0.40  # 벽 높이 = 그림 높이 (m) — 상자 높이와 같다
    panel_length: float = 1.20  # 그림 가로 (m) — 줄 가운데 ± 0.60
    default_camera_height: float = 2.65  # 카메라 높이를 받지 못했을 때 (drone_boxes 의 기본값과 같다)
    size: tuple = (144, 48)  # 기준 그림 (가로, 세로) px
    band: tuple = (-0.65, 1.3)  # 펴는 띠: 모델의 벽 바닥선에서 안쪽(−) ~ 바깥(+), 단위 = 모델이 계산한 벽 그림 높이
    band_rows: int = 112  # 편 띠의 세로 px (가로는 size[0]) — 모델의 벽 그림 높이 1.0 이 약 57 px
    grid_step: int = 4  # 벽 자리 → 화면 픽셀 계산은 이 간격으로만 하고 사이는 고르게 채운다 (빠르게)
    # 견주기
    blur_1: float = 0.9  # 윤곽을 내기 전 흐리기 (px, 편 띠 기준)
    blur_2: float = 1.1  # 윤곽을 낸 뒤 흐리기 — 윤곽 자리가 1 ~ 2 px 틀려도 맞게
    # 기준 그림에서 견주는 부분 (왼쪽 · 오른쪽 끝, 위 · 아래 끝 — 그림 크기에 대한 비율). 나머지는 자리를 옮겨 볼 여유다.
    # 세로는 가운데보다 조금 위쪽을 쓴다: 거점 안에 선 로봇은 카메라에서 볼 때 벽 그림의 아래쪽을 가린다 (출발할 때는 여섯 거점 모두에 로봇이 있다).
    # 견줘 본 값 (연습 화면 52장 · 벽 104개, 2026-11-19): 0.19 ~ 0.81 은 로봇 크기의 덩어리를 그려 넣으면 모름이 1 ~ 11개,
    # 0.05 ~ 0.60 은 그림을 지운 화면에서 틀린 답 1개, 0.10 ~ 0.72 는 덩어리를 그려 넣어도 모름 0 ~ 3개이고 그림을 지우면 모두 모름
    core_x: tuple = (0.14, 0.86)
    core_y: tuple = (0.10, 0.72)
    scales_x: tuple = (0.85, 1.0, 1.15)  # 기준 그림의 가로 배율
    scales_y: tuple = (0.75, 0.9, 1.05, 1.2, 1.35, 1.5)  # 세로 배율 — 연습 화면은 띠 안에서 그림 높이가 36 ~ 60 px (기준 그림 48 px), 모델이 맞는 카메라면 57 px
    row_tol: int = 4  # 한 벽의 세 그림이 띠 안에서 이만큼(px)까지 다른 높이에 있어도 같은 높이로 친다
    # 믿을지 정하기
    # 가장 좋은 배치와 둘째 배치의 점수 합 차이(gap)가 이보다 작으면 모름. 잰 값 (연습 화면 52장 · 벽 104개, 2026-11-19):
    #   그림이 다 보이면 0.43 ~ 1.00 (가운데값 0.91, 104개 모두 맞음) · 여섯 거점에 로봇 크기의 덩어리를 그려 넣으면 0.27 이상 (모두 맞음) ·
    #   한 벽의 그림 한 장을 가로의 20 ~ 100 % 가리면 0.23 ~ 0.76 (나머지 두 장으로 정해진다, 30번 모두 맞음) ·
    #   그림을 지우면 0.00 ~ 0.13, 화면을 뒤집어 거울상 그림이 오게 하면 0.17 이하 (모두 모름)
    min_gap: float = 0.20
    min_seen: float = 0.50  # 띠 중 화면에 실제로 찍힌 비율이 이보다 낮으면 모름 (연습 화면은 1 · 3번 줄 0.86 ~ 1.00, 2번 줄 1.00)
    # 화면에서 무늬가 전혀 없는 자리(한 가지 색으로 칠해졌거나 가려져서 온 자리)도 찍히지 않은 자리로 친다 (blank_mask).
    # 벽이 화면에 없는데 띠만 화면 안에 놓이는 경우를 막는다 — 본선 카메라는 경기장 좌우가 잘릴 수 있다 (2026-11-19 주최측 안내).
    # 잰 값 (연습 화면 두 장의 좌우를 회색 · 흰색으로 칠해 자르는 자리를 5 px 씩 옮겨 봄, 2026-11-19): 이 검사가 없으면 칠한 자리와 사진이
    #   만나는 곧은 선이 윤곽으로 잡혀 틀린 배치를 차이 0.22 ~ 0.43 으로 믿는 일이 있었다 (432번 중 9번).
    # 연습 화면에는 영향이 없다: 이 PC 의 연습 화면 145장(읽은 벽 281개)에서 이 검사를 넣기 전과 답 · 차이가 모두 같았다 (2026-11-19).
    #   JPEG 이 뭉갠 바닥은 8 px 네모 단위라 25 px 네모 기준에는 걸리지 않는다 (9 px · 13 px 네모로 하면 연습 화면의 벽 띠 안에서도 걸렸다)
    blank_window: int = 25  # 이 크기(px)의 네모 안에서 밝기가 blank_range 이하로만 다르면 무늬가 없는 자리. 0 이면 이 검사를 끈다
    blank_range: int = 1
    blank_margin: int = 10  # 그 자리의 둘레 이만큼(px)도 뺀다 — 칠한 자리와 사진이 만나는 선, JPEG 이 그 선 둘레에 남기는 얼룩
    min_depth_px: float = 16.0  # 화면에서 벽 그림의 높이(모델 계산)가 이보다 얇으면 모름
    min_length_px: float = 40.0  # 화면에서 벽 그림의 가로가 이보다 짧으면 모름


# ------------------------------------------------------------------ 벽 띠 펴기
def _row_range(height, row):
    """row 1 ~ 3 의 y 범위 (1번 줄이 위). drone_field_map._row_range 와 같다."""
    h = height / 3.0
    return (3 - row) * h, (4 - row) * h


def _band_points(boundary, camp, row, cols, offsets, camera_xy, p):
    """띠 위의 점들이 보이는 화면 픽셀. cols = 가로 비율(0 이 왼쪽), offsets = 벽 바닥선에서 밀린 정도(바닥 좌표의 배율 차). 모양 (len(offsets), len(cols), 2)."""
    width, height = float(boundary.field_w), float(boundary.field_h)
    lo, hi = _row_range(height, row)
    mid = (lo + hi) / 2.0
    y0, y1 = mid - p.panel_length / 2.0, mid + p.panel_length / 2.0
    # 경기장 안에서 벽을 보고 선 사람의 왼쪽: BLUE 벽(x = 0)을 보면 y 가 작은 쪽, RED 벽을 보면 y 가 큰 쪽
    along = y0 + cols * (y1 - y0) if camp == "BLUE" else y1 - cols * (y1 - y0)
    wall_x = 0.0 if camp == "BLUE" else width
    nx, ny = camera_xy
    lift = (1.0 + offsets)[:, None]  # 카메라 바로 아래 점에서 본 거리의 배율 (1 = 벽 바닥선)
    qx = nx + (wall_x - nx) * lift + 0.0 * along[None, :]
    qy = ny + (along[None, :] - ny) * lift
    px = boundary.field_to_pixel(np.stack([qx.ravel(), qy.ravel()], 1))
    return px.reshape(len(offsets), len(cols), 2)


def blank_mask(grey, params=None):
    """화면(회색)에서 무늬가 전혀 없는 자리 (bool, 화면 크기). 그런 자리가 없거나 검사를 껐으면 None.

    blank_window 크기의 네모 안에서 밝기가 blank_range 이하로만 다른 곳을 찾고, 그 네모 전체와 둘레 blank_margin px 를 표시한다.
    한 가지 색으로 칠해졌거나 가려져서 온 자리가 여기에 걸린다. 실제로 찍힌 바닥 · 벽은 잡음과 무늬가 있어 걸리지 않는다.
    """
    import cv2

    p = params or PanelParams()
    size = int(p.blank_window)
    if size <= 1 or grey is None or grey.ndim != 2 or min(grey.shape) < 1:
        return None
    kernel = np.ones((size, size), np.uint8)
    spread = cv2.subtract(cv2.dilate(grey, kernel), cv2.erode(grey, kernel))  # 네모 안의 가장 밝은 값 − 가장 어두운 값
    flat = (spread <= p.blank_range).astype(np.uint8)  # 이 점을 가운데로 한 네모 전체에 무늬가 없다
    if not flat.any():
        return None
    grow = size + 2 * max(0, int(p.blank_margin))
    return cv2.dilate(flat, np.ones((grow, grow), np.uint8)) > 0


def wall_band(grey, boundary, camp, row, scale=None, camera_xy=None, params=None, blank=None):
    """화면(회색)에서 한 거점의 벽 띠를 편다 → (띠 uint8, 찍힌 자리 표시 bool, "") 또는 (None, None, 못 편 까닭).

    scale: drone_boxes 가 잰 s = Hc / (Hc − 0.40) (BoxView.scale). None 이면 default_camera_height 로 계산한다.
    camera_xy: 카메라 바로 아래 점 (m). None 이면 경기장 정중앙 (drone_boxes 와 같다).
    찍힌 자리 표시: 연습 카메라는 1 · 3번 줄 벽의 바깥쪽 모서리가 화면 위아래 끝에서 잘린다. 잘린 자리는 견줄 때 뺀다.
    blank: blank_mask(grey) 의 결과 (화면에서 무늬가 전혀 없는 자리) — 그 자리도 찍히지 않은 자리로 친다.
           None 이면 여기서 구한다. False 면 그런 자리가 없다고 보고 건너뛴다 (identify 가 한 번 구해 보고 없을 때).
    """
    import cv2

    p = params or PanelParams()
    if boundary is None or boundary.hom is None:
        return None, None, "경계가 없음"
    tw, tb = int(p.size[0]), int(p.band_rows)
    h = p.wall_height
    s = float(scale) if scale and scale > 1.0 else p.default_camera_height / (p.default_camera_height - h)
    if not 1.02 <= s <= 2.0:
        return None, None, "카메라 높이 배율 %.3f 이 범위 밖" % s
    n = tuple(camera_xy) if camera_xy is not None else (float(boundary.field_w) / 2.0, float(boundary.field_h) / 2.0)
    step = max(1, int(p.grid_step))
    gw, gh = tw // step + 1, tb // step + 1
    # 편 띠의 픽셀 가운데(0.5 px)에 맞춘 성긴 격자. 맨 윗줄이 가장 바깥(band[1]), 맨 아랫줄이 가장 안쪽(band[0])
    cols = (np.arange(gw) * step + 0.5) / tw
    down = (np.arange(gh) * step + 0.5) / tb
    offsets = (p.band[1] - down * (p.band[1] - p.band[0])) * (s - 1.0)
    with np.errstate(all="ignore"):
        coarse = _band_points(boundary, camp, row, cols, offsets, n, p)
        edge = _band_points(boundary, camp, row, np.array([0.0, 0.5, 1.0]), np.array([0.0, s - 1.0]), n, p)
    if not (np.isfinite(coarse).all() and np.isfinite(edge).all()):
        return None, None, "벽 자리를 화면으로 옮길 수 없음 (렌즈 모델 밖)"
    depth = float(np.linalg.norm(edge[1, 1] - edge[0, 1]))
    length = float(np.linalg.norm(edge[0, 2] - edge[0, 0]))
    if depth < p.min_depth_px or length < p.min_length_px:
        return None, None, "화면에서 벽 그림이 너무 작음 (가로 %.0f px, 높이 %.0f px)" % (length, depth)
    # 성긴 격자 → 편 띠의 모든 픽셀 자리 (띠의 i번째 픽셀은 격자의 i / step 번째 자리 — 두 격자점 사이를 곧게 잇는다)
    at_x = np.repeat((np.arange(tw, dtype=np.float32) / step)[None, :], tb, 0)
    at_y = np.repeat((np.arange(tb, dtype=np.float32) / step)[:, None], tw, 1)
    full = cv2.remap(np.ascontiguousarray(coarse, np.float32), at_x, at_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ih, iw = grey.shape[:2]
    seen = (full[..., 0] >= 0) & (full[..., 0] <= iw - 1) & (full[..., 1] >= 0) & (full[..., 1] <= ih - 1)
    if seen.mean() < p.min_seen:
        return None, None, "벽 자리가 화면 밖 (%.0f%% 만 안)" % (100 * seen.mean())
    if blank is None:
        blank = blank_mask(grey, p)
    if isinstance(blank, np.ndarray) and blank.shape[:2] == grey.shape[:2]:
        empty = cv2.remap(blank.astype(np.uint8), np.ascontiguousarray(full[..., 0]), np.ascontiguousarray(full[..., 1]), cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        seen &= ~empty
        if seen.mean() < p.min_seen:
            return None, None, "벽 자리에 무늬가 없음 — 벽이 화면에 찍히지 않은 듯 (%.0f%% 만 보임)" % (100 * seen.mean())
    band = cv2.remap(grey, np.ascontiguousarray(full[..., 0]), np.ascontiguousarray(full[..., 1]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return band, seen, ""


# ------------------------------------------------------------------ 견주기
def _outline(grey, p, seen=None):
    """회색 그림 → 윤곽 세기 (float32). 조명 · 인쇄 색이 달라도 건물 · 천막 · 탱크의 모양은 같다. 찍히지 않은 자리는 0."""
    import cv2

    g = cv2.createCLAHE(2.0, (4, 4)).apply(np.ascontiguousarray(grey, np.uint8))
    g = cv2.GaussianBlur(g, (0, 0), p.blur_1)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    out = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), p.blur_2)
    if seen is not None and not seen.all():
        # 화면 끝 바로 안쪽 몇 px 도 뺀다 (끝에서 늘여 붙인 값과 만나는 자리)
        keep = cv2.erode(seen.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        out = np.where(keep, out, 0.0).astype(np.float32)
    return out


_CACHE = {}


def reference(kind, size=None):
    """기준 그림 (회색 uint8, size 크기 — None 이면 넣어 둔 크기 그대로). 파일 끝의 글자를 한 번만 푼다."""
    import cv2

    key = ("grey", kind, None if size is None else tuple(size))
    if key not in _CACHE:
        raw = np.frombuffer(base64.b64decode(_REFERENCE_PNG[kind]), np.uint8)
        grey = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if grey is None:
            raise ValueError("기준 그림 %s 을 풀지 못함" % kind)
        if size is not None and (grey.shape[1], grey.shape[0]) != tuple(size):
            grey = cv2.resize(grey, tuple(size), interpolation=cv2.INTER_AREA)
        _CACHE[key] = grey
    return _CACHE[key]


def _cores(kind, p):
    """기준 그림의 윤곽을 배율마다 바꾼 뒤 견줄 부분(core_x · core_y)만 잘라 둔 것 → [세로 배율 번호][가로 배율 번호] float32. 한 번 만들어 두고 다시 쓴다."""
    import cv2

    key = ("cores", kind, tuple(p.size), p.blur_1, p.blur_2, tuple(p.core_x), tuple(p.core_y), tuple(p.scales_x), tuple(p.scales_y))
    if key not in _CACHE:
        tw, th = p.size
        base = _outline(reference(kind, p.size), p)
        out = []
        for sy in p.scales_y:
            line = []
            for sx in p.scales_x:
                w, h = int(round(tw * sx)), int(round(th * sy))
                scaled = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
                x0, x1 = int(round(w * p.core_x[0])), int(round(w * p.core_x[1]))
                y0, y1 = int(round(h * p.core_y[0])), int(round(h * p.core_y[1]))
                line.append(np.ascontiguousarray(scaled[y0:y1, x0:x1]))
            out.append(line)
        _CACHE[key] = out
    return _CACHE[key]


def _band_outline(band, seen, p):
    """띠 한 장 → 윤곽 세기 (띠 크기 size[0] × band_rows 로 맞춘 것)."""
    import cv2

    tw, tb = int(p.size[0]), int(p.band_rows)
    if (band.shape[1], band.shape[0]) != (tw, tb):
        band = cv2.resize(band, (tw, tb), interpolation=cv2.INTER_AREA)
        seen = None if seen is None else cv2.resize(seen.astype(np.uint8), (tw, tb), interpolation=cv2.INTER_NEAREST) > 0
    return _outline(band, p, seen)


def _band_scores(outline, p):
    """띠 한 장의 윤곽 → 점수 [종류][세로 배율][띠 안의 세로 자리] (가로 자리 · 가로 배율은 가장 좋은 것). 무늬가 없는 띠는 모두 0.

    세로 자리 = 기준 그림(그 배율로 바꾼 것)의 윗변이 놓인 띠의 줄. 견줄 부분이 띠 밖으로 나가는 자리는 −1.
    """
    import cv2

    tb, tw = outline.shape[:2]
    if float(outline.std()) < 1e-3:
        return np.zeros((len(KINDS), len(p.scales_y), tb), np.float32)
    out = np.full((len(KINDS), len(p.scales_y), tb), -1.0, np.float32)
    for k, kind in enumerate(KINDS):
        for j, line in enumerate(_cores(kind, p)):
            best = None
            for core in line:
                if core.shape[0] > tb or core.shape[1] > tw or min(core.shape) < 4:
                    continue
                result = cv2.matchTemplate(outline, core, cv2.TM_CCOEFF_NORMED)
                rows = np.where(np.isfinite(result), result, -1.0).max(axis=1)
                best = rows if best is None else np.maximum(best, rows)
            if best is None:
                continue
            # 견준 부분의 윗변 자리 → 그림 윗변 자리 (견준 부분은 그림 윗변에서 core_y[0] 만큼 아래에서 시작한다)
            shift = int(round(p.size[1] * p.scales_y[j] * p.core_y[0]))
            n = min(len(best) - shift, tb)
            if n > 0:
                out[k, j, :n] = best[shift:shift + n]
    return out


def _spread(values, tol):
    """마지막 축을 따라, 자리마다 ±tol 안의 가장 큰 값."""
    if tol <= 0:
        return values
    pad = np.full(values.shape[:-1] + (tol,), -1.0, values.dtype)
    wide = np.concatenate([pad, values, pad], axis=-1)
    n = values.shape[-1]
    return np.maximum.reduce([wide[..., i:i + n] for i in range(2 * tol + 1)])


def wall_layout(tables, params=None):
    """한 벽의 띠 석 장의 점수(_band_scores, 1 · 2 · 3번 줄 순서) → (배치 [1번 줄 종류, 2번 줄, 3번 줄], gap, 그 배치일 때의 점수 [줄][종류]).

    세 장이 같은 세로 배율 · 같은 높이(±row_tol)에서 낸 점수만 합친다. gap = 가장 좋은 배치의 합 − 다른 배치가 (어느 높이 · 배율에서든) 낼 수 있는 가장 큰 합.
    """
    p = params or PanelParams()
    spread = [_spread(t, int(p.row_tol)) for t in tables]  # [줄][종류][세로 배율][세로 자리]
    best = []
    for layout in _LAYOUTS:
        total = spread[0][layout[0]] + spread[1][layout[1]] + spread[2][layout[2]]  # [세로 배율][세로 자리]
        j, i = np.unravel_index(int(np.argmax(total)), total.shape)
        best.append((float(total[j, i]), layout, int(j), int(i)))
    best.sort(key=lambda item: -item[0])
    top, layout, j, i = best[0]
    scores = [[float(spread[r][k][j, i]) for k in range(len(KINDS))] for r in range(3)]
    return [KINDS[k] for k in layout], top - best[1][0], scores


# ------------------------------------------------------------------ 밖에서 부르는 것
def identify(bgr, boundary, scale=None, camera_xy=None, params=None):
    """드론 화면(BGR) + 경계 → (kinds, detail).

    kinds  = {"BLUE": [1번 줄 종류, 2번 줄, 3번 줄] 또는 None, "RED": 같은 모양}. 종류는 KINDS 의 이름. 분명하지 않으면 None.
    detail = 기록용. 진영마다 {"order": 가장 좋은 배치(믿지 않아도 적는다) 또는 None, "gap": 둘째 배치와의 차이, "accepted": 믿었는가,
             "scores": 그 배치일 때의 [줄][종류] 점수, "note": 모름인 까닭}, 그리고 "seconds".
    scale · camera_xy 는 wall_band 와 같다 (drone_boxes 의 BoxView.scale · BoxView.camera[:2] 를 주면 된다).
    """
    import cv2

    p = params or PanelParams()
    began = time.monotonic()
    kinds = {camp: None for camp in CAMPS}
    detail = {}
    grey = None
    if bgr is not None and getattr(bgr, "size", 0):
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    blank = None if grey is None else blank_mask(grey, p)  # 한 번만 구해 여섯 띠에 같이 쓴다 (없으면 None)
    for camp in CAMPS:
        entry = {"order": None, "gap": None, "accepted": False, "scores": None, "note": ""}
        detail[camp] = entry
        if grey is None:
            entry["note"] = "화면이 없음"
            continue
        outlines = []
        for row in ROWS:
            band, seen, why = wall_band(grey, boundary, camp, row, scale, camera_xy, p, False if blank is None else blank)
            if band is None:
                entry["note"] = "%d번 줄: %s" % (row, why)
                break
            outlines.append(_band_outline(band, seen, p))
        if len(outlines) < 3:
            continue
        order, gap, scores = wall_layout([_band_scores(o, p) for o in outlines], p)
        entry.update(order=order, gap=round(gap, 3), scores=[[round(v, 3) for v in s] for s in scores])
        if gap >= p.min_gap:
            entry["accepted"] = True
            kinds[camp] = order
        else:
            entry["note"] = "배치 사이의 차이 %.2f 가 %.2f 보다 작음 (그림이 가려졌거나 흐리거나 없음)" % (gap, p.min_gap)
    detail["seconds"] = round(time.monotonic() - began, 3)
    return kinds, detail


def kind_regions(width, height, kinds, base_depth=0.40, base_length=1.20):
    """알아낸 진영의 거점을 보고값 이름으로 → [{"name", "min_x", "min_y", "max_x", "max_y", "row"}] (진영마다 1 · 2 · 3번 줄 순서).

    네모는 drone_field_map.base_regions 와 같은 도면값이다 (벽에서 0.40 m, 줄 가운데 ± 0.60 m).
    kinds 가 None 인 진영, 세 종류를 하나씩 담지 않은 진영은 넣지 않는다.
    """
    out = []
    for camp in CAMPS:
        order = (kinds or {}).get(camp)
        if not order or sorted(order) != sorted(KINDS):
            continue
        for row in ROWS:
            lo, hi = _row_range(float(height), row)
            mid = (lo + hi) / 2.0
            x0, x1 = (0.0, base_depth) if camp == "BLUE" else (float(width) - base_depth, float(width))
            out.append({"name": "%s_%s" % (camp, order[row - 1]), "min_x": round(x0, 4), "min_y": round(mid - base_length / 2.0, 4),
                        "max_x": round(x1, 4), "max_y": round(mid + base_length / 2.0, 4), "row": row})
    return out


def add_to_map(data, kinds):
    """지도 dict(drone_field_map.field_map 의 결과)의 bases 뒤에 보고값 이름의 거점을 덧붙인다. 이미 있는 것(자리 이름 B-1 …)은 그대로 둔다.

    같은 이름이 이미 있으면 다시 넣지 않는다. data 를 고쳐서 돌려준다.
    """
    if not data or not kinds:
        return data
    bases = data.setdefault("bases", [])
    have = {b.get("name") for b in bases}
    bases.extend(r for r in kind_regions(data["width"], data["height"], kinds) if r["name"] not in have)
    return data


def annotate(data, bgr, boundary, view=None, params=None):
    """map_maker.compute 가 부른다: 벽 그림을 읽어 지도 dict 에 덧붙이고 기록용 dict 를 돌려준다. 무슨 일이 있어도 예외를 내지 않는다.

    view: drone_boxes.BoxView (카메라 높이 배율 · 카메라 바로 아래 점을 가져다 쓴다). 없으면 기본값.
    돌려주는 것: identify 의 detail 에 "kinds": {"BLUE": [...] 또는 None, "RED": ...} 를 더한 것.
    """
    try:
        if boundary is None or not getattr(boundary, "ok", False):
            return {"kinds": {camp: None for camp in CAMPS}, "note": "경계 검사를 통과하지 못해 벽 그림을 읽지 않음"}
        scale = getattr(view, "scale", None)
        camera = getattr(view, "camera", None)
        kinds, detail = identify(bgr, boundary, scale if scale and scale > 1.0 else None, tuple(camera[:2]) if camera else None, params)
        if data is not None:
            add_to_map(data, kinds)
        return dict(detail, kinds=kinds)
    except Exception as error:  # noqa: BLE001 — 벽 그림을 못 읽어도 지도는 나가야 한다 (supervisor 가 거점을 차례로 들어가 본다)
        return {"kinds": {camp: None for camp in CAMPS}, "note": "벽 그림 읽기 오류 %s: %s" % (type(error).__name__, error)}


# ------------------------------------------------------------------ 기준 그림 (144 × 48 회색 PNG 를 base64 글자로)
_REFERENCE_PNG = {
    "COMMAND_POST": (
        "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"
    ),
    "MAINTENANCE": (
        "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"
        "j5/3nFqsXr8dH51OOpNkPxX3nq3fumQObQZ9aNOZTpmrfrxy+ZPGdlpfrlfVNGnrVptMKjUdV3rJh61TPzxFhdVvv1Cr2s6bQrz1K3lAvujTzbmMNp9mL6on"
        "pJys3KTNmXRu56shoND2No8O/eZ2ufdguFaVposh3e+eDJPFwxN62qYTM91mSlfnx5nVTXfsc6hb69seGNJ+3UdLgZNAPCNy3hEreq57TvdFm/jjAfGjb5so"
        "WX3ryc8fnEYbc75/VMvJrFywubY27utyL46OEq5EkaWeySUbrbuAztXpaqOtlyRvRm4tDVuJtaauoCmfQHu69YV0WF524qzoEKp++Gjr9R3Blq8dVdPxbq90"
        "rO74qNMpcym6jbx+vdAp+P91kG99uPuqmTKv/vrFkUDeX12ttzrRMu4s2tk0m5XbuokysoZwWJZur6QyXj56m+j53iZ7mv1olD2Vu4NR5WxX3ng+53aNxrlu"
        "pCtMCb63NsONUwN2dr80m9LzyLHbi09+Ev8X36rrWIPiB6puzo9JLstknpHHQOZdBeJOos6/+qtdsMhHVSs78g1ZjqdDJ9d64Gey/7w2YbAl+mPvZ/3UgOQo"
        "aGL18Lj7fGN3FHiPO3Gl0XSlirG4lPc0lPii6rftGYi5xdYKWsOB7a+OyJHyfimXSTWNc1sNqtQsfAkRqS9PfX/JClipqFkvBnR5ciJQs0xXUdFrgPJBhAG7"
        "orQN0kpeu7rX9PmiWMumySGESqrcul/GcdnduHHenxab8ceapNHN7Rql3R4ertsfp6cHi2hOCw+locGdm05BaVe99/m4qn7w9vSTCp3Ntde23fpuFK9X8Ton"
        "2UkaMZYtFHRYC7IO4KTSF+RSItTVPG0uHZHuARWQM+IpP7+12pCztnWiqqPudv2Ur2dX3htIeSoXZPvyvB8EwGQEkUZk9HoZs5+/0Law22U7XUviHrYdL0cZ"
        "t+evXtfL9ut1suHacqt78zJ00ST729n580nW6KIP3QmtSStH//5I/6j3+RvLbAVZjcSEo/XFoFJQNB7rY8hWbUmtq4alPaSxitLgY7gEykZDmAXFTF9Oir1S"
        "UEOtp2ifpoJQYqbDVtImnayuqJfWoMlDqvuWdFSYb6SiXj+sJua5ljuLIQTCoYC+lsu8IcXPIA2s0UH44Q3obwxaj4cxgOFGGygZzSExwgAWoA7dDEOoAQd0"
        "TEiv4SMQEoA7ZZGH8iIy9ApAHNQahK70hhiqoN4sJwqC/WKC7FuRuluGkcae/ZYsfLRMRTchtb2+m2QbggE6+MXUgBCNUBIXJqW+MopYHUnIcoho6n0YVBof"
        "5jUeaZ1a6uHDVIBfa+IZEtG0PoZtYZYAQYn9rCvjDgvjEREkPY4QZrRoAp7U3aJGqunuT2q9UL2c/n4NXovT2aCkPIvCpDsIZRuEFIIiwC0hwwvIELuVO6cL"
        "CDTN0xRdiRmNvoD0h/KL4A/fMi5B+qGnG4r14gS4woIkI1CUZA2UjjjJkMGGCgN/ooiIhG3CX0goCifSTyGHXBjQMog2xgRQRAffYClHtC7OGi58aWo/2NgU"
        "II6e2nmZqa4SYRrW+DDUjTmvGXqVQ0Qg6jkFO0FNhJN5u8ZtwsN0QIQJC/XaCTyipHbG8SCJA1HCfRPm7Z5+VISFuTPQwBqCD4lIYoSKIe2Dc2143hgs56Bf"
        "vVlVJdvJUUBoWNDwBNqVSMNL3lqagB23YdqFGHEnwqgUCae1rbhUklwMy4R1FsrSMMkjhl9hj4siQr4hRYiwloIvV7UQaSJtw1Viax00CsQJhDMRg7xarYXM"
        "XWVgRmnBoscnnnKFENegnEo66+BzONhLHkZZGonoeZzYivkG5eq0hFPC7AY6PUjG8DmQbygRYJxChg8ZQE3gTBXB6Ds1ssa0DFhpNY0oNR4ZrUJAwwwR52sb"
        "nmYIrqltJARkswjc3sOMNjAY+N9aEwIVhtRh1mbDyGxagmUIImMOOovoh4+N4DKrFyJkFqSAJyrMPi01WM+qMEzGIv8PM9Q8aReAA1UAAAAASUVORK5CYII="
    ),
    "SUPPLY_DEPOT": (
        "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"
    ),
}
