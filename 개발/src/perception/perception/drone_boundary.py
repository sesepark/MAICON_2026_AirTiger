"""드론 화면 한 장 → 경기장 경계 (화면 픽셀 ↔ 경기장 좌표 대응).

경기용 main 카메라 화면은 경기 전에 볼 수 없다 → 화면 픽셀 위치를 외우지 않고 **바닥 그림의 모양**만 안다고 보고 찾는다.
  세로 구역선 5개   x = s · W/2−c · W/2 · W/2+c · W−s   (끝 띠 선 · CM 선 · 중앙선 — 중앙선 기준 좌우 대칭)
  가로 줄 경계 2개  y = H/3 · 2H/3
  중앙 원           가운데 (W/2, H/2), 반지름 r
s · c · r 는 모른다(현장 도면 [추정])고 보고 함께 맞춘다. 원이 둥글어야 한다는 조건이 가로 · 세로 축척을 묶는다.

카메라 모델: 렌즈 왜곡(나눗셈 모델) + 바닥 평면 사영 변환 H (보정된 픽셀 → 경기장 m).
  d = p − (cx, cy)                          왜곡 중심
  r² = (dx² + (a·dy)²) / (W_img/2)²          a = 픽셀 가로세로 비 — 연습 카메라는 a ≈ 0.5 (화면이 세로로 늘어나 찍힌다)
  보정 픽셀 q = (cx, cy) + d / (1 + k1·r² + k2·r⁴)
두 단계로 맞춘다: 단순 모델(k1 · 중심 = 화면 가운데 · a = 1) → 확장 모델(k1 · k2 · a · 중심 모두).
  확장 모델이 이상하면(a · 중심 범위 밖, 역변환 불일치) 단순 모델 결과를 쓰고 notes 에 남긴다.
  근거 (2026-11-18 test2_A4: 경기장 곳곳 A4 12장): 단순 모델은 용지 가로 축척이 −18%~+12% 어긋나고(σ 10%),
  확장 모델은 바닥선만으로 σ 4% (용지 측정 자체 오차 ~2.5%). 두 카메라가 따로 낸 끝 띠 · CM · 원 치수 차이도 8 cm → 1.5 cm.
바닥(높이 0)에 그린 선만 쓴다 — 박스 윗면 · UGV 처럼 높이가 있는 것은 여기 대응으로 바로 옮기면 바깥으로 밀린다.

좌표계는 sim-strategy team/navigation/fieldmap.py 와 같다: x 는 BLUE 쪽 벽(0) → RED 쪽 벽, y 는 아래(0) → 위, 1번 줄이 위. 단위 m.
지금은 화면이 대략 「BLUE 왼쪽 · 1번 줄 위」 로 찍힌다고 본다 (연습 카메라 test1 · test2 가 그렇다). 다르게 놓인 화면은 아직 못 다룬다 ✏️.

numpy · OpenCV 만 쓴다. ROS 에 묶이지 않는다.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

FIELD_W = 6.12  # x 방향 (BLUE 벽 → RED 벽) — 공식 자료 「약 612 cm」
FIELD_H = 5.00  # y 방향 — 공식 자료 「약 500 cm」


@dataclass
class BoundaryParams:
    field_w: float = FIELD_W
    field_h: float = FIELD_H
    # 흰 선 픽셀
    tophat_px: int = 9  # 이보다 가는 밝은 것만 (선 굵기 2~5 px)
    tophat_min: int = 35
    sat_max: int = 60  # HSV 채도 — 흰색
    # 세로선 후보
    vert_len_px: int = 41  # 세로로 이만큼 이어져야 세로선 픽셀
    peak_min: float = 30.0  # 열 합(5 px 평균)이 이 이상이어야 봉우리
    sym_tol: float = 0.12  # 좌우 거리 차 / 거리
    inner_ratio: tuple = (0.20, 0.40)  # (CM 선 거리) / (끝 띠 선 거리) 허용 범위 — 화면 픽셀에서. 바닥 그림은 0.6 / 2.2 = 0.27 이고
    #                                    렌즈가 바깥을 눌러 연습 카메라 두 대는 0.295 ~ 0.30 으로 찍힌다. 2026-11-19 까지는 (0.12, 0.55) 라서
    #                                    바닥 깔개의 이음매(가운데에서 ±1.04 m, 흰 선처럼 보인다)를 CM 선으로 고른 조합(0.49)도 후보였다
    # 선 따라가기
    vtrack_px: int = 14  # 세로선 봉우리 둘레 이만큼 안에서 점을 모은다
    htrack_px: int = 40  # 가로선은 왜곡으로 많이 휜다
    run_max_px: int = 5  # 한 줄(행 · 열)에서 선 픽셀이 이보다 많으면 선이 아니다 (글자 · 박스)
    max_combos: int = 40  # 세로선 5개 조합을 이만큼까지 어림해 본다
    max_row_pairs: int = 3  # 조합마다 가로선 짝을 이만큼까지
    refine: int = 3  # 어림 점수 상위 이만큼만 정밀하게 맞춘다
    refine_stop_ok: bool = True  # 정밀 맞추기를 어림 점수 순으로 하나씩 하다가 검사를 모두 통과한 것이 나오면 멈춘다 (2026-11-19)
    # 확장 렌즈 모델이 깨져 단순 모델로 물러난 결과는 「검사를 통과했어도 덜 믿는다」 (2026-11-20, _search). 본선 카메라(2026-11-19 16:55 ~ 18:06 여섯 판,
    # 사진 819장)에서 처음부터 찾기가 두 해로 갈렸다: RED 쪽 끝 띠 선(화면 u ≈ 952) 대신 그 20 px 바깥의 **거점 테두리 선**(u ≈ 972 — R-1 · R-2 · R-3 의
    # 흰 테두리가 세로로 한 줄에 놓인다)을 끝 띠 선으로 고른 조합이 어림 점수가 더 높아 먼저 맞춰졌고, 그 조합은 확장 모델이 깨지지만(픽셀 비 0.0 ~ 0.3)
    # 단순 모델로는 검사를 모두 통과했다 (흰 선 비율 0.87 · 잔차 1.1 ~ 1.4 cm). 그 해는 경기장 오른쪽 1/3 을 0.2 ~ 0.3 m 틀리게 옮긴다 (드론이 본 UGV 자리를
    # 라이다 위치와 견준 것). 맞는 선을 고른 조합은 확장 모델이 연습 카메라와 같은 값(k1 −0.4 · 픽셀 비 0.5)으로 맞고 잔차도 작다 (0.8 ~ 1.0 cm).
    #   - 그런 결과가 나오면 멈추지 않고, **끝 띠 선을 달리 고른** 조합을 더 맞춰 본다: 그 끝 띠 선 곁의 다른 세로선으로 바꾼 것 →
    #     끝 띠 선이 다른 다음 후보 (저마다 full_lens_retry 개까지). 자세한 순서는 _search 의 설명
    #   - 확장 모델로 맞은 결과는 점수에 full_lens_bonus 를 얹어 견준다 — 단순 모델 결과보다 점수가 이만큼 넘게 낮지 않으면 그것을 쓴다
    #   - 끝까지 확장 모델로 맞는 것이 없으면 전처럼 단순 모델 결과를 쓴다 (전보다 나빠지지 않는다)
    # 꺼 두면(False) 2026-11-19 의 동작 그대로다
    prefer_full_lens: bool = True
    full_lens_bonus: float = 0.05
    full_lens_retry: int = 3
    end_line_reach_px: int = 45  # 끝 띠 선에서 이 안의 다른 세로선을 끝 띠 선으로 바꿔 본다 (본선: 거점 테두리 선 ↔ 끝 띠 선 19 ~ 20 px, 1 px ≈ 7 mm)
    # 끝 띠 선 곁에 그보다 이 배수 넘게 **긴** 세로선이 있으면 그 선으로 바꾼 조합을 먼저 맞춰 본다 — 검사를 통과하고 확장 모델로 맞으면 그것을 쓴다.
    # 진짜 구역선은 화면 위에서 아래까지 이어지고, 거점 테두리 · 상자 모서리는 토막이다. 본선 사진 275장: 끝 띠 선은 따라간 점 166 ~ 236개,
    # 그 곁의 거점 테두리 선은 100 ~ 151개 (비 1.33 ~ 1.8, 모든 사진에서). 약한 결과를 기다리지 않는 까닭: 틀린 선을 고르고도 확장 모델이
    # 범위 안으로 맞는 사진이 5 % 있었다 (819장 중 39장 — 렌즈값은 제각각이고 오른쪽이 0.3 m 틀린다). 0 이하 = 쓰지 않음
    longer_line_ratio: float = 1.25
    loose_peak_ratio: float = 0.6  # 그때 · 검사를 통과한 것이 하나도 없을 때 보는 「낮춘 문턱」 = peak_min × 이 값 (30 → 18). 1.0 이상 = 쓰지 않음
    # 한 화면에서 정밀하게 맞춰 보는 횟수의 한도 (모든 단계를 합쳐). 위의 단계들이 붙으면서, 끝내 못 맞추는 화면(경기 중 로봇이 선 위에 올라선 사진)은
    # 20번까지 맞춰 보고 PC 에서 11초가 걸렸다 (전에는 7번 · 4초). 본선 사진 819장: 통과한 795장 중 787장이 8번 안에 끝났고 677장은 1번
    max_refines: int = 8
    # 지난 경계(prior)에서 출발 — 카메라는 경기장 천장에 붙어 있어 경기마다 같다. 이 화면에서 선 점을 다시 모아 다시 맞추고 검사한다
    prior_tol_px: int = 6  # 지난 세로선 자리와 이 화면 봉우리가 이 2배(px)보다 멀면 카메라가 움직인 것으로 보고 처음부터 찾는다
    prior_min_support: float = 0.45  # 다시 맞추기 전 (rough_support_px 로 잰) 흰 선 비율이 이보다 낮으면 카메라가 바뀐 것으로 보고 처음부터 찾는다
    prior_max_change_m: float = 0.08  # 다시 맞춘 끝 띠 · CM 반폭 · 원 반지름이 지난 값과 이보다 다르면 믿지 않는다 (바닥 칠은 그대로다)
    min_line_pts: int = 20
    strip_prior_m: float = 0.90  # 어림 단계의 끝 띠 폭 (m) — 예비 경기장 test1·test2 에서 잰 값 0.88~0.92. 정밀 단계에서 다시 맞춘다
    # 노랑 (HSV, OpenCV 범위) — 거점 테두리 · 줄무늬 박스 · >> 표시
    yellow_h: tuple = (15, 40)
    yellow_s: int = 90
    yellow_v: int = 110
    yellow_weight: float = 3.0
    max_yellow: float = 0.04
    support_px: int = 3  # 모델 선에서 이 거리 안의 흰 선 픽셀을 「맞음」 으로
    rough_support_px: int = 7  # 어림 단계는 선 위치가 몇 cm 어긋나므로 넓게
    min_support: float = 0.75  # 맞게 찾은 연습 사진 305장은 0.84 ~ 1.00 (중앙값 0.99), 틀린 선을 고른 것은 대개 0.5 ~ 0.7 (2026-11-19 까지는 0.45)
    # 맞추기
    huber_m: float = 0.02  # 잔차가 이보다 크면 덜 믿는다
    iters: int = 40
    full_lens: bool = True  # 확장 렌즈 모델(k2 · 픽셀 비 · 중심)까지 맞춘다
    # 확장 모델의 추가 출발점 (k1, k2, 픽셀 비). 연습 카메라 test1 · test2 에서 잰 값 (k1 −0.48~−0.55 · k2 0.36~0.51 · 비 0.50~0.54).
    # 경기용 카메라가 같은 기종이면 도움이 되고, 아니어도 단순 모델 출발점이 따로 있으니 손해는 없다.
    lens_priors: tuple = ((-0.5, 0.4, 0.52),)
    aspect_range: tuple = (0.3, 1.6)
    center_tol: float = 0.25  # 왜곡 중심이 화면 가운데에서 화면 크기의 이 비율 안. 연습 사진 308장은 가로 31 px · 세로 35 px 안이다.
    #                           화면이 한쪽으로 잘려 오면(본선 경기장은 좌우가 잘릴 수 있다) 렌즈 가운데가 그만큼 옮겨진다 → 0.15 에서 넓힘 (2026-11-19).
    #                           이보다 멀어서 확장 모델을 버리게 되면 단순 모델로 넘어가지 않고 실패로 둔다 (_finish) — 단순 모델은 중심을
    #                           화면 가운데로 두므로, 연습 사진을 20 % 민 것에서 검사는 통과하고 귀퉁이가 14 ~ 17 cm 틀렸다
    max_inverse_px: float = 0.5  # 화면 테두리 점을 보정 → 되돌렸을 때 이 이상 어긋나면 확장 모델을 버린다
    # 검사 (통과해야 ok)
    max_rms_m: float = 0.03
    min_points: int = 400
    # 바닥 그림의 치수 — 모든 경기장이 같은 그림이다 (20 cm 격자. 가운데에서 끝 띠 선 ±2.2 m · CM 선 ±0.6 m · 줄 경계 ±0.8 m · 원 반지름 0.6 m).
    # 맞춤 좌표는 줄 경계를 ±0.8333 에 두므로 값이 2 ~ 4 % 크게 나온다. 연습 사진 수백 장에서 맞게 찾은 값:
    #   끝 띠 0.79 ~ 0.89 · CM 반폭 0.598 ~ 0.625 · 원 반지름 0.614 ~ 0.636 · 원 / CM 1.02 ~ 1.03 · (W/2 − 끝 띠) / CM 3.60 ~ 3.65
    # 틀린 선을 고른 것은 여기서 벗어난다: 이음매를 CM 선으로 고르면 CM 반폭 1.05, 상자 모서리를 줄 경계로 고르면 원 / CM 이 1.15.
    # 2026-11-19 까지는 (0.3 ~ 1.2 · 0.4 ~ 1.2 · 0.4 ~ 1.2) 라서 그런 것이 검사를 통과했다 (로봇에서 한 번, PC 에서 한 번)
    s_range: tuple = (0.75, 0.96)
    c_range: tuple = (0.56, 0.67)
    r_range: tuple = (0.57, 0.68)
    circle_cm_range: tuple = (0.97, 1.08)  # 원 반지름 / CM 반폭
    span_cm_range: tuple = (3.40, 3.85)  # (W/2 − 끝 띠) / CM 반폭 — 축척과 무관한 비율
    rough_c_range: tuple = (0.50, 0.72)  # 어림 단계의 CM 반폭 (끝 띠를 strip_prior_m 로 두고 잰 값) — 벗어나면 그 조합은 정밀하게 맞추지 않는다
    # 다시 찾기 — 첫 탐색이 통과하지 못하면 세로선 조합 몇 개에 가로선 짝을 모두 대 본다 (가로선은 상자에 가려 짧게 잡히면 길이 순 3개에 못 든다)
    retry_combos: int = 3
    retry_refine: int = 4


@dataclass(frozen=True)
class Lens:
    """렌즈 왜곡 (나눗셈 모델). n = (화면 가로/2)² — r 을 화면 크기에 무관하게."""

    k1: float = 0.0
    k2: float = 0.0
    aspect: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    n: float = 1.0

    @staticmethod
    def simple(k1, size):
        ww, hh = size
        return Lens(float(k1), 0.0, 1.0, ww / 2.0, hh / 2.0, (ww / 2.0) ** 2)

    def vector(self):
        return np.array([self.k1, self.k2, self.aspect, self.cx, self.cy])

    def with_vector(self, v):
        return Lens(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), self.n)


@dataclass
class Boundary:
    ok: bool
    k: float = 0.0  # 렌즈 k1 (예전 이름 그대로)
    hom: np.ndarray | None = None  # 보정 픽셀 → 경기장 (m)
    image_size: tuple = (0, 0)  # (가로, 세로) px
    lens: Lens | None = None  # None 이면 Lens.simple(k, image_size)
    strip: float | None = None  # s: 벽 → 끝 띠 선 (m)
    cm_half: float | None = None  # c: 중앙선 → CM 선 (m)
    circle_r: float | None = None
    rms_m: float | None = None  # 맞춘 뒤 바닥선 점들의 잔차 RMS (m)
    points: int = 0
    field_w: float = FIELD_W
    field_h: float = FIELD_H
    notes: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)  # 찾은 선 점들 (그림 그리기용)

    # ---- 대응
    def _lens(self):
        return self.lens if self.lens is not None else Lens.simple(self.k, self.image_size)

    def pixel_to_field(self, uv):
        """원본 화면 픽셀 [(u, v)] → 경기장 [(x, y)] m (바닥 높이 0)."""
        return _apply_h(self.hom, _undistort(np.asarray(uv, float).reshape(-1, 2), self._lens()))

    def field_to_pixel(self, xy):
        """경기장 [(x, y)] m → 원본 화면 픽셀 [(u, v)] (화면 밖일 수 있다)."""
        q = _apply_h(np.linalg.inv(self.hom), np.asarray(xy, float).reshape(-1, 2))
        return _distort(q, self._lens())

    def corners_px(self):
        """경기장 바닥 네 모서리 (0,0) · (W,0) · (W,H) · (0,H) 의 화면 픽셀."""
        w, h = self.field_w, self.field_h
        return self.field_to_pixel([(0, 0), (w, 0), (w, h), (0, h)])

    @staticmethod
    def from_dict(d):
        """to_dict() 로 저장한 것 → Boundary (지난 경계를 prior 로 다시 쓰려고). 대응(hom · lens)이 없으면 None."""
        if not d or d.get("hom") is None:
            return None
        size = tuple(int(v) for v in d.get("image_size") or (0, 0))
        lz = d.get("lens")
        lens = Lens(lz["k1"], lz["k2"], lz["aspect"], lz["cx"], lz["cy"], lz["n"]) if lz else Lens.simple(d.get("k", 0.0), size)
        return Boundary(bool(d.get("ok")), k=float(d.get("k", lens.k1)), hom=np.asarray(d["hom"], float), image_size=size, lens=lens,
                        strip=d.get("strip"), cm_half=d.get("cm_half"), circle_r=d.get("circle_r"), rms_m=d.get("rms_m"),
                        points=int(d.get("points") or 0), field_w=float(d.get("field_w", FIELD_W)), field_h=float(d.get("field_h", FIELD_H)))

    def to_dict(self):
        lens = self._lens() if self.hom is not None else None
        return {
            "ok": self.ok, "k": self.k, "hom": None if self.hom is None else self.hom.tolist(), "image_size": list(self.image_size),
            "lens": None if lens is None else {"k1": lens.k1, "k2": lens.k2, "aspect": lens.aspect, "cx": lens.cx, "cy": lens.cy, "n": lens.n},
            "strip": self.strip, "cm_half": self.cm_half, "circle_r": self.circle_r, "rms_m": self.rms_m, "points": self.points,
            "field_w": self.field_w, "field_h": self.field_h, "notes": list(self.notes),
            "corners_px": None if self.hom is None else np.round(self.corners_px(), 1).tolist(),
        }


# ------------------------------------------------------------------ 기하
def _factor(d, lens):
    r2 = (d[:, 0] ** 2 + (lens.aspect * d[:, 1]) ** 2)[:, None] / lens.n
    return 1.0 + lens.k1 * r2 + lens.k2 * r2 * r2


def _undistort(p, lens):
    c = np.array([lens.cx, lens.cy])
    d = p - c
    return c + d / _factor(d, lens)


def _distort(q, lens):
    """_undistort 의 역. 화면 밖이라 해가 없는 점은 NaN.

    보정은 중심에서 같은 방향으로 늘이거나 줄이기만 한다 → d = t·e (e = q − c) 로 두면 1차원 방정식
        g(t) = t − f(t·e) = t − 1 − k1·ρ·t² − k2·ρ²·t⁴ = 0,   ρ = r²(e)
    의 가장 작은 양의 근이다 (g(0) = −1). 뉴턴법으로 풀고, 근이 g′ > 0 (첫 근) 이 아니면 해 없음.
    """
    c = np.array([lens.cx, lens.cy])
    e = np.asarray(q, float) - c
    rho = (e[:, 0] ** 2 + (lens.aspect * e[:, 1]) ** 2) / lens.n
    k1, k2 = lens.k1, lens.k2
    if k2 == 0.0:
        # 단순 모델은 2차 방정식 k1·ρ·t² − t + 1 = 0 → 작은 근 t = 2 / (1 + √(1 − 4·k1·ρ)) (뉴턴법과 같은 근, 훨씬 빠르다).
        # 판별식 ≤ 0 이면 첫 근에서 g′ ≤ 0 (접힘) 이라 뉴턴법 쪽과 같이 해 없음.
        disc = 1.0 - 4.0 * k1 * rho
        with np.errstate(all="ignore"):
            t = 2.0 / (1.0 + np.sqrt(np.maximum(disc, 0.0)))
        out = c + e * t[:, None]
        out[~(disc > 0) | ~np.isfinite(t)] = np.nan
        return out
    t = np.ones(len(e))
    with np.errstate(all="ignore"):
        for _ in range(40):
            g = t - 1 - k1 * rho * t ** 2 - k2 * rho ** 2 * t ** 4
            dg = 1 - 2 * k1 * rho * t - 4 * k2 * rho ** 2 * t ** 3
            step = g / np.where(np.abs(dg) < 1e-9, 1e-9, dg)
            t = t - np.clip(step, -0.5, 0.5)
            if np.nanmax(np.abs(step)) < 1e-10:
                break
        g = t - 1 - k1 * rho * t ** 2 - k2 * rho ** 2 * t ** 4
        dg = 1 - 2 * k1 * rho * t - 4 * k2 * rho ** 2 * t ** 3
        bad = ~np.isfinite(t) | (t <= 0) | (np.abs(g) > 1e-6) | (dg <= 0)
    out = c + e * t[:, None]
    out[bad] = np.nan
    return out


def _apply_h(h, p):
    w = p[:, 0] * h[2, 0] + p[:, 1] * h[2, 1] + h[2, 2]
    return np.stack([(p[:, 0] * h[0, 0] + p[:, 1] * h[0, 1] + h[0, 2]) / w, (p[:, 0] * h[1, 0] + p[:, 1] * h[1, 1] + h[1, 2]) / w], 1)


# ------------------------------------------------------------------ 파일
def read_image(path):
    """그림 파일 → BGR. cv2.imread 는 Windows 에서 한글 경로(「개발」 등)를 못 읽는다 → 바이트로 읽어 푼다. 못 읽으면 None."""
    import cv2

    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path, img):
    """BGR → 그림 파일 (한글 경로 가능). 확장자로 형식을 정한다."""
    import os

    import cv2

    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if ok:
        buf.tofile(path)
    return bool(ok)


# ------------------------------------------------------------------ 흰 선
def line_mask(bgr, p):
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (p.tophat_px, p.tophat_px)))
    return ((th > p.tophat_min) & (hsv[..., 1] < p.sat_max)).astype(np.uint8)


def _vertical_peaks(mask, p, peak_min=None):
    """세로선 후보 열 [(u, 열 합)]. peak_min: 봉우리로 치는 열 합의 문턱 (None 이면 p.peak_min)."""
    import cv2

    peak_min = p.peak_min if peak_min is None else float(peak_min)

    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, p.vert_len_px)))
    wide = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    vert &= 1 - wide
    prof = np.convolve(vert.sum(0).astype(float), np.ones(5) / 5.0, "same")
    peaks = []
    for u in range(3, len(prof) - 3):
        if prof[u] >= peak_min and prof[u] == prof[u - 3:u + 4].max():
            if peaks and u - peaks[-1][0] <= 6:  # 평평한 봉우리는 하나로
                if prof[u] > peaks[-1][1]:
                    peaks[-1] = (u, prof[u])
                continue
            peaks.append((u, float(prof[u])))
    return peaks


def _five_line_candidates(peaks, width, p, limit=8, length=None):
    """봉우리 중 좌우 대칭인 5개 조합 (끝 띠 · CM · 중앙 · CM · 끝 띠). 가장 짧은 선의 길이(따라간 점 수)가 긴 순.

    length: {u: 그 봉우리에서 따라간 선 점 수}. 진짜 구역선은 화면 위아래로 길게 이어지고 무늬 · 박스 모서리는 짧다.
    """
    us = [u for u, _ in peaks]
    st = length if length is not None else {u: s for u, s in peaks}
    out = []
    for c in us:
        left = [u for u in us if u < c]
        right = [u for u in us if u > c]
        for b1, b2 in itertools.product(left, right):
            db1, db2 = c - b1, c + 0 - b2
            db2 = b2 - c
            if abs(db1 - db2) > p.sym_tol * max(db1, db2):
                continue
            for a1, a2 in itertools.product([u for u in left if u < b1], [u for u in right if u > b2]):
                da1, da2 = c - a1, a2 - c
                if abs(da1 - da2) > p.sym_tol * max(da1, da2):
                    continue
                ratio = (db1 + db2) / float(da1 + da2)
                if not p.inner_ratio[0] <= ratio <= p.inner_ratio[1]:
                    continue
                if da1 + da2 < 0.25 * width:
                    continue
                score = min(st[a1], st[a2], st[b1], st[b2], st[c])
                out.append((score, (a1, b1, c, b2, a2)))
    out.sort(reverse=True)
    return [combo for _s, combo in out[:limit]]


def _five_ok(combo, width, p):
    """세로선 5개 (끝 띠 · CM · 중앙 · CM · 끝 띠) 가 _five_line_candidates 의 조건(순서 · 좌우 대칭 · 안팎 비 · 폭)에 맞는가."""
    a1, b1, c, b2, a2 = combo
    if not a1 < b1 < c < b2 < a2:
        return False
    db1, db2, da1, da2 = c - b1, b2 - c, c - a1, a2 - c
    if abs(db1 - db2) > p.sym_tol * max(db1, db2) or abs(da1 - da2) > p.sym_tol * max(da1, da2):
        return False
    return p.inner_ratio[0] <= (db1 + db2) / float(da1 + da2) <= p.inner_ratio[1] and da1 + da2 >= 0.25 * width


def _robust_poly(a, b, tols=(30, 15, 8, 5, 4)):
    keep = np.ones(len(a), bool)
    for tol in tols:
        if keep.sum() < 8:
            return keep & False
        coef = np.polyfit(a[keep], b[keep], 2)
        keep = np.abs(b - np.polyval(coef, a)) < tol
    return keep


def _window_means(win, run_max):
    """win[i] (0/1 한 줄) 마다 1 인 칸 수와 그 칸 번호 평균. 칸 수가 1 ~ run_max 인 줄만 → (고른 줄 번호, 평균)."""
    win = win.astype(np.int32)
    cnt = win.sum(1)
    sel = (cnt > 0) & (cnt <= run_max)
    pos = (win[sel] * np.arange(win.shape[1])).sum(1) / cnt[sel]
    return np.nonzero(sel)[0], pos


def _track_vertical(mask, u0, v_range, p):
    """세로선 봉우리 u0 둘레에서 행마다 선 점 하나 → [(u, v)]."""
    hh, ww = mask.shape
    lo, hi = max(0, u0 - p.vtrack_px), min(ww, u0 + p.vtrack_px + 1)
    vs = np.arange(v_range[0], v_range[1], 2)
    idx, pos = _window_means(mask[vs, lo:hi], p.run_max_px)
    pts = np.stack([lo + pos, vs[idx]], 1).astype(float).reshape(-1, 2)
    if len(pts) < 8:
        return pts[:0]
    return pts[_robust_poly(pts[:, 1], pts[:, 0])]


def _track_horizontal(mask, v0, u_range, p):
    hh, ww = mask.shape
    lo, hi = max(0, v0 - p.htrack_px), min(hh, v0 + p.htrack_px + 1)
    us = np.arange(u_range[0], u_range[1], 2)
    idx, pos = _window_means(mask[lo:hi, us].T, p.run_max_px)
    pts = np.stack([us[idx], lo + pos], 1).astype(float).reshape(-1, 2)
    if len(pts) < 8:
        return pts[:0]
    return pts[_robust_poly(pts[:, 0], pts[:, 1])]


def _horizontal_peaks(mask, u_lo, u_hi, p):
    """두 바깥 세로선 사이에서 가로선 후보 행 → [(v, 강도)] 강도 순."""
    import cv2

    band = mask[:, u_lo:u_hi]
    horiz = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (p.vert_len_px, 1)))
    tall = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    horiz &= 1 - tall
    prof = np.convolve(horiz.sum(1).astype(float), np.ones(9) / 9.0, "same")
    out = []
    for v in range(5, len(prof) - 5):
        if prof[v] > 0 and prof[v] == prof[v - 5:v + 6].max():
            if out and v - out[-1][0] <= 10:
                continue
            out.append((v, float(prof[v])))
    out.sort(key=lambda t: -t[1])
    return out


# ------------------------------------------------------------------ 맞추기
#   theta = [k1, k2, aspect, cx, cy, h11..h32 (8), s, c, r]  (16)
_SIMPLE = [0] + list(range(5, 16))  # 단순 모델에서 푸는 변수 (k1 · H · s · c · r)
_FULL = list(range(16))


def _unpack(theta, n):
    return Lens(*map(float, theta[:5]), n), np.append(theta[5:13], 1.0).reshape(3, 3), theta[13], theta[14], theta[15]


def _pack(lens, hom, s, c, r):
    return np.concatenate([lens.vector(), (hom / hom[2, 2]).ravel()[:8], [s, c, r]])


_LINE_CODE = {"Bs": 0, "Bc": 1, "C": 2, "Rc": 3, "Rs": 4, "top": 5, "bot": 6}  # 그 밖(circle)은 7


class _Packed:
    """선 묶음 {이름: 점} 을 한 배열로 (순서는 dict 순서 그대로). 맞추기 동안 묶음마다 따로 보정 · 사영하지 않게."""

    def __init__(self, groups):
        groups = {k: np.asarray(v, float).reshape(-1, 2) for k, v in groups.items()}
        self.pts = np.concatenate(list(groups.values())) if groups else np.zeros((0, 2))
        self.code = np.concatenate([np.full(len(v), _LINE_CODE.get(k, 7)) for k, v in groups.items()]) if groups else np.zeros(0, int)
        self.is_x = self.code < 5
        self.is_y = (self.code >= 5) & (self.code < 7)
        self.is_c = self.code == 7


def _model_residuals(theta, groups, n, p, q=None):
    """바닥선 점 → 경기장으로 옮겨 모델 선까지의 거리 (m). groups 는 dict 또는 _Packed. q: 이미 보정한 점 (렌즈가 같을 때)."""
    pk = groups if isinstance(groups, _Packed) else _Packed(groups)
    lens, hom, s, c, r = _unpack(theta, n)
    w, h = p.field_w, p.field_h
    xy = _apply_h(hom, _undistort(pk.pts, lens) if q is None else q)
    target = np.array([s, w / 2 - c, w / 2, w / 2 + c, w - s, 2 * h / 3, h / 3, 0.0])[pk.code]
    res = np.empty(len(pk.pts))
    res[pk.is_x] = xy[pk.is_x, 0] - target[pk.is_x]
    res[pk.is_y] = xy[pk.is_y, 1] - target[pk.is_y]
    cc = pk.is_c
    res[cc] = np.hypot(xy[cc, 0] - w / 2, xy[cc, 1] - h / 2) - r
    return res


def _fit(theta, free, groups, n, p):
    """Huber 가중 Levenberg–Marquardt (수치 미분). free 에 든 변수만 푼다."""
    free = np.asarray(free)
    lam = 1e-3
    groups = groups if isinstance(groups, _Packed) else _Packed(groups)

    def full(x):
        t = theta.copy()
        t[free] = x
        return t

    x = theta[free].copy()
    scale = np.maximum(np.abs(x) * 1e-5, 1e-7)

    def cost(xx):
        r = _model_residuals(full(xx), groups, n, p)
        a = np.abs(r)
        wgt = np.where(a <= p.huber_m, 1.0, p.huber_m / np.maximum(a, 1e-12))
        return r, wgt, float((wgt * r * r).sum())

    r, wgt, cur = cost(x)
    for _ in range(p.iters):
        jac = np.empty((len(r), len(x)))
        q = None  # 렌즈 변수(0 ~ 4)를 건드리지 않는 열은 보정한 점이 같다 → 한 번만 보정
        for i in range(len(x)):
            x2 = x.copy()
            x2[i] += scale[i]
            if free[i] >= 5:
                if q is None:
                    q = _undistort(groups.pts, _unpack(full(x), n)[0])
                jac[:, i] = (_model_residuals(full(x2), groups, n, p, q) - r) / scale[i]
            else:
                jac[:, i] = (_model_residuals(full(x2), groups, n, p) - r) / scale[i]
        jw = jac * wgt[:, None]
        a = jac.T @ jw
        g = jw.T @ r
        improved = False
        step = np.zeros_like(x)
        for _try in range(12):
            try:
                step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r2, w2, c2 = cost(x + step)
            if c2 < cur:
                x, r, wgt, cur = x + step, r2, w2, c2
                lam = max(lam / 3, 1e-12)
                improved = True
                break
            lam *= 10
        if not improved or np.abs(step / np.maximum(np.abs(x), 1e-6)).max() < 1e-7:
            break
    return full(x), r


def _field_region_px(theta, n, size, w, h, margin=0.3):
    """단순 모델로 본 경기장(+ margin m) 테두리를 화면 안으로 자른 점들 — 확장 모델을 검사할 영역.

    확장 모델은 바닥선이 있는 곳에서만 맞춘 것이라, 화면 귀퉁이(경기장 밖)에서는 접힐 수 있다 (연습 카메라: 귀퉁이 r² ≈ 1.08 에서 접힘).
    그곳은 쓰지 않으므로 검사하지 않는다.
    """
    lens, hom, *_ = _unpack(theta, n)
    ww, hh = size
    t = np.linspace(0, 1, 41)
    x0, x1, y0, y1 = -margin, w + margin, -margin, h + margin
    rim = np.concatenate([np.stack([x0 + t * (x1 - x0), 0 * t + y0], 1), np.stack([x0 + t * (x1 - x0), 0 * t + y1], 1),
                          np.stack([0 * t + x0, y0 + t * (y1 - y0)], 1), np.stack([0 * t + x1, y0 + t * (y1 - y0)], 1)])
    px = _distort(_apply_h(np.linalg.inv(hom), rim), lens)
    px = px[np.isfinite(px).all(1)]
    return np.stack([np.clip(px[:, 0], 0, ww - 1), np.clip(px[:, 1], 0, hh - 1)], 1)


def _robust_cost(r, p):
    a = np.abs(r)
    return float(np.where(a <= p.huber_m, a * a, 2 * p.huber_m * a - p.huber_m ** 2).sum())


def _lens_sane(lens, size, p, region):
    """확장 렌즈 모델이 믿을 만한가: 픽셀 비 · 중심 범위, 경기장 영역 테두리에서 보정이 접히지 않고 되돌리기가 맞는가."""
    ww, hh = size
    if not p.aspect_range[0] <= lens.aspect <= p.aspect_range[1]:
        return "픽셀 비 %.2f 범위 밖" % lens.aspect
    if abs(lens.cx - ww / 2) > p.center_tol * ww or abs(lens.cy - hh / 2) > p.center_tol * hh:
        return "왜곡 중심 (%.0f, %.0f) 이 화면 가운데에서 너무 멂" % (lens.cx, lens.cy)
    edge = region
    if len(edge) < 8:
        return "검사할 경기장 영역이 없음"
    f = _factor(edge - [lens.cx, lens.cy], lens)
    if (f <= 0.05).any():
        return "경기장 영역 안에서 보정이 뒤집힘"
    # 반지름을 따라 보정 반지름이 늘어나야 한다 (줄어들면 접힌다)
    rad = np.linspace(0, 1.0, 50)[:, None, None] * (edge - [lens.cx, lens.cy])[None]
    ru = np.linalg.norm(rad / _factor(rad.reshape(-1, 2), lens).reshape(rad.shape[:2] + (1,)), axis=2)
    if (np.diff(ru, axis=0) < -1e-6).any():
        return "보정이 경기장 영역 안에서 접힘"
    back = _distort(_undistort(edge, lens), lens)
    err = float(np.nanmax(np.abs(back - edge))) if np.isfinite(back).all() else math.inf
    if err > p.max_inverse_px:
        return "역변환 불일치 %.2f px" % err
    return None


def _line_fit(pts, lens):
    q = _undistort(pts, lens)
    m = q.mean(0)
    _u, _s, vt = np.linalg.svd(q - m, full_matrices=False)
    return m, vt[0]


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    t = np.linalg.solve(np.array([d, -e]).T, q - p)
    return p + t[0] * d


class _Scene:
    """화면 한 장에서 후보마다 되풀이되는 계산을 한 번만 한다 (선 픽셀 좌표 · 넓힌 마스크 · 선마다의 직선성).

    detect_boundary 한 번 동안만 쓴다. 결과는 따로 계산할 때와 같다 — 계산 순서만 바꾼다.
    """

    def __init__(self, mask, yellow):
        self.mask = mask
        self.yellow = yellow
        vs, us = np.nonzero(mask)
        self.pts = np.stack([us, vs], 1).astype(float)
        self._near = {}
        self._plumb = {}  # id(선 점 배열) → (그 배열, k 49개 각각의 직선성 오차)
        self._simple = None

    def near(self, px):
        if px not in self._near:
            import cv2

            self._near[px] = cv2.dilate(self.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))
        return self._near[px]

    def undistorted_pts(self, lens):
        """선 픽셀 전부를 보정. 단순 모델(중심 = 화면 가운데 · 비 1 · k2 0)은 중심 거리를 한 번만 구해 둔다."""
        if lens.k2 == 0.0 and lens.aspect == 1.0:
            if self._simple is None or self._simple[0] != (lens.cx, lens.cy, lens.n):
                c = np.array([lens.cx, lens.cy])
                d = self.pts - c
                r2 = (d[:, 0] ** 2 + (1.0 * d[:, 1]) ** 2)[:, None] / lens.n
                self._simple = ((lens.cx, lens.cy, lens.n), c, d, r2)
            _key, c, d, r2 = self._simple
            return c + d / (1.0 + lens.k1 * r2 + 0.0 * r2 * r2)
        return _undistort(self.pts, lens)

    def plumb_errors(self, pts, size):
        key = id(pts)
        hit = self._plumb.get(key)
        if hit is None or hit[0] is not pts:
            hit = (pts, _plumb_errors(pts, size))
            self._plumb[key] = hit
        return hit[1]


_PLUMB_K = np.linspace(-0.6, 0.6, 49)


def _plumb_errors(pts, size):
    """선 하나를 k 49개로 각각 보정했을 때 직선에서 벗어난 정도 (작은 특이값² / 점 수) — k 마다 한 값.

    2차원 점들의 작은 특이값² 은 공분산(2×2)의 작은 고윳값이라 SVD 없이 한 번에 구한다.
    """
    ww, hh = size
    c = np.array([ww / 2.0, hh / 2.0])
    d = np.asarray(pts, float) - c
    r2 = (d[:, 0] ** 2 + d[:, 1] ** 2) / (ww / 2.0) ** 2
    f = 1.0 / (1.0 + _PLUMB_K[:, None] * r2[None, :])  # (49, 점)
    qx, qy = d[:, 0] * f, d[:, 1] * f
    qx = qx - qx.mean(1, keepdims=True)
    qy = qy - qy.mean(1, keepdims=True)
    sxx, syy, sxy = (qx * qx).sum(1), (qy * qy).sum(1), (qx * qy).sum(1)
    lam = (sxx + syy) / 2.0 - np.sqrt(((sxx - syy) / 2.0) ** 2 + sxy ** 2)
    return np.maximum(lam, 0.0) / len(d)


def _plumb_k(groups, size, scene=None):
    """선들이 가장 곧게 펴지는 단순 렌즈 k1 (−0.6 ~ 0.6, 49칸). 선마다의 값은 scene 에 기억해 두고 더하기만 한다."""
    e = sum(scene.plumb_errors(pts, size) if scene is not None else _plumb_errors(pts, size) for pts in groups.values())
    return float(_PLUMB_K[int(np.argmin(e))])


def _circle_points(mask, groups, lens, hom, p, scene=None):
    """중앙선 · CM 선 근처를 뺀 가운데 영역의 선 픽셀 중 경기장 가운데에서 같은 거리에 몰린 것."""
    if scene is not None:
        pts = scene.pts
        xy = _apply_h(hom, scene.undistorted_pts(lens))
    else:
        vs, us = np.nonzero(mask)
        pts = np.stack([us, vs], 1).astype(float)
        xy = _apply_h(hom, _undistort(pts, lens))
    w, h = p.field_w, p.field_h
    d = np.hypot(xy[:, 0] - w / 2, xy[:, 1] - h / 2)
    near_center_line = np.abs(xy[:, 0] - w / 2) < 0.05
    sel = (d > 0.3) & (d < 1.3) & ~near_center_line & (np.abs(xy[:, 1] - h / 2) < h / 6 + 0.1)
    if sel.sum() < 30:
        return pts[:0], None
    hist, edges = np.histogram(d[sel], bins=np.arange(0.3, 1.31, 0.02))
    r0 = float(edges[np.argmax(hist)] + 0.01)
    keep = sel & (np.abs(d - r0) < 0.04)
    return pts[keep][::2], r0


# ------------------------------------------------------------------ 본체
def detect_boundary(bgr, params=None, prior=None):
    """드론 화면(BGR uint8) → Boundary. 못 찾으면 ok=False 와 notes.

    prior: 지난번에 찾은 Boundary (또는 그 to_dict(), 여럿이면 목록). 주면 먼저 그 모델에서 출발해 이 화면의 선 점으로 다시 맞추고,
      검사를 모두 통과하면 그것을 낸다 (debug["via"] = "prior", 한 장에 1초 안쪽). 통과하지 못하면 아래 탐색을 처음부터 한다.

    두 단계로 고른다.
      1) 어림: 좌우 대칭인 세로선 5개 조합 × 가로선 짝마다, 직선성으로 k · 교점 넷으로 H 를 어림하고 점수를 매긴다 (빠름).
      2) 정밀: 점수가 높은 몇 개만 12 변수(k · H · s · c · r)를 함께 맞추고, 다시 점수를 매겨 가장 높은 것.
    점수 = 모델 선 위 흰 선 비율 − yellow_weight × 구역 안 노란 비율.
      거점 테두리 · 노란 줄무늬 박스 · >> 표시는 도면상 벽과 끝 띠 선 사이에만 있다 → 구역 안이 노랗다면 끝 띠 선을 잘못 고른 것.
    """
    import cv2

    p = params or BoundaryParams()
    hh, ww = bgr.shape[:2]
    size = (ww, hh)
    mask = line_mask(bgr, p)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[..., 0] >= p.yellow_h[0]) & (hsv[..., 0] <= p.yellow_h[1]) & (hsv[..., 1] >= p.yellow_s) & (hsv[..., 2] >= p.yellow_v))
    scene = _Scene(mask, yellow)
    prior_notes = []
    priors = prior if isinstance(prior, (list, tuple)) else ([] if prior is None else [prior])
    weak = None  # 지난 경계의 선 조합으로 검사는 통과했지만 확장 렌즈 모델이 깨진 결과 — 처음부터 찾기가 더 나은 것을 못 내면 이것을 쓴다
    for index, pr in enumerate(priors):
        b, why = _from_prior(scene, pr, size, p)
        if b is not None and b.ok:
            b.debug["via"] = "prior"
            b.debug["prior_index"] = index  # 받은 목록에서 몇 번째 것이 맞았는가 (부르는 쪽이 저장 파일 것인지 코드에 실린 것인지 가린다)
            b.notes.append("지난 경계로 선 조합을 골라 맞춤")
            if p.prefer_full_lens and p.full_lens and b.debug.get("lens_full") is False:
                # 2026-11-20: 지난 경계가 틀린 선(본선 카메라의 거점 테두리 선)으로 맞춘 것이었으면 그 조합을 그대로 다시 고른다 → 바로 내지 않고
                # 처음부터 찾아 본다 (_search 는 그 곁의 더 긴 선을 본다). 그것이 확장 모델로 맞으면 그것을, 아니면 전처럼 이것을 낸다
                if weak is None:
                    weak = b
                prior_notes.append("지난 경계로 고른 선 조합은 확장 렌즈 모델이 깨짐 — 처음부터 찾아 봄")
                continue
            return b
        prior_notes.append("지난 경계로 고른 선 조합 안 맞음: %s" % why)
        if why is not None and why.startswith("지난 ") and "자리에" in why:  # 선 하나가 가려져 봉우리가 없을 때
            b, why2 = _refit_prior(scene, pr, size, p)
            if b is not None and b.ok:
                b.debug["via"] = "prior_refit"
                b.debug["prior_index"] = index
                b.notes.append("지난 경계 둘레의 흰 선 점으로 다시 맞춤 (" + prior_notes.pop() + ")")
                return b
            prior_notes.append("지난 경계 둘레로 다시 맞추기 안 됨: %s" % why2)
    best = _search(scene, size, p)
    best = _reduced_if_failed(scene, size, p, best)  # 끝 띠 선이 화면 밖일 때 (평가 경기장) — 위가 실패했을 때만 해 본다
    if weak is not None and not (best.ok and best.debug.get("lens_full", True)):
        weak.notes.append("처음부터 찾기가 더 나은 것을 내지 못함 — 지난 경계의 선 조합 (단순 렌즈) 을 씀")
        return weak
    best.notes += prior_notes
    best.debug["via"] = "search"
    return best


def _search(scene, size, p):
    """세로선 · 가로선 조합을 모두 어림해 보고 상위 몇 개만 정밀하게 (prior 없이 처음부터).

    2026-11-20 — 확장 렌즈 모델이 깨져 단순 모델로 물러난 결과(「약한 결과」)는 검사를 통과했어도 거기서 멈추지 않는다 (BoundaryParams.prefer_full_lens 의 설명):
      0) 후보를 맞추기 전에, 그 끝 띠 선 곁에 훨씬 긴 세로선(longer_line_ratio)이 있으면 그 선으로 바꾼 조합을 먼저 맞춰 본다 (약한 결과가 아니어도)
      1) 약한 결과가 나오면 그 끝 띠 선 곁(end_line_reach_px)에 있는 다른 세로선으로 끝 띠 선만 바꾼 조합을 바로 맞춰 본다 — 낮춘 문턱의 봉우리까지 본다
         (본선 사진: 진짜 끝 띠 선은 가늘고 조금 휘어 열 합이 문턱 30 언저리 22 ~ 46 이라, 사진에 따라 봉우리로 뽑히지 않는다. 따라간 점은 190 ~ 220개로 가장 길다)
      2) 그래도 없으면 끝 띠 선이 다른 어림 후보를 full_lens_retry 개까지 더 맞춰 본다
      3) 검사를 통과한 것이 하나도 없으면 낮춘 문턱의 봉우리로 조합을 다시 만들어 본다 — 여기서는 확장 모델로 맞은 것만 받는다
         (약한 결과는 받지 않는다: 그럴 바에는 전처럼 축소 맞추기로 넘어가는 것이 낫다)
    """
    mask = scene.mask
    ww, hh = size
    prefer = p.prefer_full_lens and p.full_lens
    peaks = _vertical_peaks(mask, p)
    vtracks = {}

    def track(u):
        if u not in vtracks:
            vtracks[u] = _track_vertical(mask, u, (0, hh), p)
        return vtracks[u]

    loose = []  # 낮춘 문턱의 봉우리 (처음 쓸 때 한 번 만든다)

    def loose_us():
        if not loose:
            loose.append([u for u, _s in _vertical_peaks(mask, p, p.peak_min * p.loose_peak_ratio)])
        return loose[0]

    rough = []  # [(어림 점수, id, 어림한 Boundary)] — 어느 단계에서 만들었든 모두
    known = {}  # (세로선 조합, 가로선 짝) → 어림한 Boundary
    htrack_cache = {}

    def roughs(combo_list, every=False):
        """세로선 조합마다 가로선 짝을 대어 어림한 것들 (점수 순). 앞서 어림한 것은 다시 쓴다."""
        out = []
        for combo in combo_list:
            groups = {nm: track(u) for nm, u in zip(_VNAMES, combo)}
            if any(len(g) < p.min_line_pts for g in groups.values()):
                continue
            for pair, top, bot in _row_pairs(mask, combo, htrack_cache, hh, p, every=every):
                b = known.get((combo, pair))
                if b is None:
                    b = _rough(scene, dict(groups, top=top, bot=bot), combo, pair, size, p)
                    if b is None:
                        continue
                    known[(combo, pair)] = b
                    rough.append((b.debug["score"], id(b), b))
                out.append((b.debug["score"], id(b), b))
        out.sort(key=lambda t: -t[0])
        return out

    best = None
    tried = []
    refined = set()
    weak_ends = set()  # 약한 결과의 끝 띠 선 짝 (Bs, Rs) — 같은 끝 띠 선을 쓰는 후보는 다시 맞추지 않는다
    extra = [0]  # 약한 결과가 나온 뒤 더 맞춰 봐도 되는 후보 수

    def rank(b):
        full = prefer and bool(b.debug.get("lens_full"))
        return (b.ok, b.debug["score"] + (p.full_lens_bonus if full else 0.0), full)

    def settled():
        """더 볼 것 없이 멈춰도 되는가: 검사를 통과했고 (prefer_full_lens 면) 확장 렌즈 모델로 맞았다."""
        return best is not None and best.ok and (not prefer or bool(best.debug.get("lens_full")))

    def spent():
        """정밀 맞추기 한도(max_refines)를 다 썼는가."""
        return prefer and len(tried) >= max(1, p.max_refines)

    def one(rb, strict=False, how=""):
        """어림 하나를 정밀하게 맞추고 best 를 고친다 → 약한 결과였는가. strict: 약한 결과는 best 로 받지 않는다."""
        nonlocal best
        if spent():
            return False
        combo, pair = rb.debug["combo"], rb.debug["rows_px"]
        refined.add((combo, pair))
        b = _refine(scene, rb, size, p)
        weak = bool(prefer and b.ok and not b.debug.get("lens_full"))
        tried.append("%s%s rows=%s → 어림 %.2f · 정밀 %.2f (흰 선 %.2f · 노랑 %.3f) %s" % (
            how, combo, pair, rb.debug["score"], b.debug["score"], b.debug["support"], b.debug["yellow"],
            ("ok — 확장 렌즈 모델이 깨져 단순 모델 (끝 띠 선을 달리 고른 것을 더 본다)" if weak else "ok") if b.ok
            else "; ".join(t for t in b.notes if t.startswith("실패"))))
        if weak:
            weak_ends.add((combo[0], combo[4]))
        if not (strict and weak) and (best is None or rank(b) > rank(best)):
            best = b
        return weak

    def near_lines(u):
        """끝 띠 선 u 곁의 다른 세로선들 (낮춘 문턱의 봉우리까지)."""
        return [v for v in loose_us() if 6 < abs(v - u) <= p.end_line_reach_px and len(track(v)) >= p.min_line_pts]

    def swapped(rb, a1, a2):
        """어림 rb 의 끝 띠 선을 (a1, a2) 로 바꾼 어림 (CM 선 · 중앙선 · 가로선 짝은 그대로). 조합이 안 되면 None."""
        combo, pair = rb.debug["combo"], rb.debug["rows_px"]
        var = (a1,) + tuple(combo[1:4]) + (a2,)
        if var == combo or not _five_ok(var, ww, p):
            return None
        b = known.get((var, pair))
        if b is None:
            top, bot = (_track_horizontal(mask, v, (a1, a2), p) for v in pair)
            if min(len(top), len(bot)) < p.min_line_pts:
                return None
            b = _rough(scene, dict({nm: track(u) for nm, u in zip(_VNAMES, var)}, top=top, bot=bot), var, pair, size, p)
            if b is None:
                return None
            known[(var, pair)] = b
            rough.append((b.debug["score"], id(b), b))
        return b

    def longer(rb):
        """rb 의 끝 띠 선 곁에 훨씬 긴 세로선이 있으면 그 선(들)으로 바꾼 어림, 없으면 None."""
        if not prefer or p.longer_line_ratio <= 0.0:
            return None
        combo = rb.debug["combo"]
        ends = []
        for u in (combo[0], combo[4]):
            best_v = max(near_lines(u), key=lambda v: len(track(v)), default=None)
            ends.append(best_v if best_v is not None and len(track(best_v)) >= p.longer_line_ratio * len(track(u)) else u)
        b = swapped(rb, ends[0], ends[1])
        if b is None or (ends[0], ends[1]) in weak_ends or (b.debug["combo"], b.debug["rows_px"]) in refined:
            return None
        return b

    def beside(rb, strict):
        """약한 결과 rb 의 끝 띠 선을 그 곁의 다른 세로선으로 바꾼 조합들을 맞춰 본다 — 끝 띠 선이 긴 것부터."""
        combo = rb.debug["combo"]
        cands = []
        for a1 in [combo[0]] + near_lines(combo[0]):
            for a2 in [combo[4]] + near_lines(combo[4]):
                if (a1, a2) in weak_ends:
                    continue
                b = swapped(rb, a1, a2)
                if b is None or (b.debug["combo"], b.debug["rows_px"]) in refined:
                    continue
                cands.append((min(len(track(a1)), len(track(a2))), b.debug["score"], id(b), b))
        cands.sort(key=lambda t: (-t[0], -t[1]))
        for _len, _score, _id, b in cands[:max(0, p.full_lens_retry)]:
            if spent() or (b.debug["combo"][0], b.debug["combo"][4]) in weak_ends:
                continue
            one(b, strict, "끝 띠 선 곁: ")
            if p.refine_stop_ok and settled():
                return

    def refine_in_order(candidates, limit, strict=False, how=""):
        done = 0
        for _score, _id, rb in candidates:
            if done >= limit + extra[0] or spent():
                break
            if (rb.debug["combo"][0], rb.debug["combo"][4]) in weak_ends or (rb.debug["combo"], rb.debug["rows_px"]) in refined:
                continue  # 같은 끝 띠 선 — 가로선 짝 · CM 선만 다른 후보는 렌즈가 같은 식으로 깨진다 (본선 사진: 여섯 후보 모두)
            done += 1
            up = longer(rb)
            if up is not None:  # 끝 띠 선 곁에 훨씬 긴 선이 있다 → 그 선으로 바꾼 것부터
                one(up, strict, how + "끝 띠 선 곁의 더 긴 선: ")
                if p.refine_stop_ok and settled():
                    break
                if (rb.debug["combo"][0], rb.debug["combo"][4]) in weak_ends:
                    continue
            if one(rb, strict, how):
                extra[0] = max(0, p.full_lens_retry)
                beside(rb, strict)
            if p.refine_stop_ok and settled():
                break

    combos = _five_line_candidates(peaks, ww, p, limit=p.max_combos, length={u: len(track(u)) for u, _s in peaks})
    first = roughs(combos)
    refine_in_order(first, p.refine)
    if first and (best is None or not best.ok) and p.retry_combos > 0:
        # 다시 찾기: 선이 긴 순으로 앞선 세로선 조합 몇 개에, 가로선 후보의 모든 짝을 대 본다.
        # 가로 줄 경계가 상자에 가려 짧게 잡히면 「짧은 쪽 길이 순 3개」 에 들지 못하고, 상자 윗면의 글자 줄 같은 것이 대신 뽑힌다 (2026-11-19 13:06 2경기장).
        # 틀린 짝은 세로 축척이 틀려 원이 맞지 않는다 → 어림한 원 반지름 / CM 반폭이 맞는 것만 남기고, 최종 검사(_finish)가 다시 거른다
        again = [t for t in roughs(combos[:p.retry_combos], every=True)
                 if p.circle_cm_range[0] - 0.1 <= t[2].circle_r / t[2].cm_half <= p.circle_cm_range[1] + 0.1]
        refine_in_order(again, p.retry_refine)
    if prefer and (best is None or not best.ok) and p.loose_peak_ratio < 1.0:
        # 낮춘 문턱의 봉우리로 다시: 진짜 구역선이 봉우리로 뽑히지 않은 화면 (위 3)). 새로 생긴 봉우리가 든 조합만 본다
        normal = {u for u, _s in peaks}
        more = [(u, 0.0) for u in loose_us()]
        if any(u not in normal for u, _s in more):
            wide = _five_line_candidates(more, ww, p, limit=p.max_combos, length={u: len(track(u)) for u, _s in more})
            refine_in_order(roughs([c for c in wide if any(u not in normal for u in c)]), p.refine, strict=True, how="낮춘 문턱: ")
    if best is None:
        if not combos:
            return Boundary(False, image_size=size, notes=["좌우 대칭인 세로선 5개를 찾지 못함 (봉우리 %d개)" % len(peaks)])
        return Boundary(False, image_size=size, notes=["후보 %d개 모두 어림 실패" % len(combos)])
    runner = sorted((t[0] for t in rough if t[2].debug["combo"] != best.debug["combo"]), reverse=True)
    best.debug["margin"] = best.debug["score"] - (runner[0] if runner else 0.0)
    best.debug["tried"] = tried
    best.debug["candidates"] = len(rough)
    return best


_VNAMES = ("Bs", "Bc", "C", "Rc", "Rs")


def _row_pairs(mask, combo, cache, hh, p, every=False):
    """두 바깥 세로선 사이의 가로선 짝 후보 [(짝 행, top 점, bot 점)] — 짧은 쪽 길이 순으로 max_row_pairs 개. every: 수를 줄이지 않고 모두."""
    key = (combo[0], combo[4])
    if key not in cache:
        hpeaks = _horizontal_peaks(mask, combo[0], combo[4], p)
        cache[key] = {v: _track_horizontal(mask, v, key, p) for v, _s in hpeaks[:10]}
    tracks = cache[key]
    pairs = []
    for v1, v2 in itertools.combinations(tracks, 2):
        if abs(v1 - v2) >= 0.2 * hh:
            pairs.append((min(len(tracks[v1]), len(tracks[v2])), tuple(sorted((v1, v2)))))
    pairs.sort(reverse=True)
    return [(pair, tracks[pair[0]], tracks[pair[1]]) for npts, pair in (pairs if every else pairs[:p.max_row_pairs]) if npts >= p.min_line_pts]


def _rough(scene, groups, combo, pair, size, p):
    """직선성으로 k, 끝 띠 선 × 가로선 교점 넷으로 H. s · c 는 화면 비율로 어림, r 은 원 점 거리의 최빈값."""
    import cv2

    k = _plumb_k(groups, size, scene)
    lens = Lens.simple(k, size)
    lines = {nm: _line_fit(g, lens) for nm, g in groups.items()}
    w, h = p.field_w, p.field_h
    span = combo[4] - combo[0]
    s0 = p.strip_prior_m
    cm0 = (combo[3] - combo[1]) / 2.0 / span * (w - 2 * s0)
    try:
        src = np.array([_intersect(lines["Bs"], lines["top"]), _intersect(lines["Rs"], lines["top"]),
                        _intersect(lines["Rs"], lines["bot"]), _intersect(lines["Bs"], lines["bot"])])
    except np.linalg.LinAlgError:
        return None
    dst = np.array([(s0, 2 * h / 3), (w - s0, 2 * h / 3), (w - s0, h / 3), (s0, h / 3)])
    hom, _ = cv2.findHomography(src, dst)
    if hom is None or abs(hom[2, 2]) < 1e-12:
        return None
    hom = hom / hom[2, 2]
    # CM 선 위치는 찾은 선 점을 경기장으로 옮겨 잰다 (화면 비율 어림은 원근 · 왜곡으로 어긋난다)
    xm = {nm: float(np.median(_apply_h(hom, _undistort(groups[nm], lens))[:, 0])) for nm in ("Bc", "C", "Rc")}
    cm0 = (xm["Rc"] - xm["Bc"]) / 2.0
    if not p.rough_c_range[0] < cm0 < p.rough_c_range[1]:
        return None
    _circ, r0 = _circle_points(scene.mask, groups, lens, hom, p, scene)
    b = Boundary(False, k=float(k), hom=hom, image_size=size, lens=lens, strip=s0, cm_half=float(cm0), circle_r=float(r0 or cm0),
                 field_w=w, field_h=h, debug={"groups": groups, "combo": combo, "rows_px": pair})
    _score(scene, b, p, p.rough_support_px)
    return b


def _refine(scene, rb, size, p):
    mask = scene.mask
    groups = dict(rb.debug["groups"])
    w, h = p.field_w, p.field_h
    n = rb.lens.n
    circ, r0 = _circle_points(mask, groups, rb.lens, rb.hom, p, scene)
    if r0 is None:
        return Boundary(False, image_size=size, debug=dict(rb.debug, score=-1.0, support=0.0, yellow=0.0), notes=["실패 중앙 원을 찾지 못함"])
    groups["circle"] = circ
    # 1) 단순 모델
    theta = _pack(rb.lens, rb.hom, rb.strip, rb.cm_half, r0)
    theta, res = _fit(theta, _SIMPLE, groups, n, p)
    lens, hom, *_ = _unpack(theta, n)
    circ, _r = _circle_points(mask, groups, lens, hom, p, scene)  # 맞춘 모델로 원 점을 다시 모아 한 번 더
    if len(circ) >= 30:
        groups["circle"] = circ
        theta, res = _fit(theta, _SIMPLE, groups, n, p)
    lens_note = "렌즈: 단순 모델 (k1)"
    lens_full = False
    # 2) 확장 모델 (k2 · 픽셀 비 · 중심) — 출발점 여럿에서 맞춰 보고, 안전 검사를 통과한 것 중 바닥선에 가장 잘 맞는 것.
    #    모두 이상하면 단순 모델을 그대로 쓴다. 출발점: 단순 모델 결과 · lens_priors (렌즈를 고정해 H 를 먼저 맞춘 뒤 전부 푼다)
    if p.full_lens:
        region = _field_region_px(theta, n, size, w, h)
        starts = [("단순 모델에서", theta)]
        ww, hh = size
        for k1, k2, aspect in p.lens_priors:
            t0 = theta.copy()
            t0[:5] = [k1, k2, aspect, ww / 2.0, hh / 2.0]
            t0, _r = _fit(t0, list(range(5, 16)), groups, n, p)
            starts.append(("사전값 (%.2f, %.2f, %.2f) 에서" % (k1, k2, aspect), t0))
        cands, why_all = [], []
        for label, t0 in starts:
            t2, r2 = _fit(t0, _FULL, groups, n, p)
            why = _lens_sane(_unpack(t2, n)[0], size, p, region)
            if why is None:
                cands.append((_robust_cost(r2, p), label, t2))
            else:
                why_all.append("%s: %s" % (label, why))
        if cands:
            cands.sort(key=lambda c: c[0])
            _cost, label, theta2 = cands[0]
            lens2, hom2, *_ = _unpack(theta2, n)
            circ2, _r = _circle_points(mask, groups, lens2, hom2, p, scene)  # 원 점을 다시 모아 한 번 더
            if len(circ2) >= 30:
                groups["circle"] = circ2
                theta3, res3 = _fit(theta2, _FULL, groups, n, p)
                if _lens_sane(_unpack(theta3, n)[0], size, p, region) is None:
                    theta2 = theta3
            theta, res = theta2, _model_residuals(theta2, groups, n, p)
            lens_note = "렌즈: 확장 모델, %s (k1 %.3f · k2 %.3f · 픽셀 비 %.3f · 중심 %.0f, %.0f)" % ((label,) + tuple(theta[:5]))
            lens_full = True
        else:
            lens_note = "렌즈: 확장 모델 버림 (%s) → 단순 모델" % " / ".join(why_all)
    b = _finish(scene, theta, res, groups, size, p, {"combo": rb.debug["combo"], "rows_px": rb.debug["rows_px"]}, lens_note)
    b.debug["lens_full"] = lens_full or not p.full_lens  # 확장 모델로 맞았는가 (full_lens 를 꺼 두었으면 따지지 않는다) — _search 가 본다
    return b


def _finish(scene, theta, res, groups, size, p, debug, lens_note):
    """맞춘 theta → Boundary 와 검사 (통과해야 ok)."""
    n = (size[0] / 2.0) ** 2
    w, h = p.field_w, p.field_h
    lens, hom, s, c, r = _unpack(theta, n)
    inl = np.abs(res) < 3 * p.huber_m
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else math.inf
    b = Boundary(False, k=lens.k1, hom=hom, image_size=size, lens=lens, strip=float(s), cm_half=float(c), circle_r=float(r),
                 rms_m=rms, points=int(inl.sum()), field_w=w, field_h=h,
                 debug=dict(debug, groups=groups, inlier_frac=float(inl.mean()) if len(inl) else 0.0))
    _score(scene, b, p)
    checks = [
        (b.debug["support"] >= p.min_support, "모델 선 위 흰 선 비율 %.0f%%" % (b.debug["support"] * 100)),
        (b.debug["yellow"] <= p.max_yellow, "구역 안 노란 비율 %.1f%%" % (b.debug["yellow"] * 100)),
        (rms <= p.max_rms_m, "잔차 RMS %.1f cm" % (rms * 100)),
        (b.points >= p.min_points, "맞은 점 %d개" % b.points),
        (p.s_range[0] <= b.strip <= p.s_range[1], "끝 띠 %.2f m" % b.strip),
        (p.c_range[0] <= b.cm_half <= p.c_range[1], "CM 반폭 %.2f m" % b.cm_half),
        (p.r_range[0] <= b.circle_r <= p.r_range[1], "원 반지름 %.2f m" % b.circle_r),
        (p.circle_cm_range[0] <= b.circle_r / max(b.cm_half, 1e-6) <= p.circle_cm_range[1], "원 반지름 / CM 반폭 %.2f" % (b.circle_r / max(b.cm_half, 1e-6))),
        (p.span_cm_range[0] <= (w / 2 - b.strip) / max(b.cm_half, 1e-6) <= p.span_cm_range[1],
         "(가운데 ~ 끝 띠 선) / CM 반폭 %.2f" % ((w / 2 - b.strip) / max(b.cm_half, 1e-6))),
    ]
    if "버림" in lens_note and "왜곡 중심" in lens_note:
        checks.append((False, "렌즈 가운데가 화면 가운데에서 너무 멀다 — 화면이 한쪽으로 잘린 듯 (가운데를 화면 가운데로 둔 단순 모델은 믿지 않는다)"))
    b.notes = [("OK " if good else "실패 ") + text for good, text in checks] + [lens_note]
    b.ok = all(good for good, _ in checks)
    return b


def _from_prior(scene, prior, size, p):
    """지난 경계로 이 화면의 세로선 5개 · 가로선 2개 조합을 바로 고르고, 그 하나만 처음 찾기와 같은 방법(어림 → 정밀)으로 맞춘다.

    → (Boundary 또는 None, 안 된 까닭). 맞추는 방법이 같으므로 같은 조합을 고르면 결과도 처음 찾기와 같다 (다른 것은 고르는 시간뿐).
    """
    if isinstance(prior, dict):
        prior = Boundary.from_dict(prior)
    if prior is None or prior.hom is None or prior.strip is None or prior.cm_half is None:
        return None, "지난 경계에 대응이 없음"
    if tuple(prior.image_size) != tuple(size):
        return None, "화면 크기 %s ≠ 지난 %s" % (tuple(size), tuple(prior.image_size))
    if abs(prior.field_w - p.field_w) > 1e-6 or abs(prior.field_h - p.field_h) > 1e-6:
        return None, "경기장 크기가 다름"
    probe = Boundary(False, k=prior.k, hom=prior.hom, image_size=size, lens=prior._lens(), strip=prior.strip, cm_half=prior.cm_half,
                     circle_r=prior.circle_r or prior.cm_half, field_w=p.field_w, field_h=p.field_h)
    _score(scene, probe, p, p.rough_support_px)
    if probe.debug["support"] < p.prior_min_support:
        return None, "그대로 그린 선 위 흰 선 비율 %.0f%% — 카메라가 바뀐 듯" % (probe.debug["support"] * 100)
    mask = scene.mask
    ww, hh = size
    w, h = p.field_w, p.field_h
    # 세로선 5개: 지난 모델의 선이 화면 가운데 높이(y = H/2)에서 지나는 열에 가장 가까운 봉우리
    peaks = [u for u, _s in _vertical_peaks(mask, p)]
    xs = (prior.strip, w / 2 - prior.cm_half, w / 2, w / 2 + prior.cm_half, w - prior.strip)
    want = prior.field_to_pixel([(x, h / 2) for x in xs])[:, 0]
    if not peaks or not np.isfinite(want).all():
        return None, "세로선 봉우리 없음"
    combo = tuple(int(min(peaks, key=lambda u: abs(u - c))) for c in want)
    off = max(abs(u - c) for u, c in zip(combo, want))
    if len(set(combo)) < 5 or off > p.prior_tol_px * 2:
        return None, "지난 세로선 자리에 봉우리가 없음 (최대 %.0f px 어긋남)" % off
    groups = {nm: _track_vertical(mask, u, (0, hh), p) for nm, u in zip(_VNAMES, combo)}
    if any(len(g) < p.min_line_pts for g in groups.values()):
        return None, "세로선 점이 모자람"
    # 가로선 2개: 지난 모델의 가로선이 중앙선(x = W/2)에서 지나는 행에 가장 가까운, 처음 찾기와 같은 가로선 후보
    hpeaks = _horizontal_peaks(mask, combo[0], combo[4], p)[:10]
    rows = prior.field_to_pixel([(w / 2, 2 * h / 3), (w / 2, h / 3)])[:, 1]
    if len(hpeaks) < 2 or not np.isfinite(rows).all():
        return None, "가로선 후보 없음"
    pick = [min((v for v, _s in hpeaks), key=lambda v: abs(v - r)) for r in rows]
    roff = max(abs(v - r) for v, r in zip(pick, rows))
    if pick[0] == pick[1] or roff > p.htrack_px:
        return None, "지난 가로선 자리에 후보가 없음 (최대 %.0f px 어긋남)" % roff
    pair = tuple(sorted(pick))
    top, bot = (_track_horizontal(mask, v, (combo[0], combo[4]), p) for v in pair)
    if min(len(top), len(bot)) < p.min_line_pts:
        return None, "가로선 점이 모자람"
    rb = _rough(scene, dict(groups, top=top, bot=bot), combo, pair, size, p)
    if rb is None:
        return None, "어림 실패"
    b = _refine(scene, rb, size, p)
    b.debug["candidates"] = 1
    if not b.ok:
        return b, "; ".join(t for t in b.notes if t.startswith("실패"))
    change = max(abs(b.strip - prior.strip), abs(b.cm_half - prior.cm_half))
    if change > p.prior_max_change_m:
        b.ok = False
        return b, "치수가 지난번과 %.0f cm 다름" % (change * 100)
    return b, None


def _prior_groups(scene, lens, hom, strip, cm_half, p, tol_m):
    """모델(lens · hom · 치수)이 그리는 바닥선마다, 화면의 흰 선 픽셀 중 그 선에서 tol_m 안의 것 → 선 점 (2 px 칸마다 평균 한 점)."""
    pts = scene.pts
    xy = _apply_h(hom, scene.undistorted_pts(lens))
    w, h = p.field_w, p.field_h
    groups = {}
    inside_y = (xy[:, 1] > -0.05) & (xy[:, 1] < h + 0.05)
    for name, x0 in zip(_VNAMES, (strip, w / 2 - cm_half, w / 2, w / 2 + cm_half, w - strip)):
        groups[name] = _bin_mean(pts[inside_y & (np.abs(xy[:, 0] - x0) < tol_m)], along=1)
    between = (xy[:, 0] > strip + 0.1) & (xy[:, 0] < w - strip - 0.1)
    for name, y0 in (("top", 2 * h / 3), ("bot", h / 3)):
        groups[name] = _bin_mean(pts[between & (np.abs(xy[:, 1] - y0) < tol_m)], along=0)
    return groups


def _bin_mean(pts, along):
    """pts 를 along 축 좌표 2 px 칸마다 묶어 평균 → 칸마다 한 점."""
    if len(pts) == 0:
        return pts.reshape(0, 2)
    key = (pts[:, along] // 2).astype(np.int64)
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return np.stack([np.bincount(inv, pts[:, 0], len(uniq)) / cnt, np.bincount(inv, pts[:, 1], len(uniq)) / cnt], 1)


def _refit_prior(scene, prior, size, p):
    """선 하나가 가려져 봉우리로 조합을 고를 수 없을 때: 지난 모델 선 둘레의 흰 선 픽셀을 그대로 모아 다시 맞춘다 → (Boundary 또는 None, 까닭)."""
    if isinstance(prior, dict):
        prior = Boundary.from_dict(prior)
    lens = prior._lens()
    n = (size[0] / 2.0) ** 2
    lens = Lens(lens.k1, lens.k2, lens.aspect, lens.cx, lens.cy, n)
    full = p.full_lens and (lens.k2 != 0.0 or lens.aspect != 1.0 or lens.cx != size[0] / 2.0 or lens.cy != size[1] / 2.0)
    theta = _pack(lens, prior.hom, prior.strip, prior.cm_half, prior.circle_r or prior.cm_half)
    mid = np.array([[size[0] / 2.0, size[1] / 2.0], [size[0] / 2.0 + 1, size[1] / 2.0]])
    m_per_px = float(np.linalg.norm(np.diff(_apply_h(prior.hom, _undistort(mid, lens)), axis=0)))  # 화면 가운데 1 px 의 바닥 길이
    tol_m = max(0.02, p.prior_tol_px * m_per_px)
    res = groups = None
    for _round in range(2):  # 모으기 → 맞추기 → 맞춘 모델로 좁게 다시 모으기 → 맞추기
        lz, hm, s_, c_, _r = _unpack(theta, n)
        groups = _prior_groups(scene, lz, hm, float(s_), float(c_), p, tol_m)
        circ, r0 = _circle_points(scene.mask, groups, lz, hm, p, scene)
        short = [nm for nm, g in groups.items() if len(g) < p.min_line_pts]
        if short or r0 is None:
            return None, "선 점이 모자람 (%s)" % (", ".join(short) or "원")
        groups["circle"] = circ
        theta, res = _fit(theta, _FULL if full else _SIMPLE, groups, n, p)
        tol_m = max(0.02, 3 * m_per_px)
    if full:
        why = _lens_sane(_unpack(theta, n)[0], size, p, _field_region_px(theta, n, size, p.field_w, p.field_h))
        if why is not None:
            return None, "다시 맞춘 렌즈가 이상함 (%s)" % why
    note = "렌즈: 지난 경계의 %s 모델에서 (k1 %.3f · k2 %.3f · 픽셀 비 %.3f · 중심 %.0f, %.0f)" % (("확장" if full else "단순",) + tuple(theta[:5]))
    b = _finish(scene, theta, res, groups, size, p, {"combo": None, "rows_px": None, "candidates": 0}, note)
    if not b.ok:
        return b, "; ".join(t for t in b.notes if t.startswith("실패"))
    change = max(abs(b.strip - prior.strip), abs(b.cm_half - prior.cm_half))
    if change > p.prior_max_change_m:
        b.ok = False
        return b, "치수가 지난번과 %.0f cm 다름" % (change * 100)
    return b, None


def _score(scene, b, p, px=None):
    if "groups" in b.debug:  # 선마다 쓴 점 수 (없는 선은 0) — 어느 길로 맞췄든 여기서 채운다
        b.debug["lines"] = {nm: len(b.debug["groups"].get(nm, ())) for nm in _LINE_NAMES}
    b.debug["support"] = _support(scene, b, p, px)
    b.debug["yellow"] = _yellow_in_sectors(scene.yellow, b, p)
    b.debug["score"] = b.debug["support"] - p.yellow_weight * b.debug["yellow"]


def _yellow_in_sectors(yellow, b, p):
    """구역(끝 띠 선 안쪽, CM 선 바깥) 바닥을 5 cm 격자로 뽑아 노란 픽셀 비율. 화면 밖은 세지 않는다."""
    hh, ww = yellow.shape
    w, h = b.field_w, b.field_h
    m = 0.05
    xs = np.concatenate([np.arange(b.strip + m, w / 2 - b.cm_half - m, 0.05), np.arange(w / 2 + b.cm_half + m, w - b.strip - m, 0.05)])
    ys = np.arange(m, h - m, 0.05)
    if len(xs) == 0:
        return 1.0
    gx, gy = np.meshgrid(xs, ys)
    uv = np.round(np.nan_to_num(b.field_to_pixel(np.stack([gx.ravel(), gy.ravel()], 1)), nan=-1e6)).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)
    uv = uv[inside]
    return float(yellow[uv[:, 1], uv[:, 0]].mean()) if len(uv) else 1.0


def _support(scene, b, p, px=None):
    """맞춘 모델의 바닥선(구역선 5 · 가로선 2 · 원)을 화면에 그렸을 때, 그 위에 실제 흰 선 픽셀이 있는 비율.

    틀린 조합도 자유 치수(s · c · r) 덕에 잔차는 작게 맞춰질 수 있다 → 「모델이 화면을 얼마나 설명하는가」 로 고른다.
    가로선은 끝 띠 선 사이만 (바깥은 거점 그림). 화면 밖 · 경기장 밖은 세지 않는다.
    """
    hh, ww = scene.mask.shape
    w, h = b.field_w, b.field_h
    segs = [[(x, 0.0), (x, h)] for x in (b.strip, w / 2 - b.cm_half, w / 2, w / 2 + b.cm_half, w - b.strip)]
    segs += [[(b.strip, y), (w - b.strip, y)] for y in (h / 3, 2 * h / 3)]
    segs += [[(w / 2 + b.circle_r * math.cos(a), h / 2 + b.circle_r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 145)]]
    px = p.support_px if px is None else px
    near = scene.near(px)
    # 선분마다 60점 — 모든 선분을 한 번에 뽑아 한 번에 화면으로 옮긴다
    starts = np.array([sg[i] for sg in segs for i in range(len(sg) - 1)], float)
    ends = np.array([sg[i + 1] for sg in segs for i in range(len(sg) - 1)], float)
    pts = np.linspace(starts, ends, 60, axis=1).reshape(-1, 2)
    uv = np.round(np.nan_to_num(b.field_to_pixel(pts), nan=-1e6)).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)
    uv = uv[inside]
    total = len(uv)
    hit = int(near[uv[:, 1], uv[:, 0]].sum())
    return hit / total if total else 0.0


# ------------------------------------------------------------------ 축소 맞추기 (끝 띠 선이 화면 밖일 때)
_LINE_NAMES = ("Bs", "Bc", "C", "Rc", "Rs", "top", "bot", "circle")


@dataclass
class ReducedParams:
    """축소 맞추기의 값들. 세로선 5개로 맞추는 길(BoundaryParams)과 섞이지 않게 따로 둔다 (2026-11-19).

    평가 경기장은 화면 좌우가 잘려 끝 띠 선이 안 보일 수 있다 → (CM · 중앙 · CM) 세 선 + 가로선 2개 + 중앙 원으로 맞춘다.
    치수 범위는 연습 화면에서 세로선 5개로 맞춘 값 (CM 반폭 0.60~0.64 · 원/CM 1.02 안팎 · (W/2 − 끝 띠)/CM 3.46~3.66) 을 둘러싼 것.
    """

    enabled: bool = True
    cm_m: float = 0.61  # 어림 단계의 CM 반폭 (m)
    strip_per_cm: float = 3.635  # (W/2 − 끝 띠) / CM 반폭 — 끝 띠 선이 안 보이면 이 비로 끝 띠를 정한다
    strip_range: tuple = (0.80, 0.90)  # 그렇게 정한 끝 띠는 연습 화면에서 잰 범위 (0.79~0.90 m) 안으로 자른다 — CM 반폭이 크게 나온 화면에서 너무 좁아지지 않게
    strip_px_ratio: float = 3.33  # 화면에서 (중앙선 → 끝 띠 선) / (중앙선 → CM 선) 픽셀 거리 비 (연습 화면 3.33~3.35)
    strip_px_tol: float = 0.12  # 위 자리에서 이 비율 안에 봉우리가 있으면 끝 띠 선으로 쓴다
    max_combos: int = 8  # 세 선 조합을 이만큼까지 어림해 본다
    max_row_pairs: int = 3
    refine: int = 2  # 어림 점수 상위 이만큼만 정밀하게
    # 렌즈를 푸는 두 방법: "full" = 다섯 값 모두, "k1k2" = k1 · k2 만 (픽셀 비는 lens_priors 값 · 중심은 화면 가운데 그대로).
    #   끝 띠 선이 둘 다 보이면 (세로선 5개) 렌즈가 잘 정해진다 → full 만 (안 되면 실패).
    #   끝 띠 선이 없거나 하나뿐이면 렌즈가 약하게만 정해진다 → k1k2 가 검사를 통과해야 하고, full 도 통과하면 두 결과가 화면 안 바닥 점을
    #   max_mode_gap_m 안에서 같게 봐야 한다 (다르면 렌즈가 정해지지 않은 것 → 실패). full 의 왜곡 중심이 화면 가운데에서 멀리 나와도 실패. 낼 때는 k1k2 결과.
    max_mode_gap_m: float = 0.06
    cm_px_range: tuple = (0.05, 0.32)  # (중앙선 → CM 선 픽셀 거리) / 화면 세로 — 연습 화면 0.11, 2.6배 확대해도 0.30
    max_circle_pts: int = 2000  # 원 점이 이보다 많으면 고르게 솎는다 (맞추는 시간 — 화면이 확대돼 있으면 원 점이 수천 개)
    # 단순 모델로 맞춘 직후의 치수가 이 범위 밖이면 틀린 선 조합으로 보고 확장 모델은 해 보지 않는다 (시간)
    simple_c_range: tuple = (0.45, 0.85)
    simple_rc_range: tuple = (0.80, 1.25)
    # 검사 (모두 통과해야 ok)
    c_range: tuple = (0.58, 0.655)  # 맞춘 CM 반폭 (m)
    rc_range: tuple = (0.97, 1.07)  # 원 반지름 / CM 반폭 — 원이 CM 선에 닿는다
    ratio_range: tuple = (3.42, 3.80)  # 끝 띠 선 하나를 썼을 때 (W/2 − 끝 띠) / CM 반폭
    max_rms_m: float = 0.02
    min_support: float = 0.75  # 실제 칠 모양대로 (가로선은 끝 띠 선 ~ CM 선만) 그린 모델 선 위 흰 선 비율 (맞는 화면 0.85 이상 · 틀린 조합 0.64)
    min_line_support: float = 0.45  # 맞추기에 쓴 선 하나하나의 비율 (중앙선은 원 · 물체에 가려 0.54 까지 내려간다)
    min_circle_support: float = 0.7  # 원의 비율 (맞는 화면 0.79 이상 · 틀린 조합 0.48~0.55)
    min_circle_bins: int = 9  # 원 점이 둘레 30° 칸 12개 중 이만큼은 채워야 한다 (원이 가로 · 세로 축척을 묶는다)
    min_circle_pts: int = 100
    k1_range: tuple = (-0.9, 0.05)
    aspect_range: tuple = (0.35, 1.1)


REDUCED = ReducedParams()


def _reduced_if_failed(scene, size, p, best, rp=None):
    """세로선 5개로 못 맞췄을 때만 축소 맞추기를 해 본다. 검사를 모두 통과하면 그것을, 아니면 best 를 그대로 (까닭만 덧붙여) 낸다."""
    rp = rp or REDUCED
    if best.ok or not rp.enabled:
        return best
    try:
        red = _search_reduced(scene, size, p, rp)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, KeyError, IndexError) as exc:  # 맞추다 난 계산 오류 — 실패로 본다
        red = Boundary(False, image_size=size, notes=["계산 오류 %s" % type(exc).__name__])
    why = "; ".join(t for t in best.notes if not t.startswith("OK") and not t.startswith("렌즈"))[:200]
    if red.ok:
        red.notes.append("세로선 5개로는 안 됨 (%s) → 축소 맞추기로 맞춤" % why)
        return red
    best.notes.append("축소 맞추기도 안 됨: %s" % ("; ".join(t for t in red.notes if t.startswith("실패") or "못" in t or "오류" in t)[:300] or "까닭 모름"))
    return best


def _search_reduced(scene, size, p, rp):
    """좌우 대칭인 세 선 (CM · 중앙 · CM) + 가로선 2개 + 중앙 원으로 맞춘다. 한쪽 끝 띠 선이 보이면 그것도 쓴다.

    세로선 5개보다 근거가 적다 → 검사(_finish_reduced)가 더 엄하고, 하나라도 어긋나면 ok=False.
    끝 띠 선이 하나도 없으면 끝 띠 폭은 CM 반폭 × strip_per_cm 으로 정한다 (notes · debug 에 남긴다).
    """
    mask = scene.mask
    ww, hh = size
    peaks = _vertical_peaks(mask, p)
    vtracks = {u: _track_vertical(mask, u, (0, hh), p) for u, _s in peaks}
    us = [u for u, _s in peaks if len(vtracks[u]) >= p.min_line_pts]
    cands = []
    for c in us:
        for b1, b2 in itertools.product([u for u in us if u < c], [u for u in us if u > c]):
            d1, d2 = c - b1, b2 - c
            if abs(d1 - d2) > p.sym_tol * max(d1, d2) or not rp.cm_px_range[0] * hh <= (d1 + d2) / 2.0 <= rp.cm_px_range[1] * hh:
                continue
            cands.append((min(len(vtracks[b1]), len(vtracks[c]), len(vtracks[b2])), (b1, c, b2)))
    if not cands:
        return Boundary(False, image_size=size, notes=["좌우 대칭인 세로선 3개 (CM · 중앙 · CM) 를 찾지 못함 (봉우리 %d개)" % len(peaks)])
    cands.sort(reverse=True)
    rough = []
    cache = {}
    for _n, (b1, c, b2) in cands[:rp.max_combos]:
        d = (b2 - b1) / 2.0
        groups = {"Bc": vtracks[b1], "C": vtracks[c], "Rc": vtracks[b2]}
        edge = {}  # 끝 띠 선: 중앙선에서 CM 선 거리의 약 3.33배 자리에 봉우리가 있으면 쓴다
        for name, want in (("Bs", c - rp.strip_px_ratio * d), ("Rs", c + rp.strip_px_ratio * d)):
            near = [u for u in us if abs(u - want) <= rp.strip_px_tol * rp.strip_px_ratio * d]
            if near:
                edge[name] = max(near, key=lambda u: len(vtracks[u]))
                groups[name] = vtracks[edge[name]]
        lo = edge.get("Bs", max(0, int(round(c - rp.strip_px_ratio * d))))
        hi = edge.get("Rs", min(ww, int(round(c + rp.strip_px_ratio * d))))
        if hi - lo < 2 * p.vert_len_px:
            continue
        for pair, top, bot in _row_pairs_reduced(mask, lo, hi, cache, hh, p, rp):
            b = _rough_reduced(scene, dict(groups, top=top, bot=bot), (edge.get("Bs"), b1, c, b2, edge.get("Rs")), pair, size, p, rp)
            if b is not None:
                rough.append((b.debug["score"], id(b), b))
    if not rough:
        return Boundary(False, image_size=size, notes=["세 선 후보 %d개 모두 어림 실패 (가로선 짝이나 원이 없음)" % len(cands)])
    rough.sort(key=lambda t: -t[0])
    best = None
    for _sc, _id, rb in rough[:rp.refine]:
        b = _refine_reduced(scene, rb, size, p, rp)
        if best is None or (b.ok, b.debug.get("score", -1.0)) > (best.ok, best.debug.get("score", -1.0)):
            best = b
        if best.ok:
            break
    best.debug["candidates"] = len(rough)
    return best


def _row_pairs_reduced(mask, lo, hi, cache, hh, p, rp):
    """열 lo ~ hi 사이의 가로선 짝 후보 [(짝 행, top 점, bot 점)] — 짧은 쪽 길이 순."""
    key = (lo, hi)
    if key not in cache:
        cache[key] = {v: _track_horizontal(mask, v, key, p) for v, _s in _horizontal_peaks(mask, lo, hi, p)[:10]}
    tracks = cache[key]
    pairs = []
    for v1, v2 in itertools.combinations(tracks, 2):
        if abs(v1 - v2) >= 0.2 * hh:
            pairs.append((min(len(tracks[v1]), len(tracks[v2])), tuple(sorted((v1, v2)))))
    pairs.sort(reverse=True)
    return [(pair, tracks[pair[0]], tracks[pair[1]]) for npts, pair in pairs[:rp.max_row_pairs] if npts >= p.min_line_pts]


def _rough_reduced(scene, groups, combo, pair, size, p, rp):
    """직선성으로 k, CM 선 × 가로선 교점 넷으로 H (CM 반폭은 cm_m 으로 어림). 가로선은 CM 선 사이에 칠이 없으므로 늘인 선끼리의 교점이다."""
    import cv2

    w, h = p.field_w, p.field_h
    c0 = rp.cm_m
    try:
        with np.errstate(all="ignore"):
            k = _plumb_k(groups, size, scene)
            lens = Lens.simple(k, size)
            lines = {nm: _line_fit(groups[nm], lens) for nm in ("Bc", "Rc", "top", "bot")}
            src = np.array([_intersect(lines["Bc"], lines["top"]), _intersect(lines["Rc"], lines["top"]),
                            _intersect(lines["Rc"], lines["bot"]), _intersect(lines["Bc"], lines["bot"])])
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.isfinite(src).all():
        return None
    dst = np.array([(w / 2 - c0, 2 * h / 3), (w / 2 + c0, 2 * h / 3), (w / 2 + c0, h / 3), (w / 2 - c0, h / 3)])
    hom, _ = cv2.findHomography(src, dst)
    if hom is None or abs(hom[2, 2]) < 1e-12:
        return None
    hom = hom / hom[2, 2]
    _circ, r0 = _circle_points(scene.mask, groups, lens, hom, p, scene)
    if r0 is None:
        return None
    s0 = w / 2 - rp.strip_per_cm * c0
    for name in ("Bs", "Rs"):  # 보이는 끝 띠 선이 있으면 그 자리로
        if name in groups:
            x = float(np.median(_apply_h(hom, _undistort(groups[name], lens))[:, 0]))
            s0 = x if name == "Bs" else w - x
    b = Boundary(False, k=float(k), hom=hom, image_size=size, lens=lens, strip=float(s0), cm_half=float(c0), circle_r=float(r0),
                 field_w=w, field_h=h, debug={"groups": groups, "combo": combo, "rows_px": pair})
    _score(scene, b, p, p.rough_support_px)
    return b


def _thin(pts, limit):
    """점이 limit 보다 많으면 고르게 솎는다."""
    step = int(math.ceil(len(pts) / float(limit))) if limit else 1
    return pts[::step] if step > 1 else pts


def _refine_reduced(scene, rb, size, p, rp):
    """_refine 과 같은 순서 (단순 모델 → 확장 모델) 로 맞춘다. 끝 띠 선이 없으면 끝 띠(theta 13)는 풀지 않는다 (잔차가 없어 정해지지 않는다).

    확장 모델은 두 방법으로 푼다 — "full" 은 렌즈 다섯 값을 모두, "k1k2" 는 k1 · k2 만 (픽셀 비는 사전값 · 중심은 화면 가운데).
    어느 것을 쓰고 어떻게 서로 견주는지는 ReducedParams 의 설명과 이 함수 끝을 본다.
    """
    mask = scene.mask
    groups = dict(rb.debug["groups"])
    w, h = p.field_w, p.field_h
    n = rb.lens.n
    debug = {"combo": rb.debug["combo"], "rows_px": rb.debug["rows_px"]}

    def failed(text):
        return Boundary(False, image_size=size, debug=dict(debug, score=-1.0, support=0.0, yellow=0.0, reduced=True), notes=["실패 " + text])

    circ, r0 = _circle_points(mask, groups, rb.lens, rb.hom, p, scene)
    if r0 is None:
        return failed("중앙 원을 찾지 못함")
    groups["circle"] = _thin(circ, rp.max_circle_pts)
    hold = set() if ("Bs" in groups or "Rs" in groups) else {13}
    free_simple = [i for i in _SIMPLE if i not in hold]
    theta = _pack(rb.lens, rb.hom, rb.strip, rb.cm_half, r0)
    theta, res = _fit(theta, free_simple, groups, n, p)
    lens, hom, *_ = _unpack(theta, n)
    circ, _r = _circle_points(mask, groups, lens, hom, p, scene)  # 맞춘 모델로 원 점을 다시 모아 한 번 더
    if len(circ) >= 30:
        groups["circle"] = _thin(circ, rp.max_circle_pts)
        theta, res = _fit(theta, free_simple, groups, n, p)
    c1, r1 = float(theta[14]), float(theta[15])
    debug["simple_dims"] = (round(c1, 3), round(r1 / c1, 3) if c1 > 1e-6 else None)
    if not (rp.simple_c_range[0] <= c1 <= rp.simple_c_range[1] and rp.simple_rc_range[0] <= r1 / c1 <= rp.simple_rc_range[1]):
        return failed("단순 모델에서 이미 치수가 안 맞음 (CM 반폭 %.2f m · 원 반지름 / CM 반폭 %.2f)" % (c1, r1 / c1 if c1 > 1e-6 else 0.0))
    if not p.full_lens:
        return _finish_reduced(scene, theta, res, groups, size, p, rp, dict(debug, lens_mode="simple"), "렌즈: 단순 모델 (k1)")
    region = _field_region_px(theta, n, size, w, h)
    ww, hh = size
    free_hom = [i for i in range(5, 16) if i not in hold]
    starts = [("단순 모델에서", theta)]
    for k1, k2, aspect in p.lens_priors:
        t0 = theta.copy()
        t0[:5] = [k1, k2, aspect, ww / 2.0, hh / 2.0]
        t0, _r = _fit(t0, free_hom, groups, n, p)
        starts.append(("사전값 (%.2f, %.2f, %.2f) 에서" % (k1, k2, aspect), t0))
    def solve(mode):
        g = dict(groups)
        lens_hold = set() if mode == "full" else {2, 3, 4}
        free_full = [i for i in _FULL if i not in hold and i not in lens_hold]
        cands, why_all = [], []
        for label, t0 in (starts if mode == "full" else starts[1:]):  # k1k2 는 픽셀 비를 사전값으로 둔 출발점만 (단순 모델의 픽셀 비 1 은 이 카메라에 안 맞는다)
            t2, r2 = _fit(t0, free_full, g, n, p)
            why = _lens_sane(_unpack(t2, n)[0], size, p, region)
            if why is None:
                cands.append((_robust_cost(r2, p), label, t2))
            else:
                why_all.append("%s: %s" % (label, why))
        if not cands:
            return failed("확장 렌즈 모델(%s)이 모두 이상함 (%s)" % (mode, " / ".join(why_all)))
        cands.sort(key=lambda t: t[0])
        _cost, label, theta2 = cands[0]
        lens2, hom2, *_ = _unpack(theta2, n)
        circ2, _r = _circle_points(mask, g, lens2, hom2, p, scene)  # 원 점을 다시 모아 한 번 더
        if len(circ2) >= 30:
            g["circle"] = _thin(circ2, rp.max_circle_pts)
            theta3, _res3 = _fit(theta2, free_full, g, n, p)
            if _lens_sane(_unpack(theta3, n)[0], size, p, region) is None:
                theta2 = theta3
        note = "렌즈: 확장 모델 (%s), %s (k1 %.3f · k2 %.3f · 픽셀 비 %.3f · 중심 %.0f, %.0f)" % (
            ("모두 풂" if mode == "full" else "k1 · k2 만 풂", label) + tuple(theta2[:5]))
        return _finish_reduced(scene, theta2, _model_residuals(theta2, g, n, p), g, size, p, rp, dict(debug, lens_mode=mode), note)

    if "Bs" in groups and "Rs" in groups:  # 세로선 5개 — 렌즈가 잘 정해진다. 모두 푼 것이 안 되면 실패 (k1k2 로 물러서면 중심이 어긋난 화면에서 10 cm 넘게 틀렸다)
        return solve("full")
    b = solve("k1k2")
    if not b.ok:
        return b
    other = solve("full")
    if other.ok:
        gap = _frame_gap_m(b, other)
        b.debug["mode_gap_m"] = gap
        good = gap <= rp.max_mode_gap_m
        b.notes.insert(0, ("OK " if good else "실패 ") + "렌즈를 달리 풀었을 때 화면 안 바닥 점 차이 최대 %.1f cm" % (gap * 100))
        b.ok = good
    elif other.hom is None and any("왜곡 중심" in t for t in other.notes):
        # 모두 풀면 왜곡 중심이 화면 가운데에서 멀리 나온다 → 「중심 = 화면 가운데」 로 둔 k1k2 결과를 믿을 수 없다
        b.notes.insert(0, "실패 렌즈를 모두 풀면 왜곡 중심이 화면 가운데에서 멀리 나옴 — 중심을 화면 가운데로 둔 결과를 믿을 수 없음")
        b.ok = False
    return b


def _frame_gap_m(a, b):
    """두 경계가 같은 화면 픽셀을 바닥의 얼마나 다른 곳으로 보는가 — 화면 안에 보이는 경기장 격자점 (0.25 m 간격) 에서의 최대 거리 (m)."""
    w, h = a.field_w, a.field_h
    gx, gy = np.meshgrid(np.arange(0.5, w - 0.49, 0.25), np.arange(0.25, h - 0.24, 0.25))
    xy = np.stack([gx.ravel(), gy.ravel()], 1)
    uv = a.field_to_pixel(xy)
    ww, hh = a.image_size
    inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)
    if not inside.any():
        return math.inf
    d = np.hypot(*(b.pixel_to_field(uv[inside]) - xy[inside]).T)
    return float(np.nanmax(d)) if np.isfinite(d).any() else math.inf


def _line_support(scene, b, p):
    """실제 칠 모양대로 그린 모델 선 하나하나 → {이름: (화면 안 표본 수, 그 위에 흰 선 픽셀이 있는 비율)}. 가로선은 끝 띠 선 ~ CM 선 구간만."""
    hh, ww = scene.mask.shape
    w, h = b.field_w, b.field_h
    t = np.linspace(0.0, 1.0, 121)[:, None]

    def seg(a, z):
        return np.asarray(a, float) + t * (np.asarray(z, float) - np.asarray(a, float))

    xl, xr = w / 2 - b.cm_half, w / 2 + b.cm_half
    shapes = {nm: seg((x, 0.0), (x, h)) for nm, x in zip(_VNAMES, (b.strip, xl, w / 2, xr, w - b.strip))}
    for nm, y in (("top", 2 * h / 3), ("bot", h / 3)):
        shapes[nm] = np.concatenate([seg((b.strip, y), (xl, y)), seg((xr, y), (w - b.strip, y))])
    ang = np.linspace(0, 2 * math.pi, 181)
    shapes["circle"] = np.stack([w / 2 + b.circle_r * np.cos(ang), h / 2 + b.circle_r * np.sin(ang)], 1)
    near = scene.near(p.support_px)
    out = {}
    for nm, pts in shapes.items():
        uv = np.round(np.nan_to_num(b.field_to_pixel(pts), nan=-1e6)).astype(int)
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)]
        out[nm] = (len(uv), float(near[uv[:, 1], uv[:, 0]].mean()) if len(uv) else 0.0)
    return out


def _finish_reduced(scene, theta, res, groups, size, p, rp, debug, lens_note):
    """축소 맞추기 결과 → Boundary 와 검사. 근거가 적으므로 _finish 보다 엄하다 — 하나라도 어긋나면 ok=False."""
    n = (size[0] / 2.0) ** 2
    w, h = p.field_w, p.field_h
    theta = theta.copy()
    seen = [nm for nm in ("Bs", "Rs") if nm in groups]
    if not seen:  # 끝 띠 선이 하나도 없다 → CM 반폭에 비례해 정한다 (직접 잰 값이 아니다)
        theta[13] = min(max(w / 2 - rp.strip_per_cm * theta[14], rp.strip_range[0]), rp.strip_range[1])
    lens, hom, s, c, r = _unpack(theta, n)
    pk = _Packed(groups)
    inl = np.abs(res) < 3 * p.huber_m
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else math.inf
    b = Boundary(False, k=lens.k1, hom=hom, image_size=size, lens=lens, strip=float(s), cm_half=float(c), circle_r=float(r),
                 rms_m=rms, points=int(inl.sum()), field_w=w, field_h=h,
                 debug=dict(debug, groups=groups, inlier_frac=float(inl.mean()) if len(inl) else 0.0, reduced=True, strip_derived=not seen))
    _score(scene, b, p)
    used = [nm for nm in _LINE_NAMES if nm in groups]
    inl_by = {nm: int(inl[pk.code == _LINE_CODE.get(nm, 7)].sum()) for nm in used}
    sup = _line_support(scene, b, p)
    b.debug["line_support"] = {nm: (cnt, round(frac, 3)) for nm, (cnt, frac) in sup.items()}
    tot = sum(sup[nm][0] for nm in used)
    support = sum(sup[nm][0] * sup[nm][1] for nm in used) / tot if tot else 0.0
    weak = [nm for nm in used if sup[nm][0] < 20 or sup[nm][1] < (rp.min_circle_support if nm == "circle" else rp.min_line_support)]
    short = [nm for nm in used if inl_by[nm] < (rp.min_circle_pts if nm == "circle" else p.min_line_pts)]
    # 원 점이 둘레를 고루 채우는가 (한쪽 호만으로는 가로 · 세로 축척을 묶지 못한다)
    cxy = _apply_h(hom, _undistort(np.asarray(groups["circle"], float).reshape(-1, 2), lens)) - [w / 2, h / 2]
    bins = len(set((np.degrees(np.arctan2(cxy[:, 1], cxy[:, 0])) % 360 // 30).astype(int).tolist())) if len(cxy) else 0
    ok_c = c > 1e-6 and np.isfinite(c)
    ratio = (w / 2 - b.strip) / b.cm_half if ok_c else math.inf
    rc = b.circle_r / b.cm_half if ok_c else math.inf
    lens_why = _lens_sane(lens, size, p, _field_region_px(theta, n, size, w, h))
    checks = [
        (support >= rp.min_support, "실제 칠 모양대로 그린 선 위 흰 선 비율 %.0f%%" % (support * 100)),
        (not weak, "선마다 흰 선 비율 (모자란 선: %s)" % (", ".join(weak) or "없음")),
        (not short, "선마다 맞은 점 수 (모자란 선: %s)" % (", ".join(short) or "없음")),
        (b.debug["yellow"] <= p.max_yellow, "구역 안 노란 비율 %.1f%%" % (b.debug["yellow"] * 100)),
        (rms <= rp.max_rms_m, "잔차 RMS %.1f cm" % (rms * 100)),
        (b.points >= p.min_points, "맞은 점 %d개" % b.points),
        (rp.c_range[0] <= b.cm_half <= rp.c_range[1], "CM 반폭 %.3f m" % b.cm_half),
        (rp.rc_range[0] <= rc <= rp.rc_range[1], "원 반지름 / CM 반폭 %.3f" % rc),
        (bins >= rp.min_circle_bins, "원 점이 둘레 30° 칸 12개 중 %d개" % bins),
        (not seen or rp.ratio_range[0] <= ratio <= rp.ratio_range[1], "(W/2 − 끝 띠) / CM 반폭 %.2f" % ratio),
        (rp.k1_range[0] <= lens.k1 <= rp.k1_range[1], "렌즈 k1 %.2f" % lens.k1),
        (rp.aspect_range[0] <= lens.aspect <= rp.aspect_range[1], "픽셀 비 %.2f" % lens.aspect),
        (lens_why is None, "렌즈 검사 (%s)" % (lens_why or "통과")),
    ]
    b.notes = [("OK " if good else "실패 ") + text for good, text in checks] + [lens_note]
    b.notes.append("축소 맞추기: 쓴 선 %s" % ", ".join("%s %d점" % (nm, len(groups[nm])) for nm in used))
    if seen:
        b.notes.append("끝 띠 선은 %s 쪽만 보여 그것으로 끝 띠 폭 %.2f m 를 맞춤 (반대쪽은 좌우 대칭으로 봄)" % (" · ".join("BLUE" if nm == "Bs" else "RED" for nm in seen), b.strip))
    else:
        b.notes.append("끝 띠 선이 화면에 없어 끝 띠 폭 %.2f m 는 CM 반폭 × %.3f 으로 (범위 %.2f~%.2f m 안에서) 정한 값 — 직접 잰 것이 아님" % (
            (b.strip, rp.strip_per_cm) + tuple(rp.strip_range)))
    b.ok = all(good for good, _ in checks)
    return b


# ------------------------------------------------------------------ 그림
def draw_before_after(bgr, b, px_per_m=100):
    """(원본 위에 경계 · 구역선을 그린 그림, 위에서 본 지도) — 둘 다 BGR."""
    import cv2

    raw = bgr.copy()
    w, h = b.field_w, b.field_h
    vx = [b.strip, w / 2 - b.cm_half, w / 2, w / 2 + b.cm_half, w - b.strip]
    hy = [h / 3, 2 * h / 3]

    def poly(img, pts_m, color, th):
        pts = np.concatenate([b.field_to_pixel(np.linspace(pts_m[i], pts_m[i + 1], 80)) for i in range(len(pts_m) - 1)])
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < 2:
            return
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, th, cv2.LINE_AA)

    for x in vx:
        poly(raw, [(x, 0), (x, h)], (0, 200, 255), 1)
    for y in hy:
        poly(raw, [(0, y), (w, y)], (0, 200, 255), 1)
    circ = [(w / 2 + b.circle_r * math.cos(a), h / 2 + b.circle_r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 73)]
    poly(raw, circ, (0, 200, 255), 1)
    poly(raw, [(0, 0), (w, 0), (w, h), (0, h), (0, 0)], (0, 255, 0) if b.ok else (0, 0, 255), 3)
    for g in b.debug.get("groups", {}).values():
        for u, v in g[::3]:
            cv2.circle(raw, (int(u), int(v)), 1, (255, 0, 255), -1)

    gw, gh = int(round(w * px_per_m)), int(round(h * px_per_m))
    gx, gy = np.meshgrid(np.arange(gw) + 0.5, np.arange(gh) + 0.5)
    pix = b.field_to_pixel(np.stack([gx.ravel() / px_per_m, h - gy.ravel() / px_per_m], 1)).reshape(gh, gw, 2).astype(np.float32)
    top = cv2.remap(bgr, pix[..., 0], pix[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 0, 60))
    for x in vx:
        cv2.line(top, (int(x * px_per_m), 0), (int(x * px_per_m), gh), (0, 200, 255), 1)
    for y in hy:
        cv2.line(top, (0, int((h - y) * px_per_m)), (gw, int((h - y) * px_per_m)), (0, 200, 255), 1)
    cv2.circle(top, (int(w / 2 * px_per_m), int(h / 2 * px_per_m)), int(b.circle_r * px_per_m), (0, 200, 255), 1)
    cv2.rectangle(top, (0, 0), (gw - 1, gh - 1), (0, 255, 0) if b.ok else (0, 0, 255), 3)
    return raw, top
