"""레이저가 켜진 사진에서 레이저 점 찾기 (M2 — 쏜 것이 수광부에 닿았는지 보려고). ROS 를 쓰지 않는다. numpy 만 쓴다.

서버는 지정표적이 맞았는지 알려 주지 않는다 (경기서버연동API 6단계). 그래서 쏘기 직전의 사진과 레이저가 켜진 동안의 사진을 견줘
레이저 점이 찍힌 픽셀을 찾고, 그것이 조준 사진에서 본 수광부 자리와 겹치는지로 「맞은 것으로 보인다」 를 정한다.
2026-11-19 레이저 캘리브레이션(제어 콘솔)에서 쓴 것과 같은 방법이다.

찾는 방법
  - 레이저는 깜박일 수 있어(대회용 펄스 신호) 여러 장의 화소별 최댓값끼리 견준다
  - 밝은 면에서는 빨강이 이미 가득 차 있으므로 「붉은 정도」(R − (G + B) / 2)가 늘어난 곳을 본다
  - 가장 많이 늘어난 화소가 속한 덩어리의 무게중심이 레이저 점이다. 늘어난 정도가 약하면 못 찾은 것으로 친다

한계: 수광 돔에 닿은 빛이 사진에 어떻게 찍히는지(번짐 · 돔 안에서 흩어짐)는 실물로 확인하지 못했다. 못 찾으면 None 이고, 그때는 맞았는지 모른다.
"""
from __future__ import annotations

import numpy as np

MIN_PEAK = 20.0  # 붉은 정도가 이만큼은 늘어야 레이저 점으로 본다 (0 ~ 255 눈금)
# 레이저 점을 찾는 범위: 식으로 계산한 레이저 점(near)에서 이 픽셀 안만 본다.
# 2026-11-19 14:52 실차(지정표적 1.2 m 앞, 세 발)의 사진으로 확인한 것 — 사진 전체에서 찾으면 뒤에서 움직이는 사람이 레이저 점으로 잡힌다:
#   · 레이저가 찍힌 두 발: 찾은 점이 수광부에서 3.5 · 3.4 픽셀 (붉은 정도 109 · 110) → 맞게 찾음
#   · 레이저가 안 찍힌 한 발(깜박임의 꺼진 때): 상자 위로 보이는 사람 머리가 28 픽셀 위에서 잡혔다 (붉은 정도 88) → 「빗나감」 으로 보고
#     조준을 25 픽셀(약 5°) 틀리게 고칠 뻔했다. 레이저가 없는 사진 둘끼리 견줘도 33 · 274 픽셀 떨어진 곳이 잡혔다 (91 · 117)
# 표적 면(상자의 그림)은 움직이지 않으므로 그 안만 보면 이런 일이 없다. 1.5 m 에서 수광부는 상자 위 끝에서 15 픽셀쯤 아래다 → 16
WINDOW_PX = 16.0


def image_rgb(image):
    """sensor_msgs/Image (rgb8 · bgr8) → HxWx3 uint8 RGB. 다른 인코딩이면 None."""
    encoding = str(image.encoding).lower()
    if encoding not in ("rgb8", "bgr8"):
        return None
    rows = np.frombuffer(bytes(image.data), np.uint8).reshape(image.height, -1)
    pixels = rows[:, :image.width * 3].reshape(image.height, image.width, 3)
    return pixels.copy() if encoding == "rgb8" else pixels[:, :, ::-1].copy()


def _blur(a, r=2):
    """(2r+1)² 네모 평균 — 화소 하나의 잡음에 끌려가지 않게."""
    k = 2 * r + 1
    pad = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)


def find_spot(off_frames, on_frames, min_peak=MIN_PEAK, near=None, window_px=WINDOW_PX):
    """레이저를 켜기 전 사진들(off)과 켠 동안의 사진들(on)에서 레이저 점의 픽셀 (u, v) 를 찾는다. 못 찾으면 None.

    사진은 HxWx3 uint8 RGB. 돌려주는 값: {"u", "v", "peak": 늘어난 붉은 정도, "area": 덩어리의 화소 수}.
    near = 식으로 계산한 레이저 점 (u, v). 주면 거기서 window_px 안만 본다 (뒤에서 움직이는 사람 · 로봇을 레이저 점으로 잡지 않게 — WINDOW_PX 의 설명).
    """
    if not off_frames or not on_frames:
        return None
    off = np.max(np.stack(off_frames), axis=0).astype(np.float32)
    on = np.max(np.stack(on_frames), axis=0).astype(np.float32)
    if off.shape != on.shape:
        return None

    def redness(img):
        return img[:, :, 0] - (img[:, :, 1] + img[:, :, 2]) / 2.0

    score = np.clip(redness(on) - redness(off), 0, None) + 0.3 * np.clip(on[:, :, 0] - off[:, :, 0], 0, None)
    score = _blur(score)
    if near is not None and window_px is not None and window_px > 0:
        rows, cols = np.mgrid[0:score.shape[0], 0:score.shape[1]]
        score = np.where((np.abs(cols - float(near[0])) <= window_px) & (np.abs(rows - float(near[1])) <= window_px), score, 0.0)
    peak = float(score.max())
    if peak < min_peak:
        return None
    py, px = np.unravel_index(int(score.argmax()), score.shape)
    # 가장 밝은 화소가 속한 덩어리만 (렌즈 반사로 생긴 다른 붉은 얼룩은 뺀다): 그 화소에서 이어진 밝은 화소를 넓혀 간다
    bright = score >= 0.5 * peak
    comp = np.zeros_like(bright)
    comp[py, px] = True
    for _ in range(60):  # 반지름 60 픽셀까지
        grown = comp.copy()
        grown[1:, :] |= comp[:-1, :]
        grown[:-1, :] |= comp[1:, :]
        grown[:, 1:] |= comp[:, :-1]
        grown[:, :-1] |= comp[:, 1:]
        grown &= bright
        if grown.sum() == comp.sum():
            break
        comp = grown
    ys, xs = np.nonzero(comp)
    weights = score[comp]
    return {"u": float((xs * weights).sum() / weights.sum()), "v": float((ys * weights).sum() / weights.sum()),
            "peak": round(peak, 1), "area": int(comp.sum())}
