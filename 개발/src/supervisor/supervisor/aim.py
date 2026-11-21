"""사진으로 조준하기 (M2 지정표적). ROS 를 쓰지 않는다.

레이저 발광부와 카메라는 팬틸트 머리에 같이 붙어 있다. 그래서 레이저 점은 팬틸트를 어디로 돌리든 사진의 같은 자리에 찍히고,
표면까지의 거리 Z 에 따라서만 조금 달라진다 (2026-11-19 측정: 640×480 사진, Z 0.385 ~ 0.82 m 에서 16번, 식과의 차이 0.36 픽셀):

    u = 332.5 + 0.9 / Z        v = 248.6 − 13.4 / Z        (v 는 아래로 갈수록 크다)

→ 수광부가 사진의 이 픽셀에 오게 팬틸트를 돌리고 쏘면 맞는다. 로봇 위치의 오차나 수광부의 높이를 몰라도 된다.

  laser_pixel     거리 Z 에서 레이저 점이 찍히는 픽셀
  surface_z       차체 가운데에서 잰 거리 → 위 식의 Z
  parse_sensors   perception 의 답(detail 글)에 실려 온 수광부 탐지를 꺼낸다
  find_receiver   사진 한 장의 탐지에서 「내 표적의 수광부」 를 고른다. 수광부 탐지가 없으면 표적 그림 상자의 정해진 자리
  AimLoop         사진 → 고침을 되풀이하는 한 차례 (발사 한 번 앞에서). 고친 뒤의 사진을 보고 「1픽셀에 몇 도」 를 다시 정한다

픽셀은 모두 640×480 사진 기준이다 (perception 의 답은 사진 크기에 대한 비율이라 여기서 곱한다).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass
class AimParams:
    """사진 조준의 값. 설정 파일의 "aim" 으로 바꾼다. [추정] 은 로봇에서 확인할 것."""

    enabled: bool = True  # False 면 사진을 쓰지 않고 계산한 각도로만 쏜다 (전의 방식)
    # ---- 레이저 점 (2026-11-19 에 16번 잰 것): u = laser_u0 + laser_ku / Z, v = laser_v0 + laser_kv / Z
    laser_u0: float = 332.5
    laser_ku: float = 0.9
    laser_v0: float = 248.6
    laser_kv: float = -13.4
    image_w: int = 640
    image_h: int = 480
    # 식의 Z 는 렌즈가 아니라 그보다 앞의 기준점에서 잰 거리다 (측정 기록: 렌즈에서의 거리가 Z 보다 약 4.8 cm 길다)
    z_offset_m: float = 0.048
    z_min_m: float = 0.30  # Z 를 이 범위로 자른다 (잰 범위는 0.385 ~ 0.82 m, 그 밖은 식으로 늘린 값)
    z_max_m: float = 5.0
    # ---- 픽셀 ↔ 팬틸트 각도
    deg_per_px: float = 0.0  # 0 이면 카메라 화각으로 계산한다 (127.4° / 640 = 0.199° — 2026-11-18 에 팬 1단위 2.51 픽셀 · 틸트 1단위 1.50 픽셀로 잼)
    pan_sign: float = 1.0  # 수광부가 레이저 점의 오른쪽에 있으면 팬을 + 로 (팬 + = 오른쪽)
    tilt_sign: float = 1.0  # 수광부가 레이저 점의 위쪽에 있으면 틸트를 + 로 (틸트 + = 위. 2026-11-18 사진으로 확인)
    # ---- 되풀이
    tol_px: float = 4.0  # 수광부와 레이저 점의 차이가 이 안이면 쏜다 (0.9 m 에서 4픽셀 ≈ 1.3 cm, 수광 돔의 반지름은 약 2.5 cm)
    tol_size_frac: float = 0.25  # 탐지한 수광부가 크게 찍혔으면(가까우면) 그 크기의 이만큼까지도 받아들인다
    max_corrections: int = 3  # 발사 한 번 앞에서 팬틸트를 고치는 횟수의 상한
    max_step_deg: float = 15.0  # 한 번에 고치는 각도의 상한
    max_misses: int = 4  # 사진에서 잇달아 이만큼 못 찾으면 그 발은 지금 각도로 쏜다
    answer_wait_s: float = 3.0  # 사진 한 장의 판독 답을 기다리는 가장 긴 시간
    budget_s: float = 10.0  # 발사 한 번 앞에서 사진 조준에 쓰는 시간의 상한 (넘으면 지금 각도로 쏜다)
    fresh_s: float = 6.0  # 확인한 뒤 이만큼 지나도록 못 쐈으면(잠김 · 재장전) 다시 확인한다
    # ---- 고친 결과로 다시 배우기: 팬틸트 단위나 방향의 가정이 틀려도 모이게
    learn: bool = True
    learn_min_px: float = 6.0  # 이만큼은 옮기려 한 때에만 배운다 (탐지 자리가 1 ~ 2픽셀 떨린다)
    learn_sign: bool = True  # 반대로 움직였으면 방향 가정을 뒤집는다
    gain_range: float = 2.5  # 배운 값은 처음 값의 1/2.5 ~ 2.5배 안으로만
    # ---- 수광부 고르기
    sensor_min_conf: float = 0.25  # 이보다 낮은 수광부 탐지는 쓰지 않는다
    picture_reach_px: float = 160.0  # 표적 그림은 겨눈 점(레이저 점)에서 이 안에 있어야 한다 (그림 상자의 가장 가까운 곳까지)
    sensor_reach_px: float = 90.0  # 표적 그림을 못 찾았을 때: 겨눈 점에서 이 안의 수광부만 쓴다 (다른 표적 · UGV 의 돔을 고르지 않게)
    # 수광부는 표적 그림 상자의 위쪽 가운데에 있다 (2026-11-19 실차 사진 4장: 가로 가운데에서 폭의 ±5%, 위 끝에서 높이의 18 ~ 21% 아래).
    region_half_w: float = 0.30  # 그림 상자 가운데에서 폭의 이만큼 안
    region_above: float = 0.35  # 그림 상자 위 끝보다 높이의 이만큼 위까지 (2026-11-18 모델은 돔 아래만 상자로 잡았다)
    region_below: float = 0.50  # 그림 상자 위 끝에서 높이의 이만큼 아래까지
    # 수광부 탐지가 없을 때 겨누는 자리: 그림 상자의 (가운데 + box_dx × 폭, 위 끝 + box_dy × 높이). 상자가 면 전체를 잡는 지금 모델 기준
    box_dx: float = 0.0
    box_dy: float = 0.20
    use_box: bool = True  # False 면 수광부 탐지가 있을 때만 사진으로 고친다

    def deg_px(self, hfov_deg):
        return self.deg_per_px if self.deg_per_px > 0.0 else float(hfov_deg) / float(self.image_w)


def laser_pixel(z, p=None):
    """표면까지의 거리 z(m)에서 레이저 점이 찍히는 픽셀 (u, v)."""
    p = p or AimParams()
    z = min(max(float(z), p.z_min_m), p.z_max_m)
    return p.laser_u0 + p.laser_ku / z, p.laser_v0 + p.laser_kv / z


def surface_z(body_dist_m, cam_forward_m=0.05, p=None):
    """차체 가운데에서 표적까지의 거리 → 레이저 점 식의 Z (렌즈가 차체 가운데보다 앞에 있고, 식의 기준점은 렌즈보다 더 앞이다)."""
    p = p or AimParams()
    return min(max(float(body_dist_m) - cam_forward_m - p.z_offset_m, p.z_min_m), p.z_max_m)


UNKNOWN_KIND = "unknown"  # Picture.msg 의 「그림은 보이는데 무엇인지 모름」
SENSORS_KEY = "sensors="  # perception 이 ClassifyImage 답의 detail 끝에 붙이는 글: sensors=[[x, y, w, h, conf], ...] (사진 크기에 대한 비율)


def parse_sensors(detail):
    """perception 의 답 detail 에서 수광부 탐지를 꺼낸다: [{"x", "y", "w", "h", "conf"}]. 그 글이 없으면(옛 perception) None."""
    text = detail if isinstance(detail, str) else ""
    at = text.rfind(SENSORS_KEY)
    if at < 0:
        return None
    try:
        rows, _end = json.JSONDecoder().raw_decode(text[at + len(SENSORS_KEY):].lstrip())
        out = []
        for row in rows:
            x, y, w, h = (float(v) for v in row[:4])
            conf = float(row[4]) if len(row) > 4 else 1.0
            if all(math.isfinite(v) for v in (x, y, w, h, conf)):
                out.append({"x": x, "y": y, "w": w, "h": h, "conf": conf})
        return out
    except (ValueError, TypeError, IndexError):
        return None


@dataclass(frozen=True)
class Receiver:
    """사진에서 고른 수광부."""

    u: float  # 가운데 (픽셀)
    v: float
    size: float  # 탐지 상자의 큰 변 (픽셀). 그림 상자에서 정한 자리면 0
    how: str  # "sensor" 표적 그림 위쪽에 있는 수광부 탐지 · "near" 그림 없이 겨눈 점 가까이의 수광부 탐지 · "box" 그림 상자의 정해진 자리
    conf: float = 0.0


def _gap_to_box(point, box):
    """점에서 상자(가운데 x, y · 폭 · 높이)의 가장 가까운 곳까지 (픽셀). 안에 있으면 0."""
    x, y, w, h = box
    dx = max(abs(point[0] - x) - w / 2.0, 0.0)
    dy = max(abs(point[1] - y) - h / 2.0, 0.0)
    return math.hypot(dx, dy)


def find_receiver(pictures, sensors, want_kind, near, p=None):
    """사진 한 장의 탐지에서 내 표적의 수광부를 고른다. 못 고르면 None.

    pictures  [{"kind", "x", "y", "w", "h", "conf", "cut"}] 그림들 (자리 · 크기는 사진에 대한 비율 — 기록 파일의 classify 줄과 같은 모양)
    sensors   [{"x", "y", "w", "h", "conf"}] 수광부 탐지 (없으면 None 이나 빈 목록)
    want_kind 표적 그림의 종류 (comms · outpost · radar)
    near      겨눈 점 = 레이저 점의 픽셀 (계산한 각도로 먼저 겨눴으므로 표적은 이 둘레에 있다)

    고르는 순서
      1. 표적 그림: 종류가 맞는 그림 중 겨눈 점에 가장 가까운 것 (picture_reach_px 안). 종류가 맞는 것이 없으면 겨눈 점이 들어 있는 그림
         (가까이서는 종류를 다르게 읽거나 모름으로 답할 수 있다. 지도는 이 면이 표적이라고 한다)
      2. 그 그림의 위쪽 가운데에 있는 수광부 탐지 → 그것 (여럿이면 정해진 자리에 가장 가까운 것)
      3. 표적 그림이 없으면: 겨눈 점에서 sensor_reach_px 안의 가장 가까운 수광부 탐지
      4. 수광부 탐지가 없으면: 표적 그림 상자의 정해진 자리. 사진 가장자리에서 잘린 상자와, 다른 종류로 분명히 읽힌 그림의 상자는 쓰지 않는다
         (수광부 탐지가 없는데 종류까지 다르면 표적 면이 아닐 수 있다 → 계산한 각도에 맡긴다)
    """
    p = p or AimParams()
    w_img, h_img = float(p.image_w), float(p.image_h)

    def box_of(d):
        return d["x"] * w_img, d["y"] * h_img, d["w"] * w_img, d["h"] * h_img

    pics = [q for q in pictures or () if q.get("w", 0.0) > 0.0 and q.get("h", 0.0) > 0.0]
    wanted = [q for q in pics if q.get("kind") == want_kind and _gap_to_box(near, box_of(q)) <= p.picture_reach_px]
    if wanted:
        pic = min(wanted, key=lambda q: (_gap_to_box(near, box_of(q)), math.hypot(q["x"] * w_img - near[0], q["y"] * h_img - near[1])))
        trust_box = True
    else:
        over = [q for q in pics if _gap_to_box(near, box_of(q)) <= 0.0]
        pic = min(over, key=lambda q: q["w"] * q["h"]) if over else None
        trust_box = pic is not None and pic.get("kind") == UNKNOWN_KIND
    cands = [s for s in sensors or () if s.get("conf", 1.0) >= p.sensor_min_conf]

    def made(s, how):
        _x, _y, sw, sh = box_of(s)
        return Receiver(s["x"] * w_img, s["y"] * h_img, max(sw, sh), how, float(s.get("conf", 1.0)))

    if pic is not None:
        x, y, w, h = box_of(pic)
        top = y - h / 2.0
        spot = (x + p.box_dx * w, top + p.box_dy * h)
        inside = [s for s in cands if abs(s["x"] * w_img - x) <= p.region_half_w * w and top - p.region_above * h <= s["y"] * h_img <= top + p.region_below * h]
        if inside:
            return made(min(inside, key=lambda s: math.hypot(s["x"] * w_img - spot[0], s["y"] * h_img - spot[1])), "sensor")
        if p.use_box and trust_box and not pic.get("cut"):
            return Receiver(spot[0], spot[1], 0.0, "box", float(pic.get("conf", 0.0)))
        return None
    if cands:
        best = min(cands, key=lambda s: math.hypot(s["x"] * w_img - near[0], s["y"] * h_img - near[1]))
        if math.hypot(best["x"] * w_img - near[0], best["y"] * h_img - near[1]) <= p.sensor_reach_px:
            return made(best, "near")
    return None


def pixel_error(receiver, laser):
    """수광부가 레이저 점에서 벗어난 픽셀: (오른쪽 +, 위쪽 +). 팬(오른쪽 +) · 틸트(위 +)를 이만큼 더 돌려야 한다."""
    return receiver.u - laser[0], laser[1] - receiver.v


class AimLoop:
    """수광부를 사진으로 겨누는 한 차례 (발사 한 번 앞에서). 사진을 볼 때마다 see() 를 부르면 다음에 할 일을 돌려준다.

    k[0] · k[1] = 수광부를 사진에서 1픽셀 옮기는 데 드는 팬 · 틸트 명령 각도(도). 처음에는 카메라 화각으로 정하고,
    고친 뒤의 사진에서 실제로 옮겨 간 픽셀을 보고 다시 정한다 — 팬틸트 1단위의 각도나 방향을 틀리게 알고 있어도 다음 고침에서 맞는다.
    배운 값은 다음 발에도 이어 쓴다 (reset() 은 횟수만 지운다).
    """

    def __init__(self, params=None, deg_per_px=None, pan_limit=50.0, tilt_limit=30.0):
        self.p = params or AimParams()
        self.k0 = float(deg_per_px) if deg_per_px else self.p.deg_px(127.4)
        self.k = [self.k0 * self.p.pan_sign, self.k0 * self.p.tilt_sign]
        self.limits = (float(pan_limit), float(tilt_limit))
        self.seen = 0  # 수광부를 찾은 사진 수 (이번 표적에서)
        self.relearned = []  # 배워서 바꾼 기록 [(축 이름, 새 값)]
        self.reset()

    def reset(self):
        """새 발 앞에서: 고친 횟수 · 못 찾은 횟수를 지운다."""
        self.corrections = 0
        self.misses = 0
        self.err = None  # 마지막으로 본 차이 (픽셀)
        self.offset = None  # 마지막으로 본 (오른쪽 +, 위쪽 +) 픽셀
        self.converged = False
        self._last = None  # 앞 고침: (그때의 차이, 보낸 각도 변화, 찾은 방법)

    def miss(self):
        self.misses += 1
        self._last = None

    def interrupted(self):
        """고침과 상관없이 팬틸트가 다른 데로 돌았다 → 앞 고침의 결과로 배우지 않는다."""
        self._last = None

    def see(self, receiver, laser, pan, tilt):
        """사진 한 장의 결과. receiver = find_receiver 의 답(못 찾았으면 None), laser = 레이저 점 픽셀, pan · tilt = 그 사진 때 보내 둔 각도(도).

        돌려주는 값
          ("look",)             못 찾았다 → 다시 찍는다
          ("turn", pan, tilt)   팬틸트를 이 각도로 돌리고 다시 찍는다
          ("fire",)             쏜다 — 허용 안에 들어왔거나(converged = True), 고칠 횟수를 다 썼거나 더 돌릴 수 없다(False)
        """
        p = self.p
        if receiver is None:
            self.miss()
            return ("look",)
        self.misses = 0
        self.seen += 1
        e = pixel_error(receiver, laser)
        self._learn(e, receiver.how)
        self.offset, self.err = e, math.hypot(e[0], e[1])
        if self.err <= max(p.tol_px, p.tol_size_frac * receiver.size):
            self.converged = True
            return ("fire",)
        self.converged = False
        if self.corrections >= p.max_corrections:
            return ("fire",)
        now = (float(pan), float(tilt))
        new = tuple(max(-self.limits[i], min(self.limits[i], now[i] + max(-p.max_step_deg, min(p.max_step_deg, self.k[i] * e[i])))) for i in (0, 1))
        step = (new[0] - now[0], new[1] - now[1])  # 실제로 보내는 변화 (한도에서 잘린 뒤)
        if abs(step[0]) < 0.02 and abs(step[1]) < 0.02:
            return ("fire",)  # 팬틸트 끝에 걸려 더 돌릴 수 없다
        self._last = (e, step, receiver.how)
        self.corrections += 1
        return ("turn", new[0], new[1])

    def _learn(self, e, how):
        """앞 고침에서 옮기려던 픽셀과 실제로 옮겨 간 픽셀을 견줘 k 를 다시 정한다."""
        last, self._last = self._last, None
        p = self.p
        if last is None or not p.learn:
            return
        e0, step, how0 = last
        if how != how0:
            return  # 앞뒤 사진에서 찾은 방법이 다르면(탐지 ↔ 그림 상자) 자리가 몇 픽셀 다를 수 있다
        for i, name in ((0, "pan"), (1, "tilt")):
            wanted = step[i] / self.k[i]  # 옮기려던 픽셀
            moved = e0[i] - e[i]  # 실제로 옮겨 간 픽셀
            if abs(wanted) < p.learn_min_px or abs(moved) < 0.4 * p.learn_min_px:
                continue  # 조금 옮긴 것으로는 배우지 않는다. 거의 안 움직였으면(팬틸트 잠김 · 덜 돎) 그대로 한 번 더 고친다
            k = step[i] / moved
            if k * self.k[i] < 0.0 and (not p.learn_sign or abs(moved) < p.learn_min_px):
                continue
            k = math.copysign(min(max(abs(k), self.k0 / p.gain_range), self.k0 * p.gain_range), k)
            if abs(k - self.k[i]) > 0.15 * abs(self.k[i]):
                self.relearned.append((name, round(k, 4)))
            self.k[i] = k
