"""상대 UGV 의 수광부를 따라가며 겨누기 (선택 사항 — 기본은 꺼져 있다). ROS 를 쓰지 않는다.

지정표적 조준(aim.py)과 같은 원리다: 레이저 점은 사진의 정해진 픽셀에 찍히므로(aim.laser_pixel), 수광부가 그 픽셀에 오게 팬틸트를 돌린다.
다른 점은 표적이 움직인다는 것 — 사진마다 「그 수광부를 맞히려면 팬틸트가 어느 각도여야 하는지」 를 적고, 변화로 각속도를 어림해 조금 앞을 겨눈다.

수광부의 자리는 perception 의 /team/perception/locate_sensors 답(SensorSpot)에서 온다. owner 가 ugv 인 것(사진 속 UGV 상자 안의 수광부)만 쓰고,
진영(camp, 안테나 색)이 상대인 것만 고른다 — 아군과 진영을 못 읽은 UGV 는 쏘지 않는다.

팬틸트의 실제 위치를 읽을 수 없으므로 주기는 「돌리고 → 기다리고 → 찍는」 한 바퀴(약 1초)다. 빠르게 가로지르는 표적은 못 따라간다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .aim import AimParams, laser_pixel

UGV = "ugv"  # SensorSpot.msg 의 owner


@dataclass
class TrackParams:
    min_confidence: float = 0.25
    ugv_width_m: float = 0.23  # UGV 가 사진에 찍힌 폭으로 거리를 어림할 때 쓰는 차체 폭 [추정]
    lead_s: float = 0.5  # 사진을 찍고 팬틸트가 그 자리에 가기까지 걸리는 시간 어림 — 이만큼 앞을 겨눈다
    max_age_s: float = 1.2  # 마지막으로 본 지 이보다 오래면 쏘지 않는다
    tol_deg: float = 1.0  # 팬틸트와 표적 자리의 차이가 이 안이면 쏜다
    max_rate_deg_s: float = 40.0  # 이보다 빨리 움직이는 것으로 계산되면 잘못 짝지은 것으로 보고 속도를 0 으로 둔다
    # 「멈춰 있다」 = 잇달아 본 still_looks 장 이상이 still_span_s 넘는 동안 모두 still_tol_deg 안에 있다.
    # 유효 피격은 수광부에 2초 넘게 이어서 닿아야 하므로, 움직이는 표적은 따라가며 쏴도 인정되지 않았다 (2026-11-20 콘솔 시험, 사용자)
    still_looks: int = 3
    still_span_s: float = 1.0
    still_tol_deg: float = 1.0
    # 멀리 있어 수광부가 작게 찍히면 허용을 줄인다: 허용 = 수광부가 사진에 찍힌 크기(도) × still_tol_size_frac, still_tol_min_deg ~ still_tol_deg 사이.
    # (레이저가 수광부 위에 머물러야 하므로 「멈췄다」 의 기준은 수광부 크기에 달려 있다.) 0 이면 늘 still_tol_deg
    still_tol_size_frac: float = 0.5
    still_tol_min_deg: float = 0.5
    # 「조금씩 움직인다」 = 최근 creep_window_s 의 관측 creep_looks 장 이상(creep_span_s 넘는 동안)이 곧은 선 위에 있고(벗어남이 허용 안),
    # 그 속도가 느리다: creep_rate_deg_s 이하이면서, 한 번 고쳐 겨누는 사이(follow_period_s)에 수광부 크기의 creep_size_frac 넘게 가지 않는다.
    # 이런 표적은 선으로 내다본 자리를 겨눠 쏘고, 레이저가 켜진 동안 계속 따라간다. creep_rate_deg_s 0 이면 끈다 (멈춘 것만 쏜다)
    creep_window_s: float = 2.6
    creep_looks: int = 4
    creep_span_s: float = 1.5
    creep_rate_deg_s: float = 1.2
    creep_min_rate_deg_s: float = 0.3  # 이보다 느리면 멈춘 것으로 친다 (평균 자리를 겨눈다)
    creep_size_frac: float = 0.5
    follow_period_s: float = 1.0
    fast_rate_deg_s: float = 6.0  # 이보다 빠르면 「많이 움직인다」 — 따라가며 기다리지 않는다
    gate_deg: float = 6.0  # 여러 대가 보일 때: 따라가던 표적에서 이 안에 있는 수광부를 같은 표적으로 본다
    # 쏜 뒤의 사진에서 수광부가 레이저 점에서 이 안에 있으면 「닿아 있다」: 수광부 크기 × hit_size_frac, 적어도 hit_min_deg
    hit_size_frac: float = 0.5
    hit_min_deg: float = 0.6


def ugv_range(spot, deg_per_px, p, image_w=640):
    """수광부가 달린 UGV 가 사진에 찍힌 폭으로 거리를 어림한다 (m). UGV 상자가 없으면 1.0."""
    if not spot.get("has_host") or spot.get("host_width", 0.0) <= 0.0:
        return 1.0
    wide = math.radians(spot["host_width"] * image_w * deg_per_px)
    return max(0.3, min(4.0, p.ugv_width_m / (2.0 * math.tan(max(0.01, min(1.5, wide / 2.0))))))


def offset_deg(spot, z, deg_per_px, aim):
    """수광부가 레이저 점에서 벗어난 만큼 팬틸트를 더 돌릴 각도 (팬 + 오른쪽, 틸트 + 위) 도."""
    lu, lv = laser_pixel(z, aim)
    du = spot["x_center"] * aim.image_w - lu
    dv = spot["y_center"] * aim.image_h - lv
    return aim.pan_sign * du * deg_per_px, -aim.tilt_sign * dv * deg_per_px


def sensor_deg(spot, deg_per_px, aim):
    """수광부가 사진에 찍힌 크기 (도) — 가로 · 세로 가운데 작은 쪽. 모르면 0."""
    wide, tall = spot.get("width", 0.0) * aim.image_w, spot.get("height", 0.0) * aim.image_h
    size = min(wide, tall) if min(wide, tall) > 0.0 else max(wide, tall)
    return max(0.0, size) * deg_per_px


def foe_spots(spots, foe_camp, p, allow_unknown_camp=False, skip_callsigns=()):
    """상대 UGV 의 수광부들. 아군 · 진영을 못 읽은 UGV · skip_callsigns(이미 목숨이 0 인 팀)는 뺀다."""
    out = []
    for s in spots or ():
        if s.get("owner") != UGV or s.get("confidence", 0.0) < p.min_confidence:
            continue
        camp = s.get("camp") or ""
        if camp != foe_camp and not (allow_unknown_camp and camp == ""):
            continue
        if s.get("callsign") and s["callsign"] in skip_callsigns:
            continue
        out.append(s)
    return out


def choose_ugv(spots, foe_camp, p, allow_unknown_camp=False, skip_callsigns=()):
    """상대 UGV 의 수광부를 고른다 (가장 크게 찍힌 = 가장 가까운 것). 아군 · 진영을 못 읽은 UGV 는 고르지 않는다."""
    best = None
    for s in foe_spots(spots, foe_camp, p, allow_unknown_camp, skip_callsigns):
        size = s["width"] * s["height"]
        if best is None or size > best[0]:
            best = (size, s)
    return None if best is None else best[1]


class UgvTracker:
    def __init__(self, params=None, aim=None, deg_per_px=None):
        self.p = params or TrackParams()
        self.aim = aim or AimParams()
        self.deg_per_px = float(deg_per_px) if deg_per_px else self.aim.deg_px(127.4)
        self.reset()

    def reset(self):
        self.obs = []  # (시각, 맞히는 팬, 맞히는 틸트, 수광부 크기 도, 그 사진에서 수광부가 레이저 점에서 벗어난 각도)
        self.range_m = None  # 마지막으로 본 표적까지의 거리 어림 (차체가 찍힌 폭으로)
        self.rate = (0.0, 0.0)

    def hit_angles(self, spot, pan_deg, tilt_deg):
        """그 수광부를 맞히려면 팬틸트가 어느 각도여야 하는가 (팬, 틸트). (pan_deg, tilt_deg) = 사진을 찍을 때의 팬틸트."""
        z = ugv_range(spot, self.deg_per_px, self.p, self.aim.image_w)
        dpan, dtilt = offset_deg(spot, z, self.deg_per_px, self.aim)
        return pan_deg + dpan, tilt_deg + dtilt

    def choose(self, spots, when, pan_deg, tilt_deg):
        """상대 수광부 여럿(foe_spots 의 답) 가운데 하나: 따라가던 표적 곁(gate_deg 안)에 있는 것, 없으면 가장 크게 찍힌 것.

        -> (고른 것 또는 None, 표적이 바뀌었는가). 매번 가장 큰 것만 고르면 두 대가 보일 때 표적이 오락가락해 「멈췄다」 가 되지 않는다.
        """
        spots = list(spots or ())
        if not spots:
            return None, False
        biggest = max(spots, key=lambda s: s["width"] * s["height"])
        if not self.obs or when - self.obs[-1][0] > self.p.max_age_s:
            return biggest, bool(self.obs)
        _when, pan, tilt = self.obs[-1][:3]

        def away(s):
            a, b = self.hit_angles(s, pan_deg, tilt_deg)
            return math.hypot(a - pan, b - tilt)

        near = min(spots, key=away)
        if away(near) <= self.p.gate_deg:
            return near, False
        return biggest, True

    def observe(self, when, spot, pan_deg, tilt_deg):
        """사진 한 장에서 고른 수광부. (pan_deg, tilt_deg) = 그 사진을 찍을 때의 팬틸트."""
        z = ugv_range(spot, self.deg_per_px, self.p, self.aim.image_w)
        dpan, dtilt = offset_deg(spot, z, self.deg_per_px, self.aim)
        seen = (when, pan_deg + dpan, tilt_deg + dtilt, sensor_deg(spot, self.deg_per_px, self.aim), math.hypot(dpan, dtilt))
        self.range_m = z
        self.obs = [o for o in self.obs if when - o[0] <= 3.0] + [seen]
        self.rate = (0.0, 0.0)
        if len(self.obs) >= 2:
            (t0, a0, b0), (t1, a1, b1) = self.obs[-2][:3], self.obs[-1][:3]
            if 0.05 <= t1 - t0 <= 2.0:
                rate = ((a1 - a0) / (t1 - t0), (b1 - b0) / (t1 - t0))
                if math.hypot(*rate) <= self.p.max_rate_deg_s:
                    self.rate = rate

    def command(self, now, pan_limit=50.0, tilt_limit=30.0):
        """지금 겨눌 (팬, 틸트). 본 적이 없거나 너무 오래됐으면 None."""
        if not self.obs or now - self.obs[-1][0] > self.p.max_age_s:
            return None
        when, pan, tilt = self.obs[-1][:3]
        ahead =(now - when) + self.p.lead_s
        pan, tilt = pan + self.rate[0] * ahead, tilt + self.rate[1] * ahead
        return max(-pan_limit, min(pan_limit, pan)), max(-tilt_limit, min(tilt_limit, tilt))

    def still_point(self, now):
        """표적이 멈춰 있으면 그 자리 (맞히는 팬, 틸트 — 본 것들의 평균), 아니면 None.

        마지막으로 본 것에서 거슬러 올라가며 still_tol_deg 안에 있는 것들만 모은다. 조금 흔들리는 것(판독 잡음 · 제자리에서 까딱임)은 멈춘 것으로 친다.
        """
        if not self.obs or now - self.obs[-1][0] > self.p.max_age_s:
            return None
        _when, pan, tilt = self.obs[-1][:3]
        run, tol = [], self.tol()
        for o in reversed(self.obs):
            if math.hypot(o[1] - pan, o[2] - tilt) > tol:
                break
            run.append(o)
        if len(run) < self.p.still_looks or run[0][0] - run[-1][0] < self.p.still_span_s:
            return None
        return sum(o[1] for o in run) / len(run), sum(o[2] for o in run) / len(run)

    def last_point(self, now, pan_limit=50.0, tilt_limit=30.0):
        """마지막으로 본 자리 (앞을 겨누지 않는다). 본 적이 없거나 너무 오래됐으면 None — 멈춘 표적만 쏠 때 쓴다."""
        if not self.obs or now - self.obs[-1][0] > self.p.max_age_s:
            return None
        pan, tilt = self.still_point(now) or self.obs[-1][1:3]
        return max(-pan_limit, min(pan_limit, pan)), max(-tilt_limit, min(tilt_limit, tilt))

    # ------------------------------------------------------------------ 움직임의 정도 (멈춤 · 조금씩 · 움직임 · 많이)
    def tol(self):
        """「멈췄다」 의 허용 (도): 수광부가 작게 찍혔으면(멀다) 줄인다."""
        p = self.p
        size = self.obs[-1][3] if self.obs else 0.0
        if size <= 0.0 or p.still_tol_size_frac <= 0.0:
            return p.still_tol_deg
        return max(min(p.still_tol_min_deg, p.still_tol_deg), min(p.still_tol_deg, p.still_tol_size_frac * size))

    def on_sensor(self, obs=None):
        """그 사진(기본: 마지막)에서 레이저 점이 수광부 위에 있는가 — 쏜 뒤의 사진으로 맞았는지 어림할 때 쓴다."""
        obs = obs or (self.obs[-1] if self.obs else None)
        if obs is None:
            return False
        return obs[4] <= max(self.p.hit_min_deg, self.p.hit_size_frac * obs[3])

    def _fit(self, now):
        """최근 creep_window_s 의 관측에 곧은 선을 맞춘다. -> (팬 속도, 틸트 속도, 지금의 팬, 지금의 틸트, 가장 큰 벗어남, 장 수, 걸친 시간) 또는 None."""
        if not self.obs or now - self.obs[-1][0] > self.p.max_age_s:
            return None
        pts = [o for o in self.obs if self.obs[-1][0] - o[0] <= self.p.creep_window_s]
        n, span = len(pts), pts[-1][0] - pts[0][0]
        if n < 2 or span <= 0.0:
            return None
        tm, am, bm = sum(o[0] for o in pts) / n, sum(o[1] for o in pts) / n, sum(o[2] for o in pts) / n
        den = sum((o[0] - tm) ** 2 for o in pts)
        ra = sum((o[0] - tm) * (o[1] - am) for o in pts) / den
        rb = sum((o[0] - tm) * (o[2] - bm) for o in pts) / den
        resid = max(math.hypot(o[1] - (am + ra * (o[0] - tm)), o[2] - (bm + rb * (o[0] - tm))) for o in pts)
        return ra, rb, am + ra * (now - tm), bm + rb * (now - tm), resid, n, span

    def creep_limit(self):
        """「조금씩 움직인다」 로 쳐 줄 속도의 상한 (도/초)."""
        p = self.p
        size = self.obs[-1][3] if self.obs else 0.0
        if size <= 0.0:
            return p.creep_rate_deg_s
        return min(p.creep_rate_deg_s, p.creep_size_frac * size / max(0.1, p.follow_period_s))

    def motion(self, now):
        """표적의 움직임: "still" 멈춤 · "creep" 조금씩 · "moving" 움직임 · "fast" 많이 · "new" 본 지 얼마 안 됨 · None 안 보임."""
        if not self.obs or now - self.obs[-1][0] > self.p.max_age_s:
            return None
        p, fit = self.p, self._fit(now)
        if fit is not None and p.creep_rate_deg_s > 0.0:
            ra, rb, _a, _b, resid, n, span = fit
            steady = n >= p.creep_looks and span >= p.creep_span_s and resid <= self.tol()
            if steady and p.creep_min_rate_deg_s <= math.hypot(ra, rb) <= self.creep_limit():
                return "creep"
        if self.still_point(now) is not None:
            return "still"
        if fit is None or fit[5] < 3:
            return "new"
        return "fast" if math.hypot(fit[0], fit[1]) >= p.fast_rate_deg_s else "moving"

    def aim_point(self, now, pan_limit=50.0, tilt_limit=30.0):
        """지금 겨눌 자리. -> (팬, 틸트, 움직임, 범위로 자르기 전의 팬, 틸트) 또는 None.

        멈춘 표적은 본 것들의 평균, 조금씩 움직이는 표적은 선으로 lead_s 앞을 내다본 자리, 그 밖에는 마지막으로 본 자리다 (쏘지는 않는다).
        """
        kind = self.motion(now)
        if kind is None:
            return None
        if kind == "creep":
            ra, rb, a, b, _r, _n, _s = self._fit(now)
            pan, tilt = a + ra * self.p.lead_s, b + rb * self.p.lead_s
        else:
            pan, tilt = self.still_point(now) or self.obs[-1][1:3]
        return max(-pan_limit, min(pan_limit, pan)), max(-tilt_limit, min(tilt_limit, tilt)), kind, pan, tilt

    def aligned(self, now, pan_deg, tilt_deg):
        """지금 팬틸트가 표적이 「지금」 있을 자리에 맞아 있는가 (쏴도 되는가). 두 번은 봐야 한다."""
        if len(self.obs) < 2 or now - self.obs[-1][0] > self.p.max_age_s:
            return False
        when, pan, tilt = self.obs[-1][:3]
        pan, tilt = pan + self.rate[0] * (now - when), tilt + self.rate[1] * (now - when)
        return math.hypot(pan - pan_deg, tilt - tilt_deg) <= self.p.tol_deg
