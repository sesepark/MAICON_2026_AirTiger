"""라이다 스캔으로 경기장의 「위아래 벽 사이」 거리를 직접 잰다 (ROS 없이 시험할 수 있는 계산만).

왜 (2026-11-19):
  연습 경기장은 벽에서 벽까지 6.12 × 4.80 m, 본선 경기장은 6.12 × 5.00 m 라고 전해 들었다 (줄자로 정확히 재지는 못함).
  지도의 높이가 10 cm 씩 틀리면 벽 가까이의 상자가 라이다와 어긋나 길이 막힌다 (오전 실차 두 판).
  드론 사진의 카메라로 높이를 고르지만(map_maker.arena_for), 본선 경기장의 5.00 은 로봇으로 확인해 본 적이 없다.
  → 경기 시작 전, 출발 칸에 서 있는 동안의 스캔에서 마주 보는 두 벽 사이를 재서, 4.80 이나 5.00 에 분명히 맞으면 그 값을 쓴다.
    애매하면(벽이 가려져 안 보임 · 두 값의 중간) 아무 답도 내지 않는다 — 그때는 카메라로 고른 값 그대로다.

재는 법: 지도도 로봇 위치도 쓰지 않는다. 스캔 안에서 벽의 방향(가장 많은 곧은 선의 방향)을 찾고, 그 방향과 수직 방향으로 점들을 나란히 세워
가장 바깥의 「길게 이어진 선」 둘 사이의 거리를 낸다. 상자 · 기둥은 짧아서(0.4 ~ 1.2 m) 길이 조건에서 빠지거나, 벽보다 안쪽이라 가장 바깥이 아니다.
"""
from __future__ import annotations

import math

import numpy as np


def wall_spans(ranges, angle_min, angle_increment, near=0.15, far=11.5, min_points=6, min_length=0.5):
    """스캔 한 장 → 마주 보는 긴 곧은 선 사이의 거리들 [m] (0 ~ 2개: 서로 수직인 두 방향). 벽의 방향은 스캔에서 스스로 찾는다.

    선의 조건(점 6개 · 길이 0.5 m)은 느슨하다 — 구석 거점에서는 4 m 떨어진 건너편 벽이 상자 사이로 조금만 보인다.
    그래도 되는 까닭: 상자 · 기둥의 면은 벽에서 0.4 m 넘게 떨어져 있어(2026-11-19 주최측: 상자는 벽에서 최소 40 cm) 건너편 벽까지의 거리가
    4.4 ~ 4.6 m 를 넘지 못한다. HeightVote 는 4.65 m 넘는 값만 받으므로, 그 범위의 값은 두 벽 사이를 잰 것뿐이다.
    """
    r = np.asarray(ranges, dtype=float)
    a = float(angle_min) + float(angle_increment) * np.arange(len(r))
    ok = np.isfinite(r) & (r > near) & (r < far)
    if int(ok.sum()) < 120:
        return []
    x, y, idx = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.nonzero(ok)[0]
    # 벽의 방향: 네 칸 떨어진 이웃 점끼리 이은 방향을 90° 로 접어 가장 많은 쪽
    k = 4
    neighbours = (idx[k:] - idx[:-k]) == k
    dx, dy = (x[k:] - x[:-k])[neighbours], (y[k:] - y[:-k])[neighbours]
    length = np.hypot(dx, dy)
    good = (length > 0.01) & (length < 0.35)
    if int(good.sum()) < 40:
        return []
    quarter = math.pi / 2.0
    direction = np.arctan2(dy[good], dx[good]) % quarter
    counts, edges = np.histogram(direction, bins=90, range=(0.0, quarter))
    peak = int(np.argmax(counts))
    centre = (edges[peak] + edges[peak + 1]) / 2.0
    off = (direction - centre + quarter / 2.0) % quarter - quarter / 2.0
    close = np.abs(off) < math.radians(3.0)
    if int(close.sum()) < 20:
        return []
    theta = centre + float(np.mean(off[close]))
    # 다듬기: 두 방향으로 세운 점들이 가장 뾰족하게 모이는 각도 (±1.5° 안)
    best = None
    for t in np.linspace(theta - math.radians(1.5), theta + math.radians(1.5), 31):
        c, s = math.cos(t), math.sin(t)
        score = 0.0
        for w in (x * c + y * s, -x * s + y * c):
            h, _ = np.histogram(w, bins=np.arange(w.min() - 0.02, w.max() + 0.03, 0.01))
            score += float(np.sort(h)[-4:].sum())
        if best is None or score > best[0]:
            best = (score, t)
    c, s = math.cos(best[1]), math.sin(best[1])
    u, v = x * c + y * s, -x * s + y * c
    spans = []
    for w, other in ((u, v), (v, u)):
        edges = np.arange(w.min() - 0.03, w.max() + 0.04, 0.01)
        h, _ = np.histogram(w, bins=edges)
        smooth = np.convolve(h, np.ones(3), mode="same")
        peaks = [i for i in range(1, len(smooth) - 1) if smooth[i] >= min_points and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]]
        if len(peaks) < 2:
            continue

        def line(i):
            on = (w >= edges[i] - 0.015) & (w <= edges[i + 1] + 0.015)
            if int(on.sum()) < 3:
                return None
            return float(np.median(w[on])), float(other[on].max() - other[on].min())

        low, high = line(peaks[0]), line(peaks[-1])
        if low is None or high is None or low[1] < min_length or high[1] < min_length:
            continue  # 가장 바깥의 선이 길지 않다 = 벽이 아니다 (상자 · 기둥 · 가려진 벽)
        spans.append(high[0] - low[0])
    return spans


class HeightVote:
    """스캔 여러 장에서 잰 값을 모아 후보(4.80 · 5.00) 가운데 하나로 정한다. 분명하지 않으면 None."""

    def __init__(self, candidates=(4.80, 5.00), tolerance=0.05, need=3, keep=12, apart=0.07):
        self.candidates = tuple(float(c) for c in candidates)
        self.tolerance, self.need, self.keep, self.apart = float(tolerance), int(need), int(keep), float(apart)
        self.values = []  # 후보 둘레(±0.15 m)에 든 잰 값 — 최근 keep 개
        self.scans = 0

    def add(self, spans):
        self.scans += 1
        low, high = min(self.candidates) - 0.15, max(self.candidates) + 0.15
        for value in spans:
            if low <= value <= high:
                self.values = (self.values + [float(value)])[-self.keep:]

    def median(self):
        return float(np.median(self.values)) if self.values else None

    def decide(self):
        """정한 높이 [m] 또는 None. 가운데값이 한 후보에 tolerance 안이고 다른 후보들과는 apart 넘게 떨어져 있어야 한다."""
        if len(self.values) < self.need:
            return None
        middle = self.median()
        near = [c for c in self.candidates if abs(middle - c) <= self.tolerance]
        if len(near) != 1 or any(abs(middle - c) < self.apart for c in self.candidates if c != near[0]):
            return None
        return near[0]
