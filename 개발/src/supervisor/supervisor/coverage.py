# 출처: sim-strategy 브랜치(e0300e3)의 team/navigation/coverage.py 를 복사했다.
# 2026-11-19 고친 곳: 설 자리는 먼저 벽 · 상자 · 기둥에서 떨어진 칸(grid.tight 가 아닌 칸)에서만 고른다 — 붙은 자리에서는 서서 돌다 모서리가 닿는다.
# 2026-11-19 (14:52 팀 테스트 뒤): 후보 간격(0.2 m)에 넓은 칸이 걸리지 않은 면은 넓은 칸을 한 칸씩(5 cm) 다시 훑은 뒤에야 붙은 칸을 쓴다 (_fine_roomy).
"""빠짐없는 정찰: 정찰 구역의 박스 면을 하나도 빼지 않고 카메라로 보는 「멈춤 지점」 을 고르고 순서를 정한다.

가정 (권우진 결정 2026-11-12): 한 곳도 빠짐없이 모든 곳을 돌아봐야 한다.
  → 보이는 면(다른 박스에 맞닿지 않은 면) 전부가 목표다. M3 는 섹터 단위 완전 일치라 한 면만 빠져도 그 섹터가 0점이다.

멈춤 지점(view) = (자리, 차체 방향). 그 자리에서 그 방향으로 서면 카메라에 「알아볼 수 있게」 들어오는 면의 집합을 미리
계산해 두고, 남은 면을 가장 싸게(이동 시간 + 돌기 + 머무는 시간) 많이 덮는 지점을 하나씩 고른다 (탐욕 집합 덮개).
가는 길에 본 면도 지워지므로, 멈춤 지점에 닿거나 새 면을 볼 때마다 다시 고른다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ViewParams:
    """카메라로 면을 알아볼 수 있는 조건. 시뮬 봇의 인식 한계(2.5 m · 40% · 65°)보다 조금 좁게 잡는다 [추정]."""

    range_m: float = 2.0
    min_range_m: float = 0.45
    max_incidence_deg: float = 55.0
    half_fov_deg: float = 50.0  # 카메라 가로 화각 128° 의 절반에서 가장자리 여유를 뺀 것
    sample_along_m: float = 0.10  # 면 가운데 ± 이만큼이 모두 보여야 「보인다」 (면의 절반 이상)
    lattice_m: float = 0.2  # 후보 자리 간격
    headings: int = 16
    dwell_s: float = 1.0  # 멈춰서 보는 시간
    speed_mps: float = 0.4  # 비용 계산용 평균 속도
    turn_rate: float = 1.5  # 비용 계산용 제자리 회전 속도 (rad/s)


def _blocked_any(px, py, qx, qy, segs):
    """선분 p→q 들이 segs 중 하나와 교차하는가. p, q: (N,) 배열. 돌려주는 값 (N,) bool."""
    ax, ay, bx, by = segs[:, 0][None, :], segs[:, 1][None, :], segs[:, 2][None, :], segs[:, 3][None, :]
    px, py, qx, qy = px[:, None], py[:, None], qx[:, None], qy[:, None]
    rx, ry = qx - px, qy - py
    sx, sy = bx - ax, by - ay
    denom = rx * sy - ry * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((ax - px) * sy - (ay - py) * sx) / denom
        u = ((ax - px) * ry - (ay - py) * rx) / denom
    hit = (np.abs(denom) > 1e-12) & (t > 1e-6) & (t < 1.0 - 1e-6) & (u > -1e-9) & (u < 1.0 + 1e-9)
    return hit.any(axis=1)


class CoveragePlanner:
    def __init__(self, fmap, grid, faces, params=None, tight=True, fine=True):
        """faces: 덮어야 할 면 목록 (fieldmap.Face).

        tight=False 면 넓은 칸(차체가 서서 돌 수 있는 자리)에서만 고른다 — 그렇게 볼 수 없는 면은 unreachable 에 남는다.
        부르는 쪽이 조건을 늦춰 넓은 칸에서 한 번 더 찾아본 뒤에 벽 · 상자에 붙은 칸을 쓰게 할 때 쓴다 (scout.FaceScout._choose).
        fine=False 면 넓은 칸을 촘촘히 다시 훑지 않는다 (2026-11-19 14:52 판 전의 방식 — 견주어 볼 때 · 되돌릴 때).
        """
        self.fmap = fmap
        self.grid = grid
        self.p = params or ViewParams()
        self.faces = list(faces)
        self.index = {f.key: k for k, f in enumerate(self.faces)}
        self.views = []  # [(x, y, 방향 rad, frozenset(면 key))]
        self.unreachable = set()  # 어느 자리에서도 볼 수 없는 면
        self.allow_tight = bool(tight)
        self.fine = bool(fine)
        self.fine_spots = 0  # 촘촘히 다시 훑어 찾은 넓은 칸의 수 (기록 · 시험용)
        self._build()

    def _build(self):
        roomy, tight = self._lattice()
        self._build_from(roomy, self.p)
        if self.unreachable and self.fine:
            # 넓은 칸을 한 칸씩(5 cm) 다시 훑는다. 후보 간격(lattice_m 0.2 m)으로는 벽을 보는 면 앞의 좁은 띠를 건너뛴다 —
            # 2026-11-19 14:52 실차: 상자 윗면과 벽 사이 0.6 m 틈에서 넓은 칸은 한 줄(벽에서 0.275 m)뿐인데 후보 줄은 벽에서 0.175 m 와 상자에서
            # 0.22 m 였다. 그래서 벽에 붙은 자리가 뽑혔고 네 곳 모두 서서 돌지 못해 면을 못 찍었다 (차체 반 대각선이 0.17 m)
            faces = [f for f in self.faces if f.key in self.unreachable]
            fine = self._fine_roomy(faces, self.p)
            self.fine_spots = len(fine)
            self._build_from(fine, self.p, only=faces)
        if not self.allow_tight:
            return
        if self.unreachable and tight:  # 넓은 자리에서는 볼 수 없는 면 → 벽 · 상자 가까운 칸에서라도
            self._build_from(tight, self.p, only=[f for f in self.faces if f.key in self.unreachable])
        if self.unreachable:
            # 벽 · 다른 박스와 40 cm 틈을 두고 마주 보는 면: 틈 안의 자리에서 비스듬히 본다 (조건을 조금 늦춘다)
            loose = ViewParams(**{**self.p.__dict__, "min_range_m": 0.30, "max_incidence_deg": 62.0})
            faces = [f for f in self.faces if f.key in self.unreachable]
            spots = []
            for face in faces:
                for depth in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
                    for k in range(-8, 9):
                        sx, sy = face.point(k * 0.05)
                        sx, sy = sx + face.nx * depth, sy + face.ny * depth
                        if self.grid.free(sx, sy):
                            spots.append((sx, sy))
            self._build_from(spots, loose, only=faces)

    def _lattice(self):
        """후보 자리 (넓은 칸들, 막힌 곳 가까운 칸들). 넓은 칸 = 부풀린 벽 · 상자 · 기둥에서 soft_m 넘게 떨어진 칸."""
        grid = self.grid
        step = max(1, int(round(self.p.lattice_m / grid.cell)))
        tight_cells = getattr(grid, "tight", ())
        roomy, tight = [], []
        for i in range(0, grid.nx, step):
            for j in range(0, grid.ny, step):
                if not grid.blocked((i, j)):
                    (tight if (i, j) in tight_cells else roomy).append(grid.center(i, j))
        return roomy, tight

    def _fine_roomy(self, faces, p):
        """faces 를 볼 수 있을 만한 넓은 칸을 격자 한 칸 간격으로 모은다 (_lattice 의 후보 간격에 걸리지 않은 칸).

        면마다 그 앞쪽(거리 · 비스듬함 조건 안)의 칸만 본다 — 시야가 가리는지는 _build_from 이 따진다.
        """
        grid = self.grid
        tight_cells = getattr(grid, "tight", ())
        step = max(1, int(round(p.lattice_m / grid.cell)))
        cos_inc = math.cos(math.radians(p.max_incidence_deg))
        cells = set()
        for face in faces:
            i0, j0 = grid.cell_of(face.cx - p.range_m, face.cy - p.range_m)
            i1, j1 = grid.cell_of(face.cx + p.range_m, face.cy + p.range_m)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    if (i % step == 0 and j % step == 0) or (i, j) in cells or (i, j) in tight_cells or grid.blocked((i, j)):
                        continue  # 후보 간격의 칸은 이미 봤다
                    x, y = grid.center(i, j)
                    dx, dy = x - face.cx, y - face.cy
                    dist = math.hypot(dx, dy)
                    if p.min_range_m <= dist <= p.range_m and dx * face.nx + dy * face.ny >= cos_inc * dist:
                        cells.add((i, j))
        return [grid.center(i, j) for i, j in sorted(cells)]

    def _build_from(self, spots, p, only=None):
        faces = self.faces if only is None else only
        segs = np.array(self.fmap.segments(), dtype=np.float64)
        if not spots or not faces:
            if only is None:
                self.unreachable = {f.key for f in self.faces}
            return
        xs = np.array([s[0] for s in spots])
        ys = np.array([s[1] for s in spots])
        cos_inc = math.cos(math.radians(p.max_incidence_deg))
        seen_by = [[] for _ in spots]  # 자리 → [(면 key, 방위)]
        for face in faces:
            dx, dy = xs - face.cx, ys - face.cy
            dist = np.hypot(dx, dy)
            facing = (dx * face.nx + dy * face.ny) / np.maximum(dist, 1e-9)
            ok = (dist <= p.range_m) & (dist >= p.min_range_m) & (facing >= cos_inc)
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            clear = np.ones(len(idx), dtype=bool)
            for along in (0.0, p.sample_along_m, -p.sample_along_m):
                qx, qy = face.point(along)
                qx += face.nx * 0.01
                qy += face.ny * 0.01
                clear &= ~_blocked_any(xs[idx], ys[idx], np.full(len(idx), qx), np.full(len(idx), qy), segs)
            for k in idx[clear]:
                seen_by[k].append((face.key, math.atan2(face.cy - ys[k], face.cx - xs[k])))
        half = math.radians(p.half_fov_deg)
        covered = set()
        for k, items in enumerate(seen_by):
            if not items:
                continue
            best = {}
            for h in range(p.headings):
                heading = -math.pi + 2.0 * math.pi * h / p.headings
                keys = frozenset(key for key, bearing in items if abs(math.atan2(math.sin(bearing - heading), math.cos(bearing - heading))) <= half)
                if keys and not any(keys <= other for other in best.values()):
                    best = {hh: ks for hh, ks in best.items() if not ks < keys}
                    best[heading] = keys
            for heading, keys in best.items():
                self.views.append((spots[k][0], spots[k][1], heading, keys))
                covered |= keys
        if only is None:
            self.unreachable = {f.key for f in self.faces} - covered
        else:
            self.unreachable -= covered

    # ------------------------------------------------------------------ 고르기
    def next_view(self, pose, remaining, weights=None, extra=(), banned=()):
        """지금 자세에서 남은 면을 덮을 다음 멈춤 지점. 돌려주는 값: (x, y, 방향, 덮는 면 집합) 또는 None.

        점수 = (덮는 남은 면의 가중치 합) / (이동 시간 + 돌기 + 머무는 시간). weights: 면 key → 가중치 (기본 1).
        """
        if not remaining:
            return None
        p = self.p
        dist = self.grid.distances((pose[0], pose[1]), extra=extra)
        best = None
        for k, (x, y, heading, keys) in enumerate(self.views):
            if k in banned:
                continue
            gain_keys = keys & remaining
            if not gain_keys:
                continue
            d = dist.get(self.grid.cell_of(x, y))
            if d is None:
                continue
            arrive = math.atan2(y - pose[1], x - pose[0]) if d > 0.15 else pose[2]
            turn = abs(math.atan2(math.sin(heading - arrive), math.cos(heading - arrive)))
            cost = d / p.speed_mps + turn / p.turn_rate + p.dwell_s
            gain = sum((weights or {}).get(key, 1.0) for key in gain_keys)
            score = gain / cost
            if best is None or score > best[0]:
                best = (score, k)
        if best is None:
            return None
        k = best[1]
        x, y, heading, keys = self.views[k]
        return k, x, y, heading, keys

    def tour(self, pose, remaining, weights=None):
        """남은 면을 다 덮을 때까지 next_view 를 이어 붙인 계획 (화면 표시 · 시간 예측용)."""
        out = []
        left = set(remaining)
        cur = pose
        banned = set()
        while left:
            pick = self.next_view(cur, left, weights, banned=banned)
            if pick is None:
                break
            k, x, y, heading, keys = pick
            out.append(pick)
            banned.add(k)
            left -= keys
            cur = (x, y, heading)
        return out
