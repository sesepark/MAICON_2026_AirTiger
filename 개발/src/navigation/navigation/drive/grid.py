"""5 cm 점유 격자 · A* 경로. 표준 라이브러리만 쓴다.

sim-strategy 브랜치 team/navigation/grid.py 를 옮겨 왔다 (박스를 반경만큼 부풀림, 좁은 칸 비용, 줄 당기기).
바뀐 점: 격자를 박스 주변만 훑어 빠르게 만든다, 임시 장애물(extra)은 칸 집합으로 받는다, 거점 금지 구역은 없다
(우리 거점 진입 금지는 supervisor 책임 — 2026-11-18 결정).
"""
from __future__ import annotations

import heapq
import math

_STEPS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))


class Grid:
    def __init__(self, spec, radius=0.15, margin=0.02, cell=0.05, soft_m=0.10, soft_cost=3.0):
        self.spec = spec
        self.cell = cell
        self.width, self.height = spec.width, spec.height
        self.nx = max(1, int(math.ceil(spec.width / cell)))
        self.ny = max(1, int(math.ceil(spec.height / cell)))
        self.grow = radius + margin
        grow = self.grow
        static = set()
        # 벽: 가장자리에서 grow 안쪽
        for i in range(self.nx):
            x = (i + 0.5) * cell
            for j in range(self.ny):
                y = (j + 0.5) * cell
                if x < grow or x > spec.width - grow or y < grow or y > spec.height - grow:
                    static.add((i, j))
        # 박스: 가운데에서 half 안쪽 (박스 주변 칸만 훑는다)
        half = spec.box_size / 2.0 + grow
        for bx, by in getattr(spec, "solids", spec.boxes):  # 박스 + 고정 기둥
            i0, i1 = self._clamp_i(int((bx - half) / cell) - 1), self._clamp_i(int((bx + half) / cell) + 1)
            j0, j1 = self._clamp_j(int((by - half) / cell) - 1), self._clamp_j(int((by + half) / cell) + 1)
            for i in range(i0, i1 + 1):
                x = (i + 0.5) * cell
                if abs(x - bx) >= half:
                    continue
                for j in range(j0, j1 + 1):
                    if abs((j + 0.5) * cell - by) < half:
                        static.add((i, j))
        self.static = static
        reach = int(math.ceil(soft_m / cell))
        offsets = [(di, dj) for di in range(-reach, reach + 1) for dj in range(-reach, reach + 1) if (di or dj) and math.hypot(di, dj) <= reach]
        tight = set()
        for i, j in static:
            for di, dj in offsets:
                c = (i + di, j + dj)
                if c not in static and 0 <= c[0] < self.nx and 0 <= c[1] < self.ny:
                    tight.add(c)
        self.tight = tight
        self.soft_cost = soft_cost
        self.soft_m = soft_m
        self.costly = frozenset()  # 이 칸에 들어가는 걸음은 costly_cost 배 (narrowed 가 채운다: 보통 격자에서는 막힌 칸)
        self.costly_cost = 1.0
        self._narrowed = {}  # 좁게 부풀린 형제 격자 (narrowed)

    def narrowed(self, grow, cost=8.0):
        """같은 지도를 grow(m)만큼만 부풀린 격자 (칸 크기 · 칸 번호는 같다 — 임시 장애물 칸 집합을 그대로 쓸 수 있다). 한 번 만들면 둔다.
        grow 가 지금보다 작지 않으면 자기 자신. 좁은 틈 계획(executor._plan)에 쓴다.
        보통 격자에서는 막힌 칸(= 구조물에 가까운 칸)은 걸음 비용이 cost 배다 — 그런 칸을 가장 적게 지나는 길, 곧 틈을 곧게 가로지르는 길이 나온다
        (비용이 같으면 상자 모서리를 0.135 m 로 스치며 비스듬히 들어가 실제 모서리에 닿았다 — 가상 gap_bench: 닿은 주기 59)."""
        if grow >= self.grow - 1e-9:
            return self
        key = (round(grow, 4), cost)
        if key not in self._narrowed:
            other = Grid(self.spec, radius=grow, margin=0.0, cell=self.cell, soft_m=self.soft_m, soft_cost=self.soft_cost)
            other.costly, other.costly_cost = frozenset(self.static - other.static), cost
            self._narrowed[key] = other
        return self._narrowed[key]

    def _clamp_i(self, i):
        return min(self.nx - 1, max(0, i))

    def _clamp_j(self, j):
        return min(self.ny - 1, max(0, j))

    # ---- 칸
    def center(self, i, j):
        return (i + 0.5) * self.cell, (j + 0.5) * self.cell

    def cell_of(self, x, y):
        return self._clamp_i(int(math.floor(x / self.cell))), self._clamp_j(int(math.floor(y / self.cell)))

    def in_bounds(self, x, y):
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def blocked(self, c, extra=()):
        return c in self.static or c in extra

    def free(self, x, y, extra=()):
        return not self.blocked(self.cell_of(x, y), extra)

    def cells_in_disc(self, x, y, radius):
        """(x, y) 둘레 radius 안의 칸 (임시 장애물 칠하기)."""
        cell = self.cell
        i0, i1 = self._clamp_i(int((x - radius) / cell)), self._clamp_i(int((x + radius) / cell))
        j0, j1 = self._clamp_j(int((y - radius) / cell)), self._clamp_j(int((y + radius) / cell))
        r2 = radius * radius
        out = []
        for i in range(i0, i1 + 1):
            cx = (i + 0.5) * cell - x
            for j in range(j0, j1 + 1):
                cy = (j + 0.5) * cell - y
                if cx * cx + cy * cy <= r2:
                    out.append((i, j))
        return out

    def nearest_free(self, x, y, extra=(), max_dist=None, accept=None):
        """(x, y) 에서 가장 가까운 빈 칸의 가운데. max_dist 를 넘으면 None.
        accept(px, py) 가 있으면 그것이 참인 칸만 본다 (돌진 목표를 옮길 때: 벽에서 멀어지지 않는 칸만 — executor._standing_shift)."""
        ci, cj = self.cell_of(x, y)
        if not self.blocked((ci, cj), extra) and (accept is None or accept(x, y)):
            return x, y
        rings = int(math.ceil((max_dist if max_dist is not None else 0.75) / self.cell)) + 1
        for ring in range(1, rings + 1):
            best = None
            for i in range(ci - ring, ci + ring + 1):
                for j in range(cj - ring, cj + ring + 1):
                    if max(abs(i - ci), abs(j - cj)) != ring or not (0 <= i < self.nx and 0 <= j < self.ny):
                        continue
                    if self.blocked((i, j), extra):
                        continue
                    px, py = self.center(i, j)
                    if accept is not None and not accept(px, py):
                        continue
                    d = math.hypot(px - x, py - y)
                    if best is None or d < best[0]:
                        best = (d, px, py)
            if best:
                if max_dist is not None and best[0] > max_dist:
                    return None
                return best[1], best[2]
        return None

    def segment_free(self, a, b, extra=()):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell * 0.5)))
        for k in range(n + 1):
            t = k / n
            if not self.free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, extra):
                return False
        return True

    def path_blocked(self, start, points, extra):
        """start → points 의 선분 중 extra(임시 장애물) 칸을 지나는 것이 있는가."""
        if not extra:
            return False
        prev = start
        for p in points or ():
            n = max(1, int(math.hypot(p[0] - prev[0], p[1] - prev[1]) / (self.cell * 0.5)))
            for k in range(n + 1):
                t = k / n
                if self.cell_of(prev[0] + (p[0] - prev[0]) * t, prev[1] + (p[1] - prev[1]) * t) in extra:
                    return True
            prev = p
        return False

    # ---- 탐색
    def _neighbors(self, cur, extra):
        ci, cj = cur
        for di, dj, cost in _STEPS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < self.nx and 0 <= nj < self.ny) or self.blocked((ni, nj), extra):
                continue
            if di and dj and (self.blocked((ci + di, cj), extra) or self.blocked((ci, cj + dj), extra)):
                continue  # 모서리를 대각선으로 스치지 않는다
            if (ni, nj) in self.costly:
                yield (ni, nj), cost * self.costly_cost
            else:
                yield (ni, nj), cost * (self.soft_cost if (ni, nj) in self.tight else 1.0)

    def plan(self, start, goal, extra=()):
        """start → goal 의 꺾이는 점 [(x, y), ...] (마지막 = goal). 길이 없으면 None.

        출발 칸은 막혀 있어도 쓴다 (벽 가까이에 섰거나 다른 UGV 가 걸쳤을 때). goal 은 막혀 있으면 안 된다
        (부르는 쪽에서 nearest_free 로 옮긴다).
        """
        s = self.nearest_free(*start, extra=extra) or self.nearest_free(*start)
        if s is None:
            return None
        s_c, g_c = self.cell_of(*s), self.cell_of(*goal)
        if self.blocked(g_c, extra):
            return None
        if s_c == g_c:
            return [tuple(goal)]
        heap = [(0.0, 0.0, s_c)]
        came = {s_c: None}
        cost = {s_c: 0.0}
        while heap:
            _f, c0, cur = heapq.heappop(heap)
            if cur == g_c:
                break
            if c0 > cost[cur] + 1e-9:
                continue
            for nxt, step in self._neighbors(cur, extra):
                c1 = c0 + step
                if c1 < cost.get(nxt, 1e18):
                    cost[nxt] = c1
                    came[nxt] = cur
                    dx, dy = abs(nxt[0] - g_c[0]), abs(nxt[1] - g_c[1])
                    heapq.heappush(heap, (c1 + max(dx, dy) + 0.4142 * min(dx, dy), c1, nxt))
        if g_c not in came:
            return None
        cells = []
        cur = g_c
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        cells.reverse()
        points = [self.center(*c) for c in cells]
        points[0] = (float(start[0]), float(start[1]))
        points[-1] = (float(goal[0]), float(goal[1]))
        # 줄 당기기: 곧게 갈 수 있는 가장 먼 점으로 건너뛴다.
        # 2026-11-19 (팀장 요청으로 supervisor 쪽에서 고침 — navigation 담당이 다시 봐 주세요): 전에는 「막힌 칸만 안 지나면」 건너뛰어서,
        #   탐색이 넓은 길(tight 가 아닌 칸)로 돌아 놓은 것을 줄 당기기가 다시 상자 모서리에 붙여 버렸다 → 부풀린 가장자리(0.17 m)를 따라가
        #   차체 모서리 여유가 0 (시험기록/2026-11-18_팀테스트_재현 #5, 실차 2차 판에서 가운데가 상자 변에서 13 cm).
        #   이제 건너뛰는 선분이 tight 칸(상자·벽에서 soft_m 안)을 지나면 안 된다. 다만 탐색 경로 자체가 그 사이 내내 tight 칸이면
        #   (40 cm 틈 · 벽 가까이의 출발/도착) 그 구간은 전처럼 곧게 편다 — 좁은 틈은 여전히 지나간다.
        tight = [self.cell_of(*q) in self.tight or self.blocked(self.cell_of(*q), extra) for q in points]
        out = []
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            # 같은 조건, 순서만 (navigation 담당의 검토 제안 — docs/줄_당기기_검토_4a5e3d0_2026-11-19.md 4절): 싼 것(tight 목록) →
            # 첫 tight 칸에서 바로 끝나는 roomy → 끝까지 훑는 segment_free. 경로는 같고 계획 시간이 4a5e3d0 이전 수준으로 돌아간다
            while j > i + 1 and not ((all(tight[i:j + 1]) or self._segment_roomy(points[i], points[j]))
                                     and self.segment_free(points[i], points[j], extra)
                                     and not (self.costly and self._segment_costly(points[i], points[j]))):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def _segment_costly(self, a, b):
        """a → b 선분이 costly 칸(좁게 부풀린 격자에서만 열린 칸 — narrowed)을 지나는가. 그런 칸은 건너뛰지 않고 탐색이 고른 칸을 하나씩 따라간다:
        틈 속은 탐색이 고른 대로 곧게, 틈 밖에서는 보통 여유(0.17 m)를 지킨 선분만 편다 (틈을 나오며 모서리를 0.125 m 로 질러가지 않게)."""
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell * 0.5)))
        for k in range(n + 1):
            t = k / n
            if self.cell_of(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) in self.costly:
                return True
        return False

    def _segment_roomy(self, a, b):
        """a → b 선분이 tight 칸(상자·벽 가까이)을 지나지 않는가."""
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell * 0.5)))
        for k in range(n + 1):
            t = k / n
            if self.cell_of(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) in self.tight:
                return False
        return True
