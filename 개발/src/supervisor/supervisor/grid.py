# 출처: sim-strategy 브랜치(e0300e3)의 team/navigation/grid.py 를 그대로 복사했다. 고칠 때는 navigation 쪽과 같은 판인지 확인한다.
"""점유 격자 · A* 경로 · 거리장. 표준 라이브러리만 쓴다 (Jetson 에서 그대로 돈다).

박스는 로봇 반지름 + 여유만큼 부풀려 막는다. 박스와 벽 · 박스와 박스 사이는 최소 40 cm 라 그 틈은 칸을 잘게(5 cm) 나눠야
지나갈 수 있다 → 막힌 곳에 가까운 칸은 비용을 높여(soft) 넓은 길을 먼저 고르고, 벽을 보는 면을 보러 갈 때만 틈에 들어간다.
거점 금지 구역처럼 「들어가면 안 되는 곳」 은 zones 로 따로 막고
켜고 끌 수 있다 (예: 목표거점이 게시되면 그 거점만 연다).
"""
from __future__ import annotations

import heapq
import math

_STEPS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))


class Grid:
    def __init__(self, fmap, radius=0.15, margin=0.02, cell=0.05, soft_m=0.10, soft_cost=3.0):
        self.cell = cell
        self.nx = int(math.ceil(fmap.width / cell))
        self.ny = int(math.ceil(fmap.height / cell))
        self.width, self.height = fmap.width, fmap.height
        grow = radius + margin
        self.grow = grow  # 벽 · 상자를 부풀린 폭 (뒤에 넣는 상자의 구역도 같은 폭으로 막는다 — mission 의 지도에 없는 상자)
        half = fmap.box_size / 2.0 + grow
        solids = list(fmap.boxes) + list(getattr(fmap, "pillars", ()) or ())  # 박스 + 고정 기둥 (2026-11-19)
        self.static = set()
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.center(i, j)
                if x < grow or x > fmap.width - grow or y < grow or y > fmap.height - grow:
                    self.static.add((i, j))
                    continue
                for bx, by in solids:
                    if abs(x - bx) < half and abs(y - by) < half:
                        self.static.add((i, j))
                        break
        self.zones = {}  # 이름 → 막는 칸 집합
        self.closed = set()  # 지금 막힌 구역 이름
        self._blocked = set(self.static)
        # 막힌 칸에서 soft_m 안쪽의 칸은 지나가는 비용을 soft_cost 배로
        reach = int(math.ceil(soft_m / cell))
        self._soft_reach = reach
        self.tight = set()
        for i, j in self.static:
            for di in range(-reach, reach + 1):
                for dj in range(-reach, reach + 1):
                    c = (i + di, j + dj)
                    if c not in self.static and 0 <= c[0] < self.nx and 0 <= c[1] < self.ny and math.hypot(di, dj) <= reach:
                        self.tight.add(c)
        self.soft_cost = soft_cost
        self._links = None  # distances() 가 쓰는 이웃 목록 (_link_table). 막힌 칸이 바뀌면 None 으로 되돌린다

    # ---- 칸
    def center(self, i, j):
        return (i + 0.5) * self.cell, (j + 0.5) * self.cell

    def cell_of(self, x, y):
        return min(self.nx - 1, max(0, int(x / self.cell))), min(self.ny - 1, max(0, int(y / self.cell)))

    def add_zone(self, name, rect, closed=True, soft=False):
        """soft: 구역 둘레 soft_m 안의 칸도 「막힌 곳에 가까운 칸」 으로 친다 (지도에 없던 상자를 뒤에 넣을 때 — 상자 곁에는 설 자리를 잡지 않게)."""
        x0, y0, x1, y1 = rect
        cells = set()
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.center(i, j)
                if x0 <= x <= x1 and y0 <= y <= y1:
                    cells.add((i, j))
        self.zones[name] = cells
        if soft:
            reach = self._soft_reach
            for i, j in cells:
                for di in range(-reach, reach + 1):
                    for dj in range(-reach, reach + 1):
                        c = (i + di, j + dj)
                        if c not in cells and c not in self.static and 0 <= c[0] < self.nx and 0 <= c[1] < self.ny and math.hypot(di, dj) <= reach:
                            self.tight.add(c)
        self.set_zone(name, closed)

    def set_zone(self, name, closed):
        if closed:
            self.closed.add(name)
        else:
            self.closed.discard(name)
        blocked = set(self.static)
        for zone in self.closed:
            blocked |= self.zones.get(zone, set())
        self._blocked = blocked
        self._links = None

    def blocked(self, c, extra=()):
        return c in self._blocked or c in extra

    def free(self, x, y, extra=()):
        return not self.blocked(self.cell_of(x, y), extra)

    def nearest_free(self, x, y, extra=(), rings=15):
        ci, cj = self.cell_of(x, y)
        for ring in range(rings):
            best = None
            for i in range(ci - ring, ci + ring + 1):
                for j in range(cj - ring, cj + ring + 1):
                    if max(abs(i - ci), abs(j - cj)) != ring or not (0 <= i < self.nx and 0 <= j < self.ny):
                        continue
                    if self.blocked((i, j), extra):
                        continue
                    px, py = self.center(i, j)
                    d = math.hypot(px - x, py - y)
                    if best is None or d < best[0]:
                        best = (d, px, py)
            if best:
                return best[1], best[2]
        return None

    def segment_free(self, a, b, extra=()):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell * 0.5)))
        for k in range(n + 1):
            t = k / n
            if not self.free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, extra):
                return False
        return True

    # ---- 탐색
    def _neighbors(self, cur, extra):
        ci, cj = cur
        for di, dj, cost in _STEPS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < self.nx and 0 <= nj < self.ny) or self.blocked((ni, nj), extra):
                continue
            if di and dj and (self.blocked((ci + di, cj), extra) or self.blocked((ci, cj + dj), extra)):
                continue  # 모서리를 대각선으로 스치지 않는다
            yield (ni, nj), cost * (self.soft_cost if (ni, nj) in self.tight else 1.0)

    def _link_table(self):
        """칸 번호(i × ny + j) → [(이웃 칸 번호, 그 이웃으로 가는 비용 m), ...]. 막히지 않은 칸만 들어 있다.

        distances() 가 칸마다 _neighbors() 를 다시 부르지 않게 한 번 만들어 두고 쓴다. 이웃의 순서와 비용은 _neighbors() 와 같다
        (비용은 「걸음 × 좁은 칸 배수 × 칸 크기」 를 같은 순서로 곱해 둔 값). 구역을 열고 닫으면(set_zone) 다시 만든다.
        """
        if self._links is not None:
            return self._links
        nx, ny, cell, blocked, tight, soft = self.nx, self.ny, self.cell, self._blocked, self.tight, self.soft_cost
        links = {}
        for ci in range(nx):
            for cj in range(ny):
                if (ci, cj) in blocked:
                    continue
                out = []
                for di, dj, cost in _STEPS:
                    ni, nj = ci + di, cj + dj
                    if not (0 <= ni < nx and 0 <= nj < ny) or (ni, nj) in blocked:
                        continue
                    if di and dj and ((ci + di, cj) in blocked or (ci, cj + dj) in blocked):
                        continue  # 모서리를 대각선으로 스치지 않는다
                    out.append((ni * ny + nj, cost * (soft if (ni, nj) in tight else 1.0) * cell))
                links[ci * ny + cj] = out
        self._links = links
        return links

    def distances(self, start, extra=(), limit=None):
        """start(x, y) 에서 모든 칸까지의 경로 비용(m, 좁은 칸은 비싸게) — 다익스트라. {칸: 비용}."""
        s = self.nearest_free(*start, extra=extra)
        if s is None:
            return {}
        s_c = self.cell_of(*s)
        if not extra:
            # 임시 장애물이 없으면 미리 만든 이웃 목록으로 돈다 (2026-11-19: 정찰 계획이 이 함수를 자리 수 + 2번 부른다 — 실차에서 계획 한 번에
            # 1.5 ~ 2.7 s 동안 틱이 멈췄다). 칸을 (i, j) 대신 번호 i × ny + j 로 다룬다 — 번호의 크기 순서가 (i, j) 의 순서와 같아
            # 비용이 같은 칸을 꺼내는 순서도 아래의 원래 계산과 같고, 돌려주는 값(비용 · 칸이 적히는 순서)도 같다
            links, ny = self._link_table(), self.ny
            first = s_c[0] * ny + s_c[1]
            cost_of = {first: 0.0}
            heap = [(0.0, first)]
            pop, push, known = heapq.heappop, heapq.heappush, cost_of.get
            while heap:
                d0, cur = pop(heap)
                if d0 > cost_of[cur] + 1e-9:
                    continue
                if limit is not None and d0 > limit:
                    break
                for nxt, step in links.get(cur, ()):
                    d1 = d0 + step
                    if d1 < known(nxt, 1e18):
                        cost_of[nxt] = d1
                        push(heap, (d1, nxt))
            return {divmod(k, ny): d for k, d in cost_of.items()}
        dist = {s_c: 0.0}
        heap = [(0.0, s_c)]
        while heap:
            d0, cur = heapq.heappop(heap)
            if d0 > dist[cur] + 1e-9:
                continue
            if limit is not None and d0 > limit:
                break
            for nxt, cost in self._neighbors(cur, extra):
                d1 = d0 + cost * self.cell
                if d1 < dist.get(nxt, 1e18):
                    dist[nxt] = d1
                    heapq.heappush(heap, (d1, nxt))
        return dist

    def plan(self, start, goal, extra=()):
        """start → goal 의 꺾이는 점 [(x, y), ...] (goal 포함). 길이 없으면 None."""
        s = self.nearest_free(*start)  # 출발 칸은 막혀 있어도(다른 UGV 가 걸쳐도) 쓴다
        g = self.nearest_free(*goal, extra=extra)
        if s is None or g is None:
            return None
        s_c, g_c = self.cell_of(*s), self.cell_of(*g)
        if s_c == g_c:
            return [g]
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
        points[-1] = g
        out = []  # 줄 당기기: 곧게 보이는 가장 먼 점으로 건너뛴다
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_free(points[i], points[j], extra):
                j -= 1
            out.append(points[j])
            i = j
        return out


def path_length(start, points):
    total, prev = 0.0, start
    for p in points or ():
        total += math.hypot(p[0] - prev[0], p[1] - prev[1])
        prev = p
    return total
