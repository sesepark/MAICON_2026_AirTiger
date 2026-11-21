"""/scan → 지도에 없는 물체(다른 UGV · 드론이 빠뜨린 박스) → 임시 장애물 칸 · 충돌 정지 판단.

지도로 설명되는 점 = 벽 · 박스 면에서 map_explain_m 안의 점. 나머지가 「지도에 없는 점」 이다.
임시 장애물은 obstacle_ttl_s 동안 남고, 같은 자리에서 다시 보이면 시간이 늘어난다.
라이다가 그 자리를 볼 수 없으면(뒤쪽 가린 90° · 값 없는 광선 · 더 가까운 것에 가림) 마지막으로 찍힌 뒤 obstacle_memory_s 까지 잊지 않는다.
광선이 그 자리 너머까지 닿으면(= 비었다) 전처럼 obstacle_ttl_s 뒤 사라지고, 우리 몸이 올라선 자리는 바로 지운다 (2026-11-19).
  왜: 오전 실차 두 판(10:50 · 11:19)의 출발 직후, 출발 앞의 물체를 피해 한쪽 우회로로 돌면 그 물체가 뒤쪽(못 보는 쪽)으로 가 2.5 s 뒤
  잊혔고, 그러면 그쪽 길이 짧아 보여 되돌아 돌기를 3 ~ 4 s 마다 되풀이했다 (41 s · 48 s). 기억은 길 찾기에만 쓴다 — 서기 · 줄이기는
  지금 스캔의 점만 보고, 기억한 칸 때문에 길이 없으면 그 칸을 빼고 짠 길로 가다 앞에서 본다 (executor._plan).
FieldMap 의 다른 UGV(드론 화면)도 vehicle_ttl_s 동안 임시 장애물로 넣는다 (LiDAR 로 보이기 전에 미리 피한다).
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Pose2D


def scan_to_body(ranges, angle_min, angle_increment, params, range_min=0.0, range_max=float("inf"), sweep=None):
    """LaserScan → 차체(base_link) 기준 점 [(x, y)] (앞 +x, 왼쪽 +y). numpy 로 한 번에 (2026-11-19, 점 505개를 파이썬으로 돌던 것).

    sweep = (dx, dy, dyaw): 스캔 한 바퀴의 **첫 광선 때의 차체 자세를 끝 광선 때의 차체에서 본 것**. 주면 광선마다 찍힌 때의 자세로 되돌려
    끝 광선 때의 차체 좌표로 옮긴다 (한 바퀴 0.1 s 동안 차체가 돈 만큼 스캔이 휘는 것을 편다 — 60°/s 로 돌면 첫 광선과 끝 광선이 6° 어긋난다)."""
    p = params
    yaw0 = math.radians(p.lidar_yaw_deg)
    sign = -1.0 if p.lidar_flip else 1.0
    lo = max(p.scan_self_mask_m, range_min)
    hi = min(p.scan_max_range, range_max if range_max > 0 else float("inf"))
    r = np.asarray(ranges, dtype=float)
    if r.size == 0:
        return []
    r_all = r.size
    k = np.nonzero(np.isfinite(r) & (r >= lo) & (r <= hi))[0]  # NaN · inf · 범위 밖은 버린다
    r = r[k]
    a = yaw0 + sign * (angle_min + k * angle_increment)
    x, y = p.lidar_x + r * np.cos(a), r * np.sin(a)
    margin = getattr(p, "scan_self_margin_m", 0.0)
    if margin > 0.0:  # 차체 사각형 안(여유 포함)에 찍힌 점은 우리 몸에 달린 것이다 — 거리 하나(scan_self_mask_m)로는 모서리 쪽이 남는다
        out = (np.abs(x) > p.robot_half_length + margin) | (np.abs(y) > p.robot_half_width + margin)
        x, y, k = x[out], y[out], k[out]
    if sweep is not None and r_all > 1:
        left = 1.0 - k / float(r_all - 1)   # 그 광선 뒤로 스캔이 끝날 때까지 남은 몫 (끝 광선 0)
        turn = sweep[2] * left
        c, s = np.cos(turn), np.sin(turn)
        x, y = sweep[0] * left + c * x - s * y, sweep[1] * left + s * x + c * y
    return list(zip(x.tolist(), y.tolist()))


def body_to_world(pose: Pose2D, pts):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return [(pose.x + c * x - s * y, pose.y + s * x + c * y) for x, y in pts]


def structure_gap(spec, x, y):
    """MapSpec.distance_to_structure 를 점 배열 x, y 에 한 번에 (같은 식): 가장 가까운 벽 · 박스 · 고정 기둥 면까지, 박스 안이면 0."""
    inside = (x >= 0.0) & (x <= spec.width) & (y >= 0.0) & (y <= spec.height)
    d = np.where(inside, np.minimum(np.minimum(x, spec.width - x), np.minimum(y, spec.height - y)),
                 np.hypot(np.maximum(np.maximum(-x, x - spec.width), 0.0), np.maximum(np.maximum(-y, y - spec.height), 0.0)))
    solids = np.asarray(spec.solids, dtype=float).reshape(-1, 2)
    if len(solids):
        half = spec.box_size / 2.0
        ax = np.abs(x[:, None] - solids[None, :, 0]) - half
        ay = np.abs(y[:, None] - solids[None, :, 1]) - half
        d = np.minimum(d, np.hypot(np.maximum(ax, 0.0), np.maximum(ay, 0.0)).min(axis=1))
        d = np.where(((ax <= 0.0) & (ay <= 0.0)).any(axis=1), 0.0, d)
    return d


class ObstacleTracker:
    def __init__(self, params):
        self.p = params
        self._seeds = {}  # 점이 찍힌 칸 → [막을 반지름, 사라질 시각, 마지막으로 찍힌 시각, 처음 찍힌 시각]
        self._discs = {}  # (칸, 반지름) → 막는 칸들 (grid 가 바뀌면 비운다)
        self._discs_grid = None
        self.ugvs = None  # 드론이 본 UGV 궤적 (drive/ugvs.UgvTracks) — 주행 노드가 넣어 준다. None 이면 아래 드론 관련 일은 모두 건너뛴다
        self._ugv_cells = set()  # 찍힐 때 그 곁에 드론이 본 UGV 가 있던 임시 장애물 칸 (sync_ugvs)
        self._ugv_sync = None  # 마지막으로 맞춘 (ugvs.version, 스캔 시각)
        self.dropped_ugv_cells = 0  # UGV 가 떠나 지운 기억 칸 수 (기록용)
        self._ugv_plan = ()  # 길 고르기에 넣는, 오래 서 있는 UGV 들의 자리 ((x, y), ...) — sync_ugvs 가 정한다 (ugv_plan_parked_s)
        self._vehicles = []  # [(x, y)]
        self._vehicles_t = None
        self.unexplained_body = []  # 마지막 스캔의 지도에 없는 점 (차체 기준)
        self.unexplained_world = []
        self._far_world = []  # 그 가운데 지도 구조물에서 displaced_m 넘게 떨어진 점 (어긋난 상자 면이 아닌 것) — 붙든 UGV 확인용 (_check_held)
        self.side_margin = None  # 진행 띠의 옆 여유를 잠깐 바꿀 때 (통로 — executor 가 주기마다 정한다). None = params.stop_side_margin
        self.scan_t = None
        self.version = 0  # 임시 장애물이 바뀔 때마다 1 씩
        self._next_expiry = None  # 가장 먼저 사라질 칸의 시각 (이보다 이르면 사라질 칸을 찾지 않는다). 늦게 잡히지는 않는다
        self._extra = frozenset()  # extra_cells 의 마지막 결과와 그 열쇠 (version, 드론 UGV 를 넣었나, grid)
        self._extra_key = None
        self._tight = frozenset()  # tight_cells 의 마지막 결과와 그 열쇠
        self._fresh_set = frozenset()  # fresh_cells 의 마지막 결과와 그 열쇠
        self._fresh_key = None
        self._tight_key = None
        self.skipped_turning = 0  # 빨리 도는 동안이라 새로 막지 않은 칸 수 (기록용, scan_turn_skip_rate)

    def clear(self):
        self._seeds.clear()
        self._next_expiry = None
        self.unexplained_body = []
        self.unexplained_world = []
        self._far_world = []
        self.version += 1

    def on_vehicles(self, t, others):
        """FieldMap.vehicles 의 다른 UGV. 지도는 위치가 나갈 때마다(약 10 Hz) 다시 오지만 perception 은 이 목록을 비워 보낸다 —
        목록이 바뀐 때만 version 을 올린다 (전: 올 때마다 올려, 임시 장애물 칸 집합을 지도 메시지마다 다시 만들었다 — 2026-11-20)."""
        others = list(others)
        if others != self._vehicles:
            self._vehicles = others
            self.version += 1
        self._vehicles_t = t

    def on_scan(self, t, pose: Pose2D, body_pts, spec, grid, rays=None, turn_rate=0.0):
        """스캔 한 번. pose = 스캔 시각의 자세 (arena). 점마다의 지도 거리는 numpy 로 한 번에 (2026-11-19 — 결과는 전과 같다).

        turn_rate = 그 스캔 동안 차체가 돌던 빠르기 (rad/s). scan_turn_skip_rate 넘게 돌던 스캔의 점은 **새 임시 장애물 칸을 만들지 않는다**
        (이미 있는 칸은 이어 간다). 빨리 도는 동안에는 스캔을 지도에 놓는 방향이 몇 도씩 틀려 벽 · 상자가 통째로 「지도에 없는 점」 으로 찍힌다
        (params.scan_turn_skip_rate 의 설명). 정지 · 감속 · 서서 돌기 확인은 전처럼 그 스캔의 점을 그대로 본다."""
        p = self.p
        fast = p.scan_turn_skip_rate > 0.0 and abs(turn_rate) >= p.scan_turn_skip_rate
        keep_b, keep_w, far_w = [], [], []
        seen_cells, shifted_cells = set(), set()
        if body_pts:
            b = np.asarray(body_pts, dtype=float).reshape(-1, 2)
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            wx = pose.x + c * b[:, 0] - s * b[:, 1]
            wy = pose.y + s * b[:, 0] + c * b[:, 1]
            inside = (wx >= 0.0) & (wx <= spec.width) & (wy >= 0.0) & (wy <= spec.height)
            gap = structure_gap(spec, wx, wy)
            keep = inside & (gap > p.map_explain_m + p.map_explain_per_m * np.hypot(b[:, 0], b[:, 1]))
            if p.wall_explain_m > 0.0:  # 외벽 바로 안쪽의 점은 벽이다
                keep &= np.minimum(np.minimum(wx, spec.width - wx), np.minimum(wy, spec.height - wy)) > p.wall_explain_m
            idx = np.nonzero(keep)[0]
            keep_b = [tuple(v) for v in b[idx].tolist()]
            keep_w = list(zip(wx[idx].tolist(), wy[idx].tolist()))
            centres = self.ugv_centres(t)   # 드론이 본 UGV 들의 가운데 — 그 곁의 점은 구조물 곁이어도 UGV 다 (아래)
            for (x, y), g in zip(keep_w, gap[idx].tolist()):
                # 지도의 구조물 바로 곁의 점 = 그 구조물이 조금 어긋나 있는 것 (면 자체) → 작게 막는다. 떨어진 점 = 다른 UGV 등 → 크게.
                # 다만 드론이 본 UGV 곁(ugv_match_m)의 점은 구조물 곁이어도 크게 막는다 (2026-11-20): 상자 곁 displaced_m(0.30 m) 안에 선 UGV 의 점이
                # 「어긋난 상자 면」 이 되어 0.03 m 만 막혔다 — 보이는 것은 그 UGV 의 한 면이고 몸통은 그 뒤로 이어진다.
                # 00:00 ~ 00:52 실차 세 판 내내 (3.05, 2.56) 에 선 UGV 가 아래 상자 윗면에서 0.32 m 였다 (그 UGV 의 상자 쪽 절반이 이 경우다)
                small = g <= p.displaced_m and not any(math.hypot(x - ux, y - uy) <= p.ugv_match_m for ux, uy in centres)
                (shifted_cells if small else seen_cells).add(grid.cell_of(x, y))
                if g > p.displaced_m:
                    far_w.append((x, y))
        self.unexplained_body = keep_b
        self.unexplained_world = keep_w
        self._far_world = far_w
        self.scan_t = t
        hit = seen_cells | shifted_cells
        if hit:
            until = t + p.obstacle_ttl_s
            for cells, radius in ((seen_cells, p.obstacle_radius + grid.grow), (shifted_cells - seen_cells, p.displaced_radius + grid.grow)):
                for c in cells:
                    seed = self._seeds.get(c)
                    if seed is None:
                        if fast:
                            self.skipped_turning += 1
                            continue
                        self._seeds[c] = [radius, until, t, t]
                    else:
                        seed[0], seed[1], seed[2] = max(seed[0], radius), max(seed[1], until), t
            self.version += 1
        if self._seeds and self._drop_under_body(t, pose, grid):
            self.version += 1
        if rays is not None and p.obstacle_memory_s > 0.0 and len(self._seeds) > len(hit):
            self._keep_unseen(t, pose, grid, rays, hit)
        self._next_expiry = min(seed[1] for seed in self._seeds.values()) if self._seeds else None
        if self.ugvs is not None and not fast:
            self._check_held(t, pose, rays)

    def _check_held(self, t, pose, rays):
        """드론 사진이 끊긴 뒤 붙들고 있는 UGV(drive/ugvs.UgvTracks._held)를 이번 스캔으로 확인한다 — 라이다가 진실이다.

        - 그 자리(ugv_match_m)에 구조물에서 떨어진 지도에 없는 점이 찍혔다 → 아직 있다 (붙드는 시간을 지금부터 다시)
        - 우리 몸이 그 자리에 올라섰다, 또는 그 자리 폭(차체 반폭)만큼의 광선이 모두 그 너머 ugv_hold_clear_m 넘게 닿았다 → 비었다.
          ugv_hold_clear_scans 번 이어지면 지운다
        - 값 없는 광선(뒤쪽 90° · 반사 없음) · 더 가까운 것에 가림 · 스캔 범위 밖 → 알 수 없다 (그대로 둔다 — ugv_hold_s 뒤 잊는다)
        사진이 오고 있는 동안에는 아무것도 하지 않는다 (드론이 사진마다 고친다). 빨리 도는 동안의 스캔으로는 부르지 않는다 (방향이 몇 도 틀린다)."""
        ugvs, p = self.ugvs, self.p
        if not ugvs.holding(t):
            return
        far = np.asarray(self._far_world, dtype=float).reshape(-1, 2)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        lx, ly = pose.x + c * p.lidar_x, pose.y + s * p.lidar_x
        sign = -1.0 if p.lidar_flip else 1.0
        for u in ugvs.fresh(t):
            if len(far) and bool((np.hypot(far[:, 0] - u.x, far[:, 1] - u.y) <= p.ugv_match_m).any()):
                ugvs.confirm(u, t)
                continue
            if math.hypot(u.x - pose.x, u.y - pose.y) <= p.robot_half_width:
                ugvs.refute(u)   # 우리가 그 자리에 서 있다
                continue
            clear = False
            dist = math.hypot(u.x - lx, u.y - ly)
            if rays is not None and p.scan_self_mask_m < dist <= p.scan_max_range - p.ugv_hold_clear_m:
                angle_min, increment, ranges = rays
                n = len(ranges)
                if n and increment > 0.0:
                    body = math.atan2(u.y - ly, u.x - lx) - pose.yaw
                    k0 = int(round(math.fmod((body - math.radians(p.lidar_yaw_deg)) * sign - angle_min, 2.0 * math.pi) % (2.0 * math.pi) / increment))
                    dk = max(1, int(math.ceil(math.atan2(p.robot_half_width, dist) / increment)))
                    full = n * increment >= 2.0 * math.pi - 2.0 * increment
                    ks = [(k0 + j) % n if full else k0 + j for j in range(-dk, dk + 1)]
                    if all(0 <= k < n for k in ks):
                        r = np.asarray([ranges[k] for k in ks], dtype=float)
                        clear = bool(np.all(np.isfinite(r) & (r > dist + p.ugv_hold_clear_m)))
            ugvs.refute(u, clear)

    def _drop_under_body(self, t, pose, grid):
        """지금 우리 몸이 있는 자리(가장자리 2 cm 안쪽)의 칸은 비어 있다 — 지운다. 라이다는 몸 바로 곁(scan_self_mask_m)을 못 봐서,
        이런 칸은 「비었다」 를 확인할 길이 없어 기억으로 오래 남는다 (실차 bag 1118_201710: 지운 적 없는 칸 위를 29번 지나감).

        몸 둘레 memory_near_m 안의 칸도, 기억으로만 남은 것(마지막으로 찍힌 지 obstacle_ttl_s 넘음)이면 지운다. 그렇게 가까운 칸은 라이다가
        「비었다」 를 확인하지 못해(가림 · 몸 곁 · 뒤쪽 가장자리) 기억이 끝까지 붙든다. 2026-11-19 14:52 실차: 위쪽 벽과 상자 2 · 3 사이 0.6 m 통로
        입구에 회전 중 찍힌 점의 칸이 3 ~ 11 s 남아 통로 길(0.5 m)을 막고 4.4 m 우회로를 짰다 (그 판 스캔 재생: 기억을 끄면 곧은 길).
        가까운 진짜 물체는 보이는 쪽이면 스캔마다 다시 찍혀 지워지지 않는다. 정지 · 감속 · 서서 돌기 확인은 기억을 쓰지 않는다 (지금 스캔만)."""
        p = self.p
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        hl, hw = p.robot_half_length - 0.02, p.robot_half_width - 0.02
        near = max(0.0, p.memory_near_m)
        nl, nw = p.robot_half_length + near, p.robot_half_width + near
        reach = max(0.2, nl + grid.cell, nw + grid.cell)
        drop = []
        for cell, seed in self._seeds.items():
            cx, cy = grid.center(*cell)
            dx, dy = cx - pose.x, cy - pose.y
            if abs(dx) > reach or abs(dy) > reach:
                continue
            bx, by = abs(c * dx + s * dy), abs(-s * dx + c * dy)
            if (bx <= hl and by <= hw) or (near > 0.0 and t - seed[2] > p.obstacle_ttl_s and bx <= nl and by <= nw):
                drop.append(cell)
        for cell in drop:
            del self._seeds[cell]
        return bool(drop)

    def _keep_unseen(self, t, pose, grid, rays, hit):
        """이번 스캔에 안 찍힌 칸 중 라이다가 그 자리를 「비었다」 고 볼 수 없는 칸(뒤쪽 가린 구간 · 값 없는 광선 · 더 가까운 것에 가림)은
        잊지 않는다 — 마지막으로 찍힌 뒤 obstacle_memory_s 까지 obstacle_ttl_s 씩 늘린다. 광선이 그 칸 너머까지 닿았으면(memory_clear_m)
        늘리지 않는다 = 전처럼 obstacle_ttl_s 뒤 사라진다.

        잠깐만 찍힌 칸(처음 찍힌 때부터 마지막으로 찍힌 때까지 memory_min_seen_s 가 안 됨)은 늘리지 않는다 — 방향이 틀린 한두 스캔에만 찍힌
        벽 · 상자의 헛칸이다 (2026-11-19 밤, params.memory_min_seen_s 의 설명)."""
        p = self.p
        angle_min, increment, ranges = rays
        n = len(ranges)
        if n == 0 or increment <= 0.0:
            return
        cells = [c for c in self._seeds if c not in hit]
        xy = np.asarray([grid.center(*c) for c in cells], dtype=float)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        dx, dy = xy[:, 0] - (pose.x + c * p.lidar_x), xy[:, 1] - (pose.y + s * p.lidar_x)
        dist = np.hypot(dx, dy)
        body = np.arctan2(dy, dx) - pose.yaw
        sign = -1.0 if p.lidar_flip else 1.0
        lidar = np.mod((body - math.radians(p.lidar_yaw_deg)) * sign - angle_min, 2.0 * math.pi)
        k = np.round(lidar / increment).astype(int)
        k = np.where(k == n, 0, k)
        ok = k < n
        r = np.full(len(cells), np.nan)
        r[ok] = ranges[k[ok]]
        through = np.isfinite(r) & (r > p.scan_self_mask_m) & (r > dist + p.memory_clear_m)
        for cell, seen_through in zip(cells, through.tolist()):
            if not seen_through:
                seed = self._seeds[cell]
                if p.memory_min_seen_s > 0.0 and seed[2] - seed[3] < p.memory_min_seen_s:
                    continue  # 잠깐만 찍힌 칸: 기억으로 붙들지 않는다 (obstacle_ttl_s 뒤 사라진다)
                seed[1] = max(seed[1], min(t + p.obstacle_ttl_s, seed[2] + p.obstacle_memory_s))

    def _disc(self, grid, cell, radius):
        if self._discs_grid is not grid:
            self._discs, self._discs_grid = {}, grid
        key = (cell, radius)
        out = self._discs.get(key)
        if out is None:
            cx, cy = grid.center(*cell)
            out = self._discs[key] = tuple(grid.cells_in_disc(cx, cy, radius))
        return out

    def extra_cells(self, t, grid):
        """지금 막을 임시 장애물 칸 집합 (읽기만 할 것 — 바뀌지 않았으면 지난번 것을 그대로 돌려준다)."""
        self.sync_ugvs(t, grid)
        if self._next_expiry is not None and t >= self._next_expiry:
            expired = [c for c, seed in self._seeds.items() if seed[1] <= t]
            for c in expired:
                del self._seeds[c]
            if expired:
                self.version += 1
            self._next_expiry = min(seed[1] for seed in self._seeds.values()) if self._seeds else None
        vehicles = self._vehicles_t is not None and t - self._vehicles_t <= self.p.vehicle_ttl_s
        key = (self.version, vehicles, grid)
        if key != self._extra_key:
            out = set()
            for c, seed in self._seeds.items():
                out.update(self._disc(grid, c, seed[0]))
            if vehicles:
                radius = self.p.vehicle_radius + grid.grow
                for x, y in self._vehicles:
                    out.update(grid.cells_in_disc(x, y, radius))
            out.update(self._ugv_plan_cells(grid))
            self._extra, self._extra_key = frozenset(out), key
        return self._extra

    def fresh_cells(self, t, grid, pose=None):
        """지금 라이다에 찍히고 있는 임시 장애물만 (마지막으로 찍힌 지 obstacle_ttl_s 안 — 기억으로만 남은 칸은 뺀다).
        기억으로만 남은 칸이 없으면 extra_cells 와 같은 집합을 그대로 돌려준다 (is 로 견줄 수 있다).

        pose 를 주면 차체 뒤쪽(라이다가 못 보는 구간 ± memory_rear_deg)의 기억은 남긴다 — 기억을 넣은 까닭(727f6d4: 피해 돌아간 물체가
        뒤로 가면 잊혀 두 우회로를 오가던 것)이 바로 그 칸들이다. 그 밖의 기억(값 없는 광선 · 가림으로 남은 것)만 뺀다."""
        full = self.extra_cells(t, grid)
        ttl = self.p.obstacle_ttl_s
        if not any(t - seed[2] > ttl for seed in self._seeds.values()):
            return full
        rear = None
        if pose is not None:
            rear = (round(pose.x, 2), round(pose.y, 2), round(pose.yaw, 2))
        key = (self.version, grid, rear)
        if key != self._fresh_key:
            out = set()
            half = math.radians(self.p.memory_rear_deg)
            dropped = False
            for c, seed in self._seeds.items():
                keep = t - seed[2] <= ttl
                if not keep and pose is not None:
                    cx, cy = grid.center(*c)
                    behind = math.atan2(cy - pose.y, cx - pose.x) - pose.yaw - math.pi
                    keep = abs(math.atan2(math.sin(behind), math.cos(behind))) <= half
                if keep:
                    out.update(self._disc(grid, c, seed[0]))
                else:
                    dropped = True
            if self._vehicles_t is not None and t - self._vehicles_t <= self.p.vehicle_ttl_s:
                radius = self.p.vehicle_radius + grid.grow
                for x, y in self._vehicles:
                    out.update(grid.cells_in_disc(x, y, radius))
            out.update(self._ugv_plan_cells(grid))
            self._fresh_set, self._fresh_key = (frozenset(out) if dropped else full), key
        return self._fresh_set

    def tight_cells(self, t, grid):
        """좁은 틈 2단계에 쓸 임시 장애물 칸: 한자리에 tight_still_s 넘게 찍혀 온 점은 tight_radius 로, 나머지는 extra_cells 와 같다.
        끔(tight_radius <= 0)이거나 줄일 점이 없으면 None."""
        p = self.p
        if p.tight_radius <= 0.0:
            return None
        self.extra_cells(t, grid)   # 사라질 칸 정리
        small = p.tight_radius + grid.grow
        parked = self._parked_cells(t, grid)  # 드론으로 보아 서 있는 UGV 에 찍힌 칸은 라이다로 tight_still_s 를 기다리지 않는다
        old = [c for c, seed in self._seeds.items() if (seed[2] - seed[3] >= p.tight_still_s or c in parked) and seed[0] > small]
        if not old and not self._ugv_plan:   # 줄일 것이 없다 (오래 서 있는 드론 UGV 는 좁은 틈 길에서 작은 반지름으로 본다)
            return None
        key = (self.version, grid, len(old), len(parked))
        if key != self._tight_key:
            still = set(old)
            out = set()
            for c, seed in self._seeds.items():
                out.update(self._disc(grid, c, small if c in still else seed[0]))
            vehicles = self._vehicles_t is not None and t - self._vehicles_t <= p.vehicle_ttl_s
            if vehicles:
                radius = p.vehicle_radius + grid.grow
                for x, y in self._vehicles:
                    out.update(grid.cells_in_disc(x, y, radius))
            out.update(self._ugv_plan_cells(grid, tight=True))
            self._tight, self._tight_key = frozenset(out), key
        return self._tight

    def standing_cover(self, t, grid, cell):
        """칸 cell 을 막고 있는 임시 장애물이 서 있는 물체인가: 길 고르기에 넣은 오래 서 있는 드론 UGV (params.ugv_plan_parked_s), 또는
        한자리에 tight_still_s 넘게 찍혀 온 점 · 드론으로 보아 서 있는 UGV 에 찍힌 점 (좁은 틈 길 tight_cells 와 같은 기준).
        지나가는 물체가 잠깐 덮은 것은 False."""
        p = self.p
        self.extra_cells(t, grid)   # 사라질 칸 정리 · 드론 UGV 자리 맞추기
        if cell in self._ugv_plan_cells(grid):
            return True
        if not self._seeds:
            return False
        parked = self._parked_cells(t, grid)
        for c, seed in self._seeds.items():
            reach = int(seed[0] / grid.cell) + 1
            if abs(c[0] - cell[0]) > reach or abs(c[1] - cell[1]) > reach:
                continue
            if (seed[2] - seed[3] >= p.tight_still_s or c in parked) and cell in self._disc(grid, c, seed[0]):
                return True
        return False

    def blocker_static(self, t, grid, still_s, share=1.0):
        """지금 우리를 막고 있는 지도에 없는 점들이 still_s 넘게 그 자리에 있었나 (서 있는 물체 — 멈춘 다른 UGV · 지도에 빠진 상자).

        막고 있는 점 = 이번 스캔의 점 가운데 차체 앞쪽 띠(앞 1.3 m · 옆 0.30 m)에 든 것, 없으면 차체 가운데에서 0.5 m 안의 것.
        점마다 그 칸이 처음 찍힌 때부터 마지막으로 찍힌 때까지를 본다 (좁은 틈 길 tight_cells · 목표 옮기기 standing_cover · near_standing 과 같은 기준 —
        칸은 다시 찍히는 동안 처음 찍힌 때를 지닌다). 2026-11-20 까지는 「처음 찍힌 지 still_s」 였다: 한두 번 찍히고 기억으로만 남은 칸
        (지나간 UGV 가 남긴 칸 · 도는 동안의 헛칸)도 그 뒤 서 있는 물체가 됐다 — 실차 일곱 판 재생: 움직이는 UGV 가 남긴 칸의 43 %, 기억으로만 남은
        헛칸의 64 % (이제 5 % · 24 %). 지금 찍히고 있는 칸은 두 기준이 같다 (서 있는 UGV 를 알아보는 때는 그대로). 그 share 이상이
        오래됐으면 True (기본은 모두 — 하나라도 새 칸이면 좁은 틈 길이 아직 안 열린다). 다가오거나 지나가는 물체는 새 칸에 찍힌다. 스캔이 낡았거나 막는 점이 없으면 False."""
        if grid is None or not self._fresh(t) or not self.unexplained_body:
            return False
        p = self.p
        pairs = list(zip(self.unexplained_body, self.unexplained_world))
        block = [w for (bx, by), w in pairs if 0.0 < bx <= p.robot_half_length + 1.3 and abs(by) <= 0.30]
        if not block:
            block = [w for (bx, by), w in pairs if math.hypot(bx, by) <= 0.5]
        if not block:
            return False
        old = 0
        for x, y in block:
            seed = self._seeds.get(grid.cell_of(x, y))
            if seed is not None and seed[2] - seed[3] >= still_s:
                old += 1
        return old >= share * len(block)

    # ------------------------------------------------------------------ 드론이 본 UGV (drive/ugvs.py)
    def _parked_cells(self, t, grid):
        """임시 장애물 칸 가운데 드론으로 보아 서 있는 UGV(ugv_parked_s 넘게 한자리)에서 ugv_match_m 안의 것."""
        ugvs = self.ugvs
        if ugvs is None or not self._seeds:
            return ()
        parked = ugvs.parked(t)
        if not parked:
            return ()
        reach = self.p.ugv_match_m
        out = set()
        for c in self._seeds:
            x, y = grid.center(*c)
            if any(abs(u.x - x) <= reach and abs(u.y - y) <= reach and math.hypot(u.x - x, u.y - y) <= reach for u in parked):
                out.add(c)
        return out

    def _blockers(self):
        """지금 우리를 막고 있는 지도에 없는 점들 (경기장 좌표) — blocker_static 과 같은 고르기: 차체 앞쪽 띠, 없으면 가운데에서 0.5 m 안."""
        p = self.p
        pairs = list(zip(self.unexplained_body, self.unexplained_world))
        block = [w for (bx, by), w in pairs if 0.0 < bx <= p.robot_half_length + 1.3 and abs(by) <= 0.30]
        return block or [w for (bx, by), w in pairs if math.hypot(bx, by) <= 0.5]

    def blocker_ugv(self, t, share=0.6):
        """지금 우리를 막고 있는 것을 드론으로 보면: ("parked", 서 있던 s) · ("moving", 서 있던 s) · ("none", 0.0) · None.

        막는 점의 share 이상이 드론이 본 한 UGV 의 ugv_match_m 안이면 그 UGV 다 — ugv_parked_s 넘게 한자리면 parked, 아니면 moving.
        "none" = 드론 자리는 오고 있는데 그 점들 곁에 UGV 가 없다 (UGV 가 아니거나, 드론 화면 밖 · 가림 — 「없다」 는 뜻이 아니다).
        None = 드론 자리가 안 온다 · 스캔이 낡았다 · 막는 점이 없다.
        사진이 끊긴 뒤에는 붙들고 있는 UGV(끊길 때 서 있던 것)하고만 견준다 — 맞으면 parked, 안 맞으면 None (「곁에 UGV 없음」 이라 할 수 없다)."""
        ugvs = self.ugvs
        if ugvs is None or not self._fresh(t) or not self.unexplained_body:
            return None
        live = ugvs.alive(t)
        if not live and not ugvs.holding(t):
            return None
        block = self._blockers()
        if not block:
            return None
        reach = self.p.ugv_match_m
        best = None
        for u in ugvs.fresh(t):
            n = sum(1 for x, y in block if math.hypot(u.x - x, u.y - y) <= reach)
            if n and (best is None or n > best[0]):
                best = (n, u)
        if best is None or best[0] < share * len(block):
            return ("none", 0.0) if live else None
        still = best[1].still_s()
        return ("parked" if still >= self.p.ugv_parked_s else "moving", still)

    def near_standing(self, t, grid, reach, share=0.6):
        """차체 가운데에서 reach 안의 지도에 없는 점들이 서 있는 물체인가 — 서서 돌기를 막은 것을 볼 때 쓴다 (blocker_ugv · blocker_static 은 앞쪽 띠를 본다).
        ("parked", 서 있던 s) = 그 점의 share 이상이 드론으로 보아 서 있는 UGV 들 곁(ugv_match_m) — s 는 그 가운데 점이 가장 많이 찍힌 UGV 의 것,
        ("static", 0.0) = share 이상이 tight_still_s 넘게 그 칸에 찍혀 왔다, None = 아니다 · 모른다 (스캔이 낡았다 · 그 안에 점이 없다)."""
        if grid is None or not self._fresh(t) or not self.unexplained_body:
            return None
        p = self.p
        pts = [w for (bx, by), w in zip(self.unexplained_body, self.unexplained_world) if math.hypot(bx, by) <= reach]
        if not pts:
            return None
        ugvs = self.ugvs
        if ugvs is not None and (ugvs.alive(t) or ugvs.holding(t)):
            parked = [u for u in ugvs.fresh(t) if u.still_s() >= p.ugv_parked_s]
            hits = [sum(1 for x, y in pts if math.hypot(u.x - x, u.y - y) <= p.ugv_match_m) for u in parked]
            on_any = sum(1 for x, y in pts if any(math.hypot(u.x - x, u.y - y) <= p.ugv_match_m for u in parked))
            if parked and on_any >= share * len(pts):
                return ("parked", parked[hits.index(max(hits))].still_s())
        old = 0
        for x, y in pts:
            seed = self._seeds.get(grid.cell_of(x, y))
            if seed is not None and seed[2] - seed[3] >= p.tight_still_s:
                old += 1
        return ("static", 0.0) if old >= share * len(pts) else None

    def ugv_centres(self, t):
        """드론이 본, 지금 믿는 다른 UGV 들의 가운데 [(x, y)] (없으면 빈 목록)."""
        return [(u.x, u.y) for u in self.ugvs.fresh(t)] if self.ugvs is not None else []

    def sync_ugvs(self, t, grid):
        """UGV 때문에 생긴 임시 장애물 칸을 드론 자리에 맞춘다 (스캔 · 드론 사진이 새로 왔을 때만 일한다).

        - 지금 찍히고 있는 칸 곁(ugv_match_m)에 드론이 본 UGV 가 있으면 그 칸을 「UGV 칸」 으로 적어 둔다
        - UGV 칸이 기억으로만 남았는데(마지막으로 찍힌 지 obstacle_ttl_s 넘음) 그 곁(ugv_match_m + ugv_still_m)에 이제 UGV 가 없으면 지운다 — 떠났다.
          라이다가 못 보는 쪽(뒤 90° · 가림)으로 간 UGV 의 자리를 obstacle_memory_s(20 s)까지 붙들던 것을 푼다.
          다만 그 곁에 **서 있다가 드론에서 사라진** UGV 의 자리(UgvTracks.lost_parked)가 있으면 지우지 않는다 — 떠나는 것을 본 적이 없다
          (2026-11-20 00:00 실차: 드론이 50 s 놓친 서 있는 UGV 의 칸 40개를, 우리가 등을 돌린 순간 지웠다). 그런 칸은 전처럼 라이다 규칙으로만 사라진다
        - 그 곁(ugv_match_m)에 UGV 가 그대로 서 있으면(tight_still_s 넘게) obstacle_memory_s 가 지나도 붙든다 (obstacle_ttl_s 씩 늘린다).
          다만 그 UGV 가 이미 길 고르기에 들어 있으면(ugv_plan_parked_s 넘게 서 있다 — 드론 자리의 원이 막는다) 붙들지 않는다 (2026-11-20):
          그 곁 0.35 m 에 한 번이라도 찍힌 칸(지나가던 것 · 방향이 틀린 스캔의 헛점 · 자세 오차로 옮겨 찍힌 그 UGV)이 UGV 가 서 있는 내내 남아
          칸마다 0.35 m 씩 막았다 — 03:28 실차: 거점 안 (0.27, 2.39) 에 선 UGV 곁에 169 s 묵은 칸 58개, 그 앞 통로(실제 0.75 m)가 0.3 m 로 좁아져
          상자 면이 조금만 어긋나 찍혀도 닫혔고 0.4 m 앞 목표를 4.2 m 돌아갔다. 그런 칸은 전처럼 라이다 규칙(obstacle_ttl_s · obstacle_memory_s)으로 사라진다
        「드론에 안 보인다」 만으로는 지우지 않는다: 적어 둔 칸(= 드론이 그 자리에서 UGV 를 본 적이 있다)만, 드론 자리가 오고 있을 때만."""
        ugvs = self.ugvs
        if ugvs is None:
            return
        key = (ugvs.version, self.scan_t)
        if key == self._ugv_sync:
            return
        self._ugv_sync = key
        self._ugv_cells &= self._seeds.keys()
        p = self.p
        tracks = ugvs.fresh(t)   # 드론 자리가 안 오면 빈 목록
        # 오래 서 있는 UGV → 길 고르기에 넣을 자리: 그 자리에서 본 사진들의 평균 (Track.spot — 평균이 3 cm 넘게 옮겨 갈 때만 따라간다).
        # 전에는 서 있기 시작한 첫 장의 값이었다 (2026-11-20)
        plan = tuple(sorted(u.spot for u in tracks if 0.0 < p.ugv_plan_parked_s <= u.still_s()))
        if plan != self._ugv_plan:
            self._ugv_plan = plan
            self.version += 1
        if not ugvs.alive(t):
            return
        tag, gone = p.ugv_match_m, p.ugv_match_m + p.ugv_still_m
        maybe = ugvs.lost_parked(t)   # 서 있다가 드론에서 사라진 UGV 들의 자리 — 아직 거기 있을 수 있다
        drop = []
        for c, seed in self._seeds.items():
            x, y = grid.center(*c)
            d, u = min(((math.hypot(u.x - x, u.y - y), u) for u in tracks), key=lambda v: v[0], default=(math.inf, None))
            if t - seed[2] <= p.obstacle_ttl_s:
                if d <= tag:
                    self._ugv_cells.add(c)
            elif c in self._ugv_cells:
                if d > gone:
                    if not any(math.hypot(gx - x, gy - y) <= gone for gx, gy in maybe):
                        drop.append(c)
                elif d <= tag and u.still_s() >= p.tight_still_s and u.spot not in plan:
                    seed[1] = max(seed[1], t + p.obstacle_ttl_s)  # 그 UGV 가 그 자리에 그대로 서 있다
        for c in drop:
            del self._seeds[c]
            self._ugv_cells.discard(c)
        if drop:
            self.dropped_ugv_cells += len(drop)
            self.version += 1
        if self._seeds:
            self._next_expiry = min(seed[1] for seed in self._seeds.values())

    def _ugv_plan_cells(self, grid, tight=False):
        """오래 서 있는 UGV 들이 막는 칸 (길 고르기용 — params.ugv_plan_parked_s). tight = 좁은 틈 길: 작은 반지름으로."""
        if not self._ugv_plan:
            return ()
        radius = (self.p.ugv_plan_tight_radius if tight else self.p.vehicle_radius) + grid.grow
        out = set()
        for x, y in self._ugv_plan:
            out.update(grid.cells_in_disc(x, y, radius))
        return out

    def ugv_body_points(self, t, pose):
        """우리 차 곁(ugv_contact_m)에 서 있는 UGV 를 차체 좌표의 점 몇 개로 — 비키기 계획이 라이다가 못 보는 뒤쪽 · 몸 바로 곁의 UGV 를 피하게.
        움직이는 UGV 는 넣지 않는다 (드론 자리가 1 s 쯤 늦다 — 그것은 라이다가 본다)."""
        ugvs = self.ugvs
        if ugvs is None or pose is None:
            return []
        p = self.p
        out = []
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        for u in ugvs.fresh(t):
            dx, dy = u.x - pose.x, u.y - pose.y
            if u.still_s() < p.tight_still_s or math.hypot(dx, dy) > p.ugv_contact_m:
                continue
            bx, by = c * dx + s * dy, -s * dx + c * dy
            r = p.ugv_contact_radius
            out.append((bx, by))
            out.extend((bx + r * math.cos(a), by + r * math.sin(a)) for a in (k * math.pi / 4.0 for k in range(8)))
        return out

    def memory_summary(self, t):
        """기록용: 지금 막는 점 칸 수 · 그중 기억으로만 남은 칸(마지막으로 찍힌 지 obstacle_ttl_s 넘음) 수 · 그 가장 오래된 나이 (s)."""
        ages = [t - seed[2] for seed in self._seeds.values() if seed[1] > t]
        old = [a for a in ages if a > self.p.obstacle_ttl_s]
        return {'seeds': len(ages), 'remembered': len(old), 'oldest_s': round(max(old), 1) if old else 0.0,
                'skipped_turning': self.skipped_turning}

    def snapshot(self, t, grid):
        """지금 막고 있는 임시 장애물 칸 (시험 화면용). extra_cells 와 달리 아무것도 바꾸지 않는다."""
        out = set()
        for c, seed in self._seeds.items():
            if seed[1] > t:
                out.update(self._disc(grid, c, seed[0]))
        if self._vehicles_t is not None and t - self._vehicles_t <= self.p.vehicle_ttl_s:
            radius = self.p.vehicle_radius + grid.grow
            for x, y in self._vehicles:
                out.update(grid.cells_in_disc(x, y, radius))
        out.update(self._ugv_plan_cells(grid))
        return sorted(out)

    def _fresh(self, t):
        return self.scan_t is not None and t - self.scan_t <= 0.5

    def gap_ahead(self, t, direction, curvature=0.0, reach=None):
        """진행 방향(direction: +1 앞, -1 뒤)으로 차체 앞면에서 지도에 없는 가장 가까운 점까지 (가야 할 거리, 없으면 inf).

        띠의 폭은 차체 반폭 + stop_side_margin. curvature(= w / v, 1/m, 앞으로 갈 때만)가 있으면 곧은 띠 대신 그 호를 따라 굽은 띠를 본다
        (정확한 원으로 계산) — 돌면서 옆으로 쓸고 가는 자리에 있는 것을 잡고, 비켜 도는 쪽의 앞에 있던 것에는 서지 않는다.
        부르는 쪽은 실제로 낼 (v, w) 가 그 호를 그리게 한다 (감속하면 w 도 같은 비율로).
        """
        if not self._fresh(t):
            return math.inf
        p = self.p
        half_w = p.robot_half_width + (p.stop_side_margin if self.side_margin is None else self.side_margin)
        reach = p.slow_dist if reach is None else reach
        bend = curvature if direction > 0 and abs(curvature) > 1e-3 else 0.0
        best = math.inf
        for x, y in self.unexplained_body:
            forward = x * direction
            if not bend:
                if abs(y) <= half_w and 0.0 <= forward - p.robot_half_length <= reach:  # 곧은 띠
                    best = min(best, forward - p.robot_half_length)
            else:
                radius = 1.0 / bend  # 왼쪽으로 돌면 양수: 원의 가운데 (0, radius)
                if abs(math.hypot(x, y - radius) - abs(radius)) > half_w:
                    continue
                # 차체가 지금 있는 원 위의 각에서 그 점까지 진행 방향으로 돈 각 → 호의 길이
                swept = math.atan2(x, -(y - radius)) if bend > 0 else math.atan2(x, y - radius)
                if swept < 0.0:
                    continue  # 뒤쪽 반원
                ahead = swept * abs(radius) - p.robot_half_length
                if 0.0 <= ahead <= reach:
                    best = min(best, ahead)
        return best

    def pass_gap(self, t, curvature=0.0, reach=0.4, width=0.15):
        """앞으로 가며 **곁을 스쳐 지날** 지도에 없는 점까지의 옆 틈 (m) — 차체 옆면에서. 진행 띠(gap_ahead)는 차체 반폭 + stop_side_margin 안만 보아,
        그 바로 밖에 선 것 곁은 제 속도로 지나간다 (2026-11-20 00:00 실차: 「그 물체를 지나는 길」 에서 가운데선 0.16 ~ 0.25 m 옆의 UGV 를 0.45 m/s 로 스쳤다).
        차체 뒷면부터 앞면 앞 reach 까지, 옆으로 차체 옆면에서 width 안에 든 점 가운데 가장 가까운 것의 틈. 없으면 inf. curvature 는 gap_ahead 와 같다."""
        if not self._fresh(t):
            return math.inf
        p = self.p
        hl, hw = p.robot_half_length, p.robot_half_width
        bend = curvature if abs(curvature) > 1e-3 else 0.0
        best = math.inf
        for x, y in self.unexplained_body:
            if not bend:
                ahead, side = x, abs(y)
            else:
                radius = 1.0 / bend
                side = abs(math.hypot(x, y - radius) - abs(radius))
                ahead = (math.atan2(x, -(y - radius)) if bend > 0 else math.atan2(x, y - radius)) * abs(radius)
            if -hl <= ahead <= hl + reach and side - hw <= width:
                best = min(best, side - hw)
        return best

    def stop_reach(self, speed=0.0, extra=0.0):
        """이 안에 지도에 없는 점이 있으면 선다: stop_dist + |speed| × stop_reaction_s + extra."""
        return self.p.stop_dist + abs(speed) * self.p.stop_reaction_s + extra

    def blocking_ahead(self, t, direction, speed=0.0, curvature=0.0, extra=0.0):
        """진행 방향으로 차체 앞 stop_reach(speed, extra) 안에 지도에 없는 점이 있는가 (slow_dist 와 상관없이 그 거리까지 본다)."""
        reach = self.stop_reach(speed, extra)
        return self.gap_ahead(t, direction, curvature, reach=reach) <= reach

    def turn_hits(self, t, angle, margin, step=math.radians(5.0), contact=0.01, approach=0.01):
        """제자리에서 angle(rad, 왼쪽 +) 만큼 돌면 차체 사각형이 지도에 없는 점에 닿는가 — 돌아가는 쪽만 본다.
        contact(라이다 잡음 정도)보다 가까워지거나, min(margin, 지금 거리)보다 approach 넘게 더 다가오면 친다.
        반대쪽으로 비켜 가는 점, 붙어 있던 물체 곁을 몇 mm 다가갔다 멀어지는 점은 막지 않는다 (2026-11-19 검토)."""
        if not self._fresh(t) or not self.unexplained_body or abs(angle) < 1e-6:
            return False
        p = self.p
        hl, hw = p.robot_half_length, p.robot_half_width
        near = [(x, y) for x, y in self.unexplained_body if math.hypot(x, y) <= math.hypot(hl, hw) + margin]
        if not near:
            return False

        def gap(x, y, a):  # 차체를 a 만큼 돌린 것 = 점을 −a 만큼 돌려 원래 사각형과 견준다
            c, s = math.cos(a), math.sin(a)
            return math.hypot(max(abs(c * x + s * y) - hl, 0.0), max(abs(-s * x + c * y) - hw, 0.0))

        now = [gap(x, y, 0.0) for x, y in near]
        n = max(1, int(math.ceil(abs(angle) / step)))
        for k in range(1, n + 1):
            a = angle * k / n
            for (x, y), g0 in zip(near, now):
                g = gap(x, y, a)
                if g < max(contact, min(margin, g0) - approach):  # contact 안으로는 안 되고, margin(또는 지금 거리)에서 approach 넘게 다가가도 안 된다
                    return True
        return False
