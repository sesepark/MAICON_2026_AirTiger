"""드론 지도 상자를 라이다로 고친다 (2026-11-18). ROS 에도 slam_node 에도 묶이지 않은 계산 모듈 — numpy 만 쓴다.

왜: 드론 사진의 상자 찾기(perception)는 자주 틀린다. 없는 상자를 넣거나(선 · 그림자 · 사람) 있는 상자를 빠뜨린다.
지금까지 SLAM 은 그 지도를 그대로 믿고 정합 · 주행 지도로 썼다.

무엇을 하나 (정합이 성공한 스캔으로만)
  1. 없는 상자 빼기   상자마다 점수(log-odds)를 둔다. 라이다 쪽 면에 점이 맞으면 올리고(l_hit), 광선이 상자 속을 지나가면 깎는다(l_miss).
                     reject 아래로 내려가면 지도 · 정합에서 뺀다. 다시 맞으면(revive) 되살린다.
                     가려져 안 보이는 상자는 증거가 없으니 그대로 둔다.
                     - 「맞은 점」 은 그 점까지의 광선이 이 상자 속을 지나지 않은 것만 센다. 상자 속을 지나 뒤쪽 면 자리에 찍힌 점은
                       이 상자가 아니라 그 뒤의 것(붙어 있는 진짜 상자 등)이다.
                     - 맞은 점이 있으면 지나가는 광선이 있어도 빼지 않는다 (조금 어긋난 진짜 상자 — 한쪽은 맞고 한쪽은 비어 있다).
                       2026-11-19 실차 기록: 면에 10점씩 맞던 상자가 15 cm 어긋나 귀퉁이로 광선이 지나갔다는 이유로 빠졌다.
                     - 귀퉁이를 스치는 광선(속을 지난 길이 through_chord_m 미만)은 증거로 쓰지 않는다.
  2. 빠진 상자 더하기  지도(벽 · 상자)로 설명되지 않는 점들에 맞는 40 cm 정사각형을 후보로 둔다.
                     같은 자리에서 promote_seen 번 넘게, promote_span_s 넘게 보이고 한 번이라도 min_span_m 넘는 폭으로 보이면
                     상자로 더한다 (움직이는 UGV · 25 cm 차체는 안 된다). 더한 상자도 1 과 같은 점수로 다시 뺀다.
                     **더한 상자는 정합에만 쓴다 (structure()). 내보내는 지도(output())에는 넣지 않는다** — 주행 노드는 지도에 있는 것을
                     「안 움직이는 벽」 으로 보고 지도에 없는 물체 대응(정지 · 감속 · 물러나기)을 끈다. 서 있던 다른 UGV 가 상자로
                     들어가면 그 대응이 꺼진다 (2026-11-19 test_team_test_replay). 센서로 본 것은 고정 지도에 섞지 않는다.
  3. 안전장치        위치가 틀렸으면 지도를 망친다. 그래서 (a) 믿는 구조(외벽 + 라이다로 확인한 상자)에 맞은 점이 충분하고
                     (b) 지도로 설명되는 점의 비율이 높고 (c) 빠르게 돌고 있지 않을 때만 고친다.
                     정합에는 확인한 상자를 1, 확인 못 한 드론 상자를 unconfirmed_weight 로 센다 (structure()) —
                     이 무게는 후보 자세의 순위에만 쓰고, 정합 통과 기준(맞은 점 수)은 무게 없이 센다 (slam_node.scan_match).
                     **경기장 밖으로 나가는 상자는 더하지 않는다** (wall_overlap_m): 2026-11-19 18:06 본선 경기장 기록에서 위치가 0.12 m
                     틀린 동안 아래 벽이 y = 0.08 에 보였고, 그 면에 맞춘 상자 4개가 벽 밖(y = −0.06 ~ −0.12)에 더해져 정합을 그 틀린 자리에 묶었다.
                     x = 6.24 (벽은 6.12) 의 상자도 있었다. 상자는 바닥에 놓이므로 벽을 넘어갈 수 없다.
                     위치가 틀렸던 것이 드러나면(slam_node 의 「벽으로 다시 보기」) 그 사이에 더하고 · 빼고 · 옮긴 것을 되돌린다 (distrust).
  4. 고정 기둥        set_map(fixed=…) 으로 받은 기둥(벽에 붙은 블록 10개, drive/fieldmap.fixed_pillars)은 외벽과 같은 「아는 구조」 다:
                     그 면의 점은 설명되는 점이고, 드론 상자의 증거나 새 상자 후보로 쓰지 않는다. 다만 믿는 점으로는 세지 않는다 (가정한 자리라서).
                     기둥 자체는 빼지도 더하지도 않는다
                     (정합에는 slam_node.structure() 가 따로 넣는다).

계산량: 정합 observe_every 번에 한 번, 점 약 250 개 × 상자 수 의 numpy 배열 몇 개. 이 PC 에서 1 ~ 3 ms (test_map_fusion).

빼려면: slam_node 의 map_fusion 파라미터를 false 로 (그대로 넘기는 PassThroughMap 이 들어간다).
지우려면: 이 파일, slam_node 의 import 한 줄과 SlamNode.__init__ 의 MapFusion 한 줄, test/test_map_fusion.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np

from navigation.drive.fieldmap import pillar_side_faces


@dataclass
class FusionParams:
    observe_every: int = 2         # 정합 성공 이만큼에 한 번 지도를 고친다 (10 Hz 스캔 → 5 Hz)
    scan_step: int = 2             # 스캔 점을 이만큼에 하나 (505 점 → 약 250). 정합(5 개에 하나)보다 촘촘해야 먼 상자 속 광선이 잡힌다
    max_range_m: float = 3.5       # 라이다 원점에서 상자 가운데까지 이보다 먼 상자에는 증거를 주지 않는다
    # 한 번의 관측에서 상자 하나에 주는 증거
    hit_m: float = 0.04            # 상자 면에서 이 안의 점 = 맞은 점 (외벽 위의 점은 빼고)
    hit_min: int = 3
    through_shrink_m: float = 0.06  # 상자를 이만큼 줄인 속을 광선이 지나가야 「빈 바닥」 (자세 · 거리 오차 몇 cm 를 견딘다)
    through_end_m: float = 0.10    # 광선 끝 이만큼은 보지 않는다 (끝점이 면 바로 뒤에 찍힌 경우)
    through_min: int = 2
    through_chord_m: float = 0.10  # 줄인 상자 속을 이만큼 넘게 지나간 광선만 센다 (귀퉁이를 스친 광선 · 비스듬한 끝점은 믿지 않는다)
    l_hit: float = 1.0
    l_miss: float = 0.5            # 드론 상자에 주는 「없다」 증거는 「있다」 의 절반 (OctoMap 의 hit 0.85 / miss 0.4 와 같은 비율). 자세가 잠깐 틀려도 진짜 상자를 바로 빼지 않는다. 라이다로 더한 상자는 l_hit 만큼 깎는다
    l_max: float = 4.0             # 점수는 ±l_max 안 (오래 쌓인 점수가 고쳐지지 않는 일이 없게)
    confirm: float = 2.0           # 이 위 = 라이다로 확인한 상자
    reject: float = -3.0           # 이 아래 = 없는 상자 (지나가는 광선만 있는 관측 6 번 = 5 Hz 에서 1.2 s)
    revive: float = 3.0            # 뺀 상자가 이 위로 다시 올라오면 되살린다. 2026-11-19 1.0 → 3.0: 본선 경기장 16:55 기록(세로 4.8 로 다시 맞춤)에서
                                   # 없는 상자 (5.08, 3.74) 가 87 s 에 빠지고 214 s 에 되살아나고 293 s 에 또 빠졌다 (그때마다 주행 격자 · 경로가 바뀐다)
    revive_span_m: float = 0.25    # 뺀 상자를 되살리는 「맞은 점」 은 한 면에서 이 폭 넘게 펼쳐져야 센다 — 그 자리를 지나거나 선 UGV(25 × 23 cm) ·
                                   # 곁 상자의 모서리로는 되살리지 않는다. 0 이면 폭을 보지 않는다
    unconfirmed_weight: float = 0.7  # 정합에서 확인 못 한 드론 상자에 맞은 점의 무게 (벽 · 확인한 상자 = 1)
    # 안전장치 (이 관측으로 지도를 고쳐도 되나)
    explain_m: float = 0.07        # 벽 · 상자 면에서 이 안 = 지도로 설명되는 점
    min_explained: float = 0.4
    min_trusted_points: int = 10   # 믿는 구조(외벽 + 확인한 상자)에 맞은 점이 이만큼은 있어야 고친다
    trust_confirmed_boxes: bool = True  # 라이다로 확인한 상자에 맞은 점도 믿는 점으로 센다. False 면 외벽 점만 (2026-11-18 처음 만든 대로).
                                        # 외벽만 세면 상자에 가려 벽이 안 보이는 자리에서는 못 고친다 (2026-11-19 실차 기록 1118_202820: 관측의 74 % 를 건너뜀)
    trust_pillar_faces: bool = True  # 고정 기둥의 옆면(붙은 벽에 수직인 면, drive/fieldmap.pillar_side_faces)에 맞은 점도 믿는 점으로 센다 (2026-11-19 밤).
                                     # 본선 경기장 9판 스캔: 옆면은 가정한 자리에서 2.3 cm 안. 기둥 전체(가정한 자리)를 믿는 것은 아니다 — 아래 observe 의 설명.
                                     # 기둥이 없는 경기장이면 끈다 (slam_node 파라미터 trust_pillars)
    max_yaw_rate: float = 0.6      # rad/s. 빠르게 도는 동안의 스캔은 번진다
    # 빠진 상자
    face_m: float = 0.035          # 후보 상자 면에서 이 안의 점이 그 면을 받친다
    min_support: int = 5
    coarse_m: float = 0.04         # 후보 가운데를 찾는 격자 간격
    max_unexplained: int = 80      # 설명 안 되는 점이 이보다 많으면 고르게 줄여서 쓴다 (계산량 상한)
    match_m: float = 0.12          # 이 안의 새 후보는 같은 것
    conflict_m: float = 0.15       # 후보 가운데에서 half + 이 안에 후보 면으로 설명 안 되는 점이 있으면 = 물체가 옮겨 갔다 → 후보 버림
    promote_seen: int = 3
    promote_span_s: float = 1.0
    # 폭은 **한 면에서** 잰다 (2026-11-19 — 전에는 두 면에 걸친 점을 합쳐 재서, 모서리에서 본 25 x 23 cm UGV 도 30 cm 를 넘을 수 있었다)
    min_span_m: float = 0.30       # 맞은 점들이 한 번이라도 이 폭 넘게 펼쳐져 보여야 상자로 더한다 (상자 40 cm, 다른 UGV 차체 25 × 23 cm)
    candidate_ttl_s: float = 3.0
    max_added: int = 20
    wall_overlap_m: float = 0.08   # 더할 상자가 외벽 밖으로 이만큼 넘게 나가면 상자가 아니다 (가운데가 벽에서 half − 이 값 안쪽이어야 한다).
                                   # 0.08 = 위치 · 벽 치수 오차 몇 cm (본선 경기장 가로는 지도보다 약 6 cm 좁게 보였다)
    carry_m: float = 0.15          # 새 드론 지도의 상자가 이 안이면 앞 지도 상자의 점수를 이어 받는다
    # 옮겨진 상자: 더한 상자(라이다로 맞춘 자리)가 드론 상자 바로 곁에 있으면 둘은 같은 상자다 — 드론 지도의 자리가 틀린 것이다.
    # 그 드론 상자를 라이다로 맞춘 자리로 옮겨 내보낸다 (output 에 그 자리로 나간다 → 주행 지도가 고쳐진다).
    # 2026-11-19 17:26 모의 경기 3차(본선 경기장): 화면 끝에서 잘린 상자가 드론 지도에 (4.18, 3.72) 로 그려졌고 진짜는 (4.26, 3.99) 였다.
    # 65초에 그 상자를 뺐고 67초에 진짜 자리의 상자를 더했는데, 더한 상자는 내보내지 않으므로 주행 지도에는 진짜 상자가 없었고,
    # 175초에는 뺀 상자가 되살아났다. 로봇은 지도상의 그 상자 곁(0.12 ~ 0.22 m)이라 32초부터 105초를 그 자리에 서 있었다.
    #   move_m     뺀 드론 상자에서 이 안에 더한 상자가 있으면 옮긴다 (상자 한 변 0.4 m 보다 조금 넓게)
    #   overlap_m  아직 빼지 않았고 확인도 못 한 드론 상자에서 이 안에 더한 상자가 있으면 옮긴다 (두 상자가 이만큼 가까이 있을 수는 없다)
    # 라이다로 확인한 드론 상자는 옮기지 않는다 (그 곁에 생긴 것은 붙어 선 다른 UGV 일 수 있다)
    move_m: float = 0.45
    overlap_m: float = 0.30


class _Track:
    """상자 하나. source = 드론 지도 번호 (더한 상자는 None)."""
    __slots__ = ('x', 'y', 'score', 'rejected', 'source', 'promoted', 'seen', 'first_ns', 'last_ns', 'span', 'changed_ns')

    def __init__(self, x, y, source=None, score=0.0, rejected=False, promoted=False, seen=0, first_ns=0, last_ns=0, span=0.0):
        self.x, self.y, self.source = float(x), float(y), source
        self.score, self.rejected, self.promoted = score, rejected, promoted
        self.seen, self.first_ns, self.last_ns = seen, first_ns, last_ns
        self.span = span          # 맞은 점들이 펼쳐진 가장 큰 폭 (m) — 더한 상자 후보만 쓴다
        self.changed_ns = None    # 드론 상자: 마지막으로 빼거나 되살리거나 옮긴 시각 (distrust 가 되돌릴 때 본다)


def _to_world(points, pose):
    x, y, yaw = pose
    c, s = cos(yaw), sin(yaw)
    return np.stack((x + c * points[:, 0] - s * points[:, 1], y + s * points[:, 0] + c * points[:, 1]), axis=1)


def _surface_distance(world, centres, half):
    """점 (P, 2) 에서 상자들 (K, 2) 의 면까지 거리 (P, K). 상자 안이면 가까운 면까지."""
    ax = np.abs(world[:, None, 0] - centres[None, :, 0]) - half
    ay = np.abs(world[:, None, 1] - centres[None, :, 1]) - half
    outside = np.hypot(np.maximum(ax, 0.0), np.maximum(ay, 0.0))
    return np.where((ax <= 0) & (ay <= 0), np.minimum(-ax, -ay), outside)


def _wall_distance(world, width, height):
    return np.minimum(np.minimum(np.abs(world[:, 0]), np.abs(width - world[:, 0])),
                      np.minimum(np.abs(world[:, 1]), np.abs(height - world[:, 1])))


def _face_distance(world, faces):
    """점 (P, 2) 에서 기둥 옆면 선분들(drive/fieldmap.pillar_side_faces)까지 거리 (P,). 선분 옆(범위 밖)이면 inf, 면이 없으면 모두 inf."""
    out = np.full(len(world), np.inf)
    for axis, at, lo, hi in faces:
        along, across = (world[:, 0], world[:, 1]) if axis == 'y' else (world[:, 1], world[:, 0])
        out = np.where((along > lo) & (along < hi), np.minimum(out, np.abs(across - at)), out)
    return out


def _ray_chords(origin, world, centres, shrunk, end_m):
    """origin → 각 점 광선이 (끝 end_m 은 빼고) 줄인 상자 속을 지나간 길이 (m), 표 (P, K). 안 지나면 0."""
    direction = world - origin[None, :]
    length = np.hypot(direction[:, 0], direction[:, 1])
    far = np.broadcast_to(np.maximum(0.0, 1.0 - end_m / np.maximum(length, 1e-6))[:, None], (len(world), len(centres))).copy()
    near = np.zeros_like(far)
    with np.errstate(divide='ignore', invalid='ignore'):
        for axis in (0, 1):
            d = direction[:, axis][:, None]
            low_edge = centres[None, :, axis] - shrunk - origin[axis]
            high_edge = centres[None, :, axis] + shrunk - origin[axis]
            a, b = low_edge / d, high_edge / d
            parallel = np.abs(d) < 1e-9
            inside = (low_edge < 0) & (high_edge > 0)
            low = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(a, b))
            high = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(a, b))
            near, far = np.maximum(near, low), np.minimum(far, high)
        chord = (far - near) * length[:, None]
        return np.where(chord > 0, chord, 0.0)       # nan(점이 원점) · -inf(비켜 감) 도 0


def _spread(points):
    """점들이 펼쳐진 폭: x · y 범위 가운데 큰 쪽 (m)."""
    return float(np.ptp(points, axis=0).max()) if len(points) else 0.0


def _face_span(points, cx, cy, half):
    """상자 (cx, cy) 의 면에 붙은 점들이 **한 면에서** 펼쳐진 가장 큰 폭 (m). 두 면에 걸친 점(모서리를 본 ㄱ 자)을 합쳐 재지 않는다."""
    if not len(points):
        return 0.0
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    on_x = np.abs(np.abs(dx) - half) < np.abs(np.abs(dy) - half)     # x 면(세로 면)에 더 가까운 점
    best = 0.0
    for mask, along, across in ((on_x, dy, dx), (~on_x, dx, dy)):
        for side in (across > 0, across <= 0):
            chosen = along[mask & side]
            if len(chosen) >= 2:
                best = max(best, float(np.ptp(chosen)))
    return best


def _face_support(points, centres, origin, half, face_m):
    """후보 가운데들 (C, 2) 마다 라이다 쪽을 향한 면을 받치는 점 (C, P) — 그리고 그 점이 x 면 · y 면 어느 쪽인지."""
    dx = points[None, :, 0] - centres[:, None, 0]
    dy = points[None, :, 1] - centres[:, None, 1]
    to_x = origin[0] - centres[:, 0]
    to_y = origin[1] - centres[:, 1]
    on_x = (np.abs(np.abs(dx) - half) < face_m) & (np.abs(dy) < half + face_m) \
        & (np.sign(dx) == np.sign(to_x)[:, None]) & (np.abs(to_x) > half)[:, None]
    on_y = (np.abs(np.abs(dy) - half) < face_m) & (np.abs(dx) < half + face_m) \
        & (np.sign(dy) == np.sign(to_y)[:, None]) & (np.abs(to_y) > half)[:, None]
    return on_x, on_y


def _fit_centre(points, on_x, on_y, guess, half):
    """면을 받치는 점으로 가운데를 다시 잰다. 보이는 면의 축은 면 위치로, 안 보이는 축은 점들이 펼쳐진 범위 안에서 guess 에 가깝게."""
    gx, gy = guess
    out = []
    for axis, own, cross, g in ((0, on_x, on_y, gx), (1, on_y, on_x, gy)):
        if own.sum() >= 2:
            values = points[own, axis]
            side = np.sign(values.mean() - g) or 1.0
            out.append(float(values.mean() - side * half))
        elif cross.sum() >= 2:
            values = points[cross, axis]
            low, high = values.max() - half, values.min() + half
            out.append(float(min(max(g, low), high)) if low <= high else float((low + high) / 2))
        else:
            out.append(float(g))
    return out[0], out[1]


class MapFusion:
    """드론 지도 상자 + 라이다 관측 → 고친 상자 목록. slam_node 가 부르는 순서:

    set_map(드론 상자, …)  → 지도가 올 때마다
    structure()            → 정합에 쓸 상자 · 무게 (빼지 않은 드론 상자 + 더한 상자)
    wants_scan() / observe(차체 좌표 점, 정합한 자세, 시각, …)  → 정합이 성공한 스캔마다
    changed() / output()   → 내보낼 지도 (빼지 않은 드론 상자만 — 더한 상자는 내보내지 않는다)
    """

    def __init__(self, params=None):
        self.params = params or FusionParams()
        self._calls = 0
        self.reset()

    # ------------------------------------------------------------------ 상태
    def reset(self):
        """다음 경기: 모두 잊는다."""
        self._drone = []          # 드론 지도 상자 _Track (지도 순서)
        self._added = []          # 더한 상자 · 후보 _Track
        self._half = 0.0
        self._size = 0.0
        self._width = self._height = 0.0
        self._fixed = np.zeros((0, 2))   # 고정 기둥 가운데 (상자와 같은 크기의 정사각형) — 외벽처럼 아는 구조
        self._faces = ()          # 믿는 기둥 옆면 (trust_pillar_faces)
        self.version = 0          # 관측으로 내보낼 상자(output)가 바뀐 횟수
        self.revision = 0         # 관측으로 정합에 쓰는 상자(structure)가 바뀐 횟수 (더한 상자 포함)
        self._serial = 0          # 지도를 받은 횟수
        self._cache = None
        self._moved = {}          # 옮긴 드론 상자: 드론 지도 번호 → (드론 지도의 x, y) — 지금 자리는 그 _Track 의 x, y
        self.last = {}            # 마지막 관측 결과 (기록 · 시험 화면용)

    def set_map(self, boxes, box_size, width, height, fixed=()):
        """새 드론 지도. 같은 자리(carry_m 안)의 상자는 앞에서 쌓은 점수를 이어 받는다.

        fixed = 고정 기둥 가운데 [(x, y)] (없으면 빈 것). 드론 상자가 이미 그 자리에 있으면 그 기둥은 뺀다 (slam_node.structure 와 같은 규칙).
        """
        p = self.params
        previous = [t for t in self._drone] + [t for t in self._added if t.promoted]
        was_moved = [(origin, t) for t in self._drone for origin in [self._moved.get(t.source)] if origin is not None]
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 2)
        drone, moved = [], {}
        for index, (x, y) in enumerate(boxes):
            track = _Track(x, y, source=index)
            # 같은 드론 지도가 다시 왔다: 라이다로 옮겨 둔 상자는 옮긴 자리를 지킨다 (드론의 자리로 되돌리지 않는다)
            kept = next((t for origin, t in was_moved if np.hypot(origin[0] - x, origin[1] - y) < p.carry_m), None)
            if kept is not None:
                track.x, track.y, track.score, track.rejected = kept.x, kept.y, kept.score, kept.rejected
                moved[index] = (float(x), float(y))
            elif previous:
                distance, nearest = min((np.hypot(t.x - x, t.y - y), i) for i, t in enumerate(previous))
                if distance < p.carry_m:
                    old = previous[nearest]
                    track.score, track.rejected = old.score, old.rejected
            drone.append(track)
        self._drone, self._moved = drone, moved
        size = float(box_size or 0.0)
        # 새 지도에 들어온 자리의 더한 상자 · 후보는 지운다 (같은 상자가 두 번 나오지 않게)
        self._added = [t for t in self._added
                       if not any(abs(t.x - d.x) < size - 0.05 and abs(t.y - d.y) < size - 0.05 for d in drone)]
        self._size, self._half = size, size / 2
        self._width, self._height = float(width or 0.0), float(height or 0.0)
        # 기둥 옆면은 드론 상자와 상관없이 늘 있다 (드론이 기둥을 상자로 그렸어도 그 자리의 기둥은 진짜다)
        self._faces = pillar_side_faces(self._width, self._height, fixed) if p.trust_pillar_faces and len(fixed) else ()
        fixed = [(float(x), float(y)) for x, y in fixed
                 if not any(abs(x - d.x) < size / 2 and abs(y - d.y) < size / 2 for d in drone)]
        self._fixed = np.array(fixed, dtype=float).reshape(-1, 2)
        # 기둥 자리에 생겨 있던 더한 상자 · 후보도 지운다
        self._added = [t for t in self._added
                       if not any(abs(t.x - x) < size - 0.05 and abs(t.y - y) < size - 0.05 for x, y in fixed)]
        self._serial += 1
        self._cache = None

    def drone_states(self):
        """드론 지도 순서대로 'drone'(아직 모름) · 'confirmed'(라이다로 확인) · 'rejected'(없는 상자)."""
        return ['rejected' if t.rejected else 'confirmed' if t.score >= self.params.confirm else 'drone' for t in self._drone]

    def _live(self):
        """정합에 쓰는 상자: 빼지 않은 드론 상자 + 더한 상자 (후보 제외)."""
        return [t for t in self._drone if not t.rejected] + [t for t in self._added if t.promoted]

    def structure(self, added=True):
        """정합에 쓸 (상자 가운데 (K, 2), 무게 (K,)). added=False 면 라이다로 더한 상자를 뺀다 (위치를 의심할 때 — 더한 상자는 그 위치로 만든 것이다)."""
        live = self._live() if added else [t for t in self._drone if not t.rejected]
        boxes = np.array([(t.x, t.y) for t in live], dtype=float).reshape(-1, 2)
        weights = np.array([1.0 if (t.promoted or t.score >= self.params.confirm) else self.params.unconfirmed_weight
                            for t in live], dtype=float)
        return boxes, weights

    def changed(self):
        """내보낼 상자가 드론 지도와 다른가 (뺀 상자나 옮긴 상자가 있나)."""
        return bool(self._moved) or any(t.rejected for t in self._drone)

    def output(self):
        """내보낼 상자 [(x, y, 드론 지도 번호)] — 빼지 않은 드론 상자, 지도 순서. 옮긴 상자는 라이다로 맞춘 자리로 나간다 (_relocate).

        더한 상자는 넣지 않는다. 라이다로만 본 것(서 있는 다른 UGV 일 수도 있다)을 고정 지도에 넣으면 주행 노드가
        그것을 벽처럼 다뤄 지도에 없는 물체 대응(정지 · 감속 · 물러나기)을 하지 않는다. 주행 노드는 그런 것을 /scan 으로 직접 본다.
        """
        key = (self._serial, self.version)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, [(t.x, t.y, t.source) for t in self._drone if not t.rejected])
        return list(self._cache[1])

    def added(self):
        """더한 상자 [(x, y)] — 정합에만 쓴다 (기록 · 시험 화면에서 볼 수 있게 꺼내 준다).

        옮긴 드론 상자의 새 자리도 여기에 넣는다: supervisor 는 처음 받은 지도와 자리가 다른 상자를 「없어진 상자」 로 보고,
        이 목록(/team/navigation/lidar_boxes)에서 새 자리의 상자를 받아 그 면을 찍는다 (지도에 없던 상자와 같은 길).
        """
        return [(t.x, t.y) for t in self._added if t.promoted] + [(t.x, t.y) for t in self._drone if t.source in self._moved and not t.rejected]

    def _relocate(self, timestamp_ns=None):
        """더한 상자가 드론 상자 바로 곁에 있으면 같은 상자다 → 그 드론 상자를 라이다로 맞춘 자리로 옮기고 더한 상자는 지운다 (FusionParams.move_m 의 설명)."""
        p = self.params
        for extra in [t for t in self._added if t.promoted]:
            best = None
            for drone in self._drone:
                if not drone.rejected and drone.score >= p.confirm:
                    continue  # 라이다로 확인한 상자는 옮기지 않는다
                distance = float(np.hypot(drone.x - extra.x, drone.y - extra.y))
                if distance <= (p.move_m if drone.rejected else p.overlap_m) and (best is None or distance < best[0]):
                    best = (distance, drone)
            if best is None:
                continue
            drone = best[1]
            self._moved.setdefault(drone.source, (drone.x, drone.y))
            drone.x, drone.y = extra.x, extra.y
            drone.rejected, drone.score = False, max(float(extra.score), p.confirm)  # 라이다로 본 자리다
            drone.changed_ns = timestamp_ns
            self._added.remove(extra)
            self.version += 1
            self.revision += 1

    def _inside(self, x, y):
        """상자 (x, y) 가 경기장 안에 놓일 수 있나: 외벽 밖으로 wall_overlap_m 넘게 나가지 않는다."""
        low = self._half - self.params.wall_overlap_m
        return low <= x <= self._width - low and low <= y <= self._height - low

    def distrust(self, since_ns):
        """since_ns 뒤로 고친 것을 되돌린다 — 그동안의 위치가 틀렸던 것이 드러났을 때 (slam_node 의 「벽으로 다시 보기」 가 위치를 옮길 때 부른다).

        - 그 뒤에 처음 본 더한 상자 · 후보는 지운다 (진짜 상자면 맞는 위치에서 몇 초 안에 다시 더해진다)
        - 그 뒤에 뺀 드론 상자는 되살리고, 되살린 것은 그대로 둔다 (드론 지도 쪽으로 되돌린다). 점수는 0 으로
        - 그 뒤에 옮긴 드론 상자는 드론 지도의 자리로 되돌린다
        바뀐 것이 있으면 True.
        """
        before = (self.version, self.revision)
        for track in list(self._added):
            if track.first_ns >= since_ns:
                self._added.remove(track)
                if track.promoted:
                    self.revision += 1
        for track in self._drone:
            if track.changed_ns is None or track.changed_ns < since_ns:
                continue
            origin = self._moved.pop(track.source, None)
            if origin is not None:
                track.x, track.y = origin
            elif not track.rejected:
                continue          # 그 사이에 되살린 것: 드론 지도와 같으니 그대로
            track.rejected, track.score, track.changed_ns = False, 0.0, None
            self.version += 1
            self.revision += 1
        return (self.version, self.revision) != before

    def summary(self):
        states = self.drone_states()
        return {'drone': len(states), 'confirmed': states.count('confirmed'), 'rejected': states.count('rejected'),
                'candidates': sum(1 for t in self._added if not t.promoted), 'added': sum(1 for t in self._added if t.promoted),
                'rejected_boxes': [[round(t.x, 2), round(t.y, 2)] for t in self._drone if t.rejected],
                'added_boxes': [[round(x, 2), round(y, 2)] for x, y in self.added()],
                'moved_boxes': [[round(origin[0], 2), round(origin[1], 2), round(t.x, 2), round(t.y, 2)]
                                for t in self._drone for origin in [self._moved.get(t.source)] if origin is not None],
                'version': self.version, **self.last}

    # ------------------------------------------------------------------ 관측
    def wants_scan(self):
        """이번 정합 결과로 지도를 고칠 차례인가 (observe_every 번에 한 번)."""
        due = self._calls % max(1, self.params.observe_every) == 0
        self._calls += 1
        return due

    def observe(self, points, pose, timestamp_ns, sensor=(0.0, 0.0), yaw_rate=0.0):
        """정합한 자세 pose 의 스캔 (차체 좌표 점 (P, 2)) 으로 지도를 고친다. 상자(내보내는 것이든 정합에만 쓰는 것이든)가 바뀌었으면 True."""
        p = self.params
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if self._width <= 0 or self._height <= 0 or self._half <= 0 or len(points) < 12:
            self.last = {'skip': '지도 없음' if self._width <= 0 else '점 부족'}
            return False
        if abs(yaw_rate) > p.max_yaw_rate:
            self.last = {'skip': '빠르게 도는 중'}
            return False
        world = _to_world(points, pose)
        origin = _to_world(np.array([sensor], dtype=float), pose)[0]
        outer = wall = _wall_distance(world, self._width, self._height)
        if len(self._fixed):     # 고정 기둥의 면도 외벽처럼 「아는 구조」 로 다룬다 (아래의 wall = 외벽 · 기둥까지의 거리)
            wall = np.minimum(outer, _surface_distance(world, self._fixed, self._half).min(axis=1))
        live = self._live()
        live_centres = np.array([(t.x, t.y) for t in live], dtype=float).reshape(-1, 2)
        live_distance = _surface_distance(world, live_centres, self._half) if len(live) else np.full((len(world), 0), np.inf)
        explained = np.minimum(wall, live_distance.min(axis=1, initial=np.inf)) < p.explain_m
        # 믿는 구조 = 외벽 + 라이다로 확인한 상자. 이 자세가 맞다는 근거는 아직 확인 못 한 드론 상자가 아니어야 한다.
        # 고정 기둥 전체는 넣지 않는다: 규칙의 기본 배치를 가정한 자리일 뿐이다 (2026-11-19 실차 기록 1118_202820 —
        # 그 자리에 기둥이 없는 모의 경기장에서 기둥을 믿는 점으로 세니 틀린 자세로도 지도를 고쳐, 5 cm 넘게 튄 위치가 35 → 133번이 됐다).
        # 본선 경기장에서 스캔으로 확인한 옆면(trust_pillar_faces)만 믿는다 — 경기장 쪽 면은 가정과 2 ~ 7 cm 달랐다
        sure = np.array([p.trust_confirmed_boxes and t.score >= p.confirm for t in live], dtype=bool)
        on_face = _face_distance(world, self._faces) < p.explain_m
        trusted = (outer < p.explain_m) | on_face | (live_distance[:, sure].min(axis=1, initial=np.inf) < p.explain_m)
        wall_points, trusted_points = int((outer < p.explain_m).sum()), int(trusted.sum())
        if explained.mean() < p.min_explained or trusted_points < p.min_trusted_points:
            self.last = {'skip': '자세를 믿기 어려움 (설명되는 점 %.0f%%, 믿는 점 %d = 벽 %d + 기둥 옆면 %d + 확인한 상자 %d)'
                         % (explained.mean() * 100, trusted_points, wall_points, int((on_face & ~(outer < p.explain_m)).sum()),
                            int((trusted & ~(outer < p.explain_m) & ~on_face).sum()))}
            return False

        before = self.revision
        tracks = self._drone + self._added
        centres = np.array([(t.x, t.y) for t in tracks], dtype=float).reshape(-1, 2)
        if len(tracks):
            self._evidence(tracks, centres, world, origin, wall, timestamp_ns)
        self._suggest(world, origin, wall, timestamp_ns)
        self._update_candidates(timestamp_ns)
        self.last = {'skip': ''}
        return self.revision != before

    def _evidence(self, tracks, centres, world, origin, wall, timestamp_ns):
        """상자마다: 라이다 쪽 면에 점이 맞으면 점수를 올리고, 광선이 속을 지나가면 깎는다.

        맞은 점 = 면에서 hit_m 안이고(외벽 위의 점은 빼고), 그 점까지의 광선이 이 상자 속을 지나지 않은 것.
        둘 다 있으면(한쪽은 맞고 한쪽은 비어 있음 = 조금 어긋난 진짜 상자) l_hit − l_miss 만큼 올린다 → 빼지 않는다.
        """
        p = self.params
        in_range = np.hypot(centres[:, 0] - origin[0], centres[:, 1] - origin[1]) <= p.max_range_m
        chords = _ray_chords(origin, world, centres, self._half - p.through_shrink_m, p.through_end_m)
        through = (chords >= p.through_chord_m).sum(axis=0)
        facing = (_surface_distance(world, centres, self._half) < p.hit_m) & (wall >= p.hit_m)[:, None] & (chords <= 0)
        hits = facing.sum(axis=0)
        for k, track in enumerate(tracks):
            if not in_range[k]:
                continue
            hit, miss = hits[k] >= p.hit_min, through[k] >= p.through_min
            narrow = p.revive_span_m > 0 and _face_span(world[facing[:, k]], track.x, track.y, self._half) < p.revive_span_m
            if hit and track.rejected and narrow:
                hit = False       # 뺀 상자 자리의 좁은 물체로는 되살리지 않는다 (revive_span_m)
            # 드론 상자는 다른 센서(사진)가 본 것이라 「없다」 를 절반만 믿는다. 라이다로 더한 것(서 있던 UGV 일 수도 있다)은 더할 때와 같은 세기로 뺀다
            l_miss = p.l_miss if track.source is not None else p.l_hit
            if hit:
                track.score = min(p.l_max, track.score + p.l_hit - (l_miss if miss else 0.0))
                if track.source is None:
                    track.seen += 1
                    track.last_ns = timestamp_ns
                    if not track.promoted:
                        track.span = max(track.span, _face_span(world[facing[:, k]], track.x, track.y, self._half))
            elif miss:
                track.score = max(-p.l_max, track.score - l_miss)
            else:
                continue
            if not track.rejected and track.score <= p.reject:
                track.rejected, track.changed_ns = True, timestamp_ns
                self._count_change(track)
            elif track.rejected and track.score >= p.revive:
                track.rejected, track.changed_ns = False, timestamp_ns
                self._count_change(track)

    def _count_change(self, track):
        """상자 하나가 빠지거나 되살아났다: 드론 상자면 내보내는 지도가, 더한 상자면 정합에 쓰는 상자만 바뀐다."""
        if track.source is not None:
            self.version += 1
            self.revision += 1
        elif track.promoted:
            self.revision += 1

    def _suggest(self, world, origin, wall, timestamp_ns):
        """지도(벽 · 기둥 · 빼지 않은 상자 · 후보)로 설명 안 되는 점에 맞는 정사각형 → 새 후보. 이미 있는 후보와 어긋난 점은 그 후보를 버리게 한다."""
        p = self.params
        known = [t for t in self._drone if not t.rejected] + list(self._added)
        known_centres = np.array([(t.x, t.y) for t in known], dtype=float).reshape(-1, 2)
        occupied = np.concatenate((known_centres, self._fixed))     # 새 후보가 겹치면 안 되는 자리 (기둥 포함)
        distance = wall.copy()
        if len(known):
            distance = np.minimum(distance, _surface_distance(world, known_centres, self._half).min(axis=1))
        reach = np.hypot(world[:, 0] - origin[0], world[:, 1] - origin[1])
        inside = (world[:, 0] > -0.05) & (world[:, 0] < self._width + 0.05) & (world[:, 1] > -0.05) & (world[:, 1] < self._height + 0.05)
        loose = world[(distance > p.explain_m) & inside & (reach <= p.max_range_m)]
        # 후보 가까이에 후보 면으로 설명 안 되는 점 = 물체가 옮겨 갔다 (움직이는 UGV) → 그 후보를 버린다
        for track in list(self._added):
            if track.promoted or not len(loose):
                continue
            close = np.maximum(np.abs(loose[:, 0] - track.x), np.abs(loose[:, 1] - track.y)) < self._half + p.conflict_m
            if int(close.sum()) >= p.hit_min:
                self._added.remove(track)
        if len(loose) < p.min_support or len(self._added) >= p.max_added:
            return
        if len(loose) > p.max_unexplained:
            loose = loose[np.linspace(0, len(loose) - 1, p.max_unexplained).astype(int)]
        # 후보 가운데: 점마다 그 점이 면 위에 오는 가운데들을 coarse_m 격자로
        step, reach_c = p.coarse_m, self._half + p.face_m
        offsets = np.arange(-reach_c, reach_c + 1e-9, step)
        grid = np.stack(np.meshgrid(offsets, offsets), axis=-1).reshape(-1, 2)
        centres = np.unique(np.round((loose[:, None, :] + grid[None, :, :]).reshape(-1, 2) / step).astype(int), axis=0) * step
        # 라이다 원점을 덮거나 아는 상자와 겹치는 가운데는 뺀다. 상자는 경기장 안에 놓여야 한다 (_inside — 벽에 붙은 고정 기둥은 fixed 로 따로 안다)
        low = self._half - p.wall_overlap_m
        keep = (np.maximum(np.abs(centres[:, 0] - origin[0]), np.abs(centres[:, 1] - origin[1])) > self._half + 0.10) \
            & (centres[:, 0] >= low) & (centres[:, 0] <= self._width - low) \
            & (centres[:, 1] >= low) & (centres[:, 1] <= self._height - low)
        for c in occupied:
            keep &= ~((np.abs(centres[:, 0] - c[0]) < self._size - 0.05) & (np.abs(centres[:, 1] - c[1]) < self._size - 0.05))
        centres = centres[keep]
        if not len(centres):
            return
        on_x, on_y = _face_support(loose, centres, origin, self._half, p.face_m)
        support = on_x | on_y
        remaining = np.ones(len(loose), dtype=bool)
        for _ in range(3):      # 한 번에 후보 셋까지
            counts = (support & remaining[None, :]).sum(axis=1)
            best = int(np.argmax(counts))
            if counts[best] < p.min_support:
                return
            used = support[best] & remaining
            x, y = _fit_centre(loose, on_x[best] & used, on_y[best] & used, centres[best], self._half)
            remaining &= ~used
            overlap = (np.abs(centres[:, 0] - x) < self._size - 0.05) & (np.abs(centres[:, 1] - y) < self._size - 0.05)
            support[overlap] = False
            chords = _ray_chords(origin, world, np.array([[x, y]]), self._half - p.through_shrink_m, p.through_end_m)
            if int((chords >= p.through_chord_m).sum()) >= p.through_min:
                continue        # 그 자리를 광선이 지나간다 = 상자가 아니다
            same = [t for t in self._added if np.hypot(t.x - x, t.y - y) < p.match_m]
            if same or not self._inside(x, y):       # 다듬은 가운데가 벽 밖으로 밀려 나갔으면 버린다
                continue
            self._added.append(_Track(x, y, seen=1, first_ns=timestamp_ns, last_ns=timestamp_ns, span=_face_span(loose[used], x, y, self._half)))

    def _update_candidates(self, timestamp_ns):
        p = self.params
        for track in list(self._added):
            if track.rejected:
                self._added.remove(track)          # 뺀 것으로 이미 revision 을 올렸다
            elif not self._inside(track.x, track.y):
                self._added.remove(track)          # 경기장 밖으로 나간 상자 (지도 치수가 바뀐 경우 등)
                if track.promoted:
                    self.revision += 1
            elif not track.promoted:
                if (timestamp_ns - track.last_ns) / 1e9 > p.candidate_ttl_s:
                    self._added.remove(track)
                elif (track.seen >= p.promote_seen and (track.last_ns - track.first_ns) / 1e9 >= p.promote_span_s
                      and track.span >= p.min_span_m):
                    track.promoted = True
                    self.revision += 1             # 정합에 쓰는 상자만 바뀐다 (내보내는 지도는 그대로)
        self._relocate(timestamp_ns)               # 더한 상자가 드론 상자 곁이면 그 드론 상자를 그 자리로 옮긴다 (내보내는 지도가 바뀐다)
