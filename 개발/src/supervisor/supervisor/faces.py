"""상자 면 기록 — M3 의 중심.

하는 일
  1. expected_faces   로봇 위치와 카메라 방향으로 「이 사진에 찍혀야 할 면」 을 왼쪽부터 계산한다
  2. match            perception 이 답한 그림들(왼쪽부터)을 그 면들에 짝짓는다. 맞지 않으면 그 사진은 버린다 (None)
  3. FaceBook         면마다 본 결과를 모아 다수결로 정하고, 구역별 개수를 센다

ROS 를 쓰지 않는다. 좌표와 각도는 fieldmap.py 기준 (x BLUE 벽 → RED 벽, y 아래 → 위, 각도 0 = +x, 반시계 +).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import arena as A
from .coverage import _blocked_any


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class CameraParams:
    """카메라가 면을 어떻게 찍는지. [추정] 값은 현장에서 재서 설정으로 바꾼다."""

    hfov_deg: float = 127.4  # 가로 화각. 2026-11-18 로봇(8호기)에서 잼: 팬을 돌려 가며 1도에 5.02 픽셀 (640 픽셀 폭). 그 전 추정은 120
    lens: str = "equidistant"  # 가운데에서의 거리가 각도에 비례(광각 렌즈) · "rectilinear" = 보통 렌즈
    forward_m: float = 0.05  # 렌즈가 차체 가운데보다 앞에 있는 거리 [추정]
    edge_margin_deg: float = 2.0  # 사진 가장자리에서 이만큼은 「잘렸을 수 있다」 고 본다
    samples: int = 9  # 면 하나를 가로로 이만큼 나눠 보이는지 본다
    sure_fraction: float = 0.95  # 면의 이만큼이 보이면 「꼭 답에 있어야 하는 면」
    maybe_fraction: float = 0.20  # 이만큼도 안 보이면 「안 찍힌 면」
    sure_incidence_deg: float = 62.0  # 면을 이보다 비스듬히 보면 「있을 수도 없을 수도」
    max_incidence_deg: float = 82.0  # 이보다 비스듬하면 「안 찍힌 면」
    sure_range_m: float = 4.0  # 이보다 먼 면은 「있을 수도 없을 수도」 (경기장이 작아 대부분 이 안이다)
    aspect: float = 0.75  # 사진의 세로 / 가로 (480 / 640). 면이 사진에 찍힐 세로 크기를 어림할 때 쓴다

    def x_of(self, rel):
        """카메라 정면에서 왼쪽으로 rel(rad) 인 방향이 사진 가로 어디에 찍히나 (0 = 왼쪽 끝, 1 = 오른쪽 끝)."""
        half = math.radians(self.hfov_deg) / 2.0
        if self.lens == "rectilinear":
            return 0.5 - math.tan(max(-1.5, min(1.5, rel))) / (2.0 * math.tan(min(half, 1.5)))
        return 0.5 - rel / (2.0 * half)


@dataclass(frozen=True)
class Expected:
    """사진에 찍혀야 할 면 하나."""

    face: object  # fieldmap.Face
    x: float  # 사진 속 예상 가로 위치 (보이는 부분의 가운데, 0 왼쪽 ~ 1 오른쪽)
    rel: float  # 카메라 정면에서 본 방향 (rad, 왼쪽 +)
    dist: float  # 렌즈에서 면 가운데까지 (m)
    incidence_deg: float  # 면을 정면에서 보면 0, 옆에서 스치듯 보면 90
    fraction: float  # 면 가로 중 보이는 비율 (가림 · 사진 가장자리)
    sure: bool  # 꼭 답에 있어야 하는 면인가 (False = 있을 수도 없을 수도)
    # 사진에 찍힐 크기의 어림 (0 = 모름). perception 이 답한 그림의 크기와 견줘, 그 자리에 찍힌 그림이 정말 이 면의 것인지 가린다
    # (상자가 옮겨져 그 자리에 먼 상자의 그림이 찍히면 훨씬 작다 — scout.pick_each)
    width: float = 0.0  # 가로: 사진 폭에 대한 비율 (보이는 부분만)
    height: float = 0.0  # 세로: 사진 높이에 대한 비율 (면에서 렌즈에 가장 가까운 보이는 곳의 거리로 계산)


def camera_pose(pose, pan_deg=0.0, cam=None):
    """로봇 위치 (x, y, yaw) 와 팬(오른쪽 +, 도) → 렌즈 자리와 카메라가 보는 방향."""
    cam = cam or CameraParams()
    x, y, yaw = pose
    heading = yaw - math.radians(pan_deg)
    return x + cam.forward_m * math.cos(heading), y + cam.forward_m * math.sin(heading), heading


def expected_faces(fmap, pose, pan_deg=0.0, cam=None, segs=None):
    """이 자리 · 이 방향에서 찍은 사진에 들어오는 면들. 왼쪽부터 (x 가 작은 것부터).

    면마다 가로로 samples 곳을 찍어, 렌즈에서 곧게 보이고(벽 · 다른 상자에 안 가림) 사진 안에 드는 곳의 비율을 센다.
    """
    cam = cam or CameraParams()
    cx, cy, heading = camera_pose(pose, pan_deg, cam)
    if segs is None:
        segs = np.array(fmap.segments(), dtype=np.float64)
    half = math.radians(cam.hfov_deg) / 2.0 - math.radians(cam.edge_margin_deg)
    n = max(3, int(cam.samples))
    size = fmap.box_size
    alongs = [(-0.5 + (k + 0.5) / n) * size for k in range(n)]
    out = []
    gone = getattr(fmap, "gone", ())
    for face in fmap.faces:
        if face.box in gone:
            continue  # navigation 이 라이다로 없다고 본 상자 — 사진에 찍힐 것으로 치지 않는다 (segments 에서도 빠져 뒤의 면을 가리지 않는다)
        dx, dy = cx - face.cx, cy - face.cy
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            continue
        cos_inc = (dx * face.nx + dy * face.ny) / dist
        if cos_inc <= 0.0:
            continue  # 면의 뒤쪽에 있다
        incidence = math.degrees(math.acos(max(-1.0, min(1.0, cos_inc))))
        if incidence > cam.max_incidence_deg:
            continue
        pts = [face.point(a) for a in alongs]
        qx = np.array([p[0] + face.nx * 0.01 for p in pts])
        qy = np.array([p[1] + face.ny * 0.01 for p in pts])
        rels = np.array([wrap(math.atan2(qy[k] - cy, qx[k] - cx) - heading) for k in range(n)])
        inside = np.abs(rels) <= half
        if not inside.any():
            continue
        blocked = _blocked_any(np.full(n, cx), np.full(n, cy), qx, qy, segs)
        seen = inside & ~blocked
        fraction = float(seen.mean())
        if fraction < cam.maybe_fraction:
            continue
        rel = float(rels[seen].mean())
        sure = fraction >= cam.sure_fraction and incidence <= cam.sure_incidence_deg and dist <= cam.sure_range_m
        # 사진에 찍힐 크기: 가로는 면의 두 끝이 찍히는 자리의 차이 × 보이는 비율, 세로는 상자 높이(= 한 변)를 가장 가까운 보이는 곳의 거리에서 본 각도
        (ax, ay), (bx, by) = face.point(-size / 2.0), face.point(size / 2.0)
        width = abs(cam.x_of(wrap(math.atan2(ay - cy, ax - cx) - heading)) - cam.x_of(wrap(math.atan2(by - cy, bx - cx) - heading))) * fraction
        near = float(np.hypot(qx - cx, qy - cy)[seen].min())
        height = 2.0 * math.atan2(size / 2.0, max(near, 0.05)) / (math.radians(cam.hfov_deg) * cam.aspect)
        out.append(Expected(face, cam.x_of(rel), rel, dist, incidence, fraction, sure, width, height))
    out.sort(key=lambda e: e.x)
    return out


@dataclass
class MatchParams:
    tol_loose: float = 0.12  # 첫 맞춤: 예상 위치와 답의 위치가 사진 폭의 이만큼 넘게 다르면 짝이 아니다
    tol_tight: float = 0.05  # 사진 전체의 밀림 · 늘어남을 고친 뒤의 허용 차이
    res_max: float = 0.035  # 적어 둘 만한 짝: 고친 뒤 차이가 이 안
    sep_min: float = 0.03  # 적어 둘 만한 짝: 그 그림에 두 번째로 가까운 면이 이만큼은 더 멀어야 한다 (헷갈리는 짝은 적지 않는다)
    skip_maybe: float = 0.03  # 「있을 수도」 인 면에 답이 없을 때의 벌점
    skip_cut: float = 0.03  # 가장자리에서 잘린 그림에 면이 없을 때의 벌점
    forbid: float = 100.0  # 「꼭 있어야 하는 면」 에 답이 없거나, 멀쩡한 그림에 면이 없을 때 (그 사진은 버린다)
    max_shift: float = 0.10  # 사진 전체가 이보다 많이 밀려 있으면 버린다 (위치가 크게 틀렸다)
    scale_range: tuple = (0.80, 1.25)  # 사진 전체의 늘어남(화각이 틀렸을 때)을 이 안에서만 고친다


def _align(xs, sure, pics, tol, p):
    """왼쪽부터 순서를 지키며 면(xs)과 그림(pics)을 짝짓는 가장 싼 방법. [(면 번호, 그림 번호)] 또는 None."""
    n, m = len(xs), len(pics)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == inf:
                continue
            if i < n:  # 면 i 에 답 없음
                step = p.forbid if sure[i] else p.skip_maybe
                if c + step < cost[i + 1][j]:
                    cost[i + 1][j], back[i + 1][j] = c + step, "face"
            if j < m:  # 그림 j 에 면 없음
                step = p.skip_cut if pics[j][2] else p.forbid
                if c + step < cost[i][j + 1]:
                    cost[i][j + 1], back[i][j + 1] = c + step, "pic"
            if i < n and j < m:
                d = abs(xs[i] - pics[j][0])
                if d <= tol and c + d < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1], back[i + 1][j + 1] = c + d, "pair"
    if cost[n][m] >= p.forbid:
        return None
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "pair":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "face":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _fit(xe, xp, p):
    """그림 위치 ≈ a + b × 예상 위치 가 되는 (a, b). 방향이 조금 틀리면 사진 전체가 밀리고(a), 화각이 틀리면 늘어난다(b)."""
    k = len(xe)
    if k == 0:
        return 0.0, 1.0
    b = 1.0
    if k >= 3 and max(xe) - min(xe) >= 0.25:
        me, mp = sum(xe) / k, sum(xp) / k
        var = sum((v - me) ** 2 for v in xe)
        if var > 1e-9:
            b = sum((xe[t] - me) * (xp[t] - mp) for t in range(k)) / var
            b = max(p.scale_range[0], min(p.scale_range[1], b))
    diffs = sorted(xp[t] - (0.5 + b * (xe[t] - 0.5)) for t in range(k))
    a = diffs[k // 2] if k % 2 else 0.5 * (diffs[k // 2 - 1] + diffs[k // 2])
    return a, b


def match(expected, pictures, params=None, info=None):
    """perception 의 답(왼쪽부터)을 사진에 찍혀야 할 면들에 짝짓는다.

    expected: expected_faces 의 결과. pictures: [(kind, x_center, cut_off)] — x_center 0~1. 뒤에 크기 등이 더 붙어 있어도 된다 (앞의 셋만 쓴다).
    돌려주는 값: [(Expected, kind)] — **적어 둘 만한 짝만**. 사진이 면 목록과 맞지 않으면 None (그 사진은 쓰지 않는다).

    순서만 세면 그림 하나가 빠졌을 때 뒤가 전부 밀린다. 그래서
      1. 위치도 같이 보며 순서대로 짝짓는다. 「있을 수도 없을 수도」 인 면(가장자리 · 많이 가린 면 · 아주 비스듬한 면)은 답이 없어도 된다.
         꼭 있어야 하는 면에 답이 없거나 멀쩡한 그림에 면이 없으면 버린다
      2. 짝들로 사진 전체의 밀림 · 늘어남을 구해 고치고(로봇 방향 · 화각의 오차), 좁은 허용 차이로 다시 짝짓는다
      3. 그림 가까이에 다른 면도 있어 헷갈리는 짝은 적지 않는다
    info(dict)를 주면 밀림 · 늘어남 · 짝 수를 넣어 준다 (기록 · 화각 보정용).
    """
    p = params or MatchParams()
    pics = sorted(((float(q[1]), q[0], bool(q[2])) for q in pictures), key=lambda q: q[0])
    xs = [e.x for e in expected]
    sure = [e.sure for e in expected]
    first = _align(xs, sure, pics, p.tol_loose, p)
    if first is None:
        return None
    anchors = [(i, j) for i, j in first if sure[i] and not pics[j][2]] or first
    a, b = _fit([xs[i] for i, _j in anchors], [pics[j][0] for _i, j in anchors], p)
    if abs(a) > p.max_shift:
        return None
    xs2 = [a + 0.5 + b * (x - 0.5) for x in xs]
    second = _align(xs2, sure, pics, p.tol_tight, p)
    if second is None:
        return None
    out = []
    for i, j in second:
        res = abs(pics[j][0] - xs2[i])
        other = min((abs(pics[j][0] - xs2[k]) for k in range(len(xs2)) if k != i), default=1.0)
        if res <= p.res_max and other - res >= p.sep_min:
            out.append((expected[i], pics[j][1]))
    if info is not None:
        info.update({"shift": a, "scale": b, "pairs": len(second), "kept": len(out)})
    return out


class FaceBook:
    """면마다 본 결과. 같은 면을 여러 번 보면 다수결로 정한다 (가깝고 정면인 「잘 본 것」 은 1표, 나머지는 0.3표)."""

    # 적는 범위: 이보다 멀거나 비스듬하게 본 것은 적지 않는다. 가상 시험(test/test_faces.py)에서 위치 3 cm · 2° 오차일 때
    # 짝이 틀리는 비율이 1.2 m · 40° 안 0%, 1.5 m · 50° 안 0.2%, 2 m · 60° 안 1%, 3 m 밖 6~17% 였다
    record_range_m = 1.6
    record_incidence_deg = 55.0
    good_range_m = 1.2
    good_incidence_deg = 40.0
    weak_weight = 0.3
    bonus_weight = 0.25  # 덤(겨누지 않은 면)의 표는 이만큼만 친다
    # 「본 것」 여럿으로 적은 면(sight)의 표. 겨눠 얻은 답 하나(가장 작아도 0.3표)보다 작게 둔다 → 겨눠 얻은 답이 오면 그것이 이긴다
    sight_weight = 0.2
    # M3 개수(counts)에 「본 것」 이 아직 한 장뿐인 면도 넣는다 — 그 면에 표가 하나도 없고, 본 답이 모두 같고, navigation 이 뺀 상자가 아닐 때만 (sighted).
    # 정찰 시간이 모자라 그 면을 다시 못 보고 끝나면, 넣지 않을 때는 그 면이 늘 빠진 채 보고된다 (구역 점수는 개수가 다 맞아야 받는다).
    # 표가 아니므로 정찰 계획(remaining) · 지정표적 찾기(targets) · 다수결에는 끼지 않는다
    count_sighted = True
    sure_share = 0.75  # 가장 많은 표가 이 몫에 못 미치면 「갈렸다」

    def __init__(self, fmap, camp):
        self.fmap = fmap
        self.camp = camp
        self.scored = A.scored_sectors(camp)
        self.by_key = {f.key: f for f in fmap.faces}
        self.goal = [f for f in fmap.faces if f.sector in self.scored]
        self.votes = {}  # 면 key → {종류: 표}
        self.looks = {}  # 면 key → {"n": 본 횟수, "good": 잘 본 횟수, "unknown": 못 알아본 횟수}
        self.last = {}  # 면 key → 마지막으로 본 종류 (표가 같을 때 쓴다)
        # 「본 것」: 겨누지 않았어도 사진(정찰 사진 · 지나가며 찍은 사진)에 또렷이 찍힌 면의 답. 한 장으로는 적지 않고, 다른 자리에서 찍은
        # 여러 장이 같은 답일 때만 적는다 (sight). 면 key → [(종류, 렌즈 x, 렌즈 y, 카메라 방향 rad, 찍은 시각 또는 None)]
        self.sights = {}
        self.voted = {}  # 면 key → 「본 것」 만으로 적은 종류 (겨눠 얻은 답이 오면 여기서 빠진다)

    def add_faces(self, faces):
        """뒤에 생긴 면(지도에 없던 상자의 면 — mission 의 지도에 없는 상자)을 넣는다. 채점 구역의 면만 목표(goal)에 든다.

        돌려주는 값: 이번에 목표에 새로 든 면들. 이미 있는 면은 다시 넣지 않는다.
        """
        new = []
        for f in faces:
            if f.key in self.by_key:
                continue
            self.by_key[f.key] = f
            if f.sector in self.scored:
                self.goal.append(f)
                new.append(f)
        return new

    def add(self, exp, kind, bonus=False, limit=None):
        """사진 한 장에서 면 하나에 대해 얻은 답 하나를 적는다. 적었으면 True (너무 멀거나 비스듬하면 적지 않는다).

        bonus = 겨눈 면이 아니라 같은 사진에 함께 찍힌 다른 면의 답 (덤). 덤은
          · 그 면에 이미 다른 종류가 적혀 있으면 적지 않는다 (겨눠서 얻은 답 · 먼저 얻은 답을 덤이 뒤집지 못한다)
          · 표를 bonus_weight 배로 줄여 준다 — 덤 하나(가장 커도 0.25표)는 겨눠 얻은 답 하나(가장 작아도 0.3표)를 이기지 못한다
        limit = (거리 m, 비스듬함 °) 를 주면 적는 범위를 그것으로 바꾼다 (정찰의 덤 설정 each_range_m · each_incidence_deg 가 그대로 듣게).
        """
        far, steep = limit or (self.record_range_m, self.record_incidence_deg)
        if exp.dist > far or exp.incidence_deg > steep:
            return False
        key = exp.face.key
        unknown = kind == A.UNKNOWN or kind not in A.ALL_KINDS
        if bonus and (unknown or self.kind(key) not in (None, kind)):
            return False
        rec = self.looks.setdefault(key, {"n": 0, "good": 0, "unknown": 0})
        rec["n"] += 1
        if unknown:
            rec["unknown"] += 1
            return True
        good = exp.dist <= self.good_range_m and exp.incidence_deg <= self.good_incidence_deg
        rec["good"] += 1 if good and not bonus else 0
        if not bonus:
            self.voted.pop(key, None)  # 겨눠 얻은 답이 왔다 → 이제 「본 것만으로 적은 면」 이 아니다 (표는 겨눠 얻은 답이 더 크다)
        votes = self.votes.setdefault(key, {})
        votes[kind] = votes.get(kind, 0.0) + (1.0 if good else self.weak_weight) * (self.bonus_weight if bonus else 1.0)
        self.last[key] = kind
        return True

    @staticmethod
    def _apart(a, b, apart):
        """두 「본 것」 이 서로 다른 사진으로 칠 만큼 떨어져 있는가. apart = (렌즈 자리 m, 카메라 방향 °, 시각 초) — 어느 하나라도 넘으면 된다 (0 이하인 것은 보지 않는다)."""
        far, turn, gap = apart
        if far > 0.0 and math.hypot(a[1] - b[1], a[2] - b[2]) >= far:
            return True
        if turn > 0.0 and abs(math.degrees(wrap(a[3] - b[3]))) >= turn:
            return True
        return gap > 0.0 and a[4] is not None and b[4] is not None and abs(a[4] - b[4]) >= gap

    def sight(self, key, kind, view, need=2, apart=(0.10, 5.0, 0.3)):
        """겨누지 않은 사진에서 면 하나를 또렷이 본 것 하나를 넣는다. 이번에 그 면이 「본 것」 으로 적혔으면 그 종류, 아니면 None.

        view = (렌즈 x, 렌즈 y, 카메라 방향 rad, 찍은 시각 또는 None). 적는 조건:
          · 그 면에 겨눠 얻은 답 · 덤이 아직 없다 (있으면 「본 것」 은 그것을 바꾸지 못한다)
          · 지금까지 본 것이 모두 같은 종류다 — 다른 종류가 하나라도 있으면 적지 않고, 「본 것」 만으로 적어 둔 것이 있으면 지운다 (다시 겨눠 찍는다)
          · 서로 떨어진 사진(apart)이 need 장 이상이다
        적을 때의 표는 sight_weight (겨눠 얻은 답보다 작다). 2026-11-19 14:52 실차: 한 판에서 사진에 또렷이 찍히고도 못 적은 면이 여럿이었다.
        """
        if key not in self.by_key or kind == A.UNKNOWN or kind not in A.ALL_KINDS:
            return None
        seen = self.sights.setdefault(key, [])
        seen.append((kind,) + tuple(view))
        mine = key in self.voted and self.votes.get(key) == {self.voted[key]: self.sight_weight}  # 「본 것」 만으로 적혀 있다
        if len({s[0] for s in seen}) > 1:  # 갈렸다
            if mine:
                del self.votes[key]
                self.last.pop(key, None)
                self.looks.pop(key, None)
            self.voted.pop(key, None)
            return None
        if key in self.votes:
            return None
        distinct = []
        for s in seen:
            if all(self._apart(s, o, apart) for o in distinct):
                distinct.append(s)
        if len(distinct) < max(1, int(need)):
            return None
        self.votes[key] = {kind: self.sight_weight}
        self.last[key] = kind
        self.looks[key] = {"n": len(seen), "good": 0, "unknown": 0}
        self.voted[key] = kind
        return kind

    def sighted(self, key):
        """표는 없는데 「본 것」 은 있는 면: 본 답이 모두 같으면 그 종류, 아니면 None.

        sight 가 적을 만큼(서로 떨어진 사진 need 장)은 안 된 면이다 — 한 장뿐이거나, 두 장이 같은 자리에서 찍혔거나.
        표가 있는 면(겨눠 얻은 답 · 덤 · 「본 것」 으로 적힌 것)은 None (그 면의 종류는 kind 로 본다).
        """
        if key in self.votes:
            return None
        seen = self.sights.get(key)
        if not seen:
            return None
        kinds = {s[0] for s in seen}
        return next(iter(kinds)) if len(kinds) == 1 else None

    def add_all(self, pairs):
        """짝들을 적고, 적은 면의 key 집합을 돌려준다."""
        return {exp.face.key for exp, kind in pairs or () if self.add(exp, kind)}

    def kind(self, key):
        """그 면의 종류 (다수결). 아직 답이 없으면 None."""
        votes = self.votes.get(key)
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: (kv[1], kv[0] == self.last.get(key)))[0]

    def share(self, key):
        votes = self.votes.get(key)
        if not votes:
            return 0.0
        return max(votes.values()) / sum(votes.values())

    def unsure(self, key):
        """잘 본 적이 없거나 표가 갈린 면인가."""
        rec = self.looks.get(key)
        return rec is None or rec["good"] == 0 or self.share(key) < self.sure_share

    def remaining(self):
        """채점 구역의 면 중 아직 종류를 하나도 못 얻은 면."""
        return {f.key for f in self.goal if f.key not in self.votes}

    def counts(self):
        """{구역: {객체 종류: 개수}} — 채점 구역 넷을 모두 담는다 (본 것이 없으면 빈 것)."""
        out = {sector: {} for sector in self.scored}
        gone = getattr(self.fmap, "gone", ())
        for f in self.goal:
            k = self.kind(f.key)
            if k is None and self.count_sighted and f.box not in gone:
                k = self.sighted(f.key)  # 다시 볼 틈이 없었던 면: 한 장뿐인 「본 것」 이라도 다른 답이 없으면 센다
            if k in A.OBJECT_KINDS:
                out[f.sector][k] = out[f.sector].get(k, 0) + 1
        return out

    def targets(self):
        """지정표적으로 판독된 면들: [(면, 종류, 그 면이 놓인 진영)]."""
        out = []
        for key in self.votes:
            k = self.kind(key)
            if k in A.TARGET_KINDS:
                face = self.by_key[key]
                out.append((face, k, A.side_of(self.fmap, face.cx)))
        return out

    def find_target(self, value):
        """지정표적 보고값(RED_RADAR)에 해당하는 면. 아직 못 봤으면 None."""
        want = A.target_kind(value)
        if want is None:
            return None
        for face, k, camp in self.targets():
            if (camp, k) == want:
                return face
        return None

    def coverage(self):
        total = len(self.goal)
        return 1.0 if not total else sum(1 for f in self.goal if f.key in self.votes) / total

    def snapshot(self):
        """파일로 저장할 수 있는 모양 (프로그램이 다시 켜졌을 때 되살리려고)."""
        return {"votes": {"%d:%s" % k: v for k, v in self.votes.items()},
                "looks": {"%d:%s" % k: v for k, v in self.looks.items()},
                "last": {"%d:%s" % k: v for k, v in self.last.items()},
                "voted": {"%d:%s" % k: v for k, v in self.voted.items()},
                "sights": {"%d:%s" % k: [list(s) for s in v] for k, v in self.sights.items()}}

    def restore(self, data):
        def key(text):
            box, side = text.split(":")
            return int(box), side
        self.votes = {key(k): dict(v) for k, v in (data.get("votes") or {}).items() if key(k) in self.by_key}
        self.looks = {key(k): dict(v) for k, v in (data.get("looks") or {}).items() if key(k) in self.by_key}
        self.last = {key(k): v for k, v in (data.get("last") or {}).items() if key(k) in self.by_key}
        self.voted = {key(k): v for k, v in (data.get("voted") or {}).items() if key(k) in self.votes}
        # 「본 것」 (아직 적히지 않은 한 장짜리 포함). 전의 저장 파일에는 없다 → 빈 것. 모양이 틀린 줄은 버린다 (되살리다 멈추면 안 된다)
        self.sights = {}
        for k, v in (data.get("sights") or {}).items():
            try:
                rows = [tuple(s) for s in v if len(s) == 5]
            except TypeError:
                continue
            if rows and key(k) in self.by_key:
                self.sights[key(k)] = rows
