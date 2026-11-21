"""FieldMap 메시지에서 주행에 쓰는 것만 꺼낸다: 경기장 크기 · 박스 · 다른 UGV. 그리고 고정 기둥을 더한다.

FieldMap 은 칸 지도가 아니라 도형 목록이다 (벽 = 경기장 사각형, 박스 = 축에 나란한 정사각형).
그래서 「모르는 칸」 이 없다 — 드론이 빠뜨린 박스는 /scan 의 임시 장애물로 메운다 (obstacles.py).

고정 기둥 (2026-11-19): 벽에 붙은 노랑 · 검정 줄무늬 블록 10개. perception 의 지도는 벽 쪽 끝 띠 안을 내지 않아 기둥이 없다.
지도에 없으면 라이다가 기둥을 「모르는 물체」 로 보고 점마다 0.35 m 를 막는다 → 기둥 옆의 목표(거점 입구 · 위아래 벽 가운데 근처)는
「길 없음」 이 되어 그 자리에 선다 (2026-11-18 3차 팀 테스트: 목표 (2.62, 4.62) 가 위쪽 가운데 기둥 옆 → 0.47 m 앞에서 15 s 서 있다 BLOCKED).
그래서 주행 · 위치 추정은 규칙의 기본 배치(fixed_pillars)를 지도에 더해 쓴다. 메시지(FieldMap.boxes)에는 넣지 않는다 —
supervisor 가 기둥을 임무 상자로 보고 면을 찍으러 가면 안 된다 (supervisor 는 같은 배치를 자기 지도에 따로 넣는다: supervisor/arena.py).
자리는 연습 경기장 1 · 2번의 드론 사진으로 확인했다 (경기장 쪽 면이 벽에서 0.40 ~ 0.44 m, 가운데가 높이의 1/3 · 2/3 에서 ±4 cm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Pose2D, yaw_from_quaternion

ROLE_SELF = 1
ROLE_OTHER = 2


PILLAR_ARENA = ((5.5, 6.7), (4.5, 5.5))  # 기본 배치를 쓰는 경기장 크기 (가로 · 세로 범위, m). 공식 크기 6.12 × 5.00 근처에서만


def fixed_pillars(width, height, size=0.40):
    """고정 기둥 10개의 가운데 ((x, y), ...) — 한 변 size 의 정사각형으로 본다. 공식 크기가 아닌 경기장(시험용 작은 방 등)이면 빈 튜플.

    양쪽 벽의 줄 경계(높이의 1/3 · 2/3) 네 곳, 위아래 벽 가운데 두 곳, 네 모서리. 모서리 기둥(40 × 20 cm)은 벽에 반쯤 걸친 정사각형으로
    적는다 (경기장 안쪽 면의 자리는 같다). 거점(벽에서 0.40 m, 줄 가운데 ±0.60 m)은 이 기둥들 사이에 있다.
    """
    (w0, w1), (h0, h1) = PILLAR_ARENA
    if not (w0 <= width <= w1 and h0 <= height <= h1):
        return ()
    w, h, a = float(width), float(height), size / 2.0
    return ((a, h * 2.0 / 3.0), (a, h / 3.0), (w - a, h * 2.0 / 3.0), (w - a, h / 3.0),
            (w / 2.0, h - a), (w / 2.0, a),
            (a, 0.0), (a, h), (w - a, 0.0), (w - a, h))


PILLAR_FACE_HALF_M = 0.21  # 기둥 옆면(벽에 수직인 면)의 가운데에서의 거리. 본선 경기장 9판 스캔: 옆면은 0.20 m 자리에서 +0.2 ~ +2.3 cm (중앙)
PILLAR_FACE_DEPTH_M = 0.46  # 옆면이 벽에서 경기장 쪽으로 뻗은 길이. 기둥의 경기장 쪽 면은 벽에서 0.42 ~ 0.47 m 에 찍혔다 (가정 0.40)


def pillar_side_faces(width, height, fixed, start=0.05, half=PILLAR_FACE_HALF_M, depth=PILLAR_FACE_DEPTH_M):
    """모서리가 아닌 고정 기둥의 옆면(붙은 벽에 수직인 면) 선분들: (축, 자리, 시작, 끝).

    축 'y' = 선 y = 자리, x 가 [시작, 끝] (옆 벽에 붙은 기둥의 위 · 아래 면) / 축 'x' = 선 x = 자리, y 가 [시작, 끝] (위 · 아래 벽 가운데 기둥의 좌 · 우 면).
    붙은 벽 바로 곁 start 안은 뺀다 (그 점은 외벽 점으로 센다). 외벽은 벽에 나란한 방향만 잡아 주는데, 이 면은 벽을 따라가는 방향을 잡아 준다.
    본선 경기장 9판(2026-11-19)의 스캔에서 이 면은 가정한 자리에서 2.3 cm 안이었고, 기둥이 덮은 벽 구간에 찍힌 점은 4개뿐이었다 (기둥이 늘 있다).
    경기장 쪽 면(벽에 나란한 면)은 2 ~ 7 cm 어긋나 있어 쓰지 않는다. 모서리 기둥도 쓰지 않는다 (두 외벽이 이미 잡아 준다)."""
    faces = []
    edge = 0.3
    for cx, cy in fixed:
        on_left, on_right = cx < edge, cx > width - edge
        on_bottom, on_top = cy < edge, cy > height - edge
        if (on_left or on_right) == (on_bottom or on_top):
            continue  # 모서리 (또는 벽에 붙지 않은 것)
        if on_left or on_right:
            lo, hi = (start, depth) if on_left else (width - depth, width - start)
            faces += [('y', cy - half, lo, hi), ('y', cy + half, lo, hi)]
        else:
            lo, hi = (start, depth) if on_bottom else (height - depth, height - start)
            faces += [('x', cx - half, lo, hi), ('x', cx + half, lo, hi)]
    return tuple(faces)


@dataclass(frozen=True)
class MapSpec:
    width: float
    height: float
    box_size: float
    boxes: tuple  # ((x, y), ...) 박스 가운데 — 지도 메시지에 온 것 (임무 상자)
    frame_id: str = ""
    fixed: tuple = ()  # ((x, y), ...) 고정 기둥 가운데 (fixed_pillars). 박스와 같은 크기의 정사각형으로 본다

    @property
    def solids(self):
        """막힌 정사각형 전부 (박스 + 고정 기둥) — 격자 · 구조물까지의 거리 · 「박스 안」 판단에 쓴다."""
        return self.boxes + self.fixed

    def geometry_key(self):
        """격자를 다시 만들어야 하는지 가르는 값 (차량 위치만 바뀐 지도는 같은 값)."""
        return (round(self.width, 4), round(self.height, 4), round(self.box_size, 4), tuple((round(x, 4), round(y, 4)) for x, y in self.solids))

    def inside(self, x, y):
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def inside_box(self, x, y):
        half = self.box_size / 2.0
        return any(abs(x - bx) < half and abs(y - by) < half for bx, by in self.solids)

    def distance_to_structure(self, x, y):
        """(x, y) 에서 가장 가까운 벽 · 박스 면까지의 거리 (안쪽이면 0 이상, 박스 안이면 0)."""
        if self.inside(x, y):
            d = min(x, self.width - x, y, self.height - y)
        else:
            d = math.hypot(max(-x, x - self.width, 0.0), max(-y, y - self.height, 0.0))
        half = self.box_size / 2.0
        for bx, by in self.solids:
            ax = abs(x - bx) - half
            ay = abs(y - by) - half
            if ax <= 0.0 and ay <= 0.0:
                return 0.0
            box_d = math.hypot(max(ax, 0.0), max(ay, 0.0))
            if box_d < d:
                d = box_d
        return d


@dataclass
class MapUpdate:
    spec: MapSpec
    others: list = field(default_factory=list)  # [(x, y)] 다른 UGV (드론 화면 기준)
    self_pose: Pose2D | None = None


class InvalidMap(ValueError):
    pass


def from_field_map(msg, pillars=False) -> MapUpdate:
    """team_interfaces/FieldMap (또는 같은 필드를 가진 객체) → MapUpdate. pillars 가 참이면 고정 기둥의 기본 배치를 더한다
    (이미 그 자리에 지도 상자가 있으면 그 기둥은 뺀다 — 시험 화면처럼 기둥을 상자로 넣어 주는 지도와 겹치지 않게)."""
    width, height, box = float(msg.width), float(msg.height), float(msg.box_size)
    if not (width > 0.0 and height > 0.0):
        raise InvalidMap("지도 크기가 0 이하: %.3f × %.3f" % (width, height))
    if box <= 0.0:
        box = 0.40
    boxes = tuple((float(p.x), float(p.y)) for p in msg.boxes)
    frame = getattr(getattr(msg, "header", None), "frame_id", "") or ""
    fixed = ()
    if pillars:
        fixed = tuple(p for p in fixed_pillars(width, height, box) if not any(abs(p[0] - bx) < box / 2.0 and abs(p[1] - by) < box / 2.0 for bx, by in boxes))
    spec = MapSpec(width, height, box, boxes, frame, fixed)
    others, me = [], None
    for vehicle in getattr(msg, "vehicles", ()) or ():
        p = vehicle.pose
        if int(vehicle.role) == ROLE_OTHER:
            others.append((float(p.position.x), float(p.position.y)))
        elif int(vehicle.role) == ROLE_SELF:
            q = p.orientation
            me = Pose2D(float(p.position.x), float(p.position.y), yaw_from_quaternion(q.x, q.y, q.z, q.w))
    return MapUpdate(spec, others, me)
