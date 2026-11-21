# 출처: sim-strategy 브랜치(e0300e3)의 team/navigation/fieldmap.py 를 복사했다.
# 2026-11-18 고친 곳: 면이 속한 구역을 서버의 채점 규칙(그림 판의 한가운데 · 경계는 오른쪽 · 아래)으로 정한다 (sector_at · face_sector).
#                  상자가 경계선 가까이 있으면 선에 맞춰 놓인 것으로 본다 (_snap).
#                  맞닿은 상자를 알아보는 여유를 3 cm → 8 cm (TOUCH_TOL) — 드론 사진에서 잰 위치로 확인.
# 2026-11-19 고친 곳: 고정 기둥(pillars) — 벽에 붙은 노랑 · 검정 블록. 막힌 곳 · 시야를 가리는 것으로만 치고 면은 만들지 않는다.
#                  지도에 없던 상자를 뒤에 넣기(add_box) — navigation 이 라이다로 찾은 상자. 번호는 맨 뒤에 붙는다.
"""경기장 지도: 벽 · 박스 · 거점 · 섹터. 이동 모델이 아는 것은 이것뿐이다.

가정 (권우진 결정 2026-11-12): 드론 캠으로 정상적인 맵을 얻었다 → 박스 위치는 안다.
박스의 어느 면에 무엇(객체 · 표적 · 비객체)이 붙었는지는 모른다 → 직접 가서 봐야 한다.
박스 위치는 team/pipeline.py 가 첫 드론 화면에서 잰다 (team/localization/drone.py). 벽 · 거점 · 섹터는 공개 도면이다.

좌표계는 mock/ 시뮬과 같다: x 는 BLUE 쪽 벽(0) → RED 쪽 벽, y 는 아래(0) → 위, 1번 줄이 위쪽. 단위 m.
실제 경기장에서 지도를 만드는 쪽(드론 영상 → 박스 좌표)이 이 좌표계로 넘겨 주면 된다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

SIDES = ("E", "N", "W", "S")
NORMAL = {"E": (1.0, 0.0), "N": (0.0, 1.0), "W": (-1.0, 0.0), "S": (0.0, -1.0)}
CAMPS = ("BLUE", "RED")
TOUCH_TOL = 0.08  # 두 상자의 가운데 사이가 상자 크기 ± 이 값이고 옆으로 이 값 안이면 맞닿은 것으로 본다 (맞닿은 면은 보이지 않는다)
LINE_SNAP = 0.07  # 상자의 변이나 가운데가 구역 경계선에서 이만큼 안이면 선에 맞춰 놓인 것으로 본다 (드론 사진에서 잰 위치는 몇 cm 틀린다)
PLATE_INSET = 0.05  # 그림 판(두께 10 cm)의 한가운데가 상자 면에서 안쪽으로 들어간 거리


def opponent(camp):
    return "RED" if camp == "BLUE" else "BLUE"


def scored_sectors(camp):
    """M3 채점 구역: 상대 진영 3구역 + CM."""
    p = "R" if camp == "BLUE" else "B"
    return ("%s-S1" % p, "%s-S2" % p, "%s-S3" % p, "CM")


@dataclass(frozen=True)
class Face:
    """박스의 한 면. key = (박스 번호, 방향)."""

    box: int
    side: str
    cx: float  # 면 가운데
    cy: float
    nx: float  # 바깥 법선
    ny: float
    sector: str | None

    @property
    def key(self):
        return (self.box, self.side)

    def point(self, along):
        """면 위의 점: 가운데에서 면을 따라 along(m) 만큼 (바깥에서 볼 때 왼쪽이 +)."""
        tx, ty = -self.ny, self.nx
        return self.cx + tx * along, self.cy + ty * along


@dataclass
class FieldMap:
    width: float
    height: float
    box_size: float
    boxes: list  # [(x, y)] 박스 가운데
    bases: dict  # 보고값(예 RED_MAINTENANCE) → (x0, y0, x1, y1)
    base_rows: dict  # 보고값 → 줄 번호 1~3
    sectors: dict  # 섹터 이름 → (x0, y0, x1, y1)
    beam_depth: float = 0.20  # 거점 광전센서 빔이 벽에서 떨어진 거리 [추정]
    faces: list = field(default_factory=list)
    pillars: list = field(default_factory=list)  # [(x, y)] 고정 기둥 가운데 — 상자와 같은 크기의 막힌 정사각형. 임무 상자가 아니라 면이 없다
    # navigation 이 라이다로 「그 자리에 없다」 고 본 상자의 번호 (지도 고치기 map_fusion, 2026-11-19). 번호 · 면 목록은 처음 지도 그대로 두고
    # (면 기록 · 정찰 자리 · 저장 파일이 번호로 묶여 있다), 시야 가림과 사진 짝짓기에서만 뺀다. 다시 보이면 navigation 이 되살린다
    gone: frozenset = frozenset()
    # 거점 종류(작전지휘소 · 정비소 · 군수보급소)가 어느 줄에 있는지를 지도 메시지가 보고값 이름으로 알려 준 진영 (perception 이 벽 그림에서 읽었거나
    # 맞은편 진영의 것으로 채운 것). 비어 있으면 그 진영의 거점 이름은 도면 기본 배치를 따른 것이라 자리가 실제와 다를 수 있다
    kinds_read: frozenset = frozenset()

    def __post_init__(self):
        if not self.faces:
            self.faces = self._make_faces()

    # ---- 조회
    def sector_at(self, x, y):
        """점이 속한 구역. 경계 위의 점은 서버와 같게 「오른쪽 · 아래」 구역으로 친다 (포털의 sectorOfPoint 규칙).

        서버의 y 는 위에서 아래로 자라고 우리 y 는 아래에서 위로 자라므로, 가로 경계 위의 점은 y 가 작은 쪽(아래 줄)에 넣는다.
        """
        eps = 1e-6
        for name, (x0, y0, x1, y1) in self.sectors.items():
            if x0 - eps <= x < x1 - eps and y0 + eps < y <= y1 + eps:
                return name
        return None

    def face_sector(self, x, y, nx, ny):
        """상자(가운데 x, y)의 한 면(바깥 방향 nx, ny)에 붙은 그림이 속한 구역 — 서버의 채점 규칙대로 (2026-11-18 포털에서 확인).

        서버는 그림(판)의 한가운데로 구역을 정한다. 판은 두께 10 cm 로 상자 변의 안쪽에 겹쳐 있으므로 한가운데는 면에서 5 cm 안쪽이다.
        상자는 구역 경계선에 맞춰 놓으므로(격자 20 cm), 잰 위치에서 상자의 변이나 가운데가 경계선에 아주 가까우면 그 선에 맞춘 뒤 계산한다
        — 잰 값이 몇 cm 틀려도 경계에 붙은 면이 옆 구역으로 넘어가지 않게. 경기장 전체의 격자 원점은 가정하지 않는다
        (지도 좌표가 벽 기준인지 바닥 격자 기준인지에 따라 달라진다).
        """
        half = self.box_size / 2.0
        xs = sorted({v for x0, _y0, x1, _y1 in self.sectors.values() for v in (x0, x1)})
        ys = sorted({v for _x0, y0, _x1, y1 in self.sectors.values() for v in (y0, y1)})
        x, y = self._snap(x, xs, half), self._snap(y, ys, half)
        reach = half - PLATE_INSET
        return self.sector_at(x + nx * reach, y + ny * reach)

    @staticmethod
    def _snap(value, lines, half):
        """value(상자 가운데의 한 좌표)를, 상자의 변(±half)이나 가운데가 가장 가까운 경계선에 놓이도록 옮긴다 (LINE_SNAP 안일 때만)."""
        best = 0.0
        gap = LINE_SNAP
        for line in lines:
            for offset in (-half, 0.0, half):
                d = line - (value + offset)
                if abs(d) < gap:
                    gap, best = abs(d), d
        return value + best

    def base_camp(self, value):
        return "BLUE" if value.startswith("BLUE") else "RED"

    def bases_of(self, camp):
        return {v: r for v, r in self.bases.items() if self.base_camp(v) == camp}

    def base_in_row(self, camp, row):
        for value, r in self.base_rows.items():
            if r == row and self.base_camp(value) == camp:
                return value
        return None

    def beam_x(self, value):
        x0, _y0, x1, _y1 = self.bases[value]
        return self.beam_depth if self.base_camp(value) == "BLUE" else self.width - self.beam_depth

    def inward(self, value):
        """거점에서 경기장 안쪽 방향 x 부호 (BLUE 거점 +1, RED 거점 -1)."""
        return 1.0 if self.base_camp(value) == "BLUE" else -1.0

    def set_gone(self, gone):
        """navigation 이 뺀 상자의 번호들을 받는다. 전과 달라졌으면 True."""
        gone = frozenset(int(i) for i in gone if 0 <= int(i) < len(self.boxes))
        if gone == self.gone:
            return False
        self.gone = gone
        self._segs = None  # 가림 계산에 쓰던 것을 다시 만든다
        return True

    def box_faces(self, x, y, number, wall_gap_m=0.30):
        """(x, y) 에 상자가 하나 더 있다면 생길 면들 — 그 상자의 번호를 number 로 적는다. 지도는 건드리지 않는다.

        _make_faces 와 같은 규칙: 다른 상자에 맞닿은 면은 뺀다 (없다고 본 상자 gone 은 치지 않고, 고정 기둥에 맞닿은 면도 뺀다).
        더해서 면 앞에서 경기장 벽까지가 wall_gap_m 보다 좁은 면도 뺀다 — 벽에 붙은 쪽이라 그 앞에 설 자리가 없어 찍을 수 없다.
        (벽에 붙은 상자의 옆 면은 남는다: 가운데가 벽에서 0.20 m 이지만 면 앞은 트여 있어 옆에서 찍을 수 있다.)
        """
        size, tol = self.box_size, TOUCH_TOL
        half = size / 2.0
        others = [b for i, b in enumerate(self.boxes) if i not in self.gone and i != number] + list(self.pillars)
        out = []
        for side in SIDES:
            nx, ny = NORMAL[side]
            if any(abs(ox - (x + nx * size)) < tol and abs(oy - (y + ny * size)) < tol for ox, oy in others):
                continue
            room = {"E": self.width - (x + half), "W": x - half, "N": self.height - (y + half), "S": y - half}[side]
            if room < wall_gap_m:
                continue
            out.append(Face(number, side, x + nx * half, y + ny * half, nx, ny, self.face_sector(x, y, nx, ny)))
        return out

    def add_box(self, x, y, faces):
        """지도에 없던 상자를 맨 뒤에 넣는다 (navigation 이 라이다로 찾은 상자). 앞의 상자 번호는 바뀌지 않는다. 돌려주는 값: 새 상자의 번호.

        faces = box_faces(x, y, len(self.boxes)) 로 만든 면들. 시야 가림 계산에 쓰던 것은 다시 만든다.
        """
        self.boxes = list(self.boxes)
        self.boxes.append((float(x), float(y)))
        self.faces = list(self.faces) + list(faces)
        self._segs = None
        return len(self.boxes) - 1

    def segments(self):
        """[(x1, y1, x2, y2)] 벽 4개 + 박스 · 고정 기둥의 면 전부 (시야 가림 계산용). navigation 이 없다고 본 상자(gone)는 뺀다."""
        w, h = self.width, self.height
        segs = [(0, 0, w, 0), (w, 0, w, h), (w, h, 0, h), (0, h, 0, 0)]
        half = self.box_size / 2.0
        for x, y in [b for i, b in enumerate(self.boxes) if i not in self.gone] + list(self.pillars):
            segs += [(x + half, y - half, x + half, y + half), (x + half, y + half, x - half, y + half),
                     (x - half, y + half, x - half, y - half), (x - half, y - half, x + half, y - half)]
        return segs

    def face_near(self, x, y, tolerance=0.15):
        """(x, y) 에서 가장 가까운 면 (카메라가 본 점을 지도의 면에 붙일 때)."""
        best = None
        for face in self.faces:
            d = math.hypot(face.cx - x, face.cy - y)
            if d <= tolerance and (best is None or d < best[0]):
                best = (d, face)
        return None if best is None else best[1]

    def match_face(self, viewer, point, range_frac=0.25, max_incidence_deg=80.0, slack_deg=3.0):
        """viewer(카메라 자리)에서 본 면 하나를 지도의 면에 붙인다. point = 인식 결과(방위 · 거리)로 어림한 면 가운데.

        카메라 인식의 거리 어림(상자 크기로 잼)은 방위보다 훨씬 부정확하다 → 「point 에서 가장 가까운 면」 이 아니라
        viewer 에서 그 방위로 보이는 면을 고른다: 앞면이 viewer 를 향하고, 다른 박스에 다 가리지 않았고, 방위가 면의 폭 안이며,
        거리가 ±range_frac 안인 면 중 방위가 가장 맞는 것. 없으면 None.
        """
        vx, vy = viewer[0], viewer[1]
        bearing = math.atan2(point[1] - vy, point[0] - vx)
        rng = math.hypot(point[0] - vx, point[1] - vy)
        cands = []
        for face in self.faces:
            dx, dy = vx - face.cx, vy - face.cy
            d = math.hypot(dx, dy)
            if d < 0.05 or (face.nx * dx + face.ny * dy) / d < math.cos(math.radians(max_incidence_deg)):
                continue
            if abs(d - rng) > range_frac * max(d, 0.3):
                continue
            off = abs(math.atan2(math.sin(math.atan2(face.cy - vy, face.cx - vx) - bearing), math.cos(math.atan2(face.cy - vy, face.cx - vx) - bearing)))
            if off > math.atan2(self.box_size / 2.0, d) + math.radians(slack_deg):
                continue
            cands.append((off + 0.5 * abs(d - rng) / max(d, 0.3), face))
        for _cost, face in sorted(cands, key=lambda c: c[0]):
            if self._visible(vx, vy, face):
                return face
        return None

    def _visible(self, vx, vy, face):
        """face 의 가운데 · 양 끝 근처 중 한 곳이라도 viewer 에서 곧게 보이는가 (벽 · 다른 박스 면에 가리지 않음)."""
        import numpy as np

        from .coverage import _blocked_any

        if getattr(self, "_segs", None) is None:
            self._segs = np.array(self.segments(), dtype=np.float64)
        pts = [face.point(a) for a in (0.0, 0.12, -0.12)]
        qx = np.array([p[0] + face.nx * 0.01 for p in pts])
        qy = np.array([p[1] + face.ny * 0.01 for p in pts])
        blocked = _blocked_any(np.full(3, vx), np.full(3, vy), qx, qy, self._segs)
        return not bool(blocked.all())

    def mirror_face(self, face):
        """좌우 대칭 위치의 면 (지정표적은 양 진영의 대칭 위치에 놓인다 — P1 p.12). 지도에 없으면 None."""
        mx = self.width - face.cx
        side = {"E": "W", "W": "E"}.get(face.side, face.side)
        for other in self.faces:
            if other.side == side and abs(other.cx - mx) < 0.05 and abs(other.cy - face.cy) < 0.05:
                return other
        return None

    # ---- 만들기
    def _make_faces(self):
        """다른 박스에 맞닿아 가려진 면은 뺀다 (보이지 않으니 채점에도 없다).

        박스 위치는 드론 화면에서 잰 값이라 몇 cm 틀린다 (2026-11-18 시험 경기장 사진: 붙은 쌍의 가운데 사이가 0.37 ~ 0.42 m).
        격자가 20 cm 라 붙지 않은 상자는 가운데 사이가 0.60 m 이상이므로, TOUCH_TOL(8 cm) 안이면 붙은 것으로 본다.
        (처음의 3 cm 는 잰 위치에서는 붙은 쌍을 놓쳐, 두 상자 사이의 보이지 않는 면을 찍으러 가게 만들었다.)
        """
        out = []
        size = self.box_size
        tol = TOUCH_TOL
        for i, (x, y) in enumerate(self.boxes):
            for side in SIDES:
                nx, ny = NORMAL[side]
                if any(abs(ox - (x + nx * size)) < tol and abs(oy - (y + ny * size)) < tol for j, (ox, oy) in enumerate(self.boxes) if j != i):
                    continue
                cx, cy = x + nx * size / 2.0, y + ny * size / 2.0
                out.append(Face(i, side, cx, cy, nx, ny, self.face_sector(x, y, nx, ny)))
        return out
