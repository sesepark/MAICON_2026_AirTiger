"""경기장 도면(벽 · 거점 · 구역)과 보고값 이름. 상자 위치는 여기 없다 — navigation 이 주는 지도로 채운다.

좌표: x 는 BLUE 쪽 벽(0) → RED 쪽 벽, y 는 아래(0) → 위, 1번 줄이 위쪽. 단위 m (fieldmap.py 와 같다).
보고값 글자는 공식 자료(경기서버연동API 2026-11-04)의 것이다.

치수는 두 가지가 있다 (2026-11-18 에 둘을 견줘 봄 — docs/포털에서_확인한_것.md 4절).
  [포털] 참가팀 포털의 경기장 그림(서버의 모형): 600 x 480 cm, 격자 20 cm, 끝 띠 80, CM 240~360, 세로 160 씩, 거점 40 x 80.
  [실측] perception 이 시험 경기장의 드론 사진에서 잰 값: 벽에서 벽까지 약 612 x 500 cm 로 놓아야 축척이 맞는다
         (상자 간격 0.40 m · CM 반폭 0.60 m 가 그대로 나온다). 끝 띠 선은 벽에서 0.88 m, 가로 줄 경계는 높이를 셋으로 나눈 곳.
위치 · 지도가 [실측] 좌표로 오므로 기본값은 [실측] 이다. 지도 메시지에 구역 · 거점이 들어 있으면 그것(그 사진에서 잰 바닥 선)을 쓴다.
거점은 위부터 작전지휘소 · 정비소 · 군수보급소 (포털), 각 UGV 는 자기 진영의 같은 번호 거점 안에서 상대 진영을 보고 출발한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields

from .fieldmap import FieldMap

CAMPS = ("BLUE", "RED")
SECTORS = ("B-S1", "B-S2", "B-S3", "CM", "R-S1", "R-S2", "R-S3")

# perception 이 답하는 종류 (team_interfaces/msg/Picture.msg 와 같은 글자)
OBJECT_KINDS = ("military_truck", "mortar", "tank", "missile", "ammo_box", "drum")
TARGET_KINDS = ("comms", "outpost", "radar")
NON_OBJECT = "non_object"
UNKNOWN = "unknown"
ALL_KINDS = OBJECT_KINDS + TARGET_KINDS + (NON_OBJECT, UNKNOWN)

# 거점 종류 (보고값의 뒷부분). 어느 줄에 어느 거점이 있는지는 설정 base_kinds 로 준다
BASE_KINDS = ("COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT")

RUNNING_S = 300.0  # 경기 시간
TARGET_AT_S = 240.0  # 목표거점 게시 시각


def opponent(camp):
    return "RED" if camp == "BLUE" else "BLUE"


def camp_of(slot):
    """호출부호(B-1 · R-3 …) → 진영."""
    return "BLUE" if slot.upper().startswith("B") else "RED"


def row_of(slot):
    return int(slot[-1])


def scored_sectors(camp):
    """M3 채점 구역: 상대 진영 3구역 + CM."""
    p = opponent(camp)[0]
    return ("%s-S1" % p, "%s-S2" % p, "%s-S3" % p, "CM")


def foe_callsigns(camp):
    p = opponent(camp)[0]
    return ("%s-1" % p, "%s-2" % p, "%s-3" % p)


def target_kind(value):
    """지정표적 보고값(RED_RADAR) → (진영, perception 종류). 모르는 값이면 None."""
    if not value or "_" not in value:
        return None
    camp, kind = value.split("_", 1)
    kind = kind.lower()
    if camp not in CAMPS or kind not in TARGET_KINDS:
        return None
    return camp, kind


@dataclass
class ArenaPlan:
    width: float = 6.12  # x 방향, 벽에서 벽까지 [실측 · 사전 자료 약 612 cm. 포털의 모형은 600]
    height: float = 5.00  # y 방향 [실측 · 사전 자료 약 500 cm. 포털의 모형은 480]
    box: float = 0.40
    base_w: float = 1.20  # 거점: 벽을 따라 [사전 자료 1200. 포털의 모형은 80 cm]
    base_d: float = 0.40  # 거점: 벽에서 안쪽으로 [사전 자료 400 · 포털 40 cm]
    end_strip: float = 0.88  # 벽에서 끝 띠 선까지 — 그 바깥은 어느 구역도 아니다 [실측 0.88. 포털의 모형은 80 cm]
    cm_width: float = 1.20  # 공통 구역 폭 [실측 반폭 0.60 · 포털 240~360 cm]
    beam_depth: float = 0.20  # 거점 센서 빔이 벽에서 떨어진 거리 [추정]
    pillars: bool = True  # 벽에 붙은 고정 기둥 10개(노랑 · 검정 블록)를 막힌 곳으로 넣는다 (fixed_pillars). 본선 경기장의 배치가 다르면 false

    def row_range(self, row):
        """row 1~3 의 y 범위 (1번 줄이 위)."""
        h = self.height / 3.0
        return (3 - row) * h, (4 - row) * h

    def sector_rect(self, name):
        cm0 = (self.width - self.cm_width) / 2.0
        cm1 = cm0 + self.cm_width
        if name == "CM":
            return cm0, 0.0, cm1, self.height
        y0, y1 = self.row_range(int(name[-1]))
        if name.startswith("B"):
            return self.end_strip, y0, cm0, y1
        return cm1, y0, self.width - self.end_strip, y1

    def base_rect(self, camp, row):
        y0, y1 = self.row_range(row)
        yc = (y0 + y1) / 2.0
        if camp == "BLUE":
            return 0.0, yc - self.base_w / 2.0, self.base_d, yc + self.base_w / 2.0
        return self.width - self.base_d, yc - self.base_w / 2.0, self.width, yc + self.base_w / 2.0


DEFAULT_BASE_KINDS = {camp: list(BASE_KINDS) for camp in CAMPS}  # [포털] 위부터 A · B · C = 작전지휘소 · 정비소 · 군수보급소 (두 진영 같다)


def arena_from(overrides):
    """설정의 "arena" 에서 ArenaPlan 에 있는 이름만 골라 덮는다."""
    known = {f.name for f in fields(ArenaPlan)}
    return ArenaPlan(**{k: (bool(v) if k == "pillars" else float(v)) for k, v in (overrides or {}).items() if k in known})


PILLAR_ARENA = ((5.5, 6.7), (4.5, 5.5))  # 기둥의 기본 배치를 쓰는 경기장 크기 (가로 · 세로 범위, m) — 공식 크기 근처에서만


def fixed_pillars(width, height, size=0.40, boxes=()):
    """고정 기둥 10개의 가운데 [(x, y)] — 한 변 size 의 정사각형으로 본다. navigation 의 drive/fieldmap.py fixed_pillars 와 같은 자리여야 한다.

    양쪽 벽의 줄 경계(높이의 1/3 · 2/3) 네 곳, 위아래 벽 가운데 두 곳, 네 모서리(벽에 반쯤 걸친 정사각형으로 적는다).
    연습 경기장 1 · 2번의 드론 사진으로 확인했다 (2026-11-19). 지도(perception)는 벽 쪽 끝 띠 안을 내지 않아 기둥이 없다 →
    모르고 정찰 자리를 기둥 바로 옆에 잡으면 그 자리에 가지 못한다 (3차 팀 테스트의 (2.62, 4.62): 위쪽 가운데 기둥에서 0.24 m).
    공식 크기가 아닌 경기장이면 빈 목록. boxes 를 주면 그 자리에 이미 상자가 있는 기둥은 뺀다.
    """
    (w0, w1), (h0, h1) = PILLAR_ARENA
    if not (w0 <= width <= w1 and h0 <= height <= h1):
        return []
    w, h, a = float(width), float(height), size / 2.0
    spots = [(a, h * 2.0 / 3.0), (a, h / 3.0), (w - a, h * 2.0 / 3.0), (w - a, h / 3.0), (w / 2.0, h - a), (w / 2.0, a),
             (a, 0.0), (a, h), (w - a, 0.0), (w - a, h)]
    return [p for p in spots if not any(abs(p[0] - bx) < a and abs(p[1] - by) < a for bx, by in boxes)]


def field_map(boxes, arena=None, base_kinds=None):
    """상자 가운데 목록 [(x, y)] 로 FieldMap 을 만든다 (면 목록 · 구역 · 거점 포함)."""
    arena = arena or ArenaPlan()
    base_kinds = base_kinds or DEFAULT_BASE_KINDS
    bases, rows = {}, {}
    for camp in CAMPS:
        kinds = base_kinds[camp]
        if sorted(kinds) != sorted(BASE_KINDS):
            raise ValueError("base_kinds[%s] 는 %s 를 한 번씩 담아야 한다" % (camp, sorted(BASE_KINDS)))
        for row in (1, 2, 3):
            value = "%s_%s" % (camp, kinds[row - 1])
            bases[value] = arena.base_rect(camp, row)
            rows[value] = row
    sectors = {name: arena.sector_rect(name) for name in SECTORS}
    boxes = [(float(x), float(y)) for x, y in boxes]
    pillars = fixed_pillars(arena.width, arena.height, arena.box, boxes) if arena.pillars else []
    return FieldMap(arena.width, arena.height, arena.box, boxes, bases, rows, sectors, beam_depth=arena.beam_depth, pillars=pillars)


_SLOT_NAME = re.compile(r"^([BR])-?([1-3])$")


def base_key(name, base_kinds=None):
    """지도의 거점 이름 → 보고값. 보고값(RED_MAINTENANCE)이면 그대로, 자리 이름(B-2)이면 그 진영 2번 줄의 거점. 모르는 이름이면 None.

    navigation 은 지도의 거점을 자리 이름(B-1 …)으로 찾고(출발 칸), supervisor 는 보고값으로 찾는다 — 같은 네모를 가리키므로 둘 다 받는다.
    """
    text = (name or "").strip().upper().replace(" ", "_")
    if text in {"%s_%s" % (c, k) for c in CAMPS for k in BASE_KINDS}:
        return text
    hit = _SLOT_NAME.match(text)
    if hit:
        camp = "BLUE" if hit.group(1) == "B" else "RED"
        return "%s_%s" % (camp, (base_kinds or DEFAULT_BASE_KINDS)[camp][int(hit.group(2)) - 1])
    return None


def field_map_from_msg(m, arena=None, base_kinds=None):
    """team_interfaces/msg/FieldMap → FieldMap. 면 목록은 상자에서 다시 만든다.

    거점 · 구역은 메시지에 든 것 가운데 이름을 아는 것만 쓰고, 빠진 것은 도면 기본값으로 채운다
    (게시된 목표거점이 지도에 없어서 멈추는 일이 없게). 거점 이름은 보고값이어도, 자리 이름(B-1 …)이어도 된다.
    """
    base = field_map([], arena, base_kinds)
    width, height = m.width or base.width, m.height or base.height
    bases, rows = dict(base.bases), dict(base.base_rows)
    told = {camp: set() for camp in CAMPS}  # 보고값 이름으로 온 거점 종류 (자리 이름 B-1 … 로 온 것은 기본 배치로 읽은 것이라 세지 않는다)
    for r in m.bases:
        key = base_key(r.name, base_kinds)
        if key is None or not (r.max_x > r.min_x and r.max_y > r.min_y):
            continue
        if _SLOT_NAME.match((r.name or "").strip().upper().replace(" ", "_")) is None:
            told[key.split("_", 1)[0]].add(key)
        bases[key] = (r.min_x, r.min_y, r.max_x, r.max_y)
        third = height / 3.0
        rows[key] = int(r.row) if 1 <= int(r.row) <= 3 else max(1, min(3, 3 - int(((r.min_y + r.max_y) / 2.0) // third)))  # 줄 번호가 비어 있으면 높이로 정한다 (1번 줄이 위)
    sectors = dict(base.sectors)
    for r in m.sectors:
        name = (r.name or "").strip().upper()
        if name in SECTORS and r.max_x > r.min_x and r.max_y > r.min_y:
            sectors[name] = (r.min_x, r.min_y, r.max_x, r.max_y)
    boxes = [(float(p.x), float(p.y)) for p in m.boxes]
    size = m.box_size or base.box_size
    pillars = fixed_pillars(width, height, size, boxes) if (arena or ArenaPlan()).pillars else []
    kinds_read = frozenset(camp for camp, values in told.items() if len(values) == len(BASE_KINDS))
    return FieldMap(width, height, size, boxes, bases, rows, sectors, beam_depth=m.beam_depth or base.beam_depth, pillars=pillars, kinds_read=kinds_read)


def side_of(fmap, x):
    """x 좌표가 어느 진영 쪽인가 (지정표적이 누구 것인지 정할 때)."""
    return "BLUE" if x < fmap.width / 2.0 else "RED"
