"""상자 찾기 결과(drone_boxes.BoxView) + 경계(drone_boundary.Boundary) → team_interfaces/FieldMap 형식.

/team/perception/field_map 으로 내보낼 내용이다. 칸 이름 · 뜻은 team_interfaces/msg/FieldMap.msg · FieldRegion · FieldFace · FieldVehicle 그대로.
  field_map(...)      ROS 없이 dict 로 (JSON 으로 저장 · 시험용). 칸 이름이 메시지와 같다
  field_map_msg(...)  위 dict → team_interfaces.msg.FieldMap (ROS 환경에서만 — team_interfaces · geometry_msgs 를 그때 import)

채우는 것
  header      frame_id "arena", stamp = 드론 사진을 찍은 시각 (부르는 쪽이 준다. 모르면 0)
  width · height · box_size   경계의 경기장 크기 · 0.40
  boxes       상자 가운데 (z = 0). 순서 = BoxView.boxes 순서 (위 줄부터, 같은 줄은 왼쪽부터) — faces[].box 가 이 번호를 쓴다
  sectors     B-S1~3 · CM · R-S1~3 (row 0). 경계가 잰 끝 띠 · CM 반폭과 줄 경계(H/3 · 2H/3)로 — 도면 추정값이 아니라 이 화면에서 잰 값
  faces       상자마다 E · N · W · S 면. 다른 상자와 맞닿은 면은 뺀다 (보이지 않음). image_ref 는 빈 문자열 (아직 사진을 붙이지 않음)
  beam_depth  0.20 (메시지 기본값, [추정])
  vehicles    비움 — 드론 사진의 UGV 찾기는 아직 없다 (navigation slam 은 SELF 가 있어야 출발 칸 · 위치를 맞춘다)
  bases       base_naming 으로 고른다 (아래). 기본은 비움

거점(bases) 이름 [팀 결정 필요]
  받는 쪽 두 곳이 이름을 다르게 읽는다 (2026-11-18 코드 기준).
    supervisor (arena.field_map_from_msg): 보고값 "BLUE_COMMAND_POST" 등으로 읽고, bases 가 비면 자기 도면 기본값을 쓴다
    navigation slam (_slot_candidate): 호출부호 "B-1" 등으로 찾는다
  base_naming=None    비움 (기본) — supervisor 는 도면 기본값을 쓰고, slam 은 어차피 vehicles 가 비어 출발 칸 맞추기를 하지 않는다
  base_naming="slot"  "B-1" ~ "R-3" (바닥에 적힌 글자, slam 방식)
  base_naming="report"  "<진영>_<거점 종류>" (supervisor 방식) — 줄마다 어느 거점인지 base_kinds 로 줘야 한다 (현장 확인 필요)
  거점 범위는 도면값이다 (벽에서 0.40 m, 줄 가운데 ±0.60 m — supervisor ArenaPlan 과 같은 [확인] 값). 화면에서 재지 않았다.
"""
from __future__ import annotations

SECTORS = ("B-S1", "B-S2", "B-S3", "CM", "R-S1", "R-S2", "R-S3")
SIDES = ("E", "N", "W", "S")  # sim-strategy fieldmap.py 와 같은 순서
NORMAL = {"E": (1.0, 0.0), "N": (0.0, 1.0), "W": (-1.0, 0.0), "S": (0.0, -1.0)}
BASE_KINDS = ("COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT")
BASE_DEPTH = 0.40  # 벽에서 안쪽으로 [확인: 400]
BASE_LENGTH = 1.20  # 벽을 따라 [확인: 1200]
BEAM_DEPTH = 0.20


def _row_range(height, row):
    """row 1 ~ 3 의 y 범위 (1번 줄이 위)."""
    h = height / 3.0
    return (3 - row) * h, (4 - row) * h


def sector_regions(width, height, strip, cm_half):
    """[(이름, x0, y0, x1, y1)] — 구역 7개."""
    cm0, cm1 = width / 2.0 - cm_half, width / 2.0 + cm_half
    out = []
    for name in SECTORS:
        if name == "CM":
            out.append((name, cm0, 0.0, cm1, height))
            continue
        y0, y1 = _row_range(height, int(name[-1]))
        x0, x1 = (strip, cm0) if name.startswith("B") else (cm1, width - strip)
        out.append((name, x0, y0, x1, y1))
    return out


def base_regions(width, height, naming, base_kinds=None):
    """[(이름, x0, y0, x1, y1, 줄)] — 거점 6개 (도면값). naming 은 "slot" · "report"."""
    out = []
    for camp, letter in (("BLUE", "B"), ("RED", "R")):
        for row in (1, 2, 3):
            y0, y1 = _row_range(height, row)
            yc = (y0 + y1) / 2.0
            x0, x1 = (0.0, BASE_DEPTH) if camp == "BLUE" else (width - BASE_DEPTH, width)
            if naming == "slot":
                name = "%s-%d" % (letter, row)
            else:
                kinds = (base_kinds or {}).get(camp)
                if not kinds or sorted(kinds) != sorted(BASE_KINDS):
                    raise ValueError("base_naming='report' 에는 base_kinds[%s] 가 %s 를 한 번씩 담아야 한다" % (camp, list(BASE_KINDS)))
                name = "%s_%s" % (camp, kinds[row - 1])
            out.append((name, x0, yc - BASE_LENGTH / 2.0, x1, yc + BASE_LENGTH / 2.0, row))
    return out


def _sector_at(sectors, x, y):
    for name, x0, y0, x1, y1 in sectors:
        if x0 <= x < x1 and y0 <= y < y1:
            return name
    return ""


def box_faces(boxes, sectors, size=0.40, touch_tol=0.06):
    """[(상자 번호, 면, cx, cy, nx, ny, 구역)] — 다른 상자와 맞닿은 면은 뺀다.

    touch_tol: 두 상자 가운데 사이가 size ± 이 값이고 옆으로 이 값 안이면 맞닿은 것으로 본다. 드론 사진에서 잰 상자 위치는
    몇 cm 틀리므로 (연습 화면에서 붙은 쌍 간격 0.37 ~ 0.42 m) sim-strategy fieldmap.py 의 3 cm 보다 넓게 둔다.
    """
    out = []
    for i, (x, y) in enumerate(boxes):
        for side in SIDES:
            nx, ny = NORMAL[side]
            touching = any(abs(ox - (x + nx * size)) < touch_tol and abs(oy - (y + ny * size)) < touch_tol for j, (ox, oy) in enumerate(boxes) if j != i)
            if touching:
                continue
            out.append((i, side, x + nx * size / 2.0, y + ny * size / 2.0, nx, ny, _sector_at(sectors, x, y)))
    return out


def field_map(view, boundary, stamp=None, base_naming=None, base_kinds=None, size=0.40, touch_tol=0.06, arena=None):
    """BoxView + Boundary → team_interfaces/FieldMap 과 같은 칸의 dict. stamp: 드론 사진을 찍은 시각 (초, epoch). 모르면 None.

    arena: (가로, 세로) m — 지도에 적을 벽에서 벽까지의 크기. None 이면 경계를 맞출 때 쓴 크기(6.12 × 5.00) 그대로 (지금까지와 같다).
      주면 경기장 정중앙을 그대로 두고 벽만 그 크기로 옮긴다: 상자는 「정중앙에서 몇 m」 를 그대로 지키고(좌표는 새 왼쪽 아래 모서리 기준),
      구역선(끝 띠 · CM)은 바닥 지도에서 잰 값이라 가로 비율만큼 줄이고, 줄 경계 · 거점은 새 세로를 셋으로 나눈다.
      왜 필요한가 (2026-11-19 2경기장 실차 두 판): 상자 자리는 정중앙 기준으로 2 ~ 4 cm 안에서 맞았는데, 라이다는 벽을 지도보다
      가로 6 cm · 세로 10 cm 씩 안쪽에서 봤다 (아래 벽이 y ≈ 0.13 에, 위쪽 가운데 기둥 면이 y ≈ 4.5 에 보임). 포털의 경기장 모형 600 × 480 과 같다.
      navigation 은 벽을 width · height 로 그리고 라이다를 벽에 맞추므로, 벽 가까이의 상자가 12 ~ 14 cm 어긋나 보였다.
    """
    w0, h0 = float(boundary.field_w), float(boundary.field_h)
    w, h = (float(arena[0]), float(arena[1])) if arena else (w0, h0)
    dx, dy, kx = (w - w0) / 2.0, (h - h0) / 2.0, w / w0
    boxes = [(float(b.x) + dx, float(b.y) + dy) for b in view.boxes]
    sectors = sector_regions(w, h, float(boundary.strip) * kx, float(boundary.cm_half) * kx) if boundary.strip is not None else []
    bases = base_regions(w, h, base_naming, base_kinds) if base_naming else []
    faces = box_faces(boxes, sectors, size, touch_tol)
    return {
        "header": {"frame_id": "arena", "stamp": stamp},
        "width": w,
        "height": h,
        "box_size": size,
        "boxes": [{"x": round(x, 4), "y": round(y, 4), "z": 0.0} for x, y in boxes],
        "bases": [{"name": n, "min_x": round(x0, 4), "min_y": round(y0, 4), "max_x": round(x1, 4), "max_y": round(y1, 4), "row": r}
                  for n, x0, y0, x1, y1, r in bases],
        "sectors": [{"name": n, "min_x": round(x0, 4), "min_y": round(y0, 4), "max_x": round(x1, 4), "max_y": round(y1, 4), "row": 0}
                    for n, x0, y0, x1, y1 in sectors],
        "beam_depth": BEAM_DEPTH,
        "faces": [{"box": i, "side": s, "cx": round(cx, 4), "cy": round(cy, 4), "nx": nx, "ny": ny, "sector": sec, "image_ref": ""}
                  for i, s, cx, cy, nx, ny, sec in faces],
        "vehicles": [],
    }


def bare_field_map(stamp=None, base_naming=None, base_kinds=None, width=6.12, height=5.00, strip=0.88, cm_half=0.60, size=0.40):
    """상자가 하나도 없는 지도 (도면값의 벽 · 거점 · 구역만). 드론 사진으로 지도를 만들지 못할 때의 임시 지도다 (map_maker 의 map_fallback_s).

    이것이라도 있으면 navigation 은 벽(과 고정 기둥)에 라이다를 맞춰 위치를 잡고, 상자는 라이다에 보이는 대로 피해 다닌다.
    정찰(M3)은 못 하지만 거점 진입(M5)은 할 수 있다. 값은 연습 경기장에서 잰 것: 벽에서 벽까지 6.12 × 5.00 m, 끝 띠 0.88 m, 가운데 구역 반폭 0.60 m.
    """
    sectors = sector_regions(width, height, strip, cm_half)
    bases = base_regions(width, height, base_naming, base_kinds) if base_naming else []
    return {
        "header": {"frame_id": "arena", "stamp": stamp},
        "width": float(width), "height": float(height), "box_size": size,
        "boxes": [],
        "bases": [{"name": n, "min_x": round(x0, 4), "min_y": round(y0, 4), "max_x": round(x1, 4), "max_y": round(y1, 4), "row": r}
                  for n, x0, y0, x1, y1, r in bases],
        "sectors": [{"name": n, "min_x": round(x0, 4), "min_y": round(y0, 4), "max_x": round(x1, 4), "max_y": round(y1, 4), "row": 0}
                    for n, x0, y0, x1, y1 in sectors],
        "beam_depth": BEAM_DEPTH, "faces": [], "vehicles": [],
    }


def field_map_msg(d):
    """field_map() 의 dict → team_interfaces.msg.FieldMap. ROS 2 환경(team_interfaces 빌드)에서만 부른다."""
    from geometry_msgs.msg import Point
    from team_interfaces.msg import FieldFace, FieldMap, FieldRegion

    m = FieldMap()
    m.header.frame_id = d["header"]["frame_id"]
    stamp = d["header"].get("stamp")
    if stamp:
        sec = int(stamp)
        m.header.stamp.sec, m.header.stamp.nanosec = sec, int(round((stamp - sec) * 1e9)) % 1_000_000_000
    m.width, m.height, m.box_size, m.beam_depth = float(d["width"]), float(d["height"]), float(d["box_size"]), float(d["beam_depth"])
    m.boxes = [Point(x=float(b["x"]), y=float(b["y"]), z=0.0) for b in d["boxes"]]

    def region(r):
        return FieldRegion(name=r["name"], min_x=float(r["min_x"]), min_y=float(r["min_y"]), max_x=float(r["max_x"]), max_y=float(r["max_y"]), row=int(r["row"]))

    m.bases = [region(r) for r in d["bases"]]
    m.sectors = [region(r) for r in d["sectors"]]
    m.faces = [FieldFace(box=int(f["box"]), side=f["side"], cx=float(f["cx"]), cy=float(f["cy"]), nx=float(f["nx"]), ny=float(f["ny"]),
                         sector=f["sector"], image_ref=f["image_ref"]) for f in d["faces"]]
    m.vehicles = []
    return m
