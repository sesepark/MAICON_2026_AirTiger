"""경기장 기본값 — 실측 612 × 500 cm (2026-11-18 팀 결정: 포털의 600 × 480 은 서버의 그림 모형. supervisor ArenaPlan · perception 과 같은 값).

좌표: x 는 BLUE 벽(0) → RED 벽, y 는 아래(0) → 위, 단위 m.
출발 칸(거점)은 perception drone_field_map.base_regions 와 같은 도면값: 벽에서 안쪽 0.40 m, 줄(높이의 1/3) 가운데에서 ±0.60 m. 1번 줄이 위.
고정 기둥 10개의 자리는 [추정] 이다 (모서리 4 · 좌우 벽의 줄 경계 4 · 위아래 벽 가운데 2). 드론 사진에서는 끝 띠 안이라 잡히지 않는다.
"""
import math

ARENA = dict(width=6.12, height=5.0, box_size=0.4)
SLOTS = ('B-1', 'B-2', 'B-3', 'R-1', 'R-2', 'R-3')
BASE_DEPTH = 0.40
BASE_LENGTH = 1.20


def pillars(width=ARENA['width'], height=ARENA['height']):
    """고정 기둥 10개 [추정]. 모서리 기둥(40 × 20 cm)은 벽에 반쯤 걸친 40 cm 상자로 적는다 (경기장 안쪽 면의 위치는 같다)."""
    w, h = width, height
    return ((0.2, 0.0), (0.2, h), (w - 0.2, 0.0), (w - 0.2, h), (0.2, h * 2 / 3), (0.2, h / 3), (w - 0.2, h * 2 / 3), (w - 0.2, h / 3),
            (w / 2, h - 0.2), (w / 2, 0.2))


PILLARS = pillars()


def base_region(slot, width=ARENA['width'], height=ARENA['height']):
    """거점(출발 칸) (min_x, min_y, max_x, max_y)."""
    side, row = slot.split('-')
    centre = height * (3 - int(row) + 0.5) / 3
    x0 = 0.0 if side == 'B' else width - BASE_DEPTH
    return (x0, centre - BASE_LENGTH / 2, x0 + BASE_DEPTH, centre + BASE_LENGTH / 2)


def slot_start_pose(slot):
    """출발 자세: 거점 한가운데에서 상대 진영을 본다."""
    x0, y0, x1, y1 = base_region(slot)
    return ((x0 + x1) / 2, (y0 + y1) / 2, 0.0 if slot.startswith('B') else math.pi)


def bases(width=ARENA['width'], height=ARENA['height']):
    return [dict(zip(('name', 'min_x', 'min_y', 'max_x', 'max_y'), (slot, *base_region(slot, width, height)))) for slot in SLOTS]


def default_config(width=ARENA['width'], height=ARENA['height']):
    return {
        'width': width, 'height': height, 'box_size': ARENA['box_size'],
        'boxes': [list(p) for p in pillars(width, height)],
        'bases': bases(width, height),
    }
