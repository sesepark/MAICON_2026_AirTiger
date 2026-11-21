"""경기 서버 API 의 경로 · 단계 · 보고값 (경기서버연동API 2026-11-04 기준). OT 최종본에서 바뀌면 이 파일만 고친다."""
from __future__ import annotations

# ---- 창구. 본 경기는 포트 8000, 참가팀 포털의 팀 테스트는 같은 경로의 시험 포트 8100 (포털 안내 2026-11-18:
#      「테스트 시에는 8100 포트 · 본 경기 시에는 8000 포트」). 예전의 8000 /api/rehearsal/ 은 없어졌다 (404)
PREFIX_REPORT = "/api/report/"
TEST_PORT = 8100
LIVE_PHASES = ("waiting", "ready", "running", "reporting")  # 우리 팀의 경기(시험)가 열려 있는 단계

EP_ME = "me"
EP_READY = "ready"
EP_ORDER = "order"
EP_TARGET = "target"
EP_DRONE = "drone"
EP_M1 = "interpretation"
EP_M3 = "detection"
EP_M6 = "situation"

# ---- 단계 (GET /me 의 phase)
PHASE_IDLE = "idle"
PHASE_WAITING = "waiting"
PHASE_READY = "ready"
PHASE_RUNNING = "running"
PHASE_REPORTING = "reporting"
PHASE_FINISHED = "finished"

CAMPS = ("BLUE", "RED")
SECTORS = ("B-S1", "B-S2", "B-S3", "CM", "R-S1", "R-S2", "R-S3")
OBJECT_TYPES = ("military_truck", "mortar", "tank", "missile", "ammo_box", "drum")
BASE_KINDS = {"COMMAND_POST": "작전지휘소", "MAINTENANCE": "정비소", "SUPPLY_DEPOT": "군수보급소"}

# ---- 보고 JSON 의 필드 이름
M1_TARGET = "target"
M1_SECTORS = "search_sectors"
M1_BASE = "base"
M3_SECTORS = "sectors"
M3_SECTOR = "sector"
M3_OBJECTS = "objects"
M3_TYPE = "type"
M3_COUNT = "count"
M6_RANKING = "ranking"


def opponent(camp):
    return "RED" if camp == "BLUE" else "BLUE"


def camp_of(slot):
    return "BLUE" if slot.upper().startswith("B") else "RED"


TARGET_KINDS = ("COMMS", "OUTPOST", "RADAR")


def base_values():
    return tuple("%s_%s" % (c, k) for c in CAMPS for k in BASE_KINDS)


def target_values():
    return tuple("%s_%s" % (c, k) for c in CAMPS for k in TARGET_KINDS)
