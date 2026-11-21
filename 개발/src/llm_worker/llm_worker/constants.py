"""보고값 이름 (경기서버연동API 2026-11-04 기준). OT 최종본에서 바뀌면 이 파일만 고친다."""
from __future__ import annotations

CAMPS = ("BLUE", "RED")
SECTORS = ("B-S1", "B-S2", "B-S3", "CM", "R-S1", "R-S2", "R-S3")
TARGET_KINDS = {"COMMS": "통신시설", "OUTPOST": "감시초소", "RADAR": "레이더시설"}
BASE_KINDS = {"COMMAND_POST": "작전지휘소", "MAINTENANCE": "정비소", "SUPPLY_DEPOT": "군수보급소"}

# M1 보고 JSON 의 필드 이름
M1_TARGET = "target"
M1_SECTORS = "search_sectors"
M1_BASE = "base"


def opponent(camp):
    return "RED" if camp == "BLUE" else "BLUE"


def facility(camp, kind):
    return "%s_%s" % (camp, kind)


def camp_sectors(camp):
    p = camp[0]
    return ("%s-S1" % p, "%s-S2" % p, "%s-S3" % p)


def foe_callsigns(camp):
    p = opponent(camp)[0]
    return ("%s-1" % p, "%s-2" % p, "%s-3" % p)
