"""ROS 메시지의 내용 ↔ 경기 서버 JSON. 보고값은 여기서만 만든다 (ROS 를 쓰지 않는다)."""
from __future__ import annotations

import re

from . import constants as C

SLOT = re.compile(r"^[BR]-[1-3]$")
_BASE_BY_LABEL = {label: kind for kind, label in C.BASE_KINDS.items()}


def clean_slot(value):
    """GET /me 의 slot (B-1 ~ B-3 · R-1 ~ R-3). 그 꼴이 아니면 None."""
    text = value.strip().upper() if isinstance(value, str) else ""
    return text if SLOT.match(text) else None


def clean_objective(order, camp):
    """GET /target 의 목표거점을 거점 보고값(RED_MAINTENANCE …)으로. 모르는 값이면 None.

    보고값이 문서와 다르게 오면 거점 이름(objective_label: 작전지휘소 · 정비소 · 군수보급소)으로 대신 찾는다 — 가야 할 곳은 상대 진영의 거점이다.
    """
    value = order.get("objective")
    text = value.strip().upper().replace(" ", "_") if isinstance(value, str) else ""
    if text in C.base_values():
        return text
    kind = _BASE_BY_LABEL.get(str(order.get("objective_label") or "").strip())
    return "%s_%s" % (C.opponent(camp), kind) if kind and camp in C.CAMPS else None


def m1_body(target, sectors, base):
    """빈 항목과 보고값이 아닌 항목은 뺀다 (빠진 항목은 0점이지만, 틀린 값을 보내면 422 로 보고 전체를 받지 않는다)."""
    body = {}
    target = str(target or "").strip().upper()
    if target in C.target_values():
        body[C.M1_TARGET] = target
    names = [s for s in C.SECTORS if s in {str(x).strip().upper() for x in sectors or []}]
    if names:
        body[C.M1_SECTORS] = names
    base = str(base or "").strip().upper()
    if base in C.base_values():
        body[C.M1_BASE] = base
    return body


def m3_body(sectors):
    """[(구역, {종류: 개수})] → 보고 본문. 받은 구역을 모두 보낸다 (통째로 교체된다). 모르는 구역 · 종류 · 0개는 뺀다."""
    out = []
    for name, objects in sectors:
        name = str(name).strip().upper()
        if name not in C.SECTORS:
            continue
        found = [{C.M3_TYPE: kind, C.M3_COUNT: int(objects[kind])} for kind in C.OBJECT_TYPES if int(objects.get(kind, 0)) > 0]
        out.append({C.M3_SECTOR: name, C.M3_OBJECTS: found})
    return {C.M3_SECTORS: out}


def m6_body(ranking, camp):
    """상대 호출부호 셋의 순서로 고친다 (모르는 값 · 중복은 버리고, 빠진 것은 뒤에 번호가 큰 것부터).
    빠진 것을 3 · 2 · 1 순으로 채우는 것은 supervisor 의 「가를 것이 없을 때」 와 같게 한 것이다 (2026-11-20 팀장 결정 — 전에는 1 · 2 · 3)."""
    p = C.opponent(camp)[0] if camp in C.CAMPS else None
    foes = ["%s-%d" % (p, n) for n in (1, 2, 3)] if p else []
    out = []
    for name in ranking or []:
        name = str(name).strip().upper()
        if (not foes or name in foes) and name not in out:
            out.append(name)
    return {C.M6_RANKING: out + [name for name in reversed(foes) if name not in out]}
