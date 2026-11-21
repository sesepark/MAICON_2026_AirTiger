"""M1 낱말 찾기 해석기 — LLM 없이 명령문에서 지정표적 · 탐색 섹터 · 목표거점을 뽑는다.

팀 결정: 이 해석기로 먼저 답을 내고, LLM 결과가 오면 덮어쓴다. LLM 이 늦거나 막혀도 M1 점수 일부를 지킨다.
멘토링 안내로는 문장 표현은 매번 달라도 지정표적 같은 핵심 용어는 바뀌지 않는다 → 낱말을 찾고, 그 낱말이 부정되는 문맥인지만 본다.

다루는 것 — 두 가지 문체
  (가) 시설 이름을 그대로 적은 문체
  - 「X 가 아니라 T」, 「X 는 제외하며 … T」, 「X 타격은 타 팀이 맡는다」, 「Y 방면은 우회하고 B」 처럼 다른 시설을 부정문으로 함께 적은 경우
  - 섹터 이름을 그대로 적은 경우 (R-S1, B-S3, CM, 공통 구역)
  - 「전 구역」, 「R-S2 를 제외한 나머지 구역」, 「상대 중앙진지를 제외하고 나머지」 (중앙진지 = 상대 2번 줄로 가정 [추정])
  (나) 시설을 기능으로 돌려 말한 문체 — 참가팀 포털의 팀 테스트 기본 명령문(2026-11-18 확인)이 이 문체다
  - 「상대 1번 진지와 중앙 구역을 정찰하고 … 적 부대 간 교신을 중계하는 시설을 타격하고, 최종적으로 작전을 지휘·통제하는 거점으로 이동하라」
    → 「N번 진지」 = 그 진영의 S{N}, 「중앙 구역」 = CM, 「…시설」 앞의 설명 = 지정표적, 「…거점」 앞의 설명 = 목표거점
  - 이름이 그대로 있으면 (가)로 풀고, 없을 때만 (나)의 단서 낱말로 푼다
모르는 것은 None · 빈 목록으로 둔다 (지어내지 않는다).

출처: 팀의 앞선 저장소(maicon-simple)의 team/m1_command/parser.py 를 옮긴 것 (팀이 직접 쓴 코드).
"""
from __future__ import annotations

import re

from . import constants as C

# 긴 낱말을 먼저 찾아야 「레이더시설」 안의 「레이더」 를 두 번 세지 않는다
TARGET_WORDS = (
    ("통신시설", "COMMS"),
    ("통신 시설", "COMMS"),
    ("감시초소", "OUTPOST"),
    ("감시 초소", "OUTPOST"),
    ("레이더시설", "RADAR"),
    ("레이더 시설", "RADAR"),
    ("레이다시설", "RADAR"),
    ("레이더", "RADAR"),
    ("레이다", "RADAR"),
)
BASE_WORDS = (
    ("작전지휘소", "COMMAND_POST"),
    ("작전 지휘소", "COMMAND_POST"),
    ("지휘소", "COMMAND_POST"),
    ("군수보급소", "SUPPLY_DEPOT"),
    ("군수 보급소", "SUPPLY_DEPOT"),
    ("보급소", "SUPPLY_DEPOT"),
    ("정비소", "MAINTENANCE"),
)
NEGATIVE = ("제외", "아니라", "아닌", "아니다", "아니며", "아니고", "타 팀", "타팀", "다른 팀", "말고", "우회", "빼고", "않는다", "않으며", "금지", "하지 마", "하지 말")
EXCLUDE = ("제외", "빼고", "말고", "뺀")
ALL_WORDS = ("전 구역", "전구역", "전체 구역", "모든 구역", "전 섹터", "모든 섹터", "전체 섹터")
CM_WORDS = ("공통 구역", "공통구역", "공통 섹터", "중앙 공통", "중앙 구역", "중앙구역", "중앙 지역", "중앙지역", "중앙 섹터", "가운데 구역")
CENTRAL_WORDS = ("중앙진지", "중앙 진지")
OWN_WORDS = ("아군 진영", "우리 진영", "아군진영", "우리진영", "자 진영")

_SECTOR = re.compile(r"(?<![A-Z0-9])([BR])\s*[-‐‑–—−]?\s*S\s*([1-3])(?![0-9])")
_CM = re.compile(r"(?<![A-Z0-9])CM(?![A-Z0-9])")
_STOP = re.compile(r"[.\n。;]|,|，")
# 「상대 1번 진지」 「적 2번 구역」 「아군 3번 진지」 — 진영 말이 없으면 상대 진영으로 본다
# 번호는 여럿일 수 있다: 「상대 1번·3번 진지」 「적 1, 2번 구역」 → 두 번째 묶음에서 숫자를 모두 꺼낸다
_NUMBERED = re.compile(r"(상대|적군|적|아군|우군|우리|자)?\s*(?:진영\s*)?(?:제\s*)?((?:[1-3]\s*(?:번)?\s*[·ㆍ・,，/와과및]\s*(?:제\s*)?)*[1-3])\s*번\s*(?:째\s*)?(진지|구역|섹터|줄|지역)")
_OWN_PREFIX = ("아군", "우군", "우리", "자")

# (나) 기능으로 돌려 말한 문체의 단서 낱말. 앞의 것일수록 뚜렷하다 (점수 = 맞은 단서의 무게 합)
TARGET_CUES = {
    "COMMS": (("교신", 3), ("중계", 3), ("통신", 3), ("연락", 1), ("무선", 2)),
    "OUTPOST": (("초소", 3), ("눈으로", 3), ("육안", 3), ("상주", 2), ("경계", 2), ("관측", 2), ("보초", 2), ("망을 보", 2)),
    "RADAR": (("레이더", 3), ("레이다", 3), ("감시체계", 3), ("감시 체계", 3), ("감시망", 3), ("탐지", 3), ("전파", 2), ("추적", 2), ("체계의 핵심", 2)),
}
BASE_CUES = {
    "COMMAND_POST": (("지휘", 3), ("통제", 2), ("사령", 2), ("본부", 1)),
    "MAINTENANCE": (("정비", 3), ("수리", 3), ("고치", 2), ("손보", 2), ("손볼", 2), ("복구", 1)),
    "SUPPLY_DEPOT": (("군수", 3), ("보급", 3), ("물자", 2), ("탄약", 1), ("연료", 1), ("지원", 1)),
}
STRIKE_WORDS = ("타격", "파괴", "제압", "무력화", "공격", "격파", "제거", "무력", "사격")
MOVE_WORDS = ("이동", "집결", "진입", "확보", "점령", "향하", "도달", "복귀", "전개", "진출")


def _mentions(text, table):
    """[(위치, 끝, 종류)] — 겹치는 낱말은 긴 것만."""
    found = []
    taken = [False] * len(text)
    for word, kind in sorted(table, key=lambda item: -len(item[0])):
        start = 0
        while True:
            at = text.find(word, start)
            if at < 0:
                break
            end = at + len(word)
            if not any(taken[at:end]):
                found.append((at, end, kind))
                for i in range(at, end):
                    taken[i] = True
            start = end
    return sorted(found)


def _window(text, end, stops):
    """낱말 뒤 문맥: 다음 시설 낱말 · 섹터 이름 · 쉼표 · 문장 끝 중 가장 먼저 오는 곳까지."""
    limit = len(text)
    hit = _STOP.search(text, end)
    if hit:
        limit = min(limit, hit.start())
    for stop in stops:
        if stop >= end:
            limit = min(limit, stop)
            break
    return text[end:limit]


def _pick(text, table, others, sector_starts):
    """부정되지 않은 마지막 낱말의 종류. 없으면 None."""
    mentions = _mentions(text, table)
    if not mentions:
        return None
    stops = sorted([m[0] for m in mentions] + [m[0] for m in others] + sector_starts)
    positive = []
    for at, end, kind in mentions:
        later = [s for s in stops if s > at]
        if not any(cue in _window(text, end, later) for cue in NEGATIVE):
            positive.append(kind)
    if positive:
        return positive[-1]
    return None


def _sectors(text, camp):
    upper = text.upper()
    foe = C.opponent(camp)
    foe_sectors = list(C.camp_sectors(foe))
    named, excluded = [], set()
    spans = []
    for hit in _SECTOR.finditer(upper):
        spans.append((hit.start(), hit.end(), "%s-S%s" % (hit.group(1), hit.group(2))))
    for hit in _CM.finditer(upper):
        spans.append((hit.start(), hit.end(), "CM"))
    for word in CM_WORDS:
        at = text.find(word)
        while at >= 0:
            spans.append((at, at + len(word), "CM"))
            at = text.find(word, at + len(word))
    for hit in _NUMBERED.finditer(text):  # 「상대 1번 진지」 → 그 진영의 S1
        side = camp if (hit.group(1) or "") in _OWN_PREFIX else foe
        for number in re.findall(r"[1-3]", hit.group(2)):
            spans.append((hit.start(), hit.end(), "%s-S%s" % (side[0], number)))
    spans.sort()
    starts = [s[0] for s in spans]
    for i, (_at, end, name) in enumerate(spans):
        # 섹터 이름 뒤 24자까지 본다 (전 12자). _window 가 이미 다음 섹터 이름 · 쉼표 · 문장 끝에서 끊는다.
        # 2026-11-19 본선 경기장 모의 경기 여섯 판의 명령문이 모두 「R-S1은 이번 탐색 대상에서 제외한다.」 꼴이었고 「제외」 가 13번째 글자부터라 놓쳤다
        # (LLM 이 답한 판이라 점수에는 영향이 없었다 — LLM 이 실패하면 탐색 구역 3점이 틀렸을 것이다)
        tail = _window(text, end, starts[i + 1 :])[:24]
        if any(cue in tail for cue in EXCLUDE) and name != "CM":
            excluded.add(name)
        elif name not in named:
            named.append(name)
    for word in CENTRAL_WORDS:  # 「상대 중앙진지를 제외하고 나머지」 — 중앙진지 = 상대 2번 줄 [추정]
        at = text.find(word)
        if at >= 0 and any(cue in text[at + len(word) : at + len(word) + 10] for cue in EXCLUDE):
            excluded.add(foe_sectors[1])
    wants_all = any(word in text for word in ALL_WORDS)
    wants_rest = "나머지" in text and bool(excluded)
    if wants_all or wants_rest:
        pool = list(C.camp_sectors(camp)) if any(word in text for word in OWN_WORDS) and not any(s[0] == foe[0] for s in excluded) else foe_sectors
        for name in pool:
            if name not in excluded and name not in named:
                named.append(name)
    return [s for s in C.SECTORS if s in named and s not in excluded], starts


def _described(text, noun, cues, verbs):
    """기능으로 돌려 말한 시설: 「<설명> 시설을 타격」 「<설명> 거점으로 이동」 의 <설명>에서 단서 낱말로 종류를 고른다.

    noun 「시설」 · 「거점」 이 나올 때마다 그 앞 설명(같은 절 안, 45자까지)을 보고, 뒤 문맥(18자)에 부정이 있으면 건너뛴다.
    뒤에 verbs(타격 · 이동 …)가 오는 것을 먼저 고르고, 점수가 같은 종류가 둘이면 모르는 것으로 둔다.
    """
    best = None  # (동사가 뒤따르는가, 점수, 종류)
    at = text.find(noun)
    prev_end = 0  # 앞에 나온 같은 낱말의 끝 — 설명은 거기서부터다 (「A 하는 시설이 아니라 B 하는 시설」 에서 B 의 설명에 A 가 섞이지 않게)
    while at >= 0:
        start = max(max((m.end() for m in _STOP.finditer(text, 0, at)), default=0), prev_end)
        prev_end = at + len(noun)
        desc = text[max(start, at - 45) : at]
        tail = text[at : at + len(noun) + 18]
        stop = _STOP.search(tail)
        if stop:
            tail = tail[: stop.start()]
        if not any(cue in tail for cue in NEGATIVE):
            scores = {kind: sum(w for word, w in words if word in desc) for kind, words in cues.items()}
            top = max(scores.values())
            winners = [kind for kind, score in scores.items() if score == top]
            if top > 0 and len(winners) == 1:
                rank = (any(verb in tail for verb in verbs), top, winners[0])
                if best is None or rank[:2] > best[:2]:
                    best = rank
        at = text.find(noun, at + len(noun))
    return best[2] if best else None


def parse(text, camp):
    """명령문 → {"target", "search_sectors", "base"} (보고값). camp = 우리 진영.

    시설 보고값의 진영은 그 시설이 놓인 진영이다 — 우리가 치는 표적 · 가는 거점은 상대 진영 것.
    """
    text = text or ""
    foe = C.opponent(camp)
    sectors, sector_starts = _sectors(text, camp)
    targets = _mentions(text, TARGET_WORDS)
    bases = _mentions(text, BASE_WORDS)
    target = _pick(text, TARGET_WORDS, bases, sector_starts)
    base = _pick(text, BASE_WORDS, targets, sector_starts)
    if target is None and not targets:  # 시설 이름이 없다 → 기능 설명에서 찾는다
        target = _described(text, "시설", TARGET_CUES, STRIKE_WORDS)
    if base is None and not bases:
        base = _described(text, "거점", BASE_CUES, MOVE_WORDS)
    return {
        C.M1_TARGET: C.facility(foe, target) if target else None,
        C.M1_SECTORS: sectors,
        C.M1_BASE: C.facility(foe, base) if base else None,
    }
