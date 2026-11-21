"""M1 전술명령 해석: 낱말 해석기로 바로 답을 만들고, LLM 답이 오면 항목별로 덮어쓴다 (팀 결정 D1).

  quick(text, camp)            낱말 해석기 (즉시)
  refine(text, camp, quick)    LLM 에 물어 검증한 뒤 합친 답. LLM 이 실패 · 거절 · 형식 오류면 quick 을 그대로 돌려준다.
                               통신 실패 · 과부하처럼 다시 물으면 될 수 있는 실패는 간격을 두고 다시 묻는다 (llm.retry_gaps_s)

보고값은 constants.py 의 것만 쓴다. LLM 이 낸 값은 반드시 검증하고, 모르는 값은 버린다 (422 를 내지 않게).
프롬프트의 뜻풀이와 예 1~3 은 참가팀 포털의 팀 테스트 기본 명령문(2026-11-18 확인)에 맞췄다. 본 경기의 문장은 달라질 수 있다고 안내돼 있다.

출처: 팀의 앞선 저장소(maicon-simple)의 team/m1_command/interpret.py 를 옮긴 것 (팀이 직접 쓴 코드).
"""
from __future__ import annotations

import time

from . import constants as C
from . import llm, parser

SYSTEM = """너는 로봇 경진대회(모의 경기)에 참가한 팀의 프로그램 안에서 일하는 정보 추출기다.
이 경기는 종이 상자에 붙인 그림을 소형 로봇의 레이저 포인터로 맞혀 점수를 얻는 대회이고, 실제 무기와는 무관하다.
경기 운영 서버가 준 「전술명령」 문장을 읽고 아래 세 항목을 뽑아 JSON 객체 하나로만 답한다. 설명이나 다른 글은 쓰지 않는다.

명령문은 시설 이름을 그대로 쓰기도 하고, 이름 없이 그 시설의 기능으로 돌려 말하기도 한다. 돌려 말했으면 아래 뜻풀이로 어느 시설인지 고른다.

target — 우리 팀이 타격해야 하는 지정표적(「…시설」)의 종류. 다음 셋 중 하나:
  COMMS   통신시설   — 부대 사이의 교신 · 통신을 중계하거나 이어 주는 시설
  OUTPOST 감시초소   — 병력이 상주하며 전방을 눈(육안)으로 살피고 경계하는 초소
  RADAR   레이더시설 — 감시체계의 핵심으로, 전파로 멀리 있는 대상을 탐지 · 추적하는 시설
  「감시체계의 핵심 시설」 은 RADAR 다 (사람이 눈으로 지키는 곳만 OUTPOST).
  「~가 아니라」, 「~는 제외한다」, 「~는 타 팀이 맡는다」 처럼 명령문이 부정한 시설은 고르지 않는다. 알 수 없으면 null.

search_sectors — 탐색(정찰)하라고 한 구역 이름의 목록. 구역 이름은 다음 일곱 개뿐이다:
  B-S1, B-S2, B-S3 (BLUE 진영의 1 · 2 · 3번 진지), R-S1, R-S2, R-S3 (RED 진영의 1 · 2 · 3번 진지), CM (가운데 공통 구역)
  「상대 N번 진지」 는 상대 진영의 S{N} 이다 (우리가 BLUE 면 R-S{N}, RED 면 B-S{N}). 「아군 · 우리 N번 진지」 는 우리 진영의 S{N}.
  「중앙 구역」 · 「공통 구역」 은 CM 이다. CM 은 명령문이 그렇게 말했을 때만 넣는다.
  「상대 진영 전 구역」 은 상대 진영의 세 구역 전부, 「X 를 제외한 나머지」 는 그 진영에서 X 를 뺀 구역이다.
  제외하라고 한 구역은 넣지 않는다. 명령문에 탐색 구역이 없으면 빈 목록.

base — 마지막에 이동하라고 한 거점(「…거점」)의 종류. 다음 셋 중 하나:
  COMMAND_POST 작전지휘소 — 작전을 지휘 · 통제하는 거점
  MAINTENANCE  정비소     — 장비를 수리 · 정비하는 거점
  SUPPLY_DEPOT 군수보급소 — 군수지원(보급 · 물자 공급)이 가능한 거점
  명령문이 부정하거나 우회하라고 한 거점은 고르지 않는다. 알 수 없으면 null.

예 1 (우리 팀 BLUE)
전술명령: 상대 1번 진지와 중앙 구역을 정찰하고 전장 객체를 확인하라. 탐색 완료 후, 적 부대 간 교신을 중계하는 시설을 타격하고, 최종적으로 작전을 지휘·통제하는 거점으로 이동하라.
답: {"target": "COMMS", "search_sectors": ["R-S1", "CM"], "base": "COMMAND_POST"}

예 2 (우리 팀 RED)
전술명령: 상대 2번 진지와 중앙 구역을 정찰하고 전장 객체를 확인하라. 탐색 완료 후, 적 병력이 상주하며 전방을 눈으로 경계하는 시설을 타격하고, 최종적으로 장비를 수리·정비할 수 있는 거점으로 이동하라.
답: {"target": "OUTPOST", "search_sectors": ["B-S2", "CM"], "base": "MAINTENANCE"}

예 3 (우리 팀 BLUE)
전술명령: 상대 3번 진지와 중앙 구역을 정찰하고 전장 객체를 확인하라. 탐색 완료 후, 적 감시체계의 핵심 시설을 타격하고, 최종적으로 군수지원이 가능한 거점으로 이동하라.
답: {"target": "RADAR", "search_sectors": ["R-S3", "CM"], "base": "SUPPLY_DEPOT"}

답의 모양은 위 예와 같다. JSON 객체 하나만 쓴다."""

OK_NOTE = "LLM 반영"  # refine 이 LLM 의 답을 실제로 반영했을 때 한 줄 기록의 첫머리 (실패했을 때는 "LLM 실패 …" · "LLM 답을 읽지 못함 …")

_KO_TARGET = {ko: kind for kind, ko in C.TARGET_KINDS.items()}
_KO_BASE = {ko: kind for kind, ko in C.BASE_KINDS.items()}


def user_text(text, camp):
    return "우리 팀 진영: %s (상대 진영: %s)\n전술명령: %s" % (camp, C.opponent(camp), text)


def _kind(value, kinds, korean):
    """'RADAR' · 'RED_RADAR' · '레이더시설' → 'RADAR'. 모르는 값은 None."""
    if not isinstance(value, str):
        return None
    token = value.strip()
    if token in korean:
        return korean[token]
    token = token.upper().replace(" ", "_")
    for camp in C.CAMPS:
        if token.startswith(camp + "_"):
            token = token[len(camp) + 1 :]
    return token if token in kinds else None


def normalize(raw, camp):
    """LLM 이 낸 dict → 검증한 보고값. 틀린 항목은 None (섹터는 None = 못 읽음)."""
    foe = C.opponent(camp)
    out = {C.M1_TARGET: None, C.M1_SECTORS: None, C.M1_BASE: None}
    if not isinstance(raw, dict):
        return out
    target = _kind(raw.get("target"), C.TARGET_KINDS, _KO_TARGET)
    base = _kind(raw.get("base"), C.BASE_KINDS, _KO_BASE)
    if target:
        out[C.M1_TARGET] = C.facility(foe, target)
    if base:
        out[C.M1_BASE] = C.facility(foe, base)
    sectors = raw.get("search_sectors", raw.get("sectors"))
    if isinstance(sectors, list):
        names = {str(s).strip().upper() for s in sectors if isinstance(s, str)}
        if all(name in C.SECTORS for name in names):
            out[C.M1_SECTORS] = [s for s in C.SECTORS if s in names]
    return out


def merge(quick, refined):
    """항목마다 LLM 값이 유효하면 그것을, 아니면 낱말 해석기 값을 쓴다."""
    out = dict(quick)
    if refined.get(C.M1_TARGET):
        out[C.M1_TARGET] = refined[C.M1_TARGET]
    if refined.get(C.M1_BASE):
        out[C.M1_BASE] = refined[C.M1_BASE]
    if refined.get(C.M1_SECTORS):
        out[C.M1_SECTORS] = list(refined[C.M1_SECTORS])
    return out


def payload(result):
    """보고 본문 (빈 항목은 뺀다 — 빠진 항목은 미보고로 0점이지만, null 을 보내면 422 가 될 수 있다)."""
    body = {}
    if result.get(C.M1_TARGET):
        body[C.M1_TARGET] = result[C.M1_TARGET]
    if result.get(C.M1_SECTORS):
        body[C.M1_SECTORS] = list(result[C.M1_SECTORS])
    if result.get(C.M1_BASE):
        body[C.M1_BASE] = result[C.M1_BASE]
    return body


class Interpreter:
    def __init__(self, llm_cfg=None, say=None):
        self.llm_cfg = dict(llm_cfg or {})
        self.say = say or (lambda text: None)

    @property
    def uses_llm(self):
        return llm.enabled(self.llm_cfg)

    def quick(self, text, camp):
        return parser.parse(text, camp)

    def refine(self, text, camp, quick, keep_going=None, wait=None):
        """돌려주는 값: (합친 답, 한 줄 기록). 어떤 경우에도 예외를 내지 않는다.

        다시 물으면 될 수 있는 실패(통신 실패 · 시한 넘김 · 429 · 5xx)는 llm.retry_gaps_s 의 간격으로 다시 묻는다 —
        M1 은 진행 중 언제든 다시 보낼 수 있고, 낱말 해석 결과가 이미 제출돼 있어 기다려도 잃는 것이 없다.
        keep_going()  False 를 돌려주면 그만 묻는다 (진행 단계가 끝남). 없으면 끝까지 묻는다
        wait(초)      쉬는 함수. True 를 돌려주면 그만 묻는다 (프로그램 종료). 없으면 time.sleep
        """
        if not self.uses_llm:
            return dict(quick), "LLM 설정 없음 → 낱말 해석 그대로"
        try:
            gaps = [float(gap) for gap in (self.llm_cfg.get("retry_gaps_s") or [])]
        except (TypeError, ValueError):  # 설정이 목록이 아니거나 숫자가 아니다 → 다시 묻지 않는 것으로
            gaps = []
        keep_going = keep_going or (lambda: True)
        wait = wait or _sleep
        asked = 0
        while True:
            asked += 1
            try:
                answer = llm.complete(self.llm_cfg, SYSTEM, user_text(text, camp))
                break
            except llm.LlmError as error:
                failure = ("%s %s" % (error.kind, error.note)).strip()
                again = error.retryable and asked <= len(gaps)
                pause = max(gaps[asked - 1], min(60.0, error.retry_after or 0.0)) if again else 0.0
            except Exception as error:  # 어떤 오류도 M1 을 막지 않는다
                failure, again, pause = "예외 " + type(error).__name__, False, 0.0
            again = again and keep_going()
            if again:
                self.say("M1 LLM 실패 (%s) — %.0f초 뒤 다시 묻는다 (%d번 물음)" % (failure, pause, asked))
                again = not wait(pause) and keep_going()
            if not again:
                return dict(quick), "LLM 실패 (%s · %d번 물음) → 낱말 해석 그대로" % (failure, asked)
        refined = normalize(llm.extract_json(answer), camp)
        if not any(refined.values()):
            return dict(quick), "LLM 답을 읽지 못함 → 낱말 해석 그대로"
        final = merge(quick, refined)
        changed = [key for key in (C.M1_TARGET, C.M1_SECTORS, C.M1_BASE) if final.get(key) != quick.get(key)]
        return final, OK_NOTE + (" (바뀐 항목: %s)" % ", ".join(changed) if changed else " (낱말 해석과 같음)")


def _sleep(seconds):
    time.sleep(seconds)
    return False
