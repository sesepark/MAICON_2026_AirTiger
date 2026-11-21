"""M1: 낱말 해석기 · LLM 답 검증 · 합치기."""
import json

import pytest

from llm_worker import constants as C
from llm_worker import interpret, llm, parser


@pytest.mark.parametrize(
    "text, camp, expected",
    [
        ("레이더시설이 아니라 통신시설을 타격대상으로 한다. 목표거점은 정비소이다.", "BLUE", ("RED_COMMS", [], "RED_MAINTENANCE")),
        ("지정표적: 감시초소. 상대 진영 전 구역과 공통 구역(CM)을 탐색하라. 군수보급소 방면은 우회하고 작전지휘소를 후속 거점으로 삼아라.", "RED", ("BLUE_OUTPOST", ["B-S1", "B-S2", "B-S3", "CM"], "BLUE_COMMAND_POST")),
        ("상대 중앙진지를 제외하고 나머지 상대진영 탐색 구역을 정찰하라.", "BLUE", (None, ["R-S1", "R-S3"], None)),
        ("객체 탐색은 r-s1, R - S3 구역에서 수행하라.", "BLUE", (None, ["R-S1", "R-S3"], None)),
        ("", "BLUE", (None, [], None)),
        # 2026-11-19 본선 경기장 모의 경기의 명령문 (줄바꿈이 낱말 가운데 들어 있기도 했다)
        ("R-S1은 이번 탐색 대상에서 제외한다. R-S2와 CM을 중심으로 객체 탐색을 실시하고 적 레이더시설을 우선 타격하\n라. 후속 임무를 위해 적 진영의 군수보급소를 목표거점으로 지정한다.",
         "BLUE", ("RED_RADAR", ["CM", "R-S2"], "RED_SUPPLY_DEPOT")),
        ("B-S1은 이번 탐색 대상에서 제외한다. B-S2와 CM을 중심으로 객체 탐색을 실시하고 적 감시초소를 우선 타격하\n라. 후속 임무를 위해 적 진영의 군수보급소를 목표거점으로 지정한다.",
         "RED", ("BLUE_OUTPOST", ["B-S2", "CM"], "BLUE_SUPPLY_DEPOT")),
        ("R-S1은 이번 탐색 대상에서 제외한다. R-S2와 CM을 중심으로 객체 탐색을 실시하고 적 통신시설을 우선 타격하라. 후속 임무를 위해 적 진영의 군수보급소를 목표거점으로 지정한다.",
         "BLUE", ("RED_COMMS", ["CM", "R-S2"], "RED_SUPPLY_DEPOT")),
        ("R-S1을 정찰하고 R-S2는 제외하라.", "BLUE", (None, ["R-S1"], None)),  # 앞 섹터의 문맥은 다음 섹터 이름에서 끊긴다
    ],
)
def test_parser_cases(text, camp, expected):
    out = parser.parse(text, camp)
    assert (out[C.M1_TARGET], out[C.M1_SECTORS], out[C.M1_BASE]) == expected


def test_parser_keeps_unknown_empty():
    """옛 명칭(1차 사전교육 p.11)은 모르는 값이다 → 지어내지 않는다."""
    out = parser.parse("레이더시설를 주요 타격대상으로 지정한다. 객체 탐색은 CHARLIE와 DELTA 구역에서 수행하도록 계획하며, 후속 작전 거점은 통신중계소로 설정하라.", "BLUE")
    assert out == {C.M1_TARGET: "RED_RADAR", C.M1_SECTORS: [], C.M1_BASE: None}


def test_normalize_accepts_code_korean_and_prefixed():
    assert interpret.normalize({"target": "RADAR", "search_sectors": ["r-s1", "CM"], "base": "정비소"}, "BLUE") == {C.M1_TARGET: "RED_RADAR", C.M1_SECTORS: ["CM", "R-S1"], C.M1_BASE: "RED_MAINTENANCE"}
    assert interpret.normalize({"target": "BLUE_COMMS", "base": "red_supply_depot"}, "RED")[C.M1_TARGET] == "BLUE_COMMS"


def test_normalize_drops_unknown_values():
    out = interpret.normalize({"target": "TANK", "search_sectors": ["R-S1", "X-9"], "base": None}, "BLUE")
    assert out == {C.M1_TARGET: None, C.M1_SECTORS: None, C.M1_BASE: None}
    assert interpret.normalize("not a dict", "BLUE")[C.M1_TARGET] is None


def test_merge_prefers_valid_llm_fields():
    quick = {C.M1_TARGET: "RED_RADAR", C.M1_SECTORS: ["R-S1"], C.M1_BASE: None}
    refined = {C.M1_TARGET: None, C.M1_SECTORS: ["CM", "R-S2"], C.M1_BASE: "RED_MAINTENANCE"}
    assert interpret.merge(quick, refined) == {C.M1_TARGET: "RED_RADAR", C.M1_SECTORS: ["CM", "R-S2"], C.M1_BASE: "RED_MAINTENANCE"}


def test_payload_omits_empty_fields():
    assert interpret.payload({C.M1_TARGET: "RED_RADAR", C.M1_SECTORS: [], C.M1_BASE: None}) == {C.M1_TARGET: "RED_RADAR"}


def test_extract_json_tolerates_fences_and_prose():
    assert llm.extract_json('답:\n```json\n{"target": "RADAR", "search_sectors": []}\n```') == {"target": "RADAR", "search_sectors": []}
    assert llm.extract_json("JSON 없음") is None


def _fake_post(replies, seen):
    def post(url, headers, body, timeout):
        seen.append((url, headers, body))
        status, payload = replies.pop(0)
        return status, json.dumps(payload).encode("utf-8"), {}

    return post


def test_anthropic_request_shape_and_text(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_post", _fake_post([(200, {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": '{"target": "RADAR"}'}]})], seen))
    cfg = {"provider": "anthropic", "api_key": "k", "extra": {"output_config": {"effort": "low"}}}
    assert llm.complete(cfg, "시스템", "사용자") == '{"target": "RADAR"}'
    url, headers, body = seen[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "k" and headers["anthropic-version"] == llm.ANTHROPIC_VERSION
    assert body["model"] == llm.ANTHROPIC_MODEL and body["system"] == "시스템" and body["messages"] == [{"role": "user", "content": "사용자"}]
    assert "temperature" not in body and body["output_config"] == {"effort": "low"}


def test_anthropic_refusal_is_an_error(monkeypatch):
    monkeypatch.setattr(llm, "_post", _fake_post([(200, {"stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []})], []))
    with pytest.raises(llm.LlmError) as error:
        llm.complete({"provider": "anthropic", "api_key": "k"}, "s", "u")
    assert error.value.kind == "refusal"


def test_openai_request_shape(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_post", _fake_post([(200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})], seen))
    assert llm.complete({"provider": "openai", "api_key": "k", "model": "m", "url": "http://proxy:9"}, "s", "u") == "{}"
    url, headers, body = seen[0]
    assert url == "http://proxy:9/v1/chat/completions" and headers["Authorization"] == "Bearer k"
    assert body["messages"][0] == {"role": "system", "content": "s"}


def test_retry_then_success(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "_post", _fake_post([(529, {"error": {"type": "overloaded_error", "message": "x"}}), (200, {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]})], []))
    assert llm.complete({"provider": "anthropic", "api_key": "k", "retries": 1}, "s", "u") == "ok"


def test_interpreter_falls_back_when_llm_fails(monkeypatch):
    def boom(cfg, system, user):
        raise llm.LlmError("network", "x")

    monkeypatch.setattr(llm, "complete", boom)
    it = interpret.Interpreter({"provider": "anthropic", "api_key": "k"})
    quick = it.quick("지정표적: 레이더시설.", "BLUE")
    final, note = it.refine("지정표적: 레이더시설.", "BLUE", quick)
    assert final == quick and "낱말 해석 그대로" in note


def test_interpreter_uses_llm_when_valid(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda cfg, system, user: '{"target": "COMMS", "search_sectors": ["R-S2"], "base": "MAINTENANCE"}')
    it = interpret.Interpreter({"provider": "openai", "api_key": "k", "model": "m"})
    final, note = it.refine("…", "BLUE", {C.M1_TARGET: None, C.M1_SECTORS: [], C.M1_BASE: None})
    assert final == {C.M1_TARGET: "RED_COMMS", C.M1_SECTORS: ["R-S2"], C.M1_BASE: "RED_MAINTENANCE"} and "LLM 반영" in note



def test_interpreter_survives_a_broken_retry_setting(monkeypatch):
    """설정값이 이상해도 refine() 은 예외를 내지 않는다 (예외가 나면 LLM 스레드가 죽는다)."""
    monkeypatch.setattr(llm, "complete", lambda cfg, system, user: '{"target": "COMMS"}')
    it = interpret.Interpreter({"provider": "anthropic", "api_key": "k", "retry_gaps_s": "5"})
    final, note = it.refine("…", "BLUE", {C.M1_TARGET: None, C.M1_SECTORS: [], C.M1_BASE: None})
    assert final[C.M1_TARGET] == "RED_COMMS" and "LLM 반영" in note
    it = interpret.Interpreter({"provider": "anthropic", "api_key": "k", "retry_gaps_s": 7})
    assert it.refine("…", "BLUE", {C.M1_TARGET: None, C.M1_SECTORS: [], C.M1_BASE: None})[0][C.M1_TARGET] == "RED_COMMS"


# 참가팀 포털의 팀 테스트 기본 명령문 세 가지 (2026-11-18 확인 — 「팀 테스트 고정 예시. 본 경기의 전술명령 형식과 임무 값은 달라질 수 있습니다」).
# 시설 이름 없이 기능으로 돌려 말한다. 정답은 포털이 같이 준 값이다.
OFFICIAL = [
    ("상대 1번 진지와 중앙 구역을 정찰하고 전장 객체를 확인하라. 탐색 완료 후, 적 부대 간 교신을 중계하는 시설을 타격하고, 최종적으로 작전을 지휘·통제하는 거점으로 이동하라.",
     "COMMS", 1, "COMMAND_POST"),
    ("상대 2번 진지와 중앙 구역을 정찰하고 전장 객체를 확인하라. 탐색 완료 후, 적 병력이 상주하며 전방을 눈으로 경계하는 시설을 타격하고, 최종적으로 장비를 수리·정비할 수 있는 거점으로 이동하라.",
     "OUTPOST", 2, "MAINTENANCE"),
    ("상대 3번 진지와 중앙 구역을 정찰하고 전장 객체를 확인하라. 탐색 완료 후, 적 감시체계의 핵심 시설을 타격하고, 최종적으로 군수지원이 가능한 거점으로 이동하라.",
     "RADAR", 3, "SUPPLY_DEPOT"),
]


@pytest.mark.parametrize("camp", ["BLUE", "RED"])
@pytest.mark.parametrize("text, target, row, base", OFFICIAL)
def test_word_rules_solve_the_official_descriptive_orders(text, target, row, base, camp):
    foe = C.opponent(camp)
    got = parser.parse(text, camp)
    assert got[C.M1_TARGET] == C.facility(foe, target)
    assert got[C.M1_SECTORS] == [s for s in C.SECTORS if s in ("%s-S%d" % (foe[0], row), "CM")]
    assert got[C.M1_BASE] == C.facility(foe, base)


@pytest.mark.parametrize(
    "text, camp, expected",
    [
        # 아래는 내가 지은 변형이다 (실제 명령문이 아니다)
        ("상대 3번 진지를 정찰하라. 병력이 눈으로 전방을 살피는 시설이 아니라 부대 간 통신을 이어 주는 시설을 타격한다. 장비를 고치는 거점은 우회하고 지휘부가 있는 거점으로 집결하라.",
         "RED", ("BLUE_COMMS", ["B-S3"], "BLUE_COMMAND_POST")),
        ("적 2번 진지를 제외한 나머지 상대 진지와 중앙 구역을 수색하라. 이후 전파로 항공기를 탐지하는 시설을 무력화하고 물자를 보급받을 수 있는 거점을 확보하라.",
         "BLUE", ("RED_RADAR", ["CM", "R-S1", "R-S3"], "RED_SUPPLY_DEPOT")),
        ("상대 1번·3번 진지를 살펴 전장 객체를 식별하라. 초병이 상주하며 육안으로 접근로를 감시하는 시설을 제압한 뒤, 고장 난 장비를 손볼 수 있는 거점으로 향하라.",
         "BLUE", ("RED_OUTPOST", ["R-S1", "R-S3"], "RED_MAINTENANCE")),
        ("우리 2번 진지와 중앙 구역을 정찰하라.", "RED", (None, ["CM", "R-S2"], None)),
    ],
)
def test_word_rules_on_descriptive_variations(text, camp, expected):
    got = parser.parse(text, camp)
    assert (got[C.M1_TARGET], got[C.M1_SECTORS], got[C.M1_BASE]) == expected


def test_prompt_carries_the_function_glossary_and_examples():
    for word in ("교신", "눈(육안)", "감시체계의 핵심", "지휘 · 통제", "수리 · 정비", "군수지원", "N번 진지", "중앙 구역"):
        assert word in interpret.SYSTEM
