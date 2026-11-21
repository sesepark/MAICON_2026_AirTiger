"""운영팀이 발급한 LLM API 를 부른다 (텍스트만). 표준 라이브러리만 쓴다.

공급자는 Anthropic 또는 OpenAI 중 한 곳이고, 주소 · 모델 · 요청 형식은 OT 에서 안내된다 (Q&A 6절) → 두 형식을 모두 준비하고
주소 · 모델은 설정(llm_worker.json)으로, 키는 비밀 파일로 받는다. 경기 중에는 운영팀 키만 쓴다 (웹 LLM · 개인 키 금지).

SDK(anthropic · openai 패키지)를 쓰지 않는 이유: 로봇의 ROS2 Docker 안에 패키지를 더 깔 수 있는지 모르고(DAY 1 확인),
운영팀 주소가 중계 서버일 수도 있다. 요청 모양은 공식 문서의 원시 HTTP 형식을 그대로 따랐다.

요청은 일부러 최소로 보낸다 (model · max_tokens · system · messages). temperature 는 넣지 않는다 — 최신 Claude 모델과
API 버전 2026-10-21 은 temperature 를 400 으로 거절한다. 모델에 맞춰 더 넣을 것(effort, 구조화 출력 등)은 설정의 "extra" 에 적는다:

  {"provider": "anthropic", "model": "claude-opus-5-5", "extra": {"output_config": {"effort": "low"}}}

max_tokens 는 넉넉히 둔다. 생각(thinking)이 늘 켜지는 모델은 그 토큰도 max_tokens 안에서 쓰므로, 작게 잡으면 답이 잘린다.
시한(timeout_s)도 넉넉히 둔다. 답을 한꺼번에 받으므로(스트리밍 아님) 모델이 답을 다 만들 때까지 첫 글자가 오지 않는다.
늦게 와도 잃는 것은 없다 — 낱말 해석 결과를 먼저 돌려주고, 이 질의는 별도 스레드에서 돈다 (node.py).

실패는 둘로 나눈다 (LlmError.retryable).
  다시 물으면 될 수 있는 것   통신 실패 · 시한 넘김 · 408 · 409 · 429 · 5xx (경기 시작 순간에는 여러 팀이 한꺼번에 묻는다)
  다시 물어도 같은 것         설정 오류 · 400 · 401 · 403 · 404 · 거절 · 빈 답
여기서는 바로 다시 묻는 횟수(retries)만 다루고, 그 뒤의 재시도는 부르는 쪽이 한다 (supervisor 가 다시 물어 오면 node.py 가 새로 요청한다).
API 키는 예외 글 · 기록 어디에도 넣지 않는다.

출처: 팀의 앞선 저장소(maicon-simple)의 team/m1_command/llm.py 를 옮긴 것 (팀이 직접 쓴 코드).
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_PATH = "/v1/messages"
ANTHROPIC_VERSION = "2026-10-21"
ANTHROPIC_MODEL = "claude-opus-5-5"
OPENAI_URL = "https://api.openai.com"
OPENAI_PATH = "/v1/chat/completions"
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)


class LlmError(Exception):
    """LLM 답을 얻지 못함. kind: config · http · network · refusal · empty.

    retryable: 다시 물으면 될 수 있는 실패인가. retry_after: 서버가 알려 준 기다릴 시간(초, Retry-After 머리) 또는 None.
    """

    def __init__(self, kind, note="", retryable=False, retry_after=None):
        super().__init__("%s %s" % (kind, note))
        self.kind = kind
        self.note = note
        self.retryable = bool(retryable)
        self.retry_after = retry_after


def _merge(body, extra):
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(body.get(key), dict):
            _merge(body[key], value)
        else:
            body[key] = value
    return body


def _post(url, headers, body, timeout):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        try:
            raw = error.read()
        except (OSError, http.client.HTTPException):
            raw = b""
        return error.code, raw, dict(error.headers.items()) if error.headers else {}
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as error:
        raise LlmError("network", type(error).__name__, retryable=True) from None


def _retry_after(headers):
    """Retry-After 머리(초). 없거나 숫자가 아니면 None."""
    for key, value in (headers or {}).items():
        if str(key).lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


def _request(url, headers, body, cfg):
    """재시도를 포함한 한 번의 질의. 돌려주는 값: 응답 JSON."""
    timeout = float(cfg.get("timeout_s", 20.0))
    retries = int(cfg.get("retries", 1))
    last = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(min(2.0, max(0.5 * attempt, last.retry_after or 0.0)))
        try:
            status, raw, reply_headers = _post(url, headers, body, timeout)
        except LlmError as error:
            last = error
            continue
        if status == 200:
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                last = LlmError("http", "200 인데 JSON 이 아님", retryable=True)  # 중계 서버의 오류 화면 등
                continue
        last = LlmError("http", "%d %s" % (status, _error_note(raw)), retryable=status in RETRY_STATUS, retry_after=_retry_after(reply_headers))
        if status not in RETRY_STATUS:
            break
    raise last or LlmError("network", "응답 없음", retryable=True)


def _error_note(raw):
    """오류 본문에서 종류만 뽑는다 (본문 전체를 기록에 남기지 않는다)."""
    try:
        out = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return ""
    error = out.get("error") if isinstance(out, dict) else None
    if isinstance(error, dict):
        return "%s: %s" % (error.get("type", ""), str(error.get("message", ""))[:120])
    return ""


def _anthropic(cfg, system, user):
    key = cfg.get("api_key")
    if not key:
        raise LlmError("config", "llm.api_key 없음")
    url = (cfg.get("url") or ANTHROPIC_URL).rstrip("/") + (cfg.get("path") or ANTHROPIC_PATH)
    headers = {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": cfg.get("api_version") or ANTHROPIC_VERSION}
    betas = cfg.get("betas") or []
    if betas:
        headers["anthropic-beta"] = ",".join(betas)
    body = {
        "model": cfg.get("model") or ANTHROPIC_MODEL,
        "max_tokens": int(cfg.get("max_tokens", 2000)),
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    out = _request(url, headers, _merge(body, cfg.get("extra")), cfg)
    if out.get("stop_reason") == "refusal":  # HTTP 200 으로 온다 — content 를 읽기 전에 확인
        details = out.get("stop_details") or {}
        raise LlmError("refusal", str(details.get("category") or ""))
    texts = [block.get("text", "") for block in out.get("content") or [] if isinstance(block, dict) and block.get("type") == "text"]
    text = "".join(texts).strip()
    if not text:
        raise LlmError("empty", "stop_reason=%s" % out.get("stop_reason"))
    return text


def _openai(cfg, system, user):
    key = cfg.get("api_key")
    if not key:
        raise LlmError("config", "llm.api_key 없음")
    if not cfg.get("model"):
        raise LlmError("config", "llm.model 없음 (OpenAI 형식은 모델 이름을 꼭 적는다)")
    url = (cfg.get("url") or OPENAI_URL).rstrip("/") + (cfg.get("path") or OPENAI_PATH)
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    body = {"model": cfg["model"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    out = _request(url, headers, _merge(body, cfg.get("extra")), cfg)
    choices = out.get("choices") or []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):  # 조각 목록으로 오는 형식
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if isinstance(message, dict) and message.get("refusal"):
        raise LlmError("refusal", "")
    text = (content or "").strip()
    if not text:
        raise LlmError("empty", "finish_reason=%s" % (choices[0].get("finish_reason") if choices else None))
    return text


PROVIDERS = {"anthropic": _anthropic, "openai": _openai}


def enabled(cfg):
    return bool(cfg) and cfg.get("provider") in PROVIDERS and bool(cfg.get("api_key"))


def complete(cfg, system, user):
    """LLM 에 한 번 묻고 답 글을 돌려준다. 실패하면 LlmError."""
    call = PROVIDERS.get((cfg or {}).get("provider"))
    if call is None:
        raise LlmError("config", "llm.provider 는 anthropic 또는 openai")
    return call(cfg, system, user)


def extract_json(text):
    """답에서 JSON 객체 하나를 꺼낸다 (앞뒤 설명이나 ``` 울타리가 붙어 있어도). 없으면 None."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    at = text.find("{")
    while at >= 0:
        try:
            value, _end = decoder.raw_decode(text[at:])
        except ValueError:
            at = text.find("{", at + 1)
            continue
        if isinstance(value, dict):
            return value
        at = text.find("{", at + 1)
    return None
