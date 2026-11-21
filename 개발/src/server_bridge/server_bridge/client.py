"""경기 서버 HTTP 클라이언트 (표준 라이브러리만).

요청마다 상태 코드 확인 · 시한 · 예외 처리를 한다. 재시도는 부르는 쪽(supervisor)이 단계를 다시 확인한 뒤 정한다 —
판정은 서버가 받은 시각으로만 하므로, 단계가 바뀐 뒤의 재전송은 409 가 된다.

응답을 기다리는 시한은 고정값이 아니다. 시한 안에 답이 안 오면 다음 요청부터 두 배로 늘리고(상한 timeout_max),
답이 빨라지면 천천히 줄인다. 고정 4초였을 때는 응답이 4초보다 늦어지는 망에서 모든 요청이 실패했다 —
GET /me 가 한 번도 성공하지 못해 로봇이 자기 자리를 모르고 경기 내내 서 있었다 (앞선 저장소의 가상 경기 기록).
시한은 창구(me · drone · detection …)마다 따로 둔다. 하나로 두면, 느린 창구(큰 드론 화면 · 보고)가 늘려 놓은 시한을
빠른 창구(GET /me)의 답이 곧바로 도로 줄여서 느린 창구는 끝내 답을 받지 못한다.
처음 쓰는 창구는 다른 창구들이 지금 쓰는 시한 가운데 가장 긴 것에서 시작한다 — 망 전체가 느릴 때 창구마다 시한 넘김을 한 번씩 다시 겪지 않게.

임무코드는 어디에도 출력하지 않는다 (예외 글에도 URL 을 넣지 않는다).

출처: 팀의 앞선 저장소(maicon-simple)의 team/server/client.py 를 옮긴 것 (팀이 직접 쓴 코드).
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import constants as C


def timed_out(error):
    """시한을 넘겨서 난 예외인가 (연결 단계의 시한은 URLError 안에 들어 있다)."""
    kinds = (TimeoutError, socket.timeout)
    return isinstance(error, kinds) or isinstance(getattr(error, "reason", None), kinds)


class ServerClient:
    def __init__(self, base, prefix, code, timeout=4.0, timeout_max=12.0):
        self.base = base.rstrip("/")
        self.prefix = "/" + prefix.strip("/") + "/"
        self._code = code
        self.timeout = float(timeout)  # 평소 시한
        self.timeout_max = max(self.timeout, float(timeout_max))
        self._patience = {}  # 창구 → 지금 쓰는 시한 (응답이 늦으면 늘어난다). 없으면 평소 시한
        self._lock = threading.Lock()
        self.counts = {"ok": 0, "timeout": 0, "cut": 0, "5xx": 0}  # ok = 서버가 답함(4xx 포함) · timeout = 시한 넘김 · cut = 연결 실패 · 끊김

    def with_base(self, base):
        """주소만 다른 같은 클라이언트 (본 경기 포트 ↔ 시험 포트). 시한 · 통계는 따로 둔다."""
        return ServerClient(base, self.prefix, self._code, self.timeout, self.timeout_max)

    @property
    def patience(self):
        """지금 가장 길게 잡힌 시한 (롱폴링의 여유 · 기록에 쓴다)."""
        with self._lock:
            return max(self._patience.values(), default=self.timeout)

    def _url(self, endpoint, params=None):
        url = self.base + self.prefix + endpoint
        if params is not None:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def _learn(self, endpoint, outcome, spent, adapt):
        """요청 하나의 결과를 세고, 그 창구의 다음 시한을 고친다 (adapt: 롱폴링처럼 서버가 일부러 붙잡는 요청은 뺀다)."""
        with self._lock:
            self.counts[outcome] += 1
            if not adapt:
                return
            now = self._limit(endpoint)
            if outcome == "timeout":
                self._patience[endpoint] = min(self.timeout_max, now * 2.0)
            elif outcome in ("ok", "5xx"):  # 서버가 답했다 → 걸린 시간의 세 배를 시한으로. 줄일 때는 한 번에 5% 씩만
                want = min(self.timeout_max, max(self.timeout, 3.0 * spent))
                self._patience[endpoint] = want if want >= now else max(want, now * 0.95)

    def note(self):
        """기록용 한 줄: 요청 결과 수와 지금 (가장 긴) 시한."""
        longest = self.patience
        with self._lock:
            c = dict(self.counts)
        return "답 %d · 시한 넘김 %d · 끊김 %d · 5xx %d · 시한 %.0f초" % (c["ok"], c["timeout"], c["cut"], c["5xx"], longest)

    def _limit(self, endpoint):
        """그 창구의 지금 시한 (잠금을 잡은 채 부른다). 처음 쓰는 창구면 다른 창구들의 시한 가운데 가장 긴 것."""
        own = self._patience.get(endpoint)
        return own if own is not None else max(self._patience.values(), default=self.timeout)

    def _open(self, endpoint, request, timeout):
        """요청 하나. 돌려주는 값: (상태 코드 또는 None, 본문 바이트 또는 None, 응답 머리 dict)."""
        with self._lock:
            limit = float(timeout) if timeout else self._limit(endpoint)
        began = time.monotonic()
        try:
            try:
                with urllib.request.urlopen(request, timeout=limit) as response:
                    out = response.status, response.read(), {k: v for k, v in response.headers.items()}
            except urllib.error.HTTPError as error:  # 4xx · 5xx 도 서버의 답이다
                try:
                    raw = error.read()
                except (OSError, http.client.HTTPException):
                    raw = None
                out = error.code, raw, {}
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as error:  # 끊긴 응답도 통신 실패다
            self._learn(endpoint, "timeout" if timed_out(error) else "cut", time.monotonic() - began, not timeout)
            return None, None, {}
        self._learn(endpoint, "5xx" if out[0] >= 500 else "ok", time.monotonic() - began, not timeout)
        return out

    def call(self, method, endpoint, params=None, body=None, timeout=None):
        """돌려주는 값: (상태 코드 또는 None, JSON 또는 None). 통신 실패면 (None, None).

        timeout 을 주지 않으면 지금 시한(patience)을 쓴다. 롱폴링은 timeout 을 wait 보다 길게 준다.
        """
        data = None
        headers = {}
        if method == "GET":
            query = {"mission_code": self._code}
            query.update(params or {})
            url = self._url(endpoint, query)
        else:
            payload = {"mission_code": self._code}
            payload.update(body or {})
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            url = self._url(endpoint)
        status, raw, _headers = self._open(endpoint, urllib.request.Request(url, data=data, headers=headers, method=method), timeout)
        return status, (_json(raw) if raw is not None else None)

    def drone(self, camera=None, timeout=None):
        """상공 화면 한 장. 돌려주는 값: (상태 코드 또는 None, JPEG 바이트 또는 None, 응답 머리 dict).

        경기 때는 camera 없이 부른다. 연습 기간 테스트는 camera="main"|"test1"|"test2" 를 꼭 붙인다 (API 4단계).
        """
        query = {"mission_code": self._code}
        if camera:
            query["camera"] = camera
        status, raw, headers = self._open(C.EP_DRONE, urllib.request.Request(self._url(C.EP_DRONE, query), method="GET"), timeout)
        return status, (raw if status == 200 else None), headers


def _json(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def detail_text(out, limit=160):
    """오류 응답 {"detail": …} 을 기록용 글로 (422 는 detail 이 목록이다)."""
    if not isinstance(out, dict):
        return ""
    detail = out.get("detail", out.get("reason", ""))
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail)[:limit]
