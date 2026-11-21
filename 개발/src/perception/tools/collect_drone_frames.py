"""연습 경기장 드론 화면 수집기 — 경계 검출 개발 · 시험 자료를 모은다.

경기용 main 카메라 화면은 경기 중에만 나오므로, 연습 기간에 예비 경기장(test1 · test2) 화면을 최대한 다양하게
(박스 배치 · 로봇 · 사람 · 조명) 모아 둔다. 표준 라이브러리만 쓴다.

    set MAICON_MISSION_CODE=<임무코드>          (PowerShell: $env:MAICON_MISSION_CODE = "<임무코드>")
    python collect_drone_frames.py --out C:/Users/Admin/MAICON/drone_frames
    python collect_drone_frames.py --interval 10 --cameras test1 test2 --count 1   # 한 번만

저장:
    <out>/<camera>/<캡처시각>.jpg       캡처시각 = X-Captured-At (없으면 받은 시각)
    <out>/index.jsonl                   한 줄에 한 장: 파일 · 카메라 · 응답 머리 · 받은 시각 · 크기 · sha1
같은 장(같은 X-Captured-At 또는 같은 내용)은 다시 저장하지 않는다.

연습 화면은 시험 포트 8100 + camera=test1/test2 로 받는다 (8000 으로 부르면 409 — 서버 오류문 기준. API 문서 §4 예시와 다름).
임무코드는 파일에 남기지 않는다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

KEEP_HEADERS = ("X-Captured-At", "X-Frame-Interval-S", "X-Camera", "X-Test-Mode", "X-Camp", "X-Destroyed", "Content-Type")


def fetch(base, code, camera, timeout):
    """(상태 코드, 머리 dict, 본문 bytes). 연결 실패면 상태 코드 None."""
    query = {"mission_code": code}
    if camera:
        query["camera"] = camera
    url = "%s/api/report/drone?%s" % (base.rstrip("/"), urllib.parse.urlencode(query))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as res:
            return res.status, {k.lower(): v for k, v in res.headers.items()}, res.read()
    except urllib.error.HTTPError as err:
        body = err.read()
        return err.code, {k.lower(): v for k, v in err.headers.items()}, body
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        return None, {}, str(err).encode("utf-8", "replace")


def safe_stamp(text):
    """2026-11-18T14:17:11.073+09:00 → 20261118_141711_073 (파일 이름용)."""
    digits = re.sub(r"[^0-9]", "", text or "")
    if len(digits) >= 14:
        return "%s_%s_%s" % (digits[:8], digits[8:14], digits[14:17] or "000")
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]


def load_seen(index_path):
    seen = set()
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                seen.add((row.get("camera"), row.get("headers", {}).get("X-Captured-At")))
                seen.add(("sha1", row.get("sha1")))
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://192.168.110.50:8100", help="서버 주소 (연습: 8100)")
    ap.add_argument("--cameras", nargs="+", default=["test1", "test2"])
    ap.add_argument("--interval", type=float, default=10.0, help="한 바퀴(카메라 전부) 간격 초")
    ap.add_argument("--count", type=int, default=0, help="이만큼 돌고 끝낸다 (0 = 멈출 때까지)")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "MAICON", "drone_frames"))
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args(argv)

    code = os.environ.get("MAICON_MISSION_CODE", "").strip()
    if not code:
        print("환경변수 MAICON_MISSION_CODE 에 임무코드를 넣으세요", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.jsonl")
    seen = load_seen(index_path)
    saved = 0
    rounds = 0
    fails = 0
    print("수집 시작: %s · %s · %.0f초 간격 → %s" % (args.base, ",".join(args.cameras), args.interval, args.out), flush=True)
    try:
        while True:
            t0 = time.monotonic()
            for cam in args.cameras:
                status, headers, body = fetch(args.base, code, cam, args.timeout)
                now = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
                if status != 200 or not body.startswith(b"\xff\xd8"):
                    fails += 1
                    print("[%s] %s: 실패 %s %s" % (now, cam, status, body[:120].decode("utf-8", "replace")), flush=True)
                    continue
                fails = 0
                captured = headers.get("x-captured-at")
                sha1 = hashlib.sha1(body).hexdigest()
                if (captured and (cam, captured) in seen) or ("sha1", sha1) in seen:
                    continue
                folder = os.path.join(args.out, cam)
                os.makedirs(folder, exist_ok=True)
                name = safe_stamp(captured or now) + ".jpg"
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(body)
                row = {"file": "%s/%s" % (cam, name), "camera": cam, "received_at": now, "bytes": len(body), "sha1": sha1,
                       "headers": {k: headers[k.lower()] for k in KEEP_HEADERS if k.lower() in headers}}
                with open(index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.update({(cam, captured), ("sha1", sha1)})
                saved += 1
                print("[%s] %s: 저장 %s (%d KB, 누적 %d장)" % (now, cam, row["file"], len(body) // 1024, saved), flush=True)
            rounds += 1
            if args.count and rounds >= args.count:
                break
            backoff = min(60.0, args.interval * (2 ** min(fails // len(args.cameras), 3))) if fails else args.interval
            time.sleep(max(0.0, backoff - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    print("끝: %d바퀴, %d장 저장" % (rounds, saved), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
