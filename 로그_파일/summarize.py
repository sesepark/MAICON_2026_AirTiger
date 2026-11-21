#!/usr/bin/env python3
"""기록 폴더 하나(로봇의 ~/team/log/<날짜_시각>/)를 읽어 무슨 일이 있었는지 한눈에 보이게 찍는다. 표준 라이브러리만 쓴다.

  python3 로그_파일/summarize.py <기록 폴더 | team_logs_….tgz | 여러 실행이 든 폴더>

tgz 를 주면 옆에 풀고(같은 이름의 폴더), 그 안의 실행을 하나씩 요약한다. 숫자를 지어내지 않는다 — 기록에 없는 것은 「기록 없음」 으로 찍는다.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import sys
import tarfile

OUTCOMES = {1: "REACHED", 2: "CANCELED", 3: "BLOCKED", 4: "TIMED_OUT", 5: "CONTROL_LOST", 6: "INVALID_GOAL", 7: "LOCALIZATION_LOST"}
REPORT_NAMES = {"ready": "준비완료", 1: "M1", 3: "M3", 6: "M6", "1": "M1", "3": "M3", "6": "M6"}
RESULT_NAMES = {1: "받음", 2: "거절", 3: "모름(통신)", None: "답 없음"}


def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass  # 프로그램이 강제로 끝나 잘린 마지막 줄
    except OSError:
        pass
    return rows


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def tt(t):
    return "  -   " if t is None else "%5.1f초" % t


def stats(values):
    values = sorted(values)
    if not values:
        return "기록 없음"
    mid = values[len(values) // 2]
    return "%d번 · 가운데값 %.2f초 · 가장 김 %.2f초" % (len(values), mid, values[-1])


def sessions_in(path):
    """기록 폴더(들)를 찾는다: version.json 이나 supervisor/ 가 있는 폴더."""
    found = []
    for root, dirs, files in os.walk(path):
        if "version.json" in files or "supervisor" in dirs or "team_node.log" in files:
            found.append(root)
            dirs[:] = []
    return sorted(found)


def summarize(folder):
    print("=" * 100)
    print("실행 기록: %s" % folder)
    version = read_json(os.path.join(folder, "version.json")) or {}
    meta = read_json(os.path.join(folder, "supervisor", "meta.json")) or {}
    ev = read_jsonl(os.path.join(folder, "supervisor", "events.jsonl"))
    photos = read_jsonl(os.path.join(folder, "supervisor", "photos.jsonl"))
    http = read_jsonl(os.path.join(folder, "server_bridge", "http.jsonl"))
    llm = read_jsonl(os.path.join(folder, "llm_worker", "llm.jsonl"))
    print("시작 %s · 켠 프로그램 %s" % (version.get("started_at", "기록 없음"), ", ".join(version.get("programs") or []) or "기록 없음"))
    print("코드 지문: team_node %s · %s%s" % (version.get("team_node"), " ".join("%s=%s" % kv for kv in sorted((version.get("packages") or {}).items())) or "기록 없음",
                                       (" · VERSION 파일: " + str(version["version_file"]).replace("\n", " / ")) if version.get("version_file") else ""))
    if meta:
        print("supervisor 설정: 정찰 %s · 화각 %s · 틸트 단위 %s · 필수 프로그램 %s%s" % (meta.get("scout_mode"), (meta.get("camera") or {}).get("hfov_deg", "기본"),
              (meta.get("gimbal") or {}).get("tilt_deg_per_unit"), meta.get("require_nodes"), " · 설정 오류: %s" % meta["_config_error"] if meta.get("_config_error") else ""))
    if not ev:
        print("supervisor 의 events.jsonl 이 없다 (supervisor 가 돌지 않았거나 기록을 끈 실행)")
    by = collections.defaultdict(list)
    for e in ev:
        by[e.get("e")].append(e)

    # ---------------------------------------------------------------- 경기 신호 · 하는 일
    print("\n[경기 신호] " + (" → ".join("%s(%s)" % (e.get("cmd"), tt(e.get("t")).strip()) for e in by["match"]) or "받은 것 없음"))
    ticks = by["tick"]
    if ticks:
        print("[자리] %s" % next((e.get("slot") for e in by["slot"]), "서버에서 받은 기록 없음"))
        changes, last = [], None
        for e in ticks:
            head = (e.get("task") or "").split(" ")[0]
            if head != last:
                changes.append("%s %s" % (tt(e.get("t")).strip(), head))
                last = head
        print("[하는 일] " + " → ".join(changes))
        none_pose = sum(1 for e in ticks if e.get("pose") is None)
        running = [e for e in ticks if e.get("phase") == "running"]
        print("[위치] 0.5초마다 %d번 적음 · 위치가 없던 때 %d번 (진행 중에는 %d번 중 %d번)" % (len(ticks), none_pose, len(running), sum(1 for e in running if e.get("pose") is None)))
        dist, turn, prev = 0.0, 0.0, None
        for e in running:
            p = e.get("pose")
            if p and prev:
                dist += math.hypot(p[0] - prev[0], p[1] - prev[1])
                turn += abs((p[2] - prev[2] + 180.0) % 360.0 - 180.0)
            prev = p or prev
        if prev:
            print("[움직임] 진행 중 간 거리 %.1f m · 돈 각도 %.0f° · 마지막 위치 (%.2f, %.2f, %.0f°)" % (dist, turn, prev[0], prev[1], prev[2]))
        not_ready = [e for e in ticks if e.get("phase") == "standby" and not e.get("nodes_ready")]
        if not_ready:
            print("[준비] 시작 대기 중 다른 프로그램이 준비되지 않았던 때 %d번 (마지막 상태 %s)" % (len(not_ready), not_ready[-1].get("node_states")))
        bat = [e.get("battery") for e in ticks if isinstance(e.get("battery"), (int, float)) and e.get("battery") >= 0]
        if bat:
            print("[배터리] %.0f%% → %.0f%%" % (bat[0], bat[-1]))
        hp = [(tt(e.get("t")).strip(), e.get("before"), e.get("hp")) for e in by["hp"]]
        if hp:
            print("[목숨] " + ", ".join("%s %s→%s" % h for h in hp))

    # ---------------------------------------------------------------- M1
    print("\n[M1] 명령문: %s" % (by["order"][0].get("text") if by["order"] else "받은 기록 없음"))
    for e in by["m1_parsed"]:
        print("     %s 해석 %s%s — %s" % (tt(e.get("t")), json.dumps(e.get("parsed"), ensure_ascii=False), " [%s]" % e["code"] if e.get("code") else "", e.get("detail")))
    for e in by["m1_failed"] + by["m1_fallback_failed"] + by["m1_ask_rejected"]:
        print("     %s %s %s" % (tt(e.get("t")), e.get("e"), {k: v for k, v in e.items() if k not in ("e", "wall", "t")}))
    for row in llm:
        if row.get("e") == "llm":
            print("     LLM 요청: %s · %.1f초 · %s · %s" % (row.get("model"), row.get("took_s") or 0.0, "반영" if row.get("ok") else "실패", row.get("note")))

    # ---------------------------------------------------------------- 이동
    goals = {e.get("no"): e for e in by["go"]}
    results = collections.Counter()
    took = []
    for r in by["go_result"]:
        results[OUTCOMES.get(r.get("outcome"), str(r.get("outcome")))] += 1
        g = goals.get(r.get("no"))
        if g and r.get("outcome") == 1:
            took.append(r["wall"] - g["wall"])
    print("\n[이동] 보낸 목표 %d개 · 결과 %s · 거절 %d · 도착까지 %s" % (len(goals), dict(results) or "없음", len(by["go_rejected"]), stats(took)))
    bad = [r for r in by["go_result"] if r.get("outcome") not in (1, 2)]
    for r in bad[:8]:
        g = goals.get(r.get("no"), {})
        print("     %s 목표 %s (%.2f, %.2f) → %s %s" % (tt(r.get("t")), r.get("no"), g.get("x", float("nan")), g.get("y", float("nan")), OUTCOMES.get(r.get("outcome"), r.get("outcome")), r.get("detail") or ""))

    # ---------------------------------------------------------------- 사진 · 판독 · 짝짓기
    shots, answers = by["photo"], by["classify"]
    ok = [a for a in answers if a.get("ok")]
    print("\n[사진] 찍음 %d장 (파일로 남긴 것 %d장) · 판독 답 %d번 (성공 %d · 실패 %d · 시한 넘김 %d) · 답까지 %s" % (
        len(shots), sum(1 for s in shots if s.get("file")), len(answers), len(ok), len(answers) - len(ok), len(by["classify_timeout"]), stats([a["took_s"] for a in answers if a.get("took_s") is not None])))
    if ok:
        kinds = collections.Counter(p.get("kind") for a in ok for p in a.get("pictures") or [])
        print("     답한 그림 종류: " + ", ".join("%s %d" % kv for kv in kinds.most_common()))
    if photos:
        paired = [p for p in photos if p.get("pairs")]
        expected = sum(len([x for x in p.get("expected") or [] if x.get("sure")]) for p in photos)
        print("[짝짓기] 사진 %d장 중 %d장을 받아들임 · 받아들인 짝 %d개 · 「확실히 보일 면」 으로 계산한 것 %d개 · 답한 그림 %d개" % (
            len(photos), len(paired), sum(len(p["pairs"]) for p in paired), expected, sum(len(p.get("pictures") or []) for p in photos)))
        rejected = [p for p in photos if not p.get("pairs")]
        for p in rejected[:6]:
            print("     버린 사진 %s: 위치 %s 팬 %s · 보일 면 %d(확실 %d) · 답 %s" % (p.get("tag"), p.get("pose"), p.get("pan_deg"), len(p.get("expected") or []),
                  len([x for x in p.get("expected") or [] if x.get("sure")]), None if p.get("pictures") is None else len(p["pictures"])))

    # ---------------------------------------------------------------- 지정표적
    notes = [(e.get("t"), e.get("text", "")) for e in by["note"]]
    for t, text in notes:
        if "지정표적" in text or "발사" in text or "탄 없음" in text or "포기" in text:
            print("[표적] %s %s" % (tt(t), text))
    fires = by["fire"]
    replies = collections.Counter(e.get("reply") for e in by["fire_reply"])
    print("[발사] 요청 %d번 · 답 %s · 팬틸트 명령 %d번 (답 %s)" % (len(fires), dict(replies) or "없음", len(by["gimbal"]), dict(collections.Counter(e.get("reply") for e in by["gimbal"])) or "없음"))
    for e in fires[:8]:
        print("     %s 쏜 자리 %s · 팬틸트 %s" % (tt(e.get("t")), e.get("pose"), e.get("pan_tilt")))

    # ---------------------------------------------------------------- 보고
    sent = collections.Counter()
    outcome = collections.defaultdict(collections.Counter)
    last_ok = {}
    for e in by["report"]:
        sent[REPORT_NAMES.get(e.get("kind"), e.get("kind"))] += 1
    for e in by["report_result"]:
        name = REPORT_NAMES.get(e.get("kind"), e.get("kind"))
        outcome[name][RESULT_NAMES.get(e.get("outcome"), e.get("outcome"))] += 1
        if e.get("outcome") == 1:
            last_ok[name] = e.get("t")
    print("\n[보고] " + (" · ".join("%s 보냄 %d번 %s (마지막으로 받아들여진 때 %s)" % (name, sent.get(name, 0), dict(outcome[name]), tt(last_ok.get(name)).strip())
                                 for name in ("준비완료", "M1", "M3", "M6") if sent.get(name) or outcome.get(name)) or "기록 없음"))
    rejected = [e for e in by["report_result"] if e.get("outcome") != 1]
    for e in rejected[:6]:
        print("     %s %s → %s HTTP %s %s" % (tt(e.get("t")), REPORT_NAMES.get(e.get("kind"), e.get("kind")), RESULT_NAMES.get(e.get("outcome"), e.get("outcome")), e.get("http"), e.get("detail") or ""))
    m3 = [e for e in by["report"] if e.get("kind") in (3, "3")]
    if m3:
        final = m3[-1].get("payload") or {}
        print("     마지막 M3 (%s): %s" % (tt(m3[-1].get("t")).strip(), json.dumps(final, ensure_ascii=False)))
    m6 = [e for e in by["report"] if e.get("kind") in (6, "6")]
    if m6:
        print("     M6 순위: %s" % (m6[-1].get("payload"),))

    # ---------------------------------------------------------------- 목표거점
    if by["objective"]:
        o = by["objective"][0]
        print("\n[목표거점] %s 에 %s 받음" % (tt(o.get("t")).strip(), o.get("code")))
        rect = None
        for m in by["map"]:
            rect = (m.get("bases") or {}).get(o.get("code"))
        last_pose = next((e.get("pose") for e in reversed(ticks) if e.get("pose")), None)
        if rect and last_pose:
            (x0, y0, x1, y1), _row = rect
            inside = x0 - 0.05 <= last_pose[0] <= x1 + 0.05 and y0 <= last_pose[1] <= y1
            print("     마지막 위치 (%.2f, %.2f) 는 그 거점 칸 (%.2f~%.2f, %.2f~%.2f) %s" % (last_pose[0], last_pose[1], x0, x1, y0, y1, "안" if inside else "밖"))
    else:
        print("\n[목표거점] 받은 기록 없음")

    # ---------------------------------------------------------------- 서버 · 드론
    if http:
        per = collections.defaultdict(list)
        for h in http:
            if h.get("e") == "http":
                per[h.get("endpoint")].append(h)
        print("\n[서버] 창구 %s" % sorted({h.get("prefix") for h in http if h.get("prefix")}))
        for name, rows in sorted(per.items()):
            codes = collections.Counter(str(r.get("status")) for r in rows)
            quick = [r["took_s"] for r in rows if r.get("took_s") is not None and not (r.get("params") or {}).get("wait")]
            print("     %-15s %3d번 · 상태 %s%s" % (name, len(rows), dict(codes), " · 응답 " + stats(quick) if quick else ""))
        drone = [h for h in http if h.get("e") == "drone"]
        if drone:
            good = [d for d in drone if d.get("status") == 200]
            print("     드론 화면: 새 화면 %d장 · 실패 %d번 · 크기 %s바이트 · 받는 데 %s" % (len(good), len(drone) - len(good), sorted({d.get("bytes") for d in good})[-1:] or "-",
                  stats([d["took_s"] for d in good if d.get("took_s") is not None])))
    else:
        print("\n[서버] server_bridge 의 http.jsonl 이 없다")

    # ---------------------------------------------------------------- 오류
    problems = collections.Counter()
    samples = {}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(".log"):
                continue
            try:
                text = open(os.path.join(root, name), encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for line in text.splitlines():
                if re.search(r"Traceback|\[ERROR\]|\[FATAL\]|예외|Error:", line):
                    key = os.path.relpath(os.path.join(root, name), folder)
                    problems[key] += 1
                    samples.setdefault(key, line.strip()[:200])
    if problems:
        print("\n[오류로 보이는 줄]")
        for key, count in problems.most_common():
            print("     %s: %d줄 — 예: %s" % (key, count, samples[key]))
    else:
        print("\n[오류로 보이는 줄] 없음")
    for line in (read_text(os.path.join(folder, "team_node.log")) or "").splitlines():
        if "다시 켭니다" in line or "못 켬" in line or "예외" in line:
            print("     team_node: " + line)


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = argv[1]
    if os.path.isfile(path) and path.endswith((".tgz", ".tar.gz")):
        out = re.sub(r"\.(tgz|tar\.gz)$", "", path)
        os.makedirs(out, exist_ok=True)
        with tarfile.open(path) as tar:
            root = os.path.realpath(out)
            members = [m for m in tar.getmembers() if os.path.realpath(os.path.join(out, m.name)).startswith(root + os.sep) and (m.isfile() or m.isdir())]
            tar.extractall(out, members=members)
        path = out
    found = sessions_in(path)
    if not found:
        print("기록 폴더를 찾지 못했다: %s" % path)
        return 1
    for folder in found:
        summarize(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
