"""드론 사진 여러 장을 지도 계산(map_maker.compute 와 같은 순서)에 넣어 보고, 사진마다 한 줄로 결과를 찍는다.

쓰는 법
    python check_pictures.py <폴더 또는 jpg 파일들> [--out 폴더] [--arena-height 5.0] [--cache 파일]

하는 일 (사진마다)
    1) 경계 찾기 (drone_boundary.detect_boundary)  2) 상자 찾기 (drone_boxes.detect_boxes)
    3) 지도 만들기 (BoxView.field_map)           4) 거점 벽 그림 읽기 (base_panels.annotate)
    단계마다 걸린 시간을 재고, 결과를 한 줄로 찍고, 사진 위에 결과를 그린 그림을 --out 폴더에 남긴다.
    끝나면 요약 표를 찍고 모든 결과를 --out 폴더의 결과.json 에 적는다.

로봇의 계산과 같은 점 · 다른 점
    map_maker.compute 가 부르는 함수를 같은 순서 · 같은 값으로 부른다 (단계마다 시간을 재려고 compute 를 직접 부르지 않는다).
    --cache 를 주지 않으면 지난 경계를 쓰지 않는다 → 사진마다 경계를 처음부터 찾는다 (새 카메라의 첫 계산과 같다).
    --cache 파일을 주면 로봇처럼 통과한 경계를 그 파일에 남기고 다음 사진은 거기서 출발한다 (로봇의 두 번째 계산부터와 같다).

numpy · OpenCV 만 쓴다. ROS 가 없어도 된다. 어느 폴더에서 돌려도 된다 (perception 꾸러미는 이 파일 자리에서 찾는다).
칸마다 무슨 뜻인지는 같은 폴더의 README.md 에 있다.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import time
import traceback
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.normpath(os.path.join(HERE, "..", "..", "개발", "src", "perception"))  # perception 꾸러미가 든 폴더
sys.path.insert(0, PACKAGE)

ARENA_WIDTH = 6.12  # 지도에 적는 가로 (m) — map_maker 의 map_arena_width 기본값
LINE_NAMES = ("Bs", "Bc", "C", "Rc", "Rs", "top", "bot", "circle")  # 경계가 쓰는 선 묶음: 세로선 다섯 · 줄 경계선 둘 · 가운데 원
SHORT = {"COMMAND_POST": "지휘", "MAINTENANCE": "정비", "SUPPLY_DEPOT": "보급"}
REACH = 1.5  # 벽 밖으로 이만큼(m)까지 화면에 찍혔는지 본다
# 연습 경기장 화면에서 나온 값의 범위 (조금 넉넉히 잡은 것). 이 밖이면 그 줄 끝의 「주의」 에 적는다 — 틀렸다는 뜻이 아니라 그림을 눈으로 보라는 뜻이다.
# 연습 화면 15장에서 잰 값 (2026-11-19): 끝띠 0.79 ~ 0.84, CM 0.61 ~ 0.63, 원 0.63 ~ 0.64, rms 0.7 ~ 0.9 cm, s 1.14 ~ 1.20
USUAL = {"strip": (0.78, 0.90), "cm_half": (0.59, 0.64), "circle_r": (0.60, 0.65), "scale": (1.13, 1.21)}
USUAL_RMS_CM = 1.5
USUAL_SIZE = [1280, 720]


def find_pictures(items):
    """폴더 · 파일 · 경로 무늬 → 사진 경로 목록 (이름 순). 폴더는 그 안의 jpg · jpeg · png 만 본다 (아래 폴더는 보지 않는다)."""
    found = []
    for item in items:
        if os.path.isdir(item):
            names = sorted(n for n in os.listdir(item) if n.lower().endswith((".jpg", ".jpeg", ".png")))
            found += [os.path.join(item, n) for n in names]
        elif os.path.isfile(item):
            found.append(item)
        else:
            matched = sorted(glob.glob(item))
            if not matched:
                print("찾지 못함: %s" % item)
            found += [m for m in matched if os.path.isfile(m)]
    seen, out = set(), []
    for path in found:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def short_names(paths):
    """찍을 때 쓸 이름. 보통은 파일 이름만 쓰고, 같은 이름의 파일이 여럿이면 서로 구별될 때까지 위 폴더 이름을 앞에 붙인다."""
    parts = [os.path.abspath(path).replace(os.sep, "/").split("/") for path in paths]
    names = [p[-1] for p in parts]
    for depth in range(2, 8):
        again = {name for name in names if names.count(name) > 1}
        if not again:
            break
        names = ["/".join(p[-depth:]) if name in again else name for p, name in zip(parts, names)]
    return names


def line_counts(debug):
    """경계가 찾은 선 묶음마다 점 수 → {이름: 수}. debug["lines"] 가 있으면 그것을, 없으면 debug["groups"](이름 → 점들)를 센다."""
    def count(value):
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return int(value)
        if isinstance(value, dict):
            for key in ("points", "count", "n"):
                if key in value:
                    return count(value[key])
            return len(value)
        try:
            return int(len(value))
        except TypeError:
            return 0

    source = debug.get("lines")
    if not isinstance(source, dict) or not source:
        source = debug.get("groups") or {}
    counts = {str(name): count(value) for name, value in source.items()}
    ordered = {name: counts.get(name, 0) for name in LINE_NAMES}
    ordered.update({name: n for name, n in counts.items() if name not in ordered})
    return ordered


def visible_part(boundary, size):
    """화면에 경기장의 어디가 찍혔는가 (경계를 맞춘 좌표: x 는 BLUE 벽 0 → RED 벽, y 는 아래 벽 0 → 위 벽).

    x: 경기장 세로 가운데 높이(y = 세로 / 2)의 가로줄에서 화면에 찍힌 x 범위. y: 경기장 가로 가운데(x = 가로 / 2)의 세로줄에서 찍힌 y 범위.
       0 보다 작거나 경기장 크기보다 크면 벽 밖까지 찍힌 것이고, 그 안쪽 값이면 그만큼 잘린 것이다 (벽 밖 REACH m 까지만 본다).
    walls: 네 벽(B = BLUE 벽 x 0, R = RED 벽, 아래 = y 0, 위)의 바닥선 중 화면 안에 있는 비율 (%).
    strip_lines: 두 끝 띠 선(B 쪽 · R 쪽) 중 화면 안에 있는 비율 (%). frame: 경계를 맞춘 좌표의 크기 (가로, 세로) m.
    """
    w, h = float(boundary.field_w), float(boundary.field_h)
    wide, high = size

    def inside(points):
        with np.errstate(all="ignore"):
            px = boundary.field_to_pixel(np.asarray(points, float))
            ok = np.isfinite(px).all(1)
            ok &= (px[:, 0] >= 0) & (px[:, 0] <= wide - 1) & (px[:, 1] >= 0) & (px[:, 1] <= high - 1)
        return ok

    def run(values, flags, centre):
        """가운데 값을 품은, 끊기지 않고 찍힌 범위."""
        i = int(np.argmin(np.abs(values - centre)))
        if not flags[i]:
            return None
        lo = hi = i
        while lo > 0 and flags[lo - 1]:
            lo -= 1
        while hi < len(values) - 1 and flags[hi + 1]:
            hi += 1
        return [round(float(values[lo]), 2), round(float(values[hi]), 2)]

    xs = np.arange(-REACH, w + REACH + 1e-9, 0.01)
    ys = np.arange(-REACH, h + REACH + 1e-9, 0.01)
    t = np.linspace(0.0, 1.0, 201)
    zero, one = np.zeros_like(t), np.ones_like(t)

    def share(xv, yv):
        return int(round(100.0 * float(inside(np.stack([xv, yv], 1)).mean())))

    out = {
        "x": run(xs, inside(np.stack([xs, np.full_like(xs, h / 2.0)], 1)), w / 2.0),
        "y": run(ys, inside(np.stack([np.full_like(ys, w / 2.0), ys], 1)), h / 2.0),
        "walls": {"B": share(zero, h * t), "R": share(w * one, h * t), "아래": share(w * t, zero), "위": share(w * t, h * one)},
        "strip_lines": None,
        "frame": [w, h],
    }
    if boundary.strip is not None:
        s = float(boundary.strip)
        out["strip_lines"] = {"B": share(s * one, h * t), "R": share((w - s) * one, h * t)}
    return out


def reject_reasons(rejected):
    """버린 후보의 까닭을 세어 {까닭: 수}. 까닭에 든 숫자(좌표 · 점수)는 지워서 같은 종류끼리 모은다."""
    counts = collections.Counter()
    for item in rejected or []:
        reason = str((item or {}).get("reason", "")) if isinstance(item, dict) else str(item)
        reason = re.sub(r"\([^)]*\)", "", reason)
        reason = re.sub(r"[-+]?\d+(\.\d+)?\s*(cm|m|px|%|개)?", "", reason)
        reason = " ".join(reason.replace("×", " ").replace(",", " ").split()).strip(" ·:—-")
        counts[reason or "까닭 없음"] += 1
    return dict(counts.most_common())


def check_one(path, arena, cache, name=None, camera=None):
    """사진 한 장 → (결과 dict, 그림에 쓸 것 (image, boundary, view)). map_maker.compute 와 같은 순서로 부른다. name: 찍을 때 쓸 이름."""
    import cv2

    from perception import base_panels, camera_boundaries, map_maker
    from perception.drone_boundary import detect_boundary
    from perception.drone_boxes import BoxParams, detect_boxes

    result = {"file": name or os.path.basename(path), "path": os.path.abspath(path), "error": None, "image": None, "map_ok": False,
              "boundary": None, "visible": None, "boxes": None, "arena": None, "panels": None, "seconds": {}}
    with open(path, "rb") as fh:
        jpeg = fh.read()
    began = time.monotonic()
    image = cv2.imdecode(np.frombuffer(bytes(jpeg), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        result["error"] = "사진을 풀지 못함"
        return result, (None, None, None)
    size = (int(image.shape[1]), int(image.shape[0]))
    result["image"] = list(size)

    # 1) 경계
    priors = map_maker.load_boundaries(cache)
    shipped = camera_boundaries.built_in(camera)  # 코드에 실린 그 카메라의 경계 (--camera 를 줬을 때만)
    boundary = detect_boundary(image, prior=(priors + shipped) or None)
    result["seconds"]["boundary"] = round(time.monotonic() - began, 2)
    if map_maker.boundary_trusted(boundary):
        map_maker.save_boundary(cache, boundary, priors)
    notes = [str(n) for n in boundary.notes]
    lens = boundary._lens() if boundary.hom is not None else None
    result["boundary"] = {
        "ok": bool(boundary.ok), "found": boundary.hom is not None,
        "via": (boundary.debug.get("via") or "") + ("·줄임" if boundary.debug.get("reduced") else "") + ("·끝띠어림" if boundary.debug.get("strip_derived") else ""),
        "fails": [n for n in notes if n.startswith("실패")], "notes": notes, "lines": line_counts(boundary.debug),
        "strip": None if boundary.strip is None else round(float(boundary.strip), 3),
        "cm_half": None if boundary.cm_half is None else round(float(boundary.cm_half), 3),
        "circle_r": None if boundary.circle_r is None else round(float(boundary.circle_r), 3),
        "rms_cm": None if boundary.rms_m is None or not np.isfinite(boundary.rms_m) else round(float(boundary.rms_m) * 100.0, 2),
        "points": int(boundary.points), "support": boundary.debug.get("support"), "yellow": boundary.debug.get("yellow"),
        "lens": None if lens is None else {"k1": lens.k1, "k2": lens.k2, "aspect": lens.aspect, "cx": lens.cx, "cy": lens.cy},
        "corners_px": None if boundary.hom is None else np.round(boundary.corners_px(), 1).tolist(),
    }
    if boundary.hom is None:
        result["seconds"]["total"] = round(time.monotonic() - began, 2)
        return result, (image, boundary, None)
    try:
        result["visible"] = visible_part(boundary, size)
    except Exception as error:  # noqa: BLE001 — 보이는 범위를 못 구해도 나머지는 계속한다
        result["visible"] = {"error": "%s: %s" % (type(error).__name__, error)}

    # 2) 상자
    at = time.monotonic()
    view = detect_boxes(image, boundary, BoxParams(camera_height=None))
    result["seconds"]["boxes"] = round(time.monotonic() - at, 2)
    scale = float(view.scale)
    result["boxes"] = {
        "ok": bool(view.ok), "count": len(view.boxes), "joined": sum(1 for b in view.boxes if b.joined), "cut": sum(1 for b in view.boxes if b.cut),
        "cut_sides": [b.cut for b in view.boxes if b.cut], "scale": round(scale, 4),
        "camera_height": round(float(view.camera[2]), 3), "height_source": view.height_source, "faces_used": int(view.faces_used),
        "divisor": [round(float(v), 4) for v in view.divisor], "divisor_source": list(view.divisor_source),
        "rejected": reject_reasons(view.rejected), "notes": [str(n) for n in view.notes],
        "positions": [[round(float(b.x), 3), round(float(b.y), 3)] for b in view.boxes],
    }

    # 3) 지도 (로봇은 경계 · 상자 검사를 모두 통과하고 상자가 하나라도 있을 때만 지도를 내보낸다)
    data = view.field_map(boundary, None, "slot", arena=tuple(arena) if arena else None)
    result["arena"] = [data["width"], data["height"]]
    result["map_ok"] = bool(boundary.ok and view.ok and len(view.boxes) > 0)

    # 4) 거점 벽 그림
    at = time.monotonic()
    panels = base_panels.annotate(data, image, boundary, view)
    result["seconds"]["panels"] = round(time.monotonic() - at, 2)
    result["panels"] = panels
    result["bases"] = [b["name"] for b in data.get("bases", [])]
    result["seconds"]["total"] = round(time.monotonic() - began, 2)
    return result, (image, boundary, view)


# ------------------------------------------------------------------ 글로 찍기
def width_of(text):
    """화면에서 차지하는 칸 수 (한글은 두 칸)."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width):
    text = str(text)
    return text + " " * max(0, width - width_of(text))


def number(value, form="%.2f"):
    return "-" if value is None else form % value


def span(pair, size=None):
    """찍힌 범위. 벽 밖 REACH m 넘게 찍혔으면 그 끝은 「밖」 으로 적는다."""
    if not pair:
        return "화면 밖"
    low = "밖" if pair[0] <= -REACH + 0.005 else "%.2f" % pair[0]
    high = "밖" if size is not None and pair[1] >= size + REACH - 0.005 else "%.2f" % pair[1]
    return "%s~%s" % (low, high)


def cautions(r):
    """눈으로 확인할 것 → [글]. 연습 경기장 화면의 값과 다른 것, 화면 밖에 있는 것, 못 찾은 것."""
    out = []
    if r.get("error"):
        return ["오류"]
    if r["image"] != USUAL_SIZE:
        out.append("사진 크기가 %dx%d 이 아님" % tuple(USUAL_SIZE))
    b = r["boundary"]
    if not b["found"]:
        return out + ["경계 없음"]
    if not b["ok"]:
        out.append("경계 검사 실패")
    missing = [name for name in LINE_NAMES if b["lines"].get(name, 0) == 0]
    if missing:
        out.append("못 찾은 선 " + " ".join(missing))
    for key, label in (("strip", "끝띠"), ("cm_half", "CM"), ("circle_r", "원")):
        low, high = USUAL[key]
        if b[key] is None or not low <= b[key] <= high:
            out.append("%s %s (연습 %.2f~%.2f)" % (label, number(b[key], "%.3f"), low, high))
    if b["rms_cm"] is None or b["rms_cm"] > USUAL_RMS_CM:
        out.append("rms %scm (연습 %.1f 이하)" % (number(b["rms_cm"], "%.1f"), USUAL_RMS_CM))
    v = r["visible"] or {}
    if "error" not in v and v:
        w = v["frame"][0]
        if not v["x"] or not v["y"]:
            out.append("경기장 가운데가 화면 밖")
        else:
            if v["x"][0] > 0.0:
                out.append("BLUE 벽이 화면 밖 (x %.2f 부터 보임)" % v["x"][0])
            if v["x"][1] < w:
                out.append("RED 벽이 화면 밖 (x %.2f 까지 보임)" % v["x"][1])
        for side, camp in (("B", "BLUE"), ("R", "RED")):
            if v.get("strip_lines") and v["strip_lines"][side] < 50:
                out.append("%s 쪽 끝 띠 선이 화면에 %d%% 만 있음" % (camp, v["strip_lines"][side]))
    x = r["boxes"]
    if x:
        if x["count"] == 0:
            out.append("상자 0개")
        elif not x["ok"]:
            out.append("상자 검사 실패")
        low, high = USUAL["scale"]
        if not low <= x["scale"] <= high:
            out.append("s %.3f (연습 %.2f~%.2f)" % (x["scale"], low, high))
        if x["height_source"] == "default":
            out.append("카메라 높이를 재지 못해 기본값을 씀")
        if x["cut"]:
            out.append("화면 끝에 걸린 상자 %d개" % x["cut"])
    kinds = (r.get("panels") or {}).get("kinds") or {}
    unread = [camp for camp in ("BLUE", "RED") if not kinds.get(camp)]
    if unread:
        out.append("거점 모름 " + " ".join(unread))
    if not r.get("map_ok"):
        out.append("지도 안 나감")
    return out


def panel_text(panels, camp, long=True):
    if not panels:
        return "-"
    order = (panels.get("kinds") or {}).get(camp)
    detail = panels.get(camp) or {}
    if order:
        text = "·".join(SHORT.get(k, k) for k in order)
        return text + (" (차 %s)" % number(detail.get("gap")) if long else "")
    why = detail.get("note") or panels.get("note") or ""
    return "모름" + (" (%s)" % why if long and why else "")


def one_line(r):
    """사진 한 장의 결과 → 한 줄."""
    parts = [r["file"]]
    if r.get("error"):
        return " | ".join(parts + ["오류: %s" % r["error"]])
    parts.append("%dx%d" % tuple(r["image"]))
    b = r["boundary"]
    if b["ok"]:
        state = "경계 OK"
    elif b["found"]:
        state = "경계 실패 [%s]" % "; ".join(n[len("실패"):].strip() for n in b["fails"])
    else:
        state = "경계 없음 [%s]" % "; ".join(b["notes"][:3])
    parts.append("%s (%s)" % (state, b["via"] or "-"))
    parts.append("선 " + " ".join("%s%d" % (name, n) for name, n in b["lines"].items()))
    if not b["found"]:
        parts.append("초 경계 %.2f 합 %.2f" % (r["seconds"].get("boundary", 0.0), r["seconds"].get("total", 0.0)))
        parts.append("주의: " + ("; ".join(r.get("cautions") or []) or "없음"))
        return " | ".join(parts)
    parts.append("끝띠 %s CM %s 원 %s rms %scm" % (number(b["strip"], "%.3f"), number(b["cm_half"], "%.3f"), number(b["circle_r"], "%.3f"), number(b["rms_cm"], "%.1f")))
    v = r["visible"] or {}
    if "error" in v:
        parts.append("보임 계산 오류 %s" % v["error"])
    else:
        walls = " ".join("%s%d" % kv for kv in v["walls"].items())
        strips = "" if not v.get("strip_lines") else " · 끝띠선 " + " ".join("%s%d" % kv for kv in v["strip_lines"].items()) + " %"
        parts.append("보임 x %s y %s · 벽 %s %%%s" % (span(v["x"], v["frame"][0]), span(v["y"], v["frame"][1]), walls, strips))
    x = r["boxes"]
    rejected = ", ".join("%s %d" % kv for kv in x["rejected"].items()) or "없음"
    parts.append("상자 %d%s (붙은 %d, 잘린 %d) s %.3f (카메라 %.2f m, %s, 옆면 %d) 나눗값 %.3f/%.3f (%s/%s) 버림 %s" % (
        x["count"], "" if x["ok"] else " 검사 실패", x["joined"], x["cut"], x["scale"], x["camera_height"], x["height_source"] or "-", x["faces_used"],
        x["divisor"][0], x["divisor"][1], x["divisor_source"][0] or "-", x["divisor_source"][1] or "-", rejected))
    parts.append("경기장 %.2fx%.2f" % tuple(r["arena"]))
    parts.append("거점 BLUE %s · RED %s" % (panel_text(r["panels"], "BLUE"), panel_text(r["panels"], "RED")))
    parts.append("지도 %s" % ("나감" if r["map_ok"] else "안 나감"))
    s = r["seconds"]
    parts.append("초 경계 %.2f 상자 %.2f 벽그림 %.2f 합 %.2f" % (s.get("boundary", 0.0), s.get("boxes", 0.0), s.get("panels", 0.0), s.get("total", 0.0)))
    parts.append("주의: " + ("; ".join(r.get("cautions") or []) or "없음"))
    return " | ".join(parts)


def summary_table(results):
    heads = ["파일", "크기", "경계", "방법", "선", "끝띠", "CM", "원", "rms", "보임 x", "보임 y", "상자", "s", "BLUE", "RED", "지도", "초", "주의"]
    rows = []
    for r in results:
        if r.get("error"):
            rows.append([r["file"], "-", "오류", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "오류"])
            continue
        b, v, x = r["boundary"], r["visible"] or {}, r["boxes"]
        lines = "%d/%d" % (sum(1 for name in LINE_NAMES if b["lines"].get(name, 0) > 0), len(LINE_NAMES))
        row = [r["file"], "%dx%d" % tuple(r["image"]), "OK" if b["ok"] else ("실패" if b["found"] else "없음"), b["via"] or "-", lines,
               number(b["strip"], "%.3f"), number(b["cm_half"], "%.3f"), number(b["circle_r"], "%.3f"), number(b["rms_cm"], "%.1f"),
               span(v.get("x"), v["frame"][0]) if v.get("frame") else "-", span(v.get("y"), v["frame"][1]) if v.get("frame") else "-"]
        if x is None:
            row += ["-", "-", "-", "-", "안 나감", "%.2f" % r["seconds"].get("total", 0.0)]
        else:
            row += [str(x["count"]), "%.3f" % x["scale"], panel_text(r["panels"], "BLUE", False), panel_text(r["panels"], "RED", False),
                    "나감" if r["map_ok"] else "안 나감", "%.2f" % r["seconds"].get("total", 0.0)]
        row.append("%d가지" % len(r.get("cautions") or []) if r.get("cautions") else "없음")
        rows.append(row)
    widths = [max(width_of(str(row[i])) for row in [heads] + rows) for i in range(len(heads))]
    out = ["  ".join(pad(h, w) for h, w in zip(heads, widths)), "  ".join("-" * w for w in widths)]
    out += ["  ".join(pad(c, w) for c, w in zip(row, widths)) for row in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ 그림
def overlay(image, boundary, view, result):
    """사진 위에 맞춘 선 · 경기장 테두리 · 상자를 그린 그림과, 위에서 본 그림을 옆으로 붙인 한 장 (BGR).

    왼쪽: 주황 = 맞춘 구역선 · 원, 초록(검사 통과) 또는 빨강(실패) 굵은 선 = 경기장 바닥 테두리, 분홍 점 = 경계가 쓴 흰 선 점,
          초록 네모 = 상자 바닥, 노란 네모 = 상자 윗면, 빨간 X = 버린 후보, 하늘색 + = 카메라 바로 아래 점.
    오른쪽: 같은 것을 위에서 본 그림 (1 m = 100 px, 보라색 = 화면에 찍히지 않은 자리).
    """
    import cv2

    from perception.drone_boundary import draw_before_after
    from perception.drone_boxes import draw_boxes

    left, right = image.copy(), None
    if boundary is not None and boundary.hom is not None:
        try:
            left, right = draw_before_after(image, boundary)
        except Exception as error:  # noqa: BLE001 — 선을 못 그려도 상자는 그린다
            print("  그림: 경계 선을 그리지 못함 (%s: %s)" % (type(error).__name__, error))
        if view is not None:
            try:
                left, right = draw_boxes(left, boundary, view)
            except Exception as error:  # noqa: BLE001
                print("  그림: 상자를 그리지 못함 (%s: %s)" % (type(error).__name__, error))
    # 맨 위에 짧은 글 (OpenCV 는 한글을 못 그려 영문으로 적는다)
    b, x = result.get("boundary") or {}, result.get("boxes") or {}
    kinds = (result.get("panels") or {}).get("kinds") or {}
    letters = {"COMMAND_POST": "C", "MAINTENANCE": "M", "SUPPLY_DEPOT": "S"}
    camps = " ".join("%s %s" % (camp, "-".join(letters.get(k, "?") for k in kinds[camp]) if kinds.get(camp) else "none") for camp in ("BLUE", "RED"))
    text = "%s  boundary %s (%s)  boxes %s  s %s  panels %s  map %s" % (
        result["file"], "OK" if b.get("ok") else ("FAIL" if b.get("found") else "NONE"), b.get("via") or "-", x.get("count", "-"),
        x.get("scale", "-"), camps, "yes" if result.get("map_ok") else "no")
    if right is not None:
        height = left.shape[0]
        wide = max(1, int(round(right.shape[1] * height / float(right.shape[0]))))
        left = np.hstack([left, cv2.resize(right, (wide, height), interpolation=cv2.INTER_AREA)])
    bar = np.zeros((26, left.shape[1], 3), np.uint8)
    cv2.putText(bar, text.encode("ascii", "replace").decode("ascii"), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, left])


def write_jpeg(path, picture):
    """OpenCV 의 imwrite 는 윈도에서 한글 경로를 쓰지 못한다 → 메모리에서 JPEG 로 만든 뒤 파이썬으로 적는다."""
    import cv2

    done, coded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not done:
        raise ValueError("JPEG 로 만들지 못함")
    with open(path, "wb") as fh:
        fh.write(coded.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description="드론 사진을 지도 계산에 넣어 보고 사진마다 한 줄로 결과를 찍는다.")
    parser.add_argument("pictures", nargs="+", help="사진이 든 폴더, 또는 jpg 파일들")
    parser.add_argument("--out", default=None, help="그림과 결과.json 을 둘 폴더 (기본: 첫 사진이 있는 폴더 아래의 점검_결과)")
    parser.add_argument("--arena-height", type=float, default=5.0, help="지도에 적을 경기장 세로 m (본선 5.0, 연습 경기장 사진은 4.8)")
    parser.add_argument("--cache", default=None, help="경계 저장 파일. 주면 통과한 경계를 남기고 다음 사진은 거기서 출발한다 (로봇과 같다)")
    parser.add_argument("--camera", default=None, help="카메라 이름 (main · test1 · test2). 주면 코드에 실린 그 카메라의 경계(camera_boundaries.BUILT_IN)에서 출발한다. "
                        "주지 않으면 쓰지 않는다 (처음 보는 카메라와 같다)")
    args = parser.parse_args(argv)

    paths = find_pictures(args.pictures)
    if not paths:
        print("사진이 없음")
        return 1
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(paths[0])), "점검_결과")
    os.makedirs(out_dir, exist_ok=True)
    arena = (ARENA_WIDTH, float(args.arena_height))
    print("사진 %d장 · 경기장 %.2f x %.2f m · 지난 경계 %s · 결과 폴더 %s" % (len(paths), arena[0], arena[1], args.cache or "쓰지 않음", out_dir))
    print("칸: 파일 | 크기 | 경계 (방법) | 선 묶음마다 점 수 | 끝띠 CM 원 rms | 보임 | 상자 | 경기장 | 거점 | 지도 | 초 | 주의  — 뜻은 README.md")
    results = []
    names = short_names(paths)
    for index, path in enumerate(paths):
        result = {"file": names[index], "path": os.path.abspath(path), "error": None}
        parts = (None, None, None)
        try:
            result, parts = check_one(path, arena, args.cache, names[index], args.camera)
        except Exception as error:  # noqa: BLE001 — 사진 한 장에서 오류가 나도 다음 사진으로 간다
            result["error"] = "%s: %s" % (type(error).__name__, error)
            result["traceback"] = traceback.format_exc()
        try:
            result["cautions"] = cautions(result)
        except Exception as error:  # noqa: BLE001
            result["cautions"] = ["주의를 뽑는 중 오류 %s: %s" % (type(error).__name__, error)]
        try:
            print(one_line(result), flush=True)
        except Exception as error:  # noqa: BLE001
            print("%s | 결과를 찍는 중 오류 %s: %s" % (result.get("file"), type(error).__name__, error), flush=True)
        if result.get("traceback"):
            print(result["traceback"], flush=True)
        if parts[0] is not None:
            try:
                name = "%02d_%s_점검.jpg" % (index + 1, os.path.splitext(names[index])[0].replace(os.sep, "_").replace("/", "_"))
                write_jpeg(os.path.join(out_dir, name), overlay(parts[0], parts[1], parts[2], result))
                result["overlay"] = name
            except Exception as error:  # noqa: BLE001
                print("  그림을 남기지 못함 (%s: %s)" % (type(error).__name__, error), flush=True)
        results.append(result)

    print()
    try:
        print(summary_table(results))
    except Exception as error:  # noqa: BLE001
        print("요약 표를 만드는 중 오류 %s: %s" % (type(error).__name__, error))
    record = {"arena": list(arena), "cache": args.cache, "made_at": time.strftime("%Y-%m-%d %H:%M:%S"), "pictures": results}
    with open(os.path.join(out_dir, "결과.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1, default=str)
    print()
    print("그림 · 결과.json: %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
