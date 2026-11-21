#!/usr/bin/env python3
"""라벨(웹 화면에서 단 것) → YOLO OBB 데이터셋.

  python export_drone_labels.py ../data/drone_sessions/20261120_101500 ../data/drone_sessions/20261120_140000   (기본 --variant undist_map)
  (웹 화면의 「내보내기」 도 이것을 부른다)

출력 (--out)
  images/{train,val}/<세션>__<camera>__<이름>.jpg     그림 복사
  labels/{train,val}/<같은 이름>.txt                  한 줄에 박스 하나: 「클래스 x1 y1 x2 y2 x3 y3 x4 y4」 (0~1 로 나눈 값)
  data.yaml                                          Ultralytics 형식 (path · train · val · names)
  fieldmaps/{train,val}/<같은 이름>.json             그 그림의 상자를 경기장 좌표(frame "arena", m)로 — 전처리 calib.json 의
                                                     out_to_field 로 옮긴다. undist_map 이면 plane_h = 0.40 이라 x · y 가 곧
                                                     지도(FieldMap boxes)의 상자 가운데다. 다른 그림 종류는 plane_h 평면의 좌표다.
  export.json                                        세션별 · 나눔별 장수, 클래스 목록, 만든 시각

status 가 done 인 라벨만 쓴다 (박스 0개 done = 박스 없는 그림으로 들어간다). draft · skip 은 뺀다.

검증 나누기 — 같은 배치의 연속 장이 학습 · 검증에 섞이면 점수가 부풀려진다.
  done 라벨이 있는 세션이 둘 이상이면 **세션 통째로** 검증에 넣는다 (이름 순으로 뒤에서부터, 장수가 val_ratio 를 넘을 때까지.
  넣으면 목표의 1.5 배를 넘는 큰 세션은 건너뛴다. 맨 앞 세션은 늘 학습).
  하나뿐이면 그 세션 안에서 찍힌 시각 순으로 뒤쪽 val_ratio 만큼을 검증으로.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys

KST = dt.timezone(dt.timedelta(hours=9))
DEFAULT_CLASSES = ["box", "stripe"]


def _labels(spath, variant):
    """그 세션 · 그림 종류의 done 라벨들 [(라벨 dict, 그림 경로)] — 그림 이름 순 (= 찍힌 시각 순)."""
    root = os.path.join(spath, "labels", variant)
    img_root = spath if variant == "raw" else os.path.join(spath, variant)
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                    lab = json.load(fh)
            except (OSError, ValueError):
                continue
            if lab.get("status") != "done" or not lab.get("image"):
                continue
            img = os.path.join(img_root, lab["image"])
            if os.path.isfile(img):
                out.append((lab, img))
    out.sort(key=lambda t: os.path.basename(t[0]["image"]))
    return out


def _yolo_lines(lab, classes):
    w, h = float(lab.get("width") or 0), float(lab.get("height") or 0)
    if w <= 0 or h <= 0:
        raise ValueError("그림 크기가 없음: %s" % lab.get("image"))
    lines = []
    for b in lab.get("boxes", []):
        if b.get("cls") not in classes:
            continue
        coords = []
        for u, v in b["pts"]:
            coords += [min(max(u / w, 0.0), 1.0), min(max(v / h, 0.0), 1.0)]
        lines.append("%d %s" % (classes.index(b["cls"]), " ".join("%.6f" % c for c in coords)))
    return lines


def _calib_lookup(spath, variant):
    """그림 파일 → 그 그림을 만든 calib 항목 (전처리 출력의 index.jsonl · calib.json). 원본이거나 없으면 빈 dict."""
    if variant == "raw":
        return {}
    root = os.path.join(spath, variant)
    try:
        with open(os.path.join(root, "calib.json"), encoding="utf-8") as fh:
            entries = {e["id"]: e for e in json.load(fh).get("entries", [])}
    except (OSError, ValueError):
        return {}
    out = {}
    try:
        with open(os.path.join(root, "index.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("calib") in entries:
                    out[row["file"]] = entries[row["calib"]]
    except OSError:
        pass
    return out


def field_boxes(lab, entry, classes):
    """라벨 박스들 → 경기장 좌표 [{cls, x, y, yaw_deg, side_m, corners}] (entry 의 out_to_field 로)."""
    m = entry["out_to_field"]
    out = []
    for b in lab.get("boxes", []):
        if b.get("cls") not in classes:
            continue
        pts = []
        for u, v in b["pts"]:
            w = m[2][0] * u + m[2][1] * v + m[2][2]
            pts.append(((m[0][0] * u + m[0][1] * v + m[0][2]) / w, (m[1][0] * u + m[1][1] * v + m[1][2]) / w))
        cx, cy = sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0
        sides = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        # 정사각형의 방향은 90° 마다 같다 → 첫 변의 각을 −45 ~ 45° 로
        yaw = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
        yaw = (yaw + 45.0) % 90.0 - 45.0
        out.append({"cls": b["cls"], "x": round(cx, 4), "y": round(cy, 4), "yaw_deg": round(yaw, 2),
                    "side_m": round(sum(sides) / 4.0, 4), "corners": [[round(x, 4), round(y, 4)] for x, y in pts]})
    return out


def split(groups, val_ratio):
    """groups: [(세션 id, [항목...])] → {세션 id: ["train" | "val", ...] 항목마다}."""
    total = sum(len(items) for _, items in groups)
    want = total * val_ratio
    assign = {sid: ["train"] * len(items) for sid, items in groups}
    nonempty = [(sid, items) for sid, items in groups if items]
    if val_ratio <= 0 or total == 0:
        return assign
    if len(nonempty) >= 2:
        got = 0
        for sid, items in sorted(nonempty, key=lambda t: t[0])[::-1][:-1]:  # 맨 앞 세션 하나는 늘 학습
            if got >= want:
                break
            if got + len(items) > max(want * 1.5, 1):  # 큰 세션 하나가 검증을 다 차지하지 않게 건너뛴다
                continue
            assign[sid] = ["val"] * len(items)
            got += len(items)
        if got == 0:  # 다 너무 크면 가장 작은 세션 하나 (맨 앞 제외)
            sid, items = min(sorted(nonempty, key=lambda t: t[0])[1:], key=lambda t: len(t[1]))
            assign[sid] = ["val"] * len(items)
    else:
        sid, items = nonempty[0]
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) >= 2 else 0
        assign[sid] = ["train"] * (len(items) - n_val) + ["val"] * n_val
    return assign


def export(sessions, variant, out, classes=None, val_ratio=0.2):
    """sessions: [(세션 id, 폴더)]. 돌려주는 값: 요약 dict (export.json 과 같다)."""
    classes = list(classes or DEFAULT_CLASSES)
    if os.path.exists(out) and os.listdir(out):
        raise ValueError("출력 폴더가 비어 있지 않습니다: %s" % out)
    groups = [(sid, _labels(spath, variant)) for sid, spath in sessions]
    assign = split(groups, val_ratio)
    for part in ("train", "val"):
        for sub in ("images", "labels", "fieldmaps"):
            os.makedirs(os.path.join(out, sub, part), exist_ok=True)
    summary = {"variant": variant, "classes": classes, "created": dt.datetime.now(KST).isoformat(timespec="seconds"),
               "val_ratio": val_ratio, "sessions": {}, "counts": {"train": 0, "val": 0}, "boxes": {c: 0 for c in classes}, "errors": [],
               "fieldmaps": 0, "plane_h": []}
    planes = set()
    paths = dict(sessions)
    for sid, items in groups:
        calib = _calib_lookup(paths[sid], variant)
        per = {"train": 0, "val": 0}
        safe_sid = sid.lstrip("@").replace(os.sep, "_").replace("/", "_")
        for (lab, img), part in zip(items, assign[sid]):
            try:
                lines = _yolo_lines(lab, classes)
            except ValueError as e:
                summary["errors"].append(str(e))
                continue
            stem = "%s__%s" % (safe_sid, lab["image"].replace("/", "__").replace("\\", "__"))
            stem, ext = os.path.splitext(stem)
            shutil.copyfile(img, os.path.join(out, "images", part, stem + ext))
            with open(os.path.join(out, "labels", part, stem + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            entry = calib.get(lab["image"])
            if entry is not None:
                b0 = entry.get("boundary") or {}
                fm = {"frame_id": "arena", "width": b0.get("field_w"), "height": b0.get("field_h"), "box_size": 0.40,
                      "plane_h": entry.get("plane_h", 0.0), "camera_xyh": entry.get("camera_xyh"), "calib": entry["id"],
                      "out_to_field": entry["out_to_field"],  # 그림 픽셀 (u, v, 1) → 경기장 (x, y) — 학습 모델 평가가 쓴다
                      "session": sid, "image": lab["image"], "boxes": field_boxes(lab, entry, classes)}
                with open(os.path.join(out, "fieldmaps", part, stem + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(fm, fh, ensure_ascii=False, indent=1)
                planes.add(fm["plane_h"])
                summary["fieldmaps"] += 1
            for b in lab.get("boxes", []):
                if b.get("cls") in summary["boxes"]:
                    summary["boxes"][b["cls"]] += 1
            per[part] += 1
            summary["counts"][part] += 1
        summary["sessions"][sid] = per
    summary["plane_h"] = sorted(planes)
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# export_drone_labels.py 가 만듦 — Ultralytics OBB\n")
        fh.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % os.path.abspath(out).replace("\\", "/"))
        for i, c in enumerate(classes):
            fh.write("  %d: %s\n" % (i, c))
    with open(os.path.join(out, "export.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    summary["out"] = os.path.abspath(out)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="세션 폴더들")
    ap.add_argument("--variant", default="undist_map", help="그림 종류: undist_map (기본, 지도 좌표) · undist_top · undist_lens")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 ../data/drone_datasets/<시각>_<variant>)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="클래스 순서 (웹 화면 label_drone_frames.CLASSES 와 같게)")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(os.path.dirname(here), "data", "drone_datasets", dt.datetime.now(KST).strftime("%Y%m%d_%H%M%S") + "_" + a.variant)
    sessions = [(os.path.basename(os.path.abspath(p)), p) for p in a.sessions]
    res = export(sessions, a.variant, out, a.classes.split(","), a.val_ratio)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["counts"]["train"] + res["counts"]["val"] else 1


if __name__ == "__main__":
    sys.exit(main())
