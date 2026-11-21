#!/usr/bin/env python3
"""학습한 모델 평가 — 경기장 좌표(cm)로 재고, 조명을 바꿔 가며(--stress) 얼마나 버티는지 본다.

  python evaluate.py --weights runs/<이름>/weights/best.pt --data <데이터셋> [--split val] [--out <폴더>]

재는 것 (클래스마다 · 전체)
  - 맞춤: 예측 상자 가운데가 같은 종류 정답 가운데에서 --match-cm (기본 10 cm) 안이면 맞음 (점수 높은 예측부터 가까운 정답 하나씩)
  - precision · recall · F1, 맞은 것의 가운데 오차 (평균 · 중앙 · 90 % · 최대, cm), 방향 오차 (도, 정사각형이라 90° 마다 같음)
  - 좌표: fieldmaps/<split>/<이름>.json 의 out_to_field (전처리 calib 의 픽셀 → 경기장 행렬)로 옮긴다.
    없으면 map 그림이라고 보고 x = u/ppm, y = 그림 높이/ppm − v/ppm (--px-per-m, 기본 100)
  - --stress: photometric.STRESS 의 조건(어둡게 · 밝게 · 한쪽 어둡게 · 그림자 · 반사 · 연막 · 색온도 · 대비 · 흐림)마다
    검증 그림을 바꿔서 다시 잰다 → 조건별 recall · precision 표. 「원본」 과 차이가 크면 그 조명에 약한 것

출력: <out>/eval.json · eval.md (표) · vis/ (--vis N: 틀린 것이 많은 그림 N 장에 정답 초록 · 예측 빨강)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import photometric  # noqa: E402
from augment import IMG_EXT, dataset_names, read_image, write_image  # noqa: E402


# ------------------------------------------------------------------ 기하
def to_field(m, pts):
    out = []
    for u, v in pts:
        w = m[2][0] * u + m[2][1] * v + m[2][2]
        out.append(((m[0][0] * u + m[0][1] * v + m[0][2]) / w, (m[1][0] * u + m[1][1] * v + m[1][2]) / w))
    return out


def center(pts):
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def yaw_deg(pts):
    """정사각형 방향 (−45 ~ 45°). 꼭짓점 순서 · 시작점에 상관없게 네 변 각을 90° 로 접어 평균."""
    angs = []
    for i in range(4):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % 4]
        a = math.atan2(y1 - y0, x1 - x0) * 4.0  # 90° 주기 → 4배 하면 같은 각
        angs.append((math.cos(a), math.sin(a)))
    c, s = sum(a[0] for a in angs), sum(a[1] for a in angs)
    return math.degrees(math.atan2(s, c)) / 4.0


def yaw_diff(a, b):
    d = (a - b + 45.0) % 90.0 - 45.0
    return abs(d)


# ------------------------------------------------------------------ 자료
class Item:
    def __init__(self, ds, split, fname, ppm):
        self.path = os.path.join(ds, "images", split, fname)
        self.stem = os.path.splitext(fname)[0]
        self.lab = os.path.join(ds, "labels", split, self.stem + ".txt")
        fm_path = os.path.join(ds, "fieldmaps", split, self.stem + ".json")
        self.fm = None
        if os.path.exists(fm_path):
            with open(fm_path, encoding="utf-8") as fh:
                self.fm = json.load(fh)
        self.ppm = ppm

    def o2f(self, h):
        if self.fm and self.fm.get("out_to_field"):
            return self.fm["out_to_field"]
        return [[1.0 / self.ppm, 0, 0], [0, -1.0 / self.ppm, h / self.ppm], [0, 0, 1]]

    def gt(self, w, h):
        """[(클래스 번호, 꼭짓점 픽셀 4개)]"""
        out = []
        if not os.path.exists(self.lab):
            return out
        with open(self.lab, encoding="utf-8") as fh:
            for line in fh:
                v = line.split()
                if len(v) != 9:
                    continue
                c = [float(x) for x in v[1:]]
                out.append((int(v[0]), [(c[i] * w, c[i + 1] * h) for i in range(0, 8, 2)]))
        return out


def match(gts, preds, o2f, match_m):
    """gts: [(cls, pts)], preds: [(cls, conf, pts)] (픽셀) → (tp, fp, fn, [(cls, 가운데 오차 m, 방향 오차 °)], 틀린 예측 · 놓친 정답)."""
    G = [(c, to_field(o2f, p)) for c, p in gts]
    P = sorted([(c, s, to_field(o2f, p)) for c, s, p in preds], key=lambda t: -t[1])
    used = set()
    hits, fps = [], []
    for c, s, pf in P:
        pc = center(pf)
        best, bd = None, match_m
        for j, (gc, gf) in enumerate(G):
            if j in used or gc != c:
                continue
            d = math.dist(pc, center(gf))
            if d <= bd:
                best, bd = j, d
        if best is None:
            fps.append((c, pc))
        else:
            used.add(best)
            hits.append((c, bd, yaw_diff(yaw_deg(pf), yaw_deg(G[best][1]))))
    misses = [(gc, center(gf)) for j, (gc, gf) in enumerate(G) if j not in used]
    return hits, fps, misses


def summarize(rows, names):
    """rows: 그림마다 (hits, fps, misses) → 클래스별 · 전체 요약 dict."""
    out = {}
    for key, cls in [("전체", None)] + [(n, i) for i, n in enumerate(names)]:
        tp = sum(1 for h, _f, _m in rows for c, *_ in h if cls is None or c == cls)
        fp = sum(1 for _h, f, _m in rows for c, _ in f if cls is None or c == cls)
        fn = sum(1 for _h, _f, m in rows for c, _ in m if cls is None or c == cls)
        errs = sorted(e * 100.0 for h, _f, _m in rows for c, e, _y in h if cls is None or c == cls)
        yaws = [y for h, _f, _m in rows for c, _e, y in h if cls is None or c == cls]
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p and r else None
        out[key] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
                    "err_cm_mean": statistics.fmean(errs) if errs else None, "err_cm_median": statistics.median(errs) if errs else None,
                    "err_cm_p90": errs[min(len(errs) - 1, int(0.9 * len(errs)))] if errs else None, "err_cm_max": errs[-1] if errs else None,
                    "yaw_deg_mean": statistics.fmean(yaws) if yaws else None}
    return out


# ------------------------------------------------------------------ 평가
def evaluate(predict, ds, split="val", names=None, match_cm=10.0, ppm=100.0, stress=True, vis=0, out=None, seed=0, log=print):
    """predict(bgr) → [(cls, conf, 꼭짓점 픽셀 4개)]. 모델과 떨어져 있어 시험에서 가짜 예측기를 넣을 수 있다."""
    import numpy as np

    names = names or dataset_names(ds)
    folder = os.path.join(ds, "images", split)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)) if os.path.isdir(folder) else []
    if not files:
        raise SystemExit("평가할 그림이 없습니다: %s" % folder)
    items = [Item(ds, split, f, ppm) for f in files]
    conds = {"원본": None}
    if stress:
        conds.update(photometric.STRESS)
    result = {"split": split, "images": len(items), "match_cm": match_cm, "conditions": {}, "per_image": {}}
    worst = []
    for cname, ops in conds.items():
        rows = []
        for i, it in enumerate(items):
            img = read_image(it.path)
            h, w = img.shape[:2]
            if ops:
                img, _ = photometric.apply(img, ops, np.random.default_rng(seed + i), force=True)
            preds = predict(img)
            gts = it.gt(w, h)
            hits, fps, misses = match(gts, preds, it.o2f(h), match_cm / 100.0)
            rows.append((hits, fps, misses))
            if ops is None:
                result["per_image"][it.stem] = {"tp": len(hits), "fp": len(fps), "fn": len(misses)}
                worst.append((len(fps) + len(misses), it, img, gts, preds))
        result["conditions"][cname] = summarize(rows, names)
        s = result["conditions"][cname]["전체"]
        log("%-14s recall %s · precision %s · 가운데 오차 중앙 %s cm" % (cname, _pct(s["recall"]), _pct(s["precision"]), _num(s["err_cm_median"])))
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        with open(os.path.join(out, "eval.md"), "w", encoding="utf-8") as fh:
            fh.write(report_md(result, names))
        if vis:
            draw_worst(worst, vis, os.path.join(out, "vis"))
    return result


def _pct(v):
    return "—" if v is None else "%.1f%%" % (100 * v)


def _num(v):
    return "—" if v is None else "%.1f" % v


def report_md(result, names):
    lines = ["# 상자 찾기 평가 (%s, %d 장, 맞춤 %g cm 안)" % (result["split"], result["images"], result["match_cm"]), "",
             "## 조건별 (전체)", "", "| 조건 | recall | precision | F1 | 가운데 오차 중앙 / 90%% (cm) | 놓침 | 헛짚음 |".replace("%%", "%"),
             "|---|---|---|---|---|---|---|"]
    base = result["conditions"]["원본"]["전체"]
    for cname, per in result["conditions"].items():
        s = per["전체"]
        drop = "" if cname == "원본" or s["recall"] is None or base["recall"] is None else " (%+.1f)" % (100 * (s["recall"] - base["recall"]))
        lines.append("| %s | %s%s | %s | %s | %s / %s | %d | %d |" % (cname, _pct(s["recall"]), drop, _pct(s["precision"]), _pct(s["f1"]),
                                                                    _num(s["err_cm_median"]), _num(s["err_cm_p90"]), s["fn"], s["fp"]))
    lines += ["", "## 클래스별 (원본)", "", "| 클래스 | 정답 | recall | precision | 가운데 오차 평균 / 최대 (cm) | 방향 오차 평균 (°) |", "|---|---|---|---|---|---|"]
    for n in names:
        s = result["conditions"]["원본"].get(n)
        if s:
            lines.append("| %s | %d | %s | %s | %s / %s | %s |" % (n, s["tp"] + s["fn"], _pct(s["recall"]), _pct(s["precision"]),
                                                                  _num(s["err_cm_mean"]), _num(s["err_cm_max"]), _num(s["yaw_deg_mean"])))
    return "\n".join(lines) + "\n"


def draw_worst(worst, n, folder):
    import cv2
    import numpy as np

    os.makedirs(folder, exist_ok=True)
    for bad, it, img, gts, preds in sorted(worst, key=lambda t: -t[0])[:n]:
        canvas = img.copy()
        for _c, pts in gts:
            cv2.polylines(canvas, [np.round(np.array(pts)).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        for _c, conf, pts in preds:
            cv2.polylines(canvas, [np.round(np.array(pts)).astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "%.2f" % conf, tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        write_image(os.path.join(folder, "%02d_%s.jpg" % (bad, it.stem)), canvas, 90)


def yolo_predictor(weights, imgsz=640, conf=0.25, device=None):
    """Ultralytics OBB 모델 → predict(bgr) 함수."""
    from ultralytics import YOLO

    model = YOLO(weights, task="obb")

    def predict(bgr):
        kw = {"imgsz": imgsz, "conf": conf, "verbose": False}
        if device is not None:
            kw["device"] = device
        r = model.predict(bgr, **kw)[0]
        if r.obb is None or len(r.obb) == 0:
            return []
        pts = r.obb.xyxyxyxy.cpu().numpy()
        return [(int(c), float(s), [tuple(map(float, p)) for p in q]) for c, s, q in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), pts)]

    return predict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="학습한 가중치 (.pt · .onnx · .engine)")
    ap.add_argument("--data", required=True, help="데이터셋 폴더 (images/<split> · labels/<split> · fieldmaps/<split>)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본 <가중치 폴더>/../eval_<split>)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="이 점수 아래 예측은 버린다")
    ap.add_argument("--match-cm", type=float, default=10.0)
    ap.add_argument("--px-per-m", type=float, default=100.0, help="fieldmaps 가 없을 때 그림 축척")
    ap.add_argument("--no-stress", action="store_true", help="조명 조건별 평가를 하지 않는다")
    ap.add_argument("--vis", type=int, default=12, help="틀린 것이 많은 그림 N 장을 그려 남긴다")
    ap.add_argument("--device", default=None, help="예: 0 · cpu")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.weights))), "eval_" + a.split)
    predict = yolo_predictor(a.weights, a.imgsz, a.conf, a.device)
    evaluate(predict, a.data, a.split, match_cm=a.match_cm, ppm=a.px_per_m, stress=not a.no_stress, vis=a.vis, out=out)
    print("결과: %s (eval.md · eval.json)" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
