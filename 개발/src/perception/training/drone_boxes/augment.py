#!/usr/bin/env python3
"""조명 강건성용 학습 데이터 만들기 — 내보낸 데이터셋(export_drone_labels.py 출력)의 학습 그림마다 조명 · 색 · 화질 변형을 여러 장 만든다.

  python augment.py <데이터셋> [--out <새 데이터셋>] [--config config/augment.yaml] [--copies 4]
  python augment.py <데이터셋> --preview 6        # 변형 예시 그림(preview.jpg)만 만든다 — 범위가 너무 세거나 약하지 않은지 눈으로 본다

  - 학습(train) 그림: 원본 + 변형 copies 장 (이름 <원래 이름>__aug<k>.jpg, 라벨은 원본 것을 그대로 — 기하를 건드리지 않으니까)
  - 보이지 않는 라벨 빼기 (drop_invisible, 기본 0.05): 라벨 안에서 그림이 찍힌 픽셀(시야 밖 검정이 아닌 곳)이 이 비율보다 적으면
    학습 · 검증 라벨에서 뺀다. 2026-11-19 라벨: 줄무늬 블록 589 개 중 284 개가 map 그림의 시야 밖(검정)에 있었다 — 보이지 않는 것을
    찾으라고 가르치면 검정 띠에 헛것을 찍는다. 라벨 파일(웹 화면)은 그대로 두고 여기서만 뺀다. 일부만 보이는 것은 남긴다
  - 시야 끝 자르기 (edge_cut, 설정 edge_cut:): 변형 그림에 확률 p 로 카메라 시야 끝(검정)을 안쪽으로 밀어 상자를 가로지르게 한다
    (edge_cut.py). 일부만 보이는 상자도 전체 라벨로 찾게 가르친다 — 본선 화면 위 · 아래 끝에서 잘린 상자를 놓쳤다 (2026-11-19)
  - --cut-val <폴더>: 검증 그림에 시야 끝 자르기만 건(조명 그대로) 평가용 데이터셋을 만든다 → evaluate.py --data <폴더> --no-stress
  - 검증(val) 그림: 손대지 않는다 (점수가 실제 화면 기준이 되도록. 조명에 대한 강건성은 evaluate.py --stress 로 따로 잰다)
  - fieldmaps/ (경기장 좌표 정답)도 그대로 복사한다
  - 출력의 data.yaml 은 출력 폴더 기준으로 다시 쓰고, augment.json 에 설정 · 장수 · 장마다 건 변형을 남긴다

numpy · OpenCV · PyYAML 만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import edge_cut  # noqa: E402
import photometric  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png")


def load_yaml(path):
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_image(path):
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)  # 한글 경로
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path, img, quality=95):
    import cv2

    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError("그림 저장 실패: %s" % path)
    buf.tofile(path)


def write_data_yaml(out, names):
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# augment.py 가 만듦 — Ultralytics OBB. path 는 train.py 가 실행하는 곳에 맞춰 다시 쓴다\n")
        fh.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % os.path.abspath(out).replace("\\", "/"))
        for i, n in enumerate(names):
            fh.write("  %d: %s\n" % (i, n))


def dataset_names(ds):
    d = load_yaml(os.path.join(ds, "data.yaml"))
    names = d.get("names", {})
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def list_images(ds, part):
    folder = os.path.join(ds, "images", part)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def visible_fraction(img, pts):
    """다각형 안에서 시야 밖(검정, ≤3)이 아닌 픽셀 비율."""
    import cv2
    import numpy as np

    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, float)).astype(np.int32)], 1)
    inside = m > 0
    area = int(inside.sum())
    if area == 0:
        return 0.0
    return 1.0 - float(((img <= 3).all(axis=2) & inside).sum()) / area


def filtered_label_lines(img, label_path, min_visible):
    """YOLO OBB 라벨 줄들 중 보이는 것만. (남긴 줄, 뺀 수)"""
    h, w = img.shape[:2]
    keep, dropped = [], 0
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            v = line.split()
            if len(v) != 9:
                continue
            c = [float(x) for x in v[1:]]
            pts = [(c[i] * w, c[i + 1] * h) for i in range(0, 8, 2)]
            if min_visible > 0 and visible_fraction(img, pts) < min_visible:
                dropped += 1
                continue
            keep.append(line.strip())
    return keep, dropped


def preview(ds, ops, n, copies, seed, out_path, cut_cfg=None):
    """학습 그림 n 장 × (원본 + 변형 copies 장) 격자."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    part = "train" if list_images(ds, "train") else "val"
    files = list_images(ds, part)
    if not files:
        raise SystemExit("그림이 없습니다: %s/images/train" % ds)
    chosen = [files[i] for i in sorted(rng.choice(len(files), size=min(n, len(files)), replace=False))]
    rows = []
    for f in chosen:
        img = read_image(os.path.join(ds, "images", part, f))
        h = 240
        scale = h / img.shape[0]
        cells = [cv2.resize(img, (int(img.shape[1] * scale), h))]
        lab = os.path.join(ds, "labels", part, os.path.splitext(f)[0] + ".txt")
        lines = open(lab, encoding="utf-8").read().split("\n") if os.path.exists(lab) else []
        for _ in range(copies):
            base, klines, info = img, lines, None
            if cut_cfg:
                base, klines, info = edge_cut.cut(img, lines, cut_cfg, rng)
            aug, used = photometric.apply(base, ops, rng)
            if info:  # 자른 그림에는 남긴 라벨을 그려 본다
                aug = aug.copy()
                hh, ww = aug.shape[:2]
                for line in klines:
                    v = line.split()
                    pts = np.array([(float(v[1 + i]) * ww, float(v[2 + i]) * hh) for i in range(0, 8, 2)], np.int32)
                    cv2.polylines(aug, [pts], True, (0, 255, 0) if v[0] == "0" else (255, 0, 255), 2)
                used = list(used) + ["CUT"]
            cell = cv2.resize(aug, (int(img.shape[1] * scale), h))
            cv2.putText(cell, ",".join(u[:5] for u in used)[:48], (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cells.append(cell)
        rows.append(np.concatenate(cells, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_image(out_path, np.concatenate(rows, 0), 90)
    return out_path


def augment(ds, out, ops, copies, seed, min_visible=0.05, cut_cfg=None):
    import numpy as np

    if os.path.abspath(ds) == os.path.abspath(out):
        raise SystemExit("출력은 다른 폴더여야 합니다")
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("출력 폴더가 비어 있지 않습니다: %s" % out)
    rng = np.random.default_rng(seed)
    names = dataset_names(ds)
    log = {"source": os.path.abspath(ds), "copies": copies, "seed": seed, "ops": ops, "counts": {}, "applied": {},
           "drop_invisible": min_visible, "dropped_labels": {"train": 0, "val": 0}, "edge_cut": cut_cfg, "edge_cut_count": 0}
    for part in ("train", "val"):
        for sub in ("images", "labels", "fieldmaps"):
            os.makedirs(os.path.join(out, sub, part), exist_ok=True)
        n = 0
        for f in list_images(ds, part):
            stem = os.path.splitext(f)[0]
            src_img = os.path.join(ds, "images", part, f)
            src_lab = os.path.join(ds, "labels", part, stem + ".txt")
            shutil.copyfile(src_img, os.path.join(out, "images", part, f))
            img = read_image(src_img)
            lines = None
            if os.path.exists(src_lab) and img is not None:
                lines, dropped = filtered_label_lines(img, src_lab, min_visible)
                log["dropped_labels"][part] += dropped
                with open(os.path.join(out, "labels", part, stem + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))
            fm = os.path.join(ds, "fieldmaps", part, stem + ".json")
            if os.path.exists(fm):
                shutil.copyfile(fm, os.path.join(out, "fieldmaps", part, stem + ".json"))
            n += 1
            if part != "train" or copies <= 0 or img is None:
                continue
            for k in range(copies):
                base, klines, cut_info = img, lines, None
                if cut_cfg and lines is not None:  # 자르기 먼저 (검정은 조명 변형에서도 검정으로 남는다)
                    base, klines, cut_info = edge_cut.cut(img, lines, cut_cfg, rng)
                aug, used = photometric.apply(base, ops, rng)
                name = "%s__aug%d" % (stem, k)
                write_image(os.path.join(out, "images", part, name + ".jpg"), aug)
                if klines is not None:
                    with open(os.path.join(out, "labels", part, name + ".txt"), "w", encoding="utf-8") as fh:
                        fh.write("\n".join(klines) + ("\n" if klines else ""))
                if cut_info:
                    used = list(used) + ["edge_cut:%s" % "+".join(c["side"] for c in cut_info["cuts"])]
                    log["edge_cut_count"] += 1
                log["applied"][name] = used
                n += 1
        log["counts"][part] = n
    for extra in ("export.json",):
        if os.path.exists(os.path.join(ds, extra)):
            shutil.copyfile(os.path.join(ds, extra), os.path.join(out, extra))
    write_data_yaml(out, names)
    with open(os.path.join(out, "augment.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
    return log


def make_cut_val(ds, out, cut_cfg, per_image=2, seed=1, min_visible=0.05):
    """평가용: 검증 그림마다 시야 끝 자르기만 건 그림 per_image 장 (조명 · 색은 그대로). fieldmaps 도 같은 이름으로."""
    import numpy as np

    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("출력 폴더가 비어 있지 않습니다: %s" % out)
    rng = np.random.default_rng(seed)
    for sub in ("images", "labels", "fieldmaps"):
        os.makedirs(os.path.join(out, sub, "val"), exist_ok=True)
    n = cut_n = 0
    for f in list_images(ds, "val"):
        stem = os.path.splitext(f)[0]
        img = read_image(os.path.join(ds, "images", "val", f))
        src_lab = os.path.join(ds, "labels", "val", stem + ".txt")
        if img is None or not os.path.exists(src_lab):
            continue
        lines, _ = filtered_label_lines(img, src_lab, min_visible)
        for k in range(per_image):
            cut_img, klines, info = edge_cut.cut(img, lines, cut_cfg, rng, force=True)
            if info is None:
                continue
            name = "%s__cut%d" % (stem, k)
            write_image(os.path.join(out, "images", "val", name + ".jpg"), cut_img)
            with open(os.path.join(out, "labels", "val", name + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(klines) + ("\n" if klines else ""))
            fm = os.path.join(ds, "fieldmaps", "val", stem + ".json")
            if os.path.exists(fm):
                shutil.copyfile(fm, os.path.join(out, "fieldmaps", "val", name + ".json"))
            n += 1
            cut_n += info["dropped"]
    os.makedirs(os.path.join(out, "images", "train"), exist_ok=True)
    write_data_yaml(out, dataset_names(ds))
    return {"images": n, "dropped_labels": cut_n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="export_drone_labels.py 가 만든 데이터셋 폴더 (data.yaml 이 있는 곳)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 <데이터셋>_aug)")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "augment.yaml"))
    ap.add_argument("--copies", type=int, default=None, help="원본 1장마다 변형 수 (설정 파일 값을 덮어쓴다)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cut-val", metavar="OUT", help="검증 그림에 시야 끝 자르기만 건 평가용 데이터셋을 OUT 에 만든다")
    ap.add_argument("--preview", type=int, default=0, metavar="N", help="변형 예시 그림만 만든다 (학습 그림 N 장)")
    a = ap.parse_args(argv)
    if not os.path.isfile(os.path.join(a.dataset, "data.yaml")):
        ap.error("%s 에 data.yaml 이 없습니다" % a.dataset)
    cfg = load_yaml(a.config)
    ops = cfg.get("ops") or {}
    photometric.check_ops(ops)
    copies = a.copies if a.copies is not None else int(cfg.get("copies", 4))
    seed = a.seed if a.seed is not None else int(cfg.get("seed", 0))
    cut_cfg = cfg.get("edge_cut") or None
    if a.cut_val:
        r = make_cut_val(a.dataset, a.cut_val, cut_cfg or {}, seed=seed + 1, min_visible=float(cfg.get("drop_invisible", 0.05)))
        print("끝: %s — 시야 끝 자른 검증 그림 %d 장 (뺀 라벨 %d)" % (a.cut_val, r["images"], r["dropped_labels"]))
        return 0
    if a.preview:
        path = preview(a.dataset, ops, a.preview, max(copies, 1), seed, os.path.join(a.out or a.dataset, "preview.jpg"), cut_cfg)
        print("변형 예시: %s" % path)
        return 0
    out = a.out or a.dataset.rstrip("/\\") + str(cfg.get("out_suffix", "_aug"))
    log = augment(a.dataset, out, ops, copies, seed, float(cfg.get("drop_invisible", 0.05)), cut_cfg)
    print("끝: %s — 학습 %d 장 (원본 × %d) · 검증 %d 장 · 안 보여 뺀 라벨 %s" % (out, log["counts"].get("train", 0), 1 + copies,
                                                               log["counts"].get("val", 0), log["dropped_labels"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
