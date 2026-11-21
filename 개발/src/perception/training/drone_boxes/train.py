#!/usr/bin/env python3
"""드론 map 그림의 상자(윗면) 찾기 모델 학습 — Ultralytics YOLO OBB.

  python train.py --data <데이터셋> [--augment] [--name <이름>] [키=값 ...]
    --data      export_drone_labels.py 가 만든 데이터셋 (또는 augment.py 결과)
    --augment   먼저 augment.py 로 조명 변형을 만든다 (<데이터셋>_aug, 이미 있으면 그것을 쓴다)
    키=값       config/train.yaml 덮어쓰기 (예: epochs=50 batch=8 model=yolo11s-obb.pt device=0)

순서
  1) (--augment) 조명 · 색 · 화질 변형 데이터셋을 만든다
  2) data.yaml 의 path 를 이 PC 의 데이터셋 위치로 고쳐 runs/<이름>/data.yaml 에 쓴다 (다른 PC 에서 내보낸 데이터셋이라도 그대로 돈다)
  3) 학습 (Ultralytics) → runs/<이름>/weights/best.pt
  4) 평가 (evaluate.py) — 원래 검증 그림으로, 경기장 좌표 cm 로, 조명 조건별(--stress)로 → runs/<이름>/eval_val/eval.md
     augment.yaml 에 edge_cut 이 있으면 시야 끝을 자른 검증 그림(<데이터셋>_cutval)으로도 → runs/<이름>/eval_cutval/eval.md
  5) 내보내기 → runs/<이름>/export/drone_boxes_<이름>.pt (+ .onnx). TensorRT .engine 은 로봇(Jetson)에서 만든다 (README)
  runs/<이름>/meta.json 에 설정 · 데이터 장수 · 버전 · GPU · git 커밋을 남긴다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from augment import load_yaml  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
OUR_KEYS = ("model",)  # train.yaml 에서 Ultralytics train() 로 넘기지 않는 키


def parse_overrides(items):
    import yaml

    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit("덮어쓰기는 키=값: %s" % it)
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def write_run_data_yaml(ds, run_dir):
    """데이터셋의 data.yaml 을 읽어 path 만 이 PC 의 위치로 바꿔 run 폴더에 쓴다."""
    import yaml

    d = load_yaml(os.path.join(ds, "data.yaml"))
    d["path"] = os.path.abspath(ds).replace("\\", "/")
    d.setdefault("train", "images/train")
    d.setdefault("val", "images/val")
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "data.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(d, fh, allow_unicode=True, sort_keys=False)
    return path


def count_images(ds):
    out = {}
    for part in ("train", "val"):
        folder = os.path.join(ds, "images", part)
        out[part] = len(os.listdir(folder)) if os.path.isdir(folder) else 0
    return out


def environment():
    info = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    try:
        info["git"] = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["git"] = None
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="데이터셋 폴더 (data.yaml 이 있는 곳)")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "train.yaml"))
    ap.add_argument("--augment", action="store_true", help="먼저 조명 변형 데이터셋(<데이터셋>_aug)을 만든다")
    ap.add_argument("--augment-config", default=os.path.join(HERE, "config", "augment.yaml"))
    ap.add_argument("--name", default=None, help="실행 이름 (기본 <시각>)")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"), help="결과 폴더")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("overrides", nargs="*", help="train.yaml 덮어쓰기 키=값")
    a = ap.parse_args(argv)

    ds = os.path.abspath(a.data)
    if not os.path.isfile(os.path.join(ds, "data.yaml")):
        ap.error("%s 에 data.yaml 이 없습니다" % ds)
    cut_val_ds = None
    eval_ds = ds  # 평가는 원래 검증 그림 (변형 데이터셋도 검증 그림은 원본 그대로 복사돼 있다)
    aug_cfg = load_yaml(a.augment_config) if a.augment else {}
    if a.augment:
        aug_ds = ds.rstrip("/\\") + str(aug_cfg.get("out_suffix", "_aug"))
        if os.path.isfile(os.path.join(aug_ds, "data.yaml")):
            print("조명 변형 데이터셋이 이미 있어 그대로 씁니다: %s (다시 만들려면 지우세요)" % aug_ds, flush=True)
        else:
            import augment

            cfg = aug_cfg
            log = augment.augment(ds, aug_ds, cfg.get("ops") or {}, int(cfg.get("copies", 4)), int(cfg.get("seed", 0)),
                                  float(cfg.get("drop_invisible", 0.05)), cfg.get("edge_cut") or None)
            print("시야 끝 자르기: 변형 %d 장" % log.get("edge_cut_count", 0), flush=True)
            print("조명 변형: 학습 %d 장 · 검증 %d 장 → %s (안 보여 뺀 라벨 %s)" % (log["counts"]["train"], log["counts"]["val"], aug_ds,
                                                                       log["dropped_labels"]), flush=True)
        if aug_cfg.get("edge_cut"):
            cut_val_ds = a.data.rstrip("/\\") + "_cutval"
            if not os.path.isfile(os.path.join(cut_val_ds, "data.yaml")):
                import augment

                r = augment.make_cut_val(os.path.abspath(a.data), cut_val_ds, aug_cfg["edge_cut"], seed=int(aug_cfg.get("seed", 0)) + 1,
                                         min_visible=float(aug_cfg.get("drop_invisible", 0.05)))
                print("시야 끝 자른 검증 그림: %d 장 → %s" % (r["images"], cut_val_ds), flush=True)
        ds = aug_ds
        eval_ds = aug_ds  # 검증 그림은 변형하지 않고 복사돼 있고, 라벨은 안 보이는 것을 뺀 것 — 평가도 같은 기준으로

    cfg = load_yaml(a.config)
    cfg.update(parse_overrides(a.overrides))
    model_path = cfg.get("model", "yolo11n-obb.pt")
    kw = {k: v for k, v in cfg.items() if k not in OUR_KEYS}
    name = a.name or dt.datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.abspath(a.runs), name)
    if os.path.exists(os.path.join(run_dir, "weights")):
        ap.error("이미 있는 실행 이름입니다: %s" % run_dir)
    data_yaml = write_run_data_yaml(ds, run_dir)
    meta = {"name": name, "started": dt.datetime.now(KST).isoformat(timespec="seconds"), "dataset": ds, "eval_dataset": eval_ds,
            "images": count_images(ds), "model": model_path, "train_args": kw, "env": environment()}
    for extra in ("export.json", "augment.json"):
        p = os.path.join(ds, extra)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            d.pop("applied", None)  # 장마다 건 변형 목록은 길다
            meta[extra.split(".")[0]] = d
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print("학습: %s · 학습 %d 장 · 검증 %d 장 · %s · %s" % (name, meta["images"]["train"], meta["images"]["val"], model_path,
                                                 meta["env"].get("gpu") or "GPU 없음"), flush=True)

    from ultralytics import YOLO

    model = YOLO(model_path, task="obb")
    model.train(data=data_yaml, project=os.path.abspath(a.runs), name=name, exist_ok=True, **kw)
    best = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best):
        raise SystemExit("best.pt 가 없습니다: %s" % best)

    export_dir = os.path.join(run_dir, "export")
    os.makedirs(export_dir, exist_ok=True)
    final_pt = os.path.join(export_dir, "drone_boxes_%s.pt" % name)
    shutil.copyfile(best, final_pt)
    meta["export"] = {"pt": final_pt}
    if not a.no_onnx:
        try:
            onnx = YOLO(best, task="obb").export(format="onnx", imgsz=int(kw.get("imgsz", 640)))
            target = os.path.join(export_dir, "drone_boxes_%s.onnx" % name)
            shutil.copyfile(onnx, target)
            meta["export"]["onnx"] = target
        except Exception as e:  # ONNX 는 덤 — 실패해도 학습 결과는 남긴다
            meta["export"]["onnx_error"] = "%s: %s" % (type(e).__name__, e)
            print("ONNX 내보내기 실패 (학습 결과는 그대로): %s" % e, flush=True)

    if not a.no_eval:
        import evaluate

        predict = evaluate.yolo_predictor(best, int(kw.get("imgsz", 640)), 0.25, kw.get("device"))
        res = evaluate.evaluate(predict, eval_ds, "val", out=os.path.join(run_dir, "eval_val"), vis=12)
        meta["eval"] = {k: v["전체"] for k, v in res["conditions"].items()}
        if cut_val_ds:
            res = evaluate.evaluate(predict, cut_val_ds, "val", out=os.path.join(run_dir, "eval_cutval"), vis=12, stress=False)
            meta["eval_cutval"] = {k: v["전체"] for k, v in res["conditions"].items()}
    meta["finished"] = dt.datetime.now(KST).isoformat(timespec="seconds")
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print("끝: %s\n  가중치 %s\n  평가 %s" % (run_dir, final_pt, os.path.join(run_dir, "eval_val", "eval.md")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
