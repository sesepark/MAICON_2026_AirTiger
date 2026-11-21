#!/usr/bin/env python3
"""수집한 드론 화면(collect_drone_frames.py 출력 · 웹 화면 수집 세션)의 렌즈 왜곡을 펴서 학습 데이터용 그림으로 만든다.

왜곡 보정은 새로 만들지 않고 이 패키지의 `perception/drone_boundary.py` 를 그대로 부른다 (detect_boundary.py 와 같은 계수).
  - 카메라 모델: 렌즈 왜곡(나눗셈 모델 k1 · k2 · 픽셀 비 · 중심) + 바닥 사영 변환 hom (보정 픽셀 → 경기장 m)
  - 계수는 체커보드가 아니라 **바닥 흰 선 모양으로 화면마다 맞춘다** (detect_boundary). 맞춘 결과의 경기장 치수(CM 반폭 · 원 반지름 ·
    잔차)가 실측 범위 밖이면 버리고 다음 장으로 다시 맞춘다 (plausible — 사람이 선을 가리면 다른 선을 고르는 일이 있다). 카메라는 고정이라 카메라마다 한 번 맞추고
    모든 장에 같은 계수를 쓴다. --recheck-every 장마다 지난 계수를 출발점(prior)으로 다시 맞춰 보고, 카메라가 움직였으면 새 계수로 바꾼다.

  python undistort_drone_frames.py --in <세션 폴더>                  # 경기장 지도 좌표 그림 (기본 --mode map — 라벨은 이것에)
  python undistort_drone_frames.py --in <세션 폴더> --mode top       # 바닥 평면 위에서 본 그림
  python undistort_drone_frames.py --in <세션 폴더> --watch          # 수집기와 같이 돌리며 새 장을 따라간다
웹 화면(label_drone_frames.py)의 「전처리」 단추도 이것을 부른다.

모드
  map   **지도(team_interfaces/FieldMap, frame "arena")와 같은 좌표** — 라벨만 달면 그대로 지도의 상자 자리가 된다.
        렌즈 왜곡을 펴고, 경기장 테두리(x 0 ~ 6.12 · y 0 ~ 5.00 m, 경계가 잰 값)로 잘라 1 cm/px 로 편다 (612 × 500 px).
        그림의 평면은 바닥이 아니라 **상자 윗면 높이(40 cm)** 다: 카메라(경기장 정중앙 위, 높이 Hc)에서 보면 높이 h 인 점은
        바닥 지도에서 s = Hc / (Hc − h) 배 바깥으로 밀려 찍힌다 (drone_boxes 와 같은 모델). map 은 그만큼 되돌려 그리므로
        상자 윗면이 자기 바닥 자리 바로 위에 40 × 40 px 정사각형으로 찍힌다 → 윗면 네 꼭짓점의 가운데 = 지도의 상자 가운데.
        대신 바닥 그림(구역선 · 원)은 가운데 쪽으로 1/s 만큼 줄어 보인다 (높이가 다르니 당연하다).
        좌표: x = u / 100, y = 5.00 − v / 100 (m, 위가 1번 줄) — calib.json 의 out_to_field.
        카메라 높이: --camera-height > 높이 캐시(../data/drone_camera_heights.json, 고정 카메라라 한 번 재면 계속) >
        이 세션 화면 여러 장에서 drone_boxes 가 옆면 폭으로 잰 값의 중앙값 (한 장은 ±15 % 흔들려 5 장 넘게 모은다).
  top   바닥 평면을 위에서 본 그림 (경기장 + 둘레 --margin-m). 박스 윗면(40 cm)은 실제보다 약 19 % 크고 바깥쪽에 있다.
  lens  렌즈 왜곡만 편다. 원근은 그대로 (박스 옆면이 보인다).

출력 (--out, 기본 <in>/undist_<mode> — 웹 화면은 이것을 그림 종류 「undist_<mode>」 로 보여 준다)
  <out>/<camera>/<원본 이름>.jpg
  <out>/index.jsonl   한 장에 한 줄: file · src(원본 index 의 file) · camera · calib · mode · headers(원본 그대로)
  <out>/calib.json    계수 목록. 항목마다 boundary(detect_boundary 결과 to_dict) · out_to_field(출력 픽셀 (u, v, 1) → 경기장 (x, y) m,
                      높이 plane_h 평면 — map 은 0.40) · camera_xyh · height_source
  <out>/failed.jsonl  못 편 장과 이유

numpy · OpenCV 가 필요하다 (perception 과 같다). 원본은 건드리지 않는다 — 계수가 나아지면 다시 돌리면 된다 (--redo).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

KST = dt.timezone(dt.timedelta(hours=9))
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PERCEPTION = os.path.dirname(HERE)  # tools → 패키지 맨 위 (detect_boundary.py 와 같다)


def load_perception(path):
    """perception 패키지의 drone_boundary 를 불러온다."""
    path = path or os.environ.get("MAICON_PERCEPTION_DIR") or DEFAULT_PERCEPTION
    if not os.path.isfile(os.path.join(path, "perception", "drone_boundary.py")):
        raise SystemExit("perception/drone_boundary.py 를 찾지 못했습니다: %s (--perception-dir 로 perception 패키지 폴더를 주세요)" % path)
    sys.path.insert(0, os.path.abspath(path))
    from perception import drone_boundary  # noqa: E402

    return drone_boundary


def now_iso():
    return dt.datetime.now(KST).isoformat(timespec="seconds")


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # 수집기가 쓰는 중인 마지막 줄
    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cam_height(text):
    cam, _, val = text.partition("=")
    try:
        return cam.strip(), float(val)
    except ValueError:
        raise argparse.ArgumentTypeError("카메라=높이 (예: test1=2.97)")


def plausible(b, args):
    """detect_boundary 가 검사를 통과했어도 다른 선 조합을 고르는 경우가 있다 (2026-11-19 test2: 사람이 B 쪽 선을 가려
    CM 반폭 0.83 · 원 0.82 m 로 맞춤 — 맞는 값은 0.63). 경기장 바닥 그림은 고정이니 잰 치수가 범위 밖이면 버린다."""
    bad = []
    for name, val, (lo, hi) in (("CM 반폭", b.cm_half, args.cm_half_m), ("원 반지름", b.circle_r, args.circle_r_m)):
        if val is None or not lo <= val <= hi:
            bad.append("%s %s m (%.2f ~ %.2f)" % (name, "?" if val is None else "%.2f" % val, lo, hi))
    if b.rms_m is None or b.rms_m > args.max_rms_m:
        bad.append("잔차 %s m > %.3f" % ("?" if b.rms_m is None else "%.3f" % b.rms_m, args.max_rms_m))
    return not bad, ", ".join(bad)


class Warper:
    """계수 하나(Boundary) → 출력 그림을 만드는 remap 표와 출력 픽셀 → 경기장 행렬."""

    def __init__(self, db, boundary, mode, margin_m, px_per_m, max_side, camera_xyh=None, plane_h=0.0):
        import numpy as np

        self.np = np
        b = boundary
        w, h, m = b.field_w, b.field_h, margin_m
        self.lift = None  # map: 높이 plane_h 평면의 점 → 같은 픽셀에 찍히는 바닥 점 (N + (P − N)·s)
        if mode == "map":
            # 경기 중 지도(perception.map_maker → drone_box_model)와 똑같은 그림이 되도록 패키지의 MapView 를 그대로 쓴다
            from perception.drone_map_view import MapView

            cx, cy, hc = camera_xyh
            view = MapView(b, hc, plane_h, px_per_m, m, (cx, cy))
            self.lift = (np.array([cx, cy], float), view.s)
            self.out_to_field, self.size = view.out_to_field, view.size
            self.map_u, self.map_v = view.maps()
            return
        if mode == "top":
            ow, oh = int(round((w + 2 * m) * px_per_m)), int(round((h + 2 * m) * px_per_m))
            # 출력 (u, v) → 경기장: x = u/ppm − m, y = h + m − v/ppm  (위 = 1번 줄, draw_before_after 와 같은 방향)
            self.out_to_field = np.array([[1 / px_per_m, 0, -m], [0, -1 / px_per_m, h + m], [0, 0, 1]], float)
        else:
            # 보정 픽셀 공간에서 경기장 + 둘레가 들어가는 상자. 사영 변환이라 네 모서리만 보면 된다
            inv = np.linalg.inv(b.hom)
            q = _apply_h(np, inv, np.array([(-m, -m), (w + m, -m), (w + m, h + m), (-m, h + m)], float))
            lo, hi = q.min(0), q.max(0)
            scale = min(1.0, max_side / float(max(hi - lo)))  # 화면 가운데는 보정 전후 축척이 같다 → 1 이 원본 해상도
            ow, oh = int(np.ceil((hi[0] - lo[0]) * scale)), int(np.ceil((hi[1] - lo[1]) * scale))
            # 출력 (u, v) → 보정 픽셀 q = lo + (u, v)/scale → 경기장 = hom · q
            to_q = np.array([[1 / scale, 0, lo[0]], [0, 1 / scale, lo[1]], [0, 0, 1]], float)
            self.out_to_field = b.hom @ to_q
        self.size = (ow, oh)
        gu, gv = np.meshgrid(np.arange(ow) + 0.0, np.arange(oh) + 0.0)
        field = _apply_h(np, self.out_to_field, np.stack([gu.ravel(), gv.ravel()], 1))
        if self.lift is not None:
            n, s = self.lift
            field = n + (field - n) * s  # 상자 윗면 높이의 점은 카메라 바로 아래 점에서 s 배 먼 바닥 점과 같은 픽셀에 찍힌다
        src = b.field_to_pixel(field).reshape(oh, ow, 2)  # 공개 함수만 쓴다: 경기장 → 원본 픽셀 (렌즈 왜곡 포함)
        src[~np.isfinite(src)] = -1.0  # 화면 밖 → 테두리 색
        self.map_u, self.map_v = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)

    def __call__(self, bgr):
        import cv2

        return cv2.remap(bgr, self.map_u, self.map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _apply_h(np, hmat, p):
    p = np.asarray(p, float).reshape(-1, 2)
    w = p[:, 0] * hmat[2, 0] + p[:, 1] * hmat[2, 1] + hmat[2, 2]
    return np.stack([(p[:, 0] * hmat[0, 0] + p[:, 1] * hmat[0, 1] + hmat[0, 2]) / w,
                     (p[:, 0] * hmat[1, 0] + p[:, 1] * hmat[1, 1] + hmat[1, 2]) / w], 1)


class Pipeline:
    def __init__(self, args, db):
        self.args, self.db = args, db
        self.calib_path = os.path.join(args.out, "calib.json")
        self.index_path = os.path.join(args.out, "index.jsonl")
        self.failed_path = os.path.join(args.out, "failed.jsonl")
        os.makedirs(args.out, exist_ok=True)
        self.entries = self._load_calib()
        self.current = {}  # camera → 지금 쓰는 계수 항목
        for e in self.entries:
            self.current[e["camera"]] = e
        if args.calib:  # 다른 데서 맞춘 계수를 먼저 쓴다 (예: 다른 날 수집분)
            for e in self._read_calib(args.calib):
                if e["camera"] not in self.current:
                    self._add_entry(e["camera"], e["boundary"], e.get("source", args.calib), note="--calib 에서 가져옴")
        self.warpers = {}
        self.done = {r["src"] for r in read_jsonl(self.index_path)} if not args.redo else set()
        self.failed_cal = {}  # camera → 계수 못 맞춘 원본 수 (다음 시도 간격)
        self.since_check = {}
        self.count = {"ok": 0, "skip": 0, "fail": 0}

    # ---------------------------------------------------------------- 계수
    def _read_calib(self, path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("entries", [])

    def _load_calib(self):
        if self.args.redo or not os.path.exists(self.calib_path):
            return []
        return [e for e in self._read_calib(self.calib_path) if e.get("mode") == self.args.mode]

    def _save_calib(self):
        tmp = self.calib_path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.entries}, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.calib_path)

    def _add_entry(self, camera, bdict, source, note="", cam=None):
        b = self.db.Boundary.from_dict(bdict)
        cam = cam or {}
        warper = Warper(self.db, b, self.args.mode, self.args.margin_m, self.args.px_per_m, self.args.max_side,
                        cam.get("camera_xyh"), cam.get("plane_h", 0.0))
        e = {
            "id": "%s-%d" % (camera, sum(1 for x in self.entries if x["camera"] == camera) + 1),
            "camera": camera, "mode": self.args.mode, "created": now_iso(), "source": source, "note": note,
            "out_size": list(warper.size), "out_to_field": warper.out_to_field.tolist(),
            "margin_m": self.args.margin_m, "px_per_m": self.args.px_per_m if self.args.mode in ("top", "map") else None,
            # out_to_field 가 가리키는 평면의 높이 (m). map 은 상자 윗면 높이 → 라벨 좌표가 곧 상자 바닥 자리 (FieldMap boxes 와 같은 뜻)
            "plane_h": cam.get("plane_h", 0.0), "camera_xyh": cam.get("camera_xyh"), "height_source": cam.get("height_source"),
            "boundary": bdict,
        }
        self.entries.append(e)
        self.current[camera] = e
        self.warpers[e["id"]] = warper
        self._save_calib()
        lz = bdict.get("lens") or {}
        self.log("[%s] 계수 %s: k1 %.3f · k2 %.3f · 픽셀 비 %.2f · 잔차 %s m · 출력 %dx%d (%s) %s" % (
            camera, e["id"], lz.get("k1", bdict.get("k", 0.0)), lz.get("k2", 0.0), lz.get("aspect", 1.0),
            "%.3f" % bdict["rms_m"] if bdict.get("rms_m") is not None else "?", warper.size[0], warper.size[1], source, note))
        if cam.get("camera_xyh"):
            self.log("[%s]   상자 윗면 평면 %.2f m · 카메라 높이 %.3f m (%s) · s %.4f" % (
                camera, cam["plane_h"], cam["camera_xyh"][2], cam.get("height_source"), warper.lift[1]))
        return e

    def _warper(self, e):
        if e["id"] not in self.warpers:
            b = self.db.Boundary.from_dict(e["boundary"])
            self.warpers[e["id"]] = Warper(self.db, b, self.args.mode, e.get("margin_m", self.args.margin_m), self.args.px_per_m,
                                           self.args.max_side, e.get("camera_xyh"), e.get("plane_h", 0.0))
        return self.warpers[e["id"]]

    def _detect(self, bgr, prior=None):
        try:
            b = self.db.detect_boundary(bgr, prior=prior)
        except Exception as ex:  # 맞추기 실패가 파이프라인을 멈추지 않게
            return None, "%s: %s" % (type(ex).__name__, ex)
        if not b.ok:
            return None, "; ".join(b.notes[-2:]) or "검사 통과 못 함"
        ok, why = plausible(b, self.args)
        if not ok:
            return None, "경기장 치수가 맞지 않아 버림 (%s) — 사람 · 로봇이 선을 가려 다른 선을 고른 듯" % why
        return b, None

    def calibrate(self, camera, rows):
        """그 카메라의 원본들 중 몇 장으로 계수를 맞춘다 (고르게 떨어진 장부터)."""
        tries = self.args.calib_tries
        idx = sorted({int(round(i * (len(rows) - 1) / max(tries - 1, 1))) for i in range(tries)}) if rows else []
        for i in idx:
            r = rows[i]
            bgr = self.db.read_image(os.path.join(self.args.inp, r["file"]))
            if bgr is None:
                continue
            t0 = time.monotonic()
            b, why = self._detect(bgr)
            if b is not None:
                cam = None
                if self.args.mode == "map":
                    cam, why = self.camera_model(camera, rows, b)
                    if cam is None:
                        self.log("[%s] 경계는 맞췄지만 %s" % (camera, why))
                        return None
                return self._add_entry(camera, b.to_dict(), r["file"], note="%.1fs" % (time.monotonic() - t0), cam=cam)
            self.log("[%s] %s 로는 계수를 못 맞춤: %s" % (camera, r["file"], why))
        return None

    # ---------------------------------------------------------------- 카메라 높이 (map)
    def _height_cache(self):
        try:
            with open(self.args.height_cache, encoding="utf-8") as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def camera_model(self, camera, rows, b):
        """(map 에 쓸 카메라 dict, 못 정한 이유). 카메라는 경기장 정중앙 위 (drone_boxes 와 같다, 현장 확인 2026-11-18).
        높이는 --camera-height > 높이 캐시(고정 카메라라 한 번 재면 계속 쓴다) > 이 세션 화면들에서 drone_boxes 로 잰 중앙값."""
        import statistics

        cx, cy = b.field_w / 2.0, b.field_h / 2.0
        plane_h = self.args.box_height
        given = dict(self.args.camera_height or [])
        if camera in given:
            hc, src = given[camera], "given"
        elif camera in self._height_cache():
            hc, src = float(self._height_cache()[camera]["height"]), "cache"
        else:
            from perception import drone_boxes  # 상자 찾기 (옆면 폭으로 카메라 높이를 잰다)

            k = self.args.height_frames
            idx = sorted({int(round(i * (len(rows) - 1) / max(k - 1, 1))) for i in range(k)}) if rows else []
            vals = []
            for i in idx:
                bgr = self.db.read_image(os.path.join(self.args.inp, rows[i]["file"]))
                if bgr is None:
                    continue
                try:
                    view = drone_boxes.detect_boxes(bgr, b)
                except Exception:  # 한 장이 실패해도 다른 장으로
                    continue
                if view.height_source == "measured":
                    vals.append(float(view.camera[2]))
            if len(vals) < self.args.min_height_frames:
                return None, "카메라 높이를 잴 장이 모자람 (잰 장 %d < %d) — 새 장이 쌓이면 다시" % (len(vals), self.args.min_height_frames)
            hc, src = statistics.median(vals), "measured"
            cache = self._height_cache()
            cache[camera] = {"height": round(hc, 4), "n": len(vals), "values": [round(v, 3) for v in vals], "measured_at": now_iso(),
                             "session": os.path.basename(self.args.inp)}
            os.makedirs(os.path.dirname(os.path.abspath(self.args.height_cache)), exist_ok=True)
            with open(self.args.height_cache, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False, indent=1)
            self.log("[%s] 카메라 높이 %.3f m — %d 장 중앙값 (%.2f ~ %.2f), %s 에 남김" % (
                camera, hc, len(vals), min(vals), max(vals), self.args.height_cache))
        if hc <= plane_h + 0.5:
            return None, "카메라 높이 %.2f m 가 이상함" % hc
        return {"camera_xyh": [cx, cy, hc], "plane_h": plane_h, "height_source": src}, None

    def recheck(self, camera, bgr, src):
        """지난 계수를 출발점으로 다시 맞춰 보고, 모서리가 --move-px 넘게 바뀌었으면 새 계수로."""
        import numpy as np

        cur = self.current[camera]
        b, why = self._detect(bgr, prior=cur["boundary"])
        if b is None:
            self.log("[%s] 다시 맞추기 실패 (%s) — 지금 계수 %s 를 계속 쓴다" % (camera, why, cur["id"]))
            return cur
        old = np.asarray(cur["boundary"].get("corners_px") or [[0, 0]] * 4, float)
        new = np.asarray(b.to_dict()["corners_px"], float)
        moved = float(np.nanmax(np.abs(new - old))) if old.shape == new.shape else float("inf")
        if moved > self.args.move_px:
            cam = None
            if cur.get("camera_xyh"):  # 카메라 높이는 그대로 (경기장 크기만 다시 맞춘 것)
                cam = {"camera_xyh": [b.field_w / 2.0, b.field_h / 2.0, cur["camera_xyh"][2]], "plane_h": cur["plane_h"],
                       "height_source": cur.get("height_source")}
            return self._add_entry(camera, b.to_dict(), src, note="모서리 %.1f px 움직임" % moved, cam=cam)
        return cur

    # ---------------------------------------------------------------- 처리
    def log(self, msg):
        print("%s %s" % (dt.datetime.now(KST).strftime("%H:%M:%S"), msg), flush=True)

    def pending(self):
        rows = [r for r in read_jsonl(os.path.join(self.args.inp, "index.jsonl")) if r.get("file") and r["file"] not in self.done]
        if self.args.cameras:
            rows = [r for r in rows if r.get("camera") in self.args.cameras]
        by_cam = {}
        for r in rows:
            by_cam.setdefault(r.get("camera") or "unknown", []).append(r)
        return by_cam

    def run_once(self):
        """밀린 장을 모두 처리. 처리한 장 수를 돌려준다."""
        n = 0
        for camera, rows in sorted(self.pending().items()):
            if camera not in self.current:
                # 처음엔 바로 시도. 못 맞춘 카메라는 그 뒤로 새 장이 --retry-after 장 쌓였을 때 다시 시도
                if camera in self.failed_cal and len(rows) < self.failed_cal[camera] + self.args.retry_after:
                    continue
                if self.calibrate(camera, rows) is None:
                    self.failed_cal[camera] = len(rows)
                    self.log("[%s] 계수를 못 맞춰 %d 장을 미룬다 (새 장이 쌓이면 다시 시도)" % (camera, len(rows)))
                    continue
            for r in rows:
                n += self.process(camera, r)
        return n

    def process(self, camera, r):
        src = r["file"]
        try:
            bgr = self.db.read_image(os.path.join(self.args.inp, src))
            if bgr is None:
                raise ValueError("그림을 읽지 못함")
            self.since_check[camera] = self.since_check.get(camera, 0) + 1
            if self.args.recheck_every and self.since_check[camera] >= self.args.recheck_every:
                self.since_check[camera] = 0
                self.recheck(camera, bgr, src)
            e = self.current[camera]
            warper = self._warper(e)
            if tuple(bgr.shape[1::-1]) != tuple(e["boundary"]["image_size"]):
                raise ValueError("화면 크기 %s 가 계수의 %s 와 다름" % (bgr.shape[1::-1], e["boundary"]["image_size"]))
            out_rel = "%s/%s" % (camera, os.path.basename(src))
            out_path = os.path.join(self.args.out, camera, os.path.basename(src))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if not self.db.write_image(out_path, warper(bgr)):
                raise ValueError("저장 실패")
            append_jsonl(self.index_path, {"file": out_rel, "src": src, "camera": camera, "calib": e["id"], "mode": self.args.mode,
                                           "headers": r.get("headers", {})})
            self.count["ok"] += 1
        except Exception as ex:  # 한 장이 실패해도 다음 장으로
            append_jsonl(self.failed_path, {"src": src, "camera": camera, "error": "%s: %s" % (type(ex).__name__, ex), "at": now_iso()})
            self.count["fail"] += 1
        self.done.add(src)
        return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="수집 폴더 (index.jsonl 이 있는 곳)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 <in>/undist_<mode>)")
    ap.add_argument("--mode", choices=("map", "top", "lens"), default="map",
                    help="map: 경기장 지도 좌표 그대로 (상자 윗면 높이 평면, 1 cm/px, 경기장만) · top: 바닥 평면 위에서 본 그림 · lens: 렌즈 왜곡만")
    ap.add_argument("--cameras", default=None, help="이 카메라만, 쉼표 구분 (기본 전부)")
    ap.add_argument("--perception-dir", default=None, help="perception 패키지 폴더 (기본 $MAICON_PERCEPTION_DIR, 없으면 ..)")
    ap.add_argument("--calib", default=None, help="다른 출력의 calib.json — 그 카메라 계수를 먼저 쓴다")
    ap.add_argument("--calib-tries", type=int, default=8, help="계수를 맞출 때 시도할 장 수 (고르게 떨어진 장)")
    ap.add_argument("--retry-after", type=int, default=2, help="계수를 못 맞춘 카메라는 새 장이 이만큼 쌓이면 다시 시도")
    ap.add_argument("--cm-half-m", type=float, nargs=2, default=(0.52, 0.72), metavar=("MIN", "MAX"),
                    help="받아들일 CM 반폭 (m) — 연습 경기장 실측 0.60 ~ 0.64 (2026-11-19, test1 · test2)")
    ap.add_argument("--circle-r-m", type=float, nargs=2, default=(0.53, 0.73), metavar=("MIN", "MAX"),
                    help="받아들일 중앙 원 반지름 (m) — 실측 0.61 ~ 0.65")
    ap.add_argument("--max-rms-m", type=float, default=0.015, help="받아들일 바닥선 잔차 RMS (m) — 맞는 계수는 0.008 ~ 0.010")
    ap.add_argument("--recheck-every", type=int, default=300, help="카메라마다 이만큼 장마다 계수를 다시 맞춰 본다 (0 = 안 함)")
    ap.add_argument("--move-px", type=float, default=2.0, help="다시 맞춘 경기장 모서리가 이보다 많이 움직이면 새 계수로")
    ap.add_argument("--margin-m", type=float, default=None, help="경기장 둘레를 이만큼 더 넣는다 (m). 기본 map 0 (경기장 테두리로 자름) · top · lens 0.4")
    ap.add_argument("--px-per-m", type=float, default=100.0, help="map · top 축척 (기본 100 = 1 cm/px)")
    ap.add_argument("--box-height", type=float, default=0.40, help="map 이 그리는 평면 높이 = 상자 높이 (m, 공식 40 cm)")
    ap.add_argument("--camera-height", type=_cam_height, action="append", default=None, metavar="카메라=높이",
                    help="카메라 높이를 정해 준다 (예: test1=2.97). 없으면 높이 캐시 → 화면들에서 잰 중앙값")
    ap.add_argument("--height-cache", default=os.path.join(DEFAULT_PERCEPTION, "data", "drone_camera_heights.json"),
                    help="카메라별로 잰 높이를 남기는 파일 (고정 카메라라 한 번 재면 계속 쓴다. 다시 재려면 지운다)")
    ap.add_argument("--height-frames", type=int, default=12, help="카메라 높이를 잴 때 쓸 장 수 (고르게 떨어진 장)")
    ap.add_argument("--min-height-frames", type=int, default=5, help="높이를 잰 장이 이보다 적으면 더 쌓일 때까지 기다린다 (한 장은 ±15 %% 흔들린다)")
    ap.add_argument("--max-side", type=int, default=2000, help="lens 모드 출력의 긴 변 최대 px")
    ap.add_argument("--redo", action="store_true", help="이미 한 것과 계수를 무시하고 처음부터 (출력 폴더를 비우고 쓰는 것을 권함)")
    ap.add_argument("--watch", action="store_true", help="끝나도 멈추지 않고 새 장을 기다린다 (Ctrl+C 로 끝)")
    ap.add_argument("--poll-s", type=float, default=5.0, help="--watch 때 index.jsonl 을 다시 읽는 간격")
    args = ap.parse_args(argv)
    if args.margin_m is None:
        args.margin_m = 0.0 if args.mode == "map" else 0.4

    args.inp = os.path.abspath(args.inp)
    if args.watch and os.path.isdir(args.inp):
        try:  # 수집기는 첫 장을 저장할 때 index.jsonl 을 만든다 → 따라가기 모드는 생길 때까지 기다린다
            while not os.path.isfile(os.path.join(args.inp, "index.jsonl")):
                time.sleep(args.poll_s)
        except KeyboardInterrupt:
            return 0
    if not os.path.isfile(os.path.join(args.inp, "index.jsonl")):
        ap.error("%s 에 index.jsonl 이 없습니다 (collect_drone_frames.py 의 출력 폴더를 주세요)" % args.inp)
    args.out = args.out or os.path.join(args.inp, "undist_" + args.mode)
    args.cameras = [c.strip() for c in args.cameras.split(",") if c.strip()] if args.cameras else None
    if args.redo and os.path.exists(os.path.join(args.out, "index.jsonl")):
        ap.error("--redo 는 빈 출력 폴더에 쓰세요 (%s 에 이미 결과가 있음)" % args.out)

    db = load_perception(args.perception_dir)
    pipe = Pipeline(args, db)
    pipe.log("시작: %s → %s (mode %s, 이미 한 것 %d 장)" % (args.inp, args.out, args.mode, len(pipe.done)))
    try:
        while True:
            n = pipe.run_once()
            if n:
                c = pipe.count
                pipe.log("처리 %d 장 (누적 성공 %d · 실패 %d)" % (n, c["ok"], c["fail"]))
            if not args.watch:
                break
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        pipe.log("Ctrl+C — 끝낸다")
    c = pipe.count
    pipe.log("끝: 성공 %d · 실패 %d (failed.jsonl) · 계수 %d 개 (calib.json)" % (c["ok"], c["fail"], len(pipe.entries)))
    return 1 if c["fail"] and not c["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
