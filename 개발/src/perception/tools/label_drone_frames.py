#!/usr/bin/env python3
"""드론 화면 수집 · 박스 라벨링 웹 화면 (학습 데이터용 — 경기 중 코드 아님).

  python label_drone_frames.py                   → http://127.0.0.1:8090/
  python label_drone_frames.py --host 0.0.0.0    다른 PC 에서도 열기 (같은 망의 팀원이 나눠 라벨링)

화면
  수집      「시작」 → 「정지」 를 누를 때까지 드론 화면을 DEFAULT_INTERVAL_S(7초)마다 한 장씩 받아 세션 폴더에 쌓는다 — 장 사이에
            박스 배치를 바꿀 시간을 두려는 간격이다 (화면에서 바꿀 수 있다). 수집은 collect_drone_frames.py 를 따로 띄워서 하고
            (임무코드는 환경변수로 넘긴다), 수집하는 동안 undistort_drone_frames.py --watch 가 받은 장의 렌즈 왜곡을 바로 편다.
  라벨링    **왜곡을 편 그림(undist_map, 그 밖에 undist_top · undist_lens)에만** 박스 윗면을 사각형 · 네 점으로 표시한다 (원본 raw 에는 달지 않는다).
            undist_map 은 지도(FieldMap, frame "arena")와 같은 좌표라, 라벨 픽셀 (u, v) → 경기장 (u/100, 5.00 − v/100) m 가 곧 상자 자리다.
            「전처리」 는 수집 때 못 편 세션을 다시 편다. 「내보내기」 는 YOLO OBB 데이터셋을 만든다 (export_drone_labels.py).
  전처리 · 렌즈 보정에는 numpy · OpenCV 가 필요하다 → 그것이 깔린 파이썬으로 이 서버를 띄운다 (전처리는 같은 파이썬으로 돈다).
    python3 -m venv ~/.venvs/maicon_perception && ~/.venvs/maicon_perception/bin/pip install -r tools/requirements_label.txt

서버 주소 · 임무코드
  서버는 DEFAULT_SERVER (연습 서버 시험 포트 — test1 · test2 만. main 은 8000 이고 우리 경기 중에만 나온다).
  임무코드는 저장소에 올리지 않는다 (.gitignore 의 비밀 정보). 환경변수 MAICON_MISSION_CODE 나 로컬 설정 파일
  ../data/label_drone_frames.json {"mission_code": "...", "server": "..."} (data/ 라 git 에 안 올라감)에 두면 화면에 따로 넣지 않아도 된다.

저장 위치 (기본 ../data/drone_sessions — data/ 는 git 에 올라가지 않는다)
  <store>/<세션>/                         수집기 출력 (index.jsonl · <camera>/*.jpg)
  <store>/<세션>/undist_<mode>/          전처리 출력
  <store>/<세션>/labels/<그림 종류>/<camera>/<이름>.json   라벨 한 장
  다른 곳에 있는 수집 폴더도 --also 로 함께 보인다.

라벨 한 장 (JSON)
  {"image": "test1/x.jpg", "width": 1280, "height": 720, "status": "done" | "draft" | "skip",
   "boxes": [{"cls": "box", "pts": [[u, v], [u, v], [u, v], [u, v]]}], "updated": "..."}
  pts 는 그 그림(왜곡을 편 그림)의 픽셀 좌표. 내보내기는 status 가 done 인 것만 쓴다 (박스 0개 done = 박스 없는 그림).

임무코드는 로그 · 화면 응답에 남기지 않는다. 웹 서버 · 수집 · 라벨 · 내보내기는 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_drone_labels  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")  # 개발/src/perception/data (git 에 안 올라감)
DEFAULT_STORE = os.path.join(DATA_DIR, "drone_sessions")
DEFAULT_DATASETS = os.path.join(DATA_DIR, "drone_datasets")
STATIC = os.path.join(HERE, "label_drone_frames")
COLLECTOR = os.path.join(HERE, "collect_drone_frames.py")
UNDISTORT = os.path.join(HERE, "undistort_drone_frames.py")
LOCAL_CONF = os.path.join(DATA_DIR, "label_drone_frames.json")  # 임무코드 등 로컬 설정 (git 에 안 올라감)
CAMERAS = ("main", "test1", "test2")
DEFAULT_SERVER = "http://<서버IP>:8100"  # 연습 서버 시험 포트. 주소는 git 에 적지 않는다 — MAICON_SERVER 환경변수나 로컬 설정(label_drone_frames.json 의 server)으로 준다
DEFAULT_CAMERAS = ("test1", "test2")  # main 은 이 포트에서 422 — 8000 에서 우리 경기 중에만
DEFAULT_INTERVAL_S = 7.0  # 장 사이 간격 — 그사이 박스 배치를 바꾼다 (5 ~ 10초)
LABEL_MODE = "map"  # 수집하며 바로 펴는 전처리 (undist_map: 지도 좌표 그대로 — 경기장 테두리로 자른 상자 윗면 평면, 1 cm/px)
MIN_FREE_GB = 2.0  # 디스크 여유가 이보다 적으면 수집을 멈춘다
# 라벨 종류 (내보내기의 클래스 번호 = 이 순서). 바꾸면 이미 단 라벨의 cls 이름과 맞는지 확인할 것
CLASSES = [
    {"id": "box", "name": "박스 윗면", "color": "#22d3ee"},
    {"id": "stripe", "name": "줄무늬 블록", "color": "#facc15"},
]
VARIANT_RE = re.compile(r"^(raw|undist_[a-z]+)$")
LABEL_VARIANT_RE = re.compile(r"^undist_[a-z]+$")  # 라벨은 왜곡을 편 그림에만
NAME_RE = re.compile(r"[^0-9A-Za-z가-힣_.-]+")


def now_iso():
    return dt.datetime.now(KST).isoformat(timespec="seconds")


def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # 쓰는 중인 마지막 줄
    except FileNotFoundError:
        pass
    return rows


def inside(root, rel):
    """root 밑의 rel 경로 (밖으로 나가면 None)."""
    if not rel or "\x00" in rel:
        return None
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, rel))
    return path if path == root or path.startswith(root + os.sep) else None


def load_local_conf():
    """../data/label_drone_frames.json (git 에 안 올라감). 없거나 깨졌으면 빈 dict."""
    try:
        with open(LOCAL_CONF, encoding="utf-8") as fh:
            conf = json.load(fh)
        return conf if isinstance(conf, dict) else {}
    except (OSError, ValueError):
        return {}


def undistort_available():
    """이 파이썬에 전처리(렌즈 보정)에 필요한 numpy · OpenCV 가 있는지."""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        return False, "%s 에 %s 가 없습니다 — numpy · OpenCV 가 깔린 파이썬(가상환경)으로 이 서버를 띄우세요" % (sys.executable, e.name)
    return True, ""


class CollectProc:
    """collect_drone_frames.py 를 띄워 두고 출력 줄로 카메라별 상태를 센다. 정지 = 프로세스 끝내기."""

    SAVED = re.compile(r"^\[[^\]]*\] (\S+): 저장 (\S+)")
    FAILED = re.compile(r"^\[[^\]]*\] (\S+): 실패 (\S+) ?(.*)$")

    def __init__(self, out, server, cameras, code, interval):
        os.makedirs(out, exist_ok=True)
        self.out, self.server = out, server
        self.cameras = {c: {"saved": 0, "errors": 0, "state": "시작", "last_file": None} for c in cameras}
        self.lines = collections.deque(maxlen=200)
        self.started, self.ended, self.end_reason = time.time(), None, None
        env = dict(os.environ, MAICON_MISSION_CODE=code, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")  # 코드는 명령줄에 안 싣는다
        cmd = [sys.executable, "-u", COLLECTOR, "--base", server, "--cameras", *cameras, "--interval", str(interval), "--out", out]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8",
                                     errors="replace", creationflags=flags)
        self.log_fh = open(os.path.join(out, "collect.log"), "a", encoding="utf-8")
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._watch_disk, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            self.lines.append(line)
            try:
                self.log_fh.write(line + "\n")
                self.log_fh.flush()
            except (OSError, ValueError):
                pass
            m = self.SAVED.match(line)
            if m and m.group(1) in self.cameras:
                c = self.cameras[m.group(1)]
                c["saved"] += 1
                c["last_file"] = m.group(2)
                c["state"] = "받는 중"
                continue
            m = self.FAILED.match(line)
            if m and m.group(1) in self.cameras:
                c = self.cameras[m.group(1)]
                c["errors"] += 1
                hint = " — 테스트 모드가 꺼져 있거나 포트가 다름 (계속 다시 시도)" if m.group(2) == "409" else ""
                hint = " — 임무코드를 확인" if m.group(2) == "401" else hint
                c["state"] = "실패 %s %s%s" % (m.group(2), m.group(3)[:120], hint)
        self.proc.wait()
        if self.ended is None:
            self.ended, self.end_reason = time.time(), "수집기가 끝남 (종료 코드 %s)" % self.proc.returncode
        try:
            self.log_fh.close()
        except OSError:
            pass

    def _watch_disk(self):
        while self.proc.poll() is None:
            free = self.free_gb()
            if free is not None and free < MIN_FREE_GB:
                self.stop("디스크 여유 %.1f GB < %.1f GB" % (free, MIN_FREE_GB))
                return
            time.sleep(5)

    def free_gb(self):
        try:
            return shutil.disk_usage(self.out).free / 1e9
        except OSError:
            return None

    @property
    def running(self):
        return self.proc.poll() is None

    def stop(self, reason="정지 요청"):
        if self.proc.poll() is None:
            self.ended, self.end_reason = time.time(), reason
            if os.name == "nt":
                self.proc.terminate()
            else:
                self.proc.send_signal(signal.SIGINT)  # 수집기가 KeyboardInterrupt 로 「끝」 을 찍고 나간다
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def status(self):
        return {"running": self.running, "out": os.path.abspath(self.out), "server": self.server, "started": self.started,
                "ended": self.ended, "end_reason": self.end_reason, "free_gb": self.free_gb(), "cameras": self.cameras,
                "log": list(self.lines)[-30:]}


class App:
    def __init__(self, store, also, datasets):
        self.store = os.path.abspath(store)
        self.also = [os.path.abspath(a) for a in also]
        self.datasets = os.path.abspath(datasets)
        os.makedirs(self.store, exist_ok=True)
        self.lock = threading.Lock()
        self.run = None  # 지금 (또는 마지막) 수집
        self.run_session = None
        conf = load_local_conf()
        self.code = os.environ.get("MAICON_MISSION_CODE") or conf.get("mission_code") or None  # 응답 · 로그에 내지 않는다
        self.collect_defaults = {"server": os.environ.get("MAICON_SERVER") or conf.get("server") or DEFAULT_SERVER,
                                 "cameras": list(conf.get("cameras") or DEFAULT_CAMERAS), "interval": float(conf.get("interval_s") or DEFAULT_INTERVAL_S),
                                 "label_mode": LABEL_MODE}
        self.prep = None  # 전처리 subprocess 상태
        self.prep_lines = collections.deque(maxlen=300)
        self.prep_ok, self.prep_why = undistort_available()

    # ---------------------------------------------------------------- 세션
    def sessions(self):
        """세션 id → 폴더. store 밑의 index.jsonl 이 있는 폴더 + --also 폴더."""
        out = {}
        try:
            names = sorted(os.listdir(self.store))
        except OSError:
            names = []
        for name in names:
            path = os.path.join(self.store, name)
            if os.path.isfile(os.path.join(path, "index.jsonl")) or (self.run_session == name):
                out[name] = path
        for path in self.also:
            if os.path.isfile(os.path.join(path, "index.jsonl")):
                sid = "@" + os.path.basename(path.rstrip("/\\"))
                while sid in out:
                    sid += "_"
                out[sid] = path
        return out

    def session_path(self, sid):
        path = self.sessions().get(sid)
        if path is None:
            raise ApiError(404, "세션이 없습니다: %s" % sid)
        return path

    @staticmethod
    def variant_root(spath, variant):
        if not VARIANT_RE.match(variant or ""):
            raise ApiError(400, "그림 종류가 이상합니다: %s" % variant)
        return spath if variant == "raw" else os.path.join(spath, variant)

    @staticmethod
    def label_path(spath, variant, image):
        root = os.path.join(spath, "labels", variant)
        path = inside(root, os.path.splitext(image)[0] + ".json")
        if path is None:
            raise ApiError(400, "그림 경로가 이상합니다")
        return path

    def session_info(self, sid, spath):
        variants = []
        for v in ["raw"] + sorted(d for d in os.listdir(spath) if d.startswith("undist_") and os.path.isdir(os.path.join(spath, d))):
            rows = read_jsonl(os.path.join(self.variant_root(spath, v), "index.jsonl"))
            cams = collections.Counter(r.get("camera", "?") for r in rows)
            lab = collections.Counter()
            lroot = os.path.join(spath, "labels", v)
            for dirpath, _, files in os.walk(lroot):
                for f in files:
                    if f.endswith(".json"):
                        try:
                            with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                                lab[json.load(fh).get("status", "draft")] += 1
                        except (OSError, ValueError):
                            continue
            variants.append({"id": v, "frames": len(rows), "cameras": dict(cams), "labels": dict(lab)})
        return {"id": sid, "path": spath, "variants": variants, "collecting": self.run_session == sid and self.run is not None and self.run.running}

    def images(self, sid, variant):
        spath = self.session_path(sid)
        root = self.variant_root(spath, variant)
        out = []
        for r in read_jsonl(os.path.join(root, "index.jsonl")):
            f = r.get("file")
            if not f:
                continue
            item = {"file": f, "camera": r.get("camera"), "captured": (r.get("headers") or {}).get("X-Captured-At")}
            lp = self.label_path(spath, variant, f)
            if os.path.exists(lp):
                try:
                    with open(lp, encoding="utf-8") as fh:
                        lab = json.load(fh)
                    item["status"], item["n"] = lab.get("status", "draft"), len(lab.get("boxes", []))
                except (OSError, ValueError):
                    item["status"] = "broken"
            out.append(item)
        return out

    # ---------------------------------------------------------------- 라벨
    def get_label(self, sid, variant, image):
        lp = self.label_path(self.session_path(sid), variant, image)
        if not os.path.exists(lp):
            return None
        with open(lp, encoding="utf-8") as fh:
            return json.load(fh)

    def put_label(self, body):
        sid, variant, image = body.get("session"), body.get("variant"), body.get("image")
        spath = self.session_path(sid)
        if not LABEL_VARIANT_RE.match(variant or ""):
            raise ApiError(400, "라벨은 왜곡을 편 그림(undist_*)에만 답니다 — 먼저 전처리하세요")
        if inside(self.variant_root(spath, variant), image) is None:
            raise ApiError(400, "그림 경로가 이상합니다")
        status = body.get("status", "draft")
        if status not in ("done", "draft", "skip"):
            raise ApiError(400, "status 는 done · draft · skip")
        ids = {c["id"] for c in CLASSES}
        boxes = []
        for b in body.get("boxes") or []:
            pts = b.get("pts")
            if b.get("cls") not in ids or not isinstance(pts, list) or len(pts) != 4:
                raise ApiError(400, "박스 형식이 이상합니다: %r" % (b,))
            boxes.append({"cls": b["cls"], "pts": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts]})
        lab = {"image": image, "width": int(body.get("width") or 0), "height": int(body.get("height") or 0), "status": status,
               "boxes": boxes, "updated": now_iso()}
        lp = self.label_path(spath, variant, image)
        if status == "draft" and not boxes and not os.path.exists(lp):
            return {"saved": False}  # 아무것도 안 한 그림은 파일을 만들지 않는다
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        tmp = lp + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(lab, fh, ensure_ascii=False)
        os.replace(tmp, lp)
        return {"saved": True, "label": lab}

    # ---------------------------------------------------------------- 수집
    def collect_start(self, body):
        with self.lock:
            if self.run is not None and self.run.running:
                raise ApiError(409, "이미 수집 중입니다")
            code = (body.get("code") or "").strip() or self.code
            if not code:
                raise ApiError(400, "임무코드를 넣으세요")
            server = (body.get("server") or "").strip()
            if not re.match(r"^https?://", server):
                raise ApiError(400, "서버 주소는 http://<서버IP>:<포트> 형식")
            name = NAME_RE.sub("_", (body.get("name") or "").strip()).strip("._")[:40]
            sid = body.get("resume") or (dt.datetime.now(KST).strftime("%Y%m%d_%H%M%S") + ("_" + name if name else ""))
            if inside(self.store, sid) is None or os.sep in sid or "/" in sid:
                raise ApiError(400, "세션 이름이 이상합니다")
            cameras = [c for c in (body.get("cameras") or CAMERAS) if c in CAMERAS]
            if not cameras:
                raise ApiError(400, "카메라를 하나 이상 고르세요")
            interval = float(body.get("interval") or DEFAULT_INTERVAL_S)
            if not 1.0 <= interval <= 600.0:
                raise ApiError(400, "간격은 1 ~ 600 초")
            if self.prep is not None and self.prep["proc"].poll() is None:
                self._stop_prep()  # 다른 세션의 전처리가 돌고 있으면 끝내고 이 세션을 따라간다
            run = CollectProc(os.path.join(self.store, sid), server, cameras, code, interval)
            self.code = code
            self.collect_defaults.update(server=server, cameras=cameras, interval=interval)
            self.run, self.run_session = run, sid
            note = None
            if self.prep_ok:
                time.sleep(0.2)  # 수집기가 세션 폴더 · index.jsonl 을 만들 틈
                self._start_prep(sid, LABEL_MODE, watch=True)
            else:
                note = "전처리 못 함: " + self.prep_why
            return {"session": sid, "note": note}

    def collect_stop(self):
        with self.lock:
            run = self.run
        if run is None:
            raise ApiError(409, "수집 중이 아닙니다")
        run.stop("웹 화면에서 정지")
        sid = self.run_session
        # 따라가던 전처리를 끝내고, 마지막 몇 장까지 한 번 더 편다
        if self.prep is not None and self.prep.get("watch") and self.prep["session"] == sid:
            self._stop_prep()
            try:
                self._start_prep(sid, self.prep["mode"], watch=False)
            except ApiError:
                pass
        return {"stopped": True}

    def status(self):
        st = {"collect": None, "session": self.run_session, "code_set": bool(self.code), "defaults": self.collect_defaults,
              "classes": CLASSES, "cameras": list(CAMERAS), "prep": self.prep_status(),
              "prep_ok": self.prep_ok, "prep_why": self.prep_why}
        if self.run is not None:
            st["collect"] = self.run.status()
        return st

    # ---------------------------------------------------------------- 전처리 · 내보내기
    def prep_status(self):
        p = self.prep
        if p is None:
            return None
        return {"session": p["session"], "mode": p["mode"], "watch": p["watch"], "running": p["proc"].poll() is None,
                "code": p["proc"].poll(), "log": list(self.prep_lines)[-40:]}

    def prep_start(self, body):
        with self.lock:
            sid, mode = body.get("session"), body.get("mode", LABEL_MODE)
            if mode not in ("map", "top", "lens"):
                raise ApiError(400, "mode 는 map · top · lens")
            return self._start_prep(sid, mode, watch=False)

    def _start_prep(self, sid, mode, watch):
        if not self.prep_ok:
            raise ApiError(409, "전처리 못 함: " + self.prep_why)
        if self.prep is not None and self.prep["proc"].poll() is None:
            raise ApiError(409, "전처리가 이미 돌고 있습니다 (%s)" % self.prep["session"])
        spath = self.session_path(sid)
        cmd = [sys.executable, "-u", UNDISTORT, "--in", spath, "--mode", mode] + (["--watch", "--poll-s", "2"] if watch else [])
        self.prep_lines.clear()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HERE, text=True, encoding="utf-8",
                                errors="replace", creationflags=flags)
        self.prep = {"session": sid, "mode": mode, "watch": watch, "proc": proc}
        threading.Thread(target=self._pump, args=(proc,), daemon=True).start()
        return {"started": True}

    def _stop_prep(self):
        proc = self.prep["proc"]
        if proc.poll() is None:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGINT)  # undistort 는 KeyboardInterrupt 로 「끝」 을 찍고 나간다
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _pump(self, proc):
        for line in proc.stdout:
            self.prep_lines.append(line.rstrip())
        proc.wait()
        self.prep_lines.append("(끝 — 종료 코드 %s)" % proc.returncode)

    def export(self, body):
        variant = body.get("variant", "raw")
        sids = body.get("sessions") or list(self.sessions())
        sessions = [(sid, self.session_path(sid)) for sid in sids]
        stamp = dt.datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.datasets, "%s_%s" % (stamp, variant))
        try:
            return export_drone_labels.export(sessions, variant, out, [c["id"] for c in CLASSES], val_ratio=float(body.get("val_ratio", 0.2)))
        except ValueError as e:
            raise ApiError(400, str(e))


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "drone-label/1"

        def log_message(self, fmt, *a):  # 요청마다 찍지 않는다 (그림 요청이 많다). 오류는 _send 에서
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            if isinstance(body, (dict, list)) or body is None:
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=60")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 5_000_000:
                raise ApiError(413, "요청이 너무 큽니다")
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                raise ApiError(400, "JSON 이 아닙니다")

        def _dispatch(self, method):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
            path = url.path
            try:
                if method == "GET":
                    if path in ("/", "/index.html"):
                        return self._file(os.path.join(STATIC, "index.html"))
                    if path.startswith("/static/"):
                        f = inside(STATIC, path[len("/static/"):])
                        return self._file(f) if f else self._send(404, {"error": "없음"})
                    if path == "/api/status":
                        return self._send(200, app.status())
                    if path == "/api/sessions":
                        return self._send(200, [app.session_info(s, p) for s, p in app.sessions().items()])
                    if path == "/api/images":
                        return self._send(200, app.images(q.get("session"), q.get("variant", "raw")))
                    if path == "/api/label":
                        return self._send(200, app.get_label(q.get("session"), q.get("variant", "raw"), q.get("image")))
                    if path == "/img":
                        root = app.variant_root(app.session_path(q.get("session")), q.get("variant", "raw"))
                        f = inside(root, q.get("file"))
                        if f is None or not f.lower().endswith((".jpg", ".jpeg", ".png")):
                            raise ApiError(400, "그림 경로가 이상합니다")
                        return self._file(f)
                elif method == "POST":
                    if path == "/api/label":
                        return self._send(200, app.put_label(self._body()))
                    if path == "/api/collect/start":
                        return self._send(200, app.collect_start(self._body()))
                    if path == "/api/collect/stop":
                        return self._send(200, app.collect_stop())
                    if path == "/api/preprocess":
                        return self._send(200, app.prep_start(self._body()))
                    if path == "/api/export":
                        return self._send(200, app.export(self._body()))
                self._send(404, {"error": "없는 주소: %s" % path})
            except ApiError as e:
                self._send(e.code, {"error": e.msg})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # 서버는 죽지 않는다
                print("오류 %s %s: %s: %s" % (method, path, type(e).__name__, e), flush=True)
                try:
                    self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})
                except OSError:
                    pass

        def _file(self, path):
            if not path or not os.path.isfile(path):
                return self._send(404, {"error": "없음"})
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            with open(path, "rb") as fh:
                self._send(200, fh.read(), ctype)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="듣는 주소 (같은 망의 다른 PC 에서 열려면 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--store", default=DEFAULT_STORE, help="수집 세션 폴더 (기본 ../data/drone_sessions)")
    ap.add_argument("--also", action="append", default=None, metavar="폴더", help="함께 보일 다른 수집 폴더 (index.jsonl 있는 곳)")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS, help="내보내기 폴더 (기본 ../data/drone_datasets)")
    a = ap.parse_args(argv)
    app = App(a.store, a.also or [], a.datasets)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    httpd.daemon_threads = True
    shown = "127.0.0.1" if a.host in ("0.0.0.0", "") else a.host
    print("드론 화면 수집 · 라벨링: http://%s:%d/   (세션 %s · Ctrl+C 로 끝)" % (shown, a.port, app.store), flush=True)
    print("  서버 %s · 임무코드 %s · 전처리 %s" % (app.collect_defaults["server"], "있음" if app.code else "없음 (화면에서 넣기)",
                                          "됨" if app.prep_ok else "안 됨 — " + app.prep_why), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.run is not None and app.run.running:
            app.run.stop("서버 종료")
        if app.prep is not None:
            app._stop_prep()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
