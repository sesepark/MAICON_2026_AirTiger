"""경기 한 판의 기록을 파일로 남긴다 — 연습 경기장에서 돌린 뒤 무슨 일이 있었는지 다시 볼 수 있게.

기록 폴더 (team_node.py 가 켠 경우 ~/team/log/<날짜_시각>/supervisor/, 따로 켠 경우 <record_dir>/<날짜_시각>_<run_id>/)
  events.jsonl   일어난 일을 한 줄씩 (경기 신호 · 0.5초마다의 위치와 하는 일 · 이동 목표와 도착 · 팬틸트 · 발사와 답 · 판독 요청과 답 · 보고와 서버의 답 …)
  photos.jsonl   사진마다 「찍혀야 할 면 · 판독 답 · 짝지은 결과」
  photos/        판독에 보낸 사진 (JPEG, cv2 가 없으면 PPM)
  meta.json      그때의 설정 (비밀 값은 넣지 않는다)

줄마다 wall(벽시계 초) · t(경기 시각 초, 시작 전이면 null) 가 붙는다.
기록하다 생긴 오류는 삼킨다 — 기록 때문에 경기가 멈추면 안 된다. 임무코드 · API 키는 여기로 오지 않는다 (supervisor 는 갖고 있지 않다).
"""
from __future__ import annotations

import json
import os
import threading
import time

SESSION_ENV = "TEAM_LOG_DIR"  # team_node.py 가 정해 주는 이번 실행의 기록 폴더
OFF_ENV = "TEAM_LOG_OFF"  # "1" 이면 기록을 남기지 않는다 (디스크가 거의 찼을 때 team_node.py 가 켠다)
LIGHT_ENV = "TEAM_LOG_LIGHT"  # "1" 이면 사진은 남기지 않는다 (글 기록만)


def photos_allowed():
    return os.environ.get(LIGHT_ENV) != "1"


def session_folder(record_dir, run_id, name="supervisor"):
    """이번 실행의 기록 폴더 경로. record_dir 가 빈 문자열이면 None (남기지 않는다)."""
    if not record_dir or os.environ.get(OFF_ENV) == "1":
        return None
    shared = os.environ.get(SESSION_ENV)
    if shared:
        return os.path.join(shared, name)
    return os.path.join(record_dir, "%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), run_id))


def _load_photo_tools():
    """save_photo() 가 쓰는 cv2 · numpy 를 켜질 때 미리 불러 둔다. 없거나 못 불러도 그대로 넘어간다 (save_photo 가 그때 다시 본다).

    2026-11-19 실차 세 판의 기록: 경기의 첫 사진을 찍는 틱만 0.3 ~ 0.6 s 더 걸렸다 (그 뒤의 사진은 아님). cv2 를 그 틱에서 처음
    불러오기 때문으로 본다 [추정 — 개발 PC 에서 cv2 를 불러오는 데 0.09 s, 로봇은 그 4 ~ 6배]. 켜질 때(시작 대기 중) 불러 두면 경기 중에는 안 걸린다.
    """
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001 — 기록 때문에 프로그램이 못 켜지면 안 된다
        pass


class Recorder:
    def __init__(self, folder, meta=None):
        self.dir = None
        self.clock = lambda: None
        self._n = 0
        self._lock = threading.Lock()  # 메인 스레드와 받기 스레드가 같이 쓴다
        self._files = {}
        if not folder:
            return
        try:
            os.makedirs(os.path.join(folder, "photos"), exist_ok=True)
            self._n = len(os.listdir(os.path.join(folder, "photos")))  # 같은 폴더에 이어 쓸 때 사진 번호가 겹치지 않게
            with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(dict(meta or {}, started_at=time.strftime("%Y-%m-%d %H:%M:%S")), fh, ensure_ascii=False, indent=1, default=str)
            self.dir = folder
        except OSError:
            self.dir = None
        if self.dir is not None and photos_allowed():
            _load_photo_tools()

    def _write(self, name, record):
        if self.dir is None:
            return
        try:
            t = self.clock()
            record = dict(record, wall=round(time.time(), 3), t=None if t is None else round(t, 2))
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            with self._lock:
                fh = self._files.get(name)
                if fh is None:
                    fh = self._files[name] = open(os.path.join(self.dir, name), "a", encoding="utf-8")
                fh.write(line)
                fh.flush()  # 프로그램이 강제로 끝나도 여기까지는 남는다
        except (OSError, TypeError, ValueError):
            pass

    def event(self, name, /, **fields):
        """일어난 일 한 줄. name 은 "e" 칸에 들어간다 (fields 에 kind · name 같은 이름을 써도 겹치지 않는다)."""
        self._write("events.jsonl", dict(fields, e=name))

    def photo_record(self, record):
        self._write("photos.jsonl", record)

    def close(self):
        with self._lock:
            for fh in self._files.values():
                try:
                    fh.close()
                except OSError:
                    pass
            self._files = {}

    def save_photo(self, tag, msg):
        """sensor_msgs/Image 한 장을 파일로. 돌려주는 값: 파일 이름 (못 썼으면 None)."""
        if self.dir is None:
            return None
        with self._lock:
            self._n += 1
            number = self._n
        stem = "%04d_%s" % (number, "-".join(str(v) for v in tag).replace("/", "_").replace(" ", ""))
        try:
            width, height, data = int(msg.width), int(msg.height), bytes(msg.data)
            size = width * height * 3
            if msg.encoding not in ("rgb8", "bgr8") or len(data) < size or size <= 0:
                return None
            try:
                import cv2
                import numpy as np

                img = np.frombuffer(data, dtype=np.uint8)[:size].reshape(height, width, 3)
                if msg.encoding == "rgb8":
                    img = img[:, :, ::-1]
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    name = stem + ".jpg"
                    with open(os.path.join(self.dir, "photos", name), "wb") as fh:
                        fh.write(buf.tobytes())
                    return name
            except ImportError:
                pass
            if msg.encoding == "bgr8":  # PPM 은 R·G·B 순서
                rgb = bytearray(size)
                rgb[0::3], rgb[1::3], rgb[2::3] = data[2:size:3], data[1:size:3], data[0:size:3]
                data = bytes(rgb)
            name = stem + ".ppm"
            with open(os.path.join(self.dir, "photos", name), "wb") as fh:
                fh.write(b"P6\n%d %d\n255\n" % (width, height) + data[:size])
            return name
        except (OSError, ValueError, AttributeError):
            return None
