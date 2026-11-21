"""tools/label_drone_frames.py (수집 · 라벨 웹 화면) · export_drone_labels.py 시험. 표준 라이브러리만, 네트워크는 이 PC 안.

  python3 -m unittest discover -s test -p "test_label_drone_frames.py"     (패키지 폴더에서, 약 10초)

가짜 드론 서버가 0.5초마다 새 JPEG 를 준다 (임무코드 GOOD 만, 아니면 401). 수집은 진짜 collect_drone_frames.py 를 띄운다.
렌즈 보정(undistort_drone_frames.py)은 numpy · OpenCV 가 필요해서 여기서는 시험하지 않는다.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import export_drone_labels  # noqa: E402
import label_drone_frames  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))


class FakeDrone(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if qs.get("mission_code") != ["GOOD"]:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"detail":"bad code"}')
            return
        n = int(time.time() * 2)  # 0.5초마다 새 장
        t = dt.datetime.fromtimestamp(n / 2, KST).isoformat(timespec="milliseconds")
        body = b"\xff\xd8" + str(n).encode() + b"\xff\xd9"
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("X-Captured-At", t)
        self.send_header("X-Camera", qs["camera"][0])
        self.send_header("X-Test-Mode", "on")
        self.end_headers()
        self.wfile.write(body)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def serve(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, "http://127.0.0.1:%d" % httpd.server_address[1]


class LabelServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="label_test_")
        label_drone_frames.LOCAL_CONF = os.path.join(cls.tmp, "no_conf.json")  # 진짜 로컬 설정(임무코드)을 읽지 않게
        cls.drone, cls.drone_url = serve(FakeDrone)
        cls.app = label_drone_frames.App(os.path.join(cls.tmp, "sessions"), [], os.path.join(cls.tmp, "datasets"))
        cls.app.code = None
        cls.web, cls.url = serve(label_drone_frames.make_handler(cls.app))

    @classmethod
    def tearDownClass(cls):
        if cls.app.run is not None:
            cls.app.run.stop()
        for httpd in (cls.web, cls.drone):
            httpd.shutdown()
            httpd.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"null")

    def test_full_flow(self):
        # 시작 → 몇 장 → 정지
        code, r = self.call("/api/collect/start", {"server": self.drone_url, "code": "GOOD", "cameras": ["test1", "main"], "name": "배치A", "interval": 1})
        self.assertEqual(code, 200, r)
        sid = r["session"]
        self.assertTrue(sid.endswith("_배치A"))
        code, r = self.call("/api/collect/start", {"server": self.drone_url, "code": "GOOD"})
        self.assertEqual(code, 409)  # 이미 수집 중
        self.assertNotIn("GOOD", " ".join(self.app.run.proc.args))  # 임무코드는 명령줄에 없다 (환경변수로)
        time.sleep(3.5)
        code, st = self.call("/api/status")
        self.assertTrue(st["collect"]["running"])
        self.assertTrue(st["code_set"])
        self.assertNotIn("GOOD", json.dumps(st))  # 임무코드는 응답에 없다
        self.assertGreaterEqual(st["collect"]["cameras"]["test1"]["saved"], 2)
        self.assertEqual(self.call("/api/collect/stop", {})[0], 200)
        code, st = self.call("/api/status")
        self.assertFalse(st["collect"]["running"])
        log = read(os.path.join(self.app.store, sid, "collect.log"))
        self.assertNotIn("GOOD", log)

        # 세션 · 그림 목록 · 그림
        code, sessions = self.call("/api/sessions")
        self.assertIn(sid, [s["id"] for s in sessions])
        code, imgs = self.call("/api/images?" + urllib.parse.urlencode({"session": sid, "variant": "raw"}))
        self.assertGreaterEqual(len(imgs), 4)
        first = imgs[0]["file"]
        code, raw = self.call("/img?" + urllib.parse.urlencode({"session": sid, "variant": "raw", "file": first}))
        self.assertEqual(code, 200)
        self.assertTrue(raw.startswith(b"\xff\xd8"))

        # 전처리 흉내: 원본을 undist_map 으로 복사 (진짜 전처리는 numpy · OpenCV 가 필요)
        spath = os.path.join(self.app.store, sid)
        os.makedirs(os.path.join(spath, "undist_map"), exist_ok=True)
        with open(os.path.join(spath, "undist_map", "index.jsonl"), "w", encoding="utf-8") as fh:
            for x in imgs:
                dst = os.path.join(spath, "undist_map", x["file"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(spath, x["file"]), dst)
                fh.write(json.dumps({"file": x["file"], "src": x["file"], "camera": x["camera"], "calib": x["camera"] + "-1"}) + "\n")
        # map 의 out_to_field: x = u/100, y = 5.00 − v/100 (상자 윗면 평면 0.40 m)
        o2f = [[0.01, 0, 0], [0, -0.01, 5.0], [0, 0, 1]]
        with open(os.path.join(spath, "undist_map", "calib.json"), "w", encoding="utf-8") as fh:
            json.dump({"entries": [{"id": c + "-1", "camera": c, "mode": "map", "out_to_field": o2f, "plane_h": 0.4,
                                    "camera_xyh": [3.06, 2.5, 2.9], "boundary": {"field_w": 6.12, "field_h": 5.0}}
                                   for c in ("test1", "main")]}, fh)
        code, sessions = self.call("/api/sessions")
        self.assertIn("undist_map", [v["id"] for s in sessions if s["id"] == sid for v in s["variants"]])
        # 원본에는 라벨을 달지 않는다
        code, r = self.call("/api/label", {"session": sid, "variant": "raw", "image": first, "width": 1000, "height": 500, "status": "done", "boxes": []})
        self.assertEqual(code, 400, r)

        # 밖으로 나가는 경로는 막는다
        for bad in ("../../../etc/passwd", "/etc/passwd", "..\\..\\x.jpg"):
            code, _ = self.call("/img?" + urllib.parse.urlencode({"session": sid, "variant": "raw", "file": bad}))
            self.assertIn(code, (400, 404), bad)
        code, _ = self.call("/api/label", {"session": sid, "variant": "undist_map", "image": "../../x.jpg", "status": "done", "boxes": []})
        self.assertEqual(code, 400)
        code, _ = self.call("/api/images?" + urllib.parse.urlencode({"session": sid, "variant": "../x"}))
        self.assertEqual(code, 400)

        # 라벨 저장 · 읽기
        box = {"cls": "box", "pts": [[100, 50], [200, 50], [200, 150], [100, 150]]}
        code, r = self.call("/api/label", {"session": sid, "variant": "undist_map", "image": first, "width": 1000, "height": 500,
                                           "status": "done", "boxes": [box, {"cls": "stripe", "pts": [[0, 0], [10, 0], [10, 10], [0, 10]]}]})
        self.assertEqual(code, 200, r)
        code, lab = self.call("/api/label?" + urllib.parse.urlencode({"session": sid, "variant": "undist_map", "image": first}))
        self.assertEqual(lab["status"], "done")
        self.assertEqual(len(lab["boxes"]), 2)
        code, _ = self.call("/api/label", {"session": sid, "variant": "undist_map", "image": first, "status": "done", "boxes": [{"cls": "cat", "pts": box["pts"]}]})
        self.assertEqual(code, 400)  # 없는 종류
        # 아무것도 안 한 draft 는 파일을 만들지 않는다
        code, r = self.call("/api/label", {"session": sid, "variant": "undist_map", "image": imgs[1]["file"], "width": 1000, "height": 500, "status": "draft", "boxes": []})
        self.assertFalse(r["saved"])
        # 박스 없는 done (음성 예)
        self.call("/api/label", {"session": sid, "variant": "undist_map", "image": imgs[2]["file"], "width": 1000, "height": 500, "status": "done", "boxes": []})
        self.call("/api/label", {"session": sid, "variant": "undist_map", "image": imgs[3]["file"], "width": 1000, "height": 500, "status": "skip", "boxes": []})
        code, imgs = self.call("/api/images?" + urllib.parse.urlencode({"session": sid, "variant": "undist_map"}))
        self.assertEqual([x.get("status") for x in imgs[:4]], ["done", None, "done", "skip"])

        # 내보내기
        code, res = self.call("/api/export", {"variant": "undist_map", "val_ratio": 0.5, "sessions": [sid]})
        self.assertEqual(code, 200, res)
        self.assertEqual(res["counts"]["train"] + res["counts"]["val"], 2)  # done 2장 (skip · 라벨 없는 장 제외)
        self.assertEqual(res["boxes"], {"box": 1, "stripe": 1})
        out = res["out"]
        txts = sorted(os.listdir(os.path.join(out, "labels", "train")) + os.listdir(os.path.join(out, "labels", "val")))
        self.assertEqual(len(txts), 2)
        part = "train" if os.path.exists(os.path.join(out, "labels", "train", txts[0])) else "val"
        lines = read(os.path.join(out, "labels", part, txts[0])).split("\n")
        self.assertEqual(lines[0], "0 0.100000 0.100000 0.200000 0.100000 0.200000 0.300000 0.100000 0.300000")
        self.assertTrue(lines[1].startswith("1 0.000000 0.000000 0.010000"))
        self.assertIn("names:\n  0: box\n  1: stripe", read(os.path.join(out, "data.yaml")))
        # 경기장 좌표 (지도와 같은 frame "arena")
        self.assertEqual(res["fieldmaps"], 2)
        self.assertEqual(res["plane_h"], [0.4])
        fm = json.loads(read(os.path.join(out, "fieldmaps", part, txts[0].replace(".txt", ".json"))))
        self.assertEqual(fm["frame_id"], "arena")
        b0 = fm["boxes"][0]
        self.assertEqual((b0["cls"], b0["x"], b0["y"], b0["side_m"], b0["yaw_deg"]), ("box", 1.5, 4.0, 1.0, 0.0))
        self.assertEqual(b0["corners"][0], [1.0, 4.5])

    def test_bad_code_shows_401(self):
        code, r = self.call("/api/collect/start", {"server": self.drone_url, "code": "WRONG", "cameras": ["test2"]})
        self.assertEqual(code, 200)
        for _ in range(40):
            time.sleep(0.25)
            st = self.call("/api/status")[1]["collect"]
            if st["cameras"]["test2"]["errors"]:
                break
        self.assertIn("401", st["cameras"]["test2"]["state"])  # 수집기는 계속 다시 시도한다 → 화면에 이유를 보여 준다
        self.assertEqual(self.call("/api/collect/stop", {})[0], 200)
        self.assertFalse(self.call("/api/status")[1]["collect"]["running"])

    def test_defaults(self):
        d = label_drone_frames.App(os.path.join(self.tmp, "s2"), [], os.path.join(self.tmp, "d2")).collect_defaults  # 새로 띄운 서버
        self.assertEqual(d["server"], label_drone_frames.DEFAULT_SERVER)
        self.assertEqual(d["interval"], 7.0)
        self.assertEqual(d["cameras"], ["test1", "test2"])
        self.assertEqual(self.call("/api/collect/start", {"server": self.drone_url, "code": "GOOD", "interval": 0.2})[0], 400)

    def test_start_needs_code_and_server(self):
        if self.app.run is not None and self.app.run.running:
            self.app.run.stop()
        saved, self.app.code = self.app.code, None
        try:
            self.assertEqual(self.call("/api/collect/start", {"server": self.drone_url})[0], 400)
            self.assertEqual(self.call("/api/collect/start", {"server": "nope", "code": "GOOD"})[0], 400)
            self.assertEqual(self.call("/api/collect/start", {"server": self.drone_url, "code": "GOOD", "resume": "../x"})[0], 400)
        finally:
            self.app.code = saved


class SplitTest(unittest.TestCase):
    def test_by_session(self):
        a = export_drone_labels.split([("s1", [1] * 10), ("s2", [1] * 3), ("s3", [1] * 4)], 0.2)
        self.assertEqual(a["s3"], ["val"] * 4)  # 뒤 세션 통째로
        self.assertEqual(a["s1"], ["train"] * 10)
        self.assertEqual(a["s2"], ["train"] * 3)

    def test_first_session_always_train(self):
        a = export_drone_labels.split([("s1", [1] * 2), ("s2", [1] * 2)], 0.9)
        self.assertEqual(a["s1"], ["train"] * 2)
        self.assertEqual(a["s2"], ["val"] * 2)

    def test_single_session_tail(self):
        a = export_drone_labels.split([("s1", list(range(10)))], 0.2)
        self.assertEqual(a["s1"], ["train"] * 8 + ["val"] * 2)

    def test_zero(self):
        self.assertEqual(export_drone_labels.split([("s1", [1, 2])], 0.0)["s1"], ["train", "train"])


if __name__ == "__main__":
    unittest.main()
