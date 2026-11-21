"""백테스트 1단계: 공식 테스트 경기(2026-11-19 16:55 ~ 18:06)의 본선 카메라 드론 사진에 실제 UGV 모델을 돌려 자리를 낸다.
경기 중 코드(perception.drone_ugvs.UgvView)를 그대로 쓴다. 카메라 높이는 경기 중 값(2.55 m)."""
import glob
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
PKG = "C:/Users/Admin/MAICON_ugv/개발/src/perception"
sys.path.insert(0, PKG)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from perception import drone_ugvs  # noqa: E402
from perception.drone_boundary import Boundary  # noqa: E402

STORE = "C:/Users/Admin/MAICON_label/개발/src/perception/data/drone_sessions"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/bt"
os.makedirs(OUT, exist_ok=True)
HC = float(sys.argv[1]) if len(sys.argv) > 1 else 2.55

model = drone_ugvs.load_model(device="cpu")
print("model", model.path, model.names)
for spath in sorted(glob.glob(STORE + "/match_*")):
    mid = os.path.basename(spath)[6:]
    calib = json.load(open(spath + "/undist_map/calib.json", encoding="utf-8"))["entries"][0]
    boundary = Boundary.from_dict(calib["boundary"])
    view = drone_ugvs.UgvView(boundary, HC, arena=(6.12, 4.80), ugv_height=0.0, conf=0.25, outside_m=1.0)   # 높이 보정은 뒤에서 (floor 좌표로 저장)
    rows = [json.loads(line) for line in open(spath + "/index.jsonl", encoding="utf-8")]
    out, took = [], []
    for r in rows:
        img = cv2.imdecode(np.fromfile(spath + "/" + r["file"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        t0 = time.perf_counter()
        ugvs = view.detect(img, model.predict)
        took.append(time.perf_counter() - t0)
        out.append({"file": r["file"], "wall": r.get("wall"), "match_t": r.get("match_t"), "interval": (r.get("headers") or {}).get("X-Frame-Interval-S"),
                    "destroyed": (r.get("headers") or {}).get("X-Destroyed"), "floor": [[round(x, 3), round(y, 3), round(s, 3)] for x, y, s in ugvs]})
    json.dump({"match": mid, "hc": HC, "camera_n": [3.06, 2.4], "seen_y_020": drone_ugvs.UgvView(boundary, HC, arena=(6.12, 4.80)).seen_y(), "frames": out},
              open("%s/det_%s.json" % (OUT, mid), "w", encoding="utf-8"))
    n = [len(o["floor"]) for o in out]
    print(mid, "frames", len(out), "ugvs/frame med", sorted(n)[len(n) // 2] if n else None, "max", max(n) if n else None,
          "detect ms med %.0f" % (1000 * sorted(took)[len(took) // 2]) if took else "", "intervals", sorted({o["interval"] for o in out}), "destroyed", sorted({str(o["destroyed"]) for o in out}))
