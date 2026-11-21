"""본선 카메라 사진 6판을 지금 코드로: 경기 첫 사진 4장 각각에서 처음부터 찾은 경계(로봇이 하는 방식) + 라벨링 도구의 경계 → 경계마다 UGV 찾기."""
import glob, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "C:/Users/Admin/MAICON_ugv/개발/src/perception")
import cv2
import numpy as np
from perception import drone_ugvs
from perception.drone_boundary import Boundary, detect_boundary

STORE = "C:/Users/Admin/MAICON_label/개발/src/perception/data/drone_sessions"
OUT = sys.argv[1]
model = drone_ugvs.load_model(device="cpu")
for spath in sorted(glob.glob(STORE + "/match_*")):
    mid = os.path.basename(spath)[6:]
    rows = [json.loads(line) for line in open(spath + "/index.jsonl", encoding="utf-8")]
    def img(r):
        return cv2.imdecode(np.fromfile(spath + "/" + r["file"], np.uint8), cv2.IMREAD_COLOR)
    views, meta = {}, {}
    calib = json.load(open(spath + "/undist_map/calib.json", encoding="utf-8"))["entries"][0]
    views["label"] = Boundary.from_dict(calib["boundary"])
    for k in range(4):
        t0 = time.perf_counter()
        b = detect_boundary(img(rows[k]), prior=None)
        meta["start%d" % k] = {"ok": bool(b.ok), "via": b.debug.get("via"), "s": round(time.perf_counter() - t0, 2), "strip": b.strip, "cm_half": b.cm_half,
                               "notes": [str(n) for n in list(b.notes)[:6]], "file": rows[k]["file"]}
        if b.hom is not None:
            views["start%d" % k] = b
    out = {"match": mid, "meta": meta, "views": {}, "frames": []}
    uv = {}
    for name, b in views.items():
        v = drone_ugvs.UgvView(b, 2.55, arena=(6.12, 4.80), conf=0.25, outside_m=1.0)
        uv[name] = v
        out["views"][name] = {"seen_y": v.seen_y(), "boundary": b.to_dict()}
    for r in rows:
        im = img(r)
        if im is None:
            continue
        h = r.get("headers") or {}
        f = {"file": r["file"], "wall": r.get("wall"), "match_t": r.get("match_t"), "interval": h.get("X-Frame-Interval-S"), "destroyed": h.get("X-Destroyed"),
             "captured": h.get("X-Captured-At"), "det": {}}
        for name, v in uv.items():
            f["det"][name] = [[round(x, 3), round(y, 3), round(s, 3)] for x, y, s in v.detect(im, model.predict)]
        out["frames"].append(f)
    json.dump(out, open("%s/v1_%s.json" % (OUT, mid), "w", encoding="utf-8"))
    print(mid, "frames", len(out["frames"]), {k: (m["ok"], m["via"], m["s"]) for k, m in meta.items()}, {k: v["seen_y"] for k, v in out["views"].items()}, flush=True)
