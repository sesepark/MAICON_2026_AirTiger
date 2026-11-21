import glob, json, os, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np
from perception import drone_boundary as db
STORE = "C:/Users/Admin/MAICON_label/개발/src/perception/data/drone_sessions"
p = db.BoundaryParams()
def q(v, f):
    v = sorted(v); return v[min(len(v) - 1, int(len(v) * f))] if v else float("nan")
R = collections.defaultdict(list); ratio = []; miss = collections.Counter(); n = 0
for sp in sorted(glob.glob(STORE + "/match_*")):
    rows = sorted((json.loads(l) for l in open(sp + "/index.jsonl", encoding="utf-8")), key=lambda r: r["wall"])
    for k, r in enumerate(rows):
        if k % 3: continue
        im = cv2.imdecode(np.fromfile(sp + "/" + r["file"], np.uint8), cv2.IMREAD_COLOR)
        mask = db.line_mask(im, p); hh = im.shape[0]; n += 1
        normal = {u for u, _ in db._vertical_peaks(mask, p)}
        loose = [u for u, _ in db._vertical_peaks(mask, p, p.peak_min * 0.6)]
        got = {}
        for name, c in (("349", 348), ("952", 952), ("972", 971.5), ("266", 266)):
            near = [u for u in loose if abs(u - c) <= 5]
            if not near: miss[name] += 1; continue
            u = min(near, key=lambda u: abs(u - c))
            t = db._track_vertical(mask, u, (0, hh), p)
            span = (t[:, 1].max() - t[:, 1].min()) if len(t) else 0
            got[name] = len(t)
            R[name].append(len(t)); R[name + "_span"].append(span)
            if u not in normal: miss[name + " (낮춘 문턱에서만)"] += 1
        if "952" in got and "972" in got: ratio.append(got["952"] / max(1, got["972"]))
print("사진", n, "| 봉우리 없음:", dict(miss))
for name in ("349", "952", "972", "266"):
    v = R[name]; s = R[name + "_span"]
    print("  선 u≈%s: 따라간 점 수 최소 %d 5%% %d 50%% %d 95%% %d 최대 %d · 세로로 걸친 길이 px 5%% %d 50%% %d" % (name, min(v), q(v, .05), q(v, .5), q(v, .95), max(v), q(s, .05), q(s, .5)))
print("  952 / 972 점 수 비: 최소 %.2f 5%% %.2f 50%% %.2f 95%% %.2f · 1.2 못 넘는 사진 %d / %d" % (min(ratio), q(ratio, .05), q(ratio, .5), q(ratio, .95), sum(x < 1.2 for x in ratio), len(ratio)))
