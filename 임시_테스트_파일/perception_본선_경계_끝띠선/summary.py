"""all_*.json 요약: 통과 · 렌즈 · 고른 끝 띠 선 · 시간, 그리고 사진마다의 경계가 같은 화면 픽셀을 경기장의 어디로 옮기는지의 퍼짐. argv: 패키지 폴더, json..., [-ref json]"""
import json, sys, collections, statistics as st
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[1])
import numpy as np
from perception.drone_boundary import Boundary
args = sys.argv[2:]
ref_path = args[args.index("-ref") + 1] if "-ref" in args else None
files = [a for a in args if a.endswith(".json") and a != ref_path]
def q(v, f):
    v = sorted(v); return v[min(len(v) - 1, int(len(v) * f))] if v else float("nan")
# 화면 픽셀 격자 (경기장 안쪽에 드는 것만 쓴다)
gu, gv = np.meshgrid(np.arange(200, 1100, 30), np.arange(20, 710, 30))
PX = np.stack([gu.ravel(), gv.ravel()], 1).astype(float)
def field(d):
    return Boundary.from_dict(d).pixel_to_field(PX)
REF = None
def reference(rows):
    full = [field(r["boundary"]) for r in rows if r["ok"] and abs(r["boundary"]["lens"]["aspect"] - 1.0) > 1e-6 and r["via"] == "search" and r["combo"] and r["combo"][0] is not None]
    return np.median(np.stack(full), 0)
for path in files:
    rows = json.load(open(path, encoding="utf-8"))
    n = len(rows)
    kind = collections.Counter()
    for r in rows:
        if not r["ok"]: kind["실패"] += 1; continue
        note = r["lens_note"] or ""
        red = any("축소" in t for t in r["notes"])
        simple = abs(r["boundary"]["lens"]["aspect"] - 1.0) < 1e-6
        kind[("축소 맞추기" if red else "세로선 5개") + (" · 단순 렌즈" if simple else " · 확장 렌즈")] += 1
    rs = collections.Counter(None if not r["combo"] else r["combo"][4] for r in rows)
    bs = collections.Counter(None if not r["combo"] else r["combo"][0] for r in rows)
    t = [r["s"] for r in rows]
    print("==", path, "사진", n)
    print("   종류:", dict(kind))
    print("   고른 RED 끝 띠 선(u):", dict(rs.most_common(6)), "· BLUE:", dict(bs.most_common(5)))
    print("   걸린 시간 s: 50%% %.2f 90%% %.2f 최대 %.2f · 정밀 맞추기 횟수 50%% %d 90%% %d 최대 %d" % (q(t, .5), q(t, .9), max(t), q([r["tried"] for r in rows], .5), q([r["tried"] for r in rows], .9), max(r["tried"] for r in rows)))
    if REF is None:
        REF = reference(json.load(open(ref_path, encoding="utf-8")) if ref_path else rows)
    inside = (REF[:, 0] > 0.1) & (REF[:, 0] < 6.02) & (REF[:, 1] > 0.3) & (REF[:, 1] < 4.7)
    zones = {"왼쪽 x<2.0": inside & (REF[:, 0] < 2.04), "가운데": inside & (REF[:, 0] >= 2.04) & (REF[:, 0] <= 4.08), "오른쪽 x>4.1": inside & (REF[:, 0] > 4.08)}
    dev = {z: [] for z in zones}; worst = []
    for r in rows:
        if not r["ok"]: continue
        d = np.linalg.norm(field(r["boundary"]) - REF, axis=1)
        for z, m in zones.items(): dev[z] += list(d[m])
        worst.append((float(np.nanmax(d[inside])), r["match"][-6:], r["file"], r["lens_note"][:40] if r["lens_note"] else None, r["combo"]))
    for z in zones:
        print("   기준 경계와의 차이 %-12s 50%% %.3f 95%% %.3f 99%% %.3f 최대 %.3f m" % (z, q(dev[z], .5), q(dev[z], .95), q(dev[z], .99), max(dev[z])))
    w = [x[0] for x in worst]
    print("   사진별 가장 큰 차이: 50%% %.3f 90%% %.3f 최대 %.3f · 0.10 m 넘는 사진 %d장 · 0.20 m 넘는 사진 %d장" % (q(w, .5), q(w, .9), max(w), sum(x > 0.10 for x in w), sum(x > 0.20 for x in w)))
    for x in sorted(worst, reverse=True)[:4]: print("        ", x)
