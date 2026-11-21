import json, sys, math, statistics as st, collections
sys.stdout.reconfigure(encoding="utf-8")
base = json.load(open("map_base.json", encoding="utf-8")); new = json.load(open("map_new.json", encoding="utf-8"))
print("예:", new[0]["boxes"][:4], new[0]["data_keys"])
def sym(boxes):
    out = []
    for x, y in boxes:
        if x < 3.06 - 0.25:
            c = [(abs(bx - (6.12 - x)) + abs(by - y), bx, by) for bx, by in boxes if bx > 3.06 + 0.25]
            if c and min(c)[0] < 0.45:
                _d, bx, by = min(c)
                out.append((bx - (6.12 - x), by - y, x))
    return out
G = collections.defaultdict(list)
for label, maps in (("전", base), ("후", new)):
    for m in maps:
        fam = "확장 렌즈" if m["lens"] and abs(m["lens"]["aspect"] - 1.0) > 1e-6 else "단순 렌즈"
        G[(label, fam)] += sym(m["boxes"])
for k in sorted(G):
    v = G[k]
    print("%s · %s: 좌우 짝 %d쌍 | 오른쪽 상자 x − (왼쪽 상자의 거울 자리) 가운데 %+.3f m · |값| 50%% %.3f 90%% %.3f | y 차이 |값| 50%% %.3f" % (
        k[0], k[1], len(v), st.median(a for a, _b, _x in v), st.median(abs(a) for a, _b, _x in v), sorted(abs(a) for a, _b, _x in v)[int(len(v) * .9)], st.median(abs(b) for _a, b, _x in v)))
# 같은 사진의 전 ↔ 후 상자 자리 차이
D = collections.defaultdict(list)
for mb, mn in zip(base, new):
    fam = "전이 확장" if abs(mb["lens"]["aspect"] - 1.0) > 1e-6 else "전이 단순"
    for x, y in mb["boxes"]:
        c = min(((math.hypot(bx - x, by - y), bx - x, by - y) for bx, by in mn["boxes"]), default=None)
        if c and c[0] < 0.5:
            zone = "오른쪽 x>4.1" if x > 4.08 else ("왼쪽 x<2.0" if x < 2.04 else "가운데")
            D[(fam, zone)].append(c)
for k in sorted(D):
    v = D[k]
    print("같은 사진 전→후 (%s · %s): 상자 %d개 | 옮겨 간 거리 50%% %.3f 90%% %.3f 최대 %.3f | dx 가운데 %+.3f dy 가운데 %+.3f" % (k[0], k[1], len(v), st.median(d for d, *_ in v), sorted(d for d, *_ in v)[int(len(v) * .9)], max(d for d, *_ in v), st.median(a for _d, a, _b in v), st.median(b for _d, _a, b in v)))
# 사진 사이의 흔들림: 같은 경기 네 장에서 같은 상자의 좌표 퍼짐
for label, maps in (("전", base), ("후", new)):
    sp = []
    bym = collections.defaultdict(list)
    for m in maps: bym[m["match"]].append(m["boxes"])
    for mid, lst in bym.items():
        for x, y in lst[0]:
            pts = [(x, y)]
            for other in lst[1:]:
                c = min(((math.hypot(bx - x, by - y), bx, by) for bx, by in other), default=None)
                if c and c[0] < 0.5: pts.append((c[1], c[2]))
            if len(pts) >= 3:
                sp.append((max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in pts for b in pts), x))
    r = [s for s, x in sp if x > 4.08]; o = [s for s, x in sp if x <= 4.08]
    print("%s: 같은 경기 첫 사진 4장 사이 같은 상자의 좌표 차이(최대) — 오른쪽 50%% %.3f 90%% %.3f 최대 %.3f · 나머지 50%% %.3f 90%% %.3f 최대 %.3f" % (
        label, st.median(r), sorted(r)[int(len(r) * .9)], max(r), st.median(o), sorted(o)[int(len(o) * .9)], max(o)))
