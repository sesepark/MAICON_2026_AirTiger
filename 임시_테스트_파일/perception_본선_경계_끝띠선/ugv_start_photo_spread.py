import glob, json, math, os, sys, statistics as st, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
def q(v, f):
    v = sorted(v); return v[min(len(v) - 1, int(len(v) * f))] if v else float("nan")
ALL = []; ZONE = collections.defaultdict(list)
for path in sorted(glob.glob(HERE + "/v1_*.json")):
    det = json.load(open(path, encoding="utf-8")); mid = det["match"]
    names = [n for n in det["views"] if n.startswith("start")]
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            d = []
            for f in det["frames"]:
                A = [p for p in f["det"][names[a]] if p[2] >= 0.5]; B = [p for p in f["det"][names[b]] if p[2] >= 0.5]
                for x, y, _s in A:
                    if not B: continue
                    m = min(B, key=lambda p: math.hypot(p[0] - x, p[1] - y))
                    e = math.hypot(m[0] - x, m[1] - y)
                    if e < 0.5:
                        d.append((e, x, y, m[0] - x, m[1] - y))
            out.append((names[a], names[b], d))
            ALL += [e for e, *_ in d]
            for e, x, y, _dx, _dy in d:
                ZONE[(min(5, int(x / 1.02)), min(3, int(y / 1.2)))].append(e)
    print(mid, " ".join("%s-%s 50%% %.3f 95%% %.3f 최대 %.3f" % (a[5:], b[5:], q([e for e, *_ in d], .5), q([e for e, *_ in d], .95), max(e for e, *_ in d)) for a, b, d in out))
    # 사진마다 검출 수가 같은가
    cnt = collections.Counter(tuple(sum(1 for p in f["det"][n] if p[2] >= 0.5) for n in names) for f in det["frames"])
    same = sum(v for k, v in cnt.items() if len(set(k)) == 1)
    print("     검출 수가 네 경계에서 같은 사진 %d/%d" % (same, len(det["frames"])))
print("모두 (시작 사진 4장의 경계 사이, 같은 UGV 의 좌표 차이): 50%% %.3f 80%% %.3f 95%% %.3f 최대 %.3f (n %d)" % (q(ALL, .5), q(ALL, .8), q(ALL, .95), max(ALL), len(ALL)))
print("자리별 95% (x 6칸 왼→오, y 4칸 위가 큰 y):")
for yy in (3, 2, 1, 0):
    print("   ", "  ".join("%.2f(%4d)" % (q(ZONE[(xx, yy)], .95), len(ZONE[(xx, yy)])) if ZONE[(xx, yy)] else "  -      " for xx in range(6)))
