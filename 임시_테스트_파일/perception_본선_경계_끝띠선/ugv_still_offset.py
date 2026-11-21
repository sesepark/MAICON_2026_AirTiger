"""서 있을 때만: 드론이 본 우리 자리 − 라이다 자세. 경계(시작 사진)별 · 자리별. 본선(v1_*) 과 연습(test/v1_*)."""
import bisect, glob, json, math, os, sys, statistics as st, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); LOGS = os.path.dirname(HERE) + "/logs"
def q(v, f):
    v = sorted(v); return v[min(len(v) - 1, int(len(v) * f))] if v else float("nan")
def run(src, label):
    Z = collections.defaultdict(list); FAM = collections.defaultdict(list); per = []
    for path in sorted(glob.glob(src + "/v1_*.json")):
        det = json.load(open(path, encoding="utf-8")); mid = det["match"]
        rows = [json.loads(l) for l in open("%s/%s/navigation_debug.jsonl" % (LOGS, mid), encoding="utf-8")]
        S = [(r["t"], r["pose"]) for r in rows if r.get("event") == "state" and r.get("pose")]
        T = [s[0] for s in S]
        map_h = next((r["size"][1] for r in rows if r.get("event") == "map"), 4.8); dy = (map_h - 4.8) / 2
        for name, view in det["views"].items():
            if name == "label": continue
            fam = "넓은" if view["seen_y"][1] > 4.13 else "좁은"
            offs = []
            for f in det["frames"]:
                t = f["wall"]; i = bisect.bisect_right(T, t); win = [s for s in S[max(0, i - 6):i + 1] if -2.2 <= s[0] - t <= 0.6]
                if len(win) < 3: continue
                xs = [w[1][0] for w in win]; ys = [w[1][1] for w in win]
                if max(xs) - min(xs) > 0.03 or max(ys) - min(ys) > 0.03: continue
                px, py = st.mean(xs), st.mean(ys)
                c = [(math.hypot(x - px, y + dy - py), x - px, y + dy - py) for x, y, s in f["det"][name] if s >= 0.5]
                if not c: continue
                d, ox, oy = min(c)
                if d > 0.8: continue
                offs.append((d, ox, oy, px, py))
            for o in offs:
                zone = "오른쪽 x>4.1" if o[3] > 4.08 else ("왼쪽 x<2.0" if o[3] < 2.04 else "가운데")
                Z[(zone, fam)].append(o); FAM[fam].append(o)
            per.append((mid[-6:], name, fam, len(offs), q([o[0] for o in offs], .5), q([o[0] for o in offs], .9)))
    print("==", label)
    for k in sorted(Z):
        v = Z[k]
        print("   %-12s 경계 %s: n %4d  |d| 50%% %.3f 90%% %.3f 최대 %.3f  dx 가운데 %+.3f dy 가운데 %+.3f  0.40 넘음 %d" % (k[0], k[1], len(v), q([o[0] for o in v], .5), q([o[0] for o in v], .9), max(o[0] for o in v), st.median(o[1] for o in v), st.median(o[2] for o in v), sum(o[0] > 0.4 for o in v)))
    return per
per = run(HERE, "본선 경기장 (서 있을 때)")
for r in per: print("      %s %s %s n %d 50%% %.3f 90%% %.3f" % r)
run(HERE + "/test", "연습 경기장 (서 있을 때)")
