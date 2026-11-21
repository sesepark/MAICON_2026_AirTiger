import glob, json, math, os, sys, statistics as st
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
HU = float(sys.argv[1])
def lift(f, n, hc, hu):
    k = (hc - hu) / hc
    return (n[0] + (f[0] - n[0]) * k, n[1] + (f[1] - n[1]) * k)
alld = []
for path in sorted(glob.glob(HERE + "/bt/det_*.json")):
    det = json.load(open(path, encoding="utf-8")); mid = det["match"]
    rows = [json.loads(l) for l in open("%s/runs/%s/navigation_debug.jsonl" % (HERE, mid), encoding="utf-8")]
    S = [(r["t"], r["pose"], (r.get("cmd") or [0, 0])[0]) for r in rows if r.get("event") == "state" and r.get("pose")]
    ds = []; second = []
    for f in det["frames"]:
        t = f["wall"]
        near = [s for s in S if -1.6 <= s[0] - t <= 0.3]
        if not near: continue
        now = min(near, key=lambda s: abs(s[0] - t))
        if not (det["seen_y_020"][0] + 0.1 <= now[1][1] <= det["seen_y_020"][1] - 0.1): continue
        pts = [lift(q, det["camera_n"], det["hc"], HU) for q in f["floor"] if q[2] >= 0.5]
        dd = sorted(min(math.hypot(p[0] - s[1][0], p[1] - s[1][1]) for s in near) for p in pts)
        if dd:
            ds.append(dd[0]); second.append(dd[1] if len(dd) > 1 else 9.0)
    alld += list(zip(ds, second))
    q = sorted(ds)
    print(mid, "n", len(q), "가장 가까운 드론 UGV 까지: 50%% %.2f 80%% %.2f 90%% %.2f 95%% %.2f | ≤0.40 %.0f%% · 0.40~0.80 %.0f%% · >0.80 %.0f%%" % (
        q[len(q)//2], q[int(len(q)*.8)], q[int(len(q)*.9)], q[int(len(q)*.95)], 100*sum(d<=.4 for d in q)/len(q), 100*sum(.4<d<=.8 for d in q)/len(q), 100*sum(d>.8 for d in q)/len(q)))
n = len(alld)
print("모두 n", n, "≤0.40 %.0f%% · 0.40~0.80 %.0f%% · >0.80 %.0f%%" % (100*sum(d<=.4 for d,_ in alld)/n, 100*sum(.4<d<=.8 for d,_ in alld)/n, 100*sum(d>.8 for d,_ in alld)/n),
      "| 첫째 ≤0.40 인데 둘째도 ≤0.80: %d" % sum(1 for d, s in alld if d <= .4 and s <= .8))
