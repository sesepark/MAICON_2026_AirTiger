import json, glob, os, sys, datetime
sys.stdout.reconfigure(encoding="utf-8")
STORE = "C:/Users/Admin/MAICON_label/개발/src/perception/data/drone_sessions"
ms = {m["id"]: m for m in json.load(open(sys.argv[1], encoding="utf-8"))["matches"]}
for sp in sorted(glob.glob(STORE + "/match_*")):
    mid = os.path.basename(sp)[6:]
    rows = sorted((json.loads(l) for l in open(sp + "/index.jsonl", encoding="utf-8")), key=lambda r: r["wall"])
    seg, prev = [], None
    for r in rows:
        h = r.get("headers") or {}
        key = (h.get("X-Frame-Interval-S"), h.get("X-Destroyed"))
        if key != prev:
            seg.append([key, round(r["wall"] - rows[0]["wall"], 1), round(r["wall"] - rows[0]["wall"], 1), 1]); prev = key
        else:
            seg[-1][2] = round(r["wall"] - rows[0]["wall"], 1); seg[-1][3] += 1
    gaps = [b["wall"] - a["wall"] for a, b in zip(rows, rows[1:])]
    lat = [r["wall"] - datetime.datetime.fromisoformat(r["headers"]["X-Captured-At"]).timestamp() for r in rows if (r.get("headers") or {}).get("X-Captured-At")]
    lat.sort()
    print(mid, "slot", ms.get(mid, {}).get("slot"), "play_s", ms.get(mid, {}).get("play_s"), "사진", len(rows), "마지막 사진 경기", round(rows[-1]["wall"] - rows[0]["wall"], 1), "s | 받음−촬영 최소 %.2f 50%% %.2f 90%% %.2f 최대 %.2f" % (lat[0], lat[len(lat)//2], lat[int(len(lat)*.9)], lat[-1]))
    for key, a, b, n in seg:
        print("    간격 %s s (파괴 %s): 경기 %5.1f ~ %5.1f s, %d장" % (key[0], key[1], a, b, n))
    big = sorted(gaps)[-3:]
    print("    사진 사이 가장 긴 틈:", [round(g, 1) for g in big])
