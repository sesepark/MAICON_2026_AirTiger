import json, sys, os
sys.stdout.reconfigure(encoding="utf-8")
base = sys.argv[1]
for m in ["20261120_000001", "20261120_003129", "20261120_005241", "20261120_015006", "20261120_022335", "20261120_023923"]:
    rows = [json.loads(l) for l in open("%s/logs/%s/perception_ugvs.jsonl" % (base, m), encoding="utf-8")]
    view = next(r for r in rows if r.get("e") == "view")
    frames, last = [], None
    for r in rows:
        if r.get("e") != "ugvs":
            continue
        gap = None if last is None else r["stamp"] - last
        last = r["stamp"]
        frames.append({"wall": r["stamp"], "captured": r.get("captured"), "interval": None if gap is None else ("1.0" if gap < 1.5 else "2.0" if gap < 3.5 else "5.0"),
                       "took_ms": r.get("took_ms"), "det": {"robot": r["ugvs"]}})
    json.dump({"match": m, "views": {"robot": {"seen_y": view["seen_y"]}}, "frames": frames}, open("%s/venue/test/v1_%s.json" % (base, m), "w", encoding="utf-8"))
    print(m, len(frames), view)
