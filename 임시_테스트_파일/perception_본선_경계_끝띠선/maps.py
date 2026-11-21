"""경기 첫 사진 4장으로 지도(map_maker.compute, 상자 모델)를 만들어 상자 자리를 JSON 으로. argv: 패키지 폴더, 출력 파일"""
import glob, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[1])
from perception import map_maker
STORE = "C:/Users/Admin/MAICON_label/개발/src/perception/data/drone_sessions"
out = []
for sp in sorted(glob.glob(STORE + "/match_*")):
    rows = sorted((json.loads(l) for l in open(sp + "/index.jsonl", encoding="utf-8")), key=lambda r: r["wall"])
    for r in rows[:4]:
        jpeg = open(sp + "/" + r["file"], "rb").read()
        t0 = time.perf_counter()
        data, info = map_maker.compute(jpeg, camera="uav", camera_height=2.55, arena=[6.12, 4.80], box_detector="model")
        dt = time.perf_counter() - t0
        view = info.pop("view", None)
        boxes = [] if data is None else [[round(b["x"], 3), round(b["y"], 3)] if isinstance(b, dict) else list(b)[:2] for b in data.get("boxes", [])]
        out.append({"match": os.path.basename(sp)[6:], "file": r["file"], "s": round(dt, 2), "ok": info.get("ok"), "n": len(boxes), "boxes": boxes, "detector": info.get("box_detector"),
                    "notes": info.get("notes"), "strip": info.get("strip"), "divisor": info.get("divisor"), "border": info.get("border"), "base_panels": (info.get("base_panels") or {}).get("kinds"),
                    "lens": None if not view else view["boundary"].get("lens"), "data_keys": None if data is None else sorted(data.keys())})
        print(out[-1]["match"], r["file"], "ok", info.get("ok"), "n", len(boxes), info.get("box_detector"), "%.1fs" % dt, "aspect", None if not view else round(view["boundary"]["lens"]["aspect"], 2), flush=True)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
