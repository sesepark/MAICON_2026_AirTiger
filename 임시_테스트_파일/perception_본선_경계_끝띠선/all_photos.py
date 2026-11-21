"""본선 사진 전부에 detect_boundary → 사진마다 결과(경계 · 고른 선 · 렌즈 · 걸린 시간)를 JSON 으로. argv: 패키지 폴더, 출력 파일, [몇 장마다 1장]"""
import glob, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np
from perception.drone_boundary import detect_boundary
STORE = "C:/Users/Admin/MAICON_label/개발/src/perception/data/drone_sessions"
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
out = []
for sp in sorted(glob.glob(STORE + "/match_*")):
    rows = sorted((json.loads(l) for l in open(sp + "/index.jsonl", encoding="utf-8")), key=lambda r: r["wall"])
    for k, r in enumerate(rows):
        if k % step and k >= 4:
            continue
        im = cv2.imdecode(np.fromfile(sp + "/" + r["file"], np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter(); b = detect_boundary(im); dt = time.perf_counter() - t0
        d = b.to_dict() if b.hom is not None else {"ok": False}
        out.append({"match": os.path.basename(sp)[6:], "file": r["file"], "k": k, "ok": bool(b.ok), "s": round(dt, 3), "combo": b.debug.get("combo"), "rows": b.debug.get("rows_px"),
                    "via": b.debug.get("via"), "support": b.debug.get("support"), "lens_note": next((n for n in b.notes if n.startswith("렌즈")), None),
                    "notes": [n for n in b.notes if not n.startswith("OK")], "tried": len(b.debug.get("tried") or []), "boundary": d})
    print(os.path.basename(sp), len(out), flush=True)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
