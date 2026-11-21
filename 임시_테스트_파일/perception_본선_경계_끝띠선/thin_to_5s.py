"""1 s 간격으로 온 판(172542 · 175723 앞 86 s)을 2 s · 5 s 로 솎아 넣으면 「오래 선 UGV(길 고르기에 넣는 것)」 지식이 얼마나 낡는가."""
import bisect, glob, json, math, os, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
NAV = "C:/Users/Admin/MAICON_ugv/개발/src/navigation"; sys.path.insert(0, NAV)
from navigation.drive.params import DriveParams
from navigation.drive.ugvs import UgvTracks
HERE = os.path.dirname(os.path.abspath(__file__)); LOGS = os.path.dirname(HERE) + "/logs"
for mid, var in (("20261119_172542", "start2"), ("20261119_175723", "start0")):
    det = json.load(open("%s/v1_%s.json" % (HERE, mid), encoding="utf-8"))
    rows = [json.loads(l) for l in open("%s/%s/navigation_debug.jsonl" % (LOGS, mid), encoding="utf-8")]
    S = [(r["t"], r["pose"]) for r in rows if r.get("event") == "state" and r.get("pose")]; T = [s[0] for s in S]
    frames = sorted((f for f in det["frames"] if f["interval"] == "1.0"), key=lambda f: f["wall"])
    def recent(t):
        i = bisect.bisect_right(T, t)
        return [(s[1][0], s[1][1]) for s in S[max(0, i - 4):i + 1] if -1.7 <= s[0] - t <= 0.3]
    p = DriveParams()
    for step in (2, 5):
        A, B = UgvTracks(p), UgvTracks(p)
        c = collections.Counter()
        for k, f in enumerate(frames):
            t = f["wall"]; pts = [(x, y) for x, y, s in f["det"][var] if s >= 0.5]; rc = recent(t)
            if not rc: continue
            A.on_frame(t, pts, rc)
            if k % step == 0:
                B.on_frame(t, pts, rc)
            pa = [(u.x, u.y) for u in A.fresh(t) if u.still_s() >= p.ugv_plan_parked_s]
            pb = [(u.x, u.y) for u in B.fresh(t) if u.still_s() >= p.ugv_plan_parked_s]
            c["frames"] += 1; c["A_discs"] += len(pa); c["B_discs"] += len(pb)
            for x, y in pb:
                if not any(math.hypot(x - q[0], y - q[1]) <= 0.25 for q in pts):
                    c["B_stale"] += 1   # 그 자리에 지금 아무도 없다
            for x, y in pa:
                if not any(math.hypot(x - q[0], y - q[1]) <= 0.2 for q in pb):
                    c["B_late"] += 1    # 1 s 로는 이미 아는 것을 아직 모른다
            for x, y in pa:
                if not any(math.hypot(x - q[0], y - q[1]) <= 0.25 for q in pts):
                    c["A_stale"] += 1
        print("%s 1 s 사진 %d장 → %d s 로 솎음: 길에 넣는 UGV·초  1 s 기준 %d (그중 빈 자리 %d) | 솎은 것 %d (빈 자리 %d = %.0f%%, 아직 모름 %d = 1 s 기준의 %.0f%%)" % (
            mid[-6:], c["frames"], step, c["A_discs"], c["A_stale"], c["B_discs"], c["B_stale"], 100.0 * c["B_stale"] / max(1, c["B_discs"]), c["B_late"], 100.0 * c["B_late"] / max(1, c["A_discs"])))
