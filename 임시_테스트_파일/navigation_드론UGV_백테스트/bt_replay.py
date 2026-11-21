"""백테스트 2단계: 드론 UGV 자리(bt_detect.py) + 그 경기의 navigation 기록(자세 2 Hz · 지도에 없는 점 · 목표 결과)을 새 코드(drive/ugvs.UgvTracks)에 넣어 본다.

  A. 우리 차 고르기: 맞춘 비율 · 드론이 본 우리 자리 − 기록된 자세 (UGV 높이 보정값별)
  B. 막힌 목표(BLOCKED)마다: 막은 점들을 드론으로 보면 무엇이었나 (서 있는 UGV · 움직이던 UGV · 곁에 UGV 없음 · 드론 자리 없음)
"""
import collections
import glob
import json
import math
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "C:/Users/Admin/MAICON_ugv/개발/src/navigation")
from navigation.drive.params import DriveParams  # noqa: E402
from navigation.drive.ugvs import UgvTracks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
VERBOSE = "-v" in sys.argv


def lift(floor, n, hc, hu):
    k = (hc - hu) / hc
    return (n[0] + (floor[0] - n[0]) * k, n[1] + (floor[1] - n[1]) * k)


def pose_at(states, t):
    """기록된 자세 (2 Hz) 를 t 로 곧게 이어 본다. 0.8 s 넘게 비면 None."""
    lo, hi = 0, len(states) - 1
    if not states or t < states[0][0] - 0.8 or t > states[-1][0] + 0.8:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if states[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    a, b = states[lo], states[hi]
    if b[0] - a[0] > 1.6 or b[0] == a[0]:
        return a[1] if abs(t - a[0]) <= 0.8 else None
    k = min(1.0, max(0.0, (t - a[0]) / (b[0] - a[0])))
    return (a[1][0] + (b[1][0] - a[1][0]) * k, a[1][1] + (b[1][1] - a[1][1]) * k, a[1][2])


def blockers(pose, pts, p):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    body = [((c * (x - pose[0]) + s * (y - pose[1])), (-s * (x - pose[0]) + c * (y - pose[1])), (x, y)) for x, y in pts]
    body = [b for b in body if abs(b[0]) > p.robot_half_length + 0.03 or abs(b[1]) > p.robot_half_width + 0.03]   # 우리 몸에 찍힌 점 (옛 빌드의 기록에는 남아 있다)
    block = [w for bx, by, w in body if 0.0 < bx <= p.robot_half_length + 1.3 and abs(by) <= 0.30]
    return block or [w for bx, by, w in body if math.hypot(bx, by) <= 0.5]


def kind_of(tracks, t, block, p):
    if not tracks.alive(t):
        return ("nofeed", 0.0, None)
    best = None
    for u in tracks.fresh(t):
        n = sum(1 for x, y in block if math.hypot(u.x - x, u.y - y) <= p.ugv_match_m)
        if n and (best is None or n > best[0]):
            best = (n, u)
    if best is None or best[0] < 0.6 * len(block):
        return ("none", 0.0, None)
    still = best[1].still_s()
    return ("parked" if still >= p.ugv_parked_s else "moving", still, (round(best[1].x, 2), round(best[1].y, 2)))


total = collections.Counter()
saved_total = 0.0
all_res = []
for path in sorted(glob.glob(HERE + "/bt/det_*.json")):
    det = json.load(open(path, encoding="utf-8"))
    mid = det["match"]
    rows = [json.loads(line) for line in open("%s/runs/%s/navigation_debug.jsonl" % (HERE, mid), encoding="utf-8")]
    states = [(r["t"], r["pose"], r) for r in rows if r.get("event") == "state" and r.get("pose")]
    p = DriveParams()
    tracks = UgvTracks(p)
    n, hc = det["camera_n"], det["hc"]
    map_h = next((r["size"][1] for r in rows if r.get("event") == "map"), 4.8)
    dy = (map_h - 4.8) / 2.0   # bt_detect 는 4.80 지도 좌표로 냈다
    n = [n[0], n[1] + dy]
    seen = [det["seen_y_020"][0] + dy, det["seen_y_020"][1] + dy]
    frames = [f for f in det["frames"] if f.get("wall")]
    events = [(f["wall"], 0, f) for f in frames] + [(t, 1, r) for t, _pose, r in states] + [(r["t"], 2, r) for r in rows if r.get("event") == "result"]
    events.sort(key=lambda e: (e[0], e[1]))
    used = matched = 0
    res = []
    per_goal = collections.defaultdict(list)   # 목표 번호 → [(t, kind, still, 막는 점 가운데, 화면 안인가)]
    goal_no = {}
    for r in rows:
        if r.get("event") == "goal":
            goal_no[r["t"]] = r["no"]
    goal_times = sorted(goal_no)
    intervals = collections.Counter()
    others_seen = []
    for t, k, r in events:
        if k == 0:
            pts = [lift((q[0], q[1] + dy), n, hc, HU) for q in r["floor"] if q[2] >= 0.5]
            recent = [pose_at(states, t - d) for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)]
            recent = [(q[0], q[1]) for q in recent if q is not None]
            now = pose_at(states, t)
            if tracks.on_frame(t, pts, recent, local=None, pose=(now[0], now[1]) if now else None):
                used += 1
                intervals[round(tracks.interval)] += 1
                if now is not None and seen[0] + 0.05 <= now[1] <= seen[1] - 0.05:
                    total["self_possible"] += 1
                    if tracks.residual is not None and any(math.hypot(q[0] - now[0] - tracks.residual[0], q[1] - now[1] - tracks.residual[1]) < 1e-6 for q in pts):
                        pass
                # 우리 차로 고른 것이 있었나: split_self 를 다시 불러 본다
                from navigation.drive.ugvs import split_self
                me, others = split_self(pts, recent, p.ugv_self_match_m)
                if me is not None:
                    matched += 1
                    if now is not None:
                        res.append((me[0] - now[0], me[1] - now[1], now[0] - n[0], now[1] - n[1]))
                others_seen.append(len(others))
        elif k == 1:
            job = r.get("job")
            if not job or not (job.get("phase", "").startswith("waiting_obstacle") or job.get("through")):
                continue
            block = blockers(r["pose"], r.get("unexplained") or [], p)
            if not block:
                continue
            cx, cy = st.mean(x for x, _y in block), st.mean(y for _x, y in block)
            kind, still, where = kind_of(tracks, t, block, p)
            no = goal_no[max(g for g in goal_times if g <= t)] if goal_times and goal_times[0] <= t else None
            apart = None if where is None else round(math.hypot(where[0] - r["pose"][0], where[1] - r["pose"][1]), 2)
            per_goal[no].append((t, kind, still, (round(cx, 2), round(cy, 2)), seen[0] <= cy <= seen[1], job.get("phase"), apart))
        else:
            if r.get("outcome") != "BLOCKED":
                continue
            obs = per_goal.get(r["no"], [])
            waits = [o for o in obs if o[5].startswith("waiting_obstacle")]
            kinds = collections.Counter(o[1] for o in obs)
            top = kinds.most_common(1)[0][0] if kinds else "점 없음"
            inview = sum(1 for o in obs if o[4]) / len(obs) if obs else 0.0
            code = (r.get("detail") or "").split("[")[-1].rstrip("]") if "[" in (r.get("detail") or "") else r.get("code", "")
            total["blocked"] += 1
            total["blocked_" + top] += 1
            if inview >= 0.5:
                total["blocked_inview"] += 1
            saved = 0.0
            if top == "parked" and "OBSTACLE" in code and waits:
                saved = max(0.0, (waits[-1][0] - waits[0][0]) - p.obstacle_give_up_parked_s)
                saved_total += saved
            if VERBOSE or top in ("parked", "moving"):
                where = obs[len(obs) // 2][3] if obs else None
                aparts = [o[6] for o in obs if o[6] is not None]
                print("   %s no%-3s %-18s 막는 점 %s 화면 안 %.0f%% | 드론: %s (서 있던 %.0f s) 그 UGV ↔ 우리 %.2f m · 기다린 줄 %d" % (
                    mid[-6:], r["no"], code, where, 100 * inview, dict(kinds), max((o[2] for o in obs), default=0.0), min(aparts) if aparts else -1, len(waits)))
    rx = [a for a, _b, _c, _d in res]
    ry = [b for _a, b, _c, _d in res]
    # 가운데(N)에서 바깥쪽으로의 어긋남: + 면 드론 자리가 더 바깥
    radial = [(a * c + b * d) / max(0.3, math.hypot(c, d)) for a, b, c, d in res if math.hypot(c, d) > 0.8]
    all_res += res
    print("%s 사진 %d 중 쓴 것 %d · 우리 차 고름 %d (%.0f%%) · 다른 UGV/장 가운데 %s · 간격별 %s · 드론−자세 가운데 dx %+.3f dy %+.3f |d| %.3f · 바깥쪽 %+.3f (n %d)" % (
        mid, len(frames), used, matched, 100.0 * matched / max(1, used), st.median(others_seen) if others_seen else None, dict(intervals),
        st.median(rx) if rx else float("nan"), st.median(ry) if ry else float("nan"),
        st.median(math.hypot(a, b) for a, b in zip(rx, ry)) if rx else float("nan"), st.median(radial) if radial else float("nan"), len(radial)))
print("합계:", dict(total), "· 서 있는 UGV 로 일찍 끝냈을 때 아낀 시간 %.1f s" % saved_total)
radial = [(a * c + b * d) / max(0.3, math.hypot(c, d)) for a, b, c, d in all_res if math.hypot(c, d) > 0.8]
print("UGV 높이 %.2f: 드론−자세 |d| 가운데 %.3f m · 바깥쪽 치우침 가운데 %+.3f m (n %d)" % (
    HU, st.median(math.hypot(a, b) for a, b, _c, _d in all_res), st.median(radial), len(radial)))
