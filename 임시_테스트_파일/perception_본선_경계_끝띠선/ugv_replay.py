"""본선 6판: v1 검출 → 지금의 UgvTracks · slam _drone_self 에 넣고, 라이다 원본(scan_raw)으로 드론 UGV 가 진짜인지 대조."""
import bisect, collections, datetime, glob, json, math, os, statistics as st, sys
sys.stdout.reconfigure(encoding="utf-8")
NAV = "C:/Users/Admin/MAICON_ugv/개발/src/navigation"
sys.path.insert(0, NAV); sys.path.insert(0, NAV + "/test")
from navigation.drive.params import DriveParams
from navigation.drive.ugvs import UgvTracks
from test_node_split import FakeNode, load_slam

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[3] if len(sys.argv) > 3 else HERE
LOGS = os.path.dirname(HERE) + "/logs"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "start0"
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
LIDAR_X = 0.0398


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * f))] if v else float("nan")


class Poses:
    def __init__(self, states):
        self.t = [s[0] for s in states]
        self.p = [s[1] for s in states]

    def at(self, t, gap=0.9):
        i = bisect.bisect_right(self.t, t)
        if i == 0:
            return self.p[0] if self.t and self.t[0] - t <= gap else None
        if i == len(self.t):
            return self.p[-1] if t - self.t[-1] <= gap else None
        a, b = self.t[i - 1], self.t[i]
        if b - a > 2 * gap:
            return self.p[i - 1] if t - a <= gap else (self.p[i] if b - t <= gap else None)
        k = (t - a) / (b - a)
        pa, pb = self.p[i - 1], self.p[i]
        return (pa[0] + (pb[0] - pa[0]) * k, pa[1] + (pb[1] - pa[1]) * k, pa[2])


def lidar_check(scan, pt):
    """scan(원본) 으로 pt 자리를 본다 → hit · through · occluded · unknown · far."""
    px, py, yaw = scan["pose"][:3]
    c, s = math.cos(yaw), math.sin(yaw)
    sx, sy = px + c * LIDAR_X, py + s * LIDAR_X
    dx, dy = pt[0] - sx, pt[1] - sy
    d = math.hypot(dx, dy)
    if d < 0.40 or d > 3.0:
        return "far"
    bx, by = c * dx + s * dy, -s * dx + c * dy
    a = math.atan2(-bx, by)
    w = math.atan(0.09 / d)
    inc, a0, n = scan["angle_increment"], scan["angle_min"], len(scan["ranges"])
    full = abs(inc * n - 2 * math.pi) < 0.1
    rays = []
    k0, k1 = int(math.floor((a - w - a0) / inc)), int(math.ceil((a + w - a0) / inc))
    for k in range(k0, k1 + 1):
        r = scan["ranges"][k % n] if full else (scan["ranges"][k] if 0 <= k < n else 0)
        if r and r > 0.05:
            rays.append(r)
    if len(rays) < 2:
        return "unknown"
    if sum(1 for r in rays if d - 0.35 <= r <= d + 0.20) >= max(1, len(rays) // 4):
        return "hit"
    if sum(1 for r in rays if r > d + 0.30) >= 0.6 * len(rays):
        return "through"
    if sum(1 for r in rays if r < d - 0.35) >= 0.6 * len(rays):
        return "occluded"
    return "unknown"


def epoch(s):
    return datetime.datetime.fromisoformat(s).timestamp() if s else None


module, _ = load_slam()
T = collections.Counter()
OFF, LAT, PARKED = [], [], []
ZONE, ZONE_N = collections.Counter(), collections.Counter()
BYSCORE = collections.defaultdict(collections.Counter)
INT = collections.Counter()
BRIEF = '-b' in sys.argv
for path in sorted(glob.glob(SRC + "/v1_*.json")):
    det = json.load(open(path, encoding="utf-8"))
    mid = det["match"]
    if VARIANT not in det["views"]:
        print(mid, "경계 없음:", VARIANT)
        continue
    rows = [json.loads(l) for l in open("%s/%s/navigation_debug.jsonl" % (LOGS, mid), encoding="utf-8")]
    states = [(r["t"], r["pose"]) for r in rows if r.get("event") == "state" and r.get("pose")]
    poses = Poses(states)
    scans = [json.loads(l) for l in open("%s/%s/navigation_slam_debug.jsonl" % (LOGS, mid), encoding="utf-8") if '"scan_raw"' in l]
    scans = [s for s in scans if s.get("pose")]
    scan_t = [s["t"] for s in scans]
    map_h = next((r["size"][1] for r in rows if r.get("event") == "map"), 4.8)
    dy = (map_h - 4.8) / 2.0
    seen = [det["views"][VARIANT]["seen_y"][0] + dy, det["views"][VARIANT]["seen_y"][1] + dy]
    p = DriveParams()
    tracks = UgvTracks(p)
    FakeNode.distinct_publishers = True
    slam = module["SlamNode"]()
    frames = sorted((f for f in det["frames"] if f.get("wall")), key=lambda f: f["wall"])
    c = collections.Counter()
    offs, lat, nper = [], [], []
    life = {}
    self_as_other = 0
    t0 = frames[0]["wall"] - (frames[0].get("match_t") or 0.0)
    for f in frames:
        t = f["wall"]
        cap = epoch(f.get("captured"))
        if cap is not None:
            lat.append(t - cap)
        pts_all = [(x, y + dy, s) for x, y, s in f["det"][VARIANT]]
        pts = [(x, y) for x, y, s in pts_all if s >= CONF]
        nper.append(len(pts))
        recent = [poses.at(t - 0.1 * k) for k in range(-1, 16)]
        recent = [(r[0], r[1]) for r in recent if r is not None]
        now = poses.at(t)
        c["frames"] += 1
        c["int_%s" % f.get("interval")] += 1
        if not recent or now is None:
            c["no_pose"] += 1
        inview = now is not None and seen[0] + 0.05 <= now[1] <= seen[1] - 0.05
        tracks.residual = None
        used = tracks.on_frame(t, pts, recent, local=None, pose=(now[0], now[1]) if now else None)
        slam._ugv_tracks.on_frame(t, pts, recent)
        slam._drone_self(int(t * 1e9), pts, recent)
        if not used:
            continue
        c["used"] += 1
        picked = tracks.residual is not None
        if inview:
            c["inview"] += 1
            zone = (min(5, int(now[0] / 1.02)), min(2, int((now[1] - seen[0]) / ((seen[1] - seen[0]) / 3))))
            ZONE_N[zone] += 1
            INT[(f.get("interval"), picked)] += 1
            if picked:
                c["picked_inview"] += 1
                offs.append((tracks.residual[0], tracks.residual[1], now[0], now[1]))
            else:
                ZONE[zone] += 1
                near = sorted(math.hypot(x - now[0], y - now[1]) for x, y in pts)
                c["miss_none_1m" if not near or near[0] > 1.0 else "miss_near_0.4_1.0"] += 1
        elif picked:
            c["picked_outview"] += 1
        if now is not None and any(math.hypot(u.x - now[0], u.y - now[1]) <= 0.25 for u in tracks.fresh(t)):
            self_as_other += 1
        scan = None
        if scan_t:
            i = bisect.bisect_left(scan_t, t - 0.35)
            j = min(range(max(0, i - 1), min(len(scans), i + 2)), key=lambda k: abs(scan_t[k] - (t - 0.35)))
            if abs(scan_t[j] - (t - 0.35)) <= 0.6:
                scan = scans[j]
        for x, y, s in pts_all:
            if now is None or math.hypot(x - now[0], y - now[1]) <= 0.45 or scan is None:
                continue   # 우리 차일 수 있는 것은 뺀다
            band = "0.25-0.5" if s < 0.5 else ("0.5-0.7" if s < 0.7 else "0.7+")
            BYSCORE[band][lidar_check(scan, (x, y))] += 1
        for u in tracks.fresh(t):
            rec = life.setdefault(id(u), {"first": t - t0, "max_still": 0.0, "lid": collections.Counter(), "frames": 0, "seen_frames": 0, "near_us": 9.0})
            rec["frames"] += 1
            rec["seen_frames"] += 1 if u.seen == t else 0
            rec["max_still"] = max(rec["max_still"], u.still_s())
            rec["pos"] = (round(u.x, 2), round(u.y, 2))
            if now is not None:
                rec["near_us"] = min(rec["near_us"], math.hypot(u.x - now[0], u.y - now[1]))
            if scan is not None and u.still_s() >= p.ugv_parked_s:
                rec["lid"][lidar_check(scan, (u.x, u.y))] += 1
        c["lost_max"] = max(c["lost_max"], len(tracks.lost))
    planned = [r for r in life.values() if r["max_still"] >= p.ugv_plan_parked_s]
    PARKED += [(mid, r) for r in planned]
    ch = slam._pose_checks
    lat0 = min(lat) if lat else 0.0
    dd = [math.hypot(a, b) for a, b, _x, _y in offs]
    print("%s 지도세로 %.2f 보이는y %.2f~%.2f | 사진 %d 간격 %s 장당UGV 가운데 %s 최대 %d | 쓴 %d 자세없음 %d | 화면 안 %d 중 우리 고름 %d (%.0f%%) [못 고름: 1m 안에 없음 %d · 0.4~1.0 에 있음 %d] 화면밖인데 고름 %d | 드론−자세 50%% %.3f 90%% %.3f | 우리 자리에 다른UGV궤적 %d장 | 10s 넘게 선 궤적 %d · lost 최대 %d | slam seen %d missed %d doubt %d | 받음−촬영 흔들림 50%% %.2f 90%% %.2f 최대 %.2f s" % (
        mid, map_h, seen[0], seen[1], c["frames"], {k[4:]: v for k, v in c.items() if k.startswith("int_")}, st.median(nper) if nper else None, max(nper) if nper else 0,
        c["used"], c["no_pose"], c["inview"], c["picked_inview"], 100.0 * c["picked_inview"] / max(1, c["inview"]), c["miss_none_1m"], c["miss_near_0.4_1.0"], c["picked_outview"],
        q(dd, 0.5), q(dd, 0.9), self_as_other, len(planned), c["lost_max"], ch["drone_seen"], ch["drone_missed"], ch["drone_doubt"],
        q([l - lat0 for l in lat], 0.5), q([l - lat0 for l in lat], 0.9), (max(lat) - lat0) if lat else 0.0))
    if offs:
        print("      드론−자세 가운데 dx %+.3f dy %+.3f (n %d)" % (st.median(o[0] for o in offs), st.median(o[1] for o in offs), len(offs)))
        right = [o for o in offs if o[2] > 4.08]
        left = [o for o in offs if o[2] < 2.04]
        meta = (det.get("meta") or {}).get(VARIANT) or {}
        print("      RIGHT %s %s strip %s | x>4.08: n %d dx %+.3f dy %+.3f |d| 50%% %.3f 90%% %.3f | x<2.04: n %d dx %+.3f |d| 50%% %.3f" % (
            mid[-6:], VARIANT, None if meta.get("strip") is None else round(meta["strip"], 3), len(right),
            st.median(o[0] for o in right) if right else float("nan"), st.median(o[1] for o in right) if right else float("nan"),
            q([math.hypot(o[0], o[1]) for o in right], .5), q([math.hypot(o[0], o[1]) for o in right], .9),
            len(left), st.median(o[0] for o in left) if left else float("nan"), q([math.hypot(o[0], o[1]) for o in left], .5)))
    T.update({k: v for k, v in c.items() if k != "lost_max"})
    T["self_as_other"] += self_as_other
    T["doubt"] += ch["drone_doubt"]
    OFF += offs
    LAT += [l - lat0 for l in lat]
print()
print("모두:", dict(T))
d = [math.hypot(a, b) for a, b, _x, _y in OFF]
if d:
    print("드론−자세 |d| (n %d): 50%% %.3f · 80%% %.3f · 90%% %.3f · 95%% %.3f · dx 가운데 %+.3f dy 가운데 %+.3f" % (
        len(d), q(d, .5), q(d, .8), q(d, .9), q(d, .95), st.median(a for a, *_ in OFF), st.median(b for _a, b, *_ in OFF)))
    print("x 띠별 dx 가운데:", {k: round(st.median(a for a, _b, x, _y in OFF if min(5, int(x / 1.02)) == k), 3) for k in range(6) if any(min(5, int(x / 1.02)) == k for _a, _b, x, _y in OFF)})
    print("y 띠별 dy 가운데:", {k: round(st.median(b for _a, b, _x, y in OFF if int(y) == k), 3) for k in range(5) if any(int(y) == k for _a, _b, _x, y in OFF)})
print("자리별 우리 차 못 고름/화면 안 (x 6칸 왼→오, y 3칸 위가 큰 y):")
for yy in (2, 1, 0):
    print("   ", "  ".join("%3d/%-3d" % (ZONE[(xx, yy)], ZONE_N[(xx, yy)]) for xx in range(6)))
print("라이다 대조 (우리 차 아닌 검출, 0.4~3 m):", {k: dict(v) for k, v in sorted(BYSCORE.items())})
print("사진 간격별 우리 차 고름 (화면 안):", {k: "%d/%d" % (INT[(k, True)], INT[(k, True)] + INT[(k, False)]) for k in sorted({i for i, _ in INT}, key=str)})
if BRIEF:
    sys.exit(0)
print("10 s 넘게 선 궤적 (길 고르기에 들어가는 것) %d개:" % len(PARKED))
for mid, r in sorted(PARKED, key=lambda e: (e[0], e[1]["first"])):
    print("   %s (%.2f, %.2f) 경기 %3.0f s 부터 · 선 시간 최대 %3.0f s · 사진 %3d장 중 %3d장 보임 · 우리와 가장 가까웠던 %.2f m · 라이다 %s" % (
        mid[-6:], r["pos"][0], r["pos"][1], r["first"], r["max_still"], r["frames"], r["seen_frames"], r["near_us"], dict(r["lid"])))
