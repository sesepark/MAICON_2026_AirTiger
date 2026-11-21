"""경기 기록의 스캔으로 slam_node 의 진짜 코드(_relocate · _fits_drone_map)를 돌린다 (가짜 ROS — test_node_split.load_slam).

  lost   <폴더> <경기 시작 t> [--no-drone]   기록의 relocate_lost 마다: 그때의 마지막 정합 자리 · /odom 으로 옮긴 자리 · 드론 UGV 자리를 넣고 _relocate 를 부른다
  normal <폴더> <경기 시작 t> [몇 장마다]     정상 판: 스캔마다 「지금 잃었다」 치고 (마지막 정합 자리 = 기록된 자리, 드론 자리 없음) _relocate 를 부른다 —
                                          기록된 자리와 같은 답인가. 그리고 그 스캔이 _fits_drone_map 을 통과하는가
환경 변수 NAV = navigation 패키지 폴더. 폴더에는 navigation.jsonl · navigation_slam.jsonl.
"""
import json
import math
import os
import sys
from types import SimpleNamespace as NS

sys.stdout.reconfigure(encoding="utf-8")
nav = os.environ["NAV"]
sys.path.insert(0, nav)
sys.path.insert(0, nav + "/test")
from test_node_split import FakeNode, load_slam  # noqa: E402

FakeNode.distinct_publishers = True
module, _ = load_slam()
mode, folder, T0 = sys.argv[1], sys.argv[2], float(sys.argv[3])


def rows(name):
    for line in open(folder + "/" + name, encoding="utf-8", errors="replace"):
        try:
            yield json.loads(line)
        except Exception:
            continue


first_map = next(r for r in rows("navigation.jsonl") if r.get("event") == "map")
field = NS(width=first_map["size"][0], height=first_map["size"][1], box_size=first_map["box_size"],
           boxes=[NS(x=b[0], y=b[1]) for b in first_map["boxes"]], bases=[], vehicles=[])
ugvs = [r for r in rows("navigation.jsonl") if r.get("event") == "ugvs"]
scans, relocs, states = [], [], []
for r in rows("navigation_slam.jsonl"):
    e = r.get("event")
    if e == "scan_raw":
        scans.append(r)
    elif e == "relocate_lost":
        relocs.append(r)
    elif e == "state" and r.get("anchor"):
        states.append(r)


def as_scan(r):
    return NS(header=NS(frame_id="base_lidar_link"), ranges=[x if x else float("nan") for x in r["ranges"]], range_min=0.02, range_max=12.0,
              angle_min=r["angle_min"], angle_increment=r["angle_increment"])


def call(slam, scan_row, anchor, last, seen):
    """마지막 정합 자리 anchor, /odom 으로 옮긴 자리 last (없으면 anchor), 드론 자리 seen 으로 _relocate 한 번."""
    ts = int(round(scan_row["t"] * 1e9))
    last = last or anchor
    slam._delta = 0.0
    slam._scan_anchor = (ts - 3_000_000_000, tuple(anchor), (0.0, 0.0, 0.0))
    slam._ugv_seen = (ts - 500_000_000, [tuple(q) for q in seen]) if seen else None
    slam._relocate_next = None
    pose = slam._relocate(field, as_scan(scan_row), ts, (last[0] - anchor[0], last[1] - anchor[1], 0.0))
    return pose, slam._relocate_note


if mode == "lost":
    slam = module["SlamNode"]()
    drone = "--no-drone" not in sys.argv
    for r in relocs:
        t = r["t"]
        scan_row = min(scans, key=lambda s: abs(s["t"] - (t - 0.9)))
        anchor = [s for s in states if s["t"] <= t][-1]["anchor"]
        seen = [u for u in ugvs if u["t"] <= t and t - u["t"] <= 6]
        seen = seen[-1]["seen"] if (seen and drone) else []
        pose, note = call(slam, scan_row, anchor, r["last"], seen)
        print("%6.1f 그때: %-40s | 지금 코드: %s" % (t - T0, r["note"][:40], note[:70]))
else:
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    same = none = other = fits = n = 0
    for k, r in enumerate(scans):
        t = r["t"] - T0
        if k % every or not r.get("pose") or not 12 <= t <= 300:
            continue
        slam = module["SlamNode"]()   # 시도마다 새로 — 「이어진 두 번」 없이 한 번에 받는 것만 센다
        p = r["pose"]
        pose, note = call(slam, r, p, None, [])
        if pose is None and "후보" in note:   # 75 % 에 못 미쳐 한 번 더 보자는 답 — 같은 스캔으로 한 번 더
            pose, note = call(slam, r, p, None, [])
        n += 1
        fits += bool(slam._fits_drone_map(field, as_scan(r), tuple(p)))
        if pose is None:
            none += 1
        elif math.hypot(pose[0] - p[0], pose[1] - p[1]) <= 0.2 and abs(math.atan2(math.sin(pose[2] - p[2]), math.cos(pose[2] - p[2]))) <= 0.3:
            same += 1
        else:
            other += 1
            print("   %6.1f 기록 (%.2f, %.2f, %.2f) → (%.2f, %.2f, %.2f) %s" % (t, p[0], p[1], p[2], pose[0], pose[1], pose[2], note[:50]))
    print("스캔 %d장: 기록된 자리와 같은 답 %d · 답 없음 %d · 다른 답 %d | 드론 지도에 맞음(지도 고치기 통과) %d장 (%.0f %%)" % (n, same, none, other, fits, 100.0 * fits / max(1, n)))
