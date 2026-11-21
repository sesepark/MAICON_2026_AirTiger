"""06:17 판의 relocate_lost 17번을 새 규칙으로: 마지막 정합 자리(anchor) · /odom 으로 옮긴 자리(last) · 드론 UGV 자리 둘레를 모든 방향으로."""
import sys, json, math, os
from types import SimpleNamespace
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
nav=os.environ["NAV"]; sys.path.insert(0,nav); sys.path.insert(0,nav+"/test")
from test_node_split import load_slam
sn=SimpleNamespace(**load_slam()[0])
D=sys.argv[1]; T0=float(sys.argv[2]); UGV="--no-drone" not in sys.argv
maps=[]; ug=[]
for l in open(D+"/navigation.jsonl",encoding="utf-8",errors="replace"):
    try: r=json.loads(l)
    except Exception: continue
    if r.get("event")=="map": maps.append(r)
    if r.get("event")=="ugvs": ug.append(r)
m=maps[0]
field=SimpleNamespace(width=m["size"][0],height=m["size"][1],box_size=m["box_size"],boxes=[SimpleNamespace(x=b[0],y=b[1]) for b in m["boxes"]],bases=[])
solid,bh,ww,wh=sn.structure(field)
def as_scan(r):
    return SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link'),ranges=[x if x else float('nan') for x in r['ranges']],range_min=0.02,range_max=12.0,angle_min=r['angle_min'],angle_increment=r['angle_increment'])
scans=[]; rel=[]; states=[]
for l in open(D+"/navigation_slam.jsonl",encoding="utf-8",errors="replace"):
    try: r=json.loads(l)
    except Exception: continue
    e=r.get("event")
    if e=="scan_raw": scans.append(r)
    elif e=="relocate_lost": rel.append(r)
    elif e=="state" and r.get("anchor"): states.append(r)
for r in rel:
    t=r["t"]; sc=min(scans,key=lambda s:abs(s["t"]-(t-0.9)))
    anchor=[s for s in states if s["t"]<=t][-1]["anchor"]; last=r["last"]
    seen=[u for u in ug if u["t"]<=t and t-u["t"]<=6]
    seen=seen[-1]["seen"] if (seen and UGV) else []
    regions=[("anchor",anchor[0],anchor[1],0.4)]
    if math.hypot(last[0]-anchor[0],last[1]-anchor[1])>0.2: regions.append(("last",last[0],last[1],0.4))
    near=sorted(seen,key=lambda q:math.hypot(q[0]-last[0],q[1]-last[1]))
    near=[q for q in near if all(math.hypot(q[0]-c[1],q[1]-c[2])>0.4 for c in regions)][:3]
    regions+=[("ugv",x,y,0.25) for x,y in near]
    scan=as_scan(sc); pts=sn.scan_points(scan); found=[]
    for src,cx,cy,half in regions:
        reg=(max(0.1,cx-half),max(0.1,cy-half),min(ww-0.1,cx+half),min(wh-0.1,cy+half))
        pose=sn.slot_search(field,scan,reg)
        if pose is not None and all(math.hypot(pose[0]-o[2][0],pose[1]-o[2][1])>0.2 for o in found):
            found.append((sn.score_pose(pts,pose,solid,bh,ww,wh)[0],src,pose))
    found.sort(key=lambda f:-f[0])
    if not found: dec="맞는 자리 없음"
    elif len(found)>1 and found[1][0]>0.7*found[0][0]: dec="둘 이상 → 버림"
    else: dec="받음 (%.2f, %.2f, %.2f) [%s]"%(*found[0][2],found[0][1])
    print("%6.1f 전: %s | 새: %s | 후보 %s"%(t-T0,r["note"][:34],dec,[(f[1],int(f[0]),round(f[2][0],2),round(f[2][1],2),round(f[2][2],2)) for f in found]))
