import sys, json, math, os
from types import SimpleNamespace
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
nav=os.environ["NAV"]; sys.path.insert(0,nav); sys.path.insert(0,nav+"/test")
from test_node_split import load_slam
sn=SimpleNamespace(**load_slam()[0])
D=sys.argv[1]; T0=float(sys.argv[2]); A,B=float(sys.argv[3]),float(sys.argv[4]); R=float(sys.argv[5]) if len(sys.argv)>5 else 1.0
maps=[]
for l in open(D+"/navigation.jsonl",encoding="utf-8",errors="replace"):
    try: r=json.loads(l)
    except Exception: continue
    if r.get("event")=="map": maps.append(r)
m=maps[0]
field=SimpleNamespace(width=m["size"][0],height=m["size"][1],box_size=m["box_size"],boxes=[SimpleNamespace(x=b[0],y=b[1]) for b in m["boxes"]])
boxes,half,w,h=sn.structure(field)
def as_scan(r):
    return SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link'),ranges=[x if x else float('nan') for x in r['ranges']],range_min=0.02,range_max=12.0,angle_min=r['angle_min'],angle_increment=r['angle_increment'])
last=None
for l in open(D+"/navigation_slam.jsonl",encoding="utf-8",errors="replace"):
    if '"scan_raw"' not in l: continue
    r=json.loads(l); t=r["t"]-T0
    if r.get("pose") and t<A: last=r["pose"]
    if not (A<=t<=B): continue
    scan=as_scan(r); pts=np.array(sn.scan_points(scan),dtype=float)
    c=last if "--global" not in sys.argv else (w/2,h/2,0)
    rr=R if "--global" not in sys.argv else 3.2
    best=[]
    n=int(rr/0.1)
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            x,y=c[0]+0.1*i,c[1]+0.1*j
            if not (0.15<x<w-0.15 and 0.15<y<h-0.15): continue
            for k in range(36):
                yaw=k*math.pi/18
                hits,err=sn.score_pose(pts,(x,y,yaw),boxes,half,w,h)
                best.append((hits,-err,x,y,yaw))
    best.sort(reverse=True)
    top=[]
    for b in best:
        if all(math.hypot(b[2]-q[2],b[3]-q[3])>0.3 or abs(math.atan2(math.sin(b[4]-q[4]),math.cos(b[4]-q[4])))>0.5 for q in top): top.append(b)
        if len(top)==3: break
    lp=r.get("pose")
    print("%6.1f 기록된 자리 %s 점 %d | 가장 잘 맞는 자리: %s"%(t,None if not lp else "(%.2f,%.2f,%.2f)"%tuple(lp),len(pts)," · ".join("(%.1f,%.1f,%.2f) %d점"%(b[2],b[3],math.atan2(math.sin(b[4]),math.cos(b[4])),b[0]) for b in top)))
