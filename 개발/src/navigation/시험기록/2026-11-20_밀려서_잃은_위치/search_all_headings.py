"""라이다만으로: 마지막으로 믿던 자리 둘레(±half m)를 모든 방향으로 훑으면(slot_search) 어디가 나오나."""
import sys, json, math, os, time
from types import SimpleNamespace
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
nav=os.environ["NAV"]; sys.path.insert(0,nav); sys.path.insert(0,nav+"/test")
from test_node_split import load_slam
sn=SimpleNamespace(**load_slam()[0])
D=sys.argv[1]; T0=float(sys.argv[2]); A,B=float(sys.argv[3]),float(sys.argv[4]); HALF=float(sys.argv[5]); EVERY=int(sys.argv[6]); MODE=sys.argv[7]  # MODE: anchor x,y  또는 logged
maps=[]
for l in open(D+"/navigation.jsonl",encoding="utf-8",errors="replace"):
    try: r=json.loads(l)
    except Exception: continue
    if r.get("event")=="map": maps.append(r)
m=maps[0]
field=SimpleNamespace(width=m["size"][0],height=m["size"][1],box_size=m["box_size"],boxes=[SimpleNamespace(x=b[0],y=b[1]) for b in m["boxes"]],bases=[])
def as_scan(r):
    return SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link'),ranges=[x if x else float('nan') for x in r['ranges']],range_min=0.02,range_max=12.0,angle_min=r['angle_min'],angle_increment=r['angle_increment'])
n=ok=none=wrong=0; k=0; ms=[]
for l in open(D+"/navigation_slam.jsonl",encoding="utf-8",errors="replace"):
    if '"scan_raw"' not in l: continue
    r=json.loads(l); t=r["t"]-T0
    if not (A<=t<=B): continue
    k+=1
    if k%EVERY: continue
    if MODE=="logged":
        if not r.get("pose"): continue
        c=r["pose"]
    else:
        c=[float(v) for v in MODE.split(",")]
    scan=as_scan(r)
    region=(max(0.1,c[0]-HALF),max(0.1,c[1]-HALF),min(field.width-0.1,c[0]+HALF),min(field.height-0.1,c[1]+HALF))
    s=time.perf_counter(); pose=sn.slot_search(field,scan,region); ms.append((time.perf_counter()-s)*1000)
    n+=1
    if MODE=="logged":
        if pose is None: none+=1
        elif math.hypot(pose[0]-c[0],pose[1]-c[1])<=0.2 and abs(math.atan2(math.sin(pose[2]-c[2]),math.cos(pose[2]-c[2])))<=0.3: ok+=1
        else:
            wrong+=1; print("   %6.1f 기록 (%.2f,%.2f,%.2f) → 훑은 답 (%.2f,%.2f,%.2f)"%(t,c[0],c[1],c[2],pose[0],pose[1],pose[2]))
    else:
        print("%6.1f 훑은 답 %s (기록된 자리 %s)"%(t,None if pose is None else "(%.2f,%.2f,%.2f)"%pose,None if not r.get("pose") else "(%.2f,%.2f,%.2f)"%tuple(r["pose"])))
if MODE=="logged": print("스캔 %d장: 기록된 자리와 같은 답 %d · 답 없음 %d · 다른 답 %d | 한 번에 %.0f ms (PC)"%(n,ok,none,wrong,sum(ms)/max(1,len(ms))))
