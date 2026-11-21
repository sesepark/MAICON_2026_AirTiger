import sys, json, math, os
from types import SimpleNamespace
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
nav=os.environ["NAV"]; sys.path.insert(0,nav); sys.path.insert(0,nav+"/test")
from test_node_split import load_slam
sn=SimpleNamespace(**load_slam()[0])
def run(D,T0,A,B,label):
    m=None
    for l in open(D+"/navigation.jsonl",encoding="utf-8",errors="replace"):
        try: r=json.loads(l)
        except Exception: continue
        if r.get("event")=="map": m=r; break
    field=SimpleNamespace(width=m["size"][0],height=m["size"][1],box_size=m["box_size"],boxes=[SimpleNamespace(x=b[0],y=b[1]) for b in m["boxes"]])
    boxes,half,w,h=sn.structure(field); sh=[]; ts=[]
    for l in open(D+"/navigation_slam.jsonl",encoding="utf-8",errors="replace"):
        if '"scan_raw"' not in l: continue
        r=json.loads(l); t=r["t"]-T0
        if not (A<=t<=B) or not r.get("pose"): continue
        scan=SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link'),ranges=[x if x else float('nan') for x in r['ranges']],range_min=0.02,range_max=12.0,angle_min=r['angle_min'],angle_increment=r['angle_increment'])
        pts=np.array(sn.scan_points(scan),dtype=float)
        if len(pts)<12: continue
        hits,_=sn.score_pose(pts,tuple(r["pose"]),boxes,half,w,h); sh.append(hits/len(pts)); ts.append(t)
    a=np.array(sh)
    # 이어서 낮은 가장 긴 구간
    def longest(th):
        best=cur=0
        for v in a:
            cur=cur+1 if v<th else 0; best=max(best,cur)
        return best
    print("%-28s 스캔 %3d · 드론 지도에 맞은 비율: 가운데 %.0f%% · 아래 5%% %.0f%% · 가장 낮음 %.0f%% | 40%% 아래 %d장 (이어서 최대 %d) · 30%% 아래 %d장 (이어서 최대 %d)"%(label,len(a),100*np.median(a),100*np.percentile(a,5),100*a.min(),(a<0.4).sum(),longest(0.4),(a<0.3).sum(),longest(0.3)))
run("bt/20261120_023923",1795109990.262,12,300,"02:39 (정상)")
run("bt/20261120_032808",1795112915.472,12,300,"03:28 (정상)")
run("bt/20261120_044829",1795117737.583,12,300,"04:48 (정상)")
run("bt/20261120_063234",1795124057.838,12,300,"06:32 (정상)")
run("bt/n61",1795123077.025,12,50.5,"06:17 맞던 구간 (~50 s)")
run("bt/n61",1795123077.025,64,229,"06:17 틀린 구간 (64~229 s)")
