"""목표 칸이 「서 있는 물체」 에 덮였다고 보는 순간: 지금 기준(R0) 과 R1(처음~마지막 찍힘 2 s) 견주기. 일곱 판 재생, 주행 중 매 스캔."""
import sys, json, math, os, glob
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.environ["NAV"])
from navigation.drive.fieldmap import MapSpec, fixed_pillars
from navigation.drive.geometry import Pose2D
from navigation.drive.grid import Grid
from navigation.drive.obstacles import ObstacleTracker, scan_to_body
from navigation.drive.params import DriveParams
from navigation.drive.ugvs import UgvTracks
p=DriveParams(); TS=p.tight_still_s
for run in sorted(glob.glob("bt/2026*")):
    nav=[json.loads(l) for l in open(run+"/navigation.jsonl",encoding="utf-8") if l.strip()]
    slam=[]
    for l in open(run+"/navigation_slam.jsonl",encoding="utf-8"):
        if '"scan_raw"' in l:
            r=json.loads(l)
            if r.get("pose"): r["event"]="scan"; slam.append(r)
    RATE={}
    for i,r in enumerate(slam):
        w=[]
        for j in (i-1,i+1):
            if 0<=j<len(slam): w.append(abs(math.atan2(math.sin(slam[j]["pose"][2]-r["pose"][2]),math.cos(slam[j]["pose"][2]-r["pose"][2])))/abs(slam[j]["t"]-r["t"]))
        RATE[r["t"]]=max(w)
    poses=sorted([(r["t"],r["pose"]) for r in nav if r.get("pose")]+[(r["t"],r["pose"]) for r in slam])
    tr=ObstacleTracker(p); ug=UgvTracks(p); tr.ugvs=ug; grid=spec=None; job=None; jt=0; t0=None
    n=cov=s0=s1=0; tight0=tight1=0; ex=[]
    for r in sorted(nav+slam,key=lambda r:r["t"]):
        t=r["t"]; e=r.get("event")
        if e=="map":
            w,h=r["size"]; spec=MapSpec(w,h,r["box_size"],tuple(tuple(b) for b in r["boxes"]),"arena",tuple(fixed_pillars(w,h,r["box_size"])))
            grid=Grid(spec,radius=p.robot_radius,margin=p.inflate_margin,cell=p.cell,soft_m=p.soft_m,soft_cost=p.soft_cost)
        elif e=="goal" and t0 is None: t0=t
        elif e=="ugvs":
            st=r["stamp"]; win=[q[1] for q in poses if st-p.ugv_self_window_s<=q[0]<=st+0.1]
            ug.on_frame(st,r["seen"],[(q[0],q[1]) for q in win])
        elif e=="state" and r.get("job") and r["job"].get("mode")!="straight": job=(tuple(r["job"]["goal"][:2]),r["job"]["mode"],r["job"].get("shifted")); jt=t
        elif e=="scan" and grid is not None:
            ranges=np.array([x if x else np.nan for x in r["ranges"]],dtype=float)
            body=scan_to_body(ranges,r["angle_min"],r["angle_increment"],p)
            tr.on_scan(t,Pose2D(*r["pose"]),body,spec,grid,(r["angle_min"],r["angle_increment"],ranges),turn_rate=RATE[t])
            extra=tr.extra_cells(t,grid)
            small=p.tight_radius+grid.grow; parked=tr._parked_cells(t,grid)
            tight0+=sum(1 for c,s in tr._seeds.items() if s[1]>t and (t-s[3]>=TS or c in parked) and s[0]>small)
            tight1+=sum(1 for c,s in tr._seeds.items() if s[1]>t and (s[2]-s[3]>=TS or c in parked) and s[0]>small)
            if not job or t-jt>1.2: continue
            n+=1
            cell=grid.cell_of(*job[0])
            if not (grid.blocked(cell,extra) and not grid.blocked(cell)): continue
            cov+=1
            plan=cell in tr._ugv_plan_cells(grid)
            def st(rule):
                if plan: return True
                for c,s in tr._seeds.items():
                    if s[1]<=t or cell not in tr._disc(grid,c,s[0]): continue
                    if c in parked or (t-s[3]>=TS if rule==0 else s[2]-s[3]>=TS): return True
                return False
            a,b=st(0),st(1); s0+=a; s1+=b
            if a and not b:
                near=min((math.hypot(u.x-job[0][0],u.y-job[0][1]) for u in ug.fresh(t)),default=None)
                ex.append((round(t-t0,1),job[0],job[1],"실제 기록의 옮김=%s"%job[2],"가장 가까운 드론 UGV %s m"%(None if near is None else round(near,2))))
    print("%s 순간 %d: 목표 칸이 임시 장애물에 덮임 %d, 그중 「서 있는 물체」 — 지금 기준 %d, R1 %d | 좁은 틈 길에서 작게 보는 칸(합) 지금 %d → R1 %d"%(run[-11:],n,cov,s0,s1,tight0,tight1))
    for x in ex[:6]: print("     달라지는 순간:",x)
