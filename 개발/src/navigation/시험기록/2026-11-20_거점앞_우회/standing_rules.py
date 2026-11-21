"""「서 있는 물체」 판정 기준 견주기 (라이다만 — 드론 판정은 정답으로만 쓴다). 일곱 판 재생, 1 Hz."""
import sys, json, math, os, glob
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.environ["NAV"])
from navigation.drive.fieldmap import MapSpec, fixed_pillars
from navigation.drive.geometry import Pose2D
from navigation.drive.grid import Grid
from navigation.drive.obstacles import ObstacleTracker, scan_to_body, structure_gap
from navigation.drive.params import DriveParams
from navigation.drive.ugvs import UgvTracks
p=DriveParams(); TS=p.tight_still_s
RULES=("R0 처음 찍힌 지 2 s (지금)","R1 처음~마지막 찍힘 2 s","R2 R1 + 곁 0.10 m 칸도 인정","R3 이어서 찍힌 시간 2 s","R4 R3 + 곁 0.10 m 칸도 인정")
tot={k:{r:[0,0] for r in RULES} for k in ("서 있는 UGV · 서서 봄","서 있는 UGV · 가면서 봄","움직이는 UGV","헛칸(기억만)","헛칸(지금 찍힘)","구조물 곁(어긋난 면)")}
for run in sorted(glob.glob("bt/2026*")):
    nav=[json.loads(l) for l in open(run+"/navigation.jsonl",encoding="utf-8") if l.strip()]
    slam=[]
    for l in open(run+"/navigation_slam.jsonl",encoding="utf-8"):
        if '"scan_raw"' in l:
            r=json.loads(l)
            if r.get("pose"): r["event"]="scan"; slam.append(r)
    RATE={}; MOVE={}
    for i,r in enumerate(slam):
        w=[];v=[]
        for j in (i-1,i+1):
            if 0<=j<len(slam):
                dt=abs(slam[j]["t"]-r["t"])
                w.append(abs(math.atan2(math.sin(slam[j]["pose"][2]-r["pose"][2]),math.cos(slam[j]["pose"][2]-r["pose"][2])))/dt)
                v.append(math.hypot(slam[j]["pose"][0]-r["pose"][0],slam[j]["pose"][1]-r["pose"][1])/dt)
        RATE[r["t"]]=max(w); MOVE[r["t"]]=max(v)>0.05 or max(w)>0.1
    poses=sorted([(r["t"],r["pose"]) for r in nav if r.get("pose")]+[(r["t"],r["pose"]) for r in slam])
    tr=ObstacleTracker(p); ug=UgvTracks(p); tr.ugvs=ug; grid=spec=None
    run_t={}   # 칸 → [이어서 찍힌 시간, 마지막으로 찍힌 스캔 시각]
    prev_scan=None
    for r in sorted(nav+slam,key=lambda r:r["t"]):
        t=r["t"]; e=r.get("event")
        if e=="map":
            w,h=r["size"]; spec=MapSpec(w,h,r["box_size"],tuple(tuple(b) for b in r["boxes"]),"arena",tuple(fixed_pillars(w,h,r["box_size"])))
            grid=Grid(spec,radius=p.robot_radius,margin=p.inflate_margin,cell=p.cell,soft_m=p.soft_m,soft_cost=p.soft_cost)
        elif e=="ugvs":
            st=r["stamp"]; win=[q[1] for q in poses if st-p.ugv_self_window_s<=q[0]<=st+0.1]
            ug.on_frame(st,r["seen"],[(q[0],q[1]) for q in win])
        elif e=="scan" and grid is not None:
            ranges=np.array([x if x else np.nan for x in r["ranges"]],dtype=float)
            body=scan_to_body(ranges,r["angle_min"],r["angle_increment"],p)
            fast=RATE[t]>=p.scan_turn_skip_rate
            tr.on_scan(t,Pose2D(*r["pose"]),body,spec,grid,(r["angle_min"],r["angle_increment"],ranges),turn_rate=RATE[t])
            tr.extra_cells(t,grid)
            hit={grid.cell_of(x,y) for x,y in tr.unexplained_world} & tr._seeds.keys()
            for c in list(run_t):
                if c not in tr._seeds: del run_t[c]
            for c in hit:
                a=run_t.get(c)
                if fast:   # 빨리 도는 스캔은 세지 않는다 (끊지도 않는다)
                    if a is None: run_t[c]=[0.0,None]
                    continue
                if a is None or a[1] is None or a[1]!=prev_scan: run_t[c]=[0.0 if a is None or a[1] is None else 0.0,t] if (a is None or a[1] is None or a[1]!=prev_scan) else a
                else: a[0]+=t-a[1]; a[1]=t
            if not fast: prev_scan=t
            if not ug.alive(t): continue
            tracks=ug.fresh(t)
            def R(c,s):
                return {RULES[0]:t-s[3]>=TS, RULES[1]:s[2]-s[3]>=TS, RULES[3]:run_t.get(c,[0])[0]>=TS}
            base={c:R(c,s) for c,s in tr._seeds.items() if s[1]>t}
            for c,s in tr._seeds.items():
                if s[1]<=t: continue
                x,y=grid.center(*c)
                res=dict(base[c])
                near=[b for d,b in base.items() if abs(d[0]-c[0])<=2 and abs(d[1]-c[1])<=2]
                res[RULES[2]]=any(b[RULES[1]] for b in near); res[RULES[4]]=any(b[RULES[3]] for b in near)
                du=min(((math.hypot(u.x-x,u.y-y),u) for u in tracks),key=lambda v:v[0],default=(9,None))
                g=float(structure_gap(spec,np.array([x]),np.array([y]))[0])
                seen_now=c in hit
                if du[0]<=0.30 and du[1].still_s()>=6.0:
                    if not seen_now or t-s[3]<TS+1.0: continue      # 지금 찍히고 있고, 본 지 3 s 는 된 칸만 (판정이 나와야 할 때)
                    k="서 있는 UGV · 가면서 봄" if MOVE[t] else "서 있는 UGV · 서서 봄"
                elif du[0]<=0.35 and du[1].still_s()<1.5: k="움직이는 UGV"
                elif du[0]>0.6 and g>0.30 and 0.6<y<4.0: k="헛칸(지금 찍힘)" if seen_now else "헛칸(기억만)"   # 드론 화면 안인데 곁에 UGV 도 구조물도 없다
                elif du[0]>0.6 and g<=0.30: k="구조물 곁(어긋난 면)"
                else: continue
                for rr in RULES:
                    tot[k][rr][1]+=1; tot[k][rr][0]+=bool(res[rr])
print("「서 있는 물체」 로 치는 칸의 비율 (칸 × 스캔)")
print("%-28s"%"", "  ".join("%-22s"%k for k in tot))
for rr in RULES:
    print("%-28s"%rr, "  ".join("%-22s"%("%3.0f%% (%d/%d)"%(100*tot[k][rr][0]/max(1,tot[k][rr][1]),tot[k][rr][0],tot[k][rr][1])) for k in tot))
