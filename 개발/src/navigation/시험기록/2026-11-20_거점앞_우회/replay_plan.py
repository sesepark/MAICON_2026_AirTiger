"""기록 재생 백테스트: 주행 중 매 스캔(1 Hz)마다 그때 자세 → 그때 목표로 길을 짜 본다. NAV 환경변수 = 쓸 코드."""
import sys, json, math, os
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.environ["NAV"])
from navigation.drive.fieldmap import MapSpec, fixed_pillars
from navigation.drive.geometry import Pose2D, path_length
from navigation.drive.grid import Grid
from navigation.drive.obstacles import ObstacleTracker, scan_to_body
from navigation.drive.params import DriveParams
from navigation.drive.ugvs import UgvTracks
run=sys.argv[1]
nav=[json.loads(l) for l in open(run+"/navigation.jsonl",encoding="utf-8") if l.strip()]
slam=[]
for l in open(run+"/navigation_slam.jsonl",encoding="utf-8"):
    if '"scan_raw"' in l:
        r=json.loads(l)
        if r.get("pose"): r["event"]="scan"; slam.append(r)
p=DriveParams()
RATE={}; STILL={}
for i,r in enumerate(slam):
    w=[];v=[]
    for j in (i-1,i+1):
        if 0<=j<len(slam):
            dt=abs(slam[j]["t"]-r["t"])
            w.append(abs(math.atan2(math.sin(slam[j]["pose"][2]-r["pose"][2]),math.cos(slam[j]["pose"][2]-r["pose"][2])))/dt)
            v.append(math.hypot(slam[j]["pose"][0]-r["pose"][0],slam[j]["pose"][1]-r["pose"][1])/dt)
    RATE[r["t"]]=max(w); STILL[r["t"]]=max(v)<0.03 and max(w)<0.05
poses=sorted([(r["t"],r["pose"]) for r in nav if r.get("pose")]+[(r["t"],r["pose"]) for r in slam])
ev=sorted(nav+slam,key=lambda r:r["t"])
tr=ObstacleTracker(p); ug=UgvTracks(p); tr.ugvs=ug
grid=spec=None; job=None; job_t=0; out=[]; t0=None; frames=0
truth={}   # 길 고르기에 든 UGV 자리 → 서서 본 스캔에서 그 곁 0.30 m 에 찍힌 점들
for r in ev:
    t=r["t"]; e=r.get("event")
    if e=="map":
        w,h=r["size"]; boxes=tuple(tuple(b) for b in r["boxes"])
        spec=MapSpec(w,h,r["box_size"],boxes,"arena",tuple(fixed_pillars(w,h,r["box_size"])) if p.fixed_pillars else ())
        grid=Grid(spec,radius=p.robot_radius,margin=p.inflate_margin,cell=p.cell,soft_m=p.soft_m,soft_cost=p.soft_cost)
    elif e=="goal" and t0 is None: t0=t
    elif e=="ugvs":
        st=r["stamp"]; win=[q[1] for q in poses if st-p.ugv_self_window_s<=q[0]<=st+0.1]
        ug.on_frame(st,r["seen"],[(q[0],q[1]) for q in win]); frames+=1
    elif e=="state":
        j=r.get("job")
        if j and j.get("target") and j.get("mode")!="straight": job=(tuple(j["target"]),j["mode"]); job_t=t
    elif e=="scan" and grid is not None:
        ranges=np.array([x if x else np.nan for x in r["ranges"]],dtype=float)
        body=scan_to_body(ranges,r["angle_min"],r["angle_increment"],p)
        pose=Pose2D(*r["pose"])
        tr.on_scan(t,pose,body,spec,grid,(r["angle_min"],r["angle_increment"],ranges),turn_rate=RATE[t])
        extra=tr.extra_cells(t,grid)
        if STILL[t] and tr._ugv_plan and tr.unexplained_world:
            for sx,sy in tr._ugv_plan:
                truth.setdefault((sx,sy),[]).extend((x,y) for x,y in tr.unexplained_world if math.hypot(x-sx,y-sy)<=0.30)
        if job and t-job_t<=1.2:
            start=(pose.x,pose.y); tgt=job[0]
            full=grid.plan(start,tgt,extra); free=grid.plan(start,tgt)
            L=lambda q: None if q is None else round(path_length(start,q),2)
            out.append({"t":round(t-(t0 or t),1),"pose":[round(pose.x,2),round(pose.y,2)],"target":list(tgt),"len":L(full),"map":L(free),"cells":len(extra),
                        "old_cells":sum(1 for s in tr._seeds.values() if s[1]>t and t-s[2]>p.obstacle_memory_s),"plan":[list(q) for q in tr._ugv_plan],
                        "path":None if full is None else [[round(a,3),round(b,3)] for a,b in [start]+list(full)]})
json.dump({"rows":out,"frames":frames,"truth":{"%s,%s"%k:v for k,v in truth.items()}},open(sys.argv[2],"w"))
print(run,"instants",len(out),"ugv frames",frames)
