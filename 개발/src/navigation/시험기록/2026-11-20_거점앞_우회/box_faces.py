import sys, json, math, glob
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\Admin\MAICON_final\개발\src\navigation")
from navigation.drive.obstacles import scan_to_body
from navigation.drive.params import DriveParams
p=DriveParams()
for d in sorted(glob.glob("bt/2026*")):
    m=[json.loads(l) for l in open(d+"/navigation.jsonl",encoding="utf-8") if '"map"' in l and json.loads(l).get("event")=="map"][-1]
    boxes=m["boxes"]; h=m["box_size"]/2
    rows=[json.loads(l) for l in open(d+"/navigation_slam.jsonl",encoding="utf-8") if '"scan_raw"' in l]; rows=[r for r in rows if r.get("pose")]
    acc={}
    for i,r in enumerate(rows[1:-1],1):
        pose=r["pose"]
        if max(math.hypot(rows[j]["pose"][0]-pose[0],rows[j]["pose"][1]-pose[1]) for j in (i-1,i+1))>0.02: continue
        if max(abs(math.atan2(math.sin(rows[j]["pose"][2]-pose[2]),math.cos(rows[j]["pose"][2]-pose[2]))) for j in (i-1,i+1))>0.03: continue
        ranges=np.array([x if x else np.nan for x in r["ranges"]],dtype=float)
        body=np.array(scan_to_body(ranges,r["angle_min"],r["angle_increment"],p))
        c,s=math.cos(pose[2]),math.sin(pose[2])
        wx=pose[0]+c*body[:,0]-s*body[:,1]; wy=pose[1]+s*body[:,0]+c*body[:,1]
        for k,(bx,by) in enumerate(boxes):
            for name,ax,sign in (("왼",0,-1),("오른",0,1),("아래",1,-1),("위",1,1)):
                face=(bx if ax==0 else by)+sign*h
                # 그 면 밖에 다른 상자가 붙어 있으면(0.3 m 안) 건너뛴다
                if any(j!=k and abs((ox if ax==0 else oy)-(bx if ax==0 else by)-sign*2*h)<0.3 and abs((oy if ax==0 else ox)-(by if ax==0 else bx))<0.3 for j,(ox,oy) in enumerate(boxes)): continue
                me=pose[ax]
                if (me-face)*sign<0.15: continue
                n_=wx if ax==0 else wy; l_=wy if ax==0 else wx; lc=by if ax==0 else bx
                msk=(abs(l_-lc)<0.12)&(abs(n_-face)<0.25)
                if msk.sum()>=5: acc.setdefault((k,name),[]).append(float(np.median(n_[msk])-face)*sign)
    out=[]
    for (k,name),v in sorted(acc.items()):
        if len(v)>=3: out.append((k,name,round(float(np.median(v)),3),len(v)))
    big=[o for o in out if abs(o[2])>=0.06]
    print(d[-15:],"상자 %d개, 서서 3번 넘게 본 면 %d개, 그중 0.06 m 넘게 어긋난 면 %d개 (+ = 상자 바깥쪽으로)"%(len(boxes),len(out),len(big)))
    for k,name,off,n in big: print("     상자 (%.2f, %.2f) %s면 %+.3f m (스캔 %d)"%(boxes[k][0],boxes[k][1],name,off,n))
    if out: print("     모든 면 어긋남 가운데값 %+.3f m, |어긋남| 90%% %.3f m"%(np.median([o[2] for o in out]),np.percentile([abs(o[2]) for o in out],90)))
