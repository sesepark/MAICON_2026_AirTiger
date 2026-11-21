import sys, json, math, glob
sys.stdout.reconfigure(encoding="utf-8")
def seg_d(px,py,a,b):
    ax,ay=a; bx,by=b; dx,dy=bx-ax,by-ay; L=dx*dx+dy*dy
    u=0 if L==0 else max(0,min(1,((px-ax)*dx+(py-ay)*dy)/L))
    return math.hypot(px-ax-u*dx,py-ay-u*dy)
def clear(path,pts):
    return min((seg_d(x,y,a,b) for x,y in pts for a,b in zip(path,path[1:])),default=None)
tot={"n":0,"short":0,"long":0,"opened":0,"closed":0,"same":0}
print("판        순간  묵은칸(전→후, 가운데)  길 같음  짧아짐(≥0.3m)  길어짐(≥0.3m)  없던 길 생김  있던 길 없어짐  줄어든 길이 합  새 길–UGV 자리 최소 틈 / 그 UGV 라이다 점까지")
for d in sorted(glob.glob("bt/2026*")):
    o=json.load(open(d+"/old.json")); n=json.load(open(d+"/new.json"))
    R=list(zip(o["rows"],n["rows"])); assert all(a["t"]==b["t"] for a,b in R)
    s=dict.fromkeys(tot,0); gain=0.0; cmin=None; tmin=None; ex=[]
    oc=sorted(a["old_cells"] for a,_ in R); nc=sorted(b["old_cells"] for _,b in R)
    for a,b in R:
        s["n"]+=1
        if a["len"] is None and b["len"] is not None: s["opened"]+=1
        elif a["len"] is not None and b["len"] is None: s["closed"]+=1
        elif a["len"] is None: s["same"]+=1
        elif a["len"]-b["len"]>=0.3: s["short"]+=1; gain+=a["len"]-b["len"]; ex.append((round(a["len"]-b["len"],2),a["t"],a["pose"],a["target"],a["len"],b["len"]))
        elif b["len"]-a["len"]>=0.3: s["long"]+=1; ex.append((round(a["len"]-b["len"],2),a["t"],a["pose"],a["target"],a["len"],b["len"]))
        else: s["same"]+=1
        if b["path"] and b["path"]!=a["path"]:
            for sx,sy in b["plan"]:
                c=clear(b["path"],[(sx,sy)]); cmin=c if cmin is None else min(cmin,c)
                pts=n["truth"].get("%s,%s"%(sx,sy),[])
                if pts:
                    c=clear(b["path"],pts); tmin=c if tmin is None else min(tmin,c)
    for k in tot: tot[k]+=s[k]
    print("%s %4d   %3d → %3d            %4d     %3d           %3d            %3d          %3d          %5.1f m     %s / %s"%(d[-11:],s["n"],oc[len(oc)//2],nc[len(nc)//2],s["same"],s["short"],s["long"],s["opened"],s["closed"],gain,None if cmin is None else round(cmin,2),None if tmin is None else round(tmin,2)))
    for e in sorted(ex,reverse=True)[:4]+[x for x in ex if x[0]<0][:4]: print("        %+.2f m  t=%s 자리 %s → 목표 %s : %s → %s m"%e)
print("합",tot)
