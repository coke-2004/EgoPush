"""Choose an assembly strip and a reachable sequence using observed box relations."""
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import cdist


STEP=.16

def plan(sources, design, initial_stand, tip):
    count=len(design)
    pts=np.asarray(sources)
    center=np.median(pts,axis=0)
    bounds=np.vstack([pts,initial_stand])
    low=bounds.min(0)-6
    high=bounds.max(0)+6
    shape=np.ceil((high-low)/STEP).astype(int)
    X,Y=np.meshgrid(low[0]+STEP*np.arange(shape[0]),low[1]+STEP*np.arange(shape[1]),indexing="ij")
    def mask(p,foot,loose):
        if foot:return (abs(X-p[0])<(.65 if loose else .71))&(abs(Y-p[1])<(.59 if loose else .65))
        return (abs(X+tip[0]-p[0])<(.37 if loose else .44))&(abs(Y-p[1])<(.37 if loose else .415))
    def idx(p):
        return tuple(np.round((p-low)/STEP).astype(int))
    directions=generate_binary_structure(2,1)
    def sequence(slots,refs,loose=False):
        locations=np.vstack([pts,slots])
        base=np.array([mask(p,True,loose) for p in locations])
        front=np.array([mask(p,False,loose) for p in locations])
        stands=locations-tip
        stand_indices=[idx(p) for p in stands]
        ix,iy=np.array(stand_indices).T
        blocks=base[:,ix,iy]
        distances=cdist(locations, locations)
        support_distance=cdist(stands,locations)
        stock=list(range(len(pts)));done=[];pairs=[]
        stand=initial_stand
        for step in range(len(pts)):
            active=stock+done
            solid=base[active].sum(0)
            grip=front[active].sum(0)
            area,_=label(solid==0,directions)
            start=area[idx(stand)]
            if not start:
                if loose:return
                nearby=(X-stand[0])**2+(Y-stand[1])**2
                nearest=np.unravel_index(np.argmin(np.where(area,nearby,np.inf)),area.shape)
                if nearby[nearest]>.14**2:return
                start=area[nearest]
            # Work from the far side toward the hands, while keeping each
            # deposit connected to the real objects that will still be there.
            good=[]
            unfinished=[j for j in range(count,2*count) if j not in done]
            allowed=[j for j in unfinished if not np.any(blocks[j,unfinished])]
            # The newly deposited box never covers its own base stance.
            if not allowed:return
            for src in stock:
                grasp=stand_indices[src]
                if area[grasp]!=start:continue
                clear=(solid-base[src]+grip-front[src])==0
                reachable,_=label(clear,directions)
                component=reachable[grasp]
                if component==0:continue
                remain=[i for i in active if i!=src]
                # During a turn the lifted box is occluded. Require another
                # nearby visible landmark at the pickup, including the last one.
                if np.count_nonzero(support_distance[src,remain]<2.95)<refs:continue
                for dest in allowed:
                    target=locations[dest]
                    finish=stand_indices[dest]
                    if reachable[finish]!=component:continue
                    # Keep access to every later slot when this cube is down.
                    if remain:
                        support=support_distance[dest,remain].min()
                        if support>3.3:continue
                    else:support=0.
                    if done:
                        support=distances[dest,done].min()
                    distance=np.linalg.norm(stand-(pts[src]-tip))+np.linalg.norm(pts[src]-target)
                    # Early pick isolated/far-side stock; preserve supports next to the assembly.
                    isolation=distances[src,count:].min()
                    # Prioritize stock that will lose visual contact with the strip.
                    score=(.29*distance+.55*support-1.35*isolation-.13*target[0] if refs==2 else .35*distance+.60*support-.65*isolation-.15*target[0])
                    good.append((score,src,dest))
            if not good:return
            _,src,dest=min(good)
            stock.remove(src);done.append(dest);stand=locations[dest]-tip
            pairs.append((src,slots[dest-count]))
        return pairs
    # Prefer a narrow assembly strip in front of the scattered stock. The
    # last trip then approaches completed strokes from their open side.
    basis=design[:,[1,0]]*[1,-1]
    for refs in (2,1,0):
        candidates=[]
        gaps=(-.9,-1.35,-.55,-1.8,-.2,.25) if refs==2 else (.30,.65,-.10,-.45)
        for sign in (1,-1):
            offsets=basis*sign
            for gap in gaps:
                front=pts[:,0].max()+gap-offsets[:,0].min()
                for dy in (0,.5,-.5,1.,-1.,1.5,-1.5):
                    slots=offsets+[front,center[1]+dy]
                    support=cdist(slots,pts).min(axis=1)
                    score=abs(gap-(-.9 if refs==2 else .30))+.25*abs(dy)+np.maximum(0,support-(2.6 if refs==2 else 2.8)).sum()
                    candidates.append((score,slots))
        for _,slots in sorted(candidates,key=lambda t:t[0]):
            pairs=sequence(slots,refs)
            if pairs:return pairs
    candidates=[]
    for basis in (design,design[:,[1,0]]*[1,-1]):
        for sign in (1,-1):
            offsets=basis*sign
            for x in np.linspace(-1.6,1.6,5):
                for y in np.linspace(-3.0,3.0,9):
                    slots=offsets+center+[x,y]
                    gap=cdist(slots,pts)
                    score=np.mean(np.maximum(0,.65-gap.min(1)))*9+np.max(gap.min(1))*.32+np.mean(gap.min(1))*.1
                    candidates.append((score,slots))
    for _,slots in sorted(candidates,key=lambda t:t[0]):
        pairs=sequence(slots,0,True)
        if pairs:return pairs
    raise RuntimeError("no full box-to-box transfer plan")
