"""Collision-free wheel waypoints in the current local view."""
import heapq
import numpy as np
from scipy.ndimage import label

def route(goal, obstacles, carrying, tip):
    goal=np.asarray(goal)
    obs=np.asarray(obstacles).reshape(-1,2)
    if not len(obs):return [goal]
    for margin,wide in [(m,True) for m in (.075,.045,.02,0.)]+[(m,False) for m in (.05,.015,-.015)]:
        def clear(p):
            gap=np.abs(p[...,None,:]-obs)
            ok=np.all((gap[...,0]>.55+margin)|(gap[...,1]>.49+margin),axis=-1)
            if carrying:
                gap=np.abs(p[...,None,:]+tip-obs)
                ok &= np.all((gap[...,0]>(.42 if wide else .30)+margin)|(gap[...,1]>(.38 if wide else .30)+margin),axis=-1)
            return ok
        def line(a,b):
            return clear(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.055)+1))).all()
        zero=np.zeros(2)
        # A safe straight approach avoids sweeping the outer corner of the
        # grasp across a previously deposited box during a sideways arrival.
        arrival=goal-np.array([.28,0.]) if carrying and wide else goal
        if carrying and not line(arrival,goal):arrival=goal
        if line(zero,arrival):return [arrival,goal] if np.linalg.norm(arrival-goal)>.01 else [goal]
        step=.13
        low=np.minimum(zero,goal)-3.6;high=np.maximum(zero,goal)+3.6
        axes=[np.arange(l,h+step,step) for l,h in zip(low,high)]
        coords=np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1)
        valid=clear(coords)
        def closest(p,mask):
            return tuple(np.unravel_index(np.argmin(np.where(mask,np.linalg.norm(coords-p,axis=-1),np.inf)),mask.shape))
        start=closest(zero,valid)
        groups,_=label(valid)
        end=closest(arrival,groups==groups[start])
        if np.linalg.norm(coords[end]-arrival)>(.13 if wide else .17):continue
        previous={start:None};costs={start:0.};queue=[(0.,start)]
        while queue:
            _,u=heapq.heappop(queue)
            if u==end:break
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
                v=(u[0]+dx,u[1]+dy)
                if not(0<=v[0]<valid.shape[0] and 0<=v[1]<valid.shape[1]) or not valid[v]:continue
                if dx and dy and not(valid[u[0],v[1]] and valid[v[0],u[1]]):continue
                cost=costs[u]+(1.414214 if dx and dy else 1.)
                if cost>=costs.get(v,np.inf):continue
                costs[v]=cost;previous[v]=u
                heapq.heappush(queue,(cost+np.hypot(v[0]-end[0],v[1]-end[1]),v))
        if end not in previous:continue
        path=[];u=end
        while u is not None:
            path.append(coords[u]);u=previous[u]
        dense=path[::-1]+[arrival]
        corners=[];position=zero
        while dense:
            for j in range(len(dense)-1,-1,-1):
                if line(position,dense[j]):break
            position=dense[j]
            if np.linalg.norm(position-(corners[-1] if corners else zero))>.025:corners.append(position)
            dense=dense[j+1:]
        if np.linalg.norm(arrival-goal)>.01:corners.append(goal)
        return corners or [goal]
    raise RuntimeError("observed obstacles block destination stance")
