"""Control using only body-frame RGB-D observations and robot commands."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .perception import face
from .planning import plan
from .route import route
from .robot import CAMERAS
def rot(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s],[s,c]])
def wrap(angle):return (angle+np.pi/4)%(np.pi/2)-np.pi/4
def pivot(policy,p,goal,error):
    rate=float(np.clip(1.15*error,-.20,.20))
    translation=1.1*(p-goal)-rate*np.array([-p[1],p[0]])
    policy.eye.turn(rate)
    policy.drive(np.clip(translation,-.24,.24),.28)
class DroppedBox(RuntimeError):
    pass

class Controller:
    def __init__(self,robot,design):
        self.eye=robot
        self.tip=robot.tip
        self.design=design
        self.count=len(design)
        self.drive,self.stop,self.grip,self.release=robot.drive,robot.stop,robot.grip,robot.release
        self.placed=0;self.stage="start";self.sensor="ego";self.carrying=False

    def scan(self, names):
        points=[];counts={}
        for name in names:
            self.sensor=name
            observed=self.eye.visible(name);counts[name]=len(observed)
            for p in observed:
                if self.carrying and np.linalg.norm(p-self.tip)<.40:continue
                if all(np.linalg.norm(p-q)>.24 for q in points):points.append(p)
        if len(names)>1:self.best_camera=max(counts,key=counts.get)
        return np.asarray(points).reshape(-1,2)
    def see(self, full=False):
        eyes=CAMERAS if self.discovery else (*CAMERAS,"ego","left","right","rear","close")
        pts=self.scan(eyes if full else [self.preferred])
        if not self.nodes:
            if not len(pts):raise RuntimeError("no cube in first scan")
            self.reference=0
            self.last=pts[0].copy()
            for i,p in enumerate(pts):self.nodes[i]=p-self.last
            self.next_id=len(pts)
            self.measured=dict(enumerate(pts))
            return
        ids=list(self.nodes)
        anchor=self.nodes[self.reference]
        expected=self.last+(np.array([self.nodes[i] for i in ids])-anchor)@rot(self.angle).T
        delta=pts[None,:,:]-expected[:,None,:]
        cost=np.linalg.norm(delta,axis=-1)
        # Vote on compatible box-to-box displacements. A single nearby blob
        # cannot rename the rest of the neighborhood when the view changes.
        shifts=delta[cost<1.0]
        if len(shifts):
            diff=np.linalg.norm(delta[None]-shifts[:,None,None,:],axis=-1)
            residual=diff.min(axis=2)
            inlier=residual<.18
            score=inlier.sum(1)-np.minimum(residual,.18).sum(1)*.40-1.2*np.linalg.norm(shifts,axis=1)
            shift=shifts[np.argmax(score)]
            cost=np.linalg.norm(delta-shift,axis=-1)
        rr,cc=linear_sum_assignment(cost)
        pairs=[(ids[i],j) for i,j in zip(rr,cc) if cost[i,j]<.26]
        if len(pairs)<3 and not full:return self.see(True)
        if not pairs:
            # Body-mounted views change when the head turns and the wrists move.
            # Stop and revisit the last measured bearing if all landmarks disappear.
            if getattr(self,"searching",False):
                raise RuntimeError(f"lost box neighborhood at {self.stage} ({self.placed})")
            self.searching=True
            try:
                v=self.last/max(.001,np.linalg.norm(self.last))
                self.stop()
                for pan in (.54,-.54,0):
                    self.eye.look(pan,.64)
                    try:self.see(True);return
                    except RuntimeError:pass
                    self.drive(.18*v,.38);self.stop()
                raise RuntimeError(f"lost box neighborhood at {self.stage} ({self.placed})")
            finally:self.searching=False
        observed={i:pts[j] for i,j in pairs}
        angles=[]
        for a,(i,j) in enumerate(pairs):
            for k,l in pairs[:a]:
                v=self.nodes[i]-self.nodes[k];w=pts[j]-pts[l]
                if np.linalg.norm(v)<.9 or abs(np.linalg.norm(v)-np.linalg.norm(w))>.14:continue
                angle=math.atan2(v[0]*w[1]-v[1]*w[0],v@w)
                if abs(angle-self.angle)<.18:angles.append(angle)
        if angles:self.angle=.35*self.angle+.65*np.median(angles)
        offsets=np.array([observed[i]-rot(self.angle)@self.nodes[i] for i in observed])
        consensus=np.median(offsets,axis=0)
        self.reference=min(observed,key=lambda i:np.linalg.norm(observed[i]-rot(self.angle)@self.nodes[i]-consensus)+.004*np.linalg.norm(observed[i]))
        self.last=observed[self.reference]
        self.measured=observed
        if self.discovery and len(observed)>=4:
            for i,p in observed.items():
                if i==self.reference:continue
                local=self.nodes[self.reference]+rot(-self.angle)@(p-self.last)
                if np.linalg.norm(self.nodes[i]-local)<.18:
                    self.nodes[i]=.87*self.nodes[i]+.13*local
        if full and self.discovery and len(pairs)>=3:
            for j,p in enumerate(pts):
                if min(np.linalg.norm(p-self.relative(q)) for q in self.nodes.values())>.50:
                    self.nodes[self.next_id]=self.nodes[self.reference]+rot(-self.angle)@(p-self.last)
                    self.measured[self.next_id]=p
                    self.next_id+=1
        if full:self.preferred=self.best_camera
        for i,p in observed.items():
            self.hits[i]=self.hits.get(i,0)+1
            if i not in self.first_bearing:self.first_bearing[i]=p.copy()
            if np.linalg.norm(p-self.first_bearing[i])>.22:self.distinct_views.add(i)
        if self.discovery and full:
            for i in list(self.nodes):
                if i in observed or i==self.reference:continue
                if any(np.linalg.norm(self.nodes[i]-self.nodes[j])<.42 for j in observed):
                    del self.nodes[i];self.hits.pop(i,None);self.distinct_views.discard(i)
    def confirmed(self):
        return sum(self.hits.get(i,0)>=3 and i in self.distinct_views for i in self.nodes)
    def grip_zone(self,radius):
        # Head, navigation, and lower front camera see different faces of a
        # close cube. At release a neighboring stroke can occlude any one view.
        for cam in ("ego","close","scan_front"):
            self.sensor=cam
            near=[p for p in self.eye.visible(cam) if np.linalg.norm(p-self.tip)<radius]
            if near:return min(near,key=lambda p:np.linalg.norm(p-self.tip))
        return None
    def relative(self, target):
        proposals=[p+rot(self.angle)@(target-self.nodes[i]) for i,p in self.measured.items()]
        return np.median(proposals,axis=0)
    def navigate(self, target, stage, exclude=None, tol=.055):
        self.stage=stage
        corners=[]
        previous=np.zeros(2)
        previous_distance=np.inf;stalled=0;replans=0
        for tick in range(420):
            self.see(tick%11==0)
            if stage=="survey scattered boxes" and self.confirmed()>=self.count:
                self.stop();return
            if not corners:
                direct=self.relative(target)-self.tip
                obstacles=[self.relative(q) for q in self.nodes.values()]
                try:local=route(direct,obstacles,self.carrying,self.tip)
                except RuntimeError as exc:
                    raise RuntimeError("no clear approach: "+stage) from exc
                corners=[target+rot(-self.angle)@(p-direct) for p in local[:-1]]+[target]
                previous_distance=np.inf;stalled=0
            final=len(corners)==1
            p=self.relative(corners[0])
            if final and exclude in self.measured and stage=="approach stock box":p=self.measured[exclude]
            error=p-self.tip
            distance=np.linalg.norm(error)
            if distance<(tol if final else .075):
                corners.pop(0)
                previous_distance=np.inf;stalled=0
                if not final:continue
                self.stop()
                if self.carrying:
                    self.see(True)
                    if np.linalg.norm(self.relative(target)-self.tip)>tol+.023 and replans<4:
                        replans+=1;previous=np.zeros(2);continue
                return
            stalled=stalled+1 if distance>previous_distance-.014 else 0
            previous_distance=distance
            if stalled>=18 and replans<4:
                corners.clear();replans+=1;self.stop();previous=np.zeros(2);continue
            move=error
            if self.carrying and tick%5==0 and not self.eye.visible("close",True):
                self.stop();self.eye.look(0,.90)
                if not (self.eye.visible("ego",True) or self.eye.visible("close",True)):
                    raise DroppedBox("carried box left hands")
                self.eye.look(0,-.345)
            self.eye.turn(np.clip(.85*self.angle,-.13,.13))
            speed=min(.31 if self.carrying else .44,max(.09,.95*np.linalg.norm(move)))
            command=move*speed/max(.001,np.linalg.norm(move))
            command=previous+np.clip(command-previous,-.14,.14)
            previous=command
            self.drive(command,.30)
        raise RuntimeError(f"{stage} timeout {self.placed}, error={error.round(2)}")
    def lift_centered(self):
        self.eye.turn(0)
        secured=False
        for offset in (-.025,-.048,.0,-.035):
            self.eye.look(0,1.1)
            for _ in range(40):
                self.sensor="ego"
                p=self.grip_zone(.66)
                if p is None:
                    self.eye.look(0,.75);p=self.grip_zone(.74)
                    if p is None:break
                e=p-self.tip-[offset,0]
                if np.linalg.norm(e)<.012:
                    self.stop();self.stage="lift";secured=self.grip()
                    if not secured:self.release()
                    break
                self.drive(np.clip(.86*e,-.16,.16),.25)
            if secured:break
        if not secured:raise RuntimeError(f"grasp not confirmed at {self.placed}")
    def recover_box(self):
        self.carrying=False
        self.release()
        self.see(True)
        candidates=self.scan(("ego","close",*CAMERAS))
        others=np.array([self.relative(q) for q in self.nodes.values()])
        loose=[p for p in candidates if np.min(np.linalg.norm(others-p,axis=1))>.32]
        if not loose:raise RuntimeError("dropped cube not separable from neighbors")
        lost=min(loose,key=lambda p:np.linalg.norm(p-self.tip))
        if np.linalg.norm(lost-self.tip)>1.4:raise RuntimeError("dropped cube identity ambiguous")
        q=self.nodes[self.reference]+rot(-self.angle)@(lost-self.last)
        self.nodes[-1]=q
        self.go(q,"approach stock box",-1,tol=.17)
        self.secure()
        self.carrying=True;self.see(True)
        if self.reference==-1:raise RuntimeError("no remaining anchor after regrasp")
        del self.nodes[-1];self.measured.pop(-1,None)
    def run(self):
        self.hits={};self.first_bearing={};self.distinct_views=set();self.nodes={};self.angle=0.;self.preferred=CAMERAS[0];self.discovery=True
        self.see(True)
        # Inspect the outer boundary, where the wheels have room to pass.
        # Follow adjacent points around the observed perimeter, starting
        # with the closest outside vantage instead of crossing unknown clutter.
        directions=np.array([[np.cos(a),np.sin(a)] for a in np.arange(16)*np.pi/8])
        def vantage(direction,rad):
            points=np.array(list(self.nodes.values()))
            return points[np.argmax(points@direction)]+rad*direction
        first=min(range(16),key=lambda i:np.linalg.norm(self.relative(vantage(directions[i],1.15))))
        for j in range(48):
            if self.confirmed()>=self.count:break
            v=directions[(first+j)%16]
            stand=vantage(v,1.15+.12*(j//16))
            try:self.go(stand+self.tip,"survey scattered boxes",tol=.25)
            except RuntimeError as exc:
                if "clear approach" not in str(exc) and "timeout" not in str(exc):raise
            self.see(True)
        if self.confirmed()<self.count:raise RuntimeError(f"survey confirmed {self.confirmed()} distinct boxes")
        keep=sorted((i for i in self.nodes if i in self.distinct_views),key=lambda i:-self.hits.get(i,0))[:self.count]
        coordinates=[self.nodes[i] for i in keep]
        valid={i:p for i,p in self.measured.items() if i in keep}
        if not valid:raise RuntimeError("no confirmed survey anchor")
        ref=next(iter(valid))
        self.reference=keep.index(ref);self.last=valid[ref]
        self.measured={keep.index(i):p for i,p in valid.items()}
        self.nodes=dict(enumerate(coordinates))
        self.discovery=False
        stock=set(self.nodes)
        # All planning coordinates are relative to the first seen cube; the
        # wheel controller continues to look at physical boxes on every leg.
        survey=np.array([self.relative(q) for q in self.nodes.values()])
        pairs=plan(survey,self.design,np.zeros(2),self.tip)
        # Express each observed-source-to-slot displacement in box axes.
        pairs=[(i,self.nodes[i]+rot(-self.angle)@(slot-survey[i])) for i,slot in pairs]
        if hasattr(self,"log_plan"):
            self.log_plan(survey,pairs,self.nodes)
        print("survey and transfer plan complete",flush=True)
        for k,(src,slot) in enumerate(pairs):
            print("pick",k,src,flush=True)
            self.go(self.nodes[src],"approach stock box",src,tol=.17)
            self.secure()
            self.carrying=True
            self.see(True)
            if self.reference==src and len(self.measured)>1:
                self.reference=next(i for i in self.measured if i!=src);self.last=self.measured[self.reference]
            if self.reference==src:
                raise RuntimeError("no independent reference on lift")
            del self.nodes[src];stock.remove(src);self.measured.pop(src,None)
            for retry in range(4):
                try:
                    self.go(slot+[.020,0],"place relative to neighbor",tol=.039)
                    break
                except DroppedBox:
                    if retry==3:raise
                    self.recover_box()
            self.stage="release"
            self.eye.turn(0);self.release();self.carrying=False
            self.see(True)
            self.eye.look(0,1.10)
            p=self.grip_zone(.32)
            if p is None:
                self.eye.look(.34,.90);p=self.grip_zone(.38)
            if p is None:raise RuntimeError("release not visible in grasp zone")
            self.nodes[self.count+k]=slot+rot(-self.angle)@(p-self.relative(slot))
            self.measured[self.count+k]=p
            self.placed+=1;self.on_placement()
            print("released",self.placed,flush=True)
            self.eye.look(0,.44)
        self.stage="finished";self.stop()
    def ground(self):
        p=self.grip_zone(.48)
        if p is not None:
            try:return p,face(self.eye,p)
            except RuntimeError:pass
        # Revisit the grasp zone from different directions with the actual head.
        for pan,pitch in ((0,.78),(.32,1.02),(-.32,1.02),(0,1.10)):
            self.eye.look(pan,pitch)
            p=self.grip_zone(.54)
            if p is not None:
                try:return p,face(self.eye,p)
                except RuntimeError:pass
        raise RuntimeError("could not reacquire cube faces in local grasp zone")
    def secure(self):
        self.eye.look(0,1.10)
        p,edge=self.ground()
        origin_angle=self.angle
        self.pivot_node=self.nodes[self.reference]+rot(-self.angle)@(p-self.last)
        start_edge=edge
        for tick in range(42):
            # The observed box stays still until the hands close. Its edges
            # and center give the orientation and offsets of nearby landmarks.
            self.angle+=wrap(origin_angle+edge-start_edge-self.angle)
            self.last=p+rot(self.angle)@(self.nodes[self.reference]-self.pivot_node)
            self.see(tick%4==0)
            if abs(edge)<np.deg2rad(2.0) and np.linalg.norm(p-self.tip+[.025,0])<.033:break
            pivot(self,p,self.tip-[.025,0],edge)
            p,edge=self.ground()
        self.eye.turn(0);self.stop()
        self.lift_centered()
    def restore(self):
        # During the pivot, the hands stay over the visually observed pickup.
        # Their bearing to one fixed neighbor can resolve rotation even if a
        # second neighbor is temporarily hidden behind the robot's arms.
        for tick in range(40):
            self.see(tick%3==0)
            angles=[]
            for i,p in self.measured.items():
                a=self.nodes[i]-self.pivot_node
                b=p-(self.tip-[.025,0])
                if np.linalg.norm(a)<1.0 or abs(np.linalg.norm(a)-np.linalg.norm(b))>.18:continue
                theta=np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)
                if abs(theta-self.angle)<.18:angles.append(theta)
            if angles and len(self.measured)<3:self.angle=.4*self.angle+.6*np.median(angles)
            error=self.angle-self.plan_angle
            if abs(error)<np.deg2rad(1.5):break
            if not angles and len(self.measured)<2:break
            pivot(self,self.tip,self.tip,error)
        self.eye.turn(0);self.stop()
    def go(self,target,stage,*args,**kwargs):
        if stage=="approach stock box" and not hasattr(self,'plan_angle'):self.plan_angle=self.angle
        self.release=self.eye.release
        if stage=="place relative to neighbor":
            self.restore()
            drive=self.drive
            def bounded(velocity,seconds):
                speed=np.linalg.norm(velocity)
                drive(velocity*min(1.,.23/max(.001,speed)),seconds)
            self.drive=bounded
            try:self.navigate(target,stage,*args,**kwargs)
            finally:self.drive=drive
            self.slot=target-np.array([.020,0])
            self.release=self.checked_release
        else:self.navigate(target,stage,*args,**kwargs)
    def checked_release(self):
        self.release=self.eye.release
        for attempt in range(9):
            # Compensate a small residual pickup yaw using visible neighbor bearings.
            self.eye.look(0,.90)
            for tick in range(24):
                self.see(tick%4==0)
                try:observed=face(self.eye,self.tip,raised=True)
                except RuntimeError:break
                error=wrap(self.angle-self.plan_angle-observed)
                delta=self.relative(self.slot)-(self.tip-[.025,0])
                if abs(error)<np.deg2rad(1.4) and np.linalg.norm(delta)<.032:break
                if abs(self.angle-self.plan_angle)>np.deg2rad(22) and abs(error)>np.deg2rad(3):break
                rate=float(np.clip(error,-.13,.13))
                self.eye.turn(rate)
                self.drive(np.clip(delta-rate*np.array([0,self.tip[0]]),-.19,.19),.27)
            self.eye.turn(0);self.stop();self.eye.release();self.eye.look(0,1.10)
            self.see(True)
            p,yaw=self.ground()
            err=abs(np.rad2deg(wrap(yaw-self.angle+self.plan_angle)))
            dist=np.linalg.norm(p-self.relative(self.slot))
            print(f"inspect box {self.placed+1} attempt {attempt+1} angle {err:.2f} deg offset {dist:.3f} m",flush=True)
            if err<=3.8 and dist<=.075: return
            if attempt<8:
                self.stage="regrasp and replace"
                self.secure()
                self.restore()
                self.stage="release"
        raise RuntimeError("box still outside visual placement tolerance after 9 releases")
