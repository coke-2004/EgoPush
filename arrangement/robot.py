"""RBY1 / WUJI2 physics, actuator commands and calibrated onboard RGB-D."""
import math
from pathlib import Path
from xml.etree import ElementTree as ET
import mujoco as mj
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from .perception import centers

ASSETS=Path(__file__).parent/'robots/rby1'
CAMERAS=('scan_front','scan_left','scan_back','scan_right')
ALIASES=dict(ego='camera_head_main_rgb',close='camera_navigation_rgb',left='flank_left',right='flank_right',rear='deck_rear',**{c:c for c in CAMERAS})
RANGE=4.15
AXES=np.array([[[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]] for t in np.deg2rad(np.arange(-45,46,3))])
WHEELS=np.array([[1,1,.488],[1,-1,-.488],[1,-1,.492],[1,1,-.492]])/.076

def xmlvec(values):
    return " ".join(map(str,values))

def scene(seed,count):
    root=ET.parse(ASSETS/'scene.xml').getroot()
    for asset in root.find('asset'):
        if asset.get('file'):asset.set('file',str(ASSETS/asset.get('file')))
    world=root.find('worldbody')
    rng=np.random.default_rng(seed)
    center=rng.uniform(-1.5,1.5,2)
    boxes=[]
    for _ in range(100000):
        xy=center+rng.uniform([-2.6,-2.4],[2.6,2.4])
        if all(np.linalg.norm(q-xy)>.80 for q in boxes):boxes.append(xy)
        if len(boxes)==count:break
    else:raise ValueError("too many boxes for the scatter area")
    for i,xy in enumerate(boxes):
        cube=ET.SubElement(world,'body',name=f'b{i}',pos=xmlvec([*xy,.132]),euler=xmlvec([0,0,np.deg2rad(rng.uniform(-180,180))]))
        ET.SubElement(cube,'freejoint')
        ET.SubElement(cube,'geom',name=f'box{i}',type='box',size='.13 .13 .13',mass='.6',rgba='.95 .28 .065 1')
        ET.SubElement(cube,'site',name=f'lid{i}',pos='0 0 .13',size='.001',rgba='0 0 0 0')
    spawn=np.random.default_rng([seed,93013])
    xy=np.array(boxes)
    while True:
        start=xy.mean(0)+spawn.uniform(-3.2,3.2,2)
        if 1.05<np.linalg.norm(xy-start,axis=1).min()<2.2:break
    base=world.find("body[@name='base']")
    base.set('pos',xmlvec([*start,.004]))
    base.set('euler',xmlvec([0,0,spawn.uniform(-np.pi,np.pi)]))
    return ET.tostring(root,encoding='unicode')

class RBY1:
    tip=np.array([.842,0.])
    def __init__(self,seed,count,width=512,height=320):
        self.width,self.height=width,height
        self.m=mj.MjModel.from_xml_string(scene(seed,count))
        self.d=mj.MjData(self.m)
        self.box_addresses=np.array([self.m.jnt_qposadr[self.m.body(f"b{i}").jntadr.item()] for i in range(count)])
        self.renderer=mj.Renderer(self.m,height,width)
        self.opt=mj.MjvOption();self.opt.geomgroup[2]=0
        with np.load(ASSETS/'grasp.npz') as archive:self.frames=archive['ctrl']
        self.wheel_ids=np.array([self.m.actuator(f"wheel_{w}_act").id for w in ("fr","fl","rr","rl")])
        self.upper=np.setdiff1d(np.arange(self.m.nu),self.wheel_ids)
        self.d.ctrl[self.upper]=self.frames[0,self.upper]
        for a in self.upper:
            j=self.m.actuator_trnid[a,0]
            self.d.qpos[self.m.jnt_qposadr[j]]=self.frames[0,a]
        self.n=0;self.frame_hook=None;self.gain=np.ones(2);self.correction=0
        self.step(1000)
        floor=self.m.geom("floor").id
        # MuJoCo maps one repeat over the extent of a finite 20.48 m floor.
        self.m.geom_size[floor,:2]=10.24
        self.m.geom_matid[floor]=self.m.material("vision_floor_material").id
        self.m.geom_rgba[floor]=[1,1,1,1]
        for i in range(count):
            self.m.geom_rgba[self.m.geom(f"box{i}").id]=[.8,.26,.18,1]

        self.sensors={}
        for alias,name in ALIASES.items():
            i=self.m.camera(name).id
            self.m.cam_fovy[i]=65 if alias=='ego' else 68 if alias in ('left','right','rear') else 58
            if alias=='close':
                # Same factory attachment point; aim across the grasp zone.
                self.m.cam_quat[i]=Rotation.from_matrix(np.array([[0,0,-1],[-1,0,0],[0,1,0]])).as_quat(scalar_first=True)
            chain=[];bid=self.m.cam_bodyid[i]
            while bid!=self.m.body('base').id:
                chain.append(bid);bid=self.m.body_parentid[bid]
            self.sensors[alias]=(i,chain[::-1])
        self.head=self.m.actuator("head_0_act").id,self.m.actuator("head_1_act").id
        self.look(0,.44)
        self.audit=[];self.rate_bias=0.;self.previous_yaw_command=0.
        for cam in CAMERAS:self.m.cam_fovy[self.m.camera(cam).id]=74
    def turn(self,rate):
        self.correction=rate
    def rgb(self,cam):
        self.renderer.update_scene(self.d,camera=ALIASES.get(cam,cam),scene_option=self.opt)
        return self.renderer.render()
    def step(self,n,grab=False):
        while n>0:
            k=min(n,200-self.n%200) if self.frame_hook else n
            mj.mj_step(self.m,self.d,nstep=k);self.n+=k;n-=k
            if self.frame_hook and self.n%200==0:self.frame_hook()
    def stop(self):
        self.d.ctrl[self.wheel_ids]=0
        self.step(180)
    def motion(self,frames):
        self.stop()
        for frame in frames:
            self.d.ctrl[self.upper]=frame[self.upper];self.step(10)
    def drive(self,v,seconds=.26):
        # Wheel-level rate damping only; the gyro never supplies an integrated
        # angle or position to the visual guidance policy.
        remaining=round(seconds/.002)
        while remaining:
            steps=min(20,remaining)
            rate=self.d.sensor("drive_rate").data[2]
            self.rate_bias=.88*self.rate_bias+.12*np.clip(self.previous_yaw_command-rate,-.27,.27)
            yaw=self.correction+self.rate_bias+.65*(self.correction-rate)
            self.previous_yaw_command=np.clip(yaw,-.32,.32)
            self.d.ctrl[self.wheel_ids]=np.clip(WHEELS@np.r_[v,np.clip(yaw,-.32,.32)],-7,7)
            self.step(steps);remaining-=steps
    def look(self,pan,pitch):
        self.d.ctrl[self.wheel_ids]=0
        self.d.ctrl[list(self.head)]=[pan,pitch]
        self.step(190)
    def observation(self,cam="ego"):
        assert cam in ALIASES
        rgb=self.rgb(cam);self.renderer.enable_depth_rendering()
        try:depth=self.renderer.render()
        finally:self.renderer.disable_depth_rendering()
        f=self.height/(2*np.tan(np.deg2rad(self.m.cam_fovy[self.sensors[cam][0]])/2))
        yy,xx=np.ogrid[:self.height,:self.width]
        far=depth**2*(1+((xx+.5-self.width/2)/f)**2+((yy+.5-self.height/2)/f)**2)>RANGE**2
        rgb[far]=0;depth[far]=np.inf
        return rgb,depth
    def geometry(self,cam):
        # Forward kinematics of body-mounted optics from joint encoders alone.
        i,chain=self.sensors[cam];t=np.zeros(3);r=np.eye(3)
        for b in chain:
            t=t+r@self.m.body_pos[b];r=r@Rotation.from_quat(self.m.body_quat[b],scalar_first=True).as_matrix()
            for j in range(self.m.body_jntadr[b],self.m.body_jntadr[b]+self.m.body_jntnum[b]):
                q=self.d.qpos[self.m.jnt_qposadr[j]]
                dr=Rotation.from_rotvec(q*self.m.jnt_axis[j]).as_matrix()
                t+=r@(self.m.jnt_pos[j]-dr@self.m.jnt_pos[j]);r=r@dr
        return t+r@self.m.cam_pos[i],r@Rotation.from_quat(self.m.cam_quat[i],scalar_first=True).as_matrix()
    def visible(self,cam,raised=False):
        if cam in CAMERAS and not raised:
            points=centers(self,cam)
            self.audit.append((self.n,cam,len(points),0))
            return points
        rgb,dep=self.observation(cam);t,r=self.geometry(cam)
        f=self.height/(2*np.tan(np.deg2rad(self.m.cam_fovy[self.sensors[cam][0]])/2))
        red,green,blue=rgb.astype(float).transpose(2,0,1)
        mask=(red>1.55*green)&(green>1.25*blue)&(dep<RANGE)
        # Remove pixels immediately across a depth discontinuity before connectivity.
        separate=np.zeros(mask.shape,bool)
        for axis in (0,1):
            prev=np.roll(dep,1,axis=axis);valid=mask & np.roll(mask,1,axis=axis)
            separate |= valid & (dep>prev+.19)
            separate |= np.roll(valid & (prev>dep+.19),-1,axis=axis)
        labels,n=ndimage.label(mask & ~separate);points=[];up=False
        for i,sl in enumerate(ndimage.find_objects(labels),1):
            if sl is None:continue
            v,u=np.nonzero(labels[sl]==i)
            if len(u)<6:continue
            v+=sl[0].start;u+=sl[1].start;d=dep[v,u]
            local=np.column_stack(((u+.5-self.width/2)*d/f,(self.height/2-v-.5)*d/f,-d))
            cloud=local[np.linalg.norm(local,axis=1)<RANGE]@r.T+t
            if len(cloud)<5:continue
            high=cloud[(cloud[:,2]>.37)&(cloud[:,2]<.83)]
            if len(high)>12 and np.min(np.linalg.norm(high[:,:2]-self.tip,axis=1))<.29:up=True
            if cloud[:,2].min()>.34 or cloud[:,2].max()<-.07:continue
            if t[2]<.32:
                xy=cloud[::max(1,len(cloud)//160),:2]
                mean=xy.mean(0);proj=(xy-mean)@AXES
                lo=proj.min(axis=1);hi=proj.max(axis=1);span=hi-lo
                ray=(mean-t[:2])@AXES
                signs=np.sign(ray)
                edge=np.where(signs>0,lo,hi)
                # A cube viewed below its lid exposes one or two near sides.
                residual=np.min(abs(proj-edge[:,None,:]),axis=2)
                score=np.mean(np.minimum(residual,.09),axis=1)+np.maximum(0,span-.275).sum(axis=1)
                j=np.argmin(score)
                if max(span[j])>.39:continue
                mid=edge[j]+signs[j]*.13
                mid=np.where(span[j]>.235,(lo[j]+hi[j])/2,mid)
                points.append(mean+mid@AXES[j].T)
                continue
            lid=cloud[cloud[:,2]>cloud[:,2].max()-.045,:2]
            if len(lid)<3:continue
            lo=lid.min(0);hi=lid.max(0)
            if max(hi-lo)>.48:continue
            p=(lo+hi)/2
            ray=p-t[:2];ray=ray/max(np.linalg.norm(ray),.01)
            p+=np.sign(ray)*np.maximum(0,.26-(hi-lo))*.5*np.clip(abs(ray)*2,0,1)
            points.append(p)
        self.audit.append((self.n,cam,len(points),int(up)))
        return up if raised else points
    def grip(self):
        self.motion(self.frames);self.look(0,.90)
        ok=self.visible('ego',True) or self.visible('close',True)
        self.look(0,-.345)
        return ok
    def release(self):
        self.motion(self.frames[::-1]);self.look(0,.44)
