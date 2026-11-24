"""Detect cube tops from orange local RGB-D pixels and camera calibration."""
import numpy as np
from scipy import ndimage

def centers(world, cam):
    rgb,depth=world.observation(cam)
    t,r=world.geometry(cam)
    h,w=depth.shape
    f=h/(2*np.tan(np.deg2rad(world.m.cam_fovy[world.sensors[cam][0]])/2))
    red,green,blue=rgb.astype(np.float32).transpose(2,0,1)
    mask=(red>1.55*green)&(green>1.25*blue)&np.isfinite(depth)
    yy,xx=np.nonzero(mask)
    if not len(xx):return []
    d=depth[yy,xx]
    pts=np.column_stack(((xx+.5-w/2)*d/f,(h/2-yy-.5)*d/f,-d))@r.T+t
    # Horizontal patches have locally constant body-relative height, even
    # at oblique incidence. Remove vertical side returns and edge pixels.
    height=np.full((h,w),np.nan,np.float32);height[yy,xx]=pts[:,2]
    below=np.roll(height,1,axis=0);above=np.roll(height,-1,axis=0)
    flat=(abs(height-below)<.003)|(abs(height-above)<.003)
    top=flat&(height>.16)&(height<.35)
    labels,n=ndimage.label(top)
    if not n:return []
    idx=labels[yy,xx]
    output=[]
    for lbl in np.unique(idx[idx>0]):
        xy=pts[idx==lbl,:2]
        if len(xy)<7:continue
        mean=xy.mean(0)
        if len(xy)>250:xy=xy[::int(np.ceil(len(xy)/250))]
        angles=np.arange(0,np.pi/2,np.pi/24)
        cs,sn=np.cos(angles),np.sin(angles)
        axes=np.stack([cs,-sn,sn,cs],axis=-1).reshape(-1,2,2)
        projected=(xy-mean)@axes
        hi=projected.max(1);lo=projected.min(1);span=hi-lo
        i=np.argmin(span.prod(1))
        if span[i].min()<.135 or span[i].max()>.32:continue
        center=mean+(hi[i]+lo[i])*.5@axes[i].T
        output.append(center)
    return output

def face(world,center,raised=False):
    """Measure side-plane azimuth from consecutive RGB-D pixels on this cube."""
    for cam in ('ego','scan_front','close'):
        rgb,depth=world.observation(cam)
        red,green,blue=rgb.astype(float).transpose(2,0,1)
        orange=(red>1.55*green)&(green>1.25*blue)&(depth<2.8)
        y,x=np.nonzero(orange[:-1,:-1]&orange[1:,:-1]&orange[:-1,1:])
        if len(x)<16:continue
        t,r=world.geometry(cam)
        h,w=depth.shape
        f=h/(2*np.tan(np.deg2rad(world.m.cam_fovy[world.sensors[cam][0]])/2))
        def points(v,u):
            d=depth[v,u]
            return np.column_stack(((u+.5-w/2)*d/f,(h/2-v-.5)*d/f,-d))@r.T+t
        p=points(y,x)
        n=np.cross(points(y,x+1)-p,points(y+1,x)-p)
        n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-10)
        near=(np.linalg.norm(p[:,:2]-center,axis=1)<(.235 if raised else .208))
        valid=near&(p[:,2]>(.29 if raised else .03))&(abs(n[:,2])<.23)
        if valid.sum()<15:continue
        angles=np.arctan2(n[valid,1],n[valid,0])
        vote=np.mean(np.exp(4j*angles))
        if abs(vote)>.94:return np.angle(vote)/4
    raise RuntimeError("cube side not visible at hands")
