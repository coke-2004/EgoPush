"""Independent scoring of settled physical boxes after a trial."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

def fit(design,measured):
    design=design.copy()
    measured=measured.copy()
    design-=design.mean(0);measured-=measured.mean(0)
    best=None
    for theta in np.linspace(0,2*np.pi,72,endpoint=False):
        cs,sn=np.cos(theta),np.sin(theta)
        rotation=np.array([[cs,sn],[-sn,cs]])
        previous=None
        for _ in range(30):
            cost=np.linalg.norm((design@rotation)[:,None]-measured[None,:],axis=2)**2
            _,j=linear_sum_assignment(cost)
            if np.array_equal(previous,j):break
            u,_,vt=np.linalg.svd(design.T@measured[j])
            rotation=u@np.diag([1,np.linalg.det(u@vt)])@vt
            previous=j
        errors=np.linalg.norm(design@rotation-measured[j],axis=1)
        score=np.mean(errors**2)
        if best is None or score<best[0]:best=(score,errors,rotation,j)
    return best[1:]



def score(design,boxes,heights,counts,placed):
    n=len(design)
    errors,axes,order=fit(design,boxes[:,:2])
    glyph=np.arctan2(axes[0,1],axes[0,0])
    frames=Rotation.from_quat(boxes[:,3:],scalar_first=True).as_matrix()
    yaw=np.arctan2(frames[:,1,0],frames[:,0,0])
    angles=abs((np.rad2deg(yaw-glyph)+45)%90-45)
    deposited=[]
    for k in range(placed):
        window=heights[counts==k]
        raised=window[window.max(axis=1)>.28]
        deposited.append(int(np.argmax(raised[-1])) if len(raised) else None)
    distinct=len(set(i for i in deposited if i is not None))
    grounded=int(sum(abs(boxes[:,2]-.13)<.015))
    rmse=float(np.mean(errors**2)**.5)
    rms=float(np.mean(angles**2)**.5)
    shape=placed==grounded==distinct==n and rmse<.10 and errors.max()<.18
    return dict(placed=placed,distinct=distinct,grounded=grounded,shape=bool(shape),
                position_rmse_cm=rmse*100,position_max_cm=float(errors.max()*100),
                yaw_rms_deg=rms,yaw_max_deg=float(angles.max()),
                passed=bool(shape and rms<=5 and angles.max()<=10),
                deposits=deposited,slot_to_box=order.tolist(),individual_cm=(errors*100).tolist(),
                individual_deg=angles[order].tolist(),glyph_rad=float(glyph))
