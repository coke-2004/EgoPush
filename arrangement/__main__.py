"""Run a fresh fixture-free placement trial."""
import argparse
import importlib
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
from .controller import Controller
from .metrics import score

DEFAULT_SHAPE=Path(__file__).parent/'shapes/nyu.json'
DEFAULT_ROBOT='arrangement.robot:RBY1'

def load_robot(spec):
    module,name=spec.split(':')
    return getattr(importlib.import_module(module),name)

def run(seed,root,shape,robot_spec):
    pattern=json.loads(shape.read_text())
    design=np.asarray(list(pattern.values()),dtype=float)
    if (design.ndim!=2 or design.shape[1]!=2 or not 3<=len(design)<=20 or
        not np.isfinite(design).all() or len(np.unique(design,axis=0))!=len(design)):
        raise ValueError("Shape requires 3 to 20 unique, finite XY slot centers in metres")
    world=load_robot(robot_spec)(seed,len(design))
    out=root/f'seed_{seed}'
    try:
        out.mkdir(parents=True,exist_ok=False)
        (out/'config.json').write_text(json.dumps(dict(seed=seed,robot=robot_spec,slots=pattern),indent=2))
        policy=Controller(world,design)
        frames=[];stages=[];counts=[]
        addresses=world.box_addresses
        def capture():
            frames.append(world.d.qpos.copy());stages.append(policy.stage);counts.append(policy.placed)
        def save_progress():
            print(f"completed {policy.placed}/{len(design)}",flush=True)
        policy.on_placement=save_progress
        world.frame_hook=capture
        capture()
        failure=''
        try:
            policy.run()
            world.step(100)
        except RuntimeError as exc:failure=str(exc)
        capture()
        frames=np.asarray(frames)
        boxes=world.d.qpos[addresses[:,None]+np.arange(7)]
        verdict=score(design,boxes,frames[:,addresses+2],np.asarray(counts),policy.placed)
        np.savez_compressed(out/'trace.npz',qpos=frames,stage=stages,placed=counts)
        result=dict(seed=seed,failure=failure,**verdict)
        pending=out/'result.tmp'
        pending.write_text(json.dumps(result,indent=2))
        pending.replace(out/'result.json')
        print('PASS' if result['passed'] else 'FAIL',flush=True)
        return result
    finally:world.renderer.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('seed',type=int)
    parser.add_argument('--shape',type=Path,default=DEFAULT_SHAPE)
    parser.add_argument('--robot',default=DEFAULT_ROBOT,metavar='module:Class')
    args=parser.parse_args()
    run(args.seed,args.output,args.shape,args.robot)
if __name__=='__main__':main()
