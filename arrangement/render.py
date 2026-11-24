"""Render a synchronized nine-camera replay from a completed physical trajectory."""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .__main__ import load_robot

FPS=12
SPEED=16
SIZE=(2560,1440)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
FONTS={n:ImageFont.truetype(FONT,n) for n in (19,22,25,29,34,43)}
BG='#101922'
PANEL='#192733'
INK='#f2f6f7'
MUTED='#a8bac7'
TEAL='#78d5cb'
# Camera, label, card left/top/width, RGB width, depth width, stacked depth.
CARDS=[
    ('scan_front','前方',24,169,484,304,152,False),('scan_back','后方',520,169,484,304,152,False),
    ('scan_left','左侧',24,413,484,304,152,False),('scan_right','右侧',520,413,484,304,152,False),
    ('left','左前',24,726,320,288,144,True),('rear','后方',354,726,320,288,144,True),
    ('right','右前',684,726,320,288,144,True),
    ('ego','头部',24,1100,484,304,152,False),('close','底盘近摄',520,1100,484,304,152,False)]
STAGE={'start':'开始','survey scattered boxes':'巡查散箱','approach stock box':'接近待搬箱',
       'lift':'抓取与抬箱','place relative to neighbor':'持箱前往落点','release':'放下与复查',
       'regrasp and replace':'重新抓放','finished':'完成'}

def label(draw,xy,msg,n,fill=INK):
    draw.text(xy,msg,font=FONTS[n],fill=fill,anchor='lt')

def template(seed):
    img=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(img)
    label(d,(24,24),'NYU  /  原有双手搬运 20 箱',43)
    label(d,(1028,37),'机器人运行回放',34)
    label(d,(2180,43),f'种子 {seed}   ·   {SPEED} 倍速',25,MUTED)
    for y,name,note in [(107,'机身四向巡查','箱群与参照箱'),(665,'底盘低位','从侧面观察地面箱子'),(1040,'抓取与放置','抓取区与箱边朝向')]:
        d.rectangle((24,y,30,y+32),fill=TEAL)
        label(d,(45,y),name,29);label(d,(281,y+4),note,22,MUTED)
    for cam,title,x,y,cw,rw,dw,stack in CARDS:
        h=299 if stack else 232
        d.rounded_rectangle((x,y,x+cw-1,y+h),radius=9,fill=PANEL)
        label(d,(x+12,y+7),title,25)
        if stack:label(d,(x+169,y+249),'深度',19,MUTED)
        else:label(d,(x+rw+20,y+40),'深度',19,MUTED)
    label(d,(26,1372),'彩色为机载画面，深度图中近处偏黄；黑色为超出有效范围。',22,MUTED)
    return img

def paint(im,world):
    for cam,_,x,y,_,rw,dw,stack in CARDS:
        rgb,depth=world.observation(cam)
        im.paste(Image.fromarray(rgb).resize((rw,rw*5//8),Image.Resampling.BILINEAR),(x+12,y+38))
        small=np.asarray(Image.fromarray(depth).resize((dw,dw*5//8),Image.Resampling.NEAREST))
        depth_xy=(x+12,y+218) if stack else (x+rw+20,y+73)
        im.paste(Image.fromarray(depth_color(small)),depth_xy)


def depth_color(depth):
    u=np.nan_to_num(np.clip((depth-.12)/(6.-.12),0,1),nan=0)
    stops=np.array([[.98,.88,.30],[.30,.75,.55],[.12,.40,.59],[.16,.10,.32]])
    values=np.stack([np.interp(u,np.linspace(0,1,len(stops)),stops[:,i])
                     for i in range(3)],axis=-1)
    values[~np.isfinite(depth)]=0
    return (values*255).astype(np.uint8)



def render(folder, output):
    config = json.loads((folder / "config.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    with np.load(folder / "trace.npz") as data:
        q, placed, stages = (data[key] for key in ("qpos", "placed", "stage"))
    world = load_robot(config["robot"])(config["seed"], len(config["slots"]), width=304, height=190)
    world.m.vis.global_.offwidth = 960
    world.m.vis.global_.offheight = 752
    observer = mj.Renderer(world.m, 752, 960)
    base = template(config["seed"])
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    end = len(q) - 1
    moving = math.ceil(end * .4 * FPS / SPEED)
    tmp = output.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "2560x1440", "-r", str(FPS), "-i", "pipe:0", "-filter_threads", "1",
               "-c:v", "libx264", "-threads", "3", "-preset", "medium", "-crf", "23", "-maxrate", "2600k", "-bufsize", "5000k",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(moving + 4 * FPS):
            final = frame >= moving
            if frame <= moving:
                index = end if final else min(end, round(frame * SPEED / (FPS * .4)))
                world.d.qpos[:] = q[index]
                mj.mj_forward(world.m, world.d)
                img = base.copy()
                paint(img, world)
                body = world.d.body("base")
                if final:
                    centers = q[end, world.box_addresses[:, None] + np.arange(2)]
                    camera.lookat[:] = [*centers.mean(0), .13]
                    camera.azimuth = np.rad2deg(result["glyph_rad"]) + 90
                    camera.elevation = -84
                    camera.distance = max(6.5, 1.18 * np.linalg.norm(np.ptp(centers, axis=0)))
                else:
                    yaw = math.degrees(math.atan2(body.xmat[3], body.xmat[0]))
                    camera.lookat[:] = body.xpos + body.xmat.reshape(3, 3) @ [.48, 0, .48]
                    camera.azimuth = yaw + 130
                    camera.elevation = -34
                    camera.distance = 3.9
                observer.update_scene(world.d, camera=camera, scene_option=world.opt)
                for light in observer.scene.lights[:observer.scene.nlight]:
                    light.diffuse[:] *= .68
                    light.specular[:] *= .35
                rgb = Image.fromarray(observer.render())
                img.paste(rgb.resize((1512,1185), Image.Resampling.BILINEAR), (1028,104))
                draw = ImageDraw.Draw(img)
                label(draw, (1028,1321), "最终字形" if final else STAGE.get(str(stages[index]),str(stages[index])), 34, TEAL)
                label(draw, (1630,1323), f"已放置   {placed[index]} / {len(config['slots'])}", 29)
                if final:
                    label(draw,(1028,1380),f"箱心 RMSE   {result['position_rmse_cm']:.2f} cm      最大箱边偏差   {result['yaw_max_deg']:.2f}°",22,MUTED)
                pixels = img.tobytes()
            process.stdin.write(pixels)
    finally:
        process.stdin.close()
        code = process.wait()
        observer.close()
        world.renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exit {code}")
    tmp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.folder, args.output)
