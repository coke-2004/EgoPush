"""Summarize one complete consecutive cohort, including failed trials."""
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from math import sqrt

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def histogram(values, output):
    image = Image.new("RGB", (1000, 470), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    draw.text((45, 16), f"Center RMSE across completed trials (n={len(values)})", font=font, fill="black")
    counts, edges = np.histogram(values, bins=8, range=(0, max(2, np.ceil(max(values, default=1)*2)/2)))
    peak = max(1, counts.max())
    for i, n in enumerate(counts):
        left = 64 + i*109
        y = 350 - int(n/peak*245)
        draw.rectangle((left, y, left+100, 350), fill="#257c82")
        draw.text((left+40, y-27), str(n), font=font, fill="black")
        draw.text((left-10, 365), f"{edges[i]:.2f}", font=font, fill="black")
    draw.text((923,365), f"{edges[-1]:.2f}", font=font, fill="black")
    draw.text((425,416), "Position RMSE (cm)", font=font, fill="black")
    image.save(output)


def summarize(folder, first, n):
    rows=[]
    for seed in range(first, first+n):
        path=folder/f"seed_{seed}"/"result.json"
        result=json.loads(path.read_text())
        assert result["seed"] == seed
        rows.append(result)
    keys="seed passed shape placed distinct grounded position_rmse_cm position_max_cm yaw_rms_deg yaw_max_deg failure".split()
    with (folder/"trials.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,extrasaction="ignore",lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    good=[r for r in rows if r["passed"]]
    complete=[r for r in rows if r["placed"]==r["distinct"]==r["grounded"]==20]
    p=len(good)/n
    z=1.96
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z/den*sqrt(p*(1-p)/n+z*z/(4*n*n))
    headline=f"**{len(good)}/{n}（{p:.0%}）**，Wilson 95% 区间 {(mid-half):.1%}–{(mid+half):.1%}。"
    lines=["# 20 个随机散箱摆 NYU · 独立种子复验","",
           "## 本轮结果","",f"种子 {first}–{first+n-1}，每个种子从初始状态运行一次。联合成功率为 "+headline,
           f"完整搬运、最终落稳且由 {20} 个不同实体组成的场次为 {len(complete)}/{n}。单看字形和搬运完整性为 {sum(r['shape'] for r in rows)}/{n}。"]
    if complete:
        vals=[r["position_rmse_cm"] for r in complete]
        lines.append(f"完整场次箱心 RMSE 中位数 {median(vals):.2f} cm，范围 {min(vals):.2f}–{max(vals):.2f} cm。完整场次中最大的单箱朝向偏差为 {max(r['yaw_max_deg'] for r in complete):.2f}°。")
    lines+=["","上一轮 9600–9619 为 18/20（90%）。两轮初始布局不同，不能把比例差归因为代码优化。本轮恢复原有四个补看姿态，额外两个角度已撤回；中止的 9700 系列没有计入分母。未用本轮失败种子调整控制器。","","![完整场次的位置误差分布](histogram.png)","","## 两个代表回放",""]
    if len(good)>=2:
        typical=min(good,key=lambda r:abs(r["position_rmse_cm"]-median([x["position_rmse_cm"] for x in good])))
        difficult=max((r for r in good if r!=typical),key=lambda r:r["yaw_max_deg"])
        for name,row in [("箱心误差接近成功场次中位数",typical),("其余成功场次中单箱朝向偏差最大",difficult)]:
            seed=row["seed"]
            lines.append(f"- {name}，种子 [{seed}](seed_{seed}/replay.mp4)，箱心 RMSE {row['position_rmse_cm']:.2f} cm，最大角差 {row['yaw_max_deg']:.2f}°。")
        (folder/"representatives.json").write_text(json.dumps([typical["seed"],difficult["seed"]]))
    lines+=["","## 放手复查与失败分析",""]
    for r in rows:
        if not r["passed"]:
            lines.append(f"- 种子 {r['seed']}：停止时已登记 {r['placed']} 次摆放，轨迹中对应 {r['distinct']} 个不同箱子，终态有 {r['grounded']} 个箱子落地。停止原因：{r['failure'] or '终态评分超限'}。")
    logs=[folder/f"{row['seed']}.log" for row in rows]
    diagnostics=folder/"release-checks.json"
    if all(path.exists() for path in logs):
        evidence={}
        for path in logs:
            matches=re.findall(r"inspect box (\d+) attempt (\d+) angle ([\d.]+) deg offset ([\d.]+) m",path.read_text())
            evidence[path.stem]=[dict(box=int(k),attempt=int(t),angle_deg=float(a),offset_m=float(d)) for k,t,a,d in matches]
        diagnostics.write_text(json.dumps(evidence,indent=2))
    if diagnostics.exists():
        checks=[x for run in json.loads(diagnostics.read_text()).values() for x in run]
        repeats=sum(x["attempt"]>1 for x in checks)
        lines+=["",f"日志记录了 {len(checks)} 次松手后的视觉复查，其中第 2 次及以后的释放共 {repeats} 次。复查发生在手边，视角内误差和全局对齐后的独立终态误差分别统计。[每次复查读数](release-checks.json)"]
    lines+=["","## 回放和逐场数据","","回放按同一条物理轨迹同步重绘九路 RGB-D 和右侧跟拍，每个视频帧展示该时刻可获取的画面。控制算法实际按需取图。末尾切至外部俯视以便检查全字形。","","## 每次运行","",
            "|种子|已放 / 不同实体 / 落地|箱心 RMSE (cm)|最远 (cm)|箱边 RMS (°)|最歪 (°)|结果或停止原因|","|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        verdict="通过" if r["passed"] else r["failure"] or "终态评分超限，见该种子的 result.json"
        lines.append(f"|{r['seed']}|{r['placed']} / {r['distinct']} / {r['grounded']}|{r['position_rmse_cm']:.2f}|{r['position_max_cm']:.2f}|{r['yaw_rms_deg']:.2f}|{r['yaw_max_deg']:.2f}|{verdict}|")
    lines+=["","中途停止的场次也对现场全部 20 个物体评分，此时数值包含仍散放的箱子。各场保留 result.json 中的逐槽误差、实体分配和搬运记录。[逐场 CSV](trials.csv)","","## 设定、控制与测量","", (Path(__file__).parent.parent/"docs/method.md").read_text().strip(),""]
    notes=folder/"notes.md"
    if notes.exists():
        lines+=["## 本轮讨论与复现","",notes.read_text().strip(),""]
    histogram([r["position_rmse_cm"] for r in complete],folder/"histogram.png")
    (folder/"REPORT.md").write_text("\n".join(lines))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("first",type=int)
    parser.add_argument("count",type=int)
    args=parser.parse_args()
    summarize(args.folder,args.first,args.count)
