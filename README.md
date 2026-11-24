# EgoPush · 用局部机载视觉摆放 NYU

MuJoCo 中的 RBY1 轮式双臂机器人从随机位置出发，寻找 20 个散放的橙色方箱，再逐个搬成 NYU。控制器读取机载 RGB-D、相机关节角和底盘瞬时角速度。箱群记忆、搬运次序和行驶方向都在运行时计算。

## 运行一次

使用 Python 3.11，在 Linux 上准备 EGL / OpenGL 驱动。录像另需 ffmpeg 和 Noto Sans CJK 字体。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
MUJOCO_GL=egl .venv/bin/python -m arrangement runs/demo 10001
.venv/bin/python -m arrangement.render runs/demo/seed_10001 demo.mp4
```

已有本地工作区可直接使用 `.conda/bin/python`。一次运行会创建独立种子目录，其中 config.json 保存机器人和槽位，result.json 保存逐箱评分及停止原因，trace.npz 保存仿真回放。重复同一种子时使用新的输出目录。

## 从哪里读代码

| 想做什么 | 入口 |
|---|---|
| 观察箱子、估计箱边朝向 | [RGB-D 识别](arrangement/perception.py) |
| 看机器人如何给箱子编号、匹配和重新抓放 | [控制流程](arrangement/controller.py) |
| 决定先搬哪箱、分配哪个槽位 | [搬运规划](arrangement/planning.py) |
| 根据障碍和持箱状态选择行驶路线 | [局部路径规划](arrangement/route.py) |
| 更换本体、标定相机或抓取动作 | [RBY1 适配器](arrangement/robot.py) 和 [扩展说明](docs/extension.md) |
| 更改目标字形 | [米制槽位模板](arrangement/shapes/nyu.json) |
| 查看物理终态如何评分 | [评分代码](arrangement/metrics.py) |
| 查看 20 个新种子的结果和代表视频 | [实验报告](experiments/nyu-9800/REPORT.md) |

本地历史研究材料集中在 `archive/legacy/`，保留此前的成片、实现和报告。活动项目直接加载自己的模型和抓取序列。上一轮实验的[逐次结果](docs/baseline/trials.csv)随库保存。第三方资产的版本和许可记录见[来源说明](THIRD_PARTY_NOTICES.md)。
