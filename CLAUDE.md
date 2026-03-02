# CLAUDE.md

## Project Overview

基于摇摆方程（Swing Equation）的电网稳定性与韧性研究项目，复现 Smith et al. (2022) *Science Advances* 论文图表，并扩展拓扑与韧性实验。

## Tech Stack

- **Language**: Python 3.13
- **Environment**: `.venv/` (virtualenv)，激活方式 `source .venv/bin/activate`
- **Core Libraries**: numpy, scipy, pandas, matplotlib, seaborn, networkx, tqdm, pyarrow
- **Backend**: 运行绘图脚本时使用 `MPLBACKEND=Agg` 避免 GUI 弹窗

## Project Structure

```
reproduction/          # 论文图表复现 (Figure 1–6, S2, WS topology)
  cache/                 # 缓存文件 (.npz)，计算成本极高
  output/                # 输出图片
2nodes/                # 两节点稳定性分析
household/             # 家庭用电 / 光伏 / 微电网分析
  consumer/              # LCL 用电统计分析
  generation/            # 光伏发电分析
  node/                  # 家庭微电网模型（含储能）
Topology2.0/           # 拓扑与韧性扩展实验
  ws_topology_effects/   # WS 拓扑效应（已模块化重构）
    ws_stability/          # 稳定性计算：κ_c 映射、Lorenz/Gini 分析
    swing_cascade/         # 级联故障：二分搜索 α_critical、级联持续时间
  experiment_ratio_simplex/ # 三元单纯形：4 种拓扑族 (WS/RGG/SBM/CP) × 节点配比扫描
  cascade_resilience/    # 级联韧性：恢复时间、服务水平 S_PCC(t)、3D 参数扫描
ratio_scan/            # 节点配比扫描与稳定性分析
  cache/                 # 扫描结果缓存 (raw_results.csv, stability_results.csv 等)
  results/               # 输出图片与摘要报告
  output/                # 额外输出图片
Q1Topology/            # Q1 作业：WS 网络分析
data/                  # 原始数据 (LCL / PV / 天气 / 电价)
reference_code/        # 教师原始 Julia/Python 参考实现
docs/                  # 文档与参考资料
```

## Critical Rules

- **NEVER delete `reproduction/cache/` 中的文件**。这些 `.npz` 缓存文件计算成本极高（数小时到数天），且可能不在 git 历史中。修改脚本时必须保留已有缓存逻辑。
- **NEVER delete `Topology2.0/` 中的缓存目录**（`ws_topology_effects/ws_stability/cache/`、`ws_topology_effects/swing_cascade/cache/`、`experiment_ratio_simplex/cache/`、`cascade_resilience/cache/`），同理。
- **NEVER delete `ratio_scan/cache/`**，同理。
- 修改复现脚本时，确保缓存命中逻辑不变（先检查缓存文件是否存在，存在则跳过计算）。

## Running Scripts

```bash
# 激活环境
source .venv/bin/activate

# 运行复现脚本（示例）
MPLBACKEND=Agg .venv/bin/python reproduction/figure1.py

# 运行两节点分析
MPLBACKEND=Agg .venv/bin/python 2nodes/two_node_stability_analysis.py

# 运行节点配比扫描
MPLBACKEND=Agg .venv/bin/python ratio_scan/run_ratio_scan.py
MPLBACKEND=Agg .venv/bin/python ratio_scan/run_stability_scan.py

# 运行绘图
MPLBACKEND=Agg .venv/bin/python ratio_scan/plot_simplex_panels.py
MPLBACKEND=Agg .venv/bin/python ratio_scan/plot_stability_panels.py
MPLBACKEND=Agg .venv/bin/python ratio_scan/plot_passive_impact.py
```

## Code Conventions

- 脚本使用中文注释
- 缓存文件格式：`.npz`（numpy compressed）、`.pkl`（pickle）、`.csv`（表格数据）
- 图表脚本命名：`figure{N}.py`，输出命名：`figure{N}_*.png`
- 每个模块有独立的 `output/` 和 `cache/` 子目录
- 计算密集任务使用 `multiprocessing.Pool(cpu_count())` 并行

## Git Workflow

- **Main branch**: `main`
- **Feature branches**: `feat/<feature-name>`
- PR 流程：feature branch → PR → merge to main
- Repo: `git@github.com:Charlotte-Yan35/GW2.git`

## User Preferences

- 沟通语言：中文
- commit message：中英文均可
- 偏好直接执行，减少确认步骤
- 图表输出为 PNG 格式（300 dpi）
