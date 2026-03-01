# GW2

基于摇摆方程（Swing Equation）的电网稳定性与韧性研究代码仓库。

主要对应论文：Smith et al., *Science Advances* 8, eabj6734 (2022) —— *The effect of renewable energy incorporation on power grid stability and resilience*。

## 目录结构

```text
GW2/
├── reproduction/           # 论文图表复现（Figure 1–6, S2）
├── 2nodes/                 # 两节点模型：分岔图、相图、稳定性分析
├── household/              # 家庭用电 / 光伏发电 / 单节点微电网
├── Topology2.0/            # 拓扑与韧性扩展实验
│   ├── ws_topology_effects/      # WS 拓扑效应 (κ_c、Gini、级联二分)
│   ├── experiment_ratio_simplex/ # 三元单纯形 × 4 种拓扑族
│   └── cascade_resilience/       # 级联恢复时间与服务水平分析
├── Q1Topology/             # Q1 作业：Watts-Strogatz 网络分析
├── data/                   # 原始数据（LCL / PV / xlsx 等）
├── docs/                   # 文档与复现说明
├── reference_code/         # 参考代码（教师原始 Julia/Python 实现）
└── README.md
```

## 核心物理模型

**摇摆方程（Swing Equation）：**

$$I \frac{d\omega_i}{dt} = P_i - D\omega_i - \kappa \sum_j A_{ij} \sin(\theta_i - \theta_j)$$

$$\frac{d\theta_i}{dt} = \omega_i$$

- $\theta_i$：相角（电压相位）
- $\omega_i$：角频率偏差
- $P_i$：功率注入（正 = 发电机，负 = 负荷）
- $\kappa$：耦合强度
- $A_{ij}$：网络邻接矩阵

**关键指标：**
- $\kappa_c$：临界耦合强度（维持同步的最小 $\kappa$，越低越好）
- $\rho = \kappa_c / \kappa^*$：韧性比（归一化指标）
- 三元单纯形（Ternary Simplex）：发电机 / 负荷 / 被动节点配比空间

## 环境依赖

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy pandas matplotlib seaborn networkx tqdm pyarrow
```

## 数据说明

仓库内已包含主要数据目录：`data/`。

关键输入包括：
1. `data/Small LCL Data/LCL-June2015v2_*.csv` — 伦敦 ~5500 户半小时用电数据
2. `data/PV Data/` — 光伏发电逐小时数据
3. `data/Small_LCL_Data.parquet`（可由脚本生成）

如果缺少 Parquet 文件，可先执行：

```bash
python household/consumer/convert_lcl_to_parquet.py
```

## 运行方式

在仓库根目录执行以下命令。

### 1) 论文图表复现（`reproduction/`）

| 脚本 | 输出 | 说明 |
|------|------|------|
| `figure1.py` | `figure1_AB.png`, `figure1_CD.png` | 三元单纯形上的临界耦合 $\kappa_c$ |
| `figure2.py` | `figure2.png` | 级联故障分析（存活 / 过载 vs $\alpha/\alpha^*$） |
| `figure3.py` | `figure3.png` | 格子 vs 小世界网络的韧性变化 |
| `figure4.py` | `figure4_DEFG.png` | 真实微电网（LCL + PV，冬夏 × 50%/100% PV） |
| `figure6.py` | `figure6_ABCD.png` | PV 渗透率对韧性的影响 |
| `figureS2.py` | `figureS2.png` | 级联持续时间 vs 归一化边容量（补充图） |
| `ws_topology.py` | `ws_topology_*.png` | Watts-Strogatz 拓扑指标 |
| `run_panel_d.py` | 缓存数据 | Figure 1 Panel D 单独计算 |

```bash
python reproduction/figure1.py
python reproduction/figure2.py
python reproduction/figure3.py
python reproduction/figure4.py
python reproduction/figure6.py
python reproduction/figureS2.py
python reproduction/ws_topology.py
```

输出目录：`reproduction/output/`
缓存目录：`reproduction/cache/`（**请勿删除**，计算量大且难以重新生成）

### 2) 两节点稳定性分析（`2nodes/`）

```bash
python 2nodes/two_node_stability_analysis.py
```

输出目录：`2nodes/output/`，包括分岔图、相图、时间序列、随机扰动、级联故障等。

另有简化脚本：

```bash
python 2nodes/two_nodes.py
```

### 3) 家庭侧分析（`household/`）

家庭用电统计：

```bash
python household/consumer/lcl_usage_analysis.py
```

输出：`household/consumer/output/`（月度曲线、热力图、CSV、Markdown 报告）。

家庭光伏发电分析：

```bash
python household/generation/analyze_generation.py
```

输出：`household/generation/output/`。

单节点家庭微电网（含储能）：

```bash
python household/node/household_microgrid.py
```

输出：`household/node/output/household_microgrid.png`。

### 4) 拓扑与韧性扩展实验（`Topology2.0/`）

扩展原论文的 WS 拓扑实验，涵盖三大子模块。

#### 4a) WS 拓扑效应（`ws_topology_effects/`）

研究 Watts-Strogatz 网络的重连概率 $q$ 与平均度 $K$ 如何影响电网稳定性。

| 脚本 | 说明 |
|------|------|
| `ws_config.py` | 全局参数配置（50 节点，K∈{4,6,8,10,12}，q∈[0,1]，3 种发电/负荷配比） |
| `ws_compute.py` | 核心计算：κ_c 二分搜索、DC 潮流、Lorenz 曲线与 Gini 系数 |
| `ws_plots.py` | 绘图：κ_c(q) 曲线、κ_c(K,q) 热力图、Lorenz 曲线、Gini 系数 |
| `ws_cascade_compute.py` | 基于摇摆方程的级联故障二分搜索（α_critical） |
| `ws_cascade_plots.py` | 级联热力图与线图（相对负载 ρ） |
| `ws_plots_combined.py` | 多配比对比图（balanced / gen_heavy / load_heavy 叠加） |
| `run_all.py` | 一键运行全部计算与绘图 |
| `run_cascade_all.py` | 仅运行级联二分计算 |

三种发电/负荷配比：
- **balanced**：24 发电 + 25 负荷
- **gen_heavy**：37 发电 + 12 负荷（75% 发电主导）
- **load_heavy**：12 发电 + 37 负荷（75% 负荷主导）

```bash
python Topology2.0/ws_topology_effects/run_all.py
```

#### 4b) 三元单纯形拓扑扫描（`experiment_ratio_simplex/`）

在 4 种拓扑族上扫描发电 / 负荷 / 被动节点配比空间（三元单纯形），比较不同网络结构对 $\kappa_c$ 的影响。

**4 种拓扑族：**
- **WS**（Watts-Strogatz）：小世界网络
- **RGG**（Random Geometric Graph）：欧氏距离随机图
- **SBM**（Stochastic Block Model）：社区结构网络
- **CP**（Core-Periphery）：核心-外围网络

| 脚本 | 说明 |
|------|------|
| `compute_ratio_simplex_kappa.py` | 多进程并行扫描三元配比空间，缓存至 CSV |
| `plot_ratio_simplex_kappa.py` | 2×2 三元热力图（4 种拓扑族的 κ_c 分布） |
| `plot_topology_structures.py` | 2×2 网络结构可视化（WS/RGG/SBM/CP 代表性图） |

```bash
python Topology2.0/experiment_ratio_simplex/compute_ratio_simplex_kappa.py
python Topology2.0/experiment_ratio_simplex/plot_ratio_simplex_kappa.py
python Topology2.0/experiment_ratio_simplex/plot_topology_structures.py
```

输出：`Topology2.0/experiment_ratio_simplex/figures/`

#### 4c) 级联韧性与恢复分析（`cascade_resilience/`）

研究级联故障后的**恢复时间**与**服务水平**动态，支持随机边故障、修复调度和 PCC 连通性追踪。

| 脚本 | 说明 |
|------|------|
| `cascade_utils.py` | 核心引擎：摇摆方程积分、故障检测、修复调度、服务水平 $S_{\text{PCC}}(t)$ 计算 |
| `experiment_recovery_time_ws.py` | CLI 驱动的 3D 参数扫描（K, q, α），支持 NPZ 缓存 |
| `plots_recovery_panels.py` | 4×3 出版级面板图（网络结构 / 未恢复比例 / $S_{\min}$ / 时间序列） |

关键指标：
- $T_{\text{rec}}$：恢复时间（$S_{\text{PCC}}$ 回到阈值以上）
- $A_{\text{res}}$：服务损失面积（$\int (1 - S_{\text{PCC}}) \, dt$）
- $S_{\min}$：最低服务水平

```bash
python Topology2.0/cascade_resilience/experiment_recovery_time_ws.py --K_list 4,6,8 --q_points 21 --alpha_points 21 --R 10
python Topology2.0/cascade_resilience/plots_recovery_panels.py
```

输出：`Topology2.0/cascade_resilience/output/`
缓存：`Topology2.0/cascade_resilience/cache/`（**请勿删除**）

### 5) Q1 作业（`Q1Topology/`）

- `ws_kappa_c/` — 临界耦合 $\kappa_c$ vs 重连概率 $q$ 与平均度 $k$
- `ws_percolation/` — 网络渗流分析

## 已有结果

仓库中已包含部分生成结果：

| 模块 | 输出位置 |
|------|----------|
| 论文复现 | `reproduction/output/` (Figure 1–6, S2) |
| 两节点分析 | `2nodes/output/` |
| 家庭分析 | `household/*/output/` |
| WS 拓扑效应 | `Topology2.0/ws_topology_effects/output/` |
| 三元单纯形 | `Topology2.0/experiment_ratio_simplex/figures/` |
| 级联韧性 | `Topology2.0/cascade_resilience/output/` |

## 说明

1. `reproduction/` 脚本计算量较大，首次运行可能较慢；已有缓存时会自动跳过重复计算。
2. 多数脚本会自动创建 `output/` 与 `cache/` 目录。
3. 文档说明参考 `docs/markdown/` 与 `2nodes/两节点稳定性分析使用指南.md`。
4. `reference_code/GridResilience/` 包含教师原始 Julia 实现，供对照参考。
