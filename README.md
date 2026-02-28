# GW2

基于摇摆方程（Swing Equation）的电网稳定性与韧性研究代码仓库。

主要对应论文：Smith et al., *Science Advances* 8, eabj6734 (2022) —— *The effect of renewable energy incorporation on power grid stability and resilience*。

## 目录结构

```text
GW2/
├── reproduction/           # 论文图表复现（Figure 1–6, S2）
├── 2nodes/                 # 两节点模型：分岔图、相图、稳定性分析
├── household/              # 家庭用电 / 光伏发电 / 单节点微电网
├── Topology2.0/            # 拓扑与韧性扩展实验（WS 拓扑效应、级联恢复）
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

扩展原论文的实验，包括：
- `ws_topology_effects/` — Watts-Strogatz 拓扑指标（路径长度、聚类系数、代数连通度）与基尼系数分析
- `experiment_ratio_simplex/` — 不同节点配比下的 $\kappa_c$ 热力图
- `cascade_resilience/` — 级联故障恢复时间分析

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
| 拓扑扩展 | `Topology2.0/*/output/` |

## 说明

1. `reproduction/` 脚本计算量较大，首次运行可能较慢；已有缓存时会自动跳过重复计算。
2. 多数脚本会自动创建 `output/` 与 `cache/` 目录。
3. 文档说明参考 `docs/markdown/` 与 `2nodes/两节点稳定性分析使用指南.md`。
4. `reference_code/GridResilience/` 包含教师原始 Julia 实现，供对照参考。
