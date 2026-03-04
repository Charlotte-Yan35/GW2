# 代码模块汇总报告

> 本文档汇总了项目中四大实验模块的代码信息，方便论文撰写时查阅。

---

## 目录

1. [Topology2.0/ws_topology_effects/ — WS 拓扑对电网稳定性与级联韧性的影响](#1-ws_topology_effects)
2. [Topology2.0/experiment_ratio_simplex/ — 四种拓扑族的三元单纯形配比扫描](#2-experiment_ratio_simplex)
3. [ratio_scan/ — 节点配比扫描与稳定性分析](#3-ratio_scan)
4. [diurnal_stability/ — 日变电网稳定性分析](#4-diurnal_stability)

---

## 1. ws_topology_effects

**研究问题**：Watts-Strogatz 小世界拓扑（重连概率 q、平均度 K）如何影响电网的同步稳定性和级联韧性？三种节点配比（balanced / gen_heavy / load_heavy）下结果有何差异？

### 1.1 目录结构

```
ws_topology_effects/
├── ws_config.py                # 全局参数配置
├── run_all.py                  # 主入口：完整流水线
├── run_cascade_all.py          # 仅运行级联二分搜索
├── run_cascade_duration.py     # 仅运行 FigS2 持续时间分析
├── ws_stability/
│   ├── compute.py              # 稳定性计算核心（κ_c、Lorenz/Gini、DC 级联）
│   ├── plots.py                # 单配比绘图
│   └── plots_combined.py       # 多配比对比绘图
└── swing_cascade/
    ├── compute.py              # Swing ODE 级联计算（二分搜索、持续时间）
    └── plots.py                # 级联绘图（热力图、折线图、FigS2）
```

### 1.2 全局参数（`ws_config.py`）

| 参数 | 值 | 含义 |
|------|----|------|
| `N` | 50 | 网络总节点数（1 PCC + 49 家庭） |
| `K_list` | [4, 6, 8, 10, 12] | WS 图平均度 |
| `q_list` | linspace(0, 1, 21) | 重连概率，21 个点 |
| `kappa_grid` | linspace(0, 20, 81) | 耦合强度网格 |
| `realizations` | 20 | 蒙特卡洛实现次数 |
| `K_ref` | 8 | Lorenz/Gini/级联参考度 |
| `alpha` | 0.2 | DC 级联过载容差 |
| `KAPPA_CASCADE` | 1.0 | Swing ODE 耦合强度 |
| `SYNCTOL` | 3.0 | 失同步判断容限 |
| `BISECT_TOL` | 5e-4 | 二分法收敛精度 |

**三种节点配比**（`RATIO_CONFIGS`）：

| 名称 | 发电节点 | 负载节点 | 比例 |
|------|---------|---------|------|
| `balanced` | 24 | 25 | ≈1:1 |
| `gen_heavy` | 37 | 12 | ≈3:1 |
| `load_heavy` | 12 | 37 | ≈1:3 |

### 1.3 数学模型

**摇摆方程（Swing Equation）**：

$$I\dot{\omega}_i = P_i - D\omega_i - \kappa\sum_j A_{ij}\sin(\theta_i - \theta_j), \quad \dot{\theta}_i = \omega_i$$

**稳态条件**：$P_i - \kappa\sum_j A_{ij}\sin(\theta_i - \theta_j) \approx 0$

**DC 潮流**：$L\theta = P, \quad F_{ij} = \theta_i - \theta_j$

**Gini 系数**：$G = \frac{2\sum_i i \cdot s_i}{n \sum_i s_i} - \frac{n+1}{n}$（$s$ 为升序 $|F_e|$）

**边功率流**：$F_e = \kappa \sin(E^T \theta)$（$E$ 为关联矩阵）

### 1.4 计算模块

#### ws_stability/compute.py — 稳定性计算

| 函数 | 功能 |
|------|------|
| `generate_ws_network(N, K, q, seed)` | 生成连通 WS 小世界图 |
| `build_power_vector(ratio_name, seed)` | 随机分配发电/负载角色，功率平衡约束 |
| `_find_kappa_c_lower(A, P, n)` | 自下而上二分搜索最小同步耦合 κ_c^low |
| `_find_kappa_c_upper(A, P, n, ...)` | 自上而下搜索过耦合失稳点 κ_c^high |
| `compute_kappa_c_map(ratio_name)` | 多进程并行全网格 (K×q) κ_c 计算 |
| `compute_lorenz_and_gini(ratio_name)` | DC 潮流 → Lorenz 曲线 + Gini 系数 |
| `compute_cascade_size(ratio_name)` | DC 级联模型：边容量 $C_e = (1+\alpha)|F_e|_0$，迭代移除过载边 |

#### swing_cascade/compute.py — Swing ODE 级联

| 函数 | 功能 |
|------|------|
| `swingfracture(E_full, active_edges, psi, ...)` | 递归级联：RK45 积分 → 过载检测 → 分量递归 |
| `_bisect_critical_alpha(ensemble)` | 二分搜索临界 α_c（50% 边存活阈值） |
| `swingfracture_timed_parameterized(...)` | 带时间记录的级联（用于 FigS2 持续时间） |
| `compute_and_cache_bisection(ratio)` | K×q 全网格临界 α 计算 |
| `compute_s2_panel(ratio, config, ...)` | FigS2 面板：单变量扫描 + per-value 缓存 |

**级联终止条件**（三重判据）：
1. 失同步：$\|\omega\| > \text{SYNCTOL}$
2. 过载：$|F_e|/F_{\max} > \alpha$
3. 收敛：稳态残差 < 1e-6

### 1.5 输出图表

#### ws_stability 图表

| 图表 | 输出文件 | 内容 |
|------|---------|------|
| κ_c 折线图 | `kappa_c_map_{ratio}.png` | κ_c(q) 各 K 一条曲线，含 ±σ 带 |
| κ_c 热力图 | `kappa_c_heatmap_{ratio}.png` | K×q 平滑等高线 |
| Lorenz 曲线 | `lorenz_{ratio}.png` | q=0, q*, q=1 三条曲线 |
| Gini 系数 | `gini_vs_q_{ratio}.png` | Gini vs q |
| DC 级联尺寸 | `cascade_size_vs_q_{ratio}.png` | S(q) 含误差带 |
| 多配比 κ_c | `combined_kappa_c_map.png` | 3 配比 × 4 K = 12 条曲线叠加 |
| 多配比热力图 | `combined_kappa_c_heatmap.png` | 1×3 子图，共享色条 |

#### swing_cascade 图表

| 图表 | 输出文件 | 内容 |
|------|---------|------|
| 临界 α 热力图 | `cascade_bisection_heatmap_{ratio}.png` | ρ(K,q) 热力图，标注 MAX/MIN |
| 临界 α 折线 | `cascade_bisection_lines_{ratio}.png` | ρ(q) 各 K 一条 |
| 对比热力图 | `combined_cascade_bisection_heatmap.png` | 1×3 三配比对比 |
| 对比折线 | `combined_cascade_bisection_lines.png` | 15 条曲线叠加 |
| 持续时间 | `cascade_duration_{ratio}.png` | T vs α |
| 存活比例 | `cascade_duration_survival_{ratio}.png` | 存活边比例 vs α |
| FigS2 | `figS2_duration.png` | 2×3 面板（q 扫/K 扫 × 3 配比） |

### 1.6 缓存文件

| 缓存路径 | 内容 |
|----------|------|
| `ws_stability/cache/{ratio}.pkl` | κ_c map、Lorenz、Gini、DC 级联 |
| `swing_cascade/cache/cascade_bisection_swing_{ratio}.pkl` | 临界 α_c 和相对负载 ρ 矩阵 |
| `swing_cascade/cache/cascade_duration_{ratio}.pkl` | 持续时间和存活比例 |
| `swing_cascade/cache/figS2/s2_*.npz` | FigS2 单条曲线缓存 |

---



## 3. ratio_scan

**研究问题**：WS 网络中节点配比 (n_g, n_c, n_p) 如何影响级联韧性（α*）和同步稳定性（κ_c）？Passive 节点对过载容忍度的缓冲效应如何量化？

### 3.1 目录结构

```
ratio_scan/
├── shared_utils.py             # 公共工具库（自包含）
├── run_ratio_scan.py           # 级联韧性扫描（α*）
├── run_stability_scan.py       # 同步稳定性扫描（κ_c）
├── plot_simplex_panels.py      # α* 三元热力图 + 失效模式图
├── plot_stability_panels.py    # κ_c 三元热力图
├── plot_stability_extra.py     # κ_c 补充分析（差异图/截面/小提琴/排名）
├── plot_passive_impact.py      # Passive 节点影响专题
├── generate_summary.py         # 文本分析报告生成
├── cache/                      # 计算缓存
├── results/                    # 输出图片与报告
└── output/                     # 额外输出
```

### 3.2 公共参数（`shared_utils.py`）

| 参数 | 值 | 含义 |
|------|----|------|
| `N` | 50 | 节点总数 |
| `N_HOUSEHOLDS` | 49 | 家庭节点数 |
| `PCC_NODE` | 0 | PCC 节点 |
| `PMAX` | 1.0 | 归一化功率 |
| `I_INERTIA` | 1.0 | 惯量 |
| `D_DAMP` | 1.0 | 阻尼 |

### 3.3 级联韧性扫描（`run_ratio_scan.py`）

**实验参数**：

| 参数 | 值 |
|------|----|
| `K_VALUES` | [4, 8] |
| `Q_VALUES` | [0.0, 0.15, 1.0] |
| `ALPHA_PAS_LIST` | [1.0, 0.7, 0.4, 0.1] |
| `REALIZATIONS` | 30 |
| `ROLE_SEEDS` | 10 |
| `DYN_SEEDS` | 10 |
| `RATIO_STEP` | 5（步长 0.2）|
| `ALPHA_TOL` | 1e-2 |

**核心机制——Per-edge 过载容量**：

$$C_e = \alpha_e \cdot F_{\max}, \quad \alpha_e = \min(\alpha_i, \alpha_j)$$

- Generator/Consumer/PCC 边：$\alpha_e = \alpha_{\text{active}}$
- Passive 边：$\alpha_e = \alpha_{\text{pas}}$（取 1.0/0.7/0.4/0.1 四档）
- 跳闸条件：$|F_e(t)| > C_e$

**级联算法** `swingfracture_typed()`：
1. 扰动稳态 → 移除 trigger 边（最大负载边）
2. RK45 积分 Swing 方程
3. 过载检测 → 移除过载边 → 分解连通分量 → 递归

**二分搜索临界 α***：Julia 风格步长递减，判据 $\overline{\text{存活比例}} \leq 0.5$

**失效模式分类**：在 α*−0.1 处统计 sync-dominant vs flow-dominant

**输出指标**：$\Delta\alpha^* = \alpha^*(\alpha_{\text{pas}}=1.0) - \alpha^*(\alpha_{\text{pas}}=0.1)$（Passive 缓冲效应）

### 3.4 同步稳定性扫描（`run_stability_scan.py`）

**实验参数**：

| 参数 | 值 |
|------|----|
| `K_VALUES` | [4, 8] |
| `Q_VALUES` | [0.0, 0.15, 0.5, 1.0] |
| `RATIO_STEP` | 10（步长 0.1，更细）|
| `REALIZATIONS` | 20 |
| `ROLE_SEEDS` | 5 |
| `KAPPA_START` | 5.5 |
| `KAPPA_TOL` | 1e-3 |

**κ_c 搜索算法**（Warm-start 递降）：
1. 从 κ=5.5 积分至稳态
2. 逐步降低 κ（warm-start 沿用上一稳态解）
3. 发散 → 步长减半回退
4. 步长 < 1e-3 时终止

### 3.5 输出图表

| 图表 | 脚本 | 输出文件 | 内容 |
|------|------|---------|------|
| α* 热力图 | `plot_simplex_panels.py` | `fig_alpha_star_panels.png` | 2×3 面板（K={4,8} × q={0,0.15,1}） |
| 失效模式 | `plot_simplex_panels.py` | `fig_failure_mode_panels.png` | 2×3 面板，红=sync-dominant，蓝=overload |
| κ_c 热力图 | `plot_stability_panels.py` | `fig_kappa_c_panels.png` | 2×3 面板（K={4,8} × q={0,0.15,0.5,1}） |
| κ_c 补充 | `plot_stability_extra.py` | `fig_kappa_c_extra.png` | 2×2 面板：差异图/截面/小提琴/排名 |
| Passive 影响 | `plot_passive_impact.py` | `fig_passive_impact.png` | 1×4 面板：热力图 + 3 个截面（n_p=0/20/35） |
| 文本报告 | `generate_summary.py` | `summary.txt` | 统计摘要（均值/最优配比/Spearman 相关） |

**`plot_stability_extra.py` 四子面板详情**：

| 面板 | 内容 |
|------|------|
| A (1×3) | 拓扑敏感度：Δκ_c = κ_c(K=8) − κ_c(K=4)，RdBu_r 对称色图 |
| B | 截面：固定 n_passive≈30%，κ_c vs n_consumers |
| C | 小提琴图：κ_c 分布按 (K,q) 分组 |
| D | 排名条：Top-8 最稳定 / Bottom-8 最不稳定 |

### 3.6 缓存文件

| 文件 | 来源 | 内容 |
|------|------|------|
| `cache/raw_results.csv` | `run_ratio_scan.py` | 级联扫描原始数据 |
| `results/k{K}_q{q}.pkl` | `run_ratio_scan.py` | 聚合结果（含 Δα*）|
| `cache/stability_results.csv` | `run_stability_scan.py` | κ_c 原始数据 |
| `cache/stability_agg.csv` | `run_stability_scan.py` | κ_c 聚合均值/标准差 |

### 3.7 数据流

```
run_ratio_scan.py → cache/raw_results.csv
    → plot_simplex_panels.py  → fig_alpha_star_panels.png, fig_failure_mode_panels.png
    → generate_summary.py     → summary.txt

run_stability_scan.py → cache/stability_results.csv → cache/stability_agg.csv
    → plot_stability_panels.py → fig_kappa_c_panels.png
    → plot_stability_extra.py  → fig_kappa_c_extra.png
    → plot_passive_impact.py   → fig_passive_impact.png
    → generate_summary.py      (含 Spearman 相关)

reproduction/cache/v4_panel_c_*.npz → plot_passive_impact.py (Panel A)
```

---

## 4. diurnal_stability

**研究问题**：基于真实数据驱动的时变家庭功率注入，一天中哪些时刻电网更难同步？夏季与冬季的 24 小时稳定性轮廓有何差异？

### 4.1 目录结构

```
diurnal_stability/
├── config.py                   # 全局参数配置
├── shared_diurnal_utils.py     # 数据加载与功率注入构建
├── compute_kappa_diurnal.py    # 核心计算（逐小时 κ_c 二分搜索）
├── plot_kappa_diurnal.py       # 可视化（双 y 轴对比图）
├── cache/                      # 计算与数据缓存
└── results/                    # 输出图片与聚合数据
```

### 4.2 参数配置（`config.py`）

| 类别 | 参数 | 值 | 含义 |
|------|------|----|------|
| 网络 | N | 50 | 节点总数 |
| 网络 | K_WS | 4 | WS 平均度 |
| 网络 | Q_WS | 0.1 | WS 重连概率 |
| Monte-Carlo | N_REALIZATIONS | 10 | 网络拓扑实现数 |
| Monte-Carlo | N_PROFILE_SAMPLES | 5 | 家庭用电采样数 |
| Monte-Carlo | PENETRATION | 49 | PV 渗透率（全户有光伏） |
| 二分搜索 | KAPPA_MIN / MAX | 0.005 / 50.0 | κ 搜索范围 |
| 二分搜索 | KAPPA_TOL | 1e-3 | 收敛精度 |
| 二分搜索 | N_IC_TRIES | 3 | 随机初始条件尝试数 |
| 季节 | summer | [6, 7, 8] | 夏季月份 |
| 季节 | winter | [12, 1, 2] | 冬季月份 |

**计算规模**：2 季 × 24 h × 10 net × 5 profile = **2400 个任务**

### 4.3 数据来源

| 数据 | 来源 | 说明 |
|------|------|------|
| 家庭用电 | `data/Small_LCL_Data.parquet` | LCL 大数据，>5000 户，半小时粒度 |
| PV 发电 | `household/generation/output/generation_range_by_season.csv` | 24h PV 均值曲线 |

### 4.4 功率注入向量构建（`shared_diurnal_utils.py`）

```
前 penetration 户（有 PV）：P[k+1] = pv[hour] − demand[k]   （净注入）
其余户（无 PV）：            P[k+1] = −demand[k]             （纯消费）
PCC 平衡节点：               P[0]   = −sum(P[1:])            （功率守恒）
```

**关键参考值**：夏季 h=13 PV 峰值 ≈ 1.347 kW，冬季 ≈ 0.662 kW

### 4.5 κ_c 搜索算法（`compute_kappa_diurnal.py`）

两阶段二分搜索：
1. **Phase 1 (Bracket)**：κ 从 KAPPA_MIN=0.005 开始翻倍，找到首个稳定点 κ_ok
2. **Phase 2 (Bisection)**：在 [κ_fail, κ_ok] 间二分，warm-start + 多随机初始条件

**收敛判定**：稳态残差 $\|\text{residual}\|_2 < 10^{-5}$

**支持断点续算**：已完成任务记录在 CSV 中，重启时跳过已计算部分

### 4.6 可视化（`plot_kappa_diurnal.py`）

**图表设计**（双 y 轴）：
- **左轴**：夏冬两季 κ_c(h) 均值曲线 ± 标准差阴影
  - 标注各季峰值（▽）和最小值（△）时刻
- **右轴**：PV 发电曲线（虚线）+ 家庭用电曲线（点线）
- 夏季色系：`#c24c51`（红），冬季色系：`#4c70b0`（蓝）

### 4.7 输入/输出文件

| 文件 | 类型 | 内容 |
|------|------|------|
| `cache/household_demand_summer.parquet` | 缓存 | 夏季各户各小时平均用电 |
| `cache/household_demand_winter.parquet` | 缓存 | 冬季各户各小时平均用电 |
| `cache/kappa_diurnal_raw.csv` | 缓存 | 原始 Monte-Carlo 结果 |
| `results/kappa_diurnal.csv` | 输出 | 聚合统计（mean/std/n_valid） |
| `results/kappa_diurnal.png` | 输出 | 最终图表 |

### 4.8 数据流

```
data/Small_LCL_Data.parquet
    → shared_diurnal_utils.py → cache/household_demand_{season}.parquet

generation_range_by_season.csv
    → shared_diurnal_utils.py → PV profile

                     ↓ build_injection_vector()

compute_kappa_diurnal.py（2400 任务，多进程并行）
    WS 图 + P(h) → Swing 积分 → 二分搜索 κ_c(h)
    → cache/kappa_diurnal_raw.csv → results/kappa_diurnal.csv

plot_kappa_diurnal.py
    → results/kappa_diurnal.png
```

### 4.9 物理解读

- **κ_c(h) 高峰**：早晚用电高峰（h≈18-20），需求大、PV 低，需更强耦合维持同步
- **κ_c(h) 低谷**：深夜（h≈3-4），需求低、功率不平衡小
- **夏冬对比**：夏季白天 PV 峰值可降低 κ_c 需求，冬季 PV 不足加剧不平衡

---

## 附录：共享数学模型

### 摇摆方程（Swing Equation）

所有模块的核心动力学方程：

$$I\dot{\omega}_i = P_i - D\omega_i - \kappa\sum_j A_{ij}\sin(\theta_i - \theta_j)$$
$$\dot{\theta}_i = \omega_i$$

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $I$ | 惯量 | 1.0 |
| $D$ | 阻尼系数 | 1.0 |
| $P_i$ | 节点功率注入 | 发电 > 0，消费 < 0 |
| $\kappa$ | 耦合强度 | 扫描范围 0–50 |
| $A_{ij}$ | 邻接矩阵元素 | 0 或 1 |
| $\theta_i$ | 节点相角 | — |
| $\omega_i$ | 节点角频率 | — |

### 网络模型

所有模块使用 **Watts-Strogatz 小世界图**，参数：
- $N$ = 50（1 PCC + 49 家庭）
- $K$ = 平均度（扫描 4–12）
- $q$ = 重连概率（扫描 0–1）

### 数值方法

- **ODE 积分**：`scipy.integrate.solve_ivp`，RK45 方法，rtol=atol=1e-8
- **稳态求解**：积分至 t_max 后检查残差
- **κ_c 搜索**：二分法（bracket + bisection），warm-start 加速
- **并行化**：`multiprocessing.Pool(cpu_count())`

### 关键指标汇总

| 指标 | 符号 | 定义 | 模块 |
|------|------|------|------|
| 临界耦合强度 | $\kappa_c$ | 系统恰好同步所需最小耦合 | 全部 |
| 临界过载容差 | $\alpha^*$ | 50% 边存活的过载阈值 | ws_topology_effects, ratio_scan |
| 相对负载 | $\bar{\rho}$ | $\alpha^*$ 的等价量 | ws_topology_effects |
| 级联尺寸 | $S$ | 移除边占总边比例 | ws_topology_effects |
| Gini 系数 | $G$ | 边功率流不均匀度 | ws_topology_effects |
| Passive 缓冲效应 | $\Delta\alpha^*$ | $\alpha^*(\alpha_p=1) - \alpha^*(\alpha_p=0.1)$ | ratio_scan |
| 日变临界耦合 | $\kappa_c(h)$ | 逐小时准静态 κ_c | diurnal_stability |
