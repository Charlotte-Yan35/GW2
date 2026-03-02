# timevarying_sync — 时变 P_k(t) 对同步的影响

报告 §4.3.1: "时变 P_k(t) 对同步的影响（基于 data-driven 的设定）"

## 运行方式

### 第一步: 计算 (compute)

```bash
source .venv/bin/activate

# Summer (典型 7 月)
MPLBACKEND=Agg python timevarying_sync/compute_timevarying_sync.py \
    --season summer --freq 5min --realizations 50 --base-seed 42

# Winter (典型 1 月)
MPLBACKEND=Agg python timevarying_sync/compute_timevarying_sync.py \
    --season winter --freq 5min --realizations 50 --base-seed 42

# 若无真实数据，可使用合成曲线:
MPLBACKEND=Agg python timevarying_sync/compute_timevarying_sync.py \
    --season summer --freq 5min --realizations 50 --synthetic
```

### 第二步: 画图 (plot)

```bash
MPLBACKEND=Agg python timevarying_sync/plot_timevarying_sync.py \
    --summer-cache timevarying_sync/cache/results_summer.npz \
    --winter-cache timevarying_sync/cache/results_winter.npz
```

### 完整参数

```bash
python timevarying_sync/compute_timevarying_sync.py --help
python timevarying_sync/plot_timevarying_sync.py --help
```

## Data-Driven 数据来源

### LCL 负荷数据

- **来源**: `data/Small LCL Data/LCL-June2015v2_*.csv` (168 个 CSV)
- **对应 reference_code**: `powerdata/data/*.csv`
- **字段**: `LCLid`, `DateTime`, `KWH/hh (per half hour) `
- **加载方式**: 对齐 `reference_code/scripts/powerreader.py::make_random_week_profiles()`
  - 随机选文件 → 随机选户 → 按月过滤 → 按 day-of-week 分组 → 取均值
  - 提取 Tuesday (DOW=1) 作为典型日 (对齐 tday_sample = t[48:96] → 第 2 天)
  - 时间轴: 秒 (从午夜 0 点起)

### PV 发电数据

- **来源**: `data/PV Data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv`
- **对应 reference_code**: `powerdata/PV/HourlyData/CustomerEndpoints.csv`
- **字段**: `Substation`, `datetime`, `P_GEN_MAX`, `P_GEN_MIN`
- **加载方式**: 对齐 `reference_code/scripts/powerreader.py::make_random_week_profiles_PV()`
  - P_GEN = (P_GEN_MAX + P_GEN_MIN) / 2

### 季节定义

| 季节 | 月份 | 对应 reference_code |
|------|------|---------------------|
| summer | 7 (July) | `month=7` |
| winter | 1 (January) | `month=1` |

## 节点映射 (对齐 reference_code/scripts/powerclasses.py)

- **N = 50** 个节点
- 节点 0–48: 住户节点 (49 个 House)
- 节点 49 (PCC): 公共耦合点，平衡总功率使 Σ P_k = 0
- **PV penetration = 49**: 节点 0–48 均有 PV 面板
- **净功率**: `P_k(t) = gen_k(t) - demand_k(t)` (对齐 `House.get_house_power()`)
- **归一化**: 使 max_t(Σ|P_k|) = 2.0 以匹配静态仿真的尺度

## 同步序参量 r(t)

$$r(t) = \left| \frac{1}{N} \sum_{k=1}^{N} e^{i\theta_k(t)} \right|$$

- **包含 PCC 节点** (reference_code 中所有节点均参与 swing dynamics)
- r = 1: 完全同步; r → 0: 失同步

## 崩溃判据

- **主判据**: `||ω(t)||_2 > synctol` (synctol = 3.0, 对齐 reference_code swing.jl line 225)
- **辅助判据**: `r(t) < 0.3` (补充)
- 崩溃时刻 `t_collapse`: 首次满足崩溃条件的时刻 (秒); 未崩溃记为 NaN

## 输出文件

### 缓存 (cache/)

| 文件 | 内容 |
|------|------|
| `results_summer.npz` | Summer 仿真结果 |
| `results_winter.npz` | Winter 仿真结果 |

每个 `.npz` 包含:
- `t_grid`: (T,) 时间网格 (秒)
- `r_ensemble`: (R, T) 同步序参量
- `omega_norm_ensemble`: (R, T) ||ω||_2
- `t_collapse`: (R,) 崩溃时刻
- `var_r`: (R, T) 滑动窗口 Var[r]
- `ac1_r`: (R, T) 滑动窗口 AC1[r]
- `metadata_json`: 参数 JSON

### 图表 (outputs/figures/)

| 文件 | 内容 |
|------|------|
| `fig_r_time_summer.png` | Summer r(t) 演化 |
| `fig_r_time_winter.png` | Winter r(t) 演化 |
| `fig_r_time_overlay.png` | Summer vs Winter 叠加 |
| `fig_collapse_hist.png` | 崩溃时刻分布直方图 |
| `fig_evening_significance.png` | 傍晚高风险显著性检验 |
| `fig_early_warning.png` | Var[r] 和 AC1 前兆指标 |

### 统计 (outputs/)

| 文件 | 内容 |
|------|------|
| `stats.json` | Fisher exact test 结果, p-value, odds ratio |

## Swing 方程

复用 `ratio_scan/shared_utils.py::fswing()` (不复制):

$$\frac{d\omega_i}{dt} = \frac{P_i(t) - D\omega_i - \kappa \sum_j A_{ij} \sin(\theta_i - \theta_j)}{I}$$
$$\frac{d\theta_i}{dt} = \omega_i$$

参数: I=1.0, D=1.0, κ=5.0 (可通过 `--kappa` 修改)

## 可复现性

- 所有随机数从 `--base-seed` 通过 `numpy.random.SeedSequence.spawn()` 派生
- 相同 seed + 相同参数 → 完全相同的输出
