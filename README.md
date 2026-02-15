# GW2

基于摇摆方程（Swing Equation）的电网稳定性与韧性研究代码仓库，包含：
1. 论文图表复现（`reproduction/`）
2. 两节点动力学与稳定性分析（`2nodes/`）
3. 家庭负荷/光伏/储能微电网分析（`household/`）

主要对应论文：Smith et al., *Science Advances* 8, eabj6734 (2022)。

## 目录结构

```text
GW2/
├── 2nodes/                 # 两节点模型与可视化
├── reproduction/           # Figure 1/2/3/4/6 复现脚本
├── household/              # 家庭用电、发电、单节点微电网
├── data/                   # 原始数据（LCL / PV / xlsx 等）
├── docs/                   # 文档与复现说明
├── fig3.png fig4.png Fig6.png
└── README.md
```

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
1. `data/Small LCL Data/LCL-June2015v2_*.csv`
2. `data/PV Data/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv`
3. `data/Small_LCL_Data.parquet`（可由脚本生成）

如果缺少 Parquet 文件，可先执行：

```bash
python household/consumer/convert_lcl_to_parquet.py
```

## 运行方式

在仓库根目录执行以下命令。

### 1) 论文图表复现（`reproduction/`）

```bash
python reproduction/figure1.py
python reproduction/figure2.py
python reproduction/figure3.py
python reproduction/figure4.py
python reproduction/figure6.py
```

额外脚本（Figure 1 Panel D 单独计算）：

```bash
python reproduction/run_panel_d.py
```

输出目录：`reproduction/output/`  
缓存目录：`reproduction/cache/`

### 2) 两节点稳定性分析（`2nodes/`）

```bash
python 2nodes/two_node_stability_analysis.py
```

输出目录：`2nodes/output/`，包括分岔图、相图、时间序列、随机扰动、级联故障、参数扫描等。

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

注意：该脚本默认读取 `household/generation/hourly_data.pkl`，请先确认该文件存在。

单节点家庭微电网（含储能）：

```bash
python household/node/household_microgrid.py
```

输出：`household/node/output/household_microgrid.png`。

## 已有结果

仓库中已包含部分生成结果，例如：
1. `reproduction/output/figure1_AB.png`
2. `reproduction/output/figure2.png`
3. `reproduction/output/figure3.png`
4. `reproduction/output/figure4_DEFG.png`
5. `2nodes/output/*.png`
6. `household/*/output/*.png`

## 说明

1. `reproduction` 脚本计算量较大，首次运行可能较慢。
2. 多数脚本会自动创建 `output/` 与 `cache/` 目录。
3. 文档说明可参考 `docs/markdown/` 与 `2nodes/两节点稳定性分析使用指南.md`。
