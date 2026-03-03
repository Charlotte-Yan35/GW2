"""
config.py — 日变稳定性分析参数配置。

研究时变家庭注入功率 P_k(h) 下的逐小时准静态临界耦合 κ_c(h)，
比较夏季与冬季 24 小时稳定性曲线。
"""

from pathlib import Path

# ── 项目路径 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = MODULE_DIR / "cache"
RESULTS_DIR = MODULE_DIR / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── 数据路径 ──────────────────────────────────────────────────────
PARQUET_PATH = PROJECT_ROOT / "data" / "Small_LCL_Data.parquet"
PV_CSV_PATH = PROJECT_ROOT / "household" / "generation" / "output" / "generation_range_by_season.csv"

# ── 网络拓扑 ──────────────────────────────────────────────────────
N = 50                  # 节点总数 (PCC + households)
PCC_NODE = 0            # PCC 节点索引
N_HOUSEHOLDS = 49       # 家庭节点数
K_WS = 4               # WS 图每节点邻居数 (同 reference_code)
Q_WS = 0.1             # WS 图重连概率
SEED = 42              # 基础随机种子

# ── Monte-Carlo 参数 ──────────────────────────────────────────────
N_REALIZATIONS = 10     # 网络拓扑随机实现数
N_PROFILE_SAMPLES = 5   # 每个拓扑的家庭采样数
PENETRATION = 49        # PV 渗透率（几户有光伏）

# ── Swing 方程参数 ────────────────────────────────────────────────
I_INERTIA = 1.0         # 惯量
D_DAMP = 1.0            # 阻尼
PMAX = 1.0              # （仅用于兼容，本模块使用原始 kW 值）

# ── κ_c 二分搜索参数 ─────────────────────────────────────────────
KAPPA_MIN = 0.005       # 搜索下界起始
KAPPA_MAX = 50.0        # 搜索上界（原始 kW 下可能需要更高）
KAPPA_TOL = 1e-3        # 二分搜索精度
MAX_ITER = 40           # 最大迭代次数
N_IC_TRIES = 3          # 每个 κ 的随机初始条件尝试数

# ── 季节定义 ──────────────────────────────────────────────────────
SEASONS = {"summer": [6, 7, 8], "winter": [12, 1, 2]}

# ── 绘图 ─────────────────────────────────────────────────────────
DPI = 300
