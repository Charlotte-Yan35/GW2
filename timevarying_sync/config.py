"""
config.py — 集中配置：时变同步实验 (报告 §4.3.1)

所有可调参数统一在此处管理，compute / plot 脚本均从此导入。
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = MODULE_DIR / "cache"
OUTPUT_DIR = MODULE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ── 数据路径 (对齐 reference_code/scripts/powerreader.py) ──
# reference_code 原始: "powerdata/data/"  →  本项目: data/Small LCL Data/
# reference_code 原始: "powerdata/PV/HourlyData/CustomerEndpoints.csv"
#   →  本项目: data/PV Data/.../EXPORT HourlyData - Customer Endpoints.csv
LCL_DATA_DIR = PROJECT_ROOT / "data" / "Small LCL Data"
LCL_PARQUET = PROJECT_ROOT / "data" / "Small_LCL_Data.parquet"
PV_DATA_FILE = (
    PROJECT_ROOT / "data" / "PV Data"
    / "2014-11-28 Cleansed and Processed"
    / "EXPORT HourlyData"
    / "EXPORT HourlyData - Customer Endpoints.csv"
)

# ── 预处理聚合数据路径 (household/ 模块输出) ──
PROCESSED_LCL_CSV = PROJECT_ROOT / "household" / "consumer" / "output" / "monthly_hourly_usage.csv"
PROCESSED_PV_CSV = PROJECT_ROOT / "household" / "generation" / "output" / "generation_range_by_season.csv"

# ── 季节 → PV 季节名映射 ──
SEASON_TO_PV_SEASON = {"summer": "Summer", "winter": "Winter"}

# ═══════════════════════════════════════════════════════════════
# 季节 → 月份 (对齐 reference_code: month=1 冬, month=7 夏)
# ═══════════════════════════════════════════════════════════════
SEASON_TO_MONTH = {"summer": 7, "winter": 1}

# ═══════════════════════════════════════════════════════════════
# 网络参数 (对齐 reference_code MicroGrid: n=50, PCC = node n-1)
# ═══════════════════════════════════════════════════════════════
N = 50
N_HOUSEHOLDS = N - 1          # 49 个住户节点
PCC_NODE = N - 1              # PCC 为最后一个节点
K_BAR = 4                     # WS 平均度
Q_REWIRE = 0.1                # WS 重连概率
PV_PENETRATION = 49           # 有 PV 的节点数 (对齐 reference_code penetration=49)

# ═══════════════════════════════════════════════════════════════
# Swing 方程参数 (对齐 reference_code: I=D=1, κ=5, synctol=3)
# ═══════════════════════════════════════════════════════════════
I_INERTIA = 1.0
D_DAMP = 1.0
KAPPA = 5.0
SYNCTOL = 3.0                 # 去同步阈值 ||ω||_2

# ═══════════════════════════════════════════════════════════════
# 仿真参数
# ═══════════════════════════════════════════════════════════════
T_TOTAL = 86400.0             # 24 h = 86400 秒
DEFAULT_FREQ = "5min"         # 默认数据/监控频率
DEFAULT_REALIZATIONS = 50
DEFAULT_BASE_SEED = 42
SETTLE_TIME = 200.0           # 初态稳定收敛积分时间 (秒)
ODE_MAX_STEP = 30.0           # ODE 求解器最大步长 (秒)

# ═══════════════════════════════════════════════════════════════
# 崩溃判据
# ═══════════════════════════════════════════════════════════════
R_COLLAPSE_THRESHOLD = 0.3    # r(t) 低于此值视为崩溃
OMEGA_NORM_THRESHOLD = SYNCTOL

# ═══════════════════════════════════════════════════════════════
# 高风险时窗 (小时)
# ═══════════════════════════════════════════════════════════════
MORNING_WINDOW = (6, 10)      # 早间高峰: 06:00–10:00
EVENING_WINDOW = (16, 20)     # 傍晚高峰: 16:00–20:00

# ═══════════════════════════════════════════════════════════════
# 早期预警参数
# ═══════════════════════════════════════════════════════════════
SLIDING_WINDOW_SEC = 3600     # 滑动窗口宽度: 1 h
EARLY_WARNING_LEAD_SEC = 7200 # 崩溃前观测起始: 2 h

# ═══════════════════════════════════════════════════════════════
# 绘图参数
# ═══════════════════════════════════════════════════════════════
DPI = 300
FONT_FAMILY = "sans-serif"

# LCL CSV 字段名 (注意: 原始列名有尾部空格)
LCL_COL_ID = "LCLid"
LCL_COL_DATETIME = "DateTime"
LCL_COL_POWER = "KWH/hh (per half hour) "

# PV CSV 字段名
PV_COL_ID = "Substation"
PV_COL_DATETIME = "datetime"
PV_COL_GEN_MAX = "P_GEN_MAX"
PV_COL_GEN_MIN = "P_GEN_MIN"
