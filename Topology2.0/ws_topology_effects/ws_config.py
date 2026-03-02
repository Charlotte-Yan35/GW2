"""
ws_config.py — Global parameters for WS topology effects study.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ── Network topology ──────────────────────────────────────────────
N = 50                  # total number of nodes (PCC + households)
PCC_NODE = 0            # index of the PCC node
HOUSEHOLD_NODES = list(range(1, N))   # nodes 1 ~ 49

# ── Watts-Strogatz sweep parameters ──────────────────────────────
K_list = [4, 6, 8, 10, 12]           # even degree parameter
q_list = np.linspace(0, 1, 21)       # rewiring probability
gamma = 1.0                          # damping coefficient
kappa_grid = np.linspace(0, 20, 81)  # coupling strength grid
realizations = 20                    # Monte-Carlo realizations per (K, q)
K_ref = 8                            # reference degree for Lorenz / cascade plots
alpha = 0.2                          # power tolerance for cascade model

# ── Swing cascade parameters ─────────────────────────────────
KAPPA_CASCADE = 5.0       # 级联计算用的耦合强度 (同论文)
I_INERTIA = 1.0           # 惯量
D_DAMP = 1.0              # 阻尼
SYNCTOL = 3.0             # 失同步容限
BISECT_TOL = 5e-4         # 二分法精度

# ── Ratio configurations (PCC does not participate in ratio) ─────
# n_gen + n_load = 49 (household nodes only); Pmax normalised to 1.0
RATIO_CONFIGS = {
    "balanced": {
        "n_gen": 24,        # ~half generators
        "n_load": 25,       # ~half loads
        "Pmax": 1.0,
    },
    "gen_heavy": {
        "n_gen": 37,        # ~75 % generators
        "n_load": 12,       # ~25 % loads
        "Pmax": 1.0,
    },
    "load_heavy": {
        "n_gen": 12,        # ~25 % generators
        "n_load": 37,       # ~75 % loads
        "Pmax": 1.0,
    },
}

# ── Cascade duration exploration ─────────────────────────────────
# 6 个兴趣点 (K, q)，用于持续时间分析
INTEREST_POINTS = [
    {"label": "P1", "K": 4,  "q": 0.00, "note": "G=C MIN ρ̄"},
    {"label": "P2", "K": 6,  "q": 0.00, "note": "G>C,G<C MAX ρ̄"},
    {"label": "P3", "K": 6,  "q": 0.35, "note": "G=C MAX ρ̄"},
    {"label": "P4", "K": 4,  "q": 0.90, "note": "G>C MIN ρ̄"},
    {"label": "P5", "K": 8,  "q": 0.50, "note": "中间参考点"},
    {"label": "P6", "K": 4,  "q": 0.10, "note": "G<C MIN ρ̄"},
]
ALPHA_DURATION = [0.5, 1.0, 1.5, 2.0, 2.5]  # alpha 扫描范围
DURATION_REALIZATIONS = 20                     # 每个 (点, alpha) 的 ensemble 数

# ── Figure S2 风格：Duration sweep 配置 ─────────────────────────────
@dataclass(frozen=True)
class DurationSweepConfig:
    """单个 sweep 面板的参数配置（q 扫描 或 K 扫描）。"""
    panel: str               # "A"=q sweep, "B"=K sweep
    variable_name: str       # "q" 或 "K"
    variable_label: str      # LaTeX 图例标题
    values: tuple
    K: int = 8
    q: float = 0.1
    gamma: float = 1.0
    kappa: float = 5.0
    y_max: float = 50.0
    color: str = "#c24c51"

DURATION_SWEEP_PANELS = {
    "A": DurationSweepConfig(
        panel="A", variable_name="q", variable_label="$q$",
        values=tuple(np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)),
        K=4, gamma=1.0, kappa=1.0,
        y_max=40.0, color="#97015E",
    ),
    "B": DurationSweepConfig(
        panel="B", variable_name="K", variable_label="$K$",
        values=(4, 6, 8, 10, 12),
        q=0.1, gamma=1.0, kappa=1.0,
        y_max=50.0, color="#c24c51",
    ),
}

S2_ALPHA_MIN = 0.5
S2_ALPHA_MAX = 2.5
S2_ALPHA_RES = 50
S2_ENSEMBLE_SIZE = 30
S2_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
