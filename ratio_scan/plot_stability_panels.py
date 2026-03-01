"""
plot_stability_panels.py — kappa_c 三元热力图
2x3 面板 (K={4,8} x q={0,0.15,1})，展示 kappa_c_mean。

数据来源: cache/stability_agg.csv
输出: results/fig_kappa_c_panels.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ── 路径 ──────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
RESULTS_DIR = _MODULE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 样式常量 ─────────────────────────────────────────────────────
_FONT = 11
_TITLE = 13
_TICK = 9
_DPI = 300

K_VALUES = [4, 8]
Q_VALUES = [0.0, 0.15, 1.0]


# ====================================================================
# 三元坐标工具
# ====================================================================

H_TRI = np.sqrt(3) / 2


def bary_to_cart(rg, rc, rp):
    """重心坐标 (rg, rc, rp) -> 2D 笛卡尔坐标。"""
    x = rc + 0.5 * rp
    y = H_TRI * rp
    return x, y


def draw_simplex_frame(ax, total, m=None):
    """绘制缩进三角形框架 + 边标签 + 刻度。

    三角形边缘缩进到实际数据范围。
    m: 最小比例 (rg_min = rc_min)，默认 1/total。
    """
    if m is None:
        m = 1.0 / total

    # 三个顶点 (缩进后):
    #   原 Generator 角 (rg=1) -> rg=1-m, rc=m, rp=0
    #   原 Consumer 角  (rc=1) -> rg=m, rc=1-m, rp=0
    #   原 Passive 角   (rp=1) -> rg=m, rc=m, rp=1-2m  (但 np 可=0)
    # 实际: ng>=1, nc>=1, np>=0 → 三角形:
    v_gen = bary_to_cart(1 - m, m, 0)        # 底边左端 (多 gen, 少 con)
    v_con = bary_to_cart(m, 1 - m, 0)        # 底边右端 (少 gen, 多 con)
    v_pas = bary_to_cart(m, m, 1 - 2 * m)    # 顶部 (少 gen, 少 con, 多 pas)

    verts_x = [v_gen[0], v_con[0], v_pas[0], v_gen[0]]
    verts_y = [v_gen[1], v_con[1], v_pas[1], v_gen[1]]
    ax.plot(verts_x, verts_y, 'k-', lw=1.0)

    # 边标签 (与 figure1.py Panel C 一致)
    mx, my = 0.5 * (v_gen[0] + v_pas[0]), 0.5 * (v_gen[1] + v_pas[1])
    ax.text(mx - 0.08, my, r"$\leftarrow$ Generators", fontsize=9,
            ha='center', va='center', rotation=60)

    ax.text(0.5 * (v_gen[0] + v_con[0]), v_gen[1] - 0.07,
            r"Consumers $\rightarrow$", fontsize=9,
            ha='center', va='top')

    mx, my = 0.5 * (v_con[0] + v_pas[0]), 0.5 * (v_con[1] + v_pas[1])
    ax.text(mx + 0.08, my, r"$\leftarrow$ Passive", fontsize=9,
            ha='center', va='center', rotation=-60)

    # 刻度标注 — 基于实际数据范围
    n_min = int(round(m * total))   # 最小节点数 (ng_min = nc_min)
    n_max = total - n_min           # 最大 (另一种 = n_min 时)

    # 底边刻度 (Consumer 从 n_min 到 n_max, np=0)
    tick_vals_bottom = sorted(set([n_min, total // 4, total // 2,
                                   3 * total // 4, n_max]))
    for val in tick_vals_bottom:
        rc = val / total
        rg = 1 - rc
        if rg < m - 1e-9:
            continue
        x_t, y_t = bary_to_cart(rg, rc, 0)
        ax.plot(x_t, y_t, 'k|', ms=4, mew=0.8)
        ax.text(x_t, y_t - 0.03, str(val), fontsize=6,
                ha='center', va='top', color='0.4')

    # 左边刻度 (Generator 从 n_min 到 n_max, nc=n_min)
    tick_vals_left = sorted(set([n_min, total // 4, total // 2,
                                 3 * total // 4, n_max]))
    for val in tick_vals_left:
        rg = val / total
        rc = m
        rp = 1 - rg - rc
        if rp < -1e-9:
            continue
        x_t, y_t = bary_to_cart(rg, rc, rp)
        ax.text(x_t - 0.03, y_t, str(val), fontsize=6,
                ha='right', va='center', color='0.4')

    # 右边刻度 (Passive 从 0 到 total-2*n_min, rg=m)
    np_max = total - 2 * n_min
    tick_vals_right = sorted(set([0, np_max // 4, np_max // 2,
                                  3 * np_max // 4, np_max]))
    for val in tick_vals_right:
        rp = val / total
        rg = m
        rc = 1 - rg - rp
        if rc < m - 1e-9:
            continue
        x_t, y_t = bary_to_cart(rg, rc, rp)
        ax.text(x_t + 0.03, y_t, str(val), fontsize=6,
                ha='left', va='center', color='0.4')

    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.12, H_TRI + 0.06)
    ax.set_aspect('equal')
    ax.axis('off')


# ====================================================================
# 数据加载
# ====================================================================

def load_agg_csv():
    """读取 stability_agg.csv。"""
    csv_path = CACHE_DIR / "stability_agg.csv"
    if not csv_path.exists():
        print(f"[ERROR] 找不到数据文件: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    for col in ("kappa_c_mean", "kappa_c_std", "K", "q"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["kappa_c_mean"])
    return df.reset_index(drop=True)


# ====================================================================
# 绘图
# ====================================================================

def plot_kappa_c_panels(out_path=None):
    """2x3 面板: K={4,8} x q={0,0.15,1} 的 kappa_c 热力图。

    tricontourf 原始渲染，三角形边框缩进到数据覆盖范围。
    """
    if out_path is None:
        out_path = RESULTS_DIR / "fig_kappa_c_panels.png"

    df = load_agg_csv()

    # 自动检测节点总数 (兼容旧 49 和新 50)
    total = int(df["ng"].iloc[0] + df["nc"].iloc[0] + df["np"].iloc[0])
    print(f"检测到节点总数: {total}")

    # 计算比例和坐标
    df["rg"] = df["ng"] / total
    df["rc"] = df["nc"] / total
    df["rp"] = df["np"] / total
    df["x"], df["y"] = bary_to_cart(df["rg"].values, df["rc"].values,
                                     df["rp"].values)

    # 从实际数据计算最小比例 (用于紧贴数据的边框)
    m_data = min(df["rg"].min(), df["rc"].min())
    print(f"数据最小比例 m_data={m_data:.4f} (ng_min=nc_min={int(round(m_data*total))})")

    # 按面板分组
    data_dict = {}
    for K in K_VALUES:
        for q in Q_VALUES:
            mask = (df["K"] == K) & (np.isclose(df["q"], q))
            panel = df[mask].copy()
            if len(panel) > 0:
                data_dict[(K, q)] = panel

    if not data_dict:
        print("[WARN] 无有效 kappa_c 数据，跳过绘图")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row_idx, K in enumerate(K_VALUES):
        for col_idx, q in enumerate(Q_VALUES):
            ax = axes[row_idx, col_idx]
            draw_simplex_frame(ax, total, m=m_data)
            title = f"K={K}, q={q:g}"

            key = (K, q)
            if key not in data_dict or len(data_dict[key]) < 3:
                ax.set_title(title, fontsize=_TITLE, fontweight='bold', pad=8)
                ax.text(0.5, 0.4, "No data", ha='center', fontsize=_FONT,
                        color='red')
                continue

            sub = data_dict[key]
            x = sub["x"].values
            y = sub["y"].values
            z = sub["kappa_c_mean"].values

            try:
                triang = mtri.Triangulation(x, y)
                tcf = ax.tricontourf(triang, z, levels=20, cmap='YlGnBu_r')
                cbar = fig.colorbar(tcf, ax=ax, shrink=0.7, pad=0.02)
                cbar.set_label(r'$\overline{\kappa}_c$', fontsize=_TICK)
                cbar.ax.tick_params(labelsize=_TICK - 1)
            except (RuntimeError, ValueError):
                title += " (sparse)"

            ax.set_title(title, fontsize=_TITLE, fontweight='bold', pad=8)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04,
                        wspace=0.15, hspace=0.12)

    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ====================================================================
# main
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="kappa_c 三元热力图")
    args = parser.parse_args()
    plot_kappa_c_panels()
