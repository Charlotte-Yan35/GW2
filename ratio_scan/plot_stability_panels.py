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
from matplotlib.tri import Triangulation, LinearTriInterpolator

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
    """重心坐标 (rg, rc, rp) -> 2D 笛卡尔坐标。

    与 figure1.py 一致的论文方向:
        Generator (rg=1) -> 左下   (0, 0)
        Consumer  (rc=1) -> 右下   (1, 0)
        Passive   (rp=1) -> 顶部   (0.5, H_TRI)
    """
    x = rc + 0.5 * rp
    y = H_TRI * rp
    return x, y


def draw_simplex_frame(ax):
    """绘制等边三角形框架和顶点标签 (论文方向)。"""
    verts = np.array([
        [0.0, 0.0],               # Generator (左下)
        [1.0, 0.0],               # Consumer (右下)
        [0.5, H_TRI],             # Passive (顶)
    ])
    triangle = plt.Polygon(verts, fill=False, edgecolor='k', linewidth=1.2)
    ax.add_patch(triangle)

    offset = 0.06
    ax.text(verts[0, 0] - offset, verts[0, 1] - offset, "Generator",
            ha='center', va='top', fontsize=_TICK, fontweight='bold')
    ax.text(verts[1, 0] + offset, verts[1, 1] - offset, "Consumer",
            ha='center', va='top', fontsize=_TICK, fontweight='bold')
    ax.text(verts[2, 0], verts[2, 1] + offset, "Passive",
            ha='center', va='bottom', fontsize=_TICK, fontweight='bold')

    # 10% 增量网格线
    for frac in np.arange(0.1, 1.0, 0.1):
        # 等 rg 线 (平行于 Consumer-Passive 边)
        x0, y0 = bary_to_cart(frac, 1 - frac, 0)
        x1, y1 = bary_to_cart(frac, 0, 1 - frac)
        ax.plot([x0, x1], [y0, y1], 'k-', lw=0.3, alpha=0.3)
        # 等 rc 线 (平行于 Generator-Passive 边)
        x0, y0 = bary_to_cart(0, frac, 1 - frac)
        x1, y1 = bary_to_cart(1 - frac, frac, 0)
        ax.plot([x0, x1], [y0, y1], 'k-', lw=0.3, alpha=0.3)
        # 等 rp 线 (平行于 Generator-Consumer 底边)
        x0, y0 = bary_to_cart(1 - frac, 0, frac)
        x1, y1 = bary_to_cart(0, 1 - frac, frac)
        ax.plot([x0, x1], [y0, y1], 'k-', lw=0.3, alpha=0.3)

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.10, H_TRI + 0.06)
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
    """2x3 面板: K={4,8} x q={0,0.15,1} 的 kappa_c 热力图。"""
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

    # 全局色阶: 判断是否需要 log 变换
    all_kc = []
    for sub in data_dict.values():
        vals = sub["kappa_c_mean"].values
        all_kc.extend(vals[np.isfinite(vals)].tolist())

    if not all_kc:
        print("[WARN] 无有效 kappa_c 数据，跳过绘图")
        return

    kc_min = np.min(all_kc)
    kc_max = np.max(all_kc)
    dynamic_range = kc_max / kc_min if kc_min > 0 else 1.0
    use_log = dynamic_range > 20

    if use_log:
        # log10 变换
        plot_values_key = "kappa_c_log"
        for sub in data_dict.values():
            sub[plot_values_key] = np.log10(
                sub["kappa_c_mean"].clip(lower=1e-10))
        cbar_label = r'$\log_{10}(\overline{\kappa}_c)$'
    else:
        plot_values_key = "kappa_c_mean"
        cbar_label = r'$\overline{\kappa}_c$'

    # 全局色阶范围
    all_z = []
    for sub in data_dict.values():
        vals = sub[plot_values_key].values
        all_z.extend(vals[np.isfinite(vals)].tolist())
    vmin = np.percentile(all_z, 2)
    vmax = np.percentile(all_z, 98)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row_idx, K in enumerate(K_VALUES):
        for col_idx, q in enumerate(Q_VALUES):
            ax = axes[row_idx, col_idx]
            draw_simplex_frame(ax)
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
            z = sub[plot_values_key].values

            try:
                tri = Triangulation(x, y)
                interp = LinearTriInterpolator(tri, z)

                xi = np.linspace(-0.05, 1.05, 200)
                yi = np.linspace(-0.05, H_TRI + 0.05, 200)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = interp(Xi, Yi)

                # 新方向: rp=y/H, rc=x-y/(2H), rg=1-rc-rp
                Rp = Yi / H_TRI
                Rc = Xi - Yi / (2 * H_TRI)
                Rg = 1 - Rc - Rp
                mask_tri = (Rg < -0.01) | (Rc < -0.01) | (Rp < -0.01)
                Zi = np.ma.masked_where(mask_tri, Zi)

                levels = np.linspace(vmin, vmax, 20)
                ax.contourf(Xi, Yi, Zi, levels=levels, cmap='YlGnBu_r',
                            extend='both')
            except (RuntimeError, ValueError):
                title += " (sparse)"

            ax.set_title(title, fontsize=_TITLE, fontweight='bold', pad=8)

            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            ax.scatter(x, y, c=z, cmap='YlGnBu_r', norm=norm,
                       s=40, edgecolors='k', linewidths=0.5,
                       zorder=5, alpha=0.8)

    # 共享 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='YlGnBu_r',
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=_FONT + 1)
    cbar.ax.tick_params(labelsize=_TICK)

    fig.suptitle(r'Critical Coupling $\kappa_c$ — Stability Scan',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.04,
                        wspace=0.12, hspace=0.18)

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
