"""
plot_passive_impact.py — Passive 节点对稳定性影响的 2×2 面板

布局:
  (A) Figure1 Panel C 复现 (simplex heatmap, K=4 q=0)
  (B) np=0  (0% passive)  — κ_c vs nc 截面线图
  (C) np=20 (40% passive) — κ_c vs nc 截面线图
  (D) np=35 (70% passive) — κ_c vs nc 截面线图

数据来源:
  Panel A: reproduction/cache/v4_panel_c_n50_k4_q0.0_r100_s2.npz
  Panels B-D: ratio_scan/cache/stability_agg.csv

输出: ratio_scan/results/fig_passive_impact.png
"""

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
_PROJECT_DIR = _MODULE_DIR.parent
CACHE_DIR = _MODULE_DIR / "cache"
RESULTS_DIR = _MODULE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REPRO_CACHE = _PROJECT_DIR / "reproduction" / "cache" / "v4_panel_c_n50_k4_q0.0_r100_s2.npz"
AGG_CSV = CACHE_DIR / "stability_agg.csv"

# ── 常量 ──────────────────────────────────────────────────────────
H_TRI = np.sqrt(3) / 2
N_TOTAL = 50  # 节点总数

# 截面线图中的 (K, q) 组合及其样式 — 6 种颜色完全不同，便于区分
LINE_STYLES = [
    # (K, q, color, linestyle, linewidth, marker, markersize)
    (4, 0.0,  "#d62728", "-",  2.5, "o",  6),   # 红 圆
    (4, 0.15, "#1f77b4", "-",  2.2, "s",  5),   # 蓝 方
    (4, 1.0,  "#2ca02c", "-",  2.2, "D",  5),   # 绿 菱
    (8, 0.0,  "#e8850c", "--", 2.5, "^",  6),   # 橙 上三角
    (8, 0.15, "#9467bd", "--", 2.2, "v",  5),   # 紫 下三角
    (8, 1.0,  "#17becf", "--", 2.2, "P",  6),   # 青 十字
]

# 三个 passive 截面 — 选择差异较大的 np 值
NP_SECTIONS = [
    (0,  "0%"),
    (20, "40%"),
    (35, "70%"),
]

_DPI = 300

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


# ====================================================================
# 三元坐标工具 (复用 plot_stability_panels.py)
# ====================================================================

def bary_to_cart(rg, rc, rp):
    """重心坐标 (rg, rc, rp) -> 2D 笛卡尔坐标。"""
    x = rc + 0.5 * rp
    y = H_TRI * rp
    return x, y


def ternary_to_cart(n_plus, n_minus, n_passive):
    """节点数 -> 2D 笛卡尔坐标 (与 figure1.py 一致)。"""
    s = n_plus + n_minus + n_passive
    if s == 0:
        return 0.0, 0.0
    x = n_minus / s + 0.5 * n_passive / s
    y = n_passive / s * H_TRI
    return x, y


def draw_simplex_frame(ax, total=N_TOTAL):
    """绘制三角形框架 + 边标签 (简化版)。"""
    corners_x = [0, 1, 0.5, 0]
    corners_y = [0, 0, H_TRI, 0]
    ax.plot(corners_x, corners_y, 'k-', lw=1.0)

    # 边标签
    mx, my = 0.5 * (0 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx - 0.08, my, r"$\leftarrow$ Generators", fontsize=15,
            ha='center', va='center', rotation=60)
    ax.text(0.5, -0.07, r"Consumers $\rightarrow$", fontsize=15,
            ha='center', va='top')
    mx, my = 0.5 * (1 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx + 0.08, my, r"$\leftarrow$ Passive", fontsize=15,
            ha='center', va='center', rotation=-60)

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.10, H_TRI + 0.04)
    ax.set_aspect('equal')
    ax.axis('off')


# ====================================================================
# Panel A: Simplex heatmap
# ====================================================================

def plot_panel_a(ax):
    """复现 figure1 Panel C: K=4, q=0 simplex heatmap，带边缘刻度和截面线标注。"""
    if not REPRO_CACHE.exists():
        print(f"[ERROR] 找不到 Panel C 缓存: {REPRO_CACHE}")
        sys.exit(1)

    data = np.load(REPRO_CACHE)
    configs = data["configs"]       # (351, 3): [n_plus, n_minus, n_passive]
    kappa_c_vals = data["kappa_c_vals"]  # (351,)

    # ── 构建查找表 ──
    step = 2
    kc_lookup = {}
    for idx, (n_plus, n_minus, n_passive) in enumerate(configs):
        if not np.isnan(kappa_c_vals[idx]):
            kc_lookup[(int(n_plus), int(n_minus))] = kappa_c_vals[idx]

    # ── 对 NaN 边界点用最近邻外推，使热力图填充到三角形边缘 ──
    for n_plus in range(0, N_TOTAL + 1, step):
        for n_minus in range(0, N_TOTAL + 1 - n_plus, step):
            if (n_plus, n_minus) in kc_lookup:
                continue
            for radius in range(1, 4):
                candidates = []
                for dp in range(-radius * step, (radius + 1) * step, step):
                    for dm in range(-radius * step, (radius + 1) * step, step):
                        if dp == 0 and dm == 0:
                            continue
                        np2, nm2 = n_plus + dp, n_minus + dm
                        npa2 = N_TOTAL - np2 - nm2
                        if (np2 >= 0 and nm2 >= 0 and npa2 >= 0
                                and (np2, nm2) in kc_lookup):
                            candidates.append(kc_lookup[(np2, nm2)])
                if candidates:
                    kc_lookup[(n_plus, n_minus)] = np.mean(candidates)
                    break

    # ── 转换坐标 ──
    xs, ys, vals = [], [], []
    for (n_plus, n_minus), val in kc_lookup.items():
        n_passive = N_TOTAL - n_plus - n_minus
        x, y = ternary_to_cart(n_plus, n_minus, n_passive)
        xs.append(x)
        ys.append(y)
        vals.append(val)
    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

    # ── 绘制热力图 ──
    if len(xs) > 3:
        triang = mtri.Triangulation(xs, ys)
        # 蒙版：去除三角形外部的 Delaunay 三角面
        cx = xs[triang.triangles].mean(axis=1)
        cy = ys[triang.triangles].mean(axis=1)
        eps = 0.01
        outside = ~((cy >= -eps)
                     & (cy <= 2 * H_TRI * cx + eps)
                     & (cy <= 2 * H_TRI * (1 - cx) + eps))
        triang.set_mask(outside)

        tcf = ax.tricontourf(triang, vals, levels=20, cmap='YlGnBu_r')

        # 裁剪到三角形边界
        clip_tri = plt.Polygon([(0, 0), (1, 0), (0.5, H_TRI)],
                                transform=ax.transData, closed=True,
                                facecolor='none', edgecolor='none')
        ax.add_patch(clip_tri)
        tcf.set_clip_path(clip_tri)

        cb = plt.colorbar(tcf, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label(r'$\kappa_c$', fontsize=16)
        cb.ax.tick_params(labelsize=13)

    # ── 三角形边框 ──
    ax.plot([0, 1, 0.5, 0], [0, 0, H_TRI, 0], 'k-', lw=1.0, zorder=5)

    # ── 边缘刻度 ──
    tick_vals = [0, 10, 20, 30, 40, 50]
    tl = 0.02  # 刻度线长度

    # 底边 (Consumers n_c): BL(0) → BR(50)
    for v in tick_vals:
        x, y = ternary_to_cart(N_TOTAL - v, v, 0)
        ax.plot([x, x], [y, y - tl], 'k-', lw=0.8, zorder=5, clip_on=False)
        ax.text(x, y - tl - 0.005, str(v), fontsize=11,
                ha='center', va='top')

    # 左边 (Generators n_g): BL(50) → Top(0)
    # 向外法向量: (-H_TRI, 0.5)
    nx_l, ny_l = -H_TRI, 0.5
    for v in tick_vals:
        x, y = ternary_to_cart(v, 0, N_TOTAL - v)
        ax.plot([x, x + tl * nx_l], [y, y + tl * ny_l],
                'k-', lw=0.8, zorder=5, clip_on=False)
        ax.text(x + (tl + 0.008) * nx_l, y + (tl + 0.008) * ny_l,
                str(v), fontsize=11, ha='right', va='center')

    # 右边 (Passive n_p): BR(0) → Top(50)
    # 向外法向量: (H_TRI, 0.5)
    nx_r, ny_r = H_TRI, 0.5
    for v in tick_vals:
        x, y = ternary_to_cart(0, N_TOTAL - v, v)
        ax.plot([x, x + tl * nx_r], [y, y + tl * ny_r],
                'k-', lw=0.8, zorder=5, clip_on=False)
        ax.text(x + (tl + 0.008) * nx_r, y + (tl + 0.008) * ny_r,
                str(v), fontsize=11, ha='left', va='center')

    # ── 边标签 ──
    mx, my = 0.5 * (0 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx - 0.14, my, r"$\leftarrow$ Generators", fontsize=15,
            ha='center', va='center', rotation=60)
    ax.text(0.5, -0.11, r"Consumers $\rightarrow$", fontsize=15,
            ha='center', va='top')
    mx, my = 0.5 * (1 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx + 0.14, my, r"$\leftarrow$ Passive", fontsize=15,
            ha='center', va='center', rotation=-60)

    # ── 截面线标注 ──
    section_colors = ["#d62728", "#9467bd", "#17becf"]
    for i, (np_val, pct_str) in enumerate(NP_SECTIONS):
        n_active = N_TOTAL - np_val
        if n_active < 2:
            continue
        x0, y0 = ternary_to_cart(n_active, 0, np_val)
        x1, y1 = ternary_to_cart(0, n_active, np_val)
        ax.plot([x0, x1], [y0, y1], '--', color=section_colors[i],
                lw=2.5, zorder=3)
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        ax.text(xm, ym, f"$n_p$={np_val}", fontsize=14, color=section_colors[i],
                ha='center', va='center', fontweight='bold',
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.14, H_TRI + 0.06)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("K=4, q=0 (baseline)", fontsize=17, pad=8)


# ====================================================================
# Panels B-D: 截面线图
# ====================================================================

def load_agg_csv():
    """读取 stability_agg.csv。"""
    if not AGG_CSV.exists():
        print(f"[ERROR] 找不到数据文件: {AGG_CSV}")
        sys.exit(1)

    df = pd.read_csv(AGG_CSV)
    for col in ("kappa_c_mean", "kappa_c_std", "K", "q"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["kappa_c_mean"])
    return df.reset_index(drop=True)


def plot_section_panel(ax, df, np_val, pct_str, show_legend=False):
    """绘制固定 np 的 κ_c vs nc 截面线图，6 条曲线 (线性 y 轴)。"""
    sub = df[df["np"] == np_val].copy()

    for K, q, color, ls, lw, marker, ms in LINE_STYLES:
        mask = (sub["K"] == K) & (np.isclose(sub["q"], q))
        line_data = sub[mask].sort_values("nc")

        if len(line_data) == 0:
            continue

        x = line_data["nc"].values
        y = line_data["kappa_c_mean"].values
        s = line_data["kappa_c_std"].values

        label = f"K={K}, q={q:g}"
        ax.plot(x, y, color=color, ls=ls, lw=lw, label=label,
                marker=marker, ms=ms, zorder=3)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, lw=0.5)

    ax.set_xlabel("Consumers ($n_c$)", fontsize=16)
    ax.set_ylabel(r"$\kappa_c$", fontsize=17, rotation=0, labelpad=24)
    ax.set_title(f"$n_p$={np_val} ({pct_str} passive)", fontsize=17, pad=8)
    ax.tick_params(labelsize=14)

    if show_legend:
        return ax.get_legend_handles_labels()


# ====================================================================
# 主函数
# ====================================================================

def main():
    print("=" * 60)
    print("Passive 节点对稳定性影响 — 1×4 面板")
    print("=" * 60)

    df = load_agg_csv()
    print(f"加载 {len(df)} 行数据")

    fig = plt.figure(figsize=(22, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1],
                          left=0.04, right=0.98, top=0.92, bottom=0.14,
                          wspace=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # Panel A: simplex heatmap
    print("绘制 Panel A: simplex heatmap ...")
    plot_panel_a(axes[0])
    axes[0].text(-0.08, 1.12, "A", transform=axes[0].transAxes,
                 fontsize=18, fontweight='bold', va='top', ha='left')

    # Panels B-D: 截面线图
    panel_axes = [axes[1], axes[2], axes[3]]
    panel_labels = ["B", "C", "D"]
    legend_info = None
    for i, (np_val, pct_str) in enumerate(NP_SECTIONS):
        print(f"绘制 Panel {panel_labels[i]}: np={np_val} ({pct_str}) ...")
        result = plot_section_panel(panel_axes[i], df, np_val, pct_str,
                                    show_legend=(i == 0))
        if result is not None:
            legend_info = result
        panel_axes[i].text(-0.12, 1.12, panel_labels[i],
                           transform=panel_axes[i].transAxes,
                           fontsize=18, fontweight='bold', va='top', ha='left')

    # 将 legend 放到 Panel A 下方
    if legend_info:
        handles, labels = legend_info
        axes[0].legend(handles, labels, fontsize=13, loc='upper center',
                       bbox_to_anchor=(0.5, -0.05), ncol=2, framealpha=0.9)

    out_path = RESULTS_DIR / "fig_passive_impact.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    print(f"\nSaved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
