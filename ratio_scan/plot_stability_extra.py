"""
plot_stability_extra.py — kappa_c 补充可视化 (2x2 面板)

Panel A: 拓扑敏感度差异图 (K=8 vs K=4)
Panel B: 截面线图 (固定 n_passive)
Panel C: 小提琴图 (kappa_c 分布按 K,q 分组)
Panel D: 排名条形图 (Top/Bottom 最稳定配比)

数据来源: cache/stability_agg.csv
输出: results/fig_kappa_c_extra.png
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

H_TRI = np.sqrt(3) / 2


# ====================================================================
# 三元坐标工具 (与 plot_stability_panels.py 一致)
# ====================================================================

def bary_to_cart(rg, rc, rp):
    x = rc + 0.5 * rp
    y = H_TRI * rp
    return x, y


def draw_simplex_frame(ax):
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H_TRI]])
    triangle = plt.Polygon(verts, fill=False, edgecolor='k', linewidth=1.2)
    ax.add_patch(triangle)
    offset = 0.06
    ax.text(verts[0, 0] - offset, verts[0, 1] - offset, "Gen",
            ha='center', va='top', fontsize=_TICK, fontweight='bold')
    ax.text(verts[1, 0] + offset, verts[1, 1] - offset, "Con",
            ha='center', va='top', fontsize=_TICK, fontweight='bold')
    ax.text(verts[2, 0], verts[2, 1] + offset, "Pas",
            ha='center', va='bottom', fontsize=_TICK, fontweight='bold')
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.10, H_TRI + 0.06)
    ax.set_aspect('equal')
    ax.axis('off')


# ====================================================================
# 数据加载
# ====================================================================

def load_agg_csv():
    csv_path = CACHE_DIR / "stability_agg.csv"
    if not csv_path.exists():
        print(f"[ERROR] 找不到数据文件: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    for col in ("kappa_c_mean", "kappa_c_std", "K", "q"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["kappa_c_mean"])
    # 自动检测节点总数
    total = int(df["ng"].iloc[0] + df["nc"].iloc[0] + df["np"].iloc[0])
    df["rg"] = df["ng"] / total
    df["rc"] = df["nc"] / total
    df["rp"] = df["np"] / total
    df["x"], df["y"] = bary_to_cart(df["rg"].values, df["rc"].values,
                                     df["rp"].values)
    return df, total


# ====================================================================
# Panel A: 拓扑敏感度差异图 (1x3 子面板)
# ====================================================================

def plot_panel_a(ax_list, df):
    """Δκ_c = κ_c(K=8) - κ_c(K=4)，对 3 个 q 值绘制 simplex 差异图。"""
    df4 = df[df["K"] == 4].copy()
    df8 = df[df["K"] == 8].copy()

    if len(df4) == 0 or len(df8) == 0:
        for ax in ax_list:
            ax.text(0.5, 0.5, "Need K=4 & K=8", transform=ax.transAxes,
                    ha='center', fontsize=_FONT, color='red')
        return

    # 全局 delta 范围 (用于对称色标)
    all_delta = []

    for col_idx, q in enumerate(Q_VALUES):
        ax = ax_list[col_idx]
        draw_simplex_frame(ax)

        sub4 = df4[np.isclose(df4["q"], q)][["ng", "nc", "np", "kappa_c_mean", "x", "y"]]
        sub8 = df8[np.isclose(df8["q"], q)][["ng", "nc", "np", "kappa_c_mean"]]

        if len(sub4) == 0 or len(sub8) == 0:
            ax.set_title(f"q={q:g} (no data)", fontsize=_TICK + 1, pad=4)
            continue

        merged = sub4.merge(sub8, on=["ng", "nc", "np"], suffixes=("_4", "_8"))
        if len(merged) < 3:
            ax.set_title(f"q={q:g} (sparse)", fontsize=_TICK + 1, pad=4)
            continue

        merged["delta_kc"] = merged["kappa_c_mean_8"] - merged["kappa_c_mean_4"]
        all_delta.extend(merged["delta_kc"].values.tolist())

    # 对称色标
    if not all_delta:
        return
    vabs = max(abs(np.percentile(all_delta, 2)), abs(np.percentile(all_delta, 98)))

    for col_idx, q in enumerate(Q_VALUES):
        ax = ax_list[col_idx]
        sub4 = df4[np.isclose(df4["q"], q)][["ng", "nc", "np", "kappa_c_mean", "x", "y"]]
        sub8 = df8[np.isclose(df8["q"], q)][["ng", "nc", "np", "kappa_c_mean"]]
        merged = sub4.merge(sub8, on=["ng", "nc", "np"], suffixes=("_4", "_8"))
        if len(merged) < 3:
            continue

        merged["delta_kc"] = merged["kappa_c_mean_8"] - merged["kappa_c_mean_4"]
        x = merged["x"].values
        y = merged["y"].values
        z = merged["delta_kc"].values

        try:
            tri = Triangulation(x, y)
            interp = LinearTriInterpolator(tri, z)
            xi = np.linspace(-0.05, 1.05, 200)
            yi = np.linspace(-0.05, H_TRI + 0.05, 200)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = interp(Xi, Yi)
            Rp = Yi / H_TRI
            Rc = Xi - Yi / (2 * H_TRI)
            Rg = 1 - Rc - Rp
            mask_tri = (Rg < -0.01) | (Rc < -0.01) | (Rp < -0.01)
            Zi = np.ma.masked_where(mask_tri, Zi)
            levels = np.linspace(-vabs, vabs, 21)
            ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both')
        except (RuntimeError, ValueError):
            pass

        norm = plt.Normalize(vmin=-vabs, vmax=vabs)
        ax.scatter(x, y, c=z, cmap='RdBu_r', norm=norm,
                   s=30, edgecolors='k', linewidths=0.4, zorder=5, alpha=0.8)
        ax.set_title(f"q={q:g}", fontsize=_TICK + 1, pad=4)

    # colorbar (附在最后一个子面板右侧)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(vmin=-vabs, vmax=vabs))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_list, shrink=0.8, pad=0.02)
    cbar.set_label(r'$\Delta\kappa_c$ (K=8 $-$ K=4)', fontsize=_TICK)
    cbar.ax.tick_params(labelsize=_TICK - 1)


# ====================================================================
# Panel B: 截面线图
# ====================================================================

def plot_panel_b(ax, df, total):
    """固定 n_passive, κ_c vs n_consumers, 多条 (K,q) 曲线。"""
    # 选择最接近 total*0.3 的 n_passive
    target_np = round(total * 0.3)
    available_np = sorted(df["np"].unique())
    best_np = min(available_np, key=lambda v: abs(v - target_np))

    sub = df[df["np"] == best_np].copy()
    if len(sub) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
        return

    colors_q = {0.0: '#e8850c', 0.15: '#4a7ebb', 1.0: '#2ca02c'}
    linestyles_K = {4: '-', 8: '--'}

    for K in K_VALUES:
        for q in Q_VALUES:
            mask = (sub["K"] == K) & (np.isclose(sub["q"], q))
            panel = sub[mask].sort_values("nc")
            if len(panel) < 2:
                continue

            nc = panel["nc"].values
            kc_mean = panel["kappa_c_mean"].values
            kc_std = panel["kappa_c_std"].values

            color = colors_q.get(q, 'gray')
            ls = linestyles_K.get(K, '-')
            label = f"K={K}, q={q:g}"

            ax.plot(nc, kc_mean, color=color, ls=ls, lw=1.8, label=label)
            if kc_std is not None and not np.all(np.isnan(kc_std)):
                valid = ~np.isnan(kc_std)
                ax.fill_between(nc[valid],
                                kc_mean[valid] - kc_std[valid],
                                kc_mean[valid] + kc_std[valid],
                                color=color, alpha=0.15)

    ax.set_xlabel("$n_{consumers}$", fontsize=_FONT)
    ax.set_ylabel(r"$\overline{\kappa}_c$", fontsize=_FONT, rotation=0, labelpad=18)
    ax.set_title(f"B  Cross-section ($n_{{pas}}$={best_np})",
                 loc='left', fontsize=_TITLE, fontweight='bold')
    ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)


# ====================================================================
# Panel C: 小提琴图
# ====================================================================

def plot_panel_c(ax, df):
    """κ_c 分布小提琴图, 按 (K, q) 分组。"""
    groups = []
    labels = []
    for K in K_VALUES:
        for q in Q_VALUES:
            mask = (df["K"] == K) & (np.isclose(df["q"], q))
            vals = df[mask]["kappa_c_mean"].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(f"K={K}\nq={q:g}")

    if not groups:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
        return

    positions = range(1, len(groups) + 1)
    parts = ax.violinplot(groups, positions=positions, showmedians=False,
                          showextrema=False)

    # 颜色: K=4 蓝, K=8 橙
    colors = []
    for K in K_VALUES:
        for q in Q_VALUES:
            colors.append('#4a7ebb' if K == 4 else '#e8850c')

    for i, pc in enumerate(parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)

    # 中位数线 + 均值点
    for i, vals in enumerate(groups):
        pos = i + 1
        median = np.median(vals)
        mean = np.mean(vals)
        ax.hlines(median, pos - 0.25, pos + 0.25, colors='k', lw=1.5)
        ax.scatter([pos], [mean], color='red', s=25, zorder=5, marker='D')

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=_TICK - 1)
    ax.set_ylabel(r"$\overline{\kappa}_c$", fontsize=_FONT, rotation=0, labelpad=18)
    ax.set_title("C  Distribution by (K, q)", loc='left',
                 fontsize=_TITLE, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # 图例: 黑线=中位数, 红菱形=均值
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', lw=1.5, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markersize=6, label='Mean'),
    ]
    ax.legend(handles=legend_elements, fontsize=_TICK - 1, loc='upper right')


# ====================================================================
# Panel D: 排名条形图
# ====================================================================

def plot_panel_d(ax, df):
    """Top-8 / Bottom-8 最稳定配比 (选 K=4, q=0)。"""
    # 选代表 (K,q): 优先 K=4, q=0
    for K_try, q_try in [(4, 0.0), (4, 0.15), (8, 0.0)]:
        mask = (df["K"] == K_try) & (np.isclose(df["q"], q_try))
        sub = df[mask].copy()
        if len(sub) >= 8:
            break

    if len(sub) < 4:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
        return

    K_sel, q_sel = sub["K"].iloc[0], sub["q"].iloc[0]
    n_show = min(8, len(sub) // 2)

    top = sub.nsmallest(n_show, "kappa_c_mean")
    bot = sub.nlargest(n_show, "kappa_c_mean")

    # 合并: top (绿) + 间隔 + bottom (红), 从下到上
    combined = pd.concat([bot.iloc[::-1], top], ignore_index=True)
    y_pos = np.arange(len(combined))
    colors_bar = ['#e74c3c'] * len(bot) + ['#27ae60'] * len(top)

    labels_bar = []
    for _, row in combined.iterrows():
        labels_bar.append(f"({int(row['ng'])},{int(row['nc'])},{int(row['np'])})")

    ax.barh(y_pos, combined["kappa_c_mean"].values, color=colors_bar,
            edgecolor='k', linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_bar, fontsize=_TICK - 1)
    ax.set_xlabel(r"$\overline{\kappa}_c$", fontsize=_FONT)
    ax.set_title(f"D  Ranking (K={int(K_sel)}, q={q_sel:g})",
                 loc='left', fontsize=_TITLE, fontweight='bold')

    # 分隔线
    if len(bot) > 0 and len(top) > 0:
        ax.axhline(len(bot) - 0.5, color='gray', ls='--', lw=0.8)
        ax.text(ax.get_xlim()[1] * 0.5, len(bot) + len(top) - 0.5,
                "Most stable", fontsize=_TICK - 1, color='#27ae60',
                ha='center', va='bottom')
        ax.text(ax.get_xlim()[1] * 0.5, 0.5,
                "Least stable", fontsize=_TICK - 1, color='#e74c3c',
                ha='center', va='bottom')

    ax.grid(True, axis='x', alpha=0.3)


# ====================================================================
# 主函数
# ====================================================================

def plot_kappa_c_extra(out_path=None):
    if out_path is None:
        out_path = RESULTS_DIR / "fig_kappa_c_extra.png"

    df, total = load_agg_csv()
    print(f"节点总数: {total}, 数据行数: {len(df)}")

    fig = plt.figure(figsize=(16, 12))

    # Panel A: 差异图 (上半部分, 1×3 子面板)
    gs_top = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                               left=0.06, right=0.94, top=0.93, bottom=0.05)

    # Panel A 占据左上 (嵌套 1×3)
    gs_a = gs_top[0, 0].subgridspec(1, 3, wspace=0.15)
    ax_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    plot_panel_a(ax_a, df)
    # Panel A 总标题
    fig.text(0.06, 0.95, "A", fontsize=_TITLE + 2, fontweight='bold', va='top')
    fig.text(0.10, 0.95, r"Topology sensitivity $\Delta\kappa_c$ (K=8 $-$ K=4)",
             fontsize=_FONT, va='top')

    # Panel B: 截面线图 (右上)
    ax_b = fig.add_subplot(gs_top[0, 1])
    plot_panel_b(ax_b, df, total)

    # Panel C: 小提琴图 (左下)
    ax_c = fig.add_subplot(gs_top[1, 0])
    plot_panel_c(ax_c, df)

    # Panel D: 排名条形图 (右下)
    ax_d = fig.add_subplot(gs_top[1, 1])
    plot_panel_d(ax_d, df)

    fig.suptitle(r'Supplementary: $\kappa_c$ Analysis',
                 fontsize=15, fontweight='bold', y=0.99)

    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kappa_c 补充可视化")
    parser.parse_args()
    plot_kappa_c_extra()
