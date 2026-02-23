"""
W-S 拓扑结构 bond percolation 组合图
参考 sturcturef5.jpeg 的多面板布局:
  A: W-S 网络可视化 (不同 q_ws) + q_c 标注
  B: S_2 vs q_perc 曲线, 多面板对比不同 α
  C: 经典 S_1, S_2 vs q_perc 曲线 + inset
  D: q_c (和 κ_c) vs α，误差棒
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from ws_percolation import (
    generate_ws_network, generate_ws_network_alpha,
    bond_percolation_sweep, find_qc,
    compute_percolation_vs_qws, compute_percolation_vs_alpha,
    compute_kappa_c_vs_alpha,
    N, K_BAR, REALIZATIONS, N_PERC_TRIALS, Q_PERC_VALUES,
    Q_WS_LIST, ALPHA_LIST, CACHE_DIR, OUTPUT_DIR,
)

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'sans-serif',
})

# 颜色方案
COLORS_QWS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
COLORS_ALPHA = plt.cm.viridis(np.linspace(0, 0.9, len(ALPHA_LIST)))


# ============================================================
# Panel A: 网络可视化
# ============================================================

def draw_panel_A(fig, gs_a):
    """W-S 网络可视化，不同 q_ws，标注 q_c。"""
    q_ws_show = [0.0, 0.1, 0.5, 1.0]
    n_show = len(q_ws_show)
    n_rows = 2
    n_cols = 2
    gs_inner = gs_a.subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    rng = np.random.default_rng(42)
    q_perc_fine = np.linspace(0, 1, 201)

    for idx, q_ws in enumerate(q_ws_show):
        ax = fig.add_subplot(gs_inner[idx // n_cols, idx % n_cols])
        G = generate_ws_network(N, K_BAR, q_ws)

        # 计算 q_c
        S1, S2 = bond_percolation_sweep(G, q_perc_fine, 10, rng)
        qc = find_qc(q_perc_fine, S2)

        # 圆形布局
        pos = nx.circular_layout(G)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.5, edge_color='k')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color='k')

        ax.set_title(f"$q_{{ws}}={q_ws}$", fontsize=8, pad=2)
        ax.text(0.5, -0.12, f"$q_c={qc:.2f}$", transform=ax.transAxes,
                ha='center', fontsize=7, color='#1f77b4')
        ax.set_aspect('equal')
        ax.axis('off')


# ============================================================
# Panel B: S_2 vs q_perc, 多面板对比不同 α
# ============================================================

def draw_panel_B(fig, gs_b, data_alpha):
    """S_2 vs q_perc 渗流曲线，4 个子面板对比不同 α。"""
    alpha_list = data_alpha['alpha_list']
    q_perc = data_alpha['q_perc_values']
    S2_mean = data_alpha['S2_mean']

    # 选择 4 个代表性 α 值作为面板标题高亮
    highlight_alphas = [0, 1.0, 2.5, 5.0]
    gs_inner = gs_b.subgridspec(1, 4, wspace=0.1)

    for pi, h_alpha in enumerate(highlight_alphas):
        ax = fig.add_subplot(gs_inner[0, pi])

        # 所有曲线都画，高亮 α 加粗
        for ai, alpha in enumerate(alpha_list):
            lw = 1.8 if abs(alpha - h_alpha) < 0.01 else 0.6
            al = 1.0 if abs(alpha - h_alpha) < 0.01 else 0.4
            ax.plot(q_perc, S2_mean[ai], color=COLORS_ALPHA[ai],
                    lw=lw, alpha=al)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, S2_mean.max() * 1.15)
        ax.set_xlabel("$q$", fontsize=8)
        if pi == 0:
            ax.set_ylabel("$S_2$", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"$\\alpha={h_alpha:.0f}$" if h_alpha == int(h_alpha)
                     else f"$\\alpha={h_alpha}$", fontsize=8)
        ax.tick_params(labelsize=7)


# ============================================================
# Panel C: 经典 S_1, S_2 曲线 + inset
# ============================================================

def draw_panel_C(fig, gs_c, data_qws):
    """经典 S_1, S_2 vs q_perc 曲线 (q_ws=0 和 q_ws=1)，带 inset。"""
    ax = fig.add_subplot(gs_c)

    q_perc = data_qws['q_perc_values']
    S1_mean = data_qws['S1_mean']
    S2_mean = data_qws['S2_mean']
    q_ws_list = data_qws['q_ws_list']

    # 画 q_ws=0 (lattice) 和 q_ws=1 (random) 两条
    for qi, q_ws_target in enumerate([0.0, 1.0]):
        idx = np.argmin(np.abs(q_ws_list - q_ws_target))
        color = '#2ca02c' if q_ws_target == 0.0 else '#1f77b4'
        label = f"$q_{{ws}}={q_ws_target:.0f}$"
        ax.plot(q_perc, S1_mean[idx], '-', color=color, lw=1.5,
                label=f"$S_1$, {label}")
        ax.plot(q_perc, S2_mean[idx], '--', color=color, lw=1.5,
                label=f"$S_2$, {label}")

    ax.set_xlabel("$q$", fontsize=9)
    ax.set_ylabel("$S$", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='upper right')

    # Inset: 放大 S_2 峰值附近
    ax_in = ax.inset_axes([0.15, 0.45, 0.35, 0.35])
    for qi, q_ws_target in enumerate([0.0, 1.0]):
        idx = np.argmin(np.abs(q_ws_list - q_ws_target))
        color = '#2ca02c' if q_ws_target == 0.0 else '#1f77b4'
        ax_in.plot(q_perc, S1_mean[idx], '-', color=color, lw=1.2)
        ax_in.plot(q_perc, S2_mean[idx], '--', color=color, lw=1.2)
    ax_in.set_xlim(0, 0.2)
    ax_in.set_ylim(0, 1.05)
    ax_in.tick_params(labelsize=6)


# ============================================================
# Panel D: q_c 和 κ_c vs α
# ============================================================

def draw_panel_D(fig, gs_d, data_alpha, data_kappa_alpha=None):
    """q_c (和 κ_c) vs α，误差棒。"""
    ax = fig.add_subplot(gs_d)

    alpha_list = data_alpha['alpha_list']
    qc_mean = data_alpha['qc_mean']
    qc_std = data_alpha['qc_std']

    ax.errorbar(alpha_list, qc_mean, yerr=qc_std,
                fmt='o-', capsize=3, capthick=1.2, lw=1.8,
                color='#d62728', markersize=5, label='$q_c$ (percolation)',
                zorder=3)

    ax.set_xlabel(r"$\alpha$", fontsize=9)
    ax.set_ylabel(r"$q_c$", fontsize=9, color='#d62728')
    ax.tick_params(axis='y', labelcolor='#d62728')

    # 如果有 κ_c 数据，叠加右轴
    if data_kappa_alpha is not None:
        alpha_k = data_kappa_alpha['alpha_list']
        kc_mean = data_kappa_alpha['kappa_c_mean']
        kc_std = data_kappa_alpha['kappa_c_std']

        ax2 = ax.twinx()
        ax2.errorbar(alpha_k, kc_mean, yerr=kc_std,
                     fmt='s-', capsize=3, capthick=1.2, lw=1.8,
                     color='#1f77b4', markersize=5, label=r'$\kappa_c$ (swing)',
                     zorder=2)
        ax2.set_ylabel(r"$\kappa_c$", fontsize=9, color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
    else:
        ax.legend(fontsize=7, loc='best')

    ax.grid(True, alpha=0.3)


# ============================================================
# 主图组合
# ============================================================

def make_figure():
    """生成参考 sturcturef5 形式的四面板组合图。"""
    print("Loading data ...")

    # 加载或计算数据
    data_qws = compute_percolation_vs_qws()
    data_alpha = compute_percolation_vs_alpha()

    # 尝试加载 κ_c 数据（可能不存在）
    kappa_alpha_file = CACHE_DIR / f"kappa_c_vs_alpha_n{N}_k{K_BAR}_R{REALIZATIONS}.npz"
    data_kappa_alpha = None
    if kappa_alpha_file.exists():
        data_kappa_alpha = dict(np.load(kappa_alpha_file, allow_pickle=True))
        print("  κ_c vs α data loaded.")

    # ---- 布局 ----
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[1, 1],
                           width_ratios=[1, 1.6],
                           hspace=0.35, wspace=0.3)

    # Panel A: 左上
    gs_a = gs[0, 0]
    print("Drawing Panel A (network visualization) ...")
    draw_panel_A(fig, gs_a)

    # Panel B: 右上
    gs_b = gs[0, 1]
    print("Drawing Panel B (S2 vs q, multiple α) ...")
    draw_panel_B(fig, gs_b, data_alpha)

    # Panel C: 左下
    gs_c = gs[1, 0]
    print("Drawing Panel C (S1, S2 classic curves) ...")
    draw_panel_C(fig, gs_c, data_qws)

    # Panel D: 右下
    gs_d = gs[1, 1]
    print("Drawing Panel D (q_c and κ_c vs α) ...")
    draw_panel_D(fig, gs_d, data_alpha, data_kappa_alpha)

    # Panel 标签
    label_kw = dict(fontsize=14, fontweight='bold', va='top', ha='left')
    fig.text(0.02, 0.97, 'A', **label_kw)
    fig.text(0.42, 0.97, 'B', **label_kw)
    fig.text(0.02, 0.48, 'C', **label_kw)
    fig.text(0.42, 0.48, 'D', **label_kw)

    fig.suptitle("W-S Topology: Bond Percolation & Swing Equation Analysis",
                 fontsize=13, fontweight='bold', y=1.01)

    # 保存
    for ext in ['png', 'pdf']:
        fpath = OUTPUT_DIR / f"ws_percolation_combined.{ext}"
        fig.savefig(fpath, dpi=200, bbox_inches='tight')
        print(f"  Saved → {fpath}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
