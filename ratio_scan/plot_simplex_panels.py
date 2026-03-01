"""
plot_simplex_panels.py — Ratio Scan 可视化
生成两张 2×3 三元热力图面板:
  A) α* 热力图: fig_alpha_star_panels.png
  B) 失效模式区域: fig_failure_mode_panels.png

数据来源: cache/raw_results.csv
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
from matplotlib.lines import Line2D

# ── 路径 ──────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
RESULTS_DIR = _MODULE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 样式常量 (与项目一致) ─────────────────────────────────────────
_FONT = 11
_TITLE = 13
_TICK = 9
_DPI = 300

K_VALUES = [4, 8]
Q_VALUES = [0.0, 0.15, 1.0]
N_HOUSEHOLDS = 49

# ====================================================================
# 三元坐标工具 (复用自 experiment_ratio_simplex)
# ====================================================================

H_TRI = np.sqrt(3) / 2


def bary_to_cart(rg, rc, rp):
    """重心坐标 (rg, rc, rp) → 2D 笛卡尔坐标。

    与 figure1.py 一致的论文方向:
        Generator (rg=1) → 左下   (0, 0)
        Consumer  (rc=1) → 右下   (1, 0)
        Passive   (rp=1) → 顶部   (0.5, H_TRI)
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
# 数据加载 — 3 个辅助函数
# ====================================================================

def load_csv(path):
    """读取 CSV，确保数值类型，清洗无效行。"""
    if not path.exists():
        print(f"[ERROR] 找不到数据文件: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # 确保数值列为 float
    for col in ("alpha_star", "p_sync", "p_flow", "alpha_pas", "K", "q"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除核心字段缺失的行
    df = df.dropna(subset=["alpha_star", "p_sync", "p_flow"])

    # 如果有 n_valid 列，删除 n_valid <= 0
    if "n_valid" in df.columns:
        df = df[df["n_valid"] > 0]

    return df.reset_index(drop=True)


def select_alpha_pas(df, target=1.0):
    """返回 target 对应的 alpha_pas 值；若不存在则回退到最大值。"""
    available = sorted(df["alpha_pas"].unique())
    for v in available:
        if np.isclose(v, target):
            return target
    fallback = max(available)
    print(f"[WARN] alpha_pas={target} 不存在，使用最大值 {fallback}")
    return fallback


def load_data_by_panel(df, alpha_pas_value):
    """按 alpha_pas 筛选，计算比例和笛卡尔坐标，返回 dict[(K,q)] → DataFrame。"""
    sub = df[np.isclose(df["alpha_pas"], alpha_pas_value)].copy()

    sub["rg"] = sub["ng"] / N_HOUSEHOLDS
    sub["rc"] = sub["nc"] / N_HOUSEHOLDS
    sub["rp"] = sub["np"] / N_HOUSEHOLDS
    sub["x"], sub["y"] = bary_to_cart(sub["rg"].values, sub["rc"].values, sub["rp"].values)

    result = {}
    for K in K_VALUES:
        for q in Q_VALUES:
            mask = (sub["K"] == K) & (np.isclose(sub["q"], q))
            panel = sub[mask].copy()
            if len(panel) > 0:
                result[(K, q)] = panel
    return result


# ====================================================================
# Figure A: α* 三元热力图
# ====================================================================

def plot_alpha_star_panels(csv_path, out_path):
    """2×3 面板: K={4,8} × q={0, 0.15, 1.0} 的 α* 热力图。"""
    df = load_csv(csv_path)
    alpha_pas = select_alpha_pas(df)
    data_dict = load_data_by_panel(df, alpha_pas)

    if not data_dict:
        print("[WARN] 无有效 α* 数据，跳过 Figure A")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 全局色阶
    all_alpha = []
    for sub in data_dict.values():
        vals = sub["alpha_star"].values
        all_alpha.extend(vals[np.isfinite(vals)].tolist())

    if not all_alpha:
        print("[WARN] 无有效 α* 数据，跳过 Figure A")
        plt.close(fig)
        return

    vmin = np.percentile(all_alpha, 2)
    vmax = np.percentile(all_alpha, 98)

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
            z = sub["alpha_star"].values

            # 三角剖分 + 线性插值 (点太少或共线时回退为散点图)
            try:
                tri = Triangulation(x, y)
                interp = LinearTriInterpolator(tri, z)

                xi = np.linspace(-0.05, 1.05, 200)
                yi = np.linspace(-0.05, H_TRI + 0.05, 200)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = interp(Xi, Yi)

                # 三角形外遮罩 (与 bary_to_cart 一致的逆映射)
                Rp = Yi / H_TRI
                Rc = Xi - Yi / (2 * H_TRI)
                Rg = 1 - Rc - Rp
                mask_tri = (Rg < -0.01) | (Rc < -0.01) | (Rp < -0.01)
                Zi = np.ma.masked_where(mask_tri, Zi)

                levels = np.linspace(vmin, vmax, 20)
                ax.contourf(Xi, Yi, Zi, levels=levels, cmap='viridis',
                            extend='both')
            except (RuntimeError, ValueError):
                # 数据点不足以三角化，用彩色散点代替
                title += " (sparse)"

            ax.set_title(title, fontsize=_TITLE, fontweight='bold', pad=8)

            # 数据点标记
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            ax.scatter(x, y, c=z, cmap='viridis', norm=norm,
                       s=40, edgecolors='k', linewidths=0.5,
                       zorder=5, alpha=0.8)

    # 共享 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'$\alpha^*$', fontsize=_FONT + 1)
    cbar.ax.tick_params(labelsize=_TICK)

    fig.suptitle(r'Critical Overload Tolerance $\alpha^*$ — Ratio Scan',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.04,
                        wspace=0.12, hspace=0.18)

    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)


# ====================================================================
# Figure B: 失效模式区域
# ====================================================================

def plot_failure_mode_panels(csv_path, out_path):
    """2×3 面板: 失效模式分类 (sync-dominant vs flow-dominant)。"""
    df = load_csv(csv_path)
    alpha_pas = select_alpha_pas(df)
    data_dict = load_data_by_panel(df, alpha_pas)

    if not data_dict:
        print("[WARN] 无数据，跳过 Figure B")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row_idx, K in enumerate(K_VALUES):
        for col_idx, q in enumerate(Q_VALUES):
            ax = axes[row_idx, col_idx]
            draw_simplex_frame(ax)
            title = f"K={K}, q={q:g}"

            key = (K, q)
            if key not in data_dict or len(data_dict[key]) < 1:
                ax.set_title(title, fontsize=_TITLE, fontweight='bold', pad=8)
                ax.text(0.5, 0.4, "No data", ha='center', fontsize=_FONT,
                        color='red')
                continue

            ax.set_title(title, fontsize=_TITLE, fontweight='bold', pad=8)

            sub = data_dict[key]
            x = sub["x"].values
            y = sub["y"].values
            p_sync = sub["p_sync"].values
            p_flow = sub["p_flow"].values

            # 分类: sync-dominant vs flow-dominant
            is_sync = p_sync > p_flow
            weight = np.maximum(p_sync, p_flow)
            sizes = np.clip(30 + 120 * weight, 30, 150)

            # sync-dominant 红色, flow-dominant 蓝色
            sync_mask = is_sync
            flow_mask = ~is_sync

            if sync_mask.any():
                ax.scatter(x[sync_mask], y[sync_mask], c='#e74c3c',
                           s=sizes[sync_mask], edgecolors='k', linewidths=0.3,
                           zorder=5, alpha=0.7, label='Sync loss')
            if flow_mask.any():
                ax.scatter(x[flow_mask], y[flow_mask], c='#3498db',
                           s=sizes[flow_mask], edgecolors='k', linewidths=0.3,
                           zorder=5, alpha=0.7, label='Overload')

    # 共享图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, markeredgecolor='k', markeredgewidth=0.5,
               label='Sync loss dominant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=10, markeredgecolor='k', markeredgewidth=0.5,
               label='Overload dominant'),
    ]
    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.97, 0.5), fontsize=_FONT, framealpha=0.9)

    fig.suptitle('Failure Mode Regions — Ratio Scan',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.04, right=0.88, top=0.92, bottom=0.04,
                        wspace=0.12, hspace=0.18)

    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)


# ====================================================================
# main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Ratio Scan 三元热力图可视化")
    parser.add_argument("--only-alpha", action="store_true",
                        help="仅生成 Figure A (α* 热力图)")
    parser.add_argument("--only-modes", action="store_true",
                        help="仅生成 Figure B (失效模式)")
    args = parser.parse_args()

    csv_path = CACHE_DIR / "raw_results.csv"

    do_alpha = not args.only_modes
    do_modes = not args.only_alpha

    if do_alpha:
        out_a = RESULTS_DIR / "fig_alpha_star_panels.png"
        plot_alpha_star_panels(csv_path, out_a)

    if do_modes:
        out_b = RESULTS_DIR / "fig_failure_mode_panels.png"
        plot_failure_mode_panels(csv_path, out_b)

    print("Done.")


if __name__ == "__main__":
    main()
