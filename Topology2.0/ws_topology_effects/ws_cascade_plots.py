"""
ws_cascade_plots.py — 级联 Bisection 相对负载 ρ 绘图

参考 reference_code 中 ρ = |F_trigger| / mean(|F|) 的定义。

4 类图表：
  1. 单 ratio 热力图 ρ(K, q)
  2. 单 ratio 折线图 ρ(q) per K
  3. 1×3 三 ratio 对比热力图
  4. 三 ratio 叠加折线图
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from ws_config import (
    K_list, q_list, RATIO_CONFIGS, OUTPUT_DIR, CACHE_DIR,
)

# ── 统一样式（同 ws_plots.py）──────────────────────────────────────
_FONT = 11
_TITLE_FONT = 13
_TICK_FONT = 9
_LW = 1.8
_DPI = 300

plt.rcParams.update({
    "font.size": _FONT,
    "axes.titlesize": _TITLE_FONT,
    "axes.labelsize": _FONT,
    "xtick.labelsize": _TICK_FONT,
    "ytick.labelsize": _TICK_FONT,
    "legend.fontsize": _TICK_FONT,
    "lines.linewidth": _LW,
    "figure.dpi": _DPI,
    "savefig.dpi": _DPI,
    "savefig.bbox": "tight",
})

# Per-K 颜色（同 ws_plots.py）
_K_COLORS = {
    4:  "#1b9e77",
    6:  "#d95f02",
    8:  "#7570b3",
    10: "#e7298a",
    12: "#66a61e",
}
_K_MARKERS = {4: "o", 6: "s", 8: "D", 10: "^", 12: "v"}

# Ratio 标签和颜色（同 ws_plots_combined.py）
_RATIO_LABELS = {"balanced": "G=C", "gen_heavy": "G>C", "load_heavy": "G<C"}
_RATIO_COLORS = {
    "balanced":   "#2B547E",
    "gen_heavy":  "#8B4049",
    "load_heavy": "#5B7B50",
}
_K_STYLES = {
    4:  ("-.",  "o"),
    6:  ("--",  "s"),
    8:  ("-",   "D"),
    10: (":",   "^"),
    12: ((0, (3, 1, 1, 1)), "v"),
}


def _load_bisection_cache(ratio_name):
    # 优先加载 swing 缓存，回退到旧 DC 缓存
    path_swing = CACHE_DIR / f"cascade_bisection_swing_{ratio_name}.pkl"
    path_dc = CACHE_DIR / f"cascade_bisection_{ratio_name}.pkl"
    path = path_swing if path_swing.exists() else path_dc
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_fig(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"  Saved {stem}.png/.pdf")


def _get_rho(data):
    """从缓存中提取 alpha_c 矩阵作为 ρ̄（同论文 Figure 3 Panel C/D 定义）。"""
    if "alpha_c" in data:
        return data["alpha_c"]
    return None


# ====================================================================
# 1. 单 ratio 热力图 ρ(K, q)
# ====================================================================

def plot_bisection_heatmap(ratio_name):
    r"""单 ratio 热力图：$\overline{\rho}(K, q)$。"""
    data = _load_bisection_cache(ratio_name)
    rho = _get_rho(data)
    if rho is None:
        print(f"  [skip] No rho_mean in cache for {ratio_name}, need recompute")
        return

    Q = np.array(q_list)
    K = np.array(K_list)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 插值到精细网格
    q_fine = np.linspace(Q[0], Q[-1], 200)
    k_fine = np.linspace(K[0], K[-1], 200)
    Qf, Kf = np.meshgrid(q_fine, k_fine)

    # 填充 NaN
    rho_fill = rho.copy()
    for ki in range(len(K)):
        row = rho_fill[ki]
        valid = ~np.isnan(row)
        if valid.any():
            rho_fill[ki] = np.interp(Q, Q[valid], row[valid])

    interp = RegularGridInterpolator((K, Q), rho_fill, method="linear",
                                     bounds_error=False, fill_value=None)
    pts = np.column_stack([Kf.ravel(), Qf.ravel()])
    Zf = interp(pts).reshape(Kf.shape)

    cf = ax.contourf(Qf, Kf, Zf, levels=20, cmap="Reds")
    cs = ax.contour(Qf, Kf, Zf, levels=8, colors="k", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(r"$\overline{\rho}$")

    # 标记原始网格点
    Qg, Kg = np.meshgrid(Q, K)
    valid_pts = ~np.isnan(rho)
    ax.scatter(Qg[valid_pts], Kg[valid_pts], s=8, c="k", alpha=0.4, zorder=3)
    ax.scatter(Qg[~valid_pts], Kg[~valid_pts], s=20, c="grey", marker="x",
               alpha=0.6, zorder=3)

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel("Degree $K$")
    ax.set_title(rf"Relative load $\overline{{\rho}}$ — {ratio_name}")
    ax.set_yticks(K)

    _save_fig(fig, f"cascade_bisection_heatmap_{ratio_name}")


# ====================================================================
# 2. 单 ratio 折线图 ρ(q) per K
# ====================================================================

def plot_bisection_lines(ratio_name):
    r"""单 ratio 折线图：$\overline{\rho}(q)$ 对每个 K。"""
    data = _load_bisection_cache(ratio_name)
    rho = _get_rho(data)
    if rho is None:
        print(f"  [skip] No rho_mean in cache for {ratio_name}, need recompute")
        return

    Q = np.array(q_list)

    fig, ax = plt.subplots(figsize=(9, 3.5))

    for ki, Kv in enumerate(K_list):
        vals = rho[ki]
        valid = ~np.isnan(vals)
        if not valid.any():
            continue
        c = _K_COLORS.get(Kv, "gray")
        m = _K_MARKERS.get(Kv, "o")
        ax.plot(Q[valid], vals[valid], f"{m}-", color=c, markersize=5,
                label=f"$K={Kv}$")

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel(r"$\overline{\rho}$")
    ax.set_title(rf"Relative load $\overline{{\rho}}$ vs $q$ — {ratio_name}")
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0)
    ax.set_xlim(Q[0], Q[-1])
    ax.set_ylim(bottom=0)

    fig.subplots_adjust(right=0.78)
    _save_fig(fig, f"cascade_bisection_lines_{ratio_name}")


# ====================================================================
# 3. 1×3 三 ratio 对比热力图
# ====================================================================

def plot_bisection_combined_heatmap():
    r"""1×3 三 ratio 对比热力图 $\overline{\rho}(K, q)$。"""
    Q = np.array(q_list)
    K = np.array(K_list)
    ratio_names = list(RATIO_CONFIGS.keys())

    # 加载所有缓存
    all_data = {}
    for rn in ratio_names:
        try:
            all_data[rn] = _load_bisection_cache(rn)
        except FileNotFoundError:
            print(f"  [skip] No cache for {rn}")
            return

    # 检查是否有 rho_mean
    for rn in ratio_names:
        if _get_rho(all_data[rn]) is None:
            print(f"  [skip] No rho_mean in cache for {rn}, need recompute")
            return

    # 插值
    q_fine = np.linspace(Q[0], Q[-1], 200)
    k_fine = np.linspace(K[0], K[-1], 200)
    Qf, Kf = np.meshgrid(q_fine, k_fine)

    def _interpolate(raw):
        fill = raw.copy()
        for ki in range(len(K)):
            row = fill[ki]
            valid = ~np.isnan(row)
            if valid.any():
                fill[ki] = np.interp(Q, Q[valid], row[valid])
        interp = RegularGridInterpolator((K, Q), fill, method="linear",
                                         bounds_error=False, fill_value=None)
        pts = np.column_stack([Kf.ravel(), Qf.ravel()])
        return interp(pts).reshape(Kf.shape)

    Z_all = {rn: _interpolate(_get_rho(all_data[rn])) for rn in ratio_names}
    vmin = min(np.nanmin(z) for z in Z_all.values())
    vmax = max(np.nanmax(z) for z in Z_all.values())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True,
                             gridspec_kw={"right": 0.88})

    cf = None
    for col, rn in enumerate(ratio_names):
        ax = axes[col]
        Z = Z_all[rn]
        cf = ax.contourf(Qf, Kf, Z, levels=20, cmap="Reds",
                         vmin=vmin, vmax=vmax)
        cs = ax.contour(Qf, Kf, Z, levels=8, colors="k",
                        linewidths=0.5, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

        Qg, Kg = np.meshgrid(Q, K)
        ax.scatter(Qg.ravel(), Kg.ravel(), s=8, c="k", alpha=0.3, zorder=3)

        ax.set_xlabel("Rewiring probability $q$")
        ax.set_title(_RATIO_LABELS[rn], fontsize=_FONT)
        ax.set_yticks(K)

    axes[0].set_ylabel("Degree $K$")

    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.76])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label(r"$\overline{\rho}$")

    fig.subplots_adjust(wspace=0.08, top=0.92)
    _save_fig(fig, "combined_cascade_bisection_heatmap")


# ====================================================================
# 4. 三 ratio 叠加折线图
# ====================================================================

def plot_bisection_combined_lines():
    r"""三 ratio 叠加折线图 $\overline{\rho}(q)$ per K。"""
    Q = np.array(q_list)
    ratio_names = list(RATIO_CONFIGS.keys())

    all_data = {}
    for rn in ratio_names:
        try:
            all_data[rn] = _load_bisection_cache(rn)
        except FileNotFoundError:
            print(f"  [skip] No cache for {rn}")
            return

    for rn in ratio_names:
        if _get_rho(all_data[rn]) is None:
            print(f"  [skip] No rho_mean in cache for {rn}, need recompute")
            return

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for rn in ratio_names:
        rho = _get_rho(all_data[rn])
        color = _RATIO_COLORS[rn]
        label = _RATIO_LABELS[rn]

        for Kv in K_list:
            ki = K_list.index(Kv)
            vals = rho[ki]
            valid = ~np.isnan(vals)
            if not valid.any():
                continue

            ls, mk = _K_STYLES[Kv]
            ax.plot(Q[valid], vals[valid], linestyle=ls, marker=mk,
                    color=color, markersize=5,
                    label=f"{label}  $K={Kv}$")

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel(r"$\overline{\rho}$")
    ax.set_xlim(Q[0], Q[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=3, loc="best", framealpha=0.9,
              columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    _save_fig(fig, "combined_cascade_bisection_lines")


# ====================================================================
# 公共入口
# ====================================================================

def plot_all_bisection():
    """生成所有级联 bisection 相关图表。"""
    print("\n── Cascade bisection plots (ρ) ──")
    ratio_names = list(RATIO_CONFIGS.keys())

    for rn in ratio_names:
        try:
            plot_bisection_heatmap(rn)
            plot_bisection_lines(rn)
        except FileNotFoundError:
            print(f"  [skip] No bisection cache for {rn}")

    try:
        plot_bisection_combined_heatmap()
    except FileNotFoundError:
        print("  [skip] Missing cache for combined heatmap")

    try:
        plot_bisection_combined_lines()
    except FileNotFoundError:
        print("  [skip] Missing cache for combined lines")
