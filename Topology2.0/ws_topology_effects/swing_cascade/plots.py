"""
ws_cascade_plots.py — 级联 Bisection 相对负载 ρ 绘图

参考 reference_code 中 ρ = |F_trigger| / mean(|F|) 的定义。

4 类图表：
  1. 单 ratio 热力图 ρ(K, q)
  2. 单 ratio 折线图 ρ(q) per K
  3. 1×3 三 ratio 对比热力图
  4. 三 ratio 叠加折线图
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from ws_config import K_list, q_list, RATIO_CONFIGS

# ── 本模块的 cache / output 目录 ──
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
OUTPUT_DIR = _MODULE_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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
    ax.set_ylabel("Mean degree $k$")
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True,
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

        # ── 标注原始网格上 ρ 最大值和最小值 ──
        rho_raw = _get_rho(all_data[rn])
        valid_mask = ~np.isnan(rho_raw)
        if valid_mask.any():
            # 找 max / min
            idx_max = np.nanargmax(rho_raw)
            ki_max, qi_max = np.unravel_index(idx_max, rho_raw.shape)
            q_max, k_max, v_max = Q[qi_max], K[ki_max], rho_raw[ki_max, qi_max]

            rho_valid = np.where(valid_mask, rho_raw, np.inf)
            idx_min = np.argmin(rho_valid)
            ki_min, qi_min = np.unravel_index(idx_min, rho_raw.shape)
            q_min, k_min, v_min = Q[qi_min], K[ki_min], rho_raw[ki_min, qi_min]

            # MAX 标注始终往上, MIN 标注始终往下
            # 若两者的 K 值接近且 q 值也接近，则水平错开
            _ann_kw = dict(textcoords="offset points", fontweight="bold",
                           zorder=6, annotation_clip=False)
            _arr_kw = dict(arrowstyle="-|>", lw=1.0)

            # MAX ★
            ax.plot(q_max, k_max, marker="*", color="blue", markersize=16,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=5)
            max_off = (-60, 30) if q_max > 0.5 else (15, 30)
            ax.annotate(f"MAX $\\rho$={v_max:.3f}\n($q$={q_max:.2f}, $K$={k_max})",
                        xy=(q_max, k_max), xytext=max_off,
                        fontsize=7.5, color="blue",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="blue", alpha=0.85),
                        arrowprops=dict(color="blue", **_arr_kw), **_ann_kw)

            # MIN ★ — 标注放在图内上方，避免遮挡 x 轴
            ax.plot(q_min, k_min, marker="*", color="green", markersize=16,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=5)
            # 若 MIN 和 MAX 的 q 都在左侧，则 MIN 标注偏右；反之偏左
            if abs(q_min - q_max) < 0.3 and q_min < 0.5:
                min_off = (60, 40)
            elif q_min > 0.5:
                min_off = (-60, 40)
            else:
                min_off = (15, 40)
            ax.annotate(f"MIN $\\rho$={v_min:.3f}\n($q$={q_min:.2f}, $K$={k_min})",
                        xy=(q_min, k_min), xytext=min_off,
                        fontsize=7.5, color="green",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="green", alpha=0.85),
                        arrowprops=dict(color="green", **_arr_kw), **_ann_kw)

        ax.set_xlabel("Rewiring probability $q$")
        ax.set_title(_RATIO_LABELS[rn], fontsize=_FONT)
        ax.set_yticks(K)

    axes[0].set_ylabel("Mean degree $k$")

    cbar_ax = fig.add_axes([0.90, 0.10, 0.015, 0.78])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label(r"$\overline{\rho}$")

    fig.subplots_adjust(wspace=0.08, top=0.92, bottom=0.15)
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


# ====================================================================
# 5. 级联持续时间 duration vs alpha 折线图
# ====================================================================

_POINT_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
_POINT_MARKERS = ["o", "s", "D", "^", "v", "P"]


def _load_duration_cache(ratio_name):
    path = CACHE_DIR / f"cascade_duration_{ratio_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_cascade_duration(ratio_name):
    """Duration vs alpha 折线图，每条线对应一个兴趣点。"""
    data = _load_duration_cache(ratio_name)
    alpha_list = data["alpha_list"]
    duration_mean = data["duration_mean"]
    duration_std = data["duration_std"]
    points = data["interest_points"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for pi, pt in enumerate(points):
        label = f"{pt['label']} (K={pt['K']}, q={pt['q']:.2f})"
        c = _POINT_COLORS[pi % len(_POINT_COLORS)]
        m = _POINT_MARKERS[pi % len(_POINT_MARKERS)]

        mean = duration_mean[pi]
        std = duration_std[pi]
        valid = ~np.isnan(mean)
        if not valid.any():
            continue

        a = np.array(alpha_list)
        ax.plot(a[valid], mean[valid], f"{m}-", color=c, markersize=6,
                label=label)
        ax.fill_between(a[valid], (mean - std)[valid], (mean + std)[valid],
                        color=c, alpha=0.15)

    ax.set_xlabel(r"Overload tolerance $\alpha$")
    ax.set_ylabel("Cascade duration (simulation time)")
    ax.set_title(f"Cascade duration vs $\\alpha$ — {ratio_name}")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(bottom=0)

    _save_fig(fig, f"cascade_duration_{ratio_name}")


def plot_cascade_duration_survival(ratio_name):
    """Surviving fraction vs alpha，每条线对应一个兴趣点。"""
    data = _load_duration_cache(ratio_name)
    alpha_list = data["alpha_list"]
    surviving_mean = data["surviving_mean"]
    surviving_std = data["surviving_std"]
    points = data["interest_points"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for pi, pt in enumerate(points):
        label = f"{pt['label']} (K={pt['K']}, q={pt['q']:.2f})"
        c = _POINT_COLORS[pi % len(_POINT_COLORS)]
        m = _POINT_MARKERS[pi % len(_POINT_MARKERS)]

        mean = surviving_mean[pi]
        std = surviving_std[pi]
        valid = ~np.isnan(mean)
        if not valid.any():
            continue

        a = np.array(alpha_list)
        ax.plot(a[valid], mean[valid], f"{m}-", color=c, markersize=6,
                label=label)
        ax.fill_between(a[valid], (mean - std)[valid], (mean + std)[valid],
                        color=c, alpha=0.15)

    ax.set_xlabel(r"Overload tolerance $\alpha$")
    ax.set_ylabel("Surviving edge fraction")
    ax.set_title(f"Surviving fraction vs $\\alpha$ — {ratio_name}")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0, 1.05)

    _save_fig(fig, f"cascade_duration_survival_{ratio_name}")


def plot_cascade_duration_combined():
    """三 ratio 对比: 2×3 子图，上排 duration，下排 surviving。"""
    ratio_names = list(RATIO_CONFIGS.keys())

    all_data = {}
    for rn in ratio_names:
        try:
            all_data[rn] = _load_duration_cache(rn)
        except FileNotFoundError:
            print(f"  [skip] No duration cache for {rn}")
            return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)

    for col, rn in enumerate(ratio_names):
        data = all_data[rn]
        alpha_list = data["alpha_list"]
        points = data["interest_points"]
        a = np.array(alpha_list)

        # 上排: duration
        ax_dur = axes[0, col]
        for pi, pt in enumerate(points):
            mean = data["duration_mean"][pi]
            std = data["duration_std"][pi]
            valid = ~np.isnan(mean)
            if not valid.any():
                continue
            c = _POINT_COLORS[pi % len(_POINT_COLORS)]
            m = _POINT_MARKERS[pi % len(_POINT_MARKERS)]
            label = f"{pt['label']}" if col == 0 else None
            ax_dur.plot(a[valid], mean[valid], f"{m}-", color=c,
                        markersize=5, label=label)
            ax_dur.fill_between(a[valid], (mean - std)[valid],
                                (mean + std)[valid], color=c, alpha=0.1)

        ax_dur.set_title(_RATIO_LABELS.get(rn, rn))
        ax_dur.set_ylim(bottom=0)
        if col == 0:
            ax_dur.set_ylabel("Cascade duration")
            ax_dur.legend(fontsize=7, loc="best")

        # 下排: surviving
        ax_srv = axes[1, col]
        for pi, pt in enumerate(points):
            mean = data["surviving_mean"][pi]
            std = data["surviving_std"][pi]
            valid = ~np.isnan(mean)
            if not valid.any():
                continue
            c = _POINT_COLORS[pi % len(_POINT_COLORS)]
            m = _POINT_MARKERS[pi % len(_POINT_MARKERS)]
            ax_srv.plot(a[valid], mean[valid], f"{m}-", color=c, markersize=5)
            ax_srv.fill_between(a[valid], (mean - std)[valid],
                                (mean + std)[valid], color=c, alpha=0.1)

        ax_srv.set_xlabel(r"$\alpha$")
        ax_srv.set_ylim(0, 1.05)
        if col == 0:
            ax_srv.set_ylabel("Surviving fraction")

    fig.tight_layout()
    _save_fig(fig, "combined_cascade_duration")


def plot_all_duration():
    """生成所有级联持续时间图表。"""
    print("\n── Cascade duration plots ──")
    ratio_names = list(RATIO_CONFIGS.keys())

    for rn in ratio_names:
        try:
            plot_cascade_duration(rn)
            plot_cascade_duration_survival(rn)
        except FileNotFoundError:
            print(f"  [skip] No duration cache for {rn}")

    try:
        plot_cascade_duration_combined()
    except FileNotFoundError:
        print("  [skip] Missing duration cache for combined plot")
