"""
ws_plots_combined.py — Combined multi-ratio plots for WS topology effects.

Generates true overlay comparisons across ratio configurations:
  1. combined_kappa_c_map.png    — all ratios overlaid on one κ_c(q) line plot
  2. combined_kappa_c_heatmap.png — 1×3 absolute κ_c heatmaps
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

from ws_config import (
    K_list, q_list, K_ref, RATIO_CONFIGS, OUTPUT_DIR, CACHE_DIR,
)

# ── Style ────────────────────────────────────────────────────────
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

# ── Label mapping ────────────────────────────────────────────────
_RATIO_LABELS = {"balanced": "G=C", "gen_heavy": "G>C", "load_heavy": "G<C"}

# ── Visual encoding ──────────────────────────────────────────────
# Colour distinguishes ratio
_RATIO_COLORS = {
    "balanced":   "#2B547E",  # muted deep blue
    "gen_heavy":  "#8B4049",  # soft brick red / burgundy
    "load_heavy": "#5B7B50",  # sage / forest green
}

# Line style + marker distinguishes K
K_PLOT = [6, 8, 10]
_K_STYLES = {
    6:  ("--", "s"),   # dashed, square
    8:  ("-",  "D"),   # solid, diamond
    10: (":",  "^"),   # dotted, triangle
}


def _load_cache(ratio_name: str) -> dict:
    path = CACHE_DIR / f"{ratio_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_fig(fig, stem: str):
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"  Saved {stem}.png/.pdf")


# ====================================================================
# 1. Combined κ_c line plot — all curves overlaid on one axes
# ====================================================================

def plot_combined_kappa_c_map() -> None:
    """Single plot with 9 curves: 3 ratios × 3 K values, overlaid."""
    Q = np.array(q_list)
    ratio_names = list(RATIO_CONFIGS.keys())

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Collect q* info for dual-colour marker handling
    q_stars = {}  # ratio_name -> (q_star, kc_star)
    line_handles = {}  # (ratio_name, K) -> line handle

    for ratio_name in ratio_names:
        data = _load_cache(ratio_name)
        kc_mean = data["kappa_c_map"]
        kc_std = data["kappa_c_std"]
        q_star = data["q_star_Kref"]
        color = _RATIO_COLORS[ratio_name]
        label = _RATIO_LABELS[ratio_name]

        for Kv in K_PLOT:
            ki = K_list.index(Kv)
            mean = kc_mean[ki]
            std = kc_std[ki]
            valid = ~np.isnan(mean)
            if not valid.any():
                continue

            ls, mk = _K_STYLES[Kv]
            line = ax.plot(Q[valid], mean[valid], linestyle=ls, marker=mk,
                           color=color, markersize=5,
                           label=f"{label}  $K={Kv}$")[0]
            line_handles[(ratio_name, Kv)] = line
            ax.fill_between(Q[valid], (mean - std)[valid], (mean + std)[valid],
                            color=color, alpha=0.04)

        # Collect q* for deferred drawing
        if not np.isnan(q_star) and K_ref in K_list:
            ki_ref = K_list.index(K_ref)
            qi_ref = int(np.argmin(np.abs(Q - q_star)))
            kc_star = kc_mean[ki_ref, qi_ref]
            if not np.isnan(kc_star):
                q_stars[ratio_name] = (q_star, kc_star)

    # Draw q* markers, handling overlaps with dual-colour markers
    drawn = set()
    for rn, (qs, kcs) in q_stars.items():
        if rn in drawn:
            continue
        # Check for overlapping q* with another ratio
        overlap_rn = None
        for other_rn, (oqs, okcs) in q_stars.items():
            if other_rn != rn and other_rn not in drawn and abs(qs - oqs) < 1e-6:
                overlap_rn = other_rn
                break

        if overlap_rn is not None:
            # Bright vivid gold focal star for shared q*
            l1 = _RATIO_LABELS[rn]
            l2 = _RATIO_LABELS[overlap_rn]
            y_off = kcs + 0.05
            ax.plot(qs, y_off, marker="*", color="#FFD700",
                    markersize=12, zorder=6, linestyle="none",
                    markeredgecolor="#000000", markeredgewidth=1.0)
            drawn.add(rn)
            drawn.add(overlap_rn)
        else:
            ax.plot(qs, kcs + 0.05, marker="*", color="#FF2222",
                    markersize=12, zorder=5, linestyle="none",
                    markeredgecolor="#000000", markeredgewidth=1.0)
            drawn.add(rn)

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel(r"Critical coupling $\kappa_c$")
    ax.set_xlim(Q[0], Q[-1])
    ax.set_ylim(bottom=0)

    # 4-column grouped legend:
    # col1=G=C, col2=G>C, col3=G<C, col4=star definitions
    star_shared = Line2D([0], [0], marker="*", linestyle="none",
                         color="#FFD700", markeredgecolor="#000000",
                         markeredgewidth=1.0, markersize=12,
                         label=r"$q^*$ (G=C & G<C)")
    star_gen = Line2D([0], [0], marker="*", linestyle="none",
                      color="#FF2222", markeredgecolor="#000000",
                      markeredgewidth=1.0, markersize=12,
                      label=r"$q^*$ (G>C)")
    blank = Line2D([0], [0], linestyle="none", alpha=0.0, label="")

    # NOTE: Matplotlib fills legend entries column-wise for ncol>1 in this setup.
    # Use strict column-major ordering to force:
    # col1=G=C, col2=G>C, col3=G<C, col4=star definitions.
    ordered_handles = [
        line_handles[("balanced", 6)], line_handles[("balanced", 8)],
        line_handles[("balanced", 10)],
        line_handles[("gen_heavy", 6)], line_handles[("gen_heavy", 8)],
        line_handles[("gen_heavy", 10)],
        line_handles[("load_heavy", 6)], line_handles[("load_heavy", 8)],
        line_handles[("load_heavy", 10)],
        star_shared, star_gen, blank,
    ]
    ordered_labels = [
        r"G=C  $K=6$", r"G=C  $K=8$", r"G=C  $K=10$",
        r"G>C  $K=6$", r"G>C  $K=8$", r"G>C  $K=10$",
        r"G<C  $K=6$", r"G<C  $K=8$", r"G<C  $K=10$",
        r"$q^*$ (G=C & G<C)", r"$q^*$ (G>C)", "",
    ]
    ax.legend(ordered_handles, ordered_labels, fontsize=8, ncol=4,
              loc="upper right", framealpha=0.9, columnspacing=1.0,
              handletextpad=0.5)
    fig.tight_layout()
    _save_fig(fig, "combined_kappa_c_map")


# ====================================================================
# 2. Absolute κ_c heatmaps — 1×3 layout
# ====================================================================

def plot_combined_kappa_c_heatmap() -> None:
    """1×3 heatmaps showing absolute κ_c(K, q) for each ratio."""
    Q = np.array(q_list)
    K = np.array(K_list)
    ratio_names = list(RATIO_CONFIGS.keys())

    # Load all caches
    all_data = {rn: _load_cache(rn) for rn in ratio_names}

    # Fine grids for interpolation
    q_fine = np.linspace(Q[0], Q[-1], 200)
    k_fine = np.linspace(K[0], K[-1], 200)
    Qf, Kf = np.meshgrid(q_fine, k_fine)

    def _interpolate(kc_raw):
        """Fill NaNs per row and interpolate onto fine grid."""
        kc_fill = kc_raw.copy()
        for ki in range(len(K)):
            row = kc_fill[ki]
            valid = ~np.isnan(row)
            if valid.any():
                kc_fill[ki] = np.interp(Q, Q[valid], row[valid])
        interp = RegularGridInterpolator((K, Q), kc_fill, method="linear",
                                         bounds_error=False, fill_value=None)
        pts = np.column_stack([Kf.ravel(), Qf.ravel()])
        return interp(pts).reshape(Kf.shape)

    # Interpolate all and find shared colour range
    Z_all = {rn: _interpolate(all_data[rn]["kappa_c_map"]) for rn in ratio_names}
    vmin = min(np.nanmin(z) for z in Z_all.values())
    vmax = max(np.nanmax(z) for z in Z_all.values())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True,
                             gridspec_kw={"right": 0.88})

    cf = None
    for col, rn in enumerate(ratio_names):
        ax = axes[col]
        Z = Z_all[rn]
        cf = ax.contourf(Qf, Kf, Z, levels=20, cmap="YlGnBu_r",
                         vmin=vmin, vmax=vmax)
        cs = ax.contour(Qf, Kf, Z, levels=8, colors="k",
                        linewidths=0.5, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

        # Grid sample points
        Qg, Kg = np.meshgrid(Q, K)
        ax.scatter(Qg.ravel(), Kg.ravel(), s=8, c="k", alpha=0.3, zorder=3)

        # Mark q* — gold for G=C/G<C, bright red for G>C
        q_star = all_data[rn]["q_star_Kref"]
        if not np.isnan(q_star):
            star_color = "#FF2222" if rn == "gen_heavy" else "#FFD700"
            ax.plot(q_star, K_ref, marker="*", color=star_color,
                    markersize=12, zorder=5,
                    markeredgecolor="k", markeredgewidth=0.8)
            ax.annotate(rf"$q^*={q_star:.2f}$", (q_star, K_ref),
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=8, color=star_color, fontweight="bold")

        ax.set_xlabel("Rewiring probability $q$")
        ax.set_title(_RATIO_LABELS[rn], fontsize=_FONT)
        ax.set_yticks(K)

    axes[0].set_ylabel("Degree $k$")

    # Shared colorbar — dedicated axes to the right of all subplots
    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.76])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label(r"$\kappa_c$")

    fig.subplots_adjust(wspace=0.08, top=0.92)
    _save_fig(fig, "combined_kappa_c_heatmap")


# ====================================================================
# Public entry point
# ====================================================================

def plot_all_combined() -> None:
    """Generate all combined plots."""
    print("\n── Combined plots ──")
    plot_combined_kappa_c_map()
    plot_combined_kappa_c_heatmap()
