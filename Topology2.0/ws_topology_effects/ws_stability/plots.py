"""
ws_plots.py — Plotting routines for WS topology effects study.

All functions read from cache/{ratio}.pkl and write to output/.
No network generation, simulation, or numerical computation here.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator

from ws_config import (
    K_list, q_list, K_ref, OUTPUT_DIR, CACHE_DIR,
)

# ── Unified style ────────────────────────────────────────────────
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
# 1. κ_c visualisation — line plot (primary) + smoothed heatmap
# ====================================================================

# Per-K colour palette (colour-blind friendly, consistent across plots)
_K_COLORS = {
    4:  "#1b9e77",   # teal
    6:  "#d95f02",   # orange
    8:  "#7570b3",   # purple
    10: "#e7298a",   # pink
    12: "#66a61e",   # green
}
_K_MARKERS = {4: "o", 6: "s", 8: "D", 10: "^", 12: "v"}


def _k_color(K):
    return _K_COLORS.get(K, "gray")


def _k_marker(K):
    return _K_MARKERS.get(K, "o")


def plot_kappa_c_map(ratio_name: str) -> None:
    """Generate two κ_c figures:

    1. **Line plot** (primary): κ_c(q) for each K, with ±σ bands and q* marker.
       → kappa_c_map_{ratio}.png/pdf
    2. **Smoothed heatmap**: interpolated contourf with NaN regions masked grey.
       → kappa_c_heatmap_{ratio}.png/pdf
    """
    data = _load_cache(ratio_name)
    kc_mean = data["kappa_c_map"]     # (nK, nQ)
    kc_std = data["kappa_c_std"]      # (nK, nQ)
    q_star = data["q_star_Kref"]

    Q = np.array(q_list)
    K = np.array(K_list)

    # ── (a) Line plot ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 3.5))

    for ki, Kv in enumerate(K):
        mean = kc_mean[ki]
        std = kc_std[ki]
        valid = ~np.isnan(mean)
        if not valid.any():
            continue
        c = _k_color(Kv)
        m = _k_marker(Kv)
        ax.plot(Q[valid], mean[valid], f"{m}-", color=c, markersize=5,
                label=f"$K={Kv}$")
        ax.fill_between(Q[valid], (mean - std)[valid], (mean + std)[valid],
                        color=c, alpha=0.15)

    # Mark q* with red star
    if not np.isnan(q_star) and K_ref in K_list:
        ki_ref = K_list.index(K_ref)
        qi_ref = int(np.argmin(np.abs(Q - q_star)))
        kc_star = kc_mean[ki_ref, qi_ref]
        if not np.isnan(kc_star):
            ax.plot(q_star, kc_star + 0.3, marker="*", color="red",
                    markersize=15, zorder=5)
            ax.annotate(rf"$q^*={q_star:.2f}$",
                        (q_star, kc_star + 0.3),
                        textcoords="offset points", xytext=(10, 5),
                        fontsize=9, color="red", fontweight="bold")

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel(r"Critical coupling $\overline{\kappa}_c$")
    ax.set_title(rf"$\overline{{\kappa}}_c$ vs $q$ — {ratio_name}")
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0)
    ax.set_xlim(Q[0], Q[-1])
    ax.set_ylim(bottom=0)

    fig.subplots_adjust(right=0.78)
    _save_fig(fig, f"kappa_c_map_{ratio_name}")

    # ── (b) Smoothed heatmap ─────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))

    # Interpolate onto a fine grid for smooth contours
    kc_masked = np.ma.masked_invalid(kc_mean)

    # Fine grids
    q_fine = np.linspace(Q[0], Q[-1], 200)
    k_fine = np.linspace(K[0], K[-1], 200)
    Qf, Kf = np.meshgrid(q_fine, k_fine)

    # Use RegularGridInterpolator (nearest for NaN-heavy regions)
    # Fill NaN with nearest valid for interpolation
    kc_fill = kc_mean.copy()
    for ki in range(len(K)):
        row = kc_fill[ki]
        valid = ~np.isnan(row)
        if valid.any():
            kc_fill[ki] = np.interp(Q, Q[valid], row[valid])

    interp = RegularGridInterpolator((K, Q), kc_fill, method="linear",
                                     bounds_error=False, fill_value=None)
    pts = np.column_stack([Kf.ravel(), Qf.ravel()])
    Zf = interp(pts).reshape(Kf.shape)

    # NaN mask: grey out regions where original data was all NaN
    nan_mask_orig = np.all(np.isnan(kc_mean), axis=1)  # K rows fully NaN

    cf = ax2.contourf(Qf, Kf, Zf, levels=20, cmap="YlGnBu_r")
    cs = ax2.contour(Qf, Kf, Zf, levels=8, colors="k", linewidths=0.5, alpha=0.6)
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    cb = fig2.colorbar(cf, ax=ax2, pad=0.02)
    cb.set_label(r"$\overline{\kappa}_c$")

    # Show original grid points; mark NaN as grey ×
    Qg, Kg = np.meshgrid(Q, K)
    valid_pts = ~np.isnan(kc_mean)
    ax2.scatter(Qg[valid_pts], Kg[valid_pts], s=8, c="k", alpha=0.4, zorder=3)
    ax2.scatter(Qg[~valid_pts], Kg[~valid_pts], s=20, c="grey", marker="x",
                alpha=0.6, zorder=3, label="NaN (no convergence)")

    # Mark q*
    if not np.isnan(q_star) and K_ref in K_list:
        ax2.plot(q_star, K_ref, marker="*", color="red", markersize=14, zorder=5)
        ax2.annotate(rf"$q^*={q_star:.2f}$", (q_star, K_ref),
                     textcoords="offset points", xytext=(10, 8),
                     fontsize=9, color="red", fontweight="bold")

    ax2.set_xlabel("Rewiring probability $q$")
    ax2.set_ylabel("Degree $k$")
    ax2.set_title(rf"Critical coupling $\overline{{\kappa}}_c$ — {ratio_name}")
    ax2.set_yticks(K)
    if (~valid_pts).any():
        ax2.legend(loc="upper right", fontsize=8, markerscale=1.2)

    _save_fig(fig2, f"kappa_c_heatmap_{ratio_name}")


# ====================================================================
# 2. Lorenz curves (q = 0, q*, 1)
# ====================================================================

def plot_lorenz_curves(ratio_name: str) -> None:
    """Lorenz curves of |F_e| for three representative q values."""
    data = _load_cache(ratio_name)
    lorenz_list = data["lorenz_curves"]   # list of (frac, share) per q index
    q_star = data["q_star_Kref"]
    Q = np.array(q_list)

    # Pick indices: q=0, q≈q*, q=1
    idx_0 = 0
    idx_1 = len(Q) - 1
    if not np.isnan(q_star):
        idx_star = int(np.argmin(np.abs(Q - q_star)))
    else:
        idx_star = len(Q) // 2

    # Avoid duplicates while preserving order
    picks = []
    seen = set()
    for idx, label in [(idx_0, f"$q = {Q[idx_0]:.2f}$"),
                       (idx_star, rf"$q = q^* = {Q[idx_star]:.2f}$"),
                       (idx_1, f"$q = {Q[idx_1]:.2f}$")]:
        if idx not in seen:
            picks.append((idx, label))
            seen.add(idx)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Equality line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect equality")

    for ci, (idx, label) in enumerate(picks):
        frac, share = lorenz_list[idx]
        ax.plot(frac, share, color=colors[ci], label=label)

    ax.set_xlabel("Cumulative fraction of edges")
    ax.set_ylabel("Cumulative share of $|F_e|$")
    ax.set_title(rf"Lorenz curves ($K={K_ref}$) — {ratio_name}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    _save_fig(fig, f"lorenz_{ratio_name}")


# ====================================================================
# 3. Gini vs q
# ====================================================================

def plot_gini_vs_q(ratio_name: str) -> None:
    """Gini coefficient of |F_e| as a function of q, with q* marked."""
    data = _load_cache(ratio_name)
    gini_data = data["gini_vs_q"]
    q_arr = gini_data["q_list"]
    gini_mean = gini_data["gini_mean"]
    gini_std = gini_data["gini_std"]
    q_star = data["q_star_Kref"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(q_arr, gini_mean, "o-", color="#1f77b4", markersize=4)
    ax.fill_between(q_arr, gini_mean - gini_std, gini_mean + gini_std,
                    color="#1f77b4", alpha=0.2)

    if not np.isnan(q_star):
        ax.axvline(q_star, ls="--", color="red", lw=1.0, label=rf"$q^* = {q_star:.2f}$")
        ax.legend()

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel("Gini coefficient")
    ax.set_title(rf"Flow concentration ($K={K_ref}$) — {ratio_name}")

    _save_fig(fig, f"gini_vs_q_{ratio_name}")


# ====================================================================
# 4. Cascade size vs q
# ====================================================================

def plot_cascade_size_vs_q(ratio_name: str) -> None:
    """Mean cascade failure size S(q) with error band and q* marker."""
    data = _load_cache(ratio_name)
    cas = data["cascade_size_vs_q"]
    q_arr = cas["q_list"]
    s_mean = cas["cascade_size_mean"]
    s_std = cas["cascade_size_std"]
    q_star = data["q_star_Kref"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(q_arr, s_mean, "s-", color="#2ca02c", markersize=4)
    ax.fill_between(q_arr, s_mean - s_std, s_mean + s_std,
                    color="#2ca02c", alpha=0.2)

    if not np.isnan(q_star):
        ax.axvline(q_star, ls="--", color="red", lw=1.0, label=rf"$q^* = {q_star:.2f}$")
        ax.legend()

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel("Cascade size $S$")
    ax.set_title(rf"Cascade failure ($K={K_ref}$) — {ratio_name}")
    ax.set_ylim(bottom=0)

    _save_fig(fig, f"cascade_size_vs_q_{ratio_name}")
