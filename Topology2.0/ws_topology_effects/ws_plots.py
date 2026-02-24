"""
ws_plots.py — Plotting routines for WS topology effects study.

All functions read from cache/{ratio}.pkl and write to output/.
No network generation, simulation, or numerical computation here.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
# 1. κ_c phase map (heatmap + contour)
# ====================================================================

def plot_kappa_c_map(ratio_name: str) -> None:
    """Heatmap of κ_c(q, K) with contour lines and q* marker."""
    data = _load_cache(ratio_name)
    kc = data["kappa_c_map"]          # shape (nK, nQ)
    q_star = data["q_star_Kref"]

    Q = np.array(q_list)
    K = np.array(K_list)

    fig, ax = plt.subplots(figsize=(7, 4))

    # pcolormesh expects edges; build cell-edge arrays
    dq = (Q[1] - Q[0]) / 2 if len(Q) > 1 else 0.5
    dk = (K[1] - K[0]) / 2 if len(K) > 1 else 1
    q_edges = np.concatenate([[Q[0] - dq], Q + dq])
    k_edges = np.concatenate([[K[0] - dk], K + dk])

    pcm = ax.pcolormesh(q_edges, k_edges, kc, shading="flat", cmap="YlGnBu_r")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\overline{\kappa}_c$")

    # Contour overlay (interpolated on grid centres)
    Qg, Kg = np.meshgrid(Q, K)
    cs = ax.contour(Qg, Kg, kc, levels=6, colors="k", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # Mark q* at K_ref
    if not np.isnan(q_star) and K_ref in K_list:
        ki = K_list.index(K_ref)
        ax.plot(q_star, K_ref, marker="*", color="red", markersize=12, zorder=5)
        ax.annotate(
            rf"$q^*={q_star:.2f}$",
            (q_star, K_ref),
            textcoords="offset points", xytext=(8, 8),
            fontsize=9, color="red",
        )

    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel("Degree $K$")
    ax.set_title(rf"Critical coupling $\overline{{\kappa}}_c$ — {ratio_name}")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    _save_fig(fig, f"kappa_c_map_{ratio_name}")


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
