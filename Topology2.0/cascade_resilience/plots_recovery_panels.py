"""
plots_recovery_panels.py — Multi-panel figure for cascade resilience
with alpha (tolerance) as the key variable dimension.

Reads ONLY from cache/ npz files; never triggers recomputation.

Layout (4 rows × 3 columns):
  Row 1 — Panel A: Three WS network layouts (q=0, 0.2, 1.0)
  Row 2 — Panel B: Heatmaps of unrec_frac over (q, alpha), one per K
  Row 3 — Panel C: S_min vs q line plots, curves=alpha, one per K
  Row 4 — Panel D: S_PCC(t) timeseries, curves=alpha, fixed q, one per K

Usage:
    python plots_recovery_panels.py
    python plots_recovery_panels.py --Pmax_filter 3.0 --q_timeseries 0.0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BASE_DIR / "cache"
DEFAULT_OUTPUT = BASE_DIR / "output"

# ── Unified style ────────────────────────────────────────────────────
_FONT = 10
_TITLE = 12
_TICK = 8
_LW = 1.6
_DPI = 300

plt.rcParams.update({
    "font.size": _FONT,
    "axes.titlesize": _TITLE,
    "axes.labelsize": _FONT,
    "xtick.labelsize": _TICK,
    "ytick.labelsize": _TICK,
    "legend.fontsize": _TICK,
    "lines.linewidth": _LW,
    "figure.dpi": _DPI,
    "savefig.dpi": _DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# ── Colors ───────────────────────────────────────────────────────────
# Panel A node colors
_PCC_COLOR = "#E03030"
_GEN_COLOR = "#FF8C00"
_LOAD_COLOR = "#1874CD"
_EDGE_COLOR = "#C8C8C8"

# Tau (tolerance) semantic colours: lower τ → smaller threshold → red; higher → green
_TAU_COLORS = {0.2: "#CB181D", 0.3: "#FB6A4A", 0.5: "#31A354"}
_TAU_MARKERS = {0.2: "o", 0.3: "s", 0.5: "D"}

# Legacy palettes (kept for utility)
_PALETTE_BLUES = ["#BDD7E7", "#6BAED6", "#3182BD", "#08519C"]
_PALETTE_REDS = ["#FCBBA1", "#FB6A4A", "#CB181D", "#67000D"]
_PALETTE_GREENS = ["#BAE4B3", "#74C476", "#31A354", "#006D2C"]
_PALETTE_PURPLES = ["#CBC9E2", "#9E9AC8", "#756BB1", "#54278F"]


def _pick_palette(n: int, base: list[str]) -> list[str]:
    """Return n evenly-spaced colours from a base palette."""
    if n <= len(base):
        step = max(1, len(base) // n)
        return [base[min(i * step, len(base) - 1)] for i in range(n)]
    cmap = mcolors.LinearSegmentedColormap.from_list("c", base, N=n)
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# ====================================================================
# Cache helpers — 3D (K, q, alpha)
# ====================================================================

def load_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def discover_cache_files_3d(
    cache_dir: Path,
    Pmax_filter: float = 3.0,
) -> dict:
    """Scan cache_dir for 3D npz files, return {(K, q, alpha): path}.

    When multiple R exist for the same (K, q, alpha), keep the largest R.
    Filters by Pmax from meta.
    """
    pattern = re.compile(
        r"ws_K(\d+)_q([\d.]+)_a([\d.]+)_R(\d+)_seed(\d+)\.npz"
    )
    best: dict[tuple, tuple[int, Path]] = {}  # (K,q,a) -> (R, path)

    for f in sorted(cache_dir.glob("ws_K*.npz")):
        m = pattern.match(f.name)
        if not m:
            continue
        K = int(m.group(1))
        q = round(float(m.group(2)), 3)
        alpha = round(float(m.group(3)), 3)
        R = int(m.group(4))
        key = (K, q, alpha)

        if key in best and R <= best[key][0]:
            continue
        best[key] = (R, f)

    # Filter by Pmax
    result = {}
    for key, (R, path) in best.items():
        data = load_npz(path)
        meta = json.loads(str(data["meta"]))
        if abs(meta.get("Pmax", 0) - Pmax_filter) < 1e-6:
            result[key] = path

    return result


def build_grid_from_cache_3d(cache_map: dict):
    """Build 3D arrays of metrics from cache.

    Returns
    -------
    K_arr, q_arr, alpha_arr, grids
        grids: {metric_name: 3D array (nK, nQ, nA)}
    """
    Ks = sorted(set(k for k, _, _ in cache_map.keys()))
    qs = sorted(set(q for _, q, _ in cache_map.keys()))
    alphas = sorted(set(a for _, _, a in cache_map.keys()))
    nK, nQ, nA = len(Ks), len(qs), len(alphas)
    K_idx = {k: i for i, k in enumerate(Ks)}
    q_idx = {q: i for i, q in enumerate(qs)}
    a_idx = {a: i for i, a in enumerate(alphas)}

    unrec_frac = np.full((nK, nQ, nA), np.nan)
    S_min_med = np.full((nK, nQ, nA), np.nan)
    S_min_q25 = np.full((nK, nQ, nA), np.nan)
    S_min_q75 = np.full((nK, nQ, nA), np.nan)
    A_res_med = np.full((nK, nQ, nA), np.nan)
    E_max_med = np.full((nK, nQ, nA), np.nan)

    for (K, q, alpha), path in cache_map.items():
        data = load_npz(path)
        ki, qi, ai = K_idx[K], q_idx[q], a_idx[alpha]
        unrec = data["unrecovered"].astype(bool)
        unrec_frac[ki, qi, ai] = unrec.mean()
        S_min_med[ki, qi, ai] = np.median(data["S_min"])
        S_min_q25[ki, qi, ai] = np.percentile(data["S_min"], 25)
        S_min_q75[ki, qi, ai] = np.percentile(data["S_min"], 75)
        A_res_med[ki, qi, ai] = np.median(data["A_res"])
        E_max_med[ki, qi, ai] = np.median(data["E_lost_max"])

    grids = {
        "unrec_frac": unrec_frac,
        "S_min_median": S_min_med,
        "S_min_q25": S_min_q25,
        "S_min_q75": S_min_q75,
        "A_res_median": A_res_med,
        "E_lost_max_median": E_max_med,
    }
    return np.array(Ks), np.array(qs), np.array(alphas), grids


# ====================================================================
# Generic multi-line panel drawer
# ====================================================================

def _draw_multicurve(
    ax,
    x_arr: np.ndarray,
    y_med_2d: np.ndarray,
    y_lo_2d: np.ndarray | None,
    y_hi_2d: np.ndarray | None,
    curve_values: np.ndarray,
    palette: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    curve_label_fmt: str = "{}",
    legend_title: str | None = None,
    mark_unrec: np.ndarray | None = None,
    unrec_threshold: float = 0.2,
):
    """Draw a family of curves (one per row in y_med_2d)."""
    n_curves = len(curve_values)
    colors = _pick_palette(n_curves, palette)
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    for ci in range(n_curves):
        y = y_med_2d[ci]
        valid = np.isfinite(y)
        if not valid.any():
            continue

        c = colors[ci]
        m = markers[ci % len(markers)]
        label = curve_label_fmt.format(curve_values[ci])

        ax.plot(x_arr[valid], y[valid], color=c, marker=m,
                markersize=4, lw=_LW, label=label, zorder=3)

        if y_lo_2d is not None and y_hi_2d is not None:
            lo = y_lo_2d[ci]
            hi = y_hi_2d[ci]
            band_valid = valid & np.isfinite(lo) & np.isfinite(hi)
            if band_valid.any():
                ax.fill_between(x_arr[band_valid],
                                lo[band_valid], hi[band_valid],
                                color=c, alpha=0.12, zorder=1)

        if mark_unrec is not None:
            for qi in range(len(x_arr)):
                if valid[qi] and mark_unrec[ci, qi] > unrec_threshold:
                    ax.plot(x_arr[qi], y[qi], marker="x", color="red",
                            markersize=6, markeredgewidth=1.5, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=_TICK - 1, title=legend_title,
              title_fontsize=_TICK - 0.5,
              loc="best", framealpha=0.8, handlelength=1.5)


# ====================================================================
# Panel A — WS network structure examples
# ====================================================================

def draw_panel_A(
    ax_list: list, K: int = 4, N: int = 50, seed: int = 0,
):
    """Draw 3 WS networks with high-saturation node colours."""
    q_vals = [0.0, 0.2, 1.0]
    q_labels = ["$q=0$ (ring)", "$q=0.2$ (small-world)", "$q=1$ (random)"]

    rng = np.random.default_rng(seed)
    n_household = N - 1
    n_gen = n_household // 2
    perm = rng.permutation(np.arange(1, N))
    gen_set = set(perm[:n_gen])

    for idx, (ax, qv, lab) in enumerate(zip(ax_list, q_vals, q_labels)):
        G = nx.connected_watts_strogatz_graph(N, K, qv, seed=seed + idx)

        if qv == 0.0:
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=seed, k=1.8 / np.sqrt(N),
                                   iterations=100)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=_EDGE_COLOR,
                               width=0.5, alpha=0.5)

        loads = [n for n in range(1, N) if n not in gen_set]
        nx.draw_networkx_nodes(G, pos, nodelist=loads, ax=ax,
                               node_color=_LOAD_COLOR, node_size=25,
                               alpha=0.85, edgecolors="none")

        gens = [n for n in range(1, N) if n in gen_set]
        nx.draw_networkx_nodes(G, pos, nodelist=gens, ax=ax,
                               node_color=_GEN_COLOR, node_size=25,
                               alpha=0.85, edgecolors="none")

        nx.draw_networkx_nodes(G, pos, nodelist=[0], ax=ax,
                               node_color=_PCC_COLOR, node_size=120,
                               alpha=1.0, edgecolors="black",
                               linewidths=1.2, node_shape="*")

        ax.set_title(lab, fontsize=_TICK + 1, pad=4)
        ax.axis("off")

    legend_elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=_PCC_COLOR,
               markersize=12, markeredgecolor="k", markeredgewidth=0.8,
               label="PCC"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GEN_COLOR,
               markersize=7, label="Generator"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_LOAD_COLOR,
               markersize=7, label="Load"),
    ]
    ax_list[-1].legend(handles=legend_elems, loc="lower right",
                       fontsize=_TICK, frameon=True, framealpha=0.85,
                       handletextpad=0.3, borderpad=0.4)


# ====================================================================
# Panel B — Heatmaps: unrec_frac over (q, alpha), one per K
# ====================================================================

def draw_panel_B_heatmaps(ax_list, K_arr, q_arr, alpha_arr, grids):
    """Draw heatmaps of unrecovered fraction, x=q, y=alpha, one per K."""
    unrec = grids["unrec_frac"]  # (nK, nQ, nA)
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for ki, (ax, K) in enumerate(zip(ax_list, K_arr)):
        data_2d = unrec[ki, :, :].T  # (nA, nQ) — y=alpha, x=q

        im = ax.pcolormesh(
            q_arr, alpha_arr, data_2d,
            cmap=cmap, norm=norm, shading="nearest",
        )

        # Annotate cells with percentage
        for ai, a in enumerate(alpha_arr):
            for qi, q in enumerate(q_arr):
                val = data_2d[ai, qi]
                if np.isfinite(val):
                    txt = f"{val*100:.0f}%"
                    text_color = "white" if val > 0.6 or val < 0.15 else "black"
                    ax.text(q, a, txt, ha="center", va="center",
                            fontsize=_TICK - 1.5, color=text_color,
                            fontweight="bold")

        ax.set_xlabel("Rewiring $q$")
        if ki == 0:
            ax.set_ylabel(r"Tolerance $\tau$")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"$k={K}$")
        ax.set_yticks(alpha_arr)
        ax.set_xticks(q_arr[::2])

    # Shared colorbar on rightmost panel
    cbar = plt.colorbar(im, ax=ax_list, fraction=0.025, pad=0.03)
    cbar.set_label("Unrecovered fraction", fontsize=_FONT)
    cbar.ax.tick_params(labelsize=_TICK)


# ====================================================================
# Panel C — S_min vs q, curves = alpha, one per K
# ====================================================================

def draw_panel_C_smin(ax_list, K_arr, q_arr, alpha_arr, grids):
    """S_min vs q with one curve per alpha value, one subplot per K."""
    for ki, (ax, K) in enumerate(zip(ax_list, K_arr)):
        for ai, alpha in enumerate(alpha_arr):
            color = _TAU_COLORS.get(alpha, "gray")
            marker = _TAU_MARKERS.get(alpha, "o")
            y_med = grids["S_min_median"][ki, :, ai]
            y_lo = grids["S_min_q25"][ki, :, ai]
            y_hi = grids["S_min_q75"][ki, :, ai]
            valid = np.isfinite(y_med)
            if not valid.any():
                continue

            label = rf"$\tau={alpha}$"
            ax.plot(q_arr[valid], y_med[valid], color=color, marker=marker,
                    markersize=4, lw=_LW, label=label, zorder=3)

            band_valid = valid & np.isfinite(y_lo) & np.isfinite(y_hi)
            if band_valid.any():
                ax.fill_between(q_arr[band_valid],
                                y_lo[band_valid], y_hi[band_valid],
                                color=color, alpha=0.15, zorder=1)

        ax.set_xlabel("Rewiring $q$")
        if ki == 0:
            ax.set_ylabel(r"$S_{\min}$ (median)")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"$k={K}$")
        ax.set_ylim(-0.05, 1.12)
        ax.axhline(1.0, ls=":", color="grey", lw=0.4, alpha=0.4)
        ax.legend(fontsize=_TICK - 1, loc="best", framealpha=0.8,
                  title=r"Tolerance $\tau$", title_fontsize=_TICK - 0.5)


# ====================================================================
# Panel D — S_PCC(t) timeseries, curves = alpha, fixed q, one per K
# ====================================================================

def draw_panel_D_timeseries(
    ax_list, cache_map: dict,
    K_arr, alpha_arr,
    q_fixed: float = 0.0,
    t_shock: float = 5.0,
):
    """S_PCC(t) recovery curves from cached example timeseries.

    One subplot per K, one curve per alpha, all at fixed q.
    """
    q_round = round(q_fixed, 3)

    for ki, (ax, K) in enumerate(zip(ax_list, K_arr)):
        has_any = False

        for alpha in alpha_arr:
            key = (int(K), q_round, round(alpha, 3))
            if key not in cache_map:
                continue

            data = load_npz(cache_map[key])
            color = _TAU_COLORS.get(alpha, "gray")
            label = rf"$\tau={alpha}$"

            # Collect example timeseries
            ex_indices = set()
            for k in data.keys():
                m_ts = re.match(r"ts_ex(\d+)_", k)
                if m_ts:
                    ex_indices.add(int(m_ts.group(1)))

            all_t, all_S = [], []
            for ei in sorted(ex_indices):
                tk, sk = f"ts_ex{ei}_t", f"ts_ex{ei}_S"
                if tk in data and sk in data:
                    all_t.append(data[tk])
                    all_S.append(data[sk])
                    has_any = True

            if not all_t:
                # Fallback: schematic from summary metrics
                unrec = data["unrecovered"].astype(bool)
                S_min_vals = data["S_min"]
                meta = json.loads(str(data["meta"]))
                t_max = meta.get("t_max", 120.0)
                S_min_med = float(np.median(S_min_vals))

                t_pts = np.array([0, t_shock, t_shock+0.5, t_shock+2,
                                  t_shock+10, t_max])
                S_pts = np.array([1.0, 1.0, S_min_med+0.05, S_min_med,
                                  1.0 if not unrec.all() else S_min_med,
                                  1.0 if not unrec.all() else S_min_med])
                t_fine = np.linspace(0, t_max, 500)
                S_fine = np.clip(np.interp(t_fine, t_pts, S_pts), 0, 1)
                ax.plot(t_fine, S_fine, color=color, lw=_LW, label=label,
                        ls="--" if unrec.all() else "-")
                has_any = True
                continue

            # Interpolate to common grid and show median + IQR
            if len(all_t) > 1:
                t_lo = max(t.min() for t in all_t)
                t_hi = min(t.max() for t in all_t)
                t_grid = np.linspace(t_lo, t_hi, 500)
                S_interp = np.array([np.interp(t_grid, t, S)
                                     for t, S in zip(all_t, all_S)])
                S_med = np.median(S_interp, axis=0)
                S_q25 = np.percentile(S_interp, 25, axis=0)
                S_q75 = np.percentile(S_interp, 75, axis=0)
                ax.plot(t_grid, S_med, color=color, lw=_LW, label=label)
                ax.fill_between(t_grid, S_q25, S_q75, color=color, alpha=0.15)
            else:
                ax.plot(all_t[0], all_S[0], color=color, lw=_LW, label=label)

        # Annotations
        ax.axvline(t_shock, ls="--", color="grey", lw=0.8, alpha=0.6)
        ax.text(t_shock + 0.5, 0.05, "$t_{\\mathrm{shock}}$",
                fontsize=_TICK, color="grey")
        ax.axhline(1.0, ls=":", color="grey", lw=0.4, alpha=0.4)

        ax.set_xlabel("Time $t$ (s)")
        if ki == 0:
            ax.set_ylabel("$S_{\\mathrm{PCC}}(t)$")
        ax.set_title(f"$k={K}$, $q={q_fixed}$")
        ax.set_ylim(-0.05, 1.12)
        ax.legend(fontsize=_TICK - 1, loc="lower right", framealpha=0.8,
                  title=r"Tolerance $\tau$", title_fontsize=_TICK - 0.5)

        if not has_any:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_TICK, color="grey",
                    fontstyle="italic")


# ====================================================================
# Main figure assembly
# ====================================================================

def make_figure(
    cache_dir: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
    K_for_networks: int = 4,
    Pmax_filter: float = 3.0,
    q_timeseries: float = 0.0,
):
    output_dir.mkdir(exist_ok=True)
    cache_map = discover_cache_files_3d(cache_dir, Pmax_filter=Pmax_filter)

    if not cache_map:
        print(f"ERROR: No cache files found in {cache_dir}/ "
              f"with Pmax={Pmax_filter}")
        print("Run experiment_recovery_time_ws.py first.")
        return

    K_arr, q_arr, alpha_arr, grids = build_grid_from_cache_3d(cache_map)

    print(f"  Found {len(cache_map)} cache files (Pmax={Pmax_filter})")
    print(f"  K = {list(K_arr)}, q = {len(q_arr)} points, "
          f"alpha = {list(alpha_arr)}")

    # Read t_shock from any cached meta
    sample_path = next(iter(cache_map.values()))
    sample_meta = json.loads(str(load_npz(sample_path)["meta"]))
    t_shock = sample_meta.get("t_shock", 5.0)

    # ── Layout: 4 rows × 3 cols ──────────────────────────────────────
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        width_ratios=[1, 1, 1],
        height_ratios=[0.6, 0.8, 1, 1],
        hspace=0.40, wspace=0.30,
    )

    # Row 1: Panel A — 3 network diagrams
    ax_A = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_panel_A(ax_A, K=K_for_networks)

    # Row 2: Panel B — 3 heatmaps (unrec_frac)
    ax_B = [fig.add_subplot(gs[1, i]) for i in range(3)]
    draw_panel_B_heatmaps(ax_B, K_arr, q_arr, alpha_arr, grids)

    # Row 3: Panel C — 3 S_min vs q plots
    ax_C = [fig.add_subplot(gs[2, i]) for i in range(3)]
    draw_panel_C_smin(ax_C, K_arr, q_arr, alpha_arr, grids)

    # Row 4: Panel D — 3 timeseries plots
    ax_D = [fig.add_subplot(gs[3, i]) for i in range(3)]
    draw_panel_D_timeseries(
        ax_D, cache_map, K_arr, alpha_arr,
        q_fixed=q_timeseries, t_shock=t_shock,
    )

    # ── Panel labels ────────────────────────────────────────────────
    for ax, letter in zip(
        [ax_A[0], ax_B[0], ax_C[0], ax_D[0]],
        ["A", "B", "C", "D"],
    ):
        ax.text(-0.06, 1.07, f"({letter})",
                transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    # ── Save ────────────────────────────────────────────────────────
    stem = "cascade_alpha_panels"
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)


# ====================================================================
# CLI
# ====================================================================

def cli():
    parser = argparse.ArgumentParser(
        description="Generate multi-panel cascade figure with alpha dimension.",
    )
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--K_for_networks", type=int, default=4,
                        help="K for Panel A network layouts")
    parser.add_argument("--Pmax_filter", type=float, default=3.0,
                        help="Filter cache files by Pmax value")
    parser.add_argument("--q_timeseries", type=float, default=0.0,
                        help="Fixed q for Panel D timeseries")

    args = parser.parse_args()

    make_figure(
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        K_for_networks=args.K_for_networks,
        Pmax_filter=args.Pmax_filter,
        q_timeseries=args.q_timeseries,
    )


if __name__ == "__main__":
    cli()
