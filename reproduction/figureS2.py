"""
Reproduce Supplementary Figure S2 (cascade duration vs normalized edge capacity).

Python reimplementation aligned with the author workflow in:
- reference_code/GridResilience/scripts/responsetimes.jl
- reference_code/GridResilience/src/swing.jl (swingcascadeandtvalpha / swingfracturewithtime!)
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcache_gw2"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / "cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class SweepConfig:
    panel: str
    variable_name: str
    values: tuple[float, ...]
    n: int | None = None
    gamma: float | None = None
    kappa: float | None = None
    q: float | None = None
    y_max: float = 30.0
    cmap: str = "Reds"


def build_incidence(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    m = len(edges)
    e = np.zeros((n, m), dtype=float)
    for j, (u, v) in enumerate(edges):
        e[u, j] = 1.0
        e[v, j] = -1.0
    return e


def adjacency_from_incidence(e: np.ndarray) -> np.ndarray:
    deg = np.sum(np.abs(e), axis=1)
    lap = e @ e.T
    return np.diag(deg) - lap


def fsteadystate(theta: np.ndarray, a: np.ndarray, p: np.ndarray, kappa: float) -> np.ndarray:
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(a * np.sin(diff), axis=1)
    return p - kappa * coupling


def edgepower(theta: np.ndarray, e: np.ndarray, kappa: float) -> np.ndarray:
    return kappa * np.sin(e.T @ theta)


def fswing_rhs(_t: float, y: np.ndarray, a: np.ndarray, p: np.ndarray, i_inertia: float, d_damp: float, kappa: float) -> np.ndarray:
    n = p.size
    omega = y[:n]
    theta = y[n:]
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(a * np.sin(diff), axis=1)
    domega = (p - d_damp * omega - kappa * coupling) / i_inertia
    dtheta = omega
    return np.concatenate([domega, dtheta])


def connected_components_from_active(n: int, edges: list[tuple[int, int]], active_mask: np.ndarray, node_subset: np.ndarray | None = None) -> list[np.ndarray]:
    g = nx.Graph()
    if node_subset is None:
        nodes = np.arange(n, dtype=int)
    else:
        nodes = np.asarray(node_subset, dtype=int)
    g.add_nodes_from(nodes.tolist())
    node_set = set(nodes.tolist())
    for ei, (u, v) in enumerate(edges):
        if not active_mask[ei]:
            continue
        if u in node_set and v in node_set:
            g.add_edge(u, v)
    return [np.array(sorted(comp), dtype=int) for comp in nx.connected_components(g)]


def build_local_incidence(edges: list[tuple[int, int]], active_mask: np.ndarray, nodeset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_index = {node: i for i, node in enumerate(nodeset.tolist())}
    edge_ids: list[int] = []
    for ei, (u, v) in enumerate(edges):
        if not active_mask[ei]:
            continue
        if u in node_index and v in node_index:
            edge_ids.append(ei)

    n_sub = len(nodeset)
    m_sub = len(edge_ids)
    e1 = np.zeros((n_sub, m_sub), dtype=float)
    for j, ei in enumerate(edge_ids):
        u, v = edges[ei]
        e1[node_index[u], j] = 1.0
        e1[node_index[v], j] = -1.0
    return np.array(edge_ids, dtype=int), e1


def simulate_until_event(
    psi0: np.ndarray,
    p1: np.ndarray,
    a1: np.ndarray,
    e1: np.ndarray,
    kappa: float,
    d_damp: float,
    alpha: float,
    maxflow: float,
    synctol: float,
) -> tuple[np.ndarray, float, bool, bool]:
    # Approximate Julia DiscreteCallback by sampling fine t_grid.
    t_grid = np.linspace(0.0, 500.0, 1001)
    sol = solve_ivp(
        fun=lambda t, y: fswing_rhs(t, y, a1, p1, 1.0, d_damp, kappa),
        t_span=(0.0, 500.0),
        y0=psi0,
        t_eval=t_grid,
        rtol=1e-8,
        atol=1e-8,
        max_step=1.0,
        method="RK45",
    )
    if not sol.success:
        return psi0, 0.0, False, False

    n = p1.size
    y_hist = sol.y
    t_hist = sol.t
    sync_ok = True
    tripped = False
    k_hit = len(t_hist) - 1
    for k in range(1, len(t_hist)):
        omega = y_hist[:n, k]
        theta = y_hist[n:, k]
        if np.linalg.norm(omega, 2) > synctol:
            sync_ok = False
            k_hit = k
            break
        flow = edgepower(theta, e1, kappa) / maxflow
        if np.any(np.abs(flow) > alpha):
            tripped = True
            k_hit = k
            break
        resid = fsteadystate(theta, a1, p1, kappa)
        if np.linalg.norm(resid, 2) < 1e-6:
            k_hit = k
            break

    psi_end = y_hist[:, k_hit]
    return psi_end, float(t_hist[k_hit]), sync_ok, tripped


def swing_fracture_with_time(
    edges: list[tuple[int, int]],
    active_mask: np.ndarray,
    psi: np.ndarray,
    nodeset: np.ndarray,
    p_global: np.ndarray,
    synctol: float,
    alpha: float,
    kappa: float,
    maxflow: float,
    d_damp: float,
) -> tuple[int, float]:
    tol = 1e-5
    p1 = p_global[nodeset].astype(float).copy()
    source_count = int(np.sum(p1 > tol))
    sink_count = int(np.sum(p1 < -tol))
    if source_count == 0 or sink_count == 0:
        return 0, 0.0

    # Balance P1 homogeneously (same as Julia version).
    delta = float(np.sum(p1) / 2.0)
    p1[p1 < -tol] -= delta / sink_count
    p1[p1 > tol] -= delta / source_count

    edge_ids, e1 = build_local_incidence(edges, active_mask, nodeset)
    edge_cardinality = len(edge_ids)
    if edge_cardinality == 0:
        return 0, 0.0

    a1 = adjacency_from_incidence(e1)
    psi_end, t_local, sync_ok, tripped = simulate_until_event(
        psi0=psi,
        p1=p1,
        a1=a1,
        e1=e1,
        kappa=kappa,
        d_damp=d_damp,
        alpha=alpha,
        maxflow=maxflow,
        synctol=synctol,
    )
    if not sync_ok:
        return 0, 0.0
    if not tripped:
        return edge_cardinality, t_local

    n_sub = len(nodeset)
    omega_end = psi_end[:n_sub]
    theta_end = psi_end[n_sub:]
    flow = edgepower(theta_end, e1, kappa) / maxflow
    overloaded_local = np.abs(flow) > alpha
    for loc, overloaded in enumerate(overloaded_local):
        if overloaded:
            active_mask[edge_ids[loc]] = False

    comps = connected_components_from_active(
        n=max(max(u, v) for (u, v) in edges) + 1,
        edges=edges,
        active_mask=active_mask,
        node_subset=nodeset,
    )
    if not comps:
        return 0, t_local

    desc_edges = 0
    child_times: list[float] = []
    node_to_pos = {node: i for i, node in enumerate(nodeset.tolist())}
    for comp in comps:
        idx = np.array([node_to_pos[node] for node in comp], dtype=int)
        psi_comp = np.concatenate([omega_end[idx], theta_end[idx]])
        e_surv, t_surv = swing_fracture_with_time(
            edges=edges,
            active_mask=active_mask,
            psi=psi_comp,
            nodeset=comp,
            p_global=p_global,
            synctol=synctol,
            alpha=alpha,
            kappa=kappa,
            maxflow=maxflow,
            d_damp=d_damp,
        )
        desc_edges += e_surv
        child_times.append(t_surv)
    return desc_edges, t_local + (max(child_times) if child_times else 0.0)


def assign_source_sink_vector(n: int, ns: int, nd: int, rng: np.random.Generator) -> np.ndarray:
    p = np.zeros(n, dtype=float)
    perm = rng.permutation(n)
    src = perm[:ns]
    snk = perm[ns : ns + nd]
    if ns > 0:
        p[src] = 1.0 / ns
    if nd > 0:
        p[snk] = -1.0 / nd
    return p


def connected_ws_network(n: int, k: int, q: float, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.connected_watts_strogatz_graph(n, k, q, seed=seed)


def run_single_ensemble(
    *,
    n: int,
    ns: int,
    nd: int,
    q: float,
    k: int,
    alpha_values: np.ndarray,
    i_inertia: float,
    d_damp: float,
    kappa: float,
    ensemble_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(ensemble_seed)
    g = connected_ws_network(n, k, q, rng)
    edges = list(g.edges())
    m = len(edges)
    e = build_incidence(n, edges)
    a = adjacency_from_incidence(e)
    p = assign_source_sink_vector(n, ns, nd, rng)
    psi0 = rng.random(2 * n)

    sol = solve_ivp(
        fun=lambda t, y: fswing_rhs(t, y, a, p, i_inertia, d_damp, kappa),
        t_span=(0.0, 100.0),
        y0=psi0,
        rtol=1e-8,
        atol=1e-8,
        method="RK45",
        max_step=1.0,
    )
    if not sol.success:
        return np.zeros_like(alpha_values), np.zeros_like(alpha_values), np.nan

    psi_ss = sol.y[:, -1]
    omega = psi_ss[:n]
    theta = psi_ss[n:]
    resid = fsteadystate(theta, a, p, kappa)
    if np.linalg.norm(resid, 2) > 1e-3:
        # Keep going to match Julia behavior (warn but continue).
        pass

    flow = edgepower(theta, e, kappa)
    fmax = float(np.max(np.abs(flow)))
    if fmax <= 1e-12:
        return np.zeros_like(alpha_values), np.zeros_like(alpha_values), fmax
    d_idx = int(np.argmax(np.abs(flow)))

    redges = np.zeros_like(alpha_values, dtype=float)
    rtimes = np.zeros_like(alpha_values, dtype=float)
    for i, alpha in enumerate(alpha_values):
        active_mask = np.ones(m, dtype=bool)
        active_mask[d_idx] = False

        comps = connected_components_from_active(n=n, edges=edges, active_mask=active_mask)
        tot_edges = 0
        times: list[float] = []
        for comp in comps:
            idx = comp
            psi_subset = np.concatenate([omega[idx], theta[idx]])
            e_surv, t_surv = swing_fracture_with_time(
                edges=edges,
                active_mask=active_mask,
                psi=psi_subset,
                nodeset=comp,
                p_global=p,
                synctol=3.0,
                alpha=float(alpha),
                kappa=kappa,
                maxflow=fmax,
                d_damp=d_damp,
            )
            tot_edges += e_surv
            times.append(t_surv)
        redges[i] = tot_edges / float(m)
        rtimes[i] = max(times) if times else 0.0
    return redges, rtimes, fmax


def _ensemble_worker(payload: tuple[int, dict]) -> tuple[int, tuple[np.ndarray, np.ndarray, float]]:
    idx, kwargs = payload
    return idx, run_single_ensemble(**kwargs)


def swingcascade_t_vs_alpha(
    *,
    ensemble_size: int,
    n: int,
    q: float,
    d_damp: float,
    kappa: float,
    alpha_values: np.ndarray,
    seed: int,
    n_workers: int,
    k: int = 4,
    i_inertia: float = 1.0,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ns = int(round(0.2 * n))
    nd = n - ns

    payloads = []
    for z in range(ensemble_size):
        payloads.append(
            dict(
                n=n,
                ns=ns,
                nd=nd,
                q=float(q),
                k=k,
                alpha_values=alpha_values,
                i_inertia=float(i_inertia),
                d_damp=float(d_damp),
                kappa=float(kappa),
                ensemble_seed=int(seed + 1_000_003 * z + 7_919 * n + int(100 * q)),
            )
        )

    results: list[tuple[np.ndarray, np.ndarray, float]] = [None] * ensemble_size  # type: ignore[assignment]
    indexed_payloads = list(enumerate(payloads))
    if n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_ensemble_worker, indexed_payloads, chunksize=1)
            for idx, result in tqdm(
                it,
                total=ensemble_size,
                desc=progress_desc or "Ensembles",
                leave=False,
            ):
                results[idx] = result
    else:
        for item in tqdm(
            indexed_payloads,
            total=ensemble_size,
            desc=progress_desc or "Ensembles",
            leave=False,
        ):
            idx, result = _ensemble_worker(item)
            results[idx] = result

    edgemat = np.stack([r[0] for r in results], axis=0)
    tmat = np.stack([r[1] for r in results], axis=0)
    fmax_vec = np.array([r[2] for r in results], dtype=float)
    return np.nanmean(edgemat, axis=0), np.nanmean(tmat, axis=0), fmax_vec


def panel_configs() -> dict[str, SweepConfig]:
    return {
        "A": SweepConfig(
            panel="A",
            variable_name="n",
            values=tuple(float(x) for x in np.arange(45, 105, 5)),
            gamma=1.0,
            kappa=1.0,
            q=0.1,
            y_max=30.0,
            cmap="Reds",
        ),
        "B": SweepConfig(
            panel="B",
            variable_name="gamma",
            values=tuple(float(x) for x in np.round(np.arange(0.3, 2.1 + 1e-9, 0.1), 1)),
            n=50,
            kappa=1.0,
            q=0.1,
            y_max=80.0,
            cmap="Blues",
        ),
        "C": SweepConfig(
            panel="C",
            variable_name="kappa",
            values=tuple(float(x) for x in np.round(np.arange(0.1, 1.1 + 1e-9, 0.1), 1)),
            n=50,
            gamma=1.0,
            q=0.1,
            y_max=60.0,
            cmap="Greys",
        ),
        "D": SweepConfig(
            panel="D",
            variable_name="q",
            values=tuple(float(x) for x in np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)),
            n=50,
            gamma=1.0,
            kappa=1.0,
            y_max=40.0,
            cmap="PuRd",
        ),
    }


def fmt_value(x: float) -> str:
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".")


def cache_file_name(
    panel: str,
    variable_name: str,
    value: float,
    ensemble_size: int,
    alpha_min: float,
    alpha_max: float,
    alpha_res: int,
    seed: int,
) -> Path:
    v = fmt_value(value).replace(".", "p")
    return CACHE_DIR / (
        f"figS2_{panel}_{variable_name}{v}_e{ensemble_size}_"
        f"a{fmt_value(alpha_min).replace('.', 'p')}-{fmt_value(alpha_max).replace('.', 'p')}_"
        f"r{alpha_res}_s{seed}.npz"
    )


def compute_or_load_curves(
    *,
    config: SweepConfig,
    ensemble_size: int,
    alpha_values: np.ndarray,
    seed: int,
    n_workers: int,
    use_cache: bool,
    mode: str,
) -> list[dict]:
    curves: list[dict] = []
    for i, v in enumerate(
        tqdm(
            config.values,
            desc=f"Panel {config.panel} ({config.variable_name})",
            leave=False,
        )
    ):
        cfile = cache_file_name(
            panel=config.panel,
            variable_name=config.variable_name,
            value=v,
            ensemble_size=ensemble_size,
            alpha_min=float(alpha_values[0]),
            alpha_max=float(alpha_values[-1]),
            alpha_res=len(alpha_values),
            seed=seed,
        )

        if mode in {"plot"}:
            if not cfile.exists():
                raise FileNotFoundError(
                    f"Missing cache for panel {config.panel} value={v}: {cfile}"
                )
            data = np.load(cfile)
            curves.append({"value": v, "alpha": data["alpha"], "tbar": data["tbar"]})
            continue

        if use_cache and cfile.exists():
            data = np.load(cfile)
            curves.append({"value": v, "alpha": data["alpha"], "tbar": data["tbar"]})
            print(f"[{config.panel}] loaded cache {cfile.name}")
            continue

        if config.panel == "A":
            n = int(v)
            gamma = float(config.gamma)
            kappa = float(config.kappa)
            q = float(config.q)
        elif config.panel == "B":
            n = int(config.n)
            gamma = float(v)
            kappa = float(config.kappa)
            q = float(config.q)
        elif config.panel == "C":
            n = int(config.n)
            gamma = float(config.gamma)
            kappa = float(v)
            q = float(config.q)
        else:
            n = int(config.n)
            gamma = float(config.gamma)
            kappa = float(config.kappa)
            q = float(v)

        print(
            f"[{config.panel}] {config.variable_name}={fmt_value(v)} "
            f"(n={n}, gamma={gamma}, kappa={kappa}, q={q}, ensembles={ensemble_size})"
        )
        _, tbar, fmax_vec = swingcascade_t_vs_alpha(
            ensemble_size=ensemble_size,
            n=n,
            q=q,
            d_damp=gamma,
            kappa=kappa,
            alpha_values=alpha_values,
            seed=seed + 10_007 * i + ord(config.panel),
            n_workers=n_workers,
            progress_desc=f"P{config.panel}:{config.variable_name}={fmt_value(v)}",
        )
        np.savez(
            cfile,
            alpha=alpha_values,
            tbar=tbar,
            value=v,
            mean_fmax=np.nanmean(fmax_vec),
        )
        curves.append({"value": v, "alpha": alpha_values.copy(), "tbar": tbar})
        print(f"[{config.panel}] saved cache {cfile.name}")
    return curves


def plot_panel(ax: plt.Axes, config: SweepConfig, curves: list[dict]) -> None:
    values = [c["value"] for c in curves]
    order = np.argsort(values)
    cmap = plt.get_cmap(config.cmap)
    n = len(curves)
    for rank, idx in enumerate(order):
        c = curves[idx]
        color = cmap(0.25 + 0.65 * rank / max(1, n - 1))
        ax.plot(c["alpha"], c["tbar"], color=color, lw=1.3)

    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.0, config.y_max)
    ax.set_xlabel(r"$\alpha/\alpha_{\ast}$")
    ax.set_ylabel(r"$\overline{T}$", rotation=0, labelpad=10)
    ax.grid(alpha=0.25)
    ax.set_title(config.panel, loc="left", fontweight="bold")

    # Compact legend labels similar to the supplementary figure style.
    vals_sorted = sorted(values)
    if len(vals_sorted) > 8:
        shown = [vals_sorted[-1], vals_sorted[-3], vals_sorted[-5], vals_sorted[2], vals_sorted[0]]
    else:
        shown = vals_sorted[::-1]
    labels = ", ".join(fmt_value(x) for x in shown)
    ax.text(
        1.02,
        0.95,
        f"{config.variable_name}: {labels}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )


def parse_panels(panels_str: str) -> list[str]:
    valid = {"A", "B", "C", "D"}
    panels = [p.strip().upper() for p in panels_str.split(",") if p.strip()]
    if not panels:
        return ["A", "B", "C", "D"]
    bad = [p for p in panels if p not in valid]
    if bad:
        raise ValueError(f"Invalid panels: {bad}. Use subset of A,B,C,D.")
    return panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Supplementary Figure S2.")
    parser.add_argument("--ensemble-size", type=int, default=50)
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=2.5)
    parser.add_argument("--alpha-res", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=max(1, (mp.cpu_count() - 2)))
    parser.add_argument("--mode", choices=["compute", "plot", "both"], default="both")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", action="store_true", help="Disable cache reads/writes.")
    parser.add_argument("--panels", type=str, default="A,B,C,D", help="Subset, e.g. A,B")
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_DIR / "figureS2.png"),
        help="Output image path (.png). A .pdf is also saved next to it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cache = args.use_cache and (not args.no_cache)
    panels = parse_panels(args.panels)
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_res)
    cfgs = panel_configs()

    panel_curves: dict[str, list[dict]] = {}
    if args.mode in {"compute", "both", "plot"}:
        for p in panels:
            panel_curves[p] = compute_or_load_curves(
                config=cfgs[p],
                ensemble_size=args.ensemble_size,
                alpha_values=alpha_values,
                seed=args.seed,
                n_workers=max(1, args.n_workers),
                use_cache=use_cache,
                mode=args.mode,
            )

    if args.mode == "compute":
        print("Compute finished. Use --mode plot or --mode both to render figure.")
        return

    # Always render a 2x2 layout; for skipped panels we leave blank axes.
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    grid = [["A", "B"], ["C", "D"]]
    for r in range(2):
        for c in range(2):
            pid = grid[r][c]
            ax = axes[r, c]
            if pid in panels:
                plot_panel(ax, cfgs[pid], panel_curves[pid])
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, f"Panel {pid} skipped", ha="center", va="center")

    fig.tight_layout()
    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # 对于打包的程序很重要
    
    main()
