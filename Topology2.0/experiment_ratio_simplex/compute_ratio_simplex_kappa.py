"""
compute_ratio_simplex_kappa.py — Sweep (rg, rc, rp) on the ternary simplex
for WS, RGG, SBM, and Core-Periphery topologies.

Reuses _integrate_swing and _find_kappa_c from ws_compute.py.
"""

import sys
import time
import json
import csv
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

# ── Import κ_c machinery from ws_compute.py ─────────────────────
WS_DIR = Path(__file__).resolve().parent.parent / "ws_topology_effects"
sys.path.insert(0, str(WS_DIR))
from ws_compute import _integrate_swing, _steady_state_residual  # noqa: E402

# ── Directories ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
FIGURES_DIR = BASE_DIR / "figures"
CACHE_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ── Robust κ_c bisection (uses small init instead of random) ─────

def _find_kappa_c(A: np.ndarray, P: np.ndarray, n: int,
                  kappa_start: float = 10.0, step_init: float = 0.2,
                  tol: float = 1e-3) -> float:
    """Bisection search for critical coupling κ_c.

    Tries multiple initial conditions for robust convergence on
    heterogeneous network topologies like core-periphery.
    """
    kappa = kappa_start

    # Try several initial conditions at kappa_start
    inits = [
        np.zeros(2 * n),                                    # zero
        np.concatenate([np.zeros(n), 0.01 * np.random.randn(n)]),  # small θ
        np.concatenate([np.zeros(n), 0.001 * np.random.randn(n)]), # tiny θ
    ]

    converged = False
    y_last = inits[0]
    for y0 in inits:
        converged, y_last = _integrate_swing(A, P, n, kappa, y0, t_max=300.0)
        if converged:
            break

    if not converged:
        return np.nan

    stepsize = step_init
    kappa_old = kappa

    while True:
        converged, y_sol = _integrate_swing(A, P, n, kappa, y_last, t_max=100.0)
        if converged:
            y_last = y_sol
            if stepsize < tol:
                return kappa
            kappa_old = kappa
            kappa -= stepsize
        else:
            stepsize /= 2.0
            kappa = kappa_old - stepsize

        if kappa < 0 or stepsize < 1e-6:
            return kappa_old


# ── Constants ────────────────────────────────────────────────────
N = 50              # total nodes (PCC + 49 households)
PCC_NODE = 0
N_HOUSEHOLDS = 49
K_TARGET = 4        # target mean degree on household subgraph
PMAX = 1.0
FAMILIES = ["WS", "RGG", "SBM", "CP"]
FAMILY_IDX = {f: i for i, f in enumerate(FAMILIES)}
# Per-family κ bisection start (CP needs higher due to heterogeneous degree)
KAPPA_START = {"WS": 10.0, "RGG": 15.0, "SBM": 15.0, "CP": 30.0}


# ====================================================================
# 1. Simplex ratio grid
# ====================================================================

def build_ratio_grid(step: float = 0.1) -> list:
    """Generate all (rg, rc, rp) tuples on the ternary simplex with given step.

    Returns list of tuples (rg, rc, rp) where rg + rc + rp ≈ 1.
    Excludes points where rg=0 or rc=0 (need at least 1 gen and 1 consumer).
    """
    pts = []
    # Use integer multiples to avoid float issues
    n_steps = round(1.0 / step)
    for ig in range(1, n_steps + 1):          # rg >= step (at least 1 gen)
        for ic in range(1, n_steps + 1 - ig): # rc >= step (at least 1 con)
            ip = n_steps - ig - ic
            if ip < 0:
                continue
            rg = ig / n_steps
            rc = ic / n_steps
            rp = ip / n_steps
            pts.append((round(rg, 4), round(rc, 4), round(rp, 4)))
    return pts


def ratios_to_counts(rg: float, rc: float, rp: float,
                     n: int = N_HOUSEHOLDS) -> tuple:
    """Convert ratio triple to integer counts summing to n.

    Returns (ng, nc, np_count).
    """
    ng = max(1, round(rg * n))
    nc = max(1, round(rc * n))
    np_count = n - ng - nc
    if np_count < 0:
        # Reduce the larger of ng, nc
        excess = -np_count
        if ng >= nc:
            ng -= excess
        else:
            nc -= excess
        np_count = 0
    return ng, nc, np_count


# ====================================================================
# 2. Network generators (each returns nx.Graph on N nodes, PCC=node 0)
# ====================================================================

def household_mean_degree(G: nx.Graph) -> float:
    """Mean degree of the induced subgraph on household nodes 1..49."""
    H = G.subgraph(range(1, N))
    if H.number_of_nodes() == 0:
        return 0.0
    return 2 * H.number_of_edges() / H.number_of_nodes()


def _ensure_connected(G: nx.Graph, max_retries: int = 50) -> nx.Graph:
    """If G is disconnected, add edges between components to make it connected."""
    if nx.is_connected(G):
        return G
    components = list(nx.connected_components(G))
    # Connect all components to the one containing PCC (or largest)
    pcc_comp = None
    for c in components:
        if PCC_NODE in c:
            pcc_comp = c
            break
    if pcc_comp is None:
        pcc_comp = max(components, key=len)

    for c in components:
        if c is pcc_comp:
            continue
        # Add edge between a random node in pcc_comp and a random node in c
        u = min(pcc_comp)
        v = min(c)
        G.add_edge(u, v)
        pcc_comp = pcc_comp | c
    return G


def generate_ws(seed: int, n: int = N, k: int = 4, q: float = 0.1) -> nx.Graph:
    """Watts-Strogatz on 49 household nodes + PCC connected to 2 random households."""
    rng = np.random.default_rng(seed)
    # Generate WS on 49 nodes (will be relabelled to 1..49)
    G_household = nx.connected_watts_strogatz_graph(n - 1, k, q, seed=int(rng.integers(2**31)))
    mapping = {i: i + 1 for i in range(n - 1)}
    G = nx.relabel_nodes(G_household, mapping)
    # Add PCC node 0 connected to 2 random household nodes
    G.add_node(PCC_NODE)
    hh = list(range(1, n))
    chosen = rng.choice(hh, size=min(2, len(hh)), replace=False)
    for v in chosen:
        G.add_edge(PCC_NODE, v)
    return G


def generate_rgg(seed: int, n: int = N, k_target: int = K_TARGET) -> nx.Graph:
    """Random geometric graph; binary-search radius to match mean degree ~ k_target."""
    rng = np.random.default_rng(seed)
    # Place n-1 household nodes uniformly in [0,1]^2
    pos = rng.uniform(0, 1, size=(n - 1, 2))

    # Binary search for radius
    r_lo, r_hi = 0.01, 1.0
    best_G = None
    best_diff = float('inf')

    for _ in range(40):
        r = (r_lo + r_hi) / 2
        dists = squareform(pdist(pos))
        adj = (dists < r).astype(float)
        np.fill_diagonal(adj, 0)
        G_hh = nx.from_numpy_array(adj)
        # Relabel 0..48 → 1..49
        mapping = {i: i + 1 for i in range(n - 1)}
        G_hh = nx.relabel_nodes(G_hh, mapping)

        md = household_mean_degree(nx.Graph(G_hh))  # just household part
        diff = md - k_target
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_G = G_hh.copy()
            best_pos = pos.copy()
        if diff < 0:
            r_lo = r
        else:
            r_hi = r
        if abs(diff) < 0.1:
            break

    G = best_G
    # Add PCC node at center, connected to 2 nearest households
    G.add_node(PCC_NODE)
    center = np.array([0.5, 0.5])
    dists_to_center = np.linalg.norm(best_pos - center, axis=1)
    nearest = np.argsort(dists_to_center)[:2] + 1  # +1 for relabelling
    for v in nearest:
        G.add_edge(PCC_NODE, int(v))

    # Store positions for plotting
    G.graph['pos'] = {0: (0.5, 0.5)}
    for i in range(n - 1):
        G.graph['pos'][i + 1] = tuple(best_pos[i])

    G = _ensure_connected(G)
    return G


def generate_sbm(seed: int, n: int = N, k_target: int = K_TARGET,
                 m: int = 4, eta: float = 8.0) -> nx.Graph:
    """Stochastic block model with m blocks; search edge-probability scale for mean degree ~ k_target.

    Within-block probability = s, between-block = s / eta.
    """
    rng = np.random.default_rng(seed)
    n_hh = n - 1  # 49 household nodes

    # Assign households to blocks
    block_sizes = [n_hh // m] * m
    for i in range(n_hh % m):
        block_sizes[i] += 1

    # Binary search for scale s
    s_lo, s_hi = 0.001, 1.0
    best_G = None
    best_diff = float('inf')

    for _ in range(40):
        s = (s_lo + s_hi) / 2
        p_in = min(s, 1.0)
        p_out = min(s / eta, 1.0)
        # Build probability matrix
        probs = [[p_out] * m for _ in range(m)]
        for i in range(m):
            probs[i][i] = p_in

        G_hh = nx.stochastic_block_model(block_sizes, probs,
                                          seed=int(rng.integers(2**31)))
        # Remove self-loops
        G_hh.remove_edges_from(nx.selfloop_edges(G_hh))
        # Relabel 0..48 → 1..49
        mapping = {i: i + 1 for i in range(n_hh)}
        G_hh = nx.relabel_nodes(G_hh, mapping)

        md = household_mean_degree(G_hh)
        diff = md - k_target
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_G = G_hh.copy()
        if diff < 0:
            s_lo = s
        else:
            s_hi = s
        if abs(diff) < 0.1:
            break

    G = best_G
    # Store block assignment for each household node (for plotting)
    offset = 1
    for b in range(m):
        for i in range(block_sizes[b]):
            G.nodes[offset + i]["block"] = b
        offset += block_sizes[b]

    # Add PCC connected to one node in each of the first 2 blocks
    G.add_node(PCC_NODE)
    offset = 1
    for b in range(min(2, m)):
        G.add_edge(PCC_NODE, offset)
        offset += block_sizes[b]

    G = _ensure_connected(G)
    return G


def generate_cp(seed: int, n: int = N, k_target: int = K_TARGET,
                f_core: float = 0.2) -> nx.Graph:
    """Core-periphery network; search p_cc (core-core prob) for mean degree ~ k_target.

    n_core = floor(f_core * (n-1)) nodes form a dense core.
    Ratio: p_cp = p_cc * 0.3, p_pp = p_cc * 0.05.
    Moderate heterogeneity to keep synchronisation tractable.
    """
    rng = np.random.default_rng(seed)
    n_hh = n - 1
    n_core = max(2, int(f_core * n_hh))
    n_periph = n_hh - n_core

    # Binary search for p_cc
    p_lo, p_hi = 0.01, 1.0
    best_G = None
    best_diff = float('inf')

    for _ in range(40):
        p_cc = (p_lo + p_hi) / 2
        p_cp = p_cc * 0.3
        p_pp = p_cc * 0.05

        sizes = [n_core, n_periph]
        probs = [[p_cc, p_cp], [p_cp, p_pp]]
        G_hh = nx.stochastic_block_model(sizes, probs,
                                          seed=int(rng.integers(2**31)))
        G_hh.remove_edges_from(nx.selfloop_edges(G_hh))
        mapping = {i: i + 1 for i in range(n_hh)}
        G_hh = nx.relabel_nodes(G_hh, mapping)

        md = household_mean_degree(G_hh)
        diff = md - k_target
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_G = G_hh.copy()
        if diff < 0:
            p_lo = p_cc
        else:
            p_hi = p_cc
        if abs(diff) < 0.1:
            break

    G = best_G
    # Store core node list for plotting
    G.graph["core_nodes"] = list(range(1, n_core + 1))

    # PCC connects to one core and one periphery node
    G.add_node(PCC_NODE)
    G.add_edge(PCC_NODE, 1)           # first core node
    G.add_edge(PCC_NODE, n_core + 1)  # first periphery node

    G = _ensure_connected(G)
    return G


# Dispatcher
GENERATORS = {
    "WS":  generate_ws,
    "RGG": generate_rgg,
    "SBM": generate_sbm,
    "CP":  generate_cp,
}


# ====================================================================
# 3. Role assignment
# ====================================================================

def assign_roles(ng: int, nc: int, n_households: int = N_HOUSEHOLDS,
                 seed: int = 0) -> np.ndarray:
    """Assign power injection vector P (length N).

    P[0] = 0 (PCC).
    Among 49 household nodes: ng generators (+Pmax/ng), nc consumers (-Pmax/nc),
    rest are passive (0). sum(P) = 0.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros(N)
    perm = rng.permutation(np.arange(1, N))
    gen_nodes = perm[:ng]
    con_nodes = perm[ng:ng + nc]
    # passive_nodes are the rest — P stays 0

    P[gen_nodes] = PMAX / ng
    P[con_nodes] = -PMAX / nc
    return P


# ====================================================================
# 4. Main computation loop
# ====================================================================

def run_experiment(R: int = 30, step: float = 0.1,
                   kappa_start: float = None) -> None:
    """Main loop: for each (ratio, family, realisation) → find κ_c.

    kappa_start: if None, uses per-family defaults from KAPPA_START dict.
    Saves raw_results.csv, agg_results.csv, and metadata.json to cache/.
    """
    ratio_grid = build_ratio_grid(step=step)
    n_ratios = len(ratio_grid)
    n_families = len(FAMILIES)
    total_tasks = n_ratios * n_families * R
    print(f"Simplex grid: {n_ratios} ratio points, {n_families} families, "
          f"{R} realisations = {total_tasks} tasks")

    raw_path = CACHE_DIR / "raw_results.csv"
    agg_path = CACHE_DIR / "agg_results.csv"
    meta_path = CACHE_DIR / "metadata.json"

    # Check for existing progress (resume support)
    done_keys = set()
    if raw_path.exists():
        with open(raw_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["rg"], row["rc"], row["rp"],
                       row["family"], row["realisation"])
                done_keys.add(key)
        print(f"Resuming: {len(done_keys)} tasks already completed")

    # Open CSV for appending
    write_header = not raw_path.exists() or len(done_keys) == 0
    raw_file = open(raw_path, "a", newline="")
    raw_writer = csv.writer(raw_file)
    if write_header:
        raw_writer.writerow(["rg", "rc", "rp", "ng", "nc", "np",
                             "family", "realisation", "seed",
                             "household_mean_deg", "kappa_c", "connected"])
        raw_file.flush()

    t_start = time.time()
    done = len(done_keys)

    for ri, (rg, rc, rp) in enumerate(ratio_grid):
        ng, nc, np_count = ratios_to_counts(rg, rc, rp)
        for fi, family in enumerate(FAMILIES):
            gen_func = GENERATORS[family]
            for r in range(R):
                key = (str(rg), str(rc), str(rp), family, str(r))
                if key in done_keys:
                    continue

                seed = ri * 1000 + fi * 100 + r
                np.random.seed(seed)

                # Generate network
                G = gen_func(seed=seed)
                md = household_mean_degree(G)
                connected = nx.is_connected(G)

                if not connected:
                    # Retry up to 10 times
                    for retry in range(1, 11):
                        G = gen_func(seed=seed + retry * 10000)
                        G = _ensure_connected(G)
                        connected = nx.is_connected(G)
                        if connected:
                            break

                # Build adjacency and power vector
                A = nx.to_numpy_array(G, nodelist=range(N))
                P = assign_roles(ng, nc, seed=seed + 50000)

                # Find κ_c (per-family start or user override)
                ks = kappa_start if kappa_start is not None else KAPPA_START[family]
                kc = _find_kappa_c(A, P, N, kappa_start=ks)

                raw_writer.writerow([rg, rc, rp, ng, nc, np_count,
                                     family, r, seed, f"{md:.3f}",
                                     f"{kc:.6f}" if not np.isnan(kc) else "NaN",
                                     int(connected)])
                raw_file.flush()

                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total_tasks - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{total_tasks}] "
                          f"ratio=({rg},{rc},{rp}) family={family} r={r} "
                          f"κ_c={kc:.4f} md={md:.2f} "
                          f"ETA={eta/60:.1f}min")

    raw_file.close()
    print(f"\nRaw results saved → {raw_path}")

    # ── Aggregate ────────────────────────────────────────────────
    _aggregate_results(raw_path, agg_path)

    # ── Metadata ─────────────────────────────────────────────────
    meta = {
        "N": N, "K_TARGET": K_TARGET, "PMAX": PMAX,
        "R": R, "step": step,
        "kappa_start_override": kappa_start,
        "kappa_start_per_family": KAPPA_START,
        "n_ratios": n_ratios, "n_families": n_families,
        "total_tasks": total_tasks,
        "families": FAMILIES,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved → {meta_path}")


def _aggregate_results(raw_path: Path, agg_path: Path) -> None:
    """Aggregate raw CSV → (rg, rc, rp, family) mean/std of κ_c."""
    from collections import defaultdict
    groups = defaultdict(list)

    with open(raw_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["rg"], row["rc"], row["rp"], row["family"])
            kc = row["kappa_c"]
            if kc != "NaN":
                groups[key].append(float(kc))

    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rg", "rc", "rp", "family",
                          "kappa_c_mean", "kappa_c_std", "n_valid"])
        for (rg, rc, rp, family), vals in sorted(groups.items()):
            if vals:
                writer.writerow([rg, rc, rp, family,
                                  f"{np.mean(vals):.6f}",
                                  f"{np.std(vals):.6f}",
                                  len(vals)])
            else:
                writer.writerow([rg, rc, rp, family, "NaN", "NaN", 0])

    print(f"Aggregated results saved → {agg_path}")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Ratio-simplex κ_c experiment across 4 topology families")
    parser.add_argument("--R", type=int, default=30,
                        help="Number of realisations per (ratio, family)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Simplex grid step size")
    parser.add_argument("--kappa-start", type=float, default=None,
                        help="Initial κ for bisection (overrides per-family defaults)")
    args = parser.parse_args()

    run_experiment(R=args.R, step=args.step, kappa_start=args.kappa_start)
