"""
cascade_utils.py — Cascade resilience with recovery-time experiment.

Implements a time-domain swing-equation simulation on a Watts–Strogatz
network with:
  - Edge removal (shock) at t_shock
  - Continuous failure detection (angle difference exceeding threshold)
  - Stochastic repair with stability verification
  - Service-level tracking via PCC connected-component fraction

Main entry point: simulate_cascade_recovery_ws(...)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from typing import Literal


# ====================================================================
# 1. Network & injection generation
# ====================================================================

def generate_ws_network(
    N: int, K: int, q: float, rng: np.random.Generator
) -> nx.Graph:
    """Generate a connected Watts–Strogatz graph.

    Uses networkx's connected_watts_strogatz_graph with an integer seed
    derived from the provided Generator for reproducibility.
    """
    seed_int = int(rng.integers(0, 2**31))
    return nx.connected_watts_strogatz_graph(N, K, q, seed=seed_int)


def generate_injections(
    N: int,
    gen_ratio: float,
    Pmax: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build power injection vector P for N nodes.

    Node 0 is PCC (slack bus).  Among nodes 1..N-1, gen_ratio fraction
    are generators (P > 0), rest are loads (P < 0).
    PCC balances total: P[0] = -sum(P[1:]).

    Generators draw P ~ +Uniform(0.1*Pmax, Pmax),
    Loads draw      P ~ -Uniform(0.1*Pmax, Pmax).
    """
    n_household = N - 1
    n_gen = max(1, int(round(gen_ratio * n_household)))
    n_load = n_household - n_gen

    P = np.zeros(N)
    perm = rng.permutation(np.arange(1, N))
    gen_idx = perm[:n_gen]
    load_idx = perm[n_gen:n_gen + n_load]

    P[gen_idx] = rng.uniform(0.1 * Pmax, Pmax, size=n_gen)
    P[load_idx] = -rng.uniform(0.1 * Pmax, Pmax, size=n_load)

    # PCC balances the system
    P[0] = -np.sum(P[1:])
    return P


# ====================================================================
# 2. Swing-equation integrator (piecewise, topology-aware)
# ====================================================================

def _build_adjacency(G: nx.Graph, N: int) -> np.ndarray:
    """Build NxN adjacency matrix from a networkx Graph."""
    A = np.zeros((N, N))
    for i, j in G.edges():
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def integrate_swing_segment(
    A: np.ndarray,
    P: np.ndarray,
    kappa: float,
    y0: np.ndarray,
    t_start: float,
    t_end: float,
    dt_max: float = 0.05,
    I: float = 1.0,
    D: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate swing equations on a fixed topology from t_start to t_end.

    State y = [omega_0..omega_{N-1}, theta_0..theta_{N-1}].

    Returns
    -------
    t_array : ndarray, shape (M,)
    y_array : ndarray, shape (2*N, M)
    """
    n = len(P)

    def rhs(t, y):
        omega = y[:n]
        theta = y[n:]
        diff = theta[:, None] - theta[None, :]
        coupling = np.sum(A * np.sin(diff), axis=1)
        domega = (P - D * omega - kappa * coupling) / I
        dtheta = omega
        return np.concatenate([domega, dtheta])

    if t_end <= t_start:
        return np.array([t_start]), y0.reshape(-1, 1)

    sol = solve_ivp(
        rhs, [t_start, t_end], y0,
        method="RK45", rtol=1e-6, atol=1e-6, max_step=dt_max,
        dense_output=False,
    )
    return sol.t, sol.y


# ====================================================================
# 3. Shock: pick edge to remove
# ====================================================================

def pick_shock_edge_max_load(
    G: nx.Graph,
    theta: np.ndarray,
    rng: np.random.Generator,
    kappa: float = 1.0,
) -> tuple[tuple[int, int], float]:
    """Select the edge with the largest pre-shock load (angle difference).

    Load proxy:  load_ij = |theta_i - theta_j|

    Parameters
    ----------
    G     : Current network graph.
    theta : Node phase angles at the pre-shock steady state.
    rng   : Random generator (used for tie-breaking).
    kappa : Coupling strength (informational; load printed as kappa*sin).

    Returns
    -------
    (u, v)   : The edge with maximum load.
    load_val : The angle-difference |theta_u - theta_v| of that edge.

    Raises
    ------
    ValueError : If G has no edges or theta contains NaN/inf.
    """
    if G.number_of_edges() == 0:
        raise ValueError("pick_shock_edge_max_load: graph has no edges")
    if not np.all(np.isfinite(theta)):
        raise ValueError("pick_shock_edge_max_load: theta contains NaN/inf")

    edges = list(G.edges())
    loads = np.array([abs(theta[u] - theta[v]) for u, v in edges])

    max_load = loads.max()
    if max_load == 0.0:
        # All angles identical — fallback to random edge
        print("    [max_load_edge] all loads = 0, fallback to random edge")
        idx = rng.integers(len(edges))
        return tuple(edges[idx]), 0.0

    # Tie-breaking: among all edges with max load, pick one at random
    max_mask = np.isclose(loads, max_load, rtol=1e-9)
    candidates = [edges[i] for i in np.where(max_mask)[0]]
    if len(candidates) > 1:
        idx = rng.integers(len(candidates))
        chosen = candidates[idx]
    else:
        chosen = candidates[0]

    return tuple(chosen), float(max_load)


def pick_shock_edge(
    G: nx.Graph,
    mode: Literal["betweenness", "random", "max_load_edge"],
    rng: np.random.Generator,
    theta: np.ndarray | None = None,
    kappa: float = 1.0,
) -> tuple[int, int]:
    """Select the edge to remove at t_shock.

    'betweenness'    — edge with highest betweenness centrality.
    'random'         — uniformly random edge.
    'max_load_edge'  — edge with largest |theta_i - theta_j| at steady state.
    """
    if mode == "betweenness":
        bc = nx.edge_betweenness_centrality(G)
        return max(bc, key=bc.get)
    elif mode == "max_load_edge":
        if theta is None:
            raise ValueError("max_load_edge mode requires theta (pre-shock angles)")
        edge, load_val = pick_shock_edge_max_load(G, theta, rng, kappa)
        print(f"    [max_load_edge] shock edge={edge}  "
              f"|dtheta|={load_val:.4f}  "
              f"|F|=kappa*sin={kappa * np.sin(load_val):.4f}")
        return edge
    else:
        edges = list(G.edges())
        rng.shuffle(edges)
        return tuple(edges[0])


# ====================================================================
# 4. Failure detection
# ====================================================================

def check_failures(
    G: nx.Graph,
    theta: np.ndarray,
    theta_max: float,
    fail_timers: dict[tuple[int, int], float],
    current_time: float,
    fail_duration: float,
    check_dt: float,
) -> list[tuple[int, int]]:
    """Scan all edges; return list of edges that have failed.

    An edge (i,j) fails when |theta_i - theta_j| > theta_max
    continuously for at least fail_duration seconds.

    fail_timers tracks when each edge first exceeded threshold.
    """
    newly_failed = []
    for u, v in list(G.edges()):
        edge = (min(u, v), max(u, v))  # canonical key
        angle_diff = abs(theta[u] - theta[v])

        if angle_diff > theta_max:
            if edge not in fail_timers:
                fail_timers[edge] = current_time
            elif current_time - fail_timers[edge] >= fail_duration:
                newly_failed.append(edge)
                del fail_timers[edge]
        else:
            # Reset timer if angle came back within limits
            fail_timers.pop(edge, None)

    return newly_failed


# ====================================================================
# 5. Repair scheduling & execution
# ====================================================================

def schedule_repairs(
    failed_edges: list[tuple[int, int]],
    t_fail: float,
    repair_mean: float,
    repair_dist: Literal["fixed", "exponential"],
    rng: np.random.Generator,
) -> list[dict]:
    """Create repair records for newly failed edges.

    Each record: {edge, t_fail, t_repair_ready, attempts, max_retries}.
    """
    records = []
    for edge in failed_edges:
        if repair_dist == "exponential":
            t_rep = rng.exponential(repair_mean)
        else:
            t_rep = repair_mean

        records.append({
            "edge": edge,
            "t_fail": t_fail,
            "t_repair_ready": t_fail + t_rep,
            "attempts": 0,
        })
    return records


def attempt_repairs(
    G: nx.Graph,
    repair_queue: list[dict],
    current_time: float,
    max_retries: int,
    retry_delay: float,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Check which repairs are ready and attempt to restore edges.

    Returns (restored_edges, updated_queue).
    Actual stability verification is done after restoring edges by
    running a short integration segment (handled in the main loop).
    """
    restored = []
    remaining = []

    for rec in repair_queue:
        if current_time >= rec["t_repair_ready"]:
            if rec["attempts"] < max_retries:
                # Attempt restoration
                restored.append(rec["edge"])
                rec["attempts"] += 1
                remaining.append(rec)  # keep for possible rollback
            else:
                pass  # exhausted retries, discard
        else:
            remaining.append(rec)

    return restored, remaining


# ====================================================================
# 6. Service level & recovery time
# ====================================================================

def compute_service_level(G: nx.Graph, N: int, pcc_node: int = 0) -> float:
    """S_PCC(t) = |connected component containing PCC| / N.

    If PCC is isolated (no edges), returns 1/N.
    """
    if pcc_node not in G.nodes():
        return 0.0
    cc = nx.node_connected_component(G, pcc_node)
    return len(cc) / N


def compute_recovery_time(
    t_array: np.ndarray,
    S_array: np.ndarray,
    t_shock: float,
    eps: float,
    hold_time: float,
) -> tuple[float, bool]:
    """Find recovery time T_rec from service-level timeseries.

    T_rec = first time (after S has dropped below threshold at least once)
    that S_PCC >= 1-eps continuously for hold_time.
    If S never drops, T_rec = 0 (no damage occurred).
    Returns (T_rec, unrecovered).
    """
    threshold = 1.0 - eps

    # Find first time after shock where S drops below threshold
    saw_drop = False
    drop_idx = None
    for i in range(len(t_array)):
        if t_array[i] < t_shock:
            continue
        if S_array[i] < threshold:
            saw_drop = True
            drop_idx = i
            break

    # If S never dropped, no real damage — T_rec = 0
    if not saw_drop:
        return 0.0, False

    # Search for recovery: first sustained S >= threshold after the drop
    first_good = None
    for i in range(drop_idx, len(t_array)):
        if S_array[i] >= threshold:
            if first_good is None:
                first_good = i
            if t_array[i] - t_array[first_good] >= hold_time:
                t_recover = t_array[first_good]
                return t_recover - t_shock, False
        else:
            first_good = None

    return np.nan, True


# ====================================================================
# 7. Metric computation
# ====================================================================

def compute_metrics(
    t_array: np.ndarray,
    S_array: np.ndarray,
    n_failed_array: np.ndarray,
    t_shock: float,
    eps: float,
    hold_time: float,
) -> dict:
    """Compute all output metrics from timeseries.

    Returns dict with: T_rec, A_res, E_lost_max, S_min, t_S_min, unrecovered.
    """
    T_rec, unrecovered = compute_recovery_time(
        t_array, S_array, t_shock, eps, hold_time
    )

    # A_res = integral of (1 - S_PCC) over time (trapezoidal)
    dt = np.diff(t_array)
    integrand = 1.0 - S_array
    A_res = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt)

    # Peak failed edges
    E_lost_max = int(np.max(n_failed_array))

    # Minimum service level
    idx_min = np.argmin(S_array)
    S_min = S_array[idx_min]
    t_S_min = t_array[idx_min]

    return {
        "T_rec": T_rec,
        "A_res": A_res,
        "E_lost_max": E_lost_max,
        "S_min": S_min,
        "t_S_min": t_S_min,
        "unrecovered": unrecovered,
    }


# ====================================================================
# 8. Main simulation entry point
# ====================================================================

def simulate_cascade_recovery_ws(
    K: int = 8,
    q: float = 0.15,
    seed: int = 42,
    # Time parameters
    t_max: float = 60.0,
    dt: float = 0.05,
    t_shock: float = 5.0,
    # Shock
    shock_mode: Literal["betweenness", "random", "max_load_edge"] = "betweenness",
    # Coupling
    kappa: float = 5.0,
    # Failure detection
    theta_max: float = 1.0,
    fail_duration: float = 0.5,
    check_dt: float = 0.25,
    # Repair
    repair_mean: float = 5.0,
    repair_dist: Literal["fixed", "exponential"] = "fixed",
    retry_delay: float = 2.0,
    max_retries: int = 3,
    # Recovery
    eps: float = 0.02,
    hold_time: float = 2.0,
    # Network
    N: int = 50,
    gen_ratio: float = 0.5,
    Pmax: float = 1.0,
) -> tuple[dict, dict]:
    """Run a single cascade-recovery simulation on a WS network.

    Parameters
    ----------
    K            : WS degree parameter (even integer).
    q            : WS rewiring probability.
    seed         : Random seed for full reproducibility.
    t_max        : Total simulation time (seconds).
    dt           : Integration max time step.
    t_shock      : Time at which the shock (edge removal) occurs.
    shock_mode   : 'betweenness' or 'random' edge selection.
    kappa        : Coupling strength for swing equation.
    theta_max    : Angle-difference threshold for edge failure (radians).
    fail_duration: How long angle must exceed theta_max before failure.
    check_dt     : Interval between failure/repair checks.
    repair_mean  : Mean repair time for failed edges.
    repair_dist  : 'fixed' or 'exponential' repair time distribution.
    retry_delay  : Delay before retrying a failed repair.
    max_retries  : Maximum repair attempts per edge.
    eps          : Recovery tolerance (S_PCC >= 1 - eps).
    hold_time    : S_PCC must stay above threshold for this duration.
    N            : Total number of nodes (node 0 = PCC).
    gen_ratio    : Fraction of household nodes that are generators.
    Pmax         : Maximum power injection magnitude.

    Returns
    -------
    metrics    : dict with T_rec, A_res, E_lost_max, S_min, t_S_min,
                 unrecovered, seed.
    timeseries : dict with t_array, S_array, n_failed_array, theta_history.
    """
    rng = np.random.default_rng(seed)

    # --- Setup network and injections ---
    G = generate_ws_network(N, K, q, rng)
    P = generate_injections(N, gen_ratio, Pmax, rng)

    # Initial state: small random perturbation
    y0 = np.zeros(2 * N)
    y0[N:] = rng.uniform(-0.01, 0.01, N)  # small theta perturbation

    # --- Pre-shock: integrate to near steady state ---
    A = _build_adjacency(G, N)
    t_seg, y_seg = integrate_swing_segment(
        A, P, kappa, y0, 0.0, t_shock, dt_max=dt
    )

    # Collect timeseries
    ts_t = list(t_seg)
    ts_S = [compute_service_level(G, N)] * len(t_seg)
    ts_nfail = [0] * len(t_seg)

    y_current = y_seg[:, -1]
    t_current = t_shock

    # --- Apply shock ---
    theta_pre = y_current[N:]  # pre-shock steady-state angles
    shock_edge = pick_shock_edge(G, shock_mode, rng,
                                 theta=theta_pre, kappa=kappa)
    G.remove_edge(*shock_edge)

    # Track failed edges (shock edge + cascade failures)
    all_failed_edges: set[tuple[int, int]] = set()
    canonical_shock = (min(shock_edge[0], shock_edge[1]),
                       max(shock_edge[0], shock_edge[1]))
    all_failed_edges.add(canonical_shock)

    # Repair queue and failure timers
    repair_queue: list[dict] = []
    fail_timers: dict[tuple[int, int], float] = {}

    # Schedule repair for shock edge
    repair_queue.extend(
        schedule_repairs(
            [canonical_shock], t_shock, repair_mean, repair_dist, rng
        )
    )

    # --- Post-shock: piecewise integration with checks ---
    next_check = t_shock + check_dt

    while t_current < t_max:
        # Determine next event time
        t_next = min(next_check, t_max)

        # Build current adjacency and integrate
        A = _build_adjacency(G, N)
        t_seg, y_seg = integrate_swing_segment(
            A, P, kappa, y_current, t_current, t_next, dt_max=dt
        )

        # Record timeseries at segment endpoints
        S_now = compute_service_level(G, N)
        n_fail_now = len(all_failed_edges)
        for ti in range(len(t_seg)):
            ts_t.append(t_seg[ti])
            ts_S.append(S_now)
            ts_nfail.append(n_fail_now)

        y_current = y_seg[:, -1]
        t_current = t_next

        if t_current < next_check:
            continue

        # --- Check point reached ---
        next_check = t_current + check_dt
        theta = y_current[N:]

        # 4a. Check for new failures
        newly_failed = check_failures(
            G, theta, theta_max, fail_timers, t_current, fail_duration, check_dt
        )
        for edge in newly_failed:
            if G.has_edge(*edge):
                G.remove_edge(*edge)
                all_failed_edges.add(edge)

        # Schedule repairs for newly failed edges
        if newly_failed:
            repair_queue.extend(
                schedule_repairs(
                    newly_failed, t_current, repair_mean, repair_dist, rng
                )
            )

        # 4b. Attempt repairs
        to_restore, repair_queue = attempt_repairs(
            G, repair_queue, t_current, max_retries, retry_delay
        )

        for edge in to_restore:
            # Restore edge temporarily
            G.add_edge(*edge)

        if to_restore:
            # Verify stability: short integration after restoration
            A_test = _build_adjacency(G, N)
            stable_dt = min(fail_duration * 2, 1.0)
            t_test, y_test = integrate_swing_segment(
                A_test, P, kappa, y_current,
                t_current, t_current + stable_dt, dt_max=dt
            )
            theta_test = y_test[N:, -1]

            # Check if any restored edge immediately re-fails
            rollback_edges = []
            for edge in to_restore:
                u, v = edge
                if abs(theta_test[u] - theta_test[v]) > theta_max:
                    rollback_edges.append(edge)

            for edge in rollback_edges:
                # Restoration failed: remove edge again, schedule retry
                G.remove_edge(*edge)
                for rec in repair_queue:
                    if rec["edge"] == edge and rec["attempts"] <= max_retries:
                        rec["t_repair_ready"] = t_current + retry_delay
                        break

            # For successfully restored edges, remove from failed set
            for edge in to_restore:
                if edge not in rollback_edges:
                    all_failed_edges.discard(edge)
                    # Remove completed repair record
                    repair_queue = [
                        r for r in repair_queue if r["edge"] != edge
                    ]

    # --- Build output arrays ---
    t_array = np.array(ts_t)
    S_array = np.array(ts_S)
    n_failed_array = np.array(ts_nfail)

    # Deduplicate and sort by time
    sort_idx = np.argsort(t_array)
    t_array = t_array[sort_idx]
    S_array = S_array[sort_idx]
    n_failed_array = n_failed_array[sort_idx]

    # Compute metrics
    metrics = compute_metrics(
        t_array, S_array, n_failed_array, t_shock, eps, hold_time
    )
    metrics["seed"] = seed

    timeseries = {
        "t_array": t_array,
        "S_array": S_array,
        "n_failed_array": n_failed_array,
    }

    return metrics, timeseries
