"""
ws_compute.py — Computation routines for WS topology effects study.

Implements:
  1. WS network generation (50 nodes, node 0 = PCC)
  2. Power vector construction with ratio-based gen/load assignment
  3. Critical coupling κ_c via swing-equation bisection
  4. DC power flow → edge-flow Lorenz curves & Gini coefficients
  5. Cascading failure simulation
  6. Per-ratio .pkl caching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from ws_config import (
    N, PCC_NODE, HOUSEHOLD_NODES,
    K_list, q_list, gamma, kappa_grid, realizations,
    K_ref, alpha,
    RATIO_CONFIGS,
    stable_seed,
)

# ── 本模块的 cache / output 目录 ──
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ====================================================================
# 1. Network generation
# ====================================================================

def generate_ws_network(N: int, K: int, q: float, seed: int | None = None) -> nx.Graph:
    """Generate a connected Watts-Strogatz small-world graph.

    Parameters
    ----------
    N : int   Number of nodes.
    K : int   Each node is connected to K nearest neighbours in ring topology.
    q : float Rewiring probability (0 = ring, 1 ≈ random).
    seed : int, optional  Random seed for reproducibility.

    Returns
    -------
    nx.Graph
    """
    return nx.connected_watts_strogatz_graph(N, K, q, seed=seed)


# ====================================================================
# 2. Power vector
# ====================================================================

def build_power_vector(ratio_name: str, seed: int | None = None) -> np.ndarray:
    """Build a power injection vector P for all N nodes.

    P[PCC_NODE] = 0 (slack bus / reference).
    Among the 49 household nodes, n_gen are randomly chosen as generators
    (P > 0), the rest as loads (P < 0).  Power is balanced: sum(P) = 0.

    Parameters
    ----------
    ratio_name : str  Key into RATIO_CONFIGS.
    seed : int, optional

    Returns
    -------
    np.ndarray, shape (N,)
    """
    cfg = RATIO_CONFIGS[ratio_name]
    n_gen = cfg["n_gen"]
    n_load = cfg["n_load"]
    Pmax = cfg["Pmax"]

    rng = np.random.default_rng(seed)
    P = np.zeros(N)

    # Randomly assign roles among household nodes
    perm = rng.permutation(HOUSEHOLD_NODES)
    gen_idx = perm[:n_gen]
    load_idx = perm[n_gen:n_gen + n_load]

    # P_gen = +Pmax / n_gen,  P_load = -Pmax / n_load  →  sum = 0
    P[gen_idx] = Pmax / n_gen
    P[load_idx] = -Pmax / n_load
    # PCC stays at 0
    return P


# ====================================================================
# 3. Swing-equation solver & κ_c bisection
# ====================================================================

def _steady_state_residual(theta: np.ndarray, A: np.ndarray,
                           P: np.ndarray, kappa: float) -> np.ndarray:
    """Power-balance residual: P_i - κ Σ_j A_ij sin(θ_i − θ_j)."""
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def _integrate_swing(A: np.ndarray, P: np.ndarray, n: int,
                     kappa: float, y0: np.ndarray,
                     I: float = 1.0, D: float = 1.0,
                     t_max: float = 200.0):
    """Integrate second-order swing equation.

    State y = [ω_1..ω_n, θ_1..θ_n].
    Returns (converged: bool, y_final: ndarray).
    """
    def rhs(t, y):
        omega = y[:n]
        theta = y[n:]
        diff = theta[:, None] - theta[None, :]
        coupling = np.sum(A * np.sin(diff), axis=1)
        domega = (P - D * omega - kappa * coupling) / I
        dtheta = omega
        return np.concatenate([domega, dtheta])

    sol = solve_ivp(rhs, [0, t_max], y0, method='RK45',
                    rtol=1e-8, atol=1e-8, max_step=1.0)
    if sol.status != 0:
        return False, y0

    y_final = sol.y[:, -1]
    theta_final = y_final[n:]
    resid = _steady_state_residual(theta_final, A, P, kappa)
    converged = np.linalg.norm(resid, 2) < 1e-5
    return converged, y_final


def _find_kappa_c(A: np.ndarray, P: np.ndarray, n: int,
                  kappa_start: float = 10.0, step_init: float = 0.2,
                  tol: float = 1e-3) -> float:
    """Bisection search for critical coupling κ_c.

    Starts at high κ, warm-starts downward, halves step on failure.
    Returns κ_c or NaN.
    """
    kappa = kappa_start
    y0 = np.random.rand(2 * n)
    converged, y_last = _integrate_swing(A, P, n, kappa, y0, t_max=200.0)
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


def _find_kappa_c_lower(A: np.ndarray, P: np.ndarray, n: int,
                         tol: float = 1e-3) -> tuple:
    """自下而上搜索下临界耦合 κ_c^low（最小同步耦合）。

    Phase 1 - Bracket: κ 从 0.005 开始翻倍，直到找到第一个稳定点。
    Phase 2 - Bisection: 在 [κ_fail, κ_ok] 间二分收敛。

    Returns
    -------
    (kappa_c_low, y_stable) : (float, ndarray or None)
        κ_c^low 和对应的稳态解（用于 warm-start upper search）。
        如果始终不收敛则返回 (NaN, None)。
    """
    kappa_max_bracket = 30.0
    n_ic_tries = 3  # 每个 κ 尝试多个随机 IC

    # Phase 1: Bracket — 指数增长找到第一个稳定点
    kappa = 0.005
    kappa_fail = 0.0
    kappa_ok = None
    y_ok = None

    while kappa <= kappa_max_bracket:
        found = False
        for ic_trial in range(n_ic_tries):
            y0 = np.random.rand(2 * n) * 0.1  # 小扰动初始条件
            converged, y_sol = _integrate_swing(A, P, n, kappa, y0, t_max=200.0)
            if converged:
                kappa_ok = kappa
                y_ok = y_sol
                found = True
                break
        if found:
            break
        kappa_fail = kappa
        kappa *= 2.0

    if kappa_ok is None:
        return np.nan, None

    # 如果第一个试的 κ=0.005 就稳定了，把 kappa_fail 设为 0
    if kappa_fail == 0.0 and kappa_ok == 0.005:
        # 再往下试一下
        converged_low, _ = _integrate_swing(A, P, n, 0.001,
                                             np.random.rand(2 * n) * 0.1,
                                             t_max=200.0)
        if converged_low:
            # 极低 κ 也能收敛，κ_c^low ≈ 0
            return 0.001, y_ok
        kappa_fail = 0.001

    # Phase 2: Bisection — warm-start from κ_ok 的稳态
    while (kappa_ok - kappa_fail) > tol:
        kappa_mid = (kappa_fail + kappa_ok) / 2.0
        converged, y_sol = _integrate_swing(A, P, n, kappa_mid, y_ok, t_max=150.0)
        if converged:
            kappa_ok = kappa_mid
            y_ok = y_sol
        else:
            # 也尝试随机 IC
            converged2, y_sol2 = _integrate_swing(A, P, n, kappa_mid,
                                                   np.random.rand(2 * n) * 0.1,
                                                   t_max=200.0)
            if converged2:
                kappa_ok = kappa_mid
                y_ok = y_sol2
            else:
                kappa_fail = kappa_mid

    return kappa_ok, y_ok


def _find_kappa_c_upper(A: np.ndarray, P: np.ndarray, n: int,
                         kappa_low: float, y_stable: np.ndarray,
                         tol: float = 1e-3,
                         kappa_max: float = 30.0) -> float:
    """自上而下搜索上临界耦合 κ_c^high（过耦合失稳点）。

    从已知稳定的 kappa_low 出发，逐步增大 κ 直到失败。

    Returns
    -------
    float : κ_c^high，如果 κ_max 仍稳定则返回 NaN（无上界）。
    """
    # Phase 1: 从稳定点出发，翻倍增大找第一个失败点
    kappa = max(kappa_low * 2.0, 0.1)
    kappa_last_ok = kappa_low
    y_last_ok = y_stable.copy()
    kappa_first_fail = None

    while kappa <= kappa_max:
        converged, y_sol = _integrate_swing(A, P, n, kappa, y_last_ok, t_max=150.0)
        if converged:
            kappa_last_ok = kappa
            y_last_ok = y_sol
            kappa *= 2.0
        else:
            kappa_first_fail = kappa
            break

    if kappa_first_fail is None:
        # 在 kappa_max 处再确认一次
        converged, y_sol = _integrate_swing(A, P, n, kappa_max, y_last_ok,
                                             t_max=200.0)
        if converged:
            return np.nan  # 无上界
        kappa_first_fail = kappa_max

    # Phase 2: Bisection
    while (kappa_first_fail - kappa_last_ok) > tol:
        kappa_mid = (kappa_last_ok + kappa_first_fail) / 2.0
        converged, y_sol = _integrate_swing(A, P, n, kappa_mid, y_last_ok,
                                             t_max=150.0)
        if converged:
            kappa_last_ok = kappa_mid
            y_last_ok = y_sol
        else:
            kappa_first_fail = kappa_mid

    return kappa_first_fail


def _compute_single_task(args):
    """单个 (K, q, realization) 的 κ_c 计算任务（用于多进程并行）。"""
    ki, K, qi, q, r, ratio_name, base_seed = args
    seed = base_seed + ki * 10000 + qi * 100 + r
    np.random.seed(seed)

    G = generate_ws_network(N, K, q, seed=seed)
    A = nx.to_numpy_array(G)
    P = build_power_vector(ratio_name, seed=seed + 1)

    kc_low, y_stable = _find_kappa_c_lower(A, P, N)

    kc_high = np.nan
    if not np.isnan(kc_low) and y_stable is not None:
        kc_high = _find_kappa_c_upper(A, P, N, kc_low, y_stable)

    return ki, qi, r, kc_low, kc_high


def compute_kappa_c_map(ratio_name: str):
    """Compute critical coupling κ_c for each (K, q) pair.

    使用双向二分搜索：
    - _find_kappa_c_lower(): 自下而上找最小同步耦合 κ_c^low
    - _find_kappa_c_upper(): 自上而下找过耦合失稳 κ_c^high

    使用 multiprocessing 并行加速。

    Returns
    -------
    kappa_c_low_mean, kappa_c_low_std, kappa_c_high_mean, kappa_c_high_std
        各为 ndarray, shape (len(K_list), len(q_list))
    """
    nK = len(K_list)
    nQ = len(q_list)

    base_seed = stable_seed(ratio_name, "kappa_c")

    # 构建所有任务
    tasks = []
    for ki, K in enumerate(K_list):
        for qi, q in enumerate(q_list):
            for r in range(realizations):
                tasks.append((ki, K, qi, q, r, ratio_name, base_seed))

    total = len(tasks)
    print(f"  Total tasks: {total} ({nK}K × {nQ}q × {realizations}r)")

    # 并行计算
    n_workers = max(1, cpu_count() - 1)
    print(f"  Using {n_workers} workers")

    kc_low_all = np.full((nK, nQ, realizations), np.nan)
    kc_high_all = np.full((nK, nQ, realizations), np.nan)

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_compute_single_task, tasks),
            total=total,
            desc=f"κ_c map [{ratio_name}]",
            unit="task"
        ))

    for ki, qi, r, kc_low, kc_high in results:
        kc_low_all[ki, qi, r] = kc_low
        kc_high_all[ki, qi, r] = kc_high

    # 聚合统计
    kc_low_mean = np.full((nK, nQ), np.nan)
    kc_low_std = np.full((nK, nQ), np.nan)
    kc_high_mean = np.full((nK, nQ), np.nan)
    kc_high_std = np.full((nK, nQ), np.nan)

    for ki in range(nK):
        for qi in range(nQ):
            vals_low = kc_low_all[ki, qi]
            valid_low = vals_low[~np.isnan(vals_low)]
            if len(valid_low) > 0:
                kc_low_mean[ki, qi] = np.mean(valid_low)
                kc_low_std[ki, qi] = np.std(valid_low)

            vals_high = kc_high_all[ki, qi]
            valid_high = vals_high[~np.isnan(vals_high)]
            if len(valid_high) > 0:
                kc_high_mean[ki, qi] = np.mean(valid_high)
                kc_high_std[ki, qi] = np.std(valid_high)

    return kc_low_mean, kc_low_std, kc_high_mean, kc_high_std


# ====================================================================
# 4. DC power flow → Lorenz curves & Gini coefficient
# ====================================================================

def _solve_dc_power_flow(G: nx.Graph, P: np.ndarray):
    """Solve DC-like power flow: L θ = P with PCC as reference (θ₀ = 0).

    Works correctly even when G is a subgraph with fewer than N nodes
    (e.g. after cascade disconnections).  Nodes not in G get θ = 0.

    Returns
    -------
    theta : ndarray, shape (N,)
    edge_flows : dict  {(i, j): F_ij}  where F_ij = θ_i − θ_j
    """
    nodes = sorted(G.nodes())
    n_sub = len(nodes)
    node2idx = {v: i for i, v in enumerate(nodes)}

    # Build Laplacian for this (possibly reduced) node set
    L = np.zeros((n_sub, n_sub))
    for i, j in G.edges():
        ii, jj = node2idx[i], node2idx[j]
        L[ii, jj] -= 1.0
        L[jj, ii] -= 1.0
        L[ii, ii] += 1.0
        L[jj, jj] += 1.0

    P_sub = np.array([P[v] for v in nodes])

    # Impose θ_PCC = 0 by removing PCC row/col
    pcc_local = node2idx.get(PCC_NODE)
    if pcc_local is None:
        # PCC not in this subgraph — pick first node as reference
        pcc_local = 0

    keep = np.arange(n_sub) != pcc_local
    L_red = L[np.ix_(keep, keep)]
    P_red = P_sub[keep]

    theta_sub = np.zeros(n_sub)
    if L_red.size > 0:
        theta_sub[keep] = np.linalg.solve(L_red, P_red)

    # Map back to global indexing
    theta = np.zeros(N)
    for v, i in node2idx.items():
        theta[v] = theta_sub[i]

    edge_flows = {}
    for i, j in G.edges():
        edge_flows[(i, j)] = theta[i] - theta[j]

    return theta, edge_flows


def _lorenz_curve(values: np.ndarray):
    """Compute the Lorenz curve for a 1-D array of non-negative values.

    Returns
    -------
    cum_fraction : ndarray, shape (len(values)+1,)  — x-axis [0, 1]
    cum_share    : ndarray, shape (len(values)+1,)  — y-axis [0, 1]
    """
    s = np.sort(values)
    n = len(s)
    cum = np.cumsum(s)
    total = cum[-1] if cum[-1] > 0 else 1.0
    cum_share = np.concatenate([[0.0], cum / total])
    cum_fraction = np.linspace(0, 1, n + 1)
    return cum_fraction, cum_share


def _gini(values: np.ndarray) -> float:
    """Gini coefficient from a 1-D array of non-negative values."""
    s = np.sort(values)
    n = len(s)
    if n == 0 or s.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * s) / (n * s.sum())) - (n + 1) / n


def compute_lorenz_and_gini(ratio_name: str) -> dict:
    """Compute Lorenz curves and Gini coefficients of edge-flow magnitude
    vs. rewiring probability q, at fixed K = K_ref.

    For each q the result is averaged over *realizations* network instances.

    Returns
    -------
    dict with keys:
        "q_list"        : ndarray (nQ,)
        "gini_mean"     : ndarray (nQ,)
        "gini_std"      : ndarray (nQ,)
        "lorenz_curves" : list of (cum_fraction, cum_share) per q
                          (from a single representative realisation)
    """
    nQ = len(q_list)
    gini_mean = np.zeros(nQ)
    gini_std = np.zeros(nQ)
    lorenz_curves = []

    base_seed = stable_seed(ratio_name, "lorenz")

    for qi, q in tqdm(list(enumerate(q_list)),
                      desc=f"Lorenz/Gini [{ratio_name}]", unit="q"):
        ginis = []
        representative_lorenz = None

        for r in range(realizations):
            seed = base_seed + qi * 100 + r
            G = generate_ws_network(N, K_ref, q, seed=seed)
            P = build_power_vector(ratio_name, seed=seed + 1)
            _, edge_flows = _solve_dc_power_flow(G, P)

            abs_flows = np.abs(list(edge_flows.values()))
            ginis.append(_gini(abs_flows))

            if r == 0:
                representative_lorenz = _lorenz_curve(abs_flows)

        gini_mean[qi] = np.mean(ginis)
        gini_std[qi] = np.std(ginis)
        lorenz_curves.append(representative_lorenz)

    return {
        "q_list": np.array(q_list),
        "gini_mean": gini_mean,
        "gini_std": gini_std,
        "lorenz_curves": lorenz_curves,
    }


# ====================================================================
# 5. Cascading failure
# ====================================================================

def _simulate_cascade(G: nx.Graph, P: np.ndarray, alpha_tol: float) -> float:
    """Simulate a single cascade event.

    1. Solve DC power flow to get initial edge flows |F_e|₀.
    2. Set capacity C_e = (1 + alpha_tol) * |F_e|₀.
    3. Remove the edge with the largest |F_e|₀.
    4. Re-solve DC flow; remove all edges where |F_e| > C_e.
    5. Repeat until no new overloads.
    6. Cascade size S = fraction of edges removed (including the trigger).

    Returns
    -------
    float  Fraction of edges removed.
    """
    G_work = G.copy()
    total_edges = G_work.number_of_edges()
    if total_edges == 0:
        return 0.0

    # Initial power flow
    _, flows0 = _solve_dc_power_flow(G_work, P)
    abs_flows0 = {e: abs(f) for e, f in flows0.items()}

    # Edge capacities (set once, never updated)
    capacity = {e: (1.0 + alpha_tol) * abs_flows0[e] for e in abs_flows0}

    # Step 1: remove the most-loaded edge
    trigger_edge = max(abs_flows0, key=abs_flows0.get)
    G_work.remove_edge(*trigger_edge)
    removed = {trigger_edge}

    # Iterative cascade
    while True:
        # Check connectivity — only solve on the largest connected component
        if not nx.is_connected(G_work):
            largest_cc = max(nx.connected_components(G_work), key=len)
            # If PCC is isolated the grid is lost
            if PCC_NODE not in largest_cc:
                removed = set(G.edges()) - set(G_work.edges())
                break
            G_work = G_work.subgraph(largest_cc).copy()

        if G_work.number_of_edges() == 0:
            break

        _, flows_new = _solve_dc_power_flow(G_work, P)

        overloaded = []
        for e, f in flows_new.items():
            # Use canonical edge key (sorted tuple) for capacity lookup
            e_key = e if e in capacity else (e[1], e[0])
            if e_key in capacity and abs(f) > capacity[e_key]:
                overloaded.append(e)

        if not overloaded:
            break

        for e in overloaded:
            if G_work.has_edge(*e):
                G_work.remove_edge(*e)
                removed.add(e)

    return len(removed) / total_edges


def compute_cascade_size(ratio_name: str) -> dict:
    """Compute cascade failure size vs. q at K = K_ref.

    Cascade size S is defined as the fraction of edges removed
    (trigger + all overload-induced removals) relative to the
    total number of edges in the original graph.

    Returns
    -------
    dict with keys:
        "q_list"            : ndarray (nQ,)
        "cascade_size_mean" : ndarray (nQ,)
        "cascade_size_std"  : ndarray (nQ,)
    """
    nQ = len(q_list)
    cascade_mean = np.zeros(nQ)
    cascade_std = np.zeros(nQ)

    base_seed = stable_seed(ratio_name, "cascade")

    for qi, q in tqdm(list(enumerate(q_list)),
                      desc=f"Cascade [{ratio_name}]", unit="q"):
        sizes = []
        for r in range(realizations):
            seed = base_seed + qi * 100 + r
            G = generate_ws_network(N, K_ref, q, seed=seed)
            P = build_power_vector(ratio_name, seed=seed + 1)
            s = _simulate_cascade(G, P, alpha)
            sizes.append(s)

        cascade_mean[qi] = np.mean(sizes)
        cascade_std[qi] = np.std(sizes)

    return {
        "q_list": np.array(q_list),
        "cascade_size_mean": cascade_mean,
        "cascade_size_std": cascade_std,
    }


# ====================================================================
# 6. Orchestration + caching
# ====================================================================

def _cache_path(ratio_name: str):
    return CACHE_DIR / f"{ratio_name}.pkl"


def _load_cache(ratio_name: str):
    path = _cache_path(ratio_name)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(ratio_name: str, data: dict):
    path = _cache_path(ratio_name)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache saved → {path}")


def compute_all_for_ratio(ratio_name: str, force_recompute_kc: bool = False) -> None:
    """Run all computations for a given ratio configuration.

    Results are cached to cache/{ratio_name}.pkl.

    Parameters
    ----------
    force_recompute_kc : bool
        如果为 True，即使缓存存在也重新计算 κ_c（使用新的双向算法），
        但保留 Lorenz/Gini/Cascade 等已有结果。
    """
    cached = _load_cache(ratio_name)

    # 检查是否需要重算 κ_c（缓存中无新 key 或 force）
    need_kc = (cached is None
               or "kappa_c_low_map" not in cached
               or force_recompute_kc)

    if cached is not None and not need_kc:
        print(f"  [cache hit] {_cache_path(ratio_name)}")
        return

    if need_kc:
        print(f"\n--- Computing κ_c map (bidirectional) for '{ratio_name}' ---")
        kc_low_mean, kc_low_std, kc_high_mean, kc_high_std = \
            compute_kappa_c_map(ratio_name)

        # Optimal q* per K (q that minimises κ_c^low)
        if K_ref in K_list:
            q_star_Kref_idx = np.nanargmin(kc_low_mean[K_list.index(K_ref)])
            q_star_Kref = q_list[q_star_Kref_idx]
        else:
            q_star_Kref = np.nan
        print(f"  q* at K_ref={K_ref}: {q_star_Kref:.2f}")
    else:
        kc_low_mean = cached["kappa_c_low_map"]
        kc_low_std = cached["kappa_c_low_std"]
        kc_high_mean = cached["kappa_c_high_map"]
        kc_high_std = cached["kappa_c_high_std"]
        q_star_Kref = cached["q_star_Kref"]

    # Lorenz/Gini/Cascade: 用已有缓存或重算
    if cached is not None and "lorenz_curves" in cached:
        lorenz_gini_data = {
            "lorenz_curves": cached["lorenz_curves"],
            "gini_vs_q": cached["gini_vs_q"],
        }
        cascade_data = cached["cascade_size_vs_q"]
    else:
        print(f"\n--- Computing Lorenz / Gini for '{ratio_name}' ---")
        lorenz_gini = compute_lorenz_and_gini(ratio_name)
        lorenz_gini_data = {
            "lorenz_curves": lorenz_gini["lorenz_curves"],
            "gini_vs_q": {
                "q_list": lorenz_gini["q_list"],
                "gini_mean": lorenz_gini["gini_mean"],
                "gini_std": lorenz_gini["gini_std"],
            },
        }
        print(f"\n--- Computing cascade size for '{ratio_name}' ---")
        cascade = compute_cascade_size(ratio_name)
        cascade_data = {
            "q_list": cascade["q_list"],
            "cascade_size_mean": cascade["cascade_size_mean"],
            "cascade_size_std": cascade["cascade_size_std"],
        }

    data = {
        # 新算法结果
        "kappa_c_low_map": kc_low_mean,
        "kappa_c_low_std": kc_low_std,
        "kappa_c_high_map": kc_high_mean,
        "kappa_c_high_std": kc_high_std,
        # 保留旧 key 名以兼容（指向 low 结果）
        "kappa_c_map": kc_low_mean,
        "kappa_c_std": kc_low_std,
        "q_star_Kref": q_star_Kref,
        "lorenz_curves": lorenz_gini_data["lorenz_curves"],
        "gini_vs_q": lorenz_gini_data["gini_vs_q"],
        "cascade_size_vs_q": cascade_data,
    }
    _save_cache(ratio_name, data)
