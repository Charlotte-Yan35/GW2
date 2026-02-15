"""
复现 Reference3 Figure 6 — PV generation 对微电网韧性的影响

Panel A: 三角 simplex 热力图 (夏季 50% PV) + failure points
Panel B: 三角 simplex 热力图 (夏季 100% PV) + failure points
Panel C: α_c 直方图 (夏季 50% PV)
Panel D: α_c 直方图 (夏季 100% PV)

计算流程:
1. 构建 ring lattice (WS q=0, k=4)
2. Swing dynamics ODE 求稳态
3. Cascade failure 仿真
4. Bisection 求 α_c
5. Simplex 遍历 → 热力图
6. 从 figure4 缓存加载 failure points
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.sparse.csgraph import connected_components as scipy_cc
from scipy.sparse import csr_matrix
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings
import sys

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================
# 物理 / 网络参数
# ============================================================
N_NODES = 50
K_NEIGHBORS = 4       # ring lattice: each node connects to k/2=2 neighbors each side
Q_REWIRE = 0.0        # no rewiring (pure lattice)
KAPPA = 5.0            # coupling strength
INERTIA = 1.0
DAMPING = 1.0
P_TOT = 1.0
SYNCTOL = 2.0          # desync tolerance (swingcascadeouterbisection uses 2.0)
ODE_TSPAN = 250.0      # steady-state integration time
ODE_RTOL = 1e-8
ODE_ATOL = 1e-8

# Bisection parameters
BISECT_ALPHA_INIT = 0.01
BISECT_STEPSIZE = 0.3
BISECT_TOL = 5e-4

# Ensemble
ENSEMBLE_SIZE = 10     # reduced from 300 for speed; increase for accuracy

# Ternary plotting
H_TRI = np.sqrt(3) / 2


# ============================================================
# Step 1: Network Construction
# ============================================================

def build_ring_lattice(n, k):
    """
    Build a ring lattice (Watts-Strogatz with q=0).
    n nodes, each connected to k/2 nearest neighbors on each side.
    Returns incidence matrix E (n x m).
    """
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            edges.append((i, target))

    m = len(edges)
    E = np.zeros((n, m))
    for idx, (origin, target) in enumerate(edges):
        E[origin, idx] = 1.0
        E[target, idx] = -1.0
    return E


def adjacency_from_incidence(E):
    """A = diag(degrees) - E @ E^T"""
    n = E.shape[0]
    degrees = np.sum(np.abs(E), axis=1)
    L = E @ E.T
    A = np.diag(degrees) - L
    return A


def get_degrees(E):
    return np.sum(np.abs(E), axis=1)


def source_sink_locs_mod(E, ns, nd, n, rng):
    """
    Random source/sink placement.
    When ns==1 or nd==1, require the chosen node to have degree > 3.
    Mirrors sourcesinklocsmod() from Julia reference.
    """
    degrees = get_degrees(E)
    sources = []
    sinks = []

    if ns == 1:
        while True:
            z = rng.integers(0, n)
            if degrees[z] > 3:
                sources.append(z)
                break
    if nd == 1:
        while True:
            z = rng.integers(0, n)
            if degrees[z] > 3 and z not in sources:
                sinks.append(z)
                break

    if ns > 1:
        while len(sources) < ns:
            z = rng.integers(0, n)
            if z not in sources and z not in sinks:
                sources.append(z)
    if nd > 1:
        while len(sinks) < nd:
            z = rng.integers(0, n)
            if z not in sources and z not in sinks:
                sinks.append(z)

    return sources, sinks


def source_sink_vector(sources, sinks, n, Ptot):
    """Create power injection vector."""
    P = np.zeros(n)
    ns = len(sources)
    nd = len(sinks)
    for i in sources:
        P[i] = Ptot / ns
    for i in sinks:
        P[i] = -Ptot / nd
    return P


# ============================================================
# Step 2: Swing Dynamics ODE
# ============================================================

def swing_rhs(t, psi, A, P, I, D, kappa):
    """
    ψ = [ω₁...ωₙ, θ₁...θₙ]
    ω̇ᵢ = (Pᵢ - D·ωᵢ - κ·Σⱼ Aᵢⱼ·sin(θᵢ - θⱼ)) / I
    θ̇ᵢ = ωᵢ
    """
    n = len(P)
    omega = psi[:n]
    theta = psi[n:]

    s = np.sin(theta)
    c = np.cos(theta)
    # sin(θᵢ - θⱼ) = sin(θᵢ)cos(θⱼ) - cos(θᵢ)sin(θⱼ)
    sin_mat = s[:, None] * c[None, :] - c[:, None] * s[None, :]
    C = kappa * A * sin_mat
    coupling = C @ np.ones(n)

    omega_dot = (P - D * omega - coupling) / I
    theta_dot = omega
    return np.concatenate([omega_dot, theta_dot])


def steady_state_residual(theta, A, P, kappa):
    """Compute residual P - κ·A·sin_matrix·1"""
    n = len(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    sin_mat = s[:, None] * c[None, :] - c[:, None] * s[None, :]
    C = kappa * A * sin_mat
    return P - C @ np.ones(n)


def edge_power(theta, E, kappa):
    """F = κ·sin(E^T·θ)"""
    delta_theta = E.T @ theta
    return kappa * np.sin(delta_theta)


def solve_steady_state(A, P, n, kappa=KAPPA, I=INERTIA, D=DAMPING,
                       tspan=ODE_TSPAN, rng=None):
    """
    Solve swing ODE to find steady state.
    Strategy: short ODE integration to get near basin, then Newton root-find.
    Returns (psi, converged) where psi = [omega, theta].
    """
    if rng is None:
        rng = np.random.default_rng()

    psi0 = rng.random(2 * n)

    def rhs(t, psi):
        return swing_rhs(t, psi, A, P, I, D, kappa)

    # Short ODE to get into basin of attraction
    sol = solve_ivp(rhs, (0, 50.0), psi0, method='RK45',
                    rtol=1e-6, atol=1e-6,
                    dense_output=False, max_step=np.inf)
    psi_approx = sol.y[:, -1]
    theta_approx = psi_approx[n:]

    # Newton root-find for steady state: P - κ·A·sin_mat·1 = 0
    def residual_func(theta):
        return steady_state_residual(theta, A, P, kappa)

    result = root(residual_func, theta_approx, method='hybr', tol=1e-10)
    if result.success:
        theta_ss = result.x
        psi_ss = np.zeros(2 * n)
        psi_ss[n:] = theta_ss  # omega = 0 at steady state
        resid = steady_state_residual(theta_ss, A, P, kappa)
        converged = np.linalg.norm(resid) < 1e-3
        return psi_ss, converged

    # Fallback: longer ODE integration
    sol = solve_ivp(rhs, (0, tspan), psi0, method='RK45',
                    rtol=ODE_RTOL, atol=ODE_ATOL,
                    dense_output=False, max_step=np.inf)
    psi = sol.y[:, -1]
    theta = psi[n:]
    resid = steady_state_residual(theta, A, P, kappa)
    converged = np.linalg.norm(resid) < 1e-3
    return psi, converged


# ============================================================
# Step 3: Cascade Failure (optimized — works with local matrices)
# ============================================================

def connected_components_dense(A):
    """
    Find connected components of adjacency matrix A (dense).
    Returns (n_components, list of lists of node indices).
    """
    n = A.shape[0]
    sparse_A = csr_matrix(np.abs(A) > 0.5)
    n_comp, labels = scipy_cc(sparse_A, directed=False)
    table = [[] for _ in range(n_comp)]
    for i in range(n):
        table[labels[i]].append(i)
    return n_comp, table


def _solve_fracture_steady_state(A1, P_bal, E1, theta_init, n_nodes, kappa, synctol, maxflow, alpha):
    """
    Find steady state for a cascade sub-problem.
    Returns (theta_ss, sync, trip) where:
      sync: True if the system synchronized (converged)
      trip: True if any edge exceeded alpha
    """
    # Try Newton root-finding first (much faster than ODE)
    def residual_func(theta):
        return steady_state_residual(theta, A1, P_bal, kappa)

    result = root(residual_func, theta_init, method='hybr', tol=1e-10)

    if result.success and np.linalg.norm(result.fun) < 1e-6:
        theta_ss = result.x
        # Check for overloaded edges
        flow = edge_power(theta_ss, E1, kappa) / maxflow
        trip = np.any(np.abs(flow) > alpha)
        return theta_ss, True, trip

    # Fallback: ODE integration with monitoring
    psi0 = np.zeros(2 * n_nodes)
    psi0[n_nodes:] = theta_init
    # Start with small omega perturbation from previous state
    # (psi_state might have nonzero omega)

    def rhs(t, psi):
        return swing_rhs(t, psi, A1, P_bal, INERTIA, DAMPING, kappa)

    sol = solve_ivp(rhs, (0, 500.0), psi0, method='RK45',
                    rtol=1e-6, atol=1e-6,
                    dense_output=False, max_step=np.inf)

    psi_end = sol.y[:, -1]
    omega_end = psi_end[:n_nodes]
    theta_end = psi_end[n_nodes:]

    # Check desync
    if np.linalg.norm(omega_end) > synctol:
        return theta_end, False, False

    # Try Newton again from ODE endpoint
    result2 = root(residual_func, theta_end, method='hybr', tol=1e-10)
    if result2.success:
        theta_ss = result2.x
    else:
        theta_ss = theta_end

    flow = edge_power(theta_ss, E1, kappa) / maxflow
    trip = np.any(np.abs(flow) > alpha)
    return theta_ss, True, trip


def swing_fracture_local(E1, P1, psi_state, synctol, alpha, kappa, maxflow):
    """
    Recursive cascade failure on a LOCAL component.
    E1: local incidence matrix (n_local x m_local)
    P1: local power vector (NOT yet balanced — balance happens here)
    psi_state: [omega, theta] for this component
    Returns: number of surviving edges
    """
    tol = 1e-5
    n_nodes = E1.shape[0]
    n_edges = E1.shape[1]

    if n_edges == 0:
        return 0

    # Check sources/sinks
    source_count = int(np.sum(P1 > tol))
    sink_count = int(np.sum(P1 < -tol))
    if source_count == 0 or sink_count == 0:
        return 0

    # Balance power homogeneously
    P_bal = P1.copy()
    delta = np.sum(P_bal) / 2.0
    source_mask = P_bal > tol
    sink_mask = P_bal < -tol
    P_bal[sink_mask] -= delta / sink_count
    P_bal[source_mask] -= delta / source_count

    A1 = adjacency_from_incidence(E1)

    # Get theta from psi_state
    theta_init = psi_state[n_nodes:]

    theta_ss, sync, trip = _solve_fracture_steady_state(
        A1, P_bal, E1, theta_init, n_nodes, kappa, synctol, maxflow, alpha
    )

    if not sync:
        return 0

    if not trip:
        return n_edges

    # Remove overloaded edges
    flow = edge_power(theta_ss, E1, kappa) / maxflow
    overloaded = np.abs(flow) > alpha
    survivor_mask = ~overloaded
    survivors = np.where(survivor_mask)[0]
    if len(survivors) == 0:
        return 0

    E2 = E1[:, survivors]
    A2 = adjacency_from_incidence(E2)
    n_comp, table = connected_components_dense(A2)

    # Recurse on each component
    total_surviving = 0
    for comp_nodes in table:
        if len(comp_nodes) == 0:
            continue
        E_sub = E2[comp_nodes, :]
        edge_mask = np.any(np.abs(E_sub) > 0.5, axis=0)
        E_sub = E_sub[:, edge_mask]

        P_sub = P1[comp_nodes]
        psi_sub = np.zeros(2 * len(comp_nodes))
        psi_sub[len(comp_nodes):] = theta_ss[comp_nodes]

        total_surviving += swing_fracture_local(
            E_sub, P_sub, psi_sub, synctol, alpha, kappa, maxflow
        )

    return total_surviving


# ============================================================
# Step 4: Bisection for α_c (per network instance)
# ============================================================

def cascade_at_alpha(E, P, psi_ss, n, kappa, alpha, maxflow, synctol):
    """
    Run cascade at a given alpha on a pre-computed network.
    Returns fraction of surviving edges S.
    """
    m = E.shape[1]
    omega = psi_ss[:n]
    theta = psi_ss[n:]

    # Compute normalized flow
    flow = edge_power(theta, E, kappa) / maxflow

    # Knock out most loaded edge
    d = np.argmax(np.abs(flow))
    E1 = np.delete(E, d, axis=1)

    # Check connected components after edge removal
    A1 = adjacency_from_incidence(E1)
    n_comp, table = connected_components_dense(A1)

    # Do cascade on each component
    total_surviving = 0
    for comp_nodes in table:
        if len(comp_nodes) == 0:
            continue
        E_sub = E1[comp_nodes, :]
        edge_mask = np.any(np.abs(E_sub) > 0.5, axis=0)
        E_sub = E_sub[:, edge_mask]

        P_sub = P[comp_nodes]
        omega_sub = omega[comp_nodes]
        theta_sub = theta[comp_nodes]
        psi_sub = np.concatenate([omega_sub, theta_sub])

        surviving = swing_fracture_local(
            E_sub, P_sub, psi_sub, synctol, alpha, kappa, maxflow
        )
        total_surviving += surviving

    return total_surviving / (m - 1)  # m-1 because we removed one edge


def compute_rho_for_config(ns, nd, n, ensemble_size, rng):
    """
    For a given (ns, nd, ne=n-ns-nd), compute the critical alpha (ρ̄) using
    ensemble bisection — matches swingcascadebisection() in reference.

    Creates an ensemble of networks, then bisects alpha so that the
    ensemble-mean surviving fraction S crosses 0.5.
    """
    ne = n - ns - nd
    if ne < 0 or ns < 1 or nd < 1:
        return np.nan

    E = build_ring_lattice(n, K_NEIGHBORS)

    # Pre-compute ensemble: network instances with steady-state solutions
    ensemble = []
    for z in range(ensemble_size):
        sources, sinks = source_sink_locs_mod(E, ns, nd, n, rng)
        P = source_sink_vector(sources, sinks, n, P_TOT)
        A = adjacency_from_incidence(E)

        psi, converged = solve_steady_state(A, P, n, rng=rng)
        if not converged:
            continue

        theta = psi[n:]
        flow = edge_power(theta, E, KAPPA)
        fmax = np.max(np.abs(flow))
        if fmax < 1e-12:
            continue

        # Normalize flow
        F_norm = flow / fmax
        ensemble.append((P, psi, F_norm, fmax))

    if len(ensemble) == 0:
        return np.nan

    # Bisection on alpha across the whole ensemble
    alpha = BISECT_ALPHA_INIT
    stepsize = BISECT_STEPSIZE
    beneath0 = True
    beneath1 = True

    while abs(stepsize) > BISECT_TOL:
        survivors_list = []
        for P, psi, F_norm, fmax in ensemble:
            S = cascade_at_alpha(E, P, psi, n, KAPPA, alpha, fmax, SYNCTOL)
            survivors_list.append(S)

        avg_S = np.mean(survivors_list)

        if avg_S > 0.5:
            beneath1 = False
        else:
            beneath1 = True

        if beneath1 != beneath0:
            stepsize = -stepsize / 2.0

        beneath0 = beneath1
        alpha += stepsize

    return alpha


# ============================================================
# Step 6: Simplex Traversal
# ============================================================

def _compute_one_config(args):
    """Worker function for parallel computation."""
    ns_val, nd_val, n, ensemble_size, seed = args
    rng = np.random.default_rng(seed)
    return compute_rho_for_config(ns_val, nd_val, n, ensemble_size, rng)


def compute_simplex_heatmap(n=N_NODES, ensemble_size=ENSEMBLE_SIZE, seed=42,
                            n_workers=None):
    """
    Compute ρ̄ for all valid (ns, nd, ne) on the simplex.
    Cache results to npz. Supports parallel computation.
    """
    cache_file = CACHE_DIR / f"fig6_rho_heatmap_n{n}.npz"

    if cache_file.exists():
        print(f"Loading cache: {cache_file.name}")
        data = np.load(cache_file)
        return data['ns_arr'], data['nd_arr'], data['ne_arr'], data['rho_arr']

    # Generate all valid configurations
    configs = []
    for nd_val in range(1, n):
        for ns_val in range(1, n - nd_val + 1):
            ne_val = n - ns_val - nd_val
            if ne_val >= 0:
                configs.append((ns_val, nd_val, ne_val))

    n_configs = len(configs)
    print(f"Total configurations: {n_configs}")

    ns_arr = np.array([c[0] for c in configs])
    nd_arr = np.array([c[1] for c in configs])
    ne_arr = np.array([c[2] for c in configs])
    rho_arr = np.full(n_configs, np.nan)

    # Prepare worker args with unique seeds
    base_rng = np.random.default_rng(seed)
    worker_args = []
    for idx, (ns_val, nd_val, ne_val) in enumerate(configs):
        worker_seed = base_rng.integers(0, 2**32)
        worker_args.append((ns_val, nd_val, n, ensemble_size, worker_seed))

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if n_workers > 1:
        print(f"Using {n_workers} parallel workers")
        # Process in chunks to show progress and save intermediate results
        chunk_size = 50
        for start in range(0, n_configs, chunk_size):
            end = min(start + chunk_size, n_configs)
            chunk_args = worker_args[start:end]
            with Pool(n_workers) as pool:
                results = pool.map(_compute_one_config, chunk_args)
            rho_arr[start:end] = results
            print(f"  Progress: {end}/{n_configs} "
                  f"({100*end/n_configs:.1f}%)")

            # Intermediate save
            np.savez_compressed(cache_file, ns_arr=ns_arr, nd_arr=nd_arr,
                                ne_arr=ne_arr, rho_arr=rho_arr)
    else:
        for idx in range(n_configs):
            if idx % 50 == 0:
                print(f"  Config {idx+1}/{n_configs}: "
                      f"ns={ns_arr[idx]}, nd={nd_arr[idx]}, ne={ne_arr[idx]}")
            rho_arr[idx] = _compute_one_config(worker_args[idx])

            # Periodic save
            if idx % 100 == 99:
                np.savez_compressed(cache_file, ns_arr=ns_arr, nd_arr=nd_arr,
                                    ne_arr=ne_arr, rho_arr=rho_arr)

    np.savez_compressed(cache_file, ns_arr=ns_arr, nd_arr=nd_arr,
                        ne_arr=ne_arr, rho_arr=rho_arr)
    print(f"Cached to {cache_file.name}")

    return ns_arr, nd_arr, ne_arr, rho_arr


# ============================================================
# Step 7: Load failure points from Figure 4 cache
# ============================================================

def load_failure_points(month, penetration, n_ensemble=50):
    """
    Load sigma trajectories from Figure 4 cache.
    Convert final sigma points to simplex coordinates.
    """
    cache_file = CACHE_DIR / f"sigmas_m{month}_p{penetration}_n{n_ensemble}.npz"
    if not cache_file.exists():
        print(f"Warning: cache file not found: {cache_file}")
        return np.array([]), np.array([]), np.array([])

    data = np.load(cache_file)
    ns_pts, nd_pts, ne_pts = [], [], []

    for i in range(n_ensemble):
        key = f"ensemble_{i}"
        if key not in data:
            break
        arr = data[key]  # shape (T, 3): sigma_s, sigma_d, sigma_p
        # Use final time step
        sigma_s, sigma_d, sigma_p = arr[-1]
        ns_val = sigma_s * N_NODES
        nd_val = sigma_d * N_NODES
        ne_val = N_NODES - ns_val - nd_val
        ns_pts.append(ns_val)
        nd_pts.append(nd_val)
        ne_pts.append(ne_val)

        # Also add a selection of time steps for denser scatter
        # Use every 10th time step from the second half of trajectory
        half = len(arr) // 2
        for t_idx in range(half, len(arr), 10):
            sigma_s, sigma_d, sigma_p = arr[t_idx]
            ns_val = sigma_s * N_NODES
            nd_val = sigma_d * N_NODES
            ne_val = N_NODES - ns_val - nd_val
            ns_pts.append(ns_val)
            nd_pts.append(nd_val)
            ne_pts.append(ne_val)

    return np.array(ns_pts), np.array(nd_pts), np.array(ne_pts)


def get_alpha_c_for_points(ns_pts, nd_pts, ne_pts,
                           ns_grid, nd_grid, ne_grid, rho_grid):
    """
    For each failure point, find the nearest grid point's ρ̄ value.
    """
    alpha_c_vals = []
    for i in range(len(ns_pts)):
        # Find nearest grid point
        dist = (ns_grid - ns_pts[i])**2 + (nd_grid - nd_pts[i])**2 + (ne_grid - ne_pts[i])**2
        nearest = np.nanargmin(dist)
        val = rho_grid[nearest]
        if not np.isnan(val):
            alpha_c_vals.append(val)
    return np.array(alpha_c_vals)


# ============================================================
# Step 8: Plotting
# ============================================================

def simplex_to_cart(ns, nd, ne, scale):
    """Simplex (ns, nd, ne) → Cartesian (x, y). Same as figure4.py."""
    a = nd - 1
    b = ne
    s = scale - 2
    x = (a + b / 2.0) / s
    y = (b * H_TRI) / s
    return x, y


def draw_ternary_frame(ax, scale):
    """Draw triangle frame and grid lines. Same as figure4.py."""
    n = scale
    s = n - 2
    bl = np.array([(-1) / s, 0.0])
    br = np.array([(n - 1) / s, 0.0])
    top = np.array([(-1 + n / 2) / s, n * H_TRI / s])

    corners = np.array([bl, br, top, bl])
    ax.plot(corners[:, 0], corners[:, 1], 'k-', lw=1.5)

    multiple = 6
    n_lines = s // multiple
    for k in range(1, n_lines + 1):
        frac = k * multiple / s
        p0 = bl + frac * (top - bl)
        p1 = br + frac * (top - br)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.3, alpha=0.5)
        p0 = bl + frac * (br - bl)
        p1 = top + frac * (br - top)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.3, alpha=0.5)
        p0 = br + frac * (bl - br)
        p1 = top + frac * (bl - top)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.3, alpha=0.5)

    mid_left_x = (bl[0] + top[0]) / 2 - 0.08
    mid_left_y = (bl[1] + top[1]) / 2
    ax.text(mid_left_x, mid_left_y, r'$\leftarrow\,\eta_+$',
            fontsize=11, rotation=60, ha='center', va='center')
    mid_right_x = (br[0] + top[0]) / 2 + 0.08
    mid_right_y = (br[1] + top[1]) / 2
    ax.text(mid_right_x, mid_right_y, r'$\eta_p\,\rightarrow$',
            fontsize=11, rotation=-60, ha='center', va='center')
    mid_bot_x = (bl[0] + br[0]) / 2
    mid_bot_y = bl[1] - 0.06
    ax.text(mid_bot_x, mid_bot_y, r'$\eta_-\,\rightarrow$',
            fontsize=11, ha='center', va='top')


def plot_heatmap_panel(ax, ns_grid, nd_grid, ne_grid, rho_grid,
                       ns_fail, nd_fail, ne_fail, scale, label,
                       vmin=None, vmax=None):
    """Plot one ternary heatmap panel with failure point scatter."""
    import matplotlib
    import matplotlib.pyplot as plt

    draw_ternary_frame(ax, scale)

    # Convert grid points to Cartesian and plot heatmap as scatter
    xs, ys, vals = [], [], []
    for i in range(len(ns_grid)):
        if np.isnan(rho_grid[i]):
            continue
        x, y = simplex_to_cart(ns_grid[i], nd_grid[i], ne_grid[i], scale)
        xs.append(x)
        ys.append(y)
        vals.append(rho_grid[i])

    xs = np.array(xs)
    ys = np.array(ys)
    vals = np.array(vals)

    if vmin is None:
        vmin = np.nanpercentile(vals, 2)
    if vmax is None:
        vmax = np.nanpercentile(vals, 98)

    sc = ax.scatter(xs, ys, c=vals, cmap='Reds', s=3, vmin=vmin, vmax=vmax,
                    edgecolors='none', zorder=1)

    # Plot failure points
    if len(ns_fail) > 0:
        xf, yf = [], []
        for i in range(len(ns_fail)):
            x, y = simplex_to_cart(ns_fail[i], nd_fail[i], ne_fail[i], scale)
            xf.append(x)
            yf.append(y)
        ax.scatter(xf, yf, c='blue', s=5, alpha=0.3, edgecolors='none', zorder=2)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(label, loc='left', fontweight='bold', fontsize=16)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, H_TRI * scale / (scale - 2) + 0.05)

    return sc


def plot_histogram_panel(ax, alpha_c_vals, label, xmax=30.0):
    """Plot histogram of α_c values."""
    if len(alpha_c_vals) == 0:
        ax.set_title(label, loc='left', fontweight='bold', fontsize=16)
        return

    bins = np.linspace(0, xmax, 60)
    ax.hist(alpha_c_vals, bins=bins, density=True, color='steelblue',
            edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.set_xlabel(r'$\alpha_c$', fontsize=13)
    ax.set_ylabel(r'$P(\alpha_c)$', fontsize=13)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 0.25)
    ax.set_title(label, loc='left', fontweight='bold', fontsize=16)


# ============================================================
# Main
# ============================================================

def main():
    import matplotlib
    import matplotlib.pyplot as plt

    font = {'size': 14}
    matplotlib.rc('font', **font)

    print("=" * 60)
    print("Figure 6: PV generation effect on microgrid resilience")
    print("=" * 60)

    # Step 1: Compute simplex heatmap (or load cache)
    print("\nStep 1: Computing simplex heatmap...")
    ns_grid, nd_grid, ne_grid, rho_grid = compute_simplex_heatmap(
        n=N_NODES, ensemble_size=ENSEMBLE_SIZE
    )
    print(f"  Grid points: {len(ns_grid)}")
    print(f"  ρ̄ range: [{np.nanmin(rho_grid):.3f}, {np.nanmax(rho_grid):.3f}]")

    # Step 2: Load failure points (summer data from figure4 cache)
    print("\nStep 2: Loading failure points...")
    ns_50, nd_50, ne_50 = load_failure_points(month=7, penetration=24)
    ns_100, nd_100, ne_100 = load_failure_points(month=7, penetration=49)
    print(f"  50% PV failure points: {len(ns_50)}")
    print(f"  100% PV failure points: {len(ns_100)}")

    # Step 3: Get α_c values for failure points
    print("\nStep 3: Extracting α_c at failure points...")
    ac_50 = get_alpha_c_for_points(ns_50, nd_50, ne_50,
                                    ns_grid, nd_grid, ne_grid, rho_grid)
    ac_100 = get_alpha_c_for_points(ns_100, nd_100, ne_100,
                                     ns_grid, nd_grid, ne_grid, rho_grid)
    print(f"  50% PV: {len(ac_50)} α_c values, mean={np.mean(ac_50):.3f}" if len(ac_50) > 0 else "  50% PV: no values")
    print(f"  100% PV: {len(ac_100)} α_c values, mean={np.mean(ac_100):.3f}" if len(ac_100) > 0 else "  100% PV: no values")

    # Step 4: Plot
    print("\nStep 4: Plotting...")
    vmin = np.nanpercentile(rho_grid[~np.isnan(rho_grid)], 2) if np.any(~np.isnan(rho_grid)) else 0.88
    vmax = np.nanpercentile(rho_grid[~np.isnan(rho_grid)], 98) if np.any(~np.isnan(rho_grid)) else 1.52

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Panel A: Summer 50% PV heatmap
    sc = plot_heatmap_panel(axes[0, 0], ns_grid, nd_grid, ne_grid, rho_grid,
                            ns_50, nd_50, ne_50, N_NODES, 'A',
                            vmin=vmin, vmax=vmax)

    # Panel B: Summer 100% PV heatmap
    plot_heatmap_panel(axes[0, 1], ns_grid, nd_grid, ne_grid, rho_grid,
                       ns_100, nd_100, ne_100, N_NODES, 'B',
                       vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(sc, ax=axes[0, :], shrink=0.6, pad=0.02)
    cbar.set_label(r'$\bar{\rho}$', fontsize=14)

    # Panel C: 50% PV histogram
    plot_histogram_panel(axes[1, 0], ac_50, 'C')

    # Panel D: 100% PV histogram
    plot_histogram_panel(axes[1, 1], ac_100, 'D')

    fig.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_DIR / "figure6_ABCD.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "figure6_ABCD.png", dpi=200, bbox_inches='tight')
    print(f"\nSaved to {OUTPUT_DIR / 'figure6_ABCD.pdf'} and .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
