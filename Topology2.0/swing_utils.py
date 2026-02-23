"""
Shared swing equation utilities.
Extracted from reproduction/figure1.py for reuse across experiments.
"""

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import random


def generate_network(n, K_bar, q):
    """生成 Watts-Strogatz 小世界网络 (保证连通)。"""
    G = nx.connected_watts_strogatz_graph(n, K_bar, q)
    return nx.to_numpy_array(G)


def compute_steady_state_residual(theta, A, P, kappa):
    """
    Compute power balance residual: P_i - κ·Σ_j A_ij·sin(θ_i - θ_j)
    """
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def integrate_swing(A, P, n, kappa, y0, I=1.0, D=1.0, t_max=200.0,
                    resid_tol=None):
    """
    Integrate the second-order swing equation.
    Returns (converged, y_final).
    resid_tol scales with ||P|| if None.
    """
    if resid_tol is None:
        resid_tol = max(1e-5, 1e-4 * np.linalg.norm(P))

    def rhs(t, y):
        omega = y[:n]
        theta = y[n:]
        diff = theta[:, None] - theta[None, :]
        coupling = np.sum(A * np.sin(diff), axis=1)
        domega = (P - D * omega - kappa * coupling) / I
        dtheta = omega
        return np.concatenate([domega, dtheta])

    sol = solve_ivp(rhs, [0, t_max], y0, method='RK45',
                    rtol=1e-6, atol=1e-6, max_step=2.0)

    if sol.status != 0:
        return False, y0

    y_final = sol.y[:, -1]
    theta_final = y_final[n:]

    resid = compute_steady_state_residual(theta_final, A, P, kappa)
    if np.linalg.norm(resid, 2) < resid_tol:
        return True, y_final
    return False, y_final


def find_kappa_c(A, P, n, kappa_start=None, tol=1e-2, I=1.0, D=1.0):
    """
    Find critical coupling κ_c using warm-started descent + bisection.
    Phase 1: converge at high kappa.
    Phase 2: halve kappa each step until failure (exponential search).
    Phase 3: bisection refinement.
    """
    if kappa_start is None:
        kappa_start = max(np.max(np.abs(P)) * 2.0, 5.5)
    kappa = kappa_start

    # Phase 1: find steady state at high kappa
    y0 = np.random.rand(2 * n)
    converged, y_last = integrate_swing(A, P, n, kappa, y0, I=I, D=D,
                                        t_max=150.0)
    if not converged:
        return np.nan

    # Phase 2: halve kappa each step until failure (log2 search)
    while True:
        kappa_try = kappa * 0.5
        if kappa_try < 0.01:
            break
        converged, y_sol = integrate_swing(A, P, n, kappa_try, y_last,
                                           I=I, D=D, t_max=60.0)
        if converged:
            y_last = y_sol
            kappa = kappa_try
        else:
            break

    # Phase 3: bisection between kappa*0.5 (failed) and kappa (succeeded)
    kappa_hi = kappa
    kappa_lo = kappa * 0.5

    while (kappa_hi - kappa_lo) > tol:
        kappa_mid = (kappa_hi + kappa_lo) / 2.0
        converged, y_sol = integrate_swing(A, P, n, kappa_mid, y_last,
                                           I=I, D=D, t_max=60.0)
        if converged:
            kappa_hi = kappa_mid
            y_last = y_sol
        else:
            kappa_lo = kappa_mid

    return kappa_hi
