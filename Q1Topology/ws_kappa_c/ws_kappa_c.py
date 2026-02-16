"""
W-S 拓扑结构对 κ_c 的影响测试
不同 rewiring probability q 下，四种 (n+, n-, np) 配置的临界耦合强度 κ_c
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count
from pathlib import Path
import warnings
import random

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 全局参数
# ============================================================
N = 50
K_BAR = 4
PMAX = 5.0
GAMMA = 1.0  # 固定阻尼系数
REALIZATIONS = 100
Q_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

CONFIGS = [
    (25, 25, 0),   # 均等无 passive
    (45, 5, 0),    # 极端发电主导
    (5, 45, 0),    # 极端用电主导
    (17, 17, 16),  # 三等分
]
CONFIG_LABELS = [
    r"$(n_+, n_-, n_p) = (25, 25, 0)$",
    r"$(n_+, n_-, n_p) = (45, 5, 0)$",
    r"$(n_+, n_-, n_p) = (5, 45, 0)$",
    r"$(n_+, n_-, n_p) = (17, 17, 16)$",
]
CONFIG_COLORS = ['#4a7ebb', '#e74c3c', '#2ecc71', '#9b59b6']

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 核心物理引擎 (from figure1.py)
# ============================================================

def generate_network(n, K_bar, q):
    """生成 Watts-Strogatz 小世界网络 (保证连通)。"""
    G = nx.connected_watts_strogatz_graph(n, K_bar, q)
    return nx.to_numpy_array(G)


def assign_powers(n, n_plus, n_minus, Pmax):
    """
    功率分配:
      P_gen = +Pmax / n_+,  P_con = -Pmax / n_-
    """
    P = np.zeros(n)
    if n_plus == 0 or n_minus == 0:
        return P
    indices = np.random.permutation(n)
    gen_idx = indices[:n_plus]
    con_idx = indices[n_plus:n_plus + n_minus]
    P[gen_idx] = Pmax / n_plus
    P[con_idx] = -Pmax / n_minus
    return P


def compute_steady_state_residual(theta, A, P, kappa):
    """Compute power balance residual."""
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def integrate_swing(A, P, n, kappa, y0, I=1.0, D=1.0, t_max=200.0):
    """
    Integrate the second-order swing equation.
    State: y = [ω_1..ω_n, θ_1..θ_n]
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

    resid = compute_steady_state_residual(theta_final, A, P, kappa)
    if np.linalg.norm(resid, 2) < 1e-5:
        return True, y_final
    return False, y_final


def find_kappa_c(A, P, n, kappa_start=25.0, step_init=0.5, tol=1e-3):
    """
    Find critical coupling κ_c using bisection from high κ downward.
    """
    kappa = kappa_start

    y0 = np.random.rand(2 * n)
    converged, y_last = integrate_swing(A, P, n, kappa, y0, t_max=200.0)
    if not converged:
        return np.nan

    stepsize = step_init
    kappa_old = kappa

    while True:
        converged, y_sol = integrate_swing(A, P, n, kappa, y_last, t_max=100.0)

        if converged:
            y_last = y_sol
            if stepsize < tol:
                return kappa
            kappa_old = kappa
            kappa = kappa - stepsize
        else:
            stepsize = stepsize / 2.0
            kappa = kappa_old - stepsize

        if kappa < 0 or stepsize < 1e-6:
            return kappa_old


def compute_kappa_c_single(args):
    """单次实现 (用于并行 map)。"""
    n, K_bar, q, n_plus, n_minus, Pmax, seed = args
    np.random.seed(seed)
    random.seed(seed)
    A = generate_network(n, K_bar, q)
    P = assign_powers(n, n_plus, n_minus, Pmax)
    return find_kappa_c(A, P, n)


def compute_kappa_c_stats_parallel(n, K_bar, q, n_plus, n_minus, Pmax,
                                   realizations, pool):
    """并行计算均值和标准差。"""
    rng = np.random.default_rng(12345)
    seeds = rng.integers(0, 10**9, size=realizations)
    args_list = [(n, K_bar, q, n_plus, n_minus, Pmax, int(s)) for s in seeds]
    results = pool.map(compute_kappa_c_single, args_list)
    values = np.array([r for r in results if not np.isnan(r)])
    if len(values) == 0:
        return np.nan, np.nan
    return np.mean(values), np.std(values)


# ============================================================
# 计算 + 绘图
# ============================================================

def compute_config(config_idx, n_plus, n_minus, n_passive):
    """计算单个配置下所有 q 值的 κ_c。"""
    cache_file = CACHE_DIR / f"ws_kappa_c_v2_config{config_idx + 1}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  Config {config_idx + 1}: loaded from cache")
        return data['q_values'], data['kappa_c_mean'], data['kappa_c_std']

    q_arr = np.array(Q_VALUES)
    kappa_c_mean = np.full(len(Q_VALUES), np.nan)
    kappa_c_std = np.full(len(Q_VALUES), np.nan)

    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        for qi, q in enumerate(Q_VALUES):
            print(f"  Config {config_idx + 1} ({n_plus},{n_minus},{n_passive}), q={q}")
            mean_val, std_val = compute_kappa_c_stats_parallel(
                N, K_BAR, q, n_plus, n_minus, PMAX, REALIZATIONS, pool
            )
            kappa_c_mean[qi] = mean_val
            kappa_c_std[qi] = std_val
            print(f"    κ_c = {mean_val:.4f} ± {std_val:.4f}")

    np.savez(cache_file, q_values=q_arr,
             kappa_c_mean=kappa_c_mean, kappa_c_std=kappa_c_std)
    print(f"  Cached to {cache_file}")
    return q_arr, kappa_c_mean, kappa_c_std


def plot_config(config_idx, q_values, kappa_c_mean, kappa_c_std):
    """绘制单个配置的 κ_c vs q 图。"""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(q_values, kappa_c_mean, yerr=kappa_c_std,
                fmt='o-', capsize=4, capthick=1.5, lw=2,
                color='#4a7ebb', ecolor='#a8c4e0', markersize=7)

    ax.set_xlabel(r"Rewiring probability $q$", fontsize=12)
    ax.set_ylabel(r"$\overline{\kappa}_c$", fontsize=12, rotation=0, labelpad=20)
    ax.set_title(CONFIG_LABELS[config_idx], fontsize=12)
    ax.set_xticks(Q_VALUES)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fname = f"ws_kappa_c_v2_config{config_idx + 1}"
    fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}.png/.pdf")
    plt.close(fig)


def plot_combined(all_results):
    """四条曲线叠在同一张图上。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (q_vals, means, stds) in enumerate(all_results):
        ax.errorbar(q_vals, means, yerr=stds,
                    fmt='o-', capsize=4, capthick=1.5, lw=2,
                    color=CONFIG_COLORS[i], ecolor=CONFIG_COLORS[i],
                    markersize=7, label=CONFIG_LABELS[i], alpha=0.85)

    ax.set_xlabel(r"Rewiring probability $q$", fontsize=12)
    ax.set_ylabel(r"$\overline{\kappa}_c$", fontsize=12, rotation=0, labelpad=20)
    ax.set_title(rf"W-S topology: $\kappa_c$ vs $q$ ($P_{{max}}={PMAX}$, $N={N}$)",
                 fontsize=13)
    ax.set_xticks(Q_VALUES)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fname = "ws_kappa_c_v2_combined"
    fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}.png/.pdf")
    plt.close(fig)


def main():
    print("=" * 60)
    print("W-S 拓扑结构对 κ_c 的影响测试 (v2)")
    print(f"n={N}, K_bar={K_BAR}, Pmax={PMAX}, γ={GAMMA}")
    print(f"q values: {Q_VALUES}")
    print(f"Realizations: {REALIZATIONS}")
    print("=" * 60)

    all_results = []
    for i, (n_plus, n_minus, n_passive) in enumerate(CONFIGS):
        print(f"\n--- Config {i + 1}: ({n_plus}, {n_minus}, {n_passive}) ---")
        q_vals, means, stds = compute_config(i, n_plus, n_minus, n_passive)
        plot_config(i, q_vals, means, stds)
        all_results.append((q_vals, means, stds))

    print("\n--- Combined plot ---")
    plot_combined(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
