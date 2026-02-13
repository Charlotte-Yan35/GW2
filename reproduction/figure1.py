"""
复现 Figure 1 — Variation in the critical coupling
输出两张图: figure1_AB.png (Panel A+B), figure1_CD.png (Panel C+D)

论文三角形方向 (Panel B & C):
  BL = all generators (n_+ = n)
  BR = all consumers  (n_- = n)
  Top = all passive   (n_p = n)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
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
N = 50            # 节点数
K_BAR = 4         # 平均度 ⟨k⟩=4 (Watts-Strogatz 小世界网络)
PMAX = 1.0        # 归一化最大功率
REALIZATIONS = 50  # 每配置点的实现数 (论文用 200)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

H_TRI = np.sqrt(3) / 2  # 等边三角形高度


# ============================================================
# 三角坐标转换 (全局统一)
# ============================================================

def ternary_to_cart(n_plus, n_minus, n_passive):
    """
    论文方向:
      BL (0,0) = all generators
      BR (1,0) = all consumers
      Top (0.5, H_TRI) = all passive
    """
    s = n_plus + n_minus + n_passive
    if s == 0:
        return 0.0, 0.0
    x = n_minus / s + 0.5 * n_passive / s
    y = n_passive / s * H_TRI
    return x, y


# ============================================================
# 核心物理引擎
# ============================================================

def generate_network(n, K_bar, q):
    """生成 Watts-Strogatz 小世界网络 (保证连通)。"""
    G = nx.connected_watts_strogatz_graph(n, K_bar, q)
    return nx.to_numpy_array(G)


def assign_powers(n, n_plus, n_minus, Pmax):
    """
    论文功率分配:
      P_gen = +Pmax / n_+,  P_con = -Pmax / n_-
      总发电 = 总消耗 = Pmax, 功率平衡 = 0
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


def find_steady_state_ode(A, P, n, kappa, gamma=1.0, t_max=200.0, tol=1e-4):
    """
    Numerically integrate the second-order swing equation with damping γ=1:
      d²θ/dt² + γ·dθ/dt = P - κ·Σ_j A_ij·sin(θ_i - θ_j)
    Returns True if ODE converges to a stable steady state, False otherwise.
    """
    def rhs(t, y):
        theta = y[:n]
        omega = y[n:]
        diff = theta[:, None] - theta[None, :]
        coupling = np.sum(A * np.sin(diff), axis=1)
        dtheta = omega
        domega = P - gamma * omega - kappa * coupling
        return np.concatenate([dtheta, domega])

    y0 = np.zeros(2 * n)  # θ=0, ω=0 (all at rest)
    sol = solve_ivp(rhs, [0, t_max], y0, method='RK45',
                    rtol=1e-6, atol=1e-8, max_step=1.0)

    if sol.status != 0:
        return False

    theta_final = sol.y[:n, -1]
    omega_final = sol.y[n:, -1]

    # Check convergence: all frequencies near zero
    if np.max(np.abs(omega_final)) > tol:
        return False

    # Check stability: |θ_i - θ_j| < π/2 for all edges
    rows, cols = np.where(np.triu(A, k=1) > 0)
    if len(rows) > 0:
        if not np.all(np.abs(theta_final[rows] - theta_final[cols]) < np.pi / 2):
            return False

    return True


def find_kappa_c(A, P, n, kappa_min=0.01, kappa_max=5.0, tol=0.005, max_iter=40):
    """
    Binary search for critical coupling κ_c using ODE integration.
    κ_c is the smallest κ for which the swing equation converges to a stable steady state.
    """
    # First check: if even kappa_max doesn't converge, return NaN
    if not find_steady_state_ode(A, P, n, kappa_max):
        return np.nan

    lo, hi = kappa_min, kappa_max
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if find_steady_state_ode(A, P, n, mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


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
    seeds = rng.integers(0, 10**9, size=REALIZATIONS)
    args_list = [(n, K_bar, q, n_plus, n_minus, Pmax, int(s)) for s in seeds]
    results = pool.map(compute_kappa_c_single, args_list)
    values = np.array([r for r in results if not np.isnan(r)])
    if len(values) == 0:
        return np.nan, np.nan
    return np.mean(values), np.std(values)


# ============================================================
# Panel A: 两节点分岔图
# ============================================================

def plot_panel_a(ax):
    """Δθ vs P/κ  —  匹配论文简洁风格"""
    s = np.linspace(0.0, 1.0, 600)
    ax.plot(s, np.arcsin(s), color='#4a7ebb', lw=2.5)
    ax.plot(s, np.pi - np.arcsin(s), color='#a8c4e0', lw=2.0)
    ax.scatter([1.0], [np.pi / 2], s=50, color='black', zorder=5)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, np.pi + 0.1)
    ax.set_xlabel(r"$P/\kappa$", fontsize=12)
    ax.set_ylabel(r"$\Delta\theta$", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, np.pi / 2, np.pi])
    ax.set_yticklabels([r"$0$", "", r"$\pi$"])
    ax.set_title("A", loc="left", fontweight="bold", fontsize=14)


# ============================================================
# Panel B: 配置单纯形示意图 (n=20)
# ============================================================

def plot_panel_b(ax):
    ax.set_aspect('equal')
    ax.axis('off')
    n = 20

    # 三角形顶点
    corners_x = [0, 1, 0.5, 0]  # BL, BR, Top, close
    corners_y = [0, 0, H_TRI, 0]
    ax.plot(corners_x, corners_y, 'k-', lw=1.5)

    # 彩色网格线 (匹配论文)
    gen_color = '#2ca02c'   # green  — generators
    con_color = '#ff7f0e'   # orange — consumers
    pas_color = '#aec7e8'   # light blue — passive

    for k in [5, 10, 15]:
        # 常数 n_+ = k (平行于 BR-Top 边)
        x0, y0 = ternary_to_cart(k, 0, n - k)
        x1, y1 = ternary_to_cart(k, n - k, 0)
        ax.plot([x0, x1], [y0, y1], color=gen_color, lw=0.7, zorder=0)

        # 常数 n_- = k (平行于 BL-Top 边)
        x0, y0 = ternary_to_cart(0, k, n - k)
        x1, y1 = ternary_to_cart(n - k, k, 0)
        ax.plot([x0, x1], [y0, y1], color=con_color, lw=0.7, zorder=0)

        # 常数 n_p = k (平行于 BL-BR 底边)
        x0, y0 = ternary_to_cart(0, n - k, k)
        x1, y1 = ternary_to_cart(n - k, 0, k)
        ax.plot([x0, x1], [y0, y1], color=pas_color, lw=0.7, zorder=0)

    # --- 刻度数字 ---
    # 左边 (BL→Top): generators 从 n→0, 标记 15,10,5
    for k in [5, 10, 15]:
        x, y = ternary_to_cart(k, 0, n - k)  # 点在左边 (n_-=0)
        ax.text(x - 0.03, y + 0.01, str(k), fontsize=7, ha='right',
                va='center', color=gen_color)

    # 底边 (BL→BR): consumers 从 0→n, 标记 5,10,15
    for k in [5, 10, 15]:
        x, y = ternary_to_cart(n - k, k, 0)  # 点在底边 (n_p=0)
        ax.text(x, y - 0.04, str(k), fontsize=7, ha='center',
                va='top', color=con_color)

    # 右边 (BR→Top): passive 从 0→n, 标记 5,10,15
    for k in [5, 10, 15]:
        x, y = ternary_to_cart(0, n - k, k)  # 点在右边 (n_+=0)
        ax.text(x + 0.03, y + 0.01, str(k), fontsize=7, ha='left',
                va='center', color=pas_color)

    # --- 边标签 ---
    # 左边: Generators (arrow ←, 从 Top 指向 BL)
    mx, my = 0.5 * (0 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx - 0.10, my + 0.02, r"$\leftarrow$ Generators", fontsize=9,
            ha='center', va='center', rotation=60, color=gen_color)

    # 底边: Consumers →
    ax.text(0.5, -0.09, r"Consumers $\rightarrow$", fontsize=9,
            ha='center', va='top', color=con_color)

    # 右边: Passive ↑
    mx, my = 0.5 * (1 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx + 0.10, my + 0.02, r"$\leftarrow$ Passive", fontsize=9,
            ha='center', va='center', rotation=-60, color=pas_color)

    # 黑点 (5, 10, 5)
    xd, yd = ternary_to_cart(5, 10, 5)
    ax.plot(xd, yd, 'o', color='k', ms=6, zorder=5)
    ax.annotate(r"$(5,10,5)$", (xd, yd), textcoords="offset points",
                xytext=(-12, 8), fontsize=7, ha='center')

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, H_TRI + 0.02)
    ax.set_title("B", loc="left", fontweight="bold", fontsize=14)


# ============================================================
# Panel C: κ_c 热力图 (单纯形)
# ============================================================

def compute_panel_c_data(n=N, K_bar=K_BAR, q=0.0, Pmax=PMAX,
                         realizations=REALIZATIONS, step=2):
    """遍历所有 n_+ + n_- + n_p = n 的配置 (step 间隔)。"""
    cache_file = CACHE_DIR / f"v3_panel_c_n{n}_k{K_bar}_q{q}_r{realizations}_s{step}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return data['configs'], data['kappa_c_vals']

    configs = []
    for n_plus in range(0, n + 1, step):
        for n_minus in range(0, n + 1 - n_plus, step):
            n_passive = n - n_plus - n_minus
            configs.append((n_plus, n_minus, n_passive))

    print(f"Panel C: {len(configs)} configurations, {realizations} realizations each")

    kappa_c_vals = np.full(len(configs), np.nan)

    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        for idx, (n_plus, n_minus, n_passive) in enumerate(configs):
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{len(configs)}")
            # n_+=0 或 n_-=0 时无功率流, κ_c 无意义 → NaN
            if n_plus == 0 or n_minus == 0:
                continue
            mean_val, _ = compute_kappa_c_stats_parallel(
                n, K_bar, q, n_plus, n_minus, Pmax, realizations, pool
            )
            kappa_c_vals[idx] = mean_val

    configs = np.array(configs)
    np.savez(cache_file, configs=configs, kappa_c_vals=kappa_c_vals)
    print(f"Panel C data cached to {cache_file}")
    return configs, kappa_c_vals


def plot_panel_c(ax, configs, kappa_c_vals, n=N):
    ax.set_aspect('equal')
    ax.axis('off')

    # 三角形边界
    corners_x = [0, 1, 0.5, 0]
    corners_y = [0, 0, H_TRI, 0]
    ax.plot(corners_x, corners_y, 'k-', lw=1.0)

    # 转换坐标 (使用全局 ternary_to_cart)
    xs, ys, vals = [], [], []
    for idx, (n_plus, n_minus, n_passive) in enumerate(configs):
        if np.isnan(kappa_c_vals[idx]):
            continue
        x, y = ternary_to_cart(n_plus, n_minus, n_passive)
        xs.append(x)
        ys.append(y)
        vals.append(kappa_c_vals[idx])

    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

    if len(xs) > 3:
        triang = mtri.Triangulation(xs, ys)
        tcf = ax.tricontourf(triang, vals, levels=20, cmap='YlGnBu_r')
        cb = plt.colorbar(tcf, ax=ax, shrink=0.7)
        cb.set_label(r'$\overline{\kappa}_c$', fontsize=11)

    # (i) 截面虚线 (n_p=0 底边)
    x0, y0 = ternary_to_cart(n, 0, 0)
    x1, y1 = ternary_to_cart(0, n, 0)
    ax.plot([x0, x1], [y0, y1], 'b--', lw=1.0, zorder=3)
    xm = 0.5 * (x0 + x1)
    ax.text(xm, -0.04, "(i)", fontsize=8, ha='center', fontstyle='italic',
            color='blue')

    # --- 三边标签 ---
    # 左边: ← Generators
    mx, my = 0.5 * (0 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx - 0.08, my, r"$\leftarrow$ Generators", fontsize=9,
            ha='center', va='center', rotation=60)

    # 底边: Consumers →
    ax.text(0.5, -0.07, r"Consumers $\rightarrow$", fontsize=9,
            ha='center', va='top')

    # 右边: Passive ↑
    mx, my = 0.5 * (1 + 0.5), 0.5 * (0 + H_TRI)
    ax.text(mx + 0.08, my, r"$\leftarrow$ Passive", fontsize=9,
            ha='center', va='center', rotation=-60)

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, H_TRI + 0.02)
    ax.set_title("C", loc="left", fontweight="bold", fontsize=14)


# ============================================================
# Panel D: 截面图 κ̄_c vs consumers
# ============================================================

def compute_panel_d_data(n=N, K_bar=K_BAR, Pmax=PMAX, realizations=REALIZATIONS):
    """n_p=0 截面, 不同 q 值。返回均值和标准差。"""
    cache_file = CACHE_DIR / f"v3_panel_d_n{n}_k{K_bar}_r{realizations}.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return (data['q_values'], data['n_minus_range'],
                data['kappa_c_mean'], data['kappa_c_std'])

    q_values = np.array([0.0, 0.1, 0.4, 1.0])
    n_minus_range = np.arange(1, n, 1)  # consumers 1 to n-1

    kappa_c_mean = np.full((len(q_values), len(n_minus_range)), np.nan)
    kappa_c_std = np.full((len(q_values), len(n_minus_range)), np.nan)

    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        for qi, q in enumerate(q_values):
            print(f"Panel D: q={q}")
            for ni, n_minus in enumerate(n_minus_range):
                n_plus = n - n_minus
                mean_val, std_val = compute_kappa_c_stats_parallel(
                    n, K_bar, q, n_plus, n_minus, Pmax, realizations, pool
                )
                kappa_c_mean[qi, ni] = mean_val
                kappa_c_std[qi, ni] = std_val

    np.savez(cache_file, q_values=q_values, n_minus_range=n_minus_range,
             kappa_c_mean=kappa_c_mean, kappa_c_std=kappa_c_std)
    print(f"Panel D data cached to {cache_file}")
    return q_values, n_minus_range, kappa_c_mean, kappa_c_std


def plot_panel_d(ax, q_values, n_minus_range, kappa_c_mean, kappa_c_std):
    """匹配论文 Panel D 风格: 仅曲线 + (i) 标注, 无置信带。"""
    colors = ['#e8850c', '#4a7ebb', '#2ca02c', '#d62728']
    lws = [2.0, 1.5, 1.5, 1.5]

    for qi, q in enumerate(q_values):
        valid = ~np.isnan(kappa_c_mean[qi])
        x = n_minus_range[valid]
        y = kappa_c_mean[qi, valid]

        ax.plot(x, y, color=colors[qi], lw=lws[qi], label=f'$q = {q}$')

    ax.set_xlabel("Consumers", fontsize=12)
    ax.set_ylabel(r"$\overline{\kappa}_c$", fontsize=12)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.text(0.15, 0.92, "(i)", transform=ax.transAxes, fontsize=10,
            fontstyle='italic')
    ax.set_title("D", loc="left", fontweight="bold", fontsize=14)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("复现 Figure 1 — Variation in the critical coupling")
    print("=" * 60)

    # --- 计算数据 ---
    print("\n--- Computing Panel C data ---")
    configs_c, kappa_c_vals_c = compute_panel_c_data(
        n=N, K_bar=K_BAR, q=0.0, Pmax=PMAX,
        realizations=REALIZATIONS, step=2
    )

    print("\n--- Computing Panel D data ---")
    q_values, n_minus_range, kd_mean, kd_std = compute_panel_d_data(
        n=N, K_bar=K_BAR, Pmax=PMAX, realizations=REALIZATIONS
    )

    out_dir = Path(__file__).parent

    # --- Figure AB ---
    print("\n--- Assembling Figure AB ---")
    fig_ab, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_panel_a(ax_a)
    plot_panel_b(ax_b)
    fig_ab.tight_layout()
    fig_ab.savefig(out_dir / "figure1_AB.pdf", dpi=300, bbox_inches='tight')
    fig_ab.savefig(out_dir / "figure1_AB.png", dpi=200, bbox_inches='tight')
    print(f"Saved figure1_AB")

    # --- Figure CD ---
    print("--- Assembling Figure CD ---")
    fig_cd, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_panel_c(ax_c, configs_c, kappa_c_vals_c, n=N)
    plot_panel_d(ax_d, q_values, n_minus_range, kd_mean, kd_std)
    fig_cd.tight_layout()
    fig_cd.savefig(out_dir / "figure1_CD.pdf", dpi=300, bbox_inches='tight')
    fig_cd.savefig(out_dir / "figure1_CD.png", dpi=200, bbox_inches='tight')
    print(f"Saved figure1_CD")

    plt.show()


if __name__ == "__main__":
    main()
