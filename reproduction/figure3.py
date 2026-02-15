"""
复现 Reference3 Figure 3 — Variation in resilience with generator-consumer numbers

Panel A: ρ = α_c/α* 分布直方图, lattice (q=0), n=100, n+=n-=50
Panel B: ρ = α_c/α* 分布直方图, small-world (q=0.1, K̄=4), n=100, n+=n-=50
Panel C: Ternary simplex 热力图 (ρ̄), lattice (q=0)
Panel D: Ternary simplex 热力图 (ρ̄), small-world (q=0.1)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import networkx as nx
from tqdm import tqdm
from scipy.integrate import solve_ivp, RK45
from scipy.stats import lognorm
from multiprocessing import Pool, cpu_count
from pathlib import Path
import warnings
import random
import time

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 全局参数
# ============================================================
N_HIST = 100          # Panels A,B: 网络节点数 (论文值=100)
K_BAR = 4             # 平均度
KAPPA = 5.0           # 耦合强度
I_INERTIA = 1.0       # 惯量
D_DAMP = 1.0          # 阻尼
PMAX = 1.0            # 最大功率
SYNCTOL = 3.0         # 同步容限

ENSEMBLE_HIST = 200   # Panels A,B: ensemble 大小 (论文值)

N_SIMPLEX = 100      # Panels C,D: simplex 节点总数 (论文值)
ENSEMBLE_SIMPLEX = 50  # Panels C,D: 每配置的 ensemble 大小
SIMPLEX_STEP = 4       # Panels C,D: simplex 网格步长 (1=全分辨率, 2=1/4计算量)

BISECT_TOL = 5e-4     # 二分法精度

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRI = np.sqrt(3) / 2


# ============================================================
# 网络生成与功率分配
# ============================================================

def generate_network(n, K_bar, q):
    """生成 Watts-Strogatz 网络; q=0 时为环格子"""
    if q == 0.0:
        return nx.watts_strogatz_graph(n, K_bar, 0.0)
    return nx.connected_watts_strogatz_graph(n, K_bar, q)


def build_incidence_matrix(G):
    """从 networkx 图构建关联矩阵 E (n×m)"""
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    E = np.zeros((n, m))
    for j, (u, v) in enumerate(edges):
        E[u, j] = 1.0
        E[v, j] = -1.0
    return E


def adjacencymatrix_from_incidence(E):
    """邻接矩阵 A = diag(deg) - E·E^T"""
    degrees = np.sum(np.abs(E), axis=1)
    L = E @ E.T
    return np.diag(degrees) - L


def assign_powers(n, n_plus, n_minus, Pmax):
    """功率分配: P_gen = +Pmax/n_+, P_con = -Pmax/n_-"""
    P = np.zeros(n)
    if n_plus == 0 or n_minus == 0:
        return P
    indices = np.random.permutation(n)
    gen_idx = indices[:n_plus]
    con_idx = indices[n_plus:n_plus + n_minus]
    P[gen_idx] = Pmax / n_plus
    P[con_idx] = -Pmax / n_minus
    return P


# ============================================================
# Swing Equation 核心函数
# ============================================================

def fswing(psi, A, P, n, I, D, kappa):
    """Swing equation 右端函数"""
    omega = psi[:n]
    theta = psi[n:]
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    domega = (P - D * omega - kappa * coupling) / I
    return np.concatenate([domega, omega])


def fsteadystate(theta, A, P, kappa):
    """稳态残差"""
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def edgepower(theta, E, kappa):
    """边功率流: κ·sin(E^T·θ)"""
    return kappa * np.sin(E.T @ theta)


# ============================================================
# 连通分量
# ============================================================

def connected_components(A):
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for u in range(n):
        if not visited[u]:
            visited[u] = True
            queue = [u]
            comp = [u]
            while queue:
                v = queue.pop(0)
                for w in range(n):
                    if A[v, w] > 0.5 and not visited[w]:
                        visited[w] = True
                        queue.append(w)
                        comp.append(w)
            components.append(comp)
    return len(components), components


# ============================================================
# 级联失效 (递归, 返回存活边数)
# ============================================================

def swingfracture(E_full, active_edges, psi, nodeset, P,
                  synctol, alpha, kappa, maxflow):
    """递归级联算法, 返回存活边数"""
    tol = 1e-5
    nc = len(nodeset)

    P1 = np.array([P[v] for v in nodeset])
    sourcecounter = int(np.sum(P1 > tol))
    sinkcounter = int(np.sum(P1 < -tol))

    # 找子网络的活跃边
    edgeset = sorted({j for v in nodeset
                      for j in active_edges
                      if abs(E_full[v, j]) > 0.5})
    ec = len(edgeset)

    if sourcecounter == 0 or sinkcounter == 0:
        return 0

    # 平衡功率
    delta = np.sum(P1) / 2.0
    for i in range(nc):
        if P1[i] < -tol:
            P1[i] -= delta / sinkcounter
        if P1[i] > tol:
            P1[i] -= delta / sourcecounter

    # 构建子网络关联矩阵
    E1 = np.zeros((nc, ec))
    for i, v in enumerate(nodeset):
        for j_local, j_global in enumerate(edgeset):
            E1[i, j_local] = E_full[v, j_global]

    A1 = adjacencymatrix_from_incidence(E1)

    def rhs(t, y):
        return fswing(y, A1, P1, nc, I_INERTIA, D_DAMP, kappa)

    solver = RK45(rhs, 0.0, psi, 500.0, rtol=1e-8, atol=1e-8)
    reason = 'timeout'
    while solver.status == 'running':
        solver.step()
        y = solver.y
        if np.linalg.norm(y[:nc], 2) > synctol:
            reason = 'desync'
            break
        flow_cur = edgepower(y[nc:], E1, kappa) / maxflow
        if np.any(np.abs(flow_cur) > alpha):
            reason = 'overload'
            break
        if np.linalg.norm(fsteadystate(y[nc:], A1, P1, kappa), 2) < 1e-6:
            reason = 'converge'
            break

    psi_f = solver.y
    omega_f, theta_f = psi_f[:nc], psi_f[nc:]

    if reason == 'desync' or np.linalg.norm(omega_f, 2) > synctol:
        return 0

    flow_f = edgepower(theta_f, E1, kappa) / maxflow
    if not np.any(np.abs(flow_f) > alpha):
        return ec

    # 移除过载边, 递归
    survivor_local = []
    for j_local in range(ec):
        if abs(flow_f[j_local]) > alpha:
            active_edges.discard(edgeset[j_local])
        else:
            survivor_local.append(j_local)

    if not survivor_local:
        return 0

    E2 = E1[:, survivor_local]
    Adj2 = adjacencymatrix_from_incidence(E2)
    _, comp_table = connected_components(Adj2)

    return sum(
        swingfracture(
            E_full, active_edges,
            np.concatenate([omega_f[comp], theta_f[comp]]),
            [nodeset[c] for c in comp],
            P, synctol, alpha, kappa, maxflow)
        for comp in comp_table
    )


# ============================================================
# 辅助: 准备单个网络 (求稳态, 计算流)
# ============================================================

def prepare_network(n, ns, nd, q, k, kappa):
    """创建网络并求稳态, 返回 dict 或 None"""
    G = generate_network(n, k, q)
    E = build_incidence_matrix(G)
    m = E.shape[1]
    A = adjacencymatrix_from_incidence(E)
    P = assign_powers(n, ns, nd, PMAX)

    psi0 = np.random.rand(2 * n)

    def rhs(t, y):
        return fswing(y, A, P, n, I_INERTIA, D_DAMP, kappa)

    sol = solve_ivp(rhs, [0, 250.0], psi0, method='RK45',
                    rtol=1e-8, atol=1e-8, max_step=1.0)
    if sol.status != 0:
        return None

    psi_ss = sol.y[:, -1]
    theta_ss = psi_ss[n:]
    omega_ss = psi_ss[:n]

    if np.linalg.norm(fsteadystate(theta_ss, A, P, kappa), 2) > 1e-3:
        return None

    flow = edgepower(theta_ss, E, kappa)
    flowmax = np.max(np.abs(flow))
    if flowmax < 1e-12:
        return None

    return {
        'E': E, 'P': P, 'omega_ss': omega_ss, 'theta_ss': theta_ss,
        'flowmax': flowmax, 'd': int(np.argmax(np.abs(flow / flowmax))),
        'm': m, 'n': n
    }


def run_cascade_single(member, alpha):
    """对单个网络在给定 alpha 下运行级联, 返回 S = 存活边比例"""
    E = member['E']
    P = member['P']
    omega_ss = member['omega_ss']
    theta_ss = member['theta_ss']
    flowmax = member['flowmax']
    d = member['d']
    m = member['m']
    n = member['n']
    nodeset = list(range(n))

    active_edges = set(range(m))
    active_edges.discard(d)

    remaining = sorted(active_edges)
    E1 = E[:, remaining]
    Adj1 = adjacencymatrix_from_incidence(E1)
    _, comp_table = connected_components(Adj1)

    tot_surv = 0
    for comp in comp_table:
        nodesubset = [nodeset[c] for c in comp]
        psi_sub = np.concatenate([omega_ss[comp], theta_ss[comp]])
        ae = set(range(m))
        ae.discard(d)
        tot_surv += swingfracture(E, ae, psi_sub, nodesubset, P,
                                  SYNCTOL, alpha, KAPPA, flowmax)

    return tot_surv / m


# ============================================================
# Panel A, B: 单网络二分法找 α_c (outer bisection)
# ============================================================

def find_alpha_c_single(args):
    """对单个网络做二分法找 α_c"""
    seed, n, ns, nd, q, k, kappa, tol = args
    np.random.seed(seed)
    random.seed(seed)

    member = prepare_network(n, ns, nd, q, k, kappa)
    if member is None:
        return np.nan

    alpha = 0.01
    stepsize = 0.3
    beneath0 = True

    while abs(stepsize) > tol:
        S = run_cascade_single(member, alpha)
        beneath1 = (S <= 0.5)
        if beneath1 != beneath0:
            stepsize = -stepsize / 2.0
        beneath0 = beneath1
        alpha += stepsize

    return alpha


def compute_histogram(ensemble_size, n, ns, nd, q, k, kappa, tol, cache_name):
    """计算 α_c 的分布 (带缓存)"""
    cache_file = CACHE_DIR / f"fig3_{cache_name}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  Loaded cached {cache_name}")
        return data['alpha_c_values']

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 10**9, size=ensemble_size)
    args_list = [(int(s), n, ns, nd, q, k, kappa, tol) for s in seeds]

    n_workers = min(cpu_count(), 8)
    print(f"  Running {ensemble_size} bisections with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = []
        for r in tqdm(pool.imap_unordered(find_alpha_c_single, args_list),
                      total=ensemble_size, desc="  Panel A/B"):
            results.append(r)

    alpha_c_values = np.array(results)
    valid_count = np.sum(~np.isnan(alpha_c_values))
    print(f"  Valid: {valid_count}/{ensemble_size}")

    np.savez(cache_file, alpha_c_values=alpha_c_values)
    print(f"  Cached to {cache_file}")
    return alpha_c_values


# ============================================================
# Panel C, D: Ensemble 二分法 (simplex 中每个配置)
# ============================================================

def ensemble_bisection(args):
    """对一个 (ns, nd, ne) 配置做 ensemble 二分法, 返回 α_c"""
    ns, nd, ne, q, k, kappa, tol, ens_size, base_seed = args
    n = ns + nd + ne

    if ns < 1 or nd < 1:
        return ns, nd, ne, np.nan

    rng = np.random.default_rng(base_seed)

    # 1. 创建所有 ensemble 成员
    members = []
    for _ in range(ens_size):
        seed = int(rng.integers(0, 10**9))
        np.random.seed(seed)
        random.seed(seed)
        m = prepare_network(n, ns, nd, q, k, kappa)
        if m is not None:
            members.append(m)

    if len(members) < 3:
        return ns, nd, ne, np.nan

    # 2. 二分法: 在所有 ensemble 成员上做
    alpha = 0.01
    stepsize = 0.3
    beneath0 = True

    while abs(stepsize) > tol:
        S_list = [run_cascade_single(m, alpha) for m in members]
        S_mean = np.mean(S_list)
        beneath1 = (S_mean <= 0.5)
        if beneath1 != beneath0:
            stepsize = -stepsize / 2.0
        beneath0 = beneath1
        alpha += stepsize

    return ns, nd, ne, alpha


def compute_simplex(n_simplex, q, k, kappa, tol, ens_size, cache_name, step=1):
    """计算 simplex 热力图数据 (带缓存)"""
    cache_file = CACHE_DIR / f"fig3_{cache_name}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  Loaded cached {cache_name}")
        return data['rho_matrix']

    # 生成所有 (ns, nd, ne) 配置
    configs = []
    for nd in range(1, n_simplex, step):
        for ns in range(1, n_simplex - nd, step):
            ne = n_simplex - ns - nd
            configs.append((ns, nd, ne))

    print(f"  {len(configs)} configs (step={step}), ensemble={ens_size}, n={n_simplex}")

    args_list = [
        (ns, nd, ne, q, k, kappa, tol, ens_size, 42 + i)
        for i, (ns, nd, ne) in enumerate(configs)
    ]

    n_workers = min(cpu_count(), 8)
    print(f"  Running with {n_workers} workers...")

    rho_matrix = np.full((n_simplex, n_simplex), np.nan)

    completed = 0
    t_start = time.time()
    with Pool(n_workers) as pool:
        for ns, nd, ne, ac in pool.imap_unordered(ensemble_bisection, args_list):
            rho_matrix[ns - 1, nd - 1] = ac
            completed += 1
            elapsed = time.time() - t_start
            eta = elapsed / completed * (len(configs) - completed)
            print(f"\r  Panel C/D: {completed}/{len(configs)} "
                  f"({100*completed/len(configs):.1f}%) "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s", end="", flush=True)
    print()

    # 填充未计算的点 (cubic 插值, 平滑渐变)
    if step > 1:
        from scipy.interpolate import griddata
        known_points = []
        known_values = []
        for i in range(n_simplex):
            for j in range(n_simplex):
                if not np.isnan(rho_matrix[i, j]):
                    known_points.append((i, j))
                    known_values.append(rho_matrix[i, j])
        if known_points:
            known_points = np.array(known_points)
            known_values = np.array(known_values)
            grid_i, grid_j = np.mgrid[0:n_simplex, 0:n_simplex]
            rho_matrix = griddata(known_points, known_values,
                                  (grid_i, grid_j), method='cubic',
                                  fill_value=np.nan)

    np.savez(cache_file, rho_matrix=rho_matrix)
    print(f"  Cached to {cache_file}")
    return rho_matrix


# ============================================================
# 三角坐标转换
# ============================================================

def ternary_to_cart(n_plus, n_minus, n_passive, n_total):
    """三角坐标 → 笛卡尔坐标
    BL = all generators, BR = all consumers, Top = all passive"""
    s = float(n_total)
    x = n_minus / s + 0.5 * n_passive / s
    y = n_passive / s * H_TRI
    return x, y


# ============================================================
# 绘图
# ============================================================

def plot_hist_panel(ax, alpha_c_values, label):
    """绘制 Panel A 或 B: ρ 分布直方图 + lognormal 拟合"""
    color = "#c24c51"
    valid = alpha_c_values[~np.isnan(alpha_c_values)]
    if len(valid) < 5:
        ax.text(0.5, 0.5, 'Insufficient data',
                transform=ax.transAxes, ha='center')
        return

    rho_mean = np.mean(valid)

    # 直方图 (原始 ρ 空间)
    H, bins = np.histogram(valid, density=True, bins=30)
    xs = (bins[:-1] + bins[1:]) / 2
    bin_width_norm = (bins[1] - bins[0]) / rho_mean

    # 绘制: x轴=ρ/ρ̄, y轴=ρ̄·P(ρ)
    ax.bar(xs / rho_mean, H * rho_mean, width=bin_width_norm,
           facecolor=color, alpha=0.7, edgecolor='none')

    # Lognormal 拟合
    args = lognorm.fit(valid)
    x_fit = np.linspace(bins[0], bins[-1], 200)
    pdf_fit = lognorm.pdf(x_fit, *args)
    ax.plot(x_fit / rho_mean, pdf_fit * rho_mean, 'k-', lw=2.5)

    ax.set_xlabel(r'$\rho/\bar{\rho}$', fontsize=14)
    ax.set_ylabel(r'$\bar{\rho}P(\rho)$', fontsize=14, rotation=0, labelpad=35)
    ax.set_xlim([0.35, 1.9])
    ax.set_xticks([0.35, 1.9])
    ax.set_yticks([0, 3])
    ax.set_ylim([0, 3])
    ax.set_title(label, loc='left', fontweight='bold', fontsize=15)


def plot_simplex_panel(ax, rho_matrix, n_simplex, label, vmin=0.9, vmax=1.6):
    """绘制三角形 simplex 热力图 (discrete cell-based, python-ternary style)"""
    N = n_simplex
    cmap = plt.cm.Reds
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    def get_val(ns, nd):
        ne = N - ns - nd
        if ns < 1 or nd < 1 or ne < 0 or ns >= N or nd >= N:
            return np.nan
        return rho_matrix[ns - 1, nd - 1]

    polygons = []
    colors = []

    # Upward triangles: for each data point (ns, nd, ne) with ns≥1, nd≥1, ne≥1
    # Triangle vertices: (ns, nd, ne), (ns+1, nd, ne-1), (ns, nd+1, ne-1)
    for nd in range(1, N):
        for ns in range(1, N - nd):
            ne = N - ns - nd
            if ne < 1:
                continue
            val = get_val(ns, nd)
            if np.isnan(val) or val <= 0:
                continue
            v1 = ternary_to_cart(ns, nd, ne, N)
            v2 = ternary_to_cart(ns + 1, nd, ne - 1, N)
            v3 = ternary_to_cart(ns, nd + 1, ne - 1, N)
            polygons.append([v1, v2, v3])
            colors.append(val)

    # Downward triangles (fill gaps): vertices at
    # (ns+1, nd, ne-1), (ns, nd+1, ne-1), (ns+1, nd+1, ne-2)
    # Color = average of surrounding upward triangle values
    for nd in range(1, N):
        for ns in range(1, N - nd):
            ne = N - ns - nd
            if ne < 2:
                continue
            # Three surrounding upward triangles:
            surrounding = [get_val(ns, nd), get_val(ns + 1, nd, ),
                           get_val(ns, nd + 1)]
            valid = [v for v in surrounding if not np.isnan(v) and v > 0]
            if not valid:
                continue
            v1 = ternary_to_cart(ns + 1, nd, ne - 1, N)
            v2 = ternary_to_cart(ns, nd + 1, ne - 1, N)
            v3 = ternary_to_cart(ns + 1, nd + 1, ne - 2, N)
            polygons.append([v1, v2, v3])
            colors.append(np.mean(valid))

    if not polygons:
        ax.text(0.5, 0.5, 'Insufficient data',
                transform=ax.transAxes, ha='center')
        return

    pc = PolyCollection(polygons, array=np.array(colors), cmap=cmap, norm=norm,
                        edgecolors='face', linewidths=0.3)
    ax.add_collection(pc)

    # 三角形顶点
    c_bl = ternary_to_cart(N, 0, 0, N)    # 左下: generators
    c_br = ternary_to_cart(0, N, 0, N)    # 右下: consumers
    c_top = ternary_to_cart(0, 0, N, N)   # 上: passive

    # 双边框: 粗外框 + 细内框
    corners = np.array([c_bl, c_br, c_top, c_bl])
    ax.plot(corners[:, 0], corners[:, 1], 'k-', lw=2.5)
    # 内缩边框
    center = np.array([(c_bl[0] + c_br[0] + c_top[0]) / 3,
                        (c_bl[1] + c_br[1] + c_top[1]) / 3])
    shrink = 0.015
    inner = np.array([c_bl, c_br, c_top, c_bl])
    for i in range(len(inner)):
        inner[i] = inner[i] + shrink * (center - inner[i])
    ax.plot(inner[:, 0], inner[:, 1], 'k-', lw=0.6)

    # 轴标签 — 沿三角形边
    mid_left_x = (c_bl[0] + c_top[0]) / 2 - 0.06
    mid_left_y = (c_bl[1] + c_top[1]) / 2
    ax.text(mid_left_x, mid_left_y, r'$\leftarrow$ Generators',
            fontsize=10, rotation=60, ha='center', va='center')

    mid_bot_x = (c_bl[0] + c_br[0]) / 2
    mid_bot_y = c_bl[1] - 0.06
    ax.text(mid_bot_x, mid_bot_y, r'Consumers $\rightarrow$',
            fontsize=10, ha='center', va='top')

    mid_right_x = (c_br[0] + c_top[0]) / 2 + 0.06
    mid_right_y = (c_br[1] + c_top[1]) / 2
    ax.text(mid_right_x, mid_right_y, r'$\leftarrow$ Passive',
            fontsize=10, rotation=-60, ha='center', va='center')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(label, loc='left', fontweight='bold', fontsize=15)

    # Set limits to match data extent
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, H_TRI + 0.08)

    return pc


def plot_figure3(hist_q0, hist_q1, rho_q0, rho_q1, n_simplex):
    """绘制完整 Figure 3 (4 panels)"""
    font = {'size': 14}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(11, 9))

    # 上排: 直方图
    ax_a = fig.add_subplot(221)
    ax_b = fig.add_subplot(222)
    # 下排: simplex
    ax_c = fig.add_subplot(223)
    ax_d = fig.add_subplot(224)

    plot_hist_panel(ax_a, hist_q0, 'A')
    plot_hist_panel(ax_b, hist_q1, 'B')

    plot_simplex_panel(ax_c, rho_q0, n_simplex, 'C')
    plot_simplex_panel(ax_d, rho_q1, n_simplex, 'D')

    # 共享 colorbar — positioned between panels C and D
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds,
                               norm=plt.Normalize(0.9, 1.6))
    sm.set_array([])
    # Between C and D, vertically centered in lower row
    cbar_ax = fig.add_axes([0.44, 0.22, 0.12, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0.9, 1.6])
    cbar.set_label(r'$\bar{\rho}$', fontsize=14)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUTPUT_DIR / "figure3.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "figure3.png", dpi=200, bbox_inches='tight')
    print(f"Saved figure3.pdf and figure3.png to {OUTPUT_DIR}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("复现 Figure 3 — Variation in resilience")
    print("=" * 60)

    # Panel A: lattice q=0, n=100, n+=n-=50
    print("\n--- Panel A: histogram q=0 ---")
    hist_q0 = compute_histogram(
        ENSEMBLE_HIST, N_HIST, 50, 50, 0.0, K_BAR, KAPPA, BISECT_TOL,
        "hist_q0_n100")

    # Panel B: small-world q=0.1, n=100, n+=n-=50
    print("\n--- Panel B: histogram q=0.1 ---")
    hist_q1 = compute_histogram(
        ENSEMBLE_HIST, N_HIST, 50, 50, 0.1, K_BAR, KAPPA, BISECT_TOL,
        "hist_q01_n100")

    # Panel C: simplex q=0
    print("\n--- Panel C: simplex q=0 ---")
    rho_q0 = compute_simplex(
        N_SIMPLEX, 0.0, K_BAR, KAPPA, BISECT_TOL, ENSEMBLE_SIMPLEX,
        f"simplex_q0_n{N_SIMPLEX}_s{SIMPLEX_STEP}", step=SIMPLEX_STEP)

    # Panel D: simplex q=0.1
    print("\n--- Panel D: simplex q=0.1 ---")
    rho_q1 = compute_simplex(
        N_SIMPLEX, 0.1, K_BAR, KAPPA, BISECT_TOL, ENSEMBLE_SIMPLEX,
        f"simplex_q01_n{N_SIMPLEX}_s{SIMPLEX_STEP}", step=SIMPLEX_STEP)

    # 绘图
    print("\n--- Plotting ---")
    plot_figure3(hist_q0, hist_q1, rho_q0, rho_q1, N_SIMPLEX)
    print("\nDone!")


if __name__ == "__main__":
    main()
