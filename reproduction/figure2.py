"""
复现 Reference3 Figure 2 — Cascading Failure vs Edge Capacity

三个 Panel:
  A: (n+,n-,np)=(15,45,0) — survivors, overloads, desyncs vs α/α*
  B: (n+,n-,np)=(30,30,0) — 同上
  C: 两种网络配置的平均级联持续时间 T̄ vs α/α*
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp, RK45
from multiprocessing import Pool, cpu_count
from pathlib import Path
import warnings
import random

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 全局参数
# ============================================================
N = 60
K_BAR = 4
Q = 0.1
KAPPA = 5.0
I_INERTIA = 1.0
D_DAMP = 1.0
PMAX = 1.0
SYNCTOL = 3.0

ENSEMBLE_SIZE = 200      # 论文值; 减小以加速测试
ALPHA_RES = 50
ALPHA_MIN = 0.1
ALPHA_MAX = 2.5

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 网络生成与功率分配
# ============================================================

def generate_network(n, K_bar, q):
    """生成 Watts-Strogatz 小世界网络，返回 networkx 图 G。"""
    G = nx.connected_watts_strogatz_graph(n, K_bar, q)
    return G


def build_incidence_matrix(G):
    """从 networkx 图构建关联矩阵 E (n×m)。
    对每条边 (u,v)，E[u,e]=+1, E[v,e]=-1。"""
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    E = np.zeros((n, m))
    for j, (u, v) in enumerate(edges):
        E[u, j] = 1.0
        E[v, j] = -1.0
    return E


def adjacencymatrix_from_incidence(E):
    """A = diag(degrees) - E·E^T"""
    n = E.shape[0]
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
    """Swing equation 右端函数。
    psi = [omega_1..omega_n, theta_1..theta_n]"""
    omega = psi[:n]
    theta = psi[n:]
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    domega = (P - D * omega - kappa * coupling) / I
    dtheta = omega
    return np.concatenate([domega, dtheta])


def fsteadystate(theta, A, P, kappa):
    """稳态残差: P - κ·Σ_j A_ij·sin(θ_i - θ_j)"""
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def edgepower(theta, E, kappa):
    """边功率流: κ·sin(E^T·θ)"""
    dtheta = E.T @ theta
    return kappa * np.sin(dtheta)


# ============================================================
# 连通分量 (用 adjacency matrix)
# ============================================================

def connected_components(A):
    """返回 (num_components, list_of_index_lists)"""
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for u in range(n):
        if not visited[u]:
            # BFS
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
# 级联失效核心算法
# ============================================================

def swing_fracture_with_breakdown(E_full, active_edges, psi, nodeset, P,
                                  synctol, alpha, kappa, maxflow):
    """
    递归级联算法 (对应 Julia swingfracturewithbreakdown!)。

    参数:
        E_full: 全网关联矩阵 (n_total × m_total)
        active_edges: set of active edge indices (全局)
        psi: 状态向量 [omega, theta] (子网络大小)
        nodeset: 子网络节点索引列表 (全局索引)
        P: 全局功率向量
        synctol, alpha, kappa, maxflow: 参数

    返回: (surviving_edges, overloads, desyncs, unbalances)
    """
    tol = 1e-5
    nc = len(nodeset)

    # 提取子网络功率
    P1 = np.array([P[v] for v in nodeset])
    sourcecounter = np.sum(P1 > tol)
    sinkcounter = np.sum(P1 < -tol)

    # 找子网络的活跃边
    edgeset = []
    for v in nodeset:
        for j in active_edges:
            if abs(E_full[v, j]) > 0.5 and j not in edgeset:
                edgeset.append(j)
    edgeset = sorted(set(edgeset))
    ec = len(edgeset)

    if sourcecounter == 0 or sinkcounter == 0:
        return 0, 0, ec, 0

    # 平衡功率: homogeneous balancing
    delta = np.sum(P1) / 2.0
    ind_delta_sink = delta / sinkcounter
    ind_delta_source = delta / sourcecounter
    for i in range(nc):
        if P1[i] < -tol:
            P1[i] -= ind_delta_sink
        if P1[i] > tol:
            P1[i] -= ind_delta_source

    # 构建子网络关联矩阵 E1
    E1 = np.zeros((nc, ec))
    for i, v in enumerate(nodeset):
        for j_local, j_global in enumerate(edgeset):
            E1[i, j_local] = E_full[v, j_global]

    A1 = adjacencymatrix_from_incidence(E1)

    # --- Step-by-step ODE integration (matching Julia DiscreteCallback) ---
    def rhs(t, y):
        return fswing(y, A1, P1, nc, I_INERTIA, D_DAMP, kappa)

    solver = RK45(rhs, 0.0, psi, 500.0, rtol=1e-8, atol=1e-8)
    reason = 'timeout'
    while solver.status == 'running':
        solver.step()
        y = solver.y
        omega_cur = y[:nc]
        theta_cur = y[nc:]
        # Priority 1: desync
        if np.linalg.norm(omega_cur, 2) > synctol:
            reason = 'desync'
            break
        # Priority 2: overload
        flow_cur = edgepower(theta_cur, E1, kappa) / maxflow
        if np.any(np.abs(flow_cur) > alpha):
            reason = 'overload'
            break
        # Priority 3: convergence
        resid_cur = fsteadystate(theta_cur, A1, P1, kappa)
        if np.linalg.norm(resid_cur, 2) < 1e-6: 
            reason = 'converge'
            break

    psi_final = solver.y
    omega_final = psi_final[:nc]
    theta_final = psi_final[nc:]

    # Desync
    if reason == 'desync':
        return 0, 0, ec, 0
    # Fallback desync (for timeout case)
    if np.linalg.norm(omega_final, 2) > synctol:
        return 0, 0, ec, 0
    # Check overload at final state
    flow_final = edgepower(theta_final, E1, kappa) / maxflow
    has_overload = np.any(np.abs(flow_final) > alpha)
    if not has_overload:
        return ec, 0, 0, 0

    # 有过载: 移除过载边，递归处理子分量
    overload_count = 0
    survivor_local = []
    for j_local in range(ec):
        if abs(flow_final[j_local]) > alpha:
            active_edges.discard(edgeset[j_local])
            overload_count += 1
        else:
            survivor_local.append(j_local)

    # 构建剩余边的关联矩阵
    new_ec = len(survivor_local)
    if new_ec == 0:
        return 0, 0, 0, 0

    E2 = np.zeros((nc, new_ec))
    for j_new, j_old in enumerate(survivor_local):
        E2[:, j_new] = E1[:, j_old]

    Adj2 = adjacencymatrix_from_incidence(E2)
    num_comp, comp_table = connected_components(Adj2)

    # 递归处理每个分量
    desc_edges = 0
    desc_overloads = 0
    desc_desyncs = 0
    desc_unbalances = 0

    for comp in comp_table:
        n_sub = len(comp)
        nodesubset = [nodeset[c] for c in comp]
        omega_sub = omega_final[comp]
        theta_sub = theta_final[comp]
        psi_sub = np.concatenate([omega_sub, theta_sub])

        xe, oe, de, ue = swing_fracture_with_breakdown(
            E_full, active_edges, psi_sub, nodesubset, P,
            synctol, alpha, kappa, maxflow)
        desc_edges += xe
        desc_overloads += oe
        desc_desyncs += de
        desc_unbalances += ue

    return desc_edges, desc_overloads, desc_desyncs, desc_unbalances


def swing_fracture_with_time(E_full, active_edges, psi, nodeset, P,
                             synctol, alpha, kappa, maxflow):
    """
    类似 swing_fracture_with_breakdown 但追踪时间。
    返回: (surviving_edges, time_taken)
    """
    tol = 1e-5
    nc = len(nodeset)

    P1 = np.array([P[v] for v in nodeset])
    sourcecounter = np.sum(P1 > tol)
    sinkcounter = np.sum(P1 < -tol)

    # 找子网络的活跃边
    edgeset = []
    for v in nodeset:
        for j in active_edges:
            if abs(E_full[v, j]) > 0.5 and j not in edgeset:
                edgeset.append(j)
    edgeset = sorted(set(edgeset))
    ec = len(edgeset)

    if sourcecounter == 0 or sinkcounter == 0:
        return 0, 0.0

    # 平衡功率
    delta = np.sum(P1) / 2.0
    ind_delta_sink = delta / sinkcounter
    ind_delta_source = delta / sourcecounter
    for i in range(nc):
        if P1[i] < -tol:
            P1[i] -= ind_delta_sink
        if P1[i] > tol:
            P1[i] -= ind_delta_source

    # 构建子网络关联矩阵
    E1 = np.zeros((nc, ec))
    for i, v in enumerate(nodeset):
        for j_local, j_global in enumerate(edgeset):
            E1[i, j_local] = E_full[v, j_global]

    A1 = adjacencymatrix_from_incidence(E1)

    # --- Step-by-step ODE integration (matching Julia DiscreteCallback) ---
    def rhs(t, y):
        return fswing(y, A1, P1, nc, I_INERTIA, D_DAMP, kappa)

    solver = RK45(rhs, 0.0, psi, 500.0, rtol=1e-8, atol=1e-8) 
    reason = 'timeout'
    while solver.status == 'running':
        solver.step()
        y = solver.y
        omega_cur = y[:nc]
        theta_cur = y[nc:]
        if np.linalg.norm(omega_cur, 2) > synctol:
            reason = 'desync'
            break
        flow_cur = edgepower(theta_cur, E1, kappa) / maxflow
        if np.any(np.abs(flow_cur) > alpha):
            reason = 'overload'
            break
        resid_cur = fsteadystate(theta_cur, A1, P1, kappa)
        if np.linalg.norm(resid_cur, 2) < 1e-5:  #与级联time有关
            reason = 'converge'
            break

    t_end = solver.t
    psi_final = solver.y
    omega_final = psi_final[:nc]
    theta_final = psi_final[nc:]

    if reason == 'desync':
        return 0, 0.0
    if np.linalg.norm(omega_final, 2) > synctol:
        return 0, 0.0
    flow_final = edgepower(theta_final, E1, kappa) / maxflow
    has_overload = np.any(np.abs(flow_final) > alpha)
    if not has_overload:
        return ec, t_end

    # 有过载: 移除过载边，递归
    survivor_local = []
    for j_local in range(ec):
        if abs(flow_final[j_local]) > alpha:
            active_edges.discard(edgeset[j_local])
        else:
            survivor_local.append(j_local)

    new_ec = len(survivor_local)
    if new_ec == 0:
        return 0, t_end

    E2 = np.zeros((nc, new_ec))
    for j_new, j_old in enumerate(survivor_local):
        E2[:, j_new] = E1[:, j_old]

    Adj2 = adjacencymatrix_from_incidence(E2)
    num_comp, comp_table = connected_components(Adj2)

    desc_edges = 0
    times = []
    for comp in comp_table:
        nodesubset = [nodeset[c] for c in comp]
        omega_sub = omega_final[comp]
        theta_sub = theta_final[comp]
        psi_sub = np.concatenate([omega_sub, theta_sub])

        xe, te = swing_fracture_with_time(
            E_full, active_edges, psi_sub, nodesubset, P,
            synctol, alpha, kappa, maxflow)
        desc_edges += xe
        times.append(te)

    t_total = t_end + max(times) if times else t_end
    return desc_edges, t_total


# ============================================================
# 单个 ensemble 成员的计算
# ============================================================

def run_single_breakdown(args):
    """计算一个 ensemble 成员的 breakdown vs alpha。"""
    seed, n, ns, nd, q, k, kappa, alphas = args
    np.random.seed(seed)
    random.seed(seed)

    alpha_res = len(alphas)
    G = generate_network(n, k, q)
    E = build_incidence_matrix(G)
    m = E.shape[1]
    A = adjacencymatrix_from_incidence(E)
    P = assign_powers(n, ns, nd, PMAX)
    nodeset = list(range(n))

    # 求稳态
    psi0 = np.random.rand(2 * n)

    def rhs(t, y):
        return fswing(y, A, P, n, I_INERTIA, D_DAMP, kappa)

    sol = solve_ivp(rhs, [0, 100.0], psi0, method='RK45',
                    rtol=1e-8, atol=1e-8, max_step=1.0)

    if sol.status != 0:
        return None

    psi_ss = sol.y[:, -1]
    omega_ss = psi_ss[:n]
    theta_ss = psi_ss[n:]

    resid = fsteadystate(theta_ss, A, P, kappa)
    if np.linalg.norm(resid, 2) > 1e-3:
        return None

    # 计算边功率流并归一化
    flow = edgepower(theta_ss, E, kappa)
    flowmax = np.max(np.abs(flow))
    if flowmax < 1e-12:
        return None
    flow_norm = flow / flowmax

    # 找最大流量边
    d = np.argmax(np.abs(flow_norm))

    redges = np.zeros(alpha_res)
    roverloads = np.zeros(alpha_res)
    rdesyncs = np.zeros(alpha_res)
    runbalances = np.zeros(alpha_res)

    for ind, alpha_val in enumerate(alphas):
        # 重新激活所有边
        active_edges = set(range(m))

        # 移除最大流量边
        active_edges.discard(d)

        # 构建无 d 的关联矩阵
        remaining = sorted(active_edges)
        E1 = E[:, remaining]
        Adj1 = adjacencymatrix_from_incidence(E1)
        num_comp, comp_table = connected_components(Adj1)

        tot_surv = 0
        tot_overloads = 0
        tot_desyncs = 0
        tot_unbalances = 0

        for comp in comp_table:
            n_sub = len(comp)
            nodesubset = [nodeset[c] for c in comp]
            omega_sub = omega_ss[comp]
            theta_sub = theta_ss[comp]
            psi_sub = np.concatenate([omega_sub, theta_sub])

            # active_edges 需要每次 alpha 独立 (因为递归会修改)
            ae = set(range(m))
            ae.discard(d)

            xe, oe, de, ue = swing_fracture_with_breakdown(
                E, ae, psi_sub, nodesubset, P,
                SYNCTOL, alpha_val, kappa, flowmax)
            tot_surv += xe
            tot_overloads += oe
            tot_desyncs += de
            tot_unbalances += ue

        redges[ind] = tot_surv / m
        roverloads[ind] = (m - tot_surv - tot_desyncs - tot_unbalances) / m
        rdesyncs[ind] = tot_desyncs / m
        runbalances[ind] = tot_unbalances / m

    # 找 alpha_c
    after_trans = np.where(redges > 0.5)[0]
    alpha_c = alphas[after_trans[0]] if len(after_trans) > 0 else np.nan

    return redges, roverloads, rdesyncs, runbalances, alpha_c, flowmax


def run_single_duration(args):
    """计算一个 ensemble 成员的 duration vs alpha。"""
    seed, n, ns, nd, q, k, kappa, alphas = args
    np.random.seed(seed)
    random.seed(seed)

    alpha_res = len(alphas)
    G = generate_network(n, k, q)
    E = build_incidence_matrix(G)
    m = E.shape[1]
    A = adjacencymatrix_from_incidence(E)
    P = assign_powers(n, ns, nd, PMAX)
    nodeset = list(range(n))

    psi0 = np.random.rand(2 * n)

    def rhs(t, y):
        return fswing(y, A, P, n, I_INERTIA, D_DAMP, kappa)

    sol = solve_ivp(rhs, [0, 100.0], psi0, method='RK45',
                    rtol=1e-8, atol=1e-8, max_step=1.0)

    if sol.status != 0:
        return None

    psi_ss = sol.y[:, -1]
    omega_ss = psi_ss[:n]
    theta_ss = psi_ss[n:]

    resid = fsteadystate(theta_ss, A, P, kappa)
    if np.linalg.norm(resid, 2) > 1e-3:
        return None

    flow = edgepower(theta_ss, E, kappa)
    flowmax = np.max(np.abs(flow))
    if flowmax < 1e-12:
        return None
    flow_norm = flow / flowmax

    d = np.argmax(np.abs(flow_norm))

    redges = np.zeros(alpha_res)
    rtimes = np.zeros(alpha_res)

    for ind, alpha_val in enumerate(alphas):
        active_edges = set(range(m))
        active_edges.discard(d)

        remaining = sorted(active_edges)
        E1 = E[:, remaining]
        Adj1 = adjacencymatrix_from_incidence(E1)
        num_comp, comp_table = connected_components(Adj1)

        tot_surv = 0
        time_vec = []

        for comp in comp_table:
            nodesubset = [nodeset[c] for c in comp]
            omega_sub = omega_ss[comp]
            theta_sub = theta_ss[comp]
            psi_sub = np.concatenate([omega_sub, theta_sub])

            ae = set(range(m))
            ae.discard(d)

            xe, te = swing_fracture_with_time(
                E, ae, psi_sub, nodesubset, P,
                SYNCTOL, alpha_val, kappa, flowmax)
            tot_surv += xe
            time_vec.append(te)

        redges[ind] = tot_surv / m
        rtimes[ind] = max(time_vec) if time_vec else 0.0

    after_trans = np.where(redges > 0.5)[0]
    alpha_c = alphas[after_trans[0]] if len(after_trans) > 0 else np.nan

    return redges, rtimes, alpha_c, flowmax


# ============================================================
# 顶层级联实验
# ============================================================

def cascade_breakdown_vs_alpha(ensemble_size, n, ns, nd, q, k,
                               alpha_res, alpha_min, alpha_max, kappa,
                               cache_name):
    """计算 breakdown vs alpha (带缓存)。"""
    cache_file = CACHE_DIR / f"fig2_{cache_name}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  Loaded cached {cache_name}")
        return (data['alphas'], data['survivors'], data['overloads'],
                data['desyncs'], data['unbalances'], data['alpha_c_mean'])

    alphas = np.linspace(alpha_min, alpha_max, alpha_res)
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 10**9, size=ensemble_size)

    args_list = [(int(s), n, ns, nd, q, k, kappa, alphas) for s in seeds]

    n_workers = min(cpu_count(), 8)
    print(f"  Running {ensemble_size} ensemble members with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_single_breakdown, args_list)):
            results.append(r)
            done = i + 1
            valid_so_far = sum(1 for x in results if x is not None)
            print(f"\r  Progress: {done}/{ensemble_size} done, {valid_so_far} valid", end="", flush=True)
        print()

    # 过滤失败
    valid = [r for r in results if r is not None]
    print(f"  Valid: {len(valid)}/{ensemble_size}")

    if len(valid) == 0:
        return alphas, np.zeros(alpha_res), np.zeros(alpha_res), \
               np.zeros(alpha_res), np.zeros(alpha_res), np.nan

    all_surv = np.array([r[0] for r in valid])
    all_over = np.array([r[1] for r in valid])
    all_desync = np.array([r[2] for r in valid])
    all_unbal = np.array([r[3] for r in valid])
    all_ac = np.array([r[4] for r in valid])

    survivors = np.mean(all_surv, axis=0)
    overloads = np.mean(all_over, axis=0)
    desyncs = np.mean(all_desync, axis=0)
    unbalances = np.mean(all_unbal, axis=0)
    alpha_c_mean = np.nanmean(all_ac)

    np.savez(cache_file, alphas=alphas, survivors=survivors,
             overloads=overloads, desyncs=desyncs, unbalances=unbalances,
             alpha_c_mean=alpha_c_mean)
    print(f"  Cached to {cache_file}")

    return alphas, survivors, overloads, desyncs, unbalances, alpha_c_mean


def cascade_duration_vs_alpha(ensemble_size, n, ns, nd, q, k,
                              alpha_res, alpha_min, alpha_max, kappa,
                              cache_name):
    """计算 cascade duration vs alpha (带缓存)。"""
    cache_file = CACHE_DIR / f"fig2_{cache_name}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  Loaded cached {cache_name}")
        return (data['alphas'], data['survivors'], data['durations'],
                data['alpha_c_mean'])

    alphas = np.linspace(alpha_min, alpha_max, alpha_res)
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 10**9, size=ensemble_size)

    args_list = [(int(s), n, ns, nd, q, k, kappa, alphas) for s in seeds]

    n_workers = min(cpu_count(), 8)
    print(f"  Running {ensemble_size} ensemble members with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_single_duration, args_list)):
            results.append(r)
            done = i + 1
            valid_so_far = sum(1 for x in results if x is not None)
            print(f"\r  Progress: {done}/{ensemble_size} done, {valid_so_far} valid", end="", flush=True)
        print()

    valid = [r for r in results if r is not None]
    print(f"  Valid: {len(valid)}/{ensemble_size}")

    if len(valid) == 0:
        return alphas, np.zeros(alpha_res), np.zeros(alpha_res), np.nan

    all_surv = np.array([r[0] for r in valid])
    all_times = np.array([r[1] for r in valid])
    all_ac = np.array([r[2] for r in valid])

    survivors = np.mean(all_surv, axis=0)
    durations = np.mean(all_times, axis=0)
    alpha_c_mean = np.nanmean(all_ac)

    np.savez(cache_file, alphas=alphas, survivors=survivors,
             durations=durations, alpha_c_mean=alpha_c_mean)
    print(f"  Cached to {cache_file}")

    return alphas, survivors, durations, alpha_c_mean


# ============================================================
# 绘图
# ============================================================

def plot_panel_breakdown(ax, alphas, survivors, overloads, desyncs,
                         alpha_c_mean, title_label, marker=None):
    """绘制 Panel A 或 B: survivors/overloads/desyncs vs α/α*"""
    ax.plot(alphas, survivors, color='#1f77b4', lw=2.0, label='Survivors')
    ax.plot(alphas, overloads, color='#2ca02c', lw=2.0, label='Overloads')
    ax.plot(alphas, desyncs, color='#ff7f0e', lw=2.0, label='Desyncs')

    if not np.isnan(alpha_c_mean):
        ax.axvline(alpha_c_mean, color='red', ls='--', lw=1.0,
                   label=r'$\bar{\rho}$')
        
        if marker is not None:
            ymax = ax.get_ylim()[1]
            ax.text(alpha_c_mean, ymax * 1.04, marker,
                    ha='center', va='top', fontsize=11, color='black')

    ax.set_xlabel(r'$\alpha / \alpha^*$', fontsize=11)
    ax.set_ylabel('P', fontsize=11) 
    ax.set_xlim(ALPHA_MIN, ALPHA_MAX)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title_label, loc='left', fontweight='bold', fontsize=13)
    ax.legend(fontsize=8, loc='center right')


def plot_figure2(data_case1, data_case2, dur_case1, dur_case2):
    """绘制 Figure 2 (3 panels)。"""
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: (15, 45, 0)
    alphas1, surv1, over1, desync1, unbal1, ac1 = data_case1
    plot_panel_breakdown(ax_a, alphas1, surv1, over1, desync1, ac1, 'A', marker='(i)')
    ax_a.text(0.05, 0.92, r'$(n_+,n_-,n_p)=(15,45,0)$',
              transform=ax_a.transAxes, fontsize=8)

    # Panel B: (30, 30, 0)
    alphas2, surv2, over2, desync2, unbal2, ac2 = data_case2
    plot_panel_breakdown(ax_b, alphas2, surv2, over2, desync2, ac2, 'B', marker='(ii)')
    ax_b.text(0.05, 0.92, r'$(n_+,n_-,n_p)=(30,30,0)$',
              transform=ax_b.transAxes, fontsize=8)

    # Panel C: 平均级联持续时间
    a_dur1, _, dur1, ac_dur1 = dur_case1
    a_dur2, _, dur2, ac_dur2 = dur_case2

    ax_c.plot(a_dur1, dur1, color='#1f77b4', lw=2.0,
              label=r'$(15,45,0)$')
    ax_c.plot(a_dur2, dur2, color='#ff7f0e', lw=2.0,
              label=r'$(30,30,0)$')

    ax_c.set_xlabel(r'$\alpha / \alpha^*$', fontsize=11)
    ax_c.set_ylabel(r'$\bar{T}$', fontsize=11)
    ax_c.set_xlim(ALPHA_MIN, ALPHA_MAX)
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=13)
    ax_c.legend(fontsize=9)
    
    if not np.isnan(ac1):
        ax_c.axvline(ac1, color='red', lw=2.0, alpha=0.6,ls='--')
        ax_c.text(ac1, 1.02, '(i)', transform=ax_c.get_xaxis_transform(),
              ha='center', va='bottom', fontsize=10, fontstyle='italic')

    if not np.isnan(ac2):
        ax_c.axvline(ac2, color='red', lw=2.0, alpha=0.6,ls='--')
        ax_c.text(ac2, 1.02, '(ii)', transform=ax_c.get_xaxis_transform(),
              ha='center', va='bottom', fontsize=10, fontstyle='italic')

    # 标注 (i) 和 (ii)
    #ax_a.text(0.85, 0.85, '(i)', transform=ax_a.transAxes,
              #fontsize=10, fontstyle='italic')
    #ax_b.text(0.85, 0.85, '(ii)', transform=ax_b.transAxes,
              #fontsize=10, fontstyle='italic')
    #ax_c.text(0.15, 0.90, '(i)', transform=ax_c.transAxes,
              #fontsize=10, fontstyle='italic', color='#1f77b4')
    #ax_c.text(0.35, 0.90, '(ii)', transform=ax_c.transAxes,
              #fontsize=10, fontstyle='italic', color='#ff7f0e')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure2.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "figure2.png", dpi=200, bbox_inches='tight')
    print(f"Saved figure2.pdf and figure2.png to {OUTPUT_DIR}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("复现 Figure 2 — Cascading Failure vs Edge Capacity")
    print("=" * 60)

    # Case 1: (15, 45, 0)
    print("\n--- Case 1: (n+,n-,np)=(15,45,0) breakdown ---")
    data_case1 = cascade_breakdown_vs_alpha(
        ENSEMBLE_SIZE, N, 15, 45, Q, K_BAR,
        ALPHA_RES, ALPHA_MIN, ALPHA_MAX, KAPPA,
        "breakdown_15_45_0")

    print("\n--- Case 1: (n+,n-,np)=(15,45,0) duration ---")
    dur_case1 = cascade_duration_vs_alpha(
        ENSEMBLE_SIZE, N, 15, 45, Q, K_BAR,
        ALPHA_RES, ALPHA_MIN, ALPHA_MAX, KAPPA,
        "duration_15_45_0")

    # Case 2: (30, 30, 0)
    print("\n--- Case 2: (n+,n-,np)=(30,30,0) breakdown ---")
    data_case2 = cascade_breakdown_vs_alpha(
        ENSEMBLE_SIZE, N, 30, 30, Q, K_BAR,
        ALPHA_RES, ALPHA_MIN, ALPHA_MAX, KAPPA,
        "breakdown_30_30_0")

    print("\n--- Case 2: (n+,n-,np)=(30,30,0) duration ---")
    dur_case2 = cascade_duration_vs_alpha(
        ENSEMBLE_SIZE, N, 30, 30, Q, K_BAR,
        ALPHA_RES, ALPHA_MIN, ALPHA_MAX, KAPPA,
        "duration_30_30_0")

    # 绘图
    print("\n--- Plotting ---")
    plot_figure2(data_case1, data_case2, dur_case1, dur_case2)

    print("\nDone!")


if __name__ == "__main__":
    main()
