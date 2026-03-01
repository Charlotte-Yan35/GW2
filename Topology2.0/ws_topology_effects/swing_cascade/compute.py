"""
ws_cascade_compute.py — 级联 Bisection 临界 Alpha 计算 (Swing ODE 方法)

基于 Swing 方程的级联失效计算，移植自 reproduction/figure3.py 的
swingfracture 实现。

核心算法：
  1. 生成 ensemble 网络，用 RK45 积分 Swing 方程求稳态
  2. 移除最大负载边作为触发
  3. 递归级联：RK45 实时积分 → desync/overload/converge 三重监测
  4. 二分搜索临界 alpha（50% 边存活阈值）
  5. 在 K×q 全网格上扫描
"""

import pickle
import numpy as np
from scipy.integrate import solve_ivp, RK45
from tqdm import tqdm

from ws_config import (
    N, PCC_NODE, HOUSEHOLD_NODES,
    K_list, q_list, realizations,
    RATIO_CONFIGS, CACHE_DIR,
    KAPPA_CASCADE, I_INERTIA, D_DAMP, SYNCTOL, BISECT_TOL,
)
from ws_compute import generate_ws_network, build_power_vector


# ====================================================================
# 1. Swing 方程核心函数（移植自 figure3.py）
# ====================================================================

def fswing(psi, A, P, n, I, D, kappa):
    """Swing equation 右端函数
    psi = [omega_0..omega_{n-1}, theta_0..theta_{n-1}]
    """
    omega = psi[:n]
    theta = psi[n:]
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    domega = (P - D * omega - kappa * coupling) / I
    return np.concatenate([domega, omega])


def fsteadystate(theta, A, P, kappa):
    """稳态残差: P - κ·Σ_j A_ij·sin(θ_i − θ_j)"""
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def edgepower(theta, E, kappa):
    """边功率流: κ·sin(E^T·θ)"""
    return kappa * np.sin(E.T @ theta)


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


def connected_components(A):
    """BFS 连通分量检测"""
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


# ====================================================================
# 2. 递归级联 swingfracture（移植自 figure3.py）
# ====================================================================

def swingfracture(E_full, active_edges, psi, nodeset, P,
                  synctol, alpha, kappa, maxflow):
    """递归级联算法, 返回存活边数

    Parameters
    ----------
    E_full : ndarray (n_total, m_total)  全局关联矩阵
    active_edges : set of int            当前活跃边的全局索引集合（会被修改）
    psi : ndarray (2*nc,)                当前分量的初始状态 [omega, theta]
    nodeset : list of int                当前分量的全局节点索引
    P : ndarray (n_total,)               全局功率向量
    synctol : float                      失同步容限
    alpha : float                        过载阈值
    kappa : float                        耦合强度
    maxflow : float                      初始最大边流（用于归一化）

    Returns
    -------
    int  存活边数
    """
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

    if ec == 0:
        return 0

    # 齐次再平衡
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


# ====================================================================
# 3. Ensemble 准备（Swing ODE 求稳态）
# ====================================================================

def _prepare_ensemble(ratio_name, K, q, n_realizations, base_seed):
    """生成 ensemble 网络并用 Swing ODE 求稳态。

    Returns
    -------
    ensemble : list of dict, 每个 dict 包含:
        E, P, omega_ss, theta_ss, flowmax, d, m, n
    """
    kappa = KAPPA_CASCADE
    ensemble = []
    for r in range(n_realizations):
        seed = base_seed + r
        G = generate_ws_network(N, K, q, seed=seed)
        P = build_power_vector(ratio_name, seed=seed + 1)

        E = build_incidence_matrix(G)
        m = E.shape[1]
        n = N
        A = adjacencymatrix_from_incidence(E)

        # 随机初始条件, 积分 Swing 方程求稳态
        rng = np.random.default_rng(seed + 2)
        psi0 = rng.random(2 * n)

        def rhs(t, y):
            return fswing(y, A, P, n, I_INERTIA, D_DAMP, kappa)

        sol = solve_ivp(rhs, [0, 250.0], psi0, method='RK45',
                        rtol=1e-8, atol=1e-8, max_step=1.0)
        if sol.status != 0:
            continue

        psi_ss = sol.y[:, -1]
        omega_ss = psi_ss[:n]
        theta_ss = psi_ss[n:]

        # 检查是否收敛到稳态
        if np.linalg.norm(fsteadystate(theta_ss, A, P, kappa), 2) > 1e-3:
            continue

        flow = edgepower(theta_ss, E, kappa)
        flowmax = np.max(np.abs(flow))
        if flowmax < 1e-12:
            continue

        d = int(np.argmax(np.abs(flow)))

        ensemble.append({
            'E': E,
            'P': P,
            'omega_ss': omega_ss,
            'theta_ss': theta_ss,
            'flowmax': flowmax,
            'd': d,
            'm': m,
            'n': n,
        })

    return ensemble


# ====================================================================
# 4. 单次级联（给定 alpha）
# ====================================================================

def _cascade_with_alpha(member, alpha):
    """执行一次 swing-based 级联：移除最大负载边 → 递归 swingfracture。

    Parameters
    ----------
    member : dict  ensemble 成员
    alpha : float  过载阈值

    Returns
    -------
    float  存活边比例
    """
    E = member['E']
    P = member['P']
    omega_ss = member['omega_ss']
    theta_ss = member['theta_ss']
    flowmax = member['flowmax']
    d = member['d']
    m = member['m']
    n = member['n']

    if m == 0:
        return 0.0

    # 构建活跃边集合（全局边索引）
    active_edges = set(range(m))

    # 移除最大负载边（trigger）
    active_edges.discard(d)

    # 找连通分量
    E_remain = E[:, sorted(active_edges)]
    A_remain = adjacencymatrix_from_incidence(E_remain)
    _, comp_table = connected_components(A_remain)

    # 对每个连通分量调用递归 swingfracture
    total_surviving = 0
    for comp in comp_table:
        nodeset = comp  # 局部索引即全局索引（因为 N 个节点编号 0..N-1）
        psi_comp = np.concatenate([omega_ss[comp], theta_ss[comp]])
        total_surviving += swingfracture(
            E, active_edges, psi_comp, nodeset,
            P, SYNCTOL, alpha, KAPPA_CASCADE, flowmax
        )

    return total_surviving / m


# ====================================================================
# 5. Bisection 搜索临界 alpha
# ====================================================================

def _bisect_critical_alpha(ensemble):
    """对 ensemble 进行二分搜索临界 alpha。

    参考 Julia cascadebisection:
    - alpha 从 0.01 开始，stepsize=0.3
    - 判据：mean(surviving_frac) > 0.5 时 beneath=False
    - 当 beneath 状态翻转时，stepsize 反向减半
    - 收敛条件：|stepsize| < BISECT_TOL

    Returns
    -------
    float  临界 alpha
    """
    if not ensemble:
        return np.nan

    alpha = 0.01
    stepsize = 0.3
    beneath0 = True
    beneath1 = True

    max_iterations = 200

    for _ in range(max_iterations):
        if abs(stepsize) <= BISECT_TOL:
            break

        survivors = []
        for member in ensemble:
            frac = _cascade_with_alpha(member, alpha)
            survivors.append(frac)

        av_remaining = np.mean(survivors)

        if av_remaining > 0.5:
            beneath1 = False
        else:
            beneath1 = True

        if beneath1 != beneath0:
            stepsize = -stepsize / 2.0

        beneath0 = beneath1
        alpha += stepsize

    return alpha


# ====================================================================
# 6. 全网格计算
# ====================================================================

def compute_cascade_bisection(ratio_name):
    """遍历 K_list × q_list 全网格，计算临界 alpha。

    Returns
    -------
    dict with keys:
        "alpha_c"      : ndarray (nK, nQ)  临界 alpha 矩阵
        "rho_mean"     : ndarray (nK, nQ)  相对负载 ρ 矩阵
        "K_list"       : list
        "q_list"       : ndarray
        "ratio_name"   : str
        "realizations" : int
    """
    nK = len(K_list)
    nQ = len(q_list)
    alpha_c = np.full((nK, nQ), np.nan)
    rho_mean = np.full((nK, nQ), np.nan)

    total = nK * nQ
    pbar = tqdm(total=total, desc=f"Cascade bisection [{ratio_name}]", unit="pt")

    for ki, K in enumerate(K_list):
        for qi, q_val in enumerate(q_list):
            base_seed = (hash((ratio_name, K, qi)) % (2**31))

            # 生成 ensemble
            ensemble = _prepare_ensemble(ratio_name, K, q_val, realizations, base_seed)

            if ensemble:
                ac = _bisect_critical_alpha(ensemble)
                alpha_c[ki, qi] = ac

                # ρ = flowmax / mean(|flow|)
                rhos = []
                for member in ensemble:
                    flow = edgepower(member['theta_ss'], member['E'], KAPPA_CASCADE)
                    abs_flows = np.abs(flow)
                    mean_flow = np.mean(abs_flows)
                    if mean_flow > 1e-12:
                        rhos.append(member['flowmax'] / mean_flow)
                if rhos:
                    rho_mean[ki, qi] = np.mean(rhos)

            pbar.set_postfix(K=K, q=f"{q_val:.2f}", ac=f"{alpha_c[ki, qi]:.4f}",
                             rho=f"{rho_mean[ki, qi]:.4f}")
            pbar.update(1)

    pbar.close()

    return {
        "alpha_c": alpha_c,
        "rho_mean": rho_mean,
        "K_list": K_list,
        "q_list": np.array(q_list),
        "ratio_name": ratio_name,
        "realizations": realizations,
    }


# ====================================================================
# 7. 缓存
# ====================================================================

def _cache_path(ratio_name):
    return CACHE_DIR / f"cascade_bisection_swing_{ratio_name}.pkl"


def compute_and_cache_bisection(ratio_name):
    """带缓存的级联 bisection 计算。"""
    path = _cache_path(ratio_name)
    if path.exists():
        print(f"  [cache hit] {path}")
        return

    print(f"\n--- Computing swing cascade bisection for '{ratio_name}' ---")
    result = compute_cascade_bisection(ratio_name)

    with open(path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache saved → {path}")


# ====================================================================
# 独立运行测试
# ====================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ratio = sys.argv[1]
    else:
        ratio = "balanced"

    print(f"Testing swing cascade bisection for ratio='{ratio}'")

    # 单点测试：K=8, q=0.5
    print("\n--- Single point test: K=8, q=0.5 ---")
    base_seed = hash((ratio, 8, 10)) % (2**31)
    ensemble = _prepare_ensemble(ratio, 8, 0.5, realizations, base_seed)
    print(f"  Ensemble size: {len(ensemble)}")

    if ensemble:
        ac = _bisect_critical_alpha(ensemble)
        print(f"  Critical alpha: {ac:.4f}")

    # 完整计算
    print(f"\n--- Full grid computation for '{ratio}' ---")
    compute_and_cache_bisection(ratio)
    print("Done.")
