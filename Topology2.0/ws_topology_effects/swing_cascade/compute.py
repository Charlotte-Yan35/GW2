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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from scipy.integrate import solve_ivp, RK45
from tqdm import tqdm

from ws_config import (
    N, PCC_NODE, HOUSEHOLD_NODES,
    K_list, q_list, realizations,
    RATIO_CONFIGS,
    KAPPA_CASCADE, I_INERTIA, D_DAMP, SYNCTOL, BISECT_TOL,
    DurationSweepConfig, DURATION_SWEEP_PANELS,
    S2_ALPHA_MIN, S2_ALPHA_MAX, S2_ALPHA_RES, S2_ENSEMBLE_SIZE, S2_SEED,
)

# ── 本模块的 cache / output 目录 ──
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
from ws_stability.compute import generate_ws_network, build_power_vector


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
# 8. 带时间记录的级联 swingfracture（级联持续时间探索）
# ====================================================================

def swingfracture_timed(E_full, active_edges, psi, nodeset, P,
                        synctol, alpha, kappa, maxflow, t_offset=0.0):
    """带时间记录的递归级联算法。

    与 swingfracture 逻辑相同，但额外记录：
    - 每轮级联的实际仿真时间
    - 每轮移除的边数
    - 最终存活边数

    Returns
    -------
    dict with keys:
        "surviving"  : int   存活边数
        "rounds"     : list of dict  每轮信息 {"t": 时间, "removed": 移除边数}
        "total_time" : float 总仿真时间
    """
    tol = 1e-5
    nc = len(nodeset)

    P1 = np.array([P[v] for v in nodeset])
    sourcecounter = int(np.sum(P1 > tol))
    sinkcounter = int(np.sum(P1 < -tol))

    edgeset = sorted({j for v in nodeset
                      for j in active_edges
                      if abs(E_full[v, j]) > 0.5})
    ec = len(edgeset)

    empty_result = {"surviving": 0, "rounds": [], "total_time": 0.0}

    if sourcecounter == 0 or sinkcounter == 0:
        return empty_result
    if ec == 0:
        return empty_result

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

    sim_time = solver.t
    psi_f = solver.y
    omega_f, theta_f = psi_f[:nc], psi_f[nc:]

    if reason == 'desync' or np.linalg.norm(omega_f, 2) > synctol:
        return {"surviving": 0, "rounds": [{"t": sim_time, "removed": ec}],
                "total_time": t_offset + sim_time}

    flow_f = edgepower(theta_f, E1, kappa) / maxflow
    if not np.any(np.abs(flow_f) > alpha):
        return {"surviving": ec, "rounds": [], "total_time": t_offset + sim_time}

    # 移除过载边
    removed_this_round = 0
    survivor_local = []
    for j_local in range(ec):
        if abs(flow_f[j_local]) > alpha:
            active_edges.discard(edgeset[j_local])
            removed_this_round += 1
        else:
            survivor_local.append(j_local)

    rounds = [{"t": sim_time, "removed": removed_this_round}]

    if not survivor_local:
        return {"surviving": 0, "rounds": rounds,
                "total_time": t_offset + sim_time}

    E2 = E1[:, survivor_local]
    Adj2 = adjacencymatrix_from_incidence(E2)
    _, comp_table = connected_components(Adj2)

    total_surviving = 0
    for comp in comp_table:
        psi_comp = np.concatenate([omega_f[comp], theta_f[comp]])
        sub_result = swingfracture_timed(
            E_full, active_edges, psi_comp,
            [nodeset[c] for c in comp],
            P, synctol, alpha, kappa, maxflow,
            t_offset=t_offset + sim_time
        )
        total_surviving += sub_result["surviving"]
        rounds.extend(sub_result["rounds"])

    total_time = max((r["t"] for r in rounds), default=sim_time) + t_offset
    return {"surviving": total_surviving, "rounds": rounds,
            "total_time": total_time}


def _cascade_with_alpha_timed(member, alpha):
    """执行一次带时间记录的 swing-based 级联。

    Returns
    -------
    dict with keys:
        "surviving_frac" : float  存活边比例
        "total_time"     : float  级联总持续时间
        "n_rounds"       : int    级联轮数
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
        return {"surviving_frac": 0.0, "total_time": 0.0, "n_rounds": 0}

    active_edges = set(range(m))
    active_edges.discard(d)

    E_remain = E[:, sorted(active_edges)]
    A_remain = adjacencymatrix_from_incidence(E_remain)
    _, comp_table = connected_components(A_remain)

    total_surviving = 0
    all_rounds = []
    for comp in comp_table:
        nodeset = comp
        psi_comp = np.concatenate([omega_ss[comp], theta_ss[comp]])
        result = swingfracture_timed(
            E, active_edges, psi_comp, nodeset,
            P, SYNCTOL, alpha, KAPPA_CASCADE, flowmax
        )
        total_surviving += result["surviving"]
        all_rounds.extend(result["rounds"])

    total_time = max((r["t"] for r in all_rounds), default=0.0)
    return {
        "surviving_frac": total_surviving / m,
        "total_time": total_time,
        "n_rounds": len(all_rounds),
    }


def compute_cascade_duration(ratio_name):
    """对 INTEREST_POINTS × ALPHA_DURATION 计算级联持续时间。

    Returns
    -------
    dict with keys:
        "interest_points" : list of dict
        "alpha_list"      : list of float
        "duration_mean"   : ndarray (n_points, n_alpha)
        "duration_std"    : ndarray (n_points, n_alpha)
        "surviving_mean"  : ndarray (n_points, n_alpha)
        "surviving_std"   : ndarray (n_points, n_alpha)
        "ratio_name"      : str
    """
    from ws_config import INTEREST_POINTS, ALPHA_DURATION, DURATION_REALIZATIONS

    n_pts = len(INTEREST_POINTS)
    n_alpha = len(ALPHA_DURATION)

    duration_mean = np.full((n_pts, n_alpha), np.nan)
    duration_std = np.full((n_pts, n_alpha), np.nan)
    surviving_mean = np.full((n_pts, n_alpha), np.nan)
    surviving_std = np.full((n_pts, n_alpha), np.nan)

    total = n_pts * n_alpha
    pbar = tqdm(total=total, desc=f"Cascade duration [{ratio_name}]", unit="pt")

    for pi, pt in enumerate(INTEREST_POINTS):
        K = pt["K"]
        q = pt["q"]
        label = pt["label"]

        base_seed = hash((ratio_name, "duration", K, int(q * 1000))) % (2**31)
        ensemble = _prepare_ensemble(ratio_name, K, q,
                                     DURATION_REALIZATIONS, base_seed)

        for ai, alp in enumerate(ALPHA_DURATION):
            durations = []
            survivals = []

            for member in ensemble:
                result = _cascade_with_alpha_timed(member, alp)
                durations.append(result["total_time"])
                survivals.append(result["surviving_frac"])

            if durations:
                duration_mean[pi, ai] = np.mean(durations)
                duration_std[pi, ai] = np.std(durations)
                surviving_mean[pi, ai] = np.mean(survivals)
                surviving_std[pi, ai] = np.std(survivals)

            pbar.set_postfix(pt=label, alpha=f"{alp:.1f}",
                             dur=f"{duration_mean[pi, ai]:.1f}")
            pbar.update(1)

    pbar.close()

    return {
        "interest_points": INTEREST_POINTS,
        "alpha_list": ALPHA_DURATION,
        "duration_mean": duration_mean,
        "duration_std": duration_std,
        "surviving_mean": surviving_mean,
        "surviving_std": surviving_std,
        "ratio_name": ratio_name,
    }


def _duration_cache_path(ratio_name):
    return CACHE_DIR / f"cascade_duration_{ratio_name}.pkl"


def compute_and_cache_duration(ratio_name):
    """带缓存的级联持续时间计算。"""
    path = _duration_cache_path(ratio_name)
    if path.exists():
        print(f"  [cache hit] {path}")
        return

    print(f"\n--- Computing cascade duration for '{ratio_name}' ---")
    result = compute_cascade_duration(ratio_name)

    with open(path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache saved → {path}")


# ====================================================================
# 9. Figure S2 风格：参数化 ensemble / cascade / 曲线计算
# ====================================================================

def _fmt_value(x):
    """数值格式化：整数去小数点，浮点保留1位。"""
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _s2_cache_path(panel, variable_name, value, ratio_name,
                   ensemble_size, alpha_res, seed):
    """S2 风格缓存路径。"""
    subdir = CACHE_DIR / "figS2"
    subdir.mkdir(exist_ok=True)
    v_str = _fmt_value(value).replace(".", "p")
    return subdir / (
        f"s2_{panel}_{variable_name}{v_str}_{ratio_name}"
        f"_e{ensemble_size}_r{alpha_res}_s{seed}.npz"
    )


def _prepare_ensemble_parameterized(ratio_name, K, q, n_realizations, base_seed,
                                    kappa, d_damp):
    """参数化版 ensemble 准备：显式传入 kappa 和 d_damp。

    与 _prepare_ensemble 逻辑相同，但 kappa/d_damp 不从全局常量读取。
    """
    ensemble = []
    for r in range(n_realizations):
        seed = base_seed + r
        G = generate_ws_network(N, K, q, seed=seed)
        P = build_power_vector(ratio_name, seed=seed + 1)

        E = build_incidence_matrix(G)
        m = E.shape[1]
        n = N
        A = adjacencymatrix_from_incidence(E)

        rng = np.random.default_rng(seed + 2)
        psi0 = rng.random(2 * n)

        def rhs(t, y, _A=A, _P=P, _n=n, _kappa=kappa, _d=d_damp):
            return fswing(y, _A, _P, _n, I_INERTIA, _d, _kappa)

        sol = solve_ivp(rhs, [0, 250.0], psi0, method='RK45',
                        rtol=1e-8, atol=1e-8, max_step=1.0)
        if sol.status != 0:
            continue

        psi_ss = sol.y[:, -1]
        omega_ss = psi_ss[:n]
        theta_ss = psi_ss[n:]

        if np.linalg.norm(fsteadystate(theta_ss, A, P, kappa), 2) > 1e-3:
            continue

        flow = edgepower(theta_ss, E, kappa)
        flowmax = np.max(np.abs(flow))
        if flowmax < 1e-12:
            continue

        d_idx = int(np.argmax(np.abs(flow)))

        ensemble.append({
            'E': E, 'P': P,
            'omega_ss': omega_ss, 'theta_ss': theta_ss,
            'flowmax': flowmax, 'd': d_idx, 'm': m, 'n': n,
        })

    return ensemble


def swingfracture_timed_parameterized(E_full, active_edges, psi, nodeset, P,
                                       synctol, alpha, kappa, maxflow, d_damp,
                                       t_offset=0.0):
    """参数化版带时间记录的递归级联算法（d_damp 作为参数）。

    使用 solve_ivp + 离散时间网格检测事件，与 figureS2.py 的
    simulate_until_event 方式一致（模拟 Julia DiscreteCallback）。
    """
    tol = 1e-5
    nc = len(nodeset)

    P1 = np.array([P[v] for v in nodeset])
    sourcecounter = int(np.sum(P1 > tol))
    sinkcounter = int(np.sum(P1 < -tol))

    edgeset = sorted({j for v in nodeset
                      for j in active_edges
                      if abs(E_full[v, j]) > 0.5})
    ec = len(edgeset)

    empty_result = {"surviving": 0, "rounds": [], "total_time": 0.0}

    if sourcecounter == 0 or sinkcounter == 0:
        return empty_result
    if ec == 0:
        return empty_result

    delta = np.sum(P1) / 2.0
    for i in range(nc):
        if P1[i] < -tol:
            P1[i] -= delta / sinkcounter
        if P1[i] > tol:
            P1[i] -= delta / sourcecounter

    E1 = np.zeros((nc, ec))
    for i, v in enumerate(nodeset):
        for j_local, j_global in enumerate(edgeset):
            E1[i, j_local] = E_full[v, j_global]

    A1 = adjacencymatrix_from_incidence(E1)

    def rhs(t, y):
        return fswing(y, A1, P1, nc, I_INERTIA, d_damp, kappa)

    # 与 figureS2.py 一致：离散时间网格 + solve_ivp，模拟 Julia DiscreteCallback
    t_grid = np.linspace(0.0, 500.0, 1001)
    sol = solve_ivp(rhs, (0.0, 500.0), psi, method='RK45',
                    t_eval=t_grid, rtol=1e-8, atol=1e-8, max_step=1.0)

    if not sol.success:
        return empty_result

    y_hist = sol.y
    t_hist = sol.t
    reason = 'timeout'
    k_hit = len(t_hist) - 1
    for k in range(1, len(t_hist)):
        omega_k = y_hist[:nc, k]
        theta_k = y_hist[nc:, k]
        if np.linalg.norm(omega_k, 2) > synctol:
            reason = 'desync'
            k_hit = k
            break
        flow_k = edgepower(theta_k, E1, kappa) / maxflow
        if np.any(np.abs(flow_k) > alpha):
            reason = 'overload'
            k_hit = k
            break
        if np.linalg.norm(fsteadystate(theta_k, A1, P1, kappa), 2) < 1e-6:
            reason = 'converge'
            k_hit = k
            break

    sim_time = float(t_hist[k_hit])
    psi_f = y_hist[:, k_hit]
    omega_f, theta_f = psi_f[:nc], psi_f[nc:]

    if reason == 'desync' or np.linalg.norm(omega_f, 2) > synctol:
        return {"surviving": 0, "rounds": [{"t": sim_time, "removed": ec}],
                "total_time": t_offset + sim_time}

    flow_f = edgepower(theta_f, E1, kappa) / maxflow
    if not np.any(np.abs(flow_f) > alpha):
        return {"surviving": ec, "rounds": [], "total_time": t_offset + sim_time}

    removed_this_round = 0
    survivor_local = []
    for j_local in range(ec):
        if abs(flow_f[j_local]) > alpha:
            active_edges.discard(edgeset[j_local])
            removed_this_round += 1
        else:
            survivor_local.append(j_local)

    rounds = [{"t": sim_time, "removed": removed_this_round}]

    if not survivor_local:
        return {"surviving": 0, "rounds": rounds,
                "total_time": t_offset + sim_time}

    E2 = E1[:, survivor_local]
    Adj2 = adjacencymatrix_from_incidence(E2)
    _, comp_table = connected_components(Adj2)

    total_surviving = 0
    for comp in comp_table:
        psi_comp = np.concatenate([omega_f[comp], theta_f[comp]])
        sub_result = swingfracture_timed_parameterized(
            E_full, active_edges, psi_comp,
            [nodeset[c] for c in comp],
            P, synctol, alpha, kappa, maxflow, d_damp,
            t_offset=t_offset + sim_time
        )
        total_surviving += sub_result["surviving"]
        rounds.extend(sub_result["rounds"])

    total_time = max((r["t"] for r in rounds), default=sim_time) + t_offset
    return {"surviving": total_surviving, "rounds": rounds,
            "total_time": total_time}


def _cascade_with_alpha_timed_parameterized(member, alpha, kappa, d_damp, synctol):
    """参数化版带时间记录的级联（kappa, d_damp, synctol 显式传入）。"""
    E = member['E']
    P = member['P']
    omega_ss = member['omega_ss']
    theta_ss = member['theta_ss']
    flowmax = member['flowmax']
    d = member['d']
    m = member['m']

    if m == 0:
        return {"surviving_frac": 0.0, "total_time": 0.0, "n_rounds": 0}

    active_edges = set(range(m))
    active_edges.discard(d)

    E_remain = E[:, sorted(active_edges)]
    A_remain = adjacencymatrix_from_incidence(E_remain)
    _, comp_table = connected_components(A_remain)

    total_surviving = 0
    all_rounds = []
    for comp in comp_table:
        nodeset = comp
        psi_comp = np.concatenate([omega_ss[comp], theta_ss[comp]])
        result = swingfracture_timed_parameterized(
            E, active_edges, psi_comp, nodeset,
            P, synctol, alpha, kappa, flowmax, d_damp
        )
        total_surviving += result["surviving"]
        all_rounds.extend(result["rounds"])

    total_time = max((r["t"] for r in all_rounds), default=0.0)
    return {
        "surviving_frac": total_surviving / m,
        "total_time": total_time,
        "n_rounds": len(all_rounds),
    }


def compute_s2_curve(ratio_name, config, value, alpha_values, ensemble_size, seed):
    """计算单条 S2 曲线：固定一个 sweep 变量值，遍历 alpha_values。

    Parameters
    ----------
    ratio_name : str        "balanced" / "gen_heavy" / "load_heavy"
    config : DurationSweepConfig
    value : float/int       当前 sweep 变量值
    alpha_values : ndarray  alpha 扫描数组
    ensemble_size : int
    seed : int

    Returns
    -------
    dict with "value", "alpha", "tbar"
    """
    # 确定当前 (K, q) 和物理参数
    if config.variable_name == "q":
        K_val = config.K
        q_val = float(value)
    else:  # "K"
        K_val = int(value)
        q_val = config.q

    kappa = config.kappa
    d_damp = config.gamma  # gamma 即阻尼系数

    base_seed = hash((ratio_name, config.panel, _fmt_value(value), seed)) % (2**31)
    ensemble = _prepare_ensemble_parameterized(
        ratio_name, K_val, q_val, ensemble_size, base_seed, kappa, d_damp
    )

    tbar = np.zeros(len(alpha_values))
    if not ensemble:
        return {"value": value, "alpha": alpha_values.copy(), "tbar": tbar}

    for ai, alp in enumerate(alpha_values):
        times = []
        for member in ensemble:
            result = _cascade_with_alpha_timed_parameterized(
                member, alp, kappa, d_damp, SYNCTOL
            )
            times.append(result["total_time"])
        tbar[ai] = np.mean(times) if times else 0.0

    return {"value": value, "alpha": alpha_values.copy(), "tbar": tbar}


def compute_s2_panel(ratio_name, config, alpha_values=None,
                     ensemble_size=None, seed=None):
    """计算一个面板所有曲线（per-value .npz 缓存）。

    Parameters
    ----------
    ratio_name : str
    config : DurationSweepConfig
    alpha_values : ndarray, optional (default from S2_ALPHA_*)
    ensemble_size : int, optional (default S2_ENSEMBLE_SIZE)
    seed : int, optional (default S2_SEED)

    Returns
    -------
    list of dict, each {"value", "alpha", "tbar"}
    """
    if alpha_values is None:
        alpha_values = np.linspace(S2_ALPHA_MIN, S2_ALPHA_MAX, S2_ALPHA_RES)
    if ensemble_size is None:
        ensemble_size = S2_ENSEMBLE_SIZE
    if seed is None:
        seed = S2_SEED

    alpha_res = len(alpha_values)
    curves = []

    for i, v in enumerate(tqdm(config.values,
                                desc=f"[{ratio_name}] Panel {config.panel} "
                                     f"({config.variable_name})",
                                leave=False)):
        cfile = _s2_cache_path(config.panel, config.variable_name, v,
                               ratio_name, ensemble_size, alpha_res, seed)

        if cfile.exists():
            data = np.load(cfile)
            curves.append({"value": v, "alpha": data["alpha"], "tbar": data["tbar"]})
            print(f"  [cache hit] {cfile.name}")
            continue

        print(f"  [{config.panel}] {config.variable_name}={_fmt_value(v)} "
              f"ratio={ratio_name} ens={ensemble_size}")
        curve = compute_s2_curve(ratio_name, config, v, alpha_values,
                                 ensemble_size, seed + 10_007 * i + ord(config.panel))
        np.savez(cfile, alpha=curve["alpha"], tbar=curve["tbar"], value=v)
        curves.append(curve)
        print(f"  [saved] {cfile.name}")

    return curves


# ====================================================================
# 独立运行测试
# ====================================================================

if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1:
        ratio = _sys.argv[1]
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
