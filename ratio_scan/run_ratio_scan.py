"""
run_ratio_scan.py — Experiment 4.2: Ratio Scan
研究节点组成 (n_g, n_c, n_p) 如何影响 WS 网络的级联韧性。
对所有 ratio 组合 × 6 种拓扑 (K×q) × 4 种 α_pas，
通过二分搜索找到临界过载容忍度 α*，并分析失效模式。

α 机制:
  - capacity: C_e = α_e · maxflow,  maxflow = max(|F_e^(0)|)
  - 跳闸: |F_e(t)| > α_e · maxflow
  - edge α: α_e = min(α_i, α_j)
    pcc/gen/con → α_active;  pas → α_pas

用法:
  python run_ratio_scan.py          # 完整实验
  python run_ratio_scan.py --test   # 小规模验证
"""

import time
import csv
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.integrate import solve_ivp, RK45

try:
    from ratio_scan.shared_utils import (
        generate_ws_network,
        fswing, fsteadystate, edgepower,
        build_incidence_matrix, adjacencymatrix_from_incidence,
        connected_components,
        build_ratio_grid, assign_roles,
    )
except ModuleNotFoundError:
    from shared_utils import (
        generate_ws_network,
        fswing, fsteadystate, edgepower,
        build_incidence_matrix, adjacencymatrix_from_incidence,
        connected_components,
        build_ratio_grid, assign_roles,
    )

# ── 路径设置 ──────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
RESULTS_DIR = _MODULE_DIR / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ====================================================================
# 参数
# ====================================================================
N = 50
N_HOUSEHOLDS = 49
PCC_NODE = 0

K_VALUES = [4, 8]
Q_VALUES = [0.0, 0.15, 1.0]

ALPHA_TOL = 1e-2
ALPHA_PAS_LIST = [1.0, 0.7, 0.4, 0.1]

REALIZATIONS = 30
ROLE_SEEDS = 10
DYN_SEEDS = 10

RATIO_STEP = 5        # simplex step = 1/RATIO_STEP = 0.2
KAPPA = 1.0
SYNCTOL = 3.0
PMAX = 1.0
I_INERTIA = 1.0
D_DAMP = 1.0

CSV_HEADER = [
    "K", "q", "ng", "nc", "np", "alpha_pas",
    "alpha_star", "p_sync", "p_flow",
]


# ====================================================================
# 1. Per-edge α 构建
# ====================================================================

def build_edge_alpha(E, node_types, alpha_active, alpha_pas):
    """根据节点类型构建每条边的 α 向量。

    α_e = min(α_i, α_j)
    pcc/gen/con → alpha_active;  pas → alpha_pas

    Parameters
    ----------
    E : ndarray (n, m)  关联矩阵
    node_types : ndarray (n,) int  0=pcc,1=gen,2=con,3=pas
    alpha_active : float
    alpha_pas : float

    Returns
    -------
    edge_alpha : ndarray (m,)
    """
    n, m = E.shape
    node_alpha = np.where(node_types == 3, alpha_pas, alpha_active)
    edge_alpha = np.zeros(m)
    for j in range(m):
        col = E[:, j]
        endpoints = np.where(np.abs(col) > 0.5)[0]
        if len(endpoints) == 2:
            edge_alpha[j] = min(node_alpha[endpoints[0]], node_alpha[endpoints[1]])
        else:
            edge_alpha[j] = alpha_active  # fallback
    return edge_alpha


# ====================================================================
# 4. 带 typed-α 的递归级联
# ====================================================================

def swingfracture_typed(E_full, active_edges, psi, nodeset, P,
                        synctol, edge_alpha_full, kappa, maxflow):
    """带 per-edge α 的递归级联算法。

    与原 swingfracture 的关键差异:
      - edge_alpha_full 是全局 per-edge α 向量 (m_total,)
      - 过载判断: |flow[e]| > edge_alpha_local[e] * maxflow
      - 额外返回 reason ("sync"/"flow"/"converge")

    Returns
    -------
    (surviving_count: int, reason: str)
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
        return 0, "sync"
    if ec == 0:
        return 0, "sync"

    # 齐次再平衡
    delta = np.sum(P1) / 2.0
    for i in range(nc):
        if P1[i] < -tol:
            P1[i] -= delta / sinkcounter
        if P1[i] > tol:
            P1[i] -= delta / sourcecounter

    # 子网络关联矩阵
    E1 = np.zeros((nc, ec))
    for i, v in enumerate(nodeset):
        for j_local, j_global in enumerate(edgeset):
            E1[i, j_local] = E_full[v, j_global]

    # 局部 edge alpha
    edge_alpha_local = np.array([edge_alpha_full[j] for j in edgeset])

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
        flow_cur = edgepower(y[nc:], E1, kappa)
        if np.any(np.abs(flow_cur) > edge_alpha_local * maxflow):
            reason = 'overload'
            break
        if np.linalg.norm(fsteadystate(y[nc:], A1, P1, kappa), 2) < 1e-6:
            reason = 'converge'
            break

    psi_f = solver.y
    omega_f, theta_f = psi_f[:nc], psi_f[nc:]

    if reason == 'desync' or np.linalg.norm(omega_f, 2) > synctol:
        return 0, "sync"

    flow_f = edgepower(theta_f, E1, kappa)
    if not np.any(np.abs(flow_f) > edge_alpha_local * maxflow):
        return ec, "converge"

    # 移除过载边, 递归
    survivor_local = []
    for j_local in range(ec):
        if abs(flow_f[j_local]) > edge_alpha_local[j_local] * maxflow:
            active_edges.discard(edgeset[j_local])
        else:
            survivor_local.append(j_local)

    if not survivor_local:
        return 0, "flow"

    E2 = E1[:, survivor_local]
    Adj2 = adjacencymatrix_from_incidence(E2)
    _, comp_table = connected_components(Adj2)

    total_surviving = 0
    first_reason = "flow"  # 首次触发是 flow（过载移边）
    for comp in comp_table:
        sub_surv, sub_reason = swingfracture_typed(
            E_full, active_edges,
            np.concatenate([omega_f[comp], theta_f[comp]]),
            [nodeset[c] for c in comp],
            P, synctol, edge_alpha_full, kappa, maxflow
        )
        total_surviving += sub_surv
        # 保留首次失效原因
        if sub_reason == "sync":
            first_reason = "sync"

    return total_surviving, first_reason


# ====================================================================
# 5. Ensemble 准备
# ====================================================================

def prepare_ensemble(K, q, ng, nc, n_realizations, role_seeds, base_seed):
    """生成 ensemble：外层 n_realizations 网络 × 内层 role_seeds 角色分配。

    Returns
    -------
    ensemble : list[dict]
        每个 dict: {E, P, node_types, omega_ss, theta_ss, flowmax, d, m, n}
    """
    kappa = KAPPA
    ensemble = []

    for r in range(n_realizations):
        net_seed = base_seed + r
        G = generate_ws_network(N, K, q, seed=net_seed)
        E = build_incidence_matrix(G)
        m = E.shape[1]
        n = N
        A = adjacencymatrix_from_incidence(E)

        for rs in range(role_seeds):
            role_seed = base_seed + n_realizations + r * role_seeds + rs
            P, node_types = assign_roles(ng, nc, seed=role_seed)

            # 随机初始条件，积分 Swing 方程求稳态
            rng = np.random.default_rng(role_seed + 7777)
            psi0 = rng.random(2 * n)

            def rhs(t, y, _A=A, _P=P, _n=n, _kappa=kappa):
                return fswing(y, _A, _P, _n, I_INERTIA, D_DAMP, _kappa)

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

            d = int(np.argmax(np.abs(flow)))

            ensemble.append({
                'E': E,
                'P': P,
                'node_types': node_types,
                'omega_ss': omega_ss,
                'theta_ss': theta_ss,
                'flowmax': flowmax,
                'd': d,
                'm': m,
                'n': n,
            })

    return ensemble


# ====================================================================
# 6. 带 typed-α 的单次级联
# ====================================================================

def cascade_with_typed_alpha(member, alpha_active, alpha_pas, dyn_seed):
    """执行一次级联：构建 edge_alpha → 移除 trigger 边 → 递归。

    Returns
    -------
    (frac: float, reason: str)
        frac = 存活边比例, reason = "sync"/"flow"/"converge"
    """
    E = member['E']
    P = member['P']
    node_types = member['node_types']
    omega_ss = member['omega_ss'].copy()
    theta_ss = member['theta_ss'].copy()
    flowmax = member['flowmax']
    d = member['d']
    m = member['m']
    n = member['n']

    if m == 0:
        return 0.0, "flow"

    # 构建 per-edge α
    edge_alpha_full = build_edge_alpha(E, node_types, alpha_active, alpha_pas)

    # 初始条件扰动（用 dyn_seed）
    rng = np.random.default_rng(dyn_seed)
    omega_init = omega_ss + rng.normal(0, 1e-4, n)
    theta_init = theta_ss + rng.normal(0, 1e-4, n)

    # 移除 trigger 边
    active_edges = set(range(m))
    active_edges.discard(d)

    # 找连通分量
    E_remain = E[:, sorted(active_edges)]
    A_remain = adjacencymatrix_from_incidence(E_remain)
    _, comp_table = connected_components(A_remain)

    total_surviving = 0
    overall_reason = "converge"
    for comp in comp_table:
        nodeset = comp
        psi_comp = np.concatenate([omega_init[comp], theta_init[comp]])
        surv, reason = swingfracture_typed(
            E, active_edges, psi_comp, nodeset,
            P, SYNCTOL, edge_alpha_full, KAPPA, flowmax
        )
        total_surviving += surv
        if reason == "sync":
            overall_reason = "sync"
        elif reason == "flow" and overall_reason == "converge":
            overall_reason = "flow"

    return total_surviving / m, overall_reason


# ====================================================================
# 7. 二分搜索 α*
# ====================================================================

def bisect_alpha_star(ensemble, alpha_pas, dyn_seeds):
    """Julia 风格 stepsize 递减二分搜索临界 α。

    Returns
    -------
    float  临界 alpha_active
    """
    if not ensemble:
        return np.nan

    alpha = 0.01
    stepsize = 0.3
    beneath0 = True
    beneath1 = True
    max_iterations = 200

    for _ in range(max_iterations):
        if abs(stepsize) <= ALPHA_TOL:
            break

        survivors = []
        for member in ensemble:
            for ds in dyn_seeds:
                frac, _ = cascade_with_typed_alpha(member, alpha, alpha_pas, ds)
                survivors.append(frac)

        av_remaining = np.mean(survivors)

        beneath1 = av_remaining <= 0.5

        if beneath1 != beneath0:
            stepsize = -stepsize / 2.0

        beneath0 = beneath1
        alpha += stepsize

    return alpha


# ====================================================================
# 8. 失效模式分类
# ====================================================================

def classify_failure_modes(ensemble, alpha_star, alpha_pas, dyn_seeds,
                           offset=0.1):
    """在 α* - offset 处统计失效原因比例。

    Returns
    -------
    dict  {"p_sync": float, "p_flow": float}
    """
    alpha_test = max(0.01, alpha_star - offset)

    n_sync = 0
    n_flow = 0
    n_total = 0

    for member in ensemble:
        for ds in dyn_seeds:
            frac, reason = cascade_with_typed_alpha(
                member, alpha_test, alpha_pas, ds
            )
            n_total += 1
            if reason == "sync":
                n_sync += 1
            elif reason == "flow":
                n_flow += 1
            # "converge" 表示存活，不计入失效

    if n_total == 0:
        return {"p_sync": 0.0, "p_flow": 0.0}

    return {
        "p_sync": n_sync / n_total,
        "p_flow": n_flow / n_total,
    }


# ====================================================================
# 9. 单任务计算（for multiprocessing）
# ====================================================================

def _compute_one_task(args):
    """计算单个 (K, q, ng, nc, np_count, alpha_pas) 的 α* 和失效模式。

    Returns
    -------
    list  CSV 行数据
    """
    K, q, ng, nc, np_count, alpha_pas, base_seed, n_real, n_role, n_dyn = args

    dyn_seeds = list(range(base_seed + 90000, base_seed + 90000 + n_dyn))

    ensemble = prepare_ensemble(K, q, ng, nc, n_real, n_role, base_seed)

    if not ensemble:
        return [K, q, ng, nc, np_count, alpha_pas,
                np.nan, np.nan, np.nan]

    alpha_star = bisect_alpha_star(ensemble, alpha_pas, dyn_seeds)

    # 失效模式分类
    if np.isnan(alpha_star):
        p_sync, p_flow = np.nan, np.nan
    else:
        modes = classify_failure_modes(ensemble, alpha_star, alpha_pas, dyn_seeds)
        p_sync = modes["p_sync"]
        p_flow = modes["p_flow"]

    return [K, q, ng, nc, np_count, alpha_pas,
            alpha_star, p_sync, p_flow]


# ====================================================================
# 10. 主实验循环
# ====================================================================

def run_experiment(test_mode=False):
    """构建比例网格 × 拓扑参数 × alpha_pas → 并行计算 → 聚合。"""
    # 参数覆盖（test 模式）
    if test_mode:
        n_real = 2
        n_role = 2
        n_dyn = 2
        k_values = [4]
        q_values = [0.0]
        alpha_pas_list = [1.0]
        ratio_step = 5
    else:
        n_real = REALIZATIONS
        n_role = ROLE_SEEDS
        n_dyn = DYN_SEEDS
        k_values = K_VALUES
        q_values = Q_VALUES
        alpha_pas_list = ALPHA_PAS_LIST
        ratio_step = RATIO_STEP

    ratio_grid = build_ratio_grid(ratio_step)
    if test_mode:
        ratio_grid = ratio_grid[:3]

    n_ratios = len(ratio_grid)
    n_topos = len(k_values) * len(q_values)
    n_apas = len(alpha_pas_list)
    total_tasks = n_ratios * n_topos * n_apas
    print(f"Ratio scan: {n_ratios} ratios × {n_topos} topos × {n_apas} α_pas "
          f"= {total_tasks} tasks")
    print(f"  REALIZATIONS={n_real}, ROLE_SEEDS={n_role}, DYN_SEEDS={n_dyn}")

    raw_path = CACHE_DIR / "raw_results.csv"

    # Resume: 读已有结果
    done_keys = set()
    if raw_path.exists():
        with open(raw_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["K"], row["q"], row["ng"], row["nc"],
                       row["np"], row["alpha_pas"])
                done_keys.add(key)
        print(f"Resume: {len(done_keys)} tasks already done")

    # 构建任务列表
    tasks = []
    for ng, nc, np_count in ratio_grid:
        for K in k_values:
            for q in q_values:
                for alpha_pas in alpha_pas_list:
                    key = (str(K), str(q), str(ng), str(nc),
                           str(np_count), str(alpha_pas))
                    if key in done_keys:
                        continue
                    base_seed = hash((K, int(q * 1000), ng, nc,
                                      int(alpha_pas * 100))) % (2**31)
                    tasks.append((K, q, ng, nc, np_count, alpha_pas,
                                  base_seed, n_real, n_role, n_dyn))

    print(f"Tasks to compute: {len(tasks)} (skipped {len(done_keys)} done)")

    if not tasks:
        print("All tasks already completed!")
        _aggregate_to_pkl(raw_path, k_values, q_values)
        return

    # 打开 CSV
    write_header = not raw_path.exists() or len(done_keys) == 0
    raw_file = open(raw_path, "a", newline="")
    raw_writer = csv.writer(raw_file)
    if write_header:
        raw_writer.writerow(CSV_HEADER)
        raw_file.flush()

    t_start = time.time()
    done = len(done_keys)

    if test_mode:
        # 单进程运行便于调试
        for task in tasks:
            result = _compute_one_task(task)
            raw_writer.writerow(result)
            raw_file.flush()
            done += 1
            K, q, ng, nc, np_c, apas = task[:6]
            astar = result[6]
            astar_str = f"{astar:.4f}" if not np.isnan(astar) else "NaN"
            print(f"  [{done}/{total_tasks}] K={K} q={q} "
                  f"ng={ng} nc={nc} np={np_c} α_pas={apas} → α*={astar_str}")
    else:
        n_workers = max(1, cpu_count() - 1)
        print(f"Using {n_workers} worker processes")

        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_compute_one_task, tasks,
                                              chunksize=1):
                raw_writer.writerow(result)
                raw_file.flush()
                done += 1
                if done % 5 == 0 or done == total_tasks:
                    elapsed = time.time() - t_start
                    computed = done - len(done_keys) + len(tasks) - len(tasks)
                    rate = (done - (total_tasks - len(tasks))) / elapsed \
                        if elapsed > 0 else 0
                    remaining = total_tasks - done
                    eta = remaining / rate if rate > 0 else 0
                    astar = result[6]
                    astar_str = f"{astar:.4f}" if not np.isnan(astar) else "NaN"
                    print(f"  [{done}/{total_tasks}] α*={astar_str} "
                          f"rate={rate:.2f}/s ETA={eta/60:.1f}min")

    raw_file.close()
    print(f"\nRaw results saved → {raw_path}")

    # 聚合
    _aggregate_to_pkl(raw_path, k_values, q_values)


# ====================================================================
# 11. 聚合为 PKL
# ====================================================================

def _aggregate_to_pkl(raw_path, k_values, q_values):
    """从 raw CSV 聚合为 per-(K,q) 的 PKL，计算 Δα*。"""
    if not raw_path.exists():
        print("No raw results to aggregate.")
        return

    import pandas as pd

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows from {raw_path}")

    for K in k_values:
        for q in q_values:
            mask = (df["K"] == K) & (np.abs(df["q"] - q) < 1e-6)
            sub = df[mask].copy()
            if sub.empty:
                continue

            result = {
                "K": K,
                "q": q,
                "data": sub.to_dict("records"),
            }

            # 计算 Δα* = α*(α_pas=1.0) - α*(α_pas=0.1)
            pivot = sub.pivot_table(
                index=["ng", "nc", "np"],
                columns="alpha_pas",
                values="alpha_star",
                aggfunc="mean",
            )
            if 1.0 in pivot.columns and 0.1 in pivot.columns:
                result["delta_alpha_star"] = (
                    pivot[1.0] - pivot[0.1]
                ).to_dict()

            # 按 α_pas 分组的均值
            grouped = sub.groupby(["ng", "nc", "np", "alpha_pas"]).agg({
                "alpha_star": "mean",
                "p_sync": "mean",
                "p_flow": "mean",
            }).reset_index()
            result["grouped"] = grouped.to_dict("records")

            q_str = f"{q:.2f}".replace(".", "p")
            pkl_path = RESULTS_DIR / f"k{K}_q{q_str}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Aggregated → {pkl_path}")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 4.2: Ratio Scan — 节点组成对级联韧性的影响")
    parser.add_argument("--test", action="store_true",
                        help="小规模测试模式 (R=2, seeds=2, 3 ratios, K=4 q=0)")
    args = parser.parse_args()

    run_experiment(test_mode=args.test)
