"""
compute_kappa_diurnal.py — 逐小时准静态临界耦合 κ_c(h) 计算。

We compute quasi-static critical coupling κ_c(h) under data-driven
time-varying injections P_k(h). Higher κ_c(h) means stronger coupling
is required for synchronisation, indicating lower stability at that hour.
This module compares summer and winter diurnal stability profiles.

用法:
  MPLBACKEND=Agg python compute_kappa_diurnal.py          # 全量运行
  MPLBACKEND=Agg python compute_kappa_diurnal.py --test    # 快速测试
"""

import sys
import argparse
import hashlib
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    N, PCC_NODE, N_HOUSEHOLDS, K_WS, Q_WS, SEED,
    N_REALIZATIONS, N_PROFILE_SAMPLES, PENETRATION,
    I_INERTIA, D_DAMP,
    KAPPA_MIN, KAPPA_MAX, KAPPA_TOL, N_IC_TRIES,
    CACHE_DIR, RESULTS_DIR, SEASONS,
)
from shared_diurnal_utils import (
    load_pv_generation,
    load_household_demand_profiles,
    preprocess_demand_pools,
    build_injection_vector,
)


# ====================================================================
# 全局变量 (多进程 worker 共享，通过 initializer 设置)
# ====================================================================
_pv_profiles = None
_demand_pools = None


def _init_worker(pv_profiles, demand_pools):
    """多进程 worker 初始化：设置全局数据。"""
    global _pv_profiles, _demand_pools
    _pv_profiles = pv_profiles
    _demand_pools = demand_pools


# ====================================================================
# 1. 网络生成
# ====================================================================

def generate_ws_network(n, K, q, seed=None):
    """生成连通的 WS 小世界图。"""
    return nx.connected_watts_strogatz_graph(n, K, q, seed=seed)


# ====================================================================
# 2. Swing 方程求解器 & κ_c 二分搜索
#    (复制自 ws_stability/compute.py，保持独立)
# ====================================================================

def _steady_state_residual(theta, A, P, kappa):
    """功率平衡残差: P_i - κ Σ_j A_ij sin(θ_i − θ_j)。"""
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def _integrate_swing(A, P, n, kappa, y0,
                     I=I_INERTIA, D=D_DAMP, t_max=200.0):
    """积分二阶 swing 方程。

    状态 y = [ω_1..ω_n, θ_1..θ_n]。
    返回 (converged: bool, y_final: ndarray)。
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


def _find_kappa_c_lower(A, P, n, tol=KAPPA_TOL):
    """自下而上搜索下临界耦合 κ_c^low（最小同步耦合）。

    Phase 1 - Bracket: κ 从 KAPPA_MIN 开始翻倍，直到找到第一个稳定点。
    Phase 2 - Bisection: 在 [κ_fail, κ_ok] 间二分收敛。

    Returns
    -------
    (kappa_c_low, y_stable) : (float, ndarray or None)
    """
    kappa_max_bracket = KAPPA_MAX

    # Phase 1: Bracket
    kappa = KAPPA_MIN
    kappa_fail = 0.0
    kappa_ok = None
    y_ok = None

    while kappa <= kappa_max_bracket:
        found = False
        for _ in range(N_IC_TRIES):
            y0 = np.random.rand(2 * n) * 0.1
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

    # 如果第一个 κ 就稳定了
    if kappa_fail == 0.0 and kappa_ok == KAPPA_MIN:
        converged_low, _ = _integrate_swing(
            A, P, n, KAPPA_MIN / 5,
            np.random.rand(2 * n) * 0.1, t_max=200.0,
        )
        if converged_low:
            return KAPPA_MIN / 5, y_ok
        kappa_fail = KAPPA_MIN / 5

    # Phase 2: Bisection — 多次随机 IC 避免卡在 bracket 边界
    iterations = 0
    while (kappa_ok - kappa_fail) > tol and iterations < 40:
        kappa_mid = (kappa_fail + kappa_ok) / 2.0
        # 先用 warm-start
        converged, y_sol = _integrate_swing(A, P, n, kappa_mid, y_ok, t_max=150.0)
        if converged:
            kappa_ok = kappa_mid
            y_ok = y_sol
        else:
            # 尝试多个随机 IC
            found_mid = False
            for _ in range(N_IC_TRIES):
                y0_rand = np.random.rand(2 * n) * 0.1
                converged2, y_sol2 = _integrate_swing(
                    A, P, n, kappa_mid, y0_rand, t_max=200.0,
                )
                if converged2:
                    kappa_ok = kappa_mid
                    y_ok = y_sol2
                    found_mid = True
                    break
            if not found_mid:
                kappa_fail = kappa_mid
        iterations += 1

    return kappa_ok, y_ok


# ====================================================================
# 3. 单任务计算
# ====================================================================

def _make_seed(season, hour, net_r, profile_r):
    """确定性种子 = hash(season, hour, net_r, profile_r)。"""
    key = f"{season}_{hour}_{net_r}_{profile_r}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2**31)


def _compute_single_task(args):
    """单个 (season, hour, net_r, profile_r) → kappa_c。"""
    season, hour, net_r, profile_r = args
    seed = _make_seed(season, hour, net_r, profile_r)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1. 生成 WS 图
    G = generate_ws_network(N, K_WS, Q_WS, seed=seed)
    A = nx.to_numpy_array(G)

    # 2. 构建注入向量
    P = build_injection_vector(
        hour, season,
        _pv_profiles[season],
        _demand_pools[season],
        PENETRATION, rng,
    )

    # 3. 二分搜索 κ_c
    kc, _ = _find_kappa_c_lower(A, P, N)

    return season, hour, net_r, profile_r, kc


# ====================================================================
# 4. 主计算流程
# ====================================================================

def run_computation(test_mode=False):
    """运行全量/测试计算。

    全量: 2 季 × 24 h × 10 net × 5 profile = 2400 任务
    测试: 2 季 × 24 h × 2 net × 1 profile = 96 任务

    支持断点续算: 从 cache/kappa_diurnal_raw.csv 读取已有结果。
    """
    if test_mode:
        n_net = 2
        n_prof = 1
        print(">>> TEST MODE: 2 net × 1 profile = 96 tasks")
    else:
        n_net = N_REALIZATIONS
        n_prof = N_PROFILE_SAMPLES
        print(f">>> FULL MODE: {n_net} net × {n_prof} profile = "
              f"{2 * 24 * n_net * n_prof} tasks")

    # --- 加载数据 ---
    print("\n--- Loading data ---")
    pv_profiles = load_pv_generation()
    demand_profiles = load_household_demand_profiles()
    demand_pools = preprocess_demand_pools(demand_profiles)

    # --- 断点续算: 加载已有结果 ---
    raw_csv = CACHE_DIR / "kappa_diurnal_raw.csv"
    existing = set()
    rows_existing = []
    if raw_csv.exists():
        df_existing = pd.read_csv(raw_csv)
        for _, row in df_existing.iterrows():
            key = (row["season"], int(row["hour"]),
                   int(row["net_realization"]), int(row["profile_realization"]))
            existing.add(key)
            rows_existing.append(row.to_dict())
        print(f"  Loaded {len(existing)} existing results from cache")

    # --- 构建任务列表 ---
    tasks = []
    for season in SEASONS:
        for hour in range(24):
            for net_r in range(n_net):
                for prof_r in range(n_prof):
                    key = (season, hour, net_r, prof_r)
                    if key not in existing:
                        tasks.append(key)

    if not tasks:
        print("  All tasks already completed!")
    else:
        total = len(tasks)
        print(f"\n--- Computing {total} tasks ---")

        n_workers = max(1, cpu_count() - 1)
        print(f"  Using {n_workers} workers")

        # 多进程计算
        new_rows = []
        with Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(pv_profiles, demand_pools),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(_compute_single_task, tasks)):
                season, hour, net_r, prof_r, kc = result
                new_rows.append({
                    "season": season,
                    "hour": hour,
                    "net_realization": net_r,
                    "profile_realization": prof_r,
                    "kappa_c": kc,
                })
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    print(f"  Progress: {i + 1}/{total} "
                          f"({100 * (i + 1) / total:.0f}%)")

        # 合并已有 + 新结果，保存 raw CSV
        all_rows = rows_existing + new_rows
        df_raw = pd.DataFrame(all_rows)
        df_raw.to_csv(raw_csv, index=False)
        print(f"  Raw results saved → {raw_csv}")

    # --- 聚合统计 ---
    print("\n--- Aggregating results ---")
    df_raw = pd.read_csv(raw_csv)

    agg_rows = []
    for season in SEASONS:
        for hour in range(24):
            mask = (df_raw["season"] == season) & (df_raw["hour"] == hour)
            vals = df_raw.loc[mask, "kappa_c"].dropna()
            agg_rows.append({
                "season": season,
                "hour": hour,
                "kappa_c_mean": vals.mean() if len(vals) > 0 else np.nan,
                "kappa_c_std": vals.std() if len(vals) > 0 else np.nan,
                "n_valid": int((~vals.isna()).sum()) if len(vals) > 0 else 0,
                "n_total": int(len(df_raw.loc[mask])),
            })

    df_agg = pd.DataFrame(agg_rows)
    agg_csv = RESULTS_DIR / "kappa_diurnal.csv"
    df_agg.to_csv(agg_csv, index=False)
    print(f"  Aggregated results saved → {agg_csv}")

    # 打印摘要
    print("\n--- Summary ---")
    for season in SEASONS:
        sub = df_agg[df_agg["season"] == season]
        peak_idx = sub["kappa_c_mean"].idxmax()
        min_idx = sub["kappa_c_mean"].idxmin()
        peak_row = sub.loc[peak_idx]
        min_row = sub.loc[min_idx]
        print(f"  {season}:")
        print(f"    Peak κ_c = {peak_row['kappa_c_mean']:.3f} ± {peak_row['kappa_c_std']:.3f} "
              f"at hour {int(peak_row['hour'])}")
        print(f"    Min  κ_c = {min_row['kappa_c_mean']:.3f} ± {min_row['kappa_c_std']:.3f} "
              f"at hour {int(min_row['hour'])}")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute diurnal κ_c")
    parser.add_argument("--test", action="store_true", help="快速测试模式")
    args = parser.parse_args()

    run_computation(test_mode=args.test)
