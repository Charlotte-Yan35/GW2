#!/usr/bin/env python3
"""
compute_diurnal_stability.py — 昼夜稳定性剖面: 逐小时 κ_c 二分搜索

核心思路: 对每个小时 h, 取该时刻的静态功率快照 P(h),
二分搜索临界耦合 κ_c(h)。κ_c 高 → 该时段需要更强耦合才能同步 → 更脆弱。

用法:
  python timevarying_sync/compute_diurnal_stability.py --season summer
  python timevarying_sync/compute_diurnal_stability.py --season winter
  python timevarying_sync/compute_diurnal_stability.py --season summer --synthetic --realizations 3
  python timevarying_sync/compute_diurnal_stability.py --season summer --sweep
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from tqdm import tqdm

# ── 路径设置 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from ratio_scan.shared_utils import fswing, generate_ws_network

from config import (
    N, N_HOUSEHOLDS, PCC_NODE, K_BAR, Q_REWIRE,
    I_INERTIA, D_DAMP, KAPPA,
    T_TOTAL, DEFAULT_BASE_SEED,
    CACHE_DIR,
)
from shared_io import build_daily_profiles_for_realization, build_node_injections


# ═══════════════════════════════════════════════════════════════
# 同步检测
# ═══════════════════════════════════════════════════════════════

def _check_sync(A, P_vec, n, kappa, I=I_INERTIA, D=D_DAMP,
                t_settle=500.0, omega_tol=0.5, max_step=5.0):
    """确定性 IC + 积分 t_settle 秒, 检查 ||ω||₂ < omega_tol。

    初始条件: ω=0, θ=linspace(0, π/4, n) — 确定性, 保证二分搜索可复现。
    带发散提前终止: 若 ||ω|| > 100 则停止。
    """
    # 确定性初始条件
    psi0 = np.zeros(2 * n)
    psi0[n:] = np.linspace(0, np.pi / 4, n)

    diverged = [False]

    def rhs(t, psi):
        omega = psi[:n]
        if np.linalg.norm(omega, 2) > 100.0:
            diverged[0] = True
            return np.zeros(2 * n)
        return fswing(psi, A, P_vec, n, I, D, kappa)

    def diverge_event(t, psi):
        return 100.0 - np.linalg.norm(psi[:n], 2)
    diverge_event.terminal = True
    diverge_event.direction = -1

    sol = solve_ivp(rhs, [0, t_settle], psi0,
                    method="RK45", rtol=1e-6, atol=1e-6,
                    max_step=max_step, events=[diverge_event])

    if diverged[0] or sol.status == 1:  # terminated by event
        return False

    omega_final = sol.y[:n, -1]
    return np.linalg.norm(omega_final, 2) < omega_tol


def binary_search_kc(A, P_vec, n, kc_lo=0.1, kc_hi=10.0, precision=0.05):
    """二分搜索临界耦合 κ_c。

    返回 κ_c: 最小的 κ 使得系统同步。
    精度 0.05 → ~8 次迭代。
    """
    # 边界检查: 如果最大 κ 仍不同步, 返回 kc_hi
    if not _check_sync(A, P_vec, n, kc_hi):
        return kc_hi

    # 如果最小 κ 就同步, 返回 kc_lo
    if _check_sync(A, P_vec, n, kc_lo):
        return kc_lo

    while (kc_hi - kc_lo) > precision:
        kc_mid = (kc_lo + kc_hi) / 2.0
        if _check_sync(A, P_vec, n, kc_mid):
            kc_hi = kc_mid
        else:
            kc_lo = kc_mid

    return (kc_lo + kc_hi) / 2.0


# ═══════════════════════════════════════════════════════════════
# 逐小时 κ_c 工作函数
# ═══════════════════════════════════════════════════════════════

def _worker_hourly_kc(args):
    """单次 realization: 建 P(t) + 网络, 24 小时各做一次二分搜索。

    返回 dict:
      kappa_c: (24,) 每小时的临界耦合
      power_l1: (24,) 每小时 Σ|P_k|
      pcc_abs_power: (24,) PCC 节点 |P_PCC|
      node_power_std: (24,) 节点功率标准差 std(P_k)
    """
    (idx, child_seed_entropy, season, use_synthetic,
     K_bar, q_rewire) = args

    rng = np.random.default_rng(child_seed_entropy)

    # 构建日曲线
    profiles = build_daily_profiles_for_realization(
        season, rng, use_synthetic=use_synthetic)

    # 时间网格: 每小时一个点 (0, 3600, ..., 82800) + 86400
    hourly_secs = np.arange(0, 25) * 3600.0  # 25 个点覆盖 0-24h
    hourly_secs[-1] = T_TOTAL  # 确保最后一个点是 86400

    # P_matrix: (25, N)
    P_matrix, P_raw = build_node_injections(profiles, hourly_secs,
                                            balance_at_pcc=True)

    # 生成 WS 网络
    net_seed = int(rng.integers(0, 2**31))
    G = generate_ws_network(N, K_bar, q_rewire, seed=net_seed)
    A = nx.to_numpy_array(G)

    # 逐小时二分搜索 κ_c
    kappa_c = np.zeros(24)
    power_l1 = np.zeros(24)
    pcc_abs_power = np.zeros(24)
    node_power_std = np.zeros(24)

    for h in range(24):
        P_h = P_matrix[h]  # 该小时的静态功率快照
        kappa_c[h] = binary_search_kc(A, P_h, N)

        # 功率统计 (归一化后)
        power_l1[h] = np.sum(np.abs(P_h))
        pcc_abs_power[h] = np.abs(P_h[PCC_NODE])
        node_power_std[h] = np.std(P_h)

    return {
        "kappa_c": kappa_c,
        "power_l1": power_l1,
        "pcc_abs_power": pcc_abs_power,
        "node_power_std": node_power_std,
    }


def run_hourly_kc(season, n_real, base_seed, use_synthetic,
                  K_bar, q_rewire, n_workers):
    """集合仿真: R 次 realization, 每次 24 小时 × 二分搜索。"""
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(n_real)
    child_entropies = [cs.entropy for cs in child_seeds]

    job_args = [
        (i, child_entropies[i], season, use_synthetic, K_bar, q_rewire)
        for i in range(n_real)
    ]

    results_list = []
    if n_workers > 1:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            for res in tqdm(pool.imap(_worker_hourly_kc, job_args),
                            total=n_real,
                            desc=f"[{season}] κ_c 二分搜索"):
                results_list.append(res)
    else:
        for args in tqdm(job_args, desc=f"[{season}] κ_c 二分搜索"):
            results_list.append(_worker_hourly_kc(args))

    # 汇总
    hours = np.arange(24)
    kappa_c = np.array([r["kappa_c"] for r in results_list])          # (R, 24)
    power_l1 = np.array([r["power_l1"] for r in results_list])        # (R, 24)
    pcc_abs_power = np.array([r["pcc_abs_power"] for r in results_list])
    node_power_std = np.array([r["node_power_std"] for r in results_list])

    return {
        "hours": hours,
        "kappa_c": kappa_c,
        "power_l1": power_l1,
        "pcc_abs_power": pcc_abs_power,
        "node_power_std": node_power_std,
    }


# ═══════════════════════════════════════════════════════════════
# (可选) κ 扫描 + 24h r(t) 热力图
# ═══════════════════════════════════════════════════════════════

def _worker_kappa_sweep(args):
    """单次 realization: 对多个 κ 值各跑 24h 仿真, 记录 r(t)。"""
    from scipy.interpolate import interp1d as _interp1d

    (idx, child_seed_entropy, season, use_synthetic,
     K_bar, q_rewire, kappa_values) = args

    rng = np.random.default_rng(child_seed_entropy)

    # 构建日曲线
    profiles = build_daily_profiles_for_realization(
        season, rng, use_synthetic=use_synthetic)

    # 5min 时间网格
    t_grid = np.arange(0, T_TOTAL + 150, 300.0)
    P_matrix, _ = build_node_injections(profiles, t_grid, balance_at_pcc=True)

    # 网络
    net_seed = int(rng.integers(0, 2**31))
    G = generate_ws_network(N, K_bar, q_rewire, seed=net_seed)
    A = nx.to_numpy_array(G)

    # P(t) 插值器
    P_interp = _interp1d(t_grid, P_matrix, axis=0, kind="linear",
                         fill_value="extrapolate")

    # 每小时采样 r 值
    hourly_secs = np.arange(24) * 3600.0 + 1800.0  # 小时中点

    r_matrix = np.zeros((len(kappa_values), 24))

    for ki, kappa in enumerate(kappa_values):
        # 确定性初始条件
        psi0 = np.zeros(2 * N)
        psi0[N:] = np.linspace(0, np.pi / 4, N)

        # 先做初态稳定
        def rhs_settle(t, psi):
            return fswing(psi, A, P_matrix[0], N, I_INERTIA, D_DAMP, kappa)

        sol0 = solve_ivp(rhs_settle, [0, 200], psi0,
                         method="RK45", rtol=1e-6, atol=1e-6, max_step=10.0)
        psi_init = sol0.y[:, -1]

        # 24h 仿真
        def rhs(t, psi):
            P_t = P_interp(float(t))
            return fswing(psi, A, P_t, N, I_INERTIA, D_DAMP, kappa)

        sol = solve_ivp(rhs, [0, T_TOTAL], psi_init,
                        t_eval=hourly_secs, method="RK45",
                        rtol=1e-6, atol=1e-6, max_step=30.0)

        for hi in range(min(sol.y.shape[1], 24)):
            theta = sol.y[N:, hi]
            r_matrix[ki, hi] = np.abs(np.mean(np.exp(1j * theta)))

    return {"r_matrix": r_matrix}


def run_kappa_sweep(season, n_real, base_seed, use_synthetic,
                    K_bar, q_rewire, n_workers,
                    kappa_values=None):
    """多 κ × 24h 扫描, 返回平均 r(κ, h) 热力图数据。"""
    if kappa_values is None:
        kappa_values = np.linspace(0.5, 8.0, 16)

    ss = np.random.SeedSequence(base_seed + 1000)  # 不同于 hourly_kc 的种子
    child_seeds = ss.spawn(n_real)
    child_entropies = [cs.entropy for cs in child_seeds]

    job_args = [
        (i, child_entropies[i], season, use_synthetic,
         K_bar, q_rewire, kappa_values)
        for i in range(n_real)
    ]

    results_list = []
    if n_workers > 1:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            for res in tqdm(pool.imap(_worker_kappa_sweep, job_args),
                            total=n_real,
                            desc=f"[{season}] κ 扫描"):
                results_list.append(res)
    else:
        for args in tqdm(job_args, desc=f"[{season}] κ 扫描"):
            results_list.append(_worker_kappa_sweep(args))

    # 平均
    r_all = np.array([r["r_matrix"] for r in results_list])  # (R, n_kappa, 24)
    r_mean = np.mean(r_all, axis=0)  # (n_kappa, 24)

    return {
        "kappa_values": kappa_values,
        "hours": np.arange(24),
        "r_mean": r_mean,
    }


# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════

def save_results(hourly_results, season, metadata, sweep_results=None):
    """保存到 cache/diurnal_kc_{season}.npz"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = CACHE_DIR / f"diurnal_kc_{season}.npz"

    save_dict = {
        "hours": hourly_results["hours"],
        "kappa_c": hourly_results["kappa_c"],
        "power_l1": hourly_results["power_l1"],
        "pcc_abs_power": hourly_results["pcc_abs_power"],
        "node_power_std": hourly_results["node_power_std"],
        "metadata_json": json.dumps(metadata),
    }

    if sweep_results is not None:
        save_dict["sweep_kappa_values"] = sweep_results["kappa_values"]
        save_dict["sweep_r_mean"] = sweep_results["r_mean"]

    np.savez_compressed(fname, **save_dict)
    return fname


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="昼夜稳定性剖面: 逐小时 κ_c 二分搜索",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--season", choices=["summer", "winter"], required=True,
                   help="季节 (summer=7月, winter=1月)")
    p.add_argument("--realizations", type=int, default=30,
                   help="Monte Carlo realization 数")
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                   help="随机数基础种子")
    p.add_argument("--K", type=int, default=K_BAR,
                   help="WS 平均度")
    p.add_argument("--q", type=float, default=Q_REWIRE,
                   help="WS 重连概率")
    p.add_argument("--synthetic", action="store_true",
                   help="使用合成曲线 (不依赖真实数据)")
    p.add_argument("--workers", type=int, default=1,
                   help="并行 worker 数 (1=串行, 0=CPU 数)")
    p.add_argument("--sweep", action="store_true",
                   help="同时运行 κ 扫描热力图 (耗时较长)")
    p.add_argument("--sweep-realizations", type=int, default=15,
                   help="κ 扫描的 realization 数")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.workers == 0:
        from multiprocessing import cpu_count
        args.workers = cpu_count()

    print("=" * 60)
    print("昼夜稳定性剖面 — κ_c(h) 二分搜索")
    print("=" * 60)
    print(f"  季节       : {args.season}")
    print(f"  数据源     : {'合成曲线' if args.synthetic else 'data-driven (LCL+PV)'}")
    print(f"  网络       : WS(N={N}, K={args.K}, q={args.q})")
    print(f"  Realizations: {args.realizations}")
    print(f"  κ 扫描     : {'是' if args.sweep else '否'}")
    print(f"  Workers    : {args.workers}")
    print("=" * 60)

    t0 = time.time()

    # ── 逐小时 κ_c ──
    hourly_results = run_hourly_kc(
        season=args.season,
        n_real=args.realizations,
        base_seed=args.base_seed,
        use_synthetic=args.synthetic,
        K_bar=args.K,
        q_rewire=args.q,
        n_workers=args.workers,
    )

    elapsed_kc = time.time() - t0
    kc = hourly_results["kappa_c"]  # (R, 24)
    print(f"\nκ_c 搜索完成! 耗时 {elapsed_kc:.1f}s")
    print(f"  κ_c 范围: [{np.min(kc):.2f}, {np.max(kc):.2f}]")
    print(f"  κ_c 均值: {np.mean(kc):.3f} ± {np.std(np.mean(kc, axis=0)):.3f}")

    # ── (可选) κ 扫描 ──
    sweep_results = None
    if args.sweep:
        t1 = time.time()
        sweep_results = run_kappa_sweep(
            season=args.season,
            n_real=args.sweep_realizations,
            base_seed=args.base_seed,
            use_synthetic=args.synthetic,
            K_bar=args.K,
            q_rewire=args.q,
            n_workers=args.workers,
        )
        elapsed_sweep = time.time() - t1
        print(f"\nκ 扫描完成! 耗时 {elapsed_sweep:.1f}s")

    # ── 保存 ──
    elapsed_total = time.time() - t0
    metadata = {
        "season": args.season,
        "realizations": args.realizations,
        "base_seed": args.base_seed,
        "N": N,
        "K": args.K,
        "q": args.q,
        "I": I_INERTIA,
        "D": D_DAMP,
        "synthetic": args.synthetic,
        "kc_search_range": [0.1, 10.0],
        "kc_precision": 0.05,
        "sync_criterion": "||omega||_2 < 0.5 after 500s",
        "sweep": args.sweep,
        "elapsed_sec": elapsed_total,
    }

    fname = save_results(hourly_results, args.season, metadata, sweep_results)

    print(f"\n总耗时: {elapsed_total:.1f}s")
    print(f"结果已保存: {fname}")


if __name__ == "__main__":
    main()
