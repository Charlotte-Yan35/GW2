#!/usr/bin/env python3
"""
compute_timevarying_sync.py — 时变 P_k(t) 同步仿真 (只计算，不画图)

报告 §4.3.1: "时变 P_k(t) 对同步的影响（基于 data-driven 的设定）"

用法:
  python timevarying_sync/compute_timevarying_sync.py --season summer --freq 5min --realizations 50
  python timevarying_sync/compute_timevarying_sync.py --season winter --freq 5min --realizations 50
  python timevarying_sync/compute_timevarying_sync.py --help
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import tqdm

# ── 路径设置: 复用仓库已有的 swing 实现 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from ratio_scan.shared_utils import fswing, fsteadystate, generate_ws_network

from config import (
    N, N_HOUSEHOLDS, PCC_NODE, K_BAR, Q_REWIRE,
    I_INERTIA, D_DAMP, KAPPA, SYNCTOL,
    T_TOTAL, DEFAULT_FREQ, DEFAULT_REALIZATIONS, DEFAULT_BASE_SEED,
    SETTLE_TIME, ODE_MAX_STEP,
    R_COLLAPSE_THRESHOLD, OMEGA_NORM_THRESHOLD,
    SLIDING_WINDOW_SEC,
    CACHE_DIR,
)
from shared_io import build_daily_profiles_for_realization, build_node_injections


# ═══════════════════════════════════════════════════════════════
# 核心仿真函数
# ═══════════════════════════════════════════════════════════════

def find_initial_steady_state(A, P0, n, kappa, rng,
                              I=I_INERTIA, D=D_DAMP,
                              settle_time=SETTLE_TIME):
    """以 P(t=0) 为恒定注入积分到稳态，返回初始相空间状态 ψ₀。

    对齐 reference_code swing.jl: 积分到 t=200s，检查收敛。
    """
    psi0 = rng.random(2 * n) * 0.1  # 接近零的随机初态

    def rhs(t, psi):
        return fswing(psi, A, P0, n, I, D, kappa)

    sol = solve_ivp(rhs, [0, settle_time], psi0,
                    method="RK45", rtol=1e-8, atol=1e-8,
                    max_step=10.0)

    psi_final = sol.y[:, -1]
    theta_final = psi_final[n:]
    residual = np.linalg.norm(fsteadystate(theta_final, A, P0, kappa), 2)
    if residual > 1e-3:
        warnings.warn(
            f"初态未收敛: ||residual||={residual:.4e} (> 1e-3), "
            f"将使用最终状态作为初始条件", RuntimeWarning, stacklevel=2,
        )
    return psi_final


def simulate_24h(A, P_matrix, t_grid, n, kappa, psi0,
                 I=I_INERTIA, D=D_DAMP, synctol=SYNCTOL,
                 max_step=ODE_MAX_STEP):
    """在时变 P(t) 下积分 24h swing dynamics。

    P(t) 在 t_grid 各点之间线性插值。
    复用 ratio_scan.shared_utils.fswing (不复制)。

    Returns
    -------
    t_eval   : ndarray (T_eval,)
    r_vals   : ndarray (T_eval,)  同步序参量 r(t)=|mean(exp(iθ))|
    omega_n  : ndarray (T_eval,)  ||ω(t)||_2
    t_collapse : float  首次崩溃时刻 (秒), NaN 表示未崩溃
    """
    # P(t) 插值器: shape (T_grid, N) → 调用 interp(t) 返回 (N,)
    P_interp = interp1d(t_grid, P_matrix, axis=0, kind="linear",
                        fill_value="extrapolate")

    def rhs(t, psi):
        P_t = P_interp(float(t))
        return fswing(psi, A, P_t, n, I, D, kappa)

    sol = solve_ivp(rhs, [t_grid[0], t_grid[-1]], psi0,
                    t_eval=t_grid, method="RK45",
                    rtol=1e-8, atol=1e-8, max_step=max_step)

    if sol.status != 0:
        warnings.warn(f"ODE 求解器状态: {sol.message}", RuntimeWarning, stacklevel=2)

    # 提取 r(t) 和 ||ω||_2
    T_eval = sol.y.shape[1]
    r_vals = np.zeros(T_eval)
    omega_n = np.zeros(T_eval)
    for i in range(T_eval):
        theta_i = sol.y[n:, i]
        omega_i = sol.y[:n, i]
        r_vals[i] = np.abs(np.mean(np.exp(1j * theta_i)))
        omega_n[i] = np.linalg.norm(omega_i, 2)

    # 崩溃检测: ||ω||_2 > synctol (对齐 reference_code swing.jl line 225)
    collapse_mask = omega_n > synctol
    t_collapse = float(sol.t[np.argmax(collapse_mask)]) if np.any(collapse_mask) else np.nan

    return sol.t[:T_eval], r_vals, omega_n, t_collapse


# ═══════════════════════════════════════════════════════════════
# 早期预警指标
# ═══════════════════════════════════════════════════════════════

def compute_sliding_variance(r, window_size):
    """滑动窗口方差 Var[r]"""
    out = np.full_like(r, np.nan)
    for i in range(window_size, len(r)):
        out[i] = np.var(r[i - window_size:i])
    return out


def compute_sliding_autocorr(r, window_size, lag=1):
    """滑动窗口 lag-1 自相关系数"""
    out = np.full_like(r, np.nan)
    min_pts = max(lag + 2, 4)  # 至少 4 个点才有意义
    for i in range(max(window_size, min_pts), len(r)):
        w = r[i - window_size:i]
        std_w = np.std(w)
        if std_w < 1e-14 or len(w) <= lag + 1:
            out[i] = np.nan
        else:
            out[i] = np.corrcoef(w[:-lag], w[lag:])[0, 1]
    return out


# ═══════════════════════════════════════════════════════════════
# 单次 realization 工作函数
# ═══════════════════════════════════════════════════════════════

def run_one_realization(args_tuple):
    """单次 realization (可被 multiprocessing 调用)。"""
    (idx, child_seed_entropy, season, t_grid, kappa,
     K_bar, q_rewire, data_mode, window_pts) = args_tuple

    rng = np.random.default_rng(child_seed_entropy)

    # 1) 构建日曲线
    profiles = build_daily_profiles_for_realization(season, rng,
                                                    data_mode=data_mode)

    # 2) 构建 P_k(t)
    P_matrix, _ = build_node_injections(profiles, t_grid, balance_at_pcc=True)

    # 3) 生成网络
    net_seed = int(rng.integers(0, 2**31))
    G = generate_ws_network(N, K_bar, q_rewire, seed=net_seed)
    A = nx.to_numpy_array(G)

    # 4) 初态
    psi0 = find_initial_steady_state(A, P_matrix[0], N, kappa, rng)

    # 5) 24h 仿真
    t_eval, r_vals, omega_n, t_collapse = simulate_24h(
        A, P_matrix, t_grid, N, kappa, psi0,
    )

    # 6) 早期预警
    var_r = compute_sliding_variance(r_vals, window_pts)
    ac1_r = compute_sliding_autocorr(r_vals, window_pts)

    return {
        "r": r_vals,
        "omega_norm": omega_n,
        "t_collapse": t_collapse,
        "var_r": var_r,
        "ac1_r": ac1_r,
    }


# ═══════════════════════════════════════════════════════════════
# 主驱动
# ═══════════════════════════════════════════════════════════════

def run_ensemble(season, t_grid, n_real, base_seed, kappa,
                 K_bar, q_rewire, data_mode, n_workers):
    """运行 ensemble 仿真，返回汇总结果。"""
    freq_sec = t_grid[1] - t_grid[0]
    window_pts = max(2, int(SLIDING_WINDOW_SEC / freq_sec))

    # 生成独立子种子 (可复现)
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(n_real)
    child_entropies = [cs.entropy for cs in child_seeds]

    job_args = [
        (i, child_entropies[i], season, t_grid, kappa,
         K_bar, q_rewire, data_mode, window_pts)
        for i in range(n_real)
    ]

    results_list = []
    if n_workers > 1:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            for res in tqdm(pool.imap(run_one_realization, job_args),
                            total=n_real, desc=f"[{season}] 仿真"):
                results_list.append(res)
    else:
        for args in tqdm(job_args, desc=f"[{season}] 仿真"):
            results_list.append(run_one_realization(args))

    # 汇总
    T_pts = len(t_grid)
    r_ensemble = np.zeros((n_real, T_pts))
    omega_ensemble = np.zeros((n_real, T_pts))
    t_collapse_arr = np.zeros(n_real)
    var_r_ensemble = np.zeros((n_real, T_pts))
    ac1_r_ensemble = np.zeros((n_real, T_pts))

    for i, res in enumerate(results_list):
        L = min(len(res["r"]), T_pts)
        r_ensemble[i, :L] = res["r"][:L]
        omega_ensemble[i, :L] = res["omega_norm"][:L]
        t_collapse_arr[i] = res["t_collapse"]
        var_r_ensemble[i, :L] = res["var_r"][:L]
        ac1_r_ensemble[i, :L] = res["ac1_r"][:L]

    return {
        "t_grid": t_grid,
        "r_ensemble": r_ensemble,
        "omega_norm_ensemble": omega_ensemble,
        "t_collapse": t_collapse_arr,
        "var_r": var_r_ensemble,
        "ac1_r": ac1_r_ensemble,
    }


def save_results(results, season, metadata, out_prefix=None):
    """落盘到 cache/results_{season}.npz"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if out_prefix:
        fname = CACHE_DIR / f"{out_prefix}_{season}.npz"
    else:
        fname = CACHE_DIR / f"results_{season}.npz"

    np.savez_compressed(
        fname,
        t_grid=results["t_grid"],
        r_ensemble=results["r_ensemble"],
        omega_norm_ensemble=results["omega_norm_ensemble"],
        t_collapse=results["t_collapse"],
        var_r=results["var_r"],
        ac1_r=results["ac1_r"],
        metadata_json=json.dumps(metadata),
    )
    return fname


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="时变 P_k(t) 同步仿真 (§4.3.1) — 只计算，不画图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--season", choices=["summer", "winter"], required=True,
                   help="季节 (summer=7月, winter=1月)")
    p.add_argument("--freq", default=DEFAULT_FREQ,
                   help="数据/监控频率, 如 5min, 30min")
    p.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS,
                   help="Monte Carlo realization 数")
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                   help="随机数基础种子 (可复现)")
    p.add_argument("--kappa", type=float, default=KAPPA,
                   help="耦合强度 κ")
    p.add_argument("--K", type=int, default=K_BAR,
                   help="WS 平均度")
    p.add_argument("--q", type=float, default=Q_REWIRE,
                   help="WS 重连概率")
    p.add_argument("--data-mode", choices=["processed", "raw", "synthetic"],
                   default="processed",
                   help="数据模式: processed=预处理聚合CSV(快), raw=原始CSV(慢), synthetic=合成曲线")
    p.add_argument("--synthetic", action="store_true",
                   help="(快捷方式) 等同于 --data-mode synthetic")
    p.add_argument("--out-prefix", default=None,
                   help="输出文件前缀 (默认 results)")
    p.add_argument("--workers", type=int, default=1,
                   help="并行 worker 数 (1=串行, 0=CPU 数)")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.workers == 0:
        from multiprocessing import cpu_count
        args.workers = cpu_count()

    # --synthetic 快捷方式覆盖 --data-mode
    data_mode = "synthetic" if args.synthetic else args.data_mode

    # 时间网格
    freq_sec = pd.Timedelta(args.freq).total_seconds()
    t_grid = np.arange(0, T_TOTAL + freq_sec / 2, freq_sec)

    # 数据模式显示名
    _mode_labels = {
        "processed": "预处理聚合CSV (household/)",
        "raw": "原始CSV (data-driven LCL+PV)",
        "synthetic": "合成曲线",
    }

    # 打印配置摘要
    print("=" * 60)
    print("时变 P_k(t) 同步仿真 — 配置摘要")
    print("=" * 60)
    print(f"  季节       : {args.season} (month={7 if args.season=='summer' else 1})")
    print(f"  数据源     : {_mode_labels.get(data_mode, data_mode)}")
    print(f"  频率       : {args.freq} ({freq_sec:.0f} s)")
    print(f"  时间范围   : 0 – {T_TOTAL:.0f} s (24h), {len(t_grid)} 个评估点")
    print(f"  网络       : WS(N={N}, K={args.K}, q={args.q})")
    print(f"  PCC 节点   : {PCC_NODE} (最后一个, 平衡功率)")
    print(f"  κ={args.kappa}, I={I_INERTIA}, D={D_DAMP}, synctol={SYNCTOL}")
    print(f"  Realizations: {args.realizations}")
    print(f"  Base seed  : {args.base_seed}")
    print(f"  Workers    : {args.workers}")
    print("=" * 60)

    t0 = time.time()
    results = run_ensemble(
        season=args.season,
        t_grid=t_grid,
        n_real=args.realizations,
        base_seed=args.base_seed,
        kappa=args.kappa,
        K_bar=args.K,
        q_rewire=args.q,
        data_mode=data_mode,
        n_workers=args.workers,
    )
    elapsed = time.time() - t0

    # 元数据
    metadata = {
        "season": args.season,
        "freq": args.freq,
        "freq_sec": freq_sec,
        "realizations": args.realizations,
        "base_seed": args.base_seed,
        "N": N,
        "K": args.K,
        "q": args.q,
        "kappa": args.kappa,
        "I": I_INERTIA,
        "D": D_DAMP,
        "synctol": SYNCTOL,
        "PCC_node": PCC_NODE,
        "data_mode": data_mode,
        "synthetic": data_mode == "synthetic",
        "elapsed_sec": elapsed,
    }

    fname = save_results(results, args.season, metadata,
                         out_prefix=args.out_prefix)

    # 摘要统计
    tc = results["t_collapse"]
    n_collapsed = int(np.sum(~np.isnan(tc)))
    print(f"\n完成! 耗时 {elapsed:.1f}s")
    print(f"  崩溃 realization 数: {n_collapsed}/{args.realizations}")
    if n_collapsed > 0:
        valid_tc = tc[~np.isnan(tc)]
        print(f"  崩溃时刻 (h): mean={np.mean(valid_tc)/3600:.2f}, "
              f"median={np.median(valid_tc)/3600:.2f}")
    print(f"  r(t) 均值 (全时段): {np.nanmean(results['r_ensemble']):.4f}")
    print(f"  结果已保存: {fname}")


if __name__ == "__main__":
    main()
