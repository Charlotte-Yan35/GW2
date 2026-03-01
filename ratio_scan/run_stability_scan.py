"""
run_stability_scan.py — Stability 阶段: 临界耦合强度 kappa_c 扫描

kappa_c 定义: 使稳定稳态存在的最小耦合强度 (bracket 上界)。

算法: Bracket + Bisection
  1. Bracket: 找 kappa_lo (不稳定) 和 kappa_hi (稳定)
  2. Bisection: 直到 kappa_hi - kappa_lo < KAPPA_TOL
  3. 返回 kappa_hi (上界)

稳定性判断:
  - 用 fsteadystate_gauge 求解稳态 (gauge fixing: theta[PCC]=0)
  - 收敛条件: norm(residual) < 1e-5

用法:
  python -m ratio_scan.run_stability_scan          # 完整
  python -m ratio_scan.run_stability_scan --test    # 小规模验证
"""

import hashlib
import time
import csv
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

try:
    from ratio_scan.shared_utils import (
        N, PMAX, I_INERTIA, D_DAMP,
        generate_ws_network,
        fswing, fsteadystate,
        build_ratio_grid, assign_powers,
    )
except ModuleNotFoundError:
    from shared_utils import (
        N, PMAX, I_INERTIA, D_DAMP,
        generate_ws_network,
        fswing, fsteadystate,
        build_ratio_grid, assign_powers,
    )

# ── 路径 ──────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
RESULTS_DIR = _MODULE_DIR / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────────────
K_VALUES = [4, 8]
Q_VALUES = [0.0, 0.15, 1.0]
RATIO_STEP = 10
REALIZATIONS = 20
ROLE_SEEDS = 5

KAPPA_START = 5.5      # 与 reproduction/figure1.py 一致
KAPPA_STEP_INIT = 0.1  # 与 reproduction/figure1.py 一致
KAPPA_TOL = 1e-3

CSV_HEADER = ["K", "q", "ng", "nc", "np", "net_seed", "role_seed", "kappa_c"]


# ====================================================================
# 确定性种子
# ====================================================================

def make_seed(*args):
    """确定性种子: 基于参数的 MD5 哈希。"""
    s = "_".join(map(str, args))
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ====================================================================
# Swing ODE 积分 + 稳态检查
# ====================================================================

def _integrate_swing(A, P, n, kappa, y0, t_max=200.0):
    """积分 Swing 方程，返回 (converged, y_final)。

    Warm-start 友好：接受任意初始条件 y0。
    与 figure1.py integrate_swing 一致：无 gauge fixing。
    """
    def rhs(t, y, _A=A, _P=P, _n=n, _kappa=kappa):
        return fswing(y, _A, _P, _n, I_INERTIA, D_DAMP, _kappa)

    sol = solve_ivp(rhs, [0, t_max], y0, method='RK45',
                    rtol=1e-8, atol=1e-8, max_step=1.0)
    if sol.status != 0:
        return False, y0

    y_final = sol.y[:, -1].copy()
    theta_final = y_final[n:]

    resid = fsteadystate(theta_final, A, P, kappa)
    converged = np.linalg.norm(resid, 2) < 1e-5
    return converged, y_final


# ====================================================================
# Warm-start 递降搜索 kappa_c (与参考代码一致)
# ====================================================================

def find_kappa_c(A, P, rng_seed, kappa_start=None, step_init=None,
                 tol=None):
    """Warm-start 递降搜索 kappa_c。

    与 reproduction/figure1.py find_kappa_c 算法一致:
    从高 kappa 开始积分求稳态，然后逐步降低 kappa，
    每次用上一步的稳态解作为初始条件 (warm-start)。
    失败时步长减半并回退，收敛条件: stepsize < tol。

    返回 kappa_c 或 NaN。
    """
    if kappa_start is None:
        kappa_start = KAPPA_START
    if step_init is None:
        step_init = KAPPA_STEP_INIT
    if tol is None:
        tol = KAPPA_TOL

    n = len(P)
    rng = np.random.default_rng(rng_seed)

    kappa = kappa_start
    y0 = rng.random(2 * n)
    converged, y_last = _integrate_swing(A, P, n, kappa, y0, t_max=200.0)

    if not converged:
        # 尝试更大的 kappa
        for mult in [2.0, 5.0, 10.0]:
            y0 = rng.random(2 * n)
            converged, y_last = _integrate_swing(
                A, P, n, kappa_start * mult, y0, t_max=200.0)
            if converged:
                kappa = kappa_start * mult
                break
        else:
            return np.nan

    stepsize = step_init
    kappa_old = kappa

    for _ in range(500):
        converged, y_sol = _integrate_swing(
            A, P, n, kappa, y_last, t_max=100.0)

        if converged:
            y_last = y_sol
            if stepsize < tol:
                return kappa
            kappa_old = kappa
            kappa -= stepsize
        else:
            stepsize /= 2.0
            kappa = kappa_old - stepsize

        if kappa < 0 or stepsize < 1e-6:
            return kappa_old

    return kappa_old


# ====================================================================
# 单任务计算 (for multiprocessing)
# ====================================================================

def _compute_one_task(args):
    """计算单个 (K, q, ng, nc, np_count, net_seed, role_seed) 的 kappa_c。

    Returns
    -------
    list  CSV 行数据
    """
    K, q, ng, nc, np_count, net_seed, role_seed = args

    G = generate_ws_network(N, K, q, seed=net_seed)
    A = nx.to_numpy_array(G)  # 与 figure1.py 一致
    P = assign_powers(N, ng, nc, seed=role_seed)

    rng_seed = make_seed("stability", K, q, ng, nc, net_seed, role_seed)
    kc = find_kappa_c(A, P, rng_seed)

    return [K, q, ng, nc, np_count, net_seed, role_seed, kc]


# ====================================================================
# 主实验循环
# ====================================================================

def run_experiment(test_mode=False):
    """构建比例网格 x 拓扑参数 -> 并行计算 kappa_c -> 聚合。"""
    if test_mode:
        k_values = [4]
        q_values = [0.0]
        ratio_step = 5
        realizations = 2
        role_seeds = 2
    else:
        k_values = K_VALUES
        q_values = Q_VALUES
        ratio_step = RATIO_STEP
        realizations = REALIZATIONS
        role_seeds = ROLE_SEEDS

    ratio_grid = build_ratio_grid(ratio_step, total=N)  # N=50，无 PCC
    if test_mode:
        ratio_grid = ratio_grid[:3]

    raw_path = CACHE_DIR / "stability_results.csv"

    # 旧缓存检测: 若已有数据 ng+nc+np=49 (旧 PCC 模式)，提示清除
    if raw_path.exists():
        import pandas as _pd
        _df_check = _pd.read_csv(raw_path, nrows=5)
        if {"ng", "nc", "np"}.issubset(_df_check.columns) and len(_df_check) > 0:
            _row = _df_check.iloc[0]
            _total = int(_row["ng"]) + int(_row["nc"]) + int(_row["np"])
            if _total != N:
                print(f"[WARN] 已有缓存 ng+nc+np={_total} (期望 {N})，"
                      f"计算语义已变。请删除旧缓存:")
                print(f"  rm -f {raw_path}")
                print(f"  rm -f {CACHE_DIR / 'stability_agg.csv'}")
                return
        del _df_check

    # Resume: 读已有结果
    done_keys = set()
    if raw_path.exists():
        with open(raw_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["K"], row["q"], row["ng"], row["nc"],
                       row["np"], row["net_seed"], row["role_seed"])
                done_keys.add(key)
        print(f"Resume: {len(done_keys)} tasks already done")

    # 构建任务列表
    tasks = []
    for ng, nc, np_count in ratio_grid:
        for K in k_values:
            for q in q_values:
                for r in range(realizations):
                    net_seed = make_seed("net", K, q, ng, nc, r)
                    for rs in range(role_seeds):
                        role_seed = make_seed("role", K, q, ng, nc, r, rs)
                        key = (str(K), str(q), str(ng), str(nc),
                               str(np_count), str(net_seed), str(role_seed))
                        if key in done_keys:
                            continue
                        tasks.append((K, q, ng, nc, np_count,
                                      net_seed, role_seed))

    n_ratios = len(ratio_grid)
    n_topos = len(k_values) * len(q_values)
    total_possible = n_ratios * n_topos * realizations * role_seeds
    print(f"Stability scan: {n_ratios} ratios x {n_topos} topos "
          f"x {realizations} nets x {role_seeds} roles = {total_possible} total")
    print(f"Tasks to compute: {len(tasks)} (skipped {len(done_keys)} done)")

    if not tasks:
        print("All tasks already completed!")
        _aggregate(raw_path)
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
        for task in tasks:
            result = _compute_one_task(task)
            raw_writer.writerow(result)
            raw_file.flush()
            done += 1
            K, q, ng, nc, np_c, ns, rs = task
            kc = result[7]
            kc_str = f"{kc:.4f}" if not np.isnan(kc) else "NaN"
            print(f"  [{done}/{total_possible}] K={K} q={q} "
                  f"ng={ng} nc={nc} np={np_c} -> kappa_c={kc_str}")
    else:
        n_workers = max(1, cpu_count() - 1)
        print(f"Using {n_workers} worker processes")

        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_compute_one_task, tasks,
                                              chunksize=1):
                raw_writer.writerow(result)
                raw_file.flush()
                done += 1
                if done % 10 == 0 or done == total_possible:
                    elapsed = time.time() - t_start
                    computed = done - len(done_keys)
                    rate = computed / elapsed if elapsed > 0 else 0
                    remaining = total_possible - done
                    eta = remaining / rate if rate > 0 else 0
                    kc = result[7]
                    kc_str = f"{kc:.4f}" if not np.isnan(kc) else "NaN"
                    print(f"  [{done}/{total_possible}] kappa_c={kc_str} "
                          f"rate={rate:.2f}/s ETA={eta/60:.1f}min")

    raw_file.close()
    print(f"\nRaw results saved -> {raw_path}")

    _aggregate(raw_path)


# ====================================================================
# 聚合
# ====================================================================

def _aggregate(raw_path):
    """从 raw CSV 聚合为 stability_agg.csv。"""
    if not raw_path.exists():
        print("No raw results to aggregate.")
        return

    import pandas as pd

    df = pd.read_csv(raw_path)
    df["kappa_c"] = pd.to_numeric(df["kappa_c"], errors="coerce")
    print(f"Loaded {len(df)} rows from {raw_path}")

    # 按 (K, q, ng, nc, np) 聚合
    valid = df.dropna(subset=["kappa_c"])
    if valid.empty:
        print("No valid kappa_c values to aggregate.")
        return

    agg = valid.groupby(["K", "q", "ng", "nc", "np"]).agg(
        kappa_c_mean=("kappa_c", "mean"),
        kappa_c_std=("kappa_c", "std"),
        n_valid=("kappa_c", "count"),
    ).reset_index()

    agg_path = CACHE_DIR / "stability_agg.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Aggregated {len(agg)} ratio points -> {agg_path}")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stability Scan: kappa_c bracket+bisection")
    parser.add_argument("--test", action="store_true",
                        help="小规模测试模式")
    args = parser.parse_args()

    run_experiment(test_mode=args.test)
