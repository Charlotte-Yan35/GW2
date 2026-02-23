"""
实验一：κ_c vs 拓扑参数（基于均匀节点的 data-driven 设计）

用 LCL 用电均值 + PV 发电均值构造功率向量，
扫描 WS 拓扑参数 (q, K)，测量 κ_c。

输出:
  1. κ_c(q, K) 热力图 × 6时刻 × 2季节
  2. κ_c vs q 曲线（不同 penetration 对比）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from multiprocessing import Pool, cpu_count
from pathlib import Path
import random
import warnings
import time

from swing_utils import generate_network, find_kappa_c

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 路径
# ============================================================
BASE_DIR = Path(__file__).parent
HOUSEHOLD_DIR = BASE_DIR.parent / "household"
DEMAND_CSV = HOUSEHOLD_DIR / "consumer" / "output" / "monthly_hourly_usage.csv"
GEN_CSV = HOUSEHOLD_DIR / "generation" / "output" / "generation_range_by_season.csv"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 全局参数
# ============================================================
N = 50
I_INERTIA = 1.0
D_DAMPING = 1.0
REALIZATIONS = 20

Q_LIST = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]
K_LIST = [4, 6, 8, 10]
HOURS = [0, 4, 8, 12, 16, 20]
PENETRATIONS = [12, 24, 37, 49]  # ~25%, 50%, 75%, 100%

# Season → month mapping (for demand data)
SEASON_CONFIG = {
    "Winter": {"month": 1, "gen_col": "Winter_Mean"},
    "Summer": {"month": 7, "gen_col": "Summer_Mean"},
}


# ============================================================
# 数据加载
# ============================================================

def load_demand_profile(month):
    """从 monthly_hourly_usage.csv 读取指定月的 24h avg_kwh，返回长度24的数组。"""
    df = pd.read_csv(DEMAND_CSV)
    subset = df[df["month"] == month].sort_values("hour")
    return subset["avg_kwh"].values  # shape (24,)


def load_generation_profile(gen_col):
    """从 generation_range_by_season.csv 读取指定季节列的 24h Mean，返回长度24的数组。"""
    df = pd.read_csv(GEN_CSV)
    df = df.sort_values("Hour")
    return df[gen_col].values  # shape (24,)


# ============================================================
# P 向量构造
# ============================================================

def build_power_vector(n, penetration, hour, demand_24h, gen_24h):
    """
    构造 N 维功率向量：
    - P[0:penetration] = gen(hour) - demand(hour)  (PV节点)
    - P[penetration:n-1] = -demand(hour)            (纯消费节点)
    - P[n-1] = -sum(P[0:n-1])                       (slack)
    """
    demand = demand_24h[hour]
    gen = gen_24h[hour]

    P = np.zeros(n)
    P[:penetration] = gen - demand
    P[penetration:n - 1] = -demand
    P[n - 1] = -np.sum(P[:n - 1])
    return P


# ============================================================
# 单次 κ_c 计算（用于并行）
# ============================================================

def compute_kappa_c_single(args):
    """单次实现 (用于并行 map)。"""
    n, K_bar, q, P, seed = args
    np.random.seed(seed)
    random.seed(seed)
    A = generate_network(n, K_bar, q)
    return find_kappa_c(A, P, n, I=I_INERTIA, D=D_DAMPING)


# ============================================================
# 扫描（全部批量提交到一个 Pool）
# ============================================================

def run_all_sweeps(realizations=REALIZATIONS):
    """批量计算所有配置，使用单个 Pool，返回结构化结果。"""
    rng = np.random.default_rng(42)

    # Step 1: 构建所有任务
    jobs = []  # list of (key, args_list, cache_file)
    # key = (pen, season_name, hour, qi, ki)

    for pen in PENETRATIONS:
        for season_name, season_cfg in SEASON_CONFIG.items():
            demand_24h = load_demand_profile(season_cfg["month"])
            gen_24h = load_generation_profile(season_cfg["gen_col"])

            for hour in HOURS:
                P = build_power_vector(N, pen, hour, demand_24h, gen_24h)

                for qi, q in enumerate(Q_LIST):
                    for ki, K in enumerate(K_LIST):
                        key = (pen, season_name, hour, qi, ki)
                        tag = (f"exp1_kc_s{season_name}_m{season_cfg['month']}"
                               f"_p{pen}_K{K}_q{q}_R{realizations}_h{hour}")
                        cache_file = CACHE_DIR / f"{tag}.npz"

                        # Check cache
                        if cache_file.exists():
                            jobs.append((key, None, cache_file))
                        else:
                            seeds = rng.integers(0, 10**9, size=realizations)
                            args_list = [(N, K, q, P, int(s)) for s in seeds]
                            jobs.append((key, args_list, cache_file))

    # Step 2: 分离已缓存 vs 需计算的
    cached_results = {}
    to_compute = []  # (key, args_list, cache_file)

    for key, args_list, cache_file in jobs:
        if args_list is None:
            data = np.load(cache_file)
            values = data["kappa_c_values"]
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                cached_results[key] = (np.nan, np.nan)
            else:
                cached_results[key] = (np.mean(valid), np.std(valid))
        else:
            to_compute.append((key, args_list, cache_file))

    print(f"Total configs: {len(jobs)}, cached: {len(cached_results)}, "
          f"to compute: {len(to_compute)}")

    if len(to_compute) == 0:
        return cached_results

    # Step 3: 扁平化所有任务，一次性提交到 Pool
    flat_args = []
    job_indices = []  # (job_idx, realization_idx)
    for job_idx, (key, args_list, cache_file) in enumerate(to_compute):
        for r_idx, args in enumerate(args_list):
            flat_args.append(args)
            job_indices.append((job_idx, r_idx))

    print(f"Submitting {len(flat_args)} tasks to Pool "
          f"({len(to_compute)} configs × {realizations} realizations)")

    t0 = time.time()
    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        flat_results = pool.map(compute_kappa_c_single, flat_args)
    elapsed = time.time() - t0
    print(f"Pool completed in {elapsed:.1f}s "
          f"({elapsed / len(flat_args):.2f}s per task)")

    # Step 4: 重组结果，保存缓存
    all_results = dict(cached_results)

    # Group flat results back by job
    job_values = [np.full(realizations, np.nan) for _ in to_compute]
    for flat_idx, (job_idx, r_idx) in enumerate(job_indices):
        job_values[job_idx][r_idx] = flat_results[flat_idx]

    for job_idx, (key, args_list, cache_file) in enumerate(to_compute):
        values = job_values[job_idx]
        np.savez(cache_file, kappa_c_values=values)

        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            all_results[key] = (np.nan, np.nan)
        else:
            all_results[key] = (np.mean(valid), np.std(valid))

    return all_results


# ============================================================
# 可视化
# ============================================================

def plot_heatmaps(all_results, penetration):
    """
    6×2 子图：6时刻(0,4,8,12,16,20h) × 2季节 的 κ_c(q,K) 热力图
    """
    seasons = list(SEASON_CONFIG.keys())
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    # Collect all values for shared colorbar
    all_vals = []
    for season in seasons:
        for hour in HOURS:
            for qi in range(len(Q_LIST)):
                for ki in range(len(K_LIST)):
                    key = (penetration, season, hour, qi, ki)
                    v = all_results.get(key, (np.nan, 0))[0]
                    if not np.isnan(v):
                        all_vals.append(v)

    if len(all_vals) == 0:
        print("No valid data for heatmaps!")
        return

    vmin, vmax = min(all_vals), max(all_vals)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for si, season in enumerate(seasons):
        for hi, hour in enumerate(HOURS):
            ax = axes[si, hi]
            grid = np.full((len(Q_LIST), len(K_LIST)), np.nan)
            for qi in range(len(Q_LIST)):
                for ki in range(len(K_LIST)):
                    key = (penetration, season, hour, qi, ki)
                    grid[qi, ki] = all_results.get(key, (np.nan, 0))[0]

            im = ax.imshow(grid, aspect='auto', origin='lower',
                           norm=norm, cmap='YlGnBu_r',
                           extent=[-0.5, len(K_LIST) - 0.5,
                                   -0.5, len(Q_LIST) - 0.5])

            ax.set_xticks(range(len(K_LIST)))
            ax.set_xticklabels(K_LIST)
            ax.set_yticks(range(len(Q_LIST)))
            ax.set_yticklabels([f"{q:.2f}" for q in Q_LIST])

            if hi == 0:
                ax.set_ylabel(f"{season}\nq", fontsize=10)
            else:
                ax.set_yticklabels([])
            if si == 1:
                ax.set_xlabel("K", fontsize=10)
            ax.set_title(f"h={hour:02d}:00", fontsize=10)

            # Annotate cells
            for qi in range(len(Q_LIST)):
                for ki in range(len(K_LIST)):
                    val = grid[qi, ki]
                    if not np.isnan(val):
                        ax.text(ki, qi, f"{val:.2f}", ha='center',
                                va='center', fontsize=7,
                                color='white' if val > (vmin + vmax) / 2
                                else 'black')

    fig.suptitle(f"$\\kappa_c(q, K)$ — Penetration = {penetration}/{N} "
                 f"({penetration * 100 // N}%)",
                 fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=axes, shrink=0.6, label=r"$\overline{\kappa}_c$")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    fname = OUTPUT_DIR / f"exp1_heatmap_pen{penetration}.png"
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)


def plot_penetration_curves(all_results):
    """
    κ_c vs q 曲线，不同 penetration 用不同颜色。
    每个季节×时刻一张子图。
    """
    seasons = list(SEASON_CONFIG.keys())
    colors = ['#e8850c', '#4a7ebb', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    for si, season in enumerate(seasons):
        for hi, hour in enumerate(HOURS):
            ax = axes[si, hi]

            for pi, pen in enumerate(PENETRATIONS):
                ki_default = 0  # K=4
                q_vals = []
                kc_means = []
                kc_stds = []
                for qi, q in enumerate(Q_LIST):
                    key = (pen, season, hour, qi, ki_default)
                    mean, std = all_results.get(key, (np.nan, 0))
                    q_vals.append(q)
                    kc_means.append(mean)
                    kc_stds.append(std)

                q_vals = np.array(q_vals)
                kc_means = np.array(kc_means)
                kc_stds = np.array(kc_stds)
                valid = ~np.isnan(kc_means)

                ax.plot(q_vals[valid], kc_means[valid],
                        'o-', color=colors[pi], lw=1.5, ms=4,
                        label=f"pen={pen}")
                ax.fill_between(q_vals[valid],
                                kc_means[valid] - kc_stds[valid],
                                kc_means[valid] + kc_stds[valid],
                                color=colors[pi], alpha=0.2)

            if hi == 0:
                ax.set_ylabel(f"{season}\n" + r"$\overline{\kappa}_c$",
                              fontsize=10)
            if si == 1:
                ax.set_xlabel("q", fontsize=10)
            ax.set_title(f"h={hour:02d}:00", fontsize=10)
            if si == 0 and hi == 5:
                ax.legend(fontsize=7, loc='best')

    fig.suptitle(r"$\kappa_c$ vs $q$ (K=4) — different PV penetrations",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname = OUTPUT_DIR / f"exp1_penetration_curves.png"
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("实验一：κ_c vs 拓扑参数（data-driven 均匀节点）")
    print("=" * 60)

    # 一次性计算所有配置
    all_results = run_all_sweeps()

    # Print summary
    for pen in PENETRATIONS:
        print(f"\n--- Penetration = {pen}/{N} ---")
        for season_name in SEASON_CONFIG:
            for hour in HOURS:
                for qi, q in enumerate(Q_LIST):
                    for ki, K in enumerate(K_LIST):
                        key = (pen, season_name, hour, qi, ki)
                        mean, std = all_results.get(key, (np.nan, np.nan))
                        if qi == 0 and ki == 0:
                            print(f"  {season_name} h={hour:02d} "
                                  f"q={q:.2f} K={K:2d} → "
                                  f"κ_c = {mean:.4f} ± {std:.4f}")

    # 生成图表
    print("\n--- Generating heatmaps ---")
    for pen in PENETRATIONS:
        plot_heatmaps(all_results, pen)

    print("\n--- Generating penetration curves ---")
    plot_penetration_curves(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
