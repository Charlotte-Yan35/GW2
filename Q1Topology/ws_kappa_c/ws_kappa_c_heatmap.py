"""
W-S 热力图: κ_c 在 (K_bar, q) 二维空间的变化，对比不同产用电配置
"""

import sys
from pathlib import Path

# 允许 import 同目录的 ws_kappa_c
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from ws_kappa_c import (
    generate_network, assign_powers, find_kappa_c,
    compute_kappa_c_single,
)

# ============================================================
# 参数
# ============================================================
N = 50
PMAX = 1.0
GAMMA = 1.0
REALIZATIONS = 100
K_BAR_VALUES = [2, 4, 6, 8, 10]
Q_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

CONFIGS = [
    (25, 25, 0),
    (45, 5, 0),
    (5, 45, 0),
    (17, 17, 16),
]
CONFIG_LABELS = [
    r"$(n_+, n_-, n_p) = (25, 25, 0)$",
    r"$(n_+, n_-, n_p) = (45, 5, 0)$",
    r"$(n_+, n_-, n_p) = (5, 45, 0)$",
    r"$(n_+, n_-, n_p) = (17, 17, 16)$",
]

KAPPA_START = 5.5
STEP_INIT = 0.1

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 计算
# ============================================================

def compute_kappa_c_single_custom(args):
    """单次实现，使用本脚本的 kappa_start / step_init。"""
    import random as _random
    n, K_bar, q, n_plus, n_minus, Pmax, seed = args
    np.random.seed(seed)
    _random.seed(seed)
    A = generate_network(n, K_bar, q)
    P = assign_powers(n, n_plus, n_minus, Pmax)
    return find_kappa_c(A, P, n, kappa_start=KAPPA_START, step_init=STEP_INIT)


def compute_heatmap(config_idx, n_plus, n_minus, n_passive):
    """计算单个配置的 5×6 热力图矩阵 (K_bar × q)。"""
    cache_file = CACHE_DIR / f"ws_kappa_c_heatmap_config{config_idx + 1}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  Config {config_idx + 1}: loaded from cache")
        return data['kappa_c_mean'], data['kappa_c_std']

    n_kbar = len(K_BAR_VALUES)
    n_q = len(Q_VALUES)
    kappa_c_mean = np.full((n_kbar, n_q), np.nan)
    kappa_c_std = np.full((n_kbar, n_q), np.nan)

    rng = np.random.default_rng(12345)
    seeds = rng.integers(0, 10**9, size=REALIZATIONS)

    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        for ki, K_bar in enumerate(K_BAR_VALUES):
            for qi, q in enumerate(Q_VALUES):
                print(f"  Config {config_idx + 1} "
                      f"({n_plus},{n_minus},{n_passive}), "
                      f"K_bar={K_bar}, q={q}")
                args_list = [
                    (N, K_bar, q, n_plus, n_minus, PMAX, int(s))
                    for s in seeds
                ]
                results = pool.map(compute_kappa_c_single_custom, args_list)
                values = np.array([r for r in results if not np.isnan(r)])
                if len(values) > 0:
                    kappa_c_mean[ki, qi] = np.mean(values)
                    kappa_c_std[ki, qi] = np.std(values)
                print(f"    κ_c = {kappa_c_mean[ki, qi]:.4f} "
                      f"± {kappa_c_std[ki, qi]:.4f}  "
                      f"(valid: {len(values)}/{REALIZATIONS})")

    np.savez(cache_file,
             kappa_c_mean=kappa_c_mean, kappa_c_std=kappa_c_std,
             K_bar_values=np.array(K_BAR_VALUES),
             q_values=np.array(Q_VALUES))
    print(f"  Cached to {cache_file}")
    return kappa_c_mean, kappa_c_std


# ============================================================
# 绘图
# ============================================================

def plot_single_heatmap(config_idx, kappa_c_mean, ax=None, vmin=None, vmax=None):
    """绘制单个热力图，可作为独立图或子图。"""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(kappa_c_mean, aspect='auto', origin='lower',
                   cmap='viridis_r', vmin=vmin, vmax=vmax)

    # 格子内标注数值
    for i in range(len(K_BAR_VALUES)):
        for j in range(len(Q_VALUES)):
            val = kappa_c_mean[i, j]
            if not np.isnan(val):
                color = 'white' if val > (vmin + vmax) / 2 else 'black' \
                    if vmin is not None else 'white'
                ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(range(len(Q_VALUES)))
    ax.set_xticklabels([f"{q:.1f}" for q in Q_VALUES])
    ax.set_yticks(range(len(K_BAR_VALUES)))
    ax.set_yticklabels([str(k) for k in K_BAR_VALUES])
    ax.set_xlabel(r"Rewiring probability $q$", fontsize=11)
    ax.set_ylabel(r"$\bar{K}$", fontsize=12)
    ax.set_title(CONFIG_LABELS[config_idx], fontsize=11)

    if standalone:
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(r"$\overline{\kappa}_c$", fontsize=12)
        fig.tight_layout()
        return fig, im
    return im


def plot_individual(config_idx, kappa_c_mean):
    """独立热力图，每个 config 一张。"""
    fig, im = plot_single_heatmap(config_idx, kappa_c_mean,
                                  vmin=np.nanmin(kappa_c_mean),
                                  vmax=np.nanmax(kappa_c_mean))
    fname = f"ws_kappa_c_heatmap_config{config_idx + 1}"
    fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}.png/.pdf")
    plt.close(fig)


def plot_combined(all_means):
    """2×2 子图汇总，统一色阶。"""
    global_vmin = min(np.nanmin(m) for m in all_means)
    global_vmax = max(np.nanmax(m) for m in all_means)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    ims = []
    for i, (ax, mean_mat) in enumerate(zip(axes, all_means)):
        im = plot_single_heatmap(i, mean_mat, ax=ax,
                                 vmin=global_vmin, vmax=global_vmax)
        ims.append(im)

    # 统一 colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label(r"$\overline{\kappa}_c$", fontsize=13)

    fig.suptitle(
        rf"W-S topology: $\kappa_c$ heatmap ($P_{{max}}={PMAX}$, $N={N}$, "
        rf"realizations$={REALIZATIONS})",
        fontsize=14, y=0.98
    )

    fname = "ws_kappa_c_heatmap_combined"
    fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}.png/.pdf")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("W-S 热力图: κ_c vs (K_bar, q)")
    print(f"N={N}, Pmax={PMAX}, γ={GAMMA}")
    print(f"K_bar values: {K_BAR_VALUES}")
    print(f"q values: {Q_VALUES}")
    print(f"Realizations: {REALIZATIONS}")
    print("=" * 60)

    all_means = []
    all_stds = []
    for i, (n_plus, n_minus, n_passive) in enumerate(CONFIGS):
        print(f"\n--- Config {i + 1}: ({n_plus}, {n_minus}, {n_passive}) ---")
        mean_mat, std_mat = compute_heatmap(i, n_plus, n_minus, n_passive)
        plot_individual(i, mean_mat)
        all_means.append(mean_mat)
        all_stds.append(std_mat)

    print("\n--- Combined heatmap ---")
    plot_combined(all_means)

    print("\nDone!")


if __name__ == "__main__":
    main()
