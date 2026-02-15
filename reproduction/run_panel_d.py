"""
单独计算 Panel D 数据并画图。
Panel D: κ̄_c vs consumers 截面图，固定 n_passive=15，q ∈ {0.0, 0.1, 0.4, 1.0}
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

from figure1 import (
    N, K_BAR, PMAX, REALIZATIONS, CACHE_DIR,
    compute_kappa_c_stats_parallel, plot_panel_d,
)

N_PASSIVE = 15


def compute_panel_d_with_progress(n=N, K_bar=K_BAR, Pmax=PMAX,
                                  realizations=REALIZATIONS, n_passive=N_PASSIVE):
    cache_file = CACHE_DIR / f"v4_panel_d_n{n}_k{K_bar}_np{n_passive}_r{realizations}.npz"
    if cache_file.exists():
        print(f"Found cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return (data['q_values'], data['n_minus_range'],
                data['kappa_c_mean'], data['kappa_c_std'])

    q_values = np.array([0.0, 0.1, 0.4, 1.0])
    n_active = n - n_passive
    n_minus_range = np.arange(1, n_active, 1)

    total = len(q_values) * len(n_minus_range)
    kappa_c_mean = np.full((len(q_values), len(n_minus_range)), np.nan)
    kappa_c_std = np.full((len(q_values), len(n_minus_range)), np.nan)

    n_workers = min(cpu_count(), 8)
    print(f"Workers: {n_workers}, Total tasks: {total}")
    print(f"q values: {q_values}, consumers: 1..{n_active-1}")

    t0 = time.time()
    done = 0

    with Pool(n_workers) as pool:
        for qi, q in enumerate(q_values):
            print(f"\n=== q = {q} ({qi+1}/{len(q_values)}) ===")
            for ni, n_minus in enumerate(n_minus_range):
                n_plus = n_active - n_minus
                mean_val, std_val = compute_kappa_c_stats_parallel(
                    n, K_bar, q, n_plus, n_minus, Pmax, realizations, pool
                )
                kappa_c_mean[qi, ni] = mean_val
                kappa_c_std[qi, ni] = std_val
                done += 1

                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] q={q} n-={n_minus:2d} "
                      f"κ̄_c={mean_val:.4f} ± {std_val:.4f}  "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    np.savez(cache_file, q_values=q_values, n_minus_range=n_minus_range,
             kappa_c_mean=kappa_c_mean, kappa_c_std=kappa_c_std)
    print(f"\nPanel D data cached to {cache_file}")
    return q_values, n_minus_range, kappa_c_mean, kappa_c_std


if __name__ == "__main__":
    print("=" * 50)
    print("Computing Panel D — κ̄_c cross-section")
    print("=" * 50)

    q_values, n_minus_range, kd_mean, kd_std = compute_panel_d_with_progress()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_panel_d(ax, q_values, n_minus_range, kd_mean, kd_std)
    fig.tight_layout()

    out = Path(__file__).parent / "panel_d_preview.png"
    fig.savefig(out, dpi=200)
    print(f"\nSaved {out}")
