"""
plot_kappa_diurnal.py — 绘制日变临界耦合 κ_c(h) 曲线。

读取 results/kappa_diurnal.csv，输出 results/kappa_diurnal.png。

用法:
  MPLBACKEND=Agg python plot_kappa_diurnal.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS_DIR, DPI
from shared_diurnal_utils import load_pv_generation


def main():
    # --- 加载数据 ---
    csv_path = RESULTS_DIR / "kappa_diurnal.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run compute_kappa_diurnal.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    pv = load_pv_generation()

    # --- 设置绘图风格 ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.linewidth": 1.2,
    })

    fig, ax1 = plt.subplots(figsize=(10, 5))

    colors = {"summer": "#c24c51", "winter": "#4c70b0"}
    labels = {"summer": "Summer (Jun–Aug)", "winter": "Winter (Dec–Feb)"}
    hours = np.arange(24)

    # --- 主 y 轴: κ_c 曲线 ---
    for season in ["summer", "winter"]:
        sub = df[df["season"] == season].sort_values("hour")
        mean = sub["kappa_c_mean"].values
        std = sub["kappa_c_std"].values.copy()
        std = np.nan_to_num(std, nan=0.0)
        h = sub["hour"].values

        ax1.plot(h, mean, color=colors[season], linewidth=2,
                 marker="o", markersize=4, label=labels[season], zorder=5)
        ax1.fill_between(h, mean - std, mean + std,
                         color=colors[season], alpha=0.15, zorder=2)

        # 标注 peak 和 min (简单文字，避免箭头与 tight_layout 冲突)
        peak_idx = np.argmax(mean)
        min_idx = np.argmin(mean)

        ax1.plot(h[peak_idx], mean[peak_idx], "v", color=colors[season],
                 markersize=8, zorder=6)
        ax1.text(h[peak_idx], mean[peak_idx] + std[peak_idx] * 0.5 + 0.3,
                 f"h={h[peak_idx]}", fontsize=8, color=colors[season],
                 ha="center", fontweight="bold")

        ax1.plot(h[min_idx], mean[min_idx], "^", color=colors[season],
                 markersize=8, zorder=6)
        ax1.text(h[min_idx], mean[min_idx] - std[min_idx] * 0.5 - 0.3,
                 f"h={h[min_idx]}", fontsize=8, color=colors[season],
                 ha="center", va="top", fontweight="bold")

    ax1.set_xlabel("Hour of Day", fontsize=13)
    ax1.set_ylabel(r"Critical Coupling $\kappa_c$", fontsize=13)
    ax1.set_xticks(hours)
    ax1.set_xlim(-0.5, 23.5)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)

    # --- 次 y 轴: PV 发电曲线 ---
    ax2 = ax1.twinx()
    for season in ["summer", "winter"]:
        ax2.plot(hours, pv[season], color=colors[season],
                 linewidth=1.2, linestyle="--", alpha=0.4)
    ax2.set_ylabel("PV Generation (kW)", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # PV 图例（手动添加到次轴）
    from matplotlib.lines import Line2D
    pv_handles = [
        Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--", alpha=0.5),
    ]
    ax2.legend(pv_handles, ["PV generation (dashed)"],
               loc="upper right", fontsize=9, framealpha=0.7)

    fig.tight_layout()

    out_path = RESULTS_DIR / "kappa_diurnal.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
