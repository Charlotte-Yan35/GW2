"""
run_cascade_duration.py — Figure S2 风格级联持续时间入口

用法:
  MPLBACKEND=Agg python run_cascade_duration.py              # 全部 3 ratio × 2 panel
  MPLBACKEND=Agg python run_cascade_duration.py balanced     # 单 ratio, 两个 panel
  MPLBACKEND=Agg python run_cascade_duration.py balanced A   # 单 ratio, 单 panel (A=q, B=K)
"""

import sys
import numpy as np

from ws_config import (
    RATIO_CONFIGS, DURATION_SWEEP_PANELS,
    S2_ALPHA_MIN, S2_ALPHA_MAX, S2_ALPHA_RES, S2_ENSEMBLE_SIZE, S2_SEED,
)
from swing_cascade.compute import compute_s2_panel
from swing_cascade.plots import plot_figS2_duration

RATIOS = list(RATIO_CONFIGS.keys())  # ["balanced", "gen_heavy", "load_heavy"]
PANEL_IDS = list(DURATION_SWEEP_PANELS.keys())  # ["A", "B"]


def main():
    alpha_values = np.linspace(S2_ALPHA_MIN, S2_ALPHA_MAX, S2_ALPHA_RES)

    # 解析命令行参数
    target_ratios = RATIOS
    target_panels = PANEL_IDS

    if len(sys.argv) > 1:
        ratio = sys.argv[1]
        if ratio not in RATIOS:
            print(f"Unknown ratio: {ratio}. Choose from {RATIOS}")
            sys.exit(1)
        target_ratios = [ratio]

    if len(sys.argv) > 2:
        panel = sys.argv[2].upper()
        if panel not in PANEL_IDS:
            print(f"Unknown panel: {panel}. Choose from {PANEL_IDS} (A=q sweep, B=K sweep)")
            sys.exit(1)
        target_panels = [panel]

    # 计算
    all_ratio_curves = {}  # {ratio: {panel: [curves]}}
    total_jobs = len(target_ratios) * len(target_panels)
    job_idx = 0

    for rn in target_ratios:
        all_ratio_curves[rn] = {}
        for pid in target_panels:
            job_idx += 1
            config = DURATION_SWEEP_PANELS[pid]
            print(f"\n{'='*60}")
            print(f"  [{job_idx}/{total_jobs}] {rn} — Panel {pid} "
                  f"({config.variable_name} sweep)")
            print(f"{'='*60}")

            curves = compute_s2_panel(
                rn, config, alpha_values,
                ensemble_size=S2_ENSEMBLE_SIZE, seed=S2_SEED,
            )
            all_ratio_curves[rn][pid] = curves

    # 绘图
    print(f"\n{'='*60}")
    print("  Plotting figS2_duration ...")
    print(f"{'='*60}")
    plot_figS2_duration(all_ratio_curves)

    print("\nAll done.")


if __name__ == "__main__":
    main()
