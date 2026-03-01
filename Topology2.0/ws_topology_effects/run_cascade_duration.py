"""
run_cascade_duration.py — 级联持续时间探索入口
用法: MPLBACKEND=Agg python run_cascade_duration.py [ratio_name]

不带参数：跑三个 ratio 的完整计算 + 绘图
带参数：只跑指定 ratio
"""

import sys
from swing_cascade.compute import compute_and_cache_duration
from swing_cascade.plots import (
    plot_cascade_duration,
    plot_cascade_duration_survival,
    plot_cascade_duration_combined,
    plot_all_duration,
)

ratios = ["balanced", "gen_heavy", "load_heavy"]


def main():
    if len(sys.argv) > 1:
        # 单个 ratio
        ratio = sys.argv[1]
        if ratio not in ratios:
            print(f"Unknown ratio: {ratio}. Choose from {ratios}")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"  Cascade duration: {ratio}")
        print(f"{'='*60}")
        compute_and_cache_duration(ratio)
        plot_cascade_duration(ratio)
        plot_cascade_duration_survival(ratio)
    else:
        # 全部三个 ratio
        for i, ratio in enumerate(ratios, 1):
            print(f"\n{'='*60}")
            print(f"  [{i}/3] Cascade duration: {ratio}")
            print(f"{'='*60}")
            compute_and_cache_duration(ratio)

        # 绘图
        plot_all_duration()

    print("\nAll done.")


if __name__ == "__main__":
    main()
