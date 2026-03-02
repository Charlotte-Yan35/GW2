#!/usr/bin/env python3
"""
run_all.py — 一键运行时变同步实验 (§4.3.1)

直接点击运行即可，无需命令行参数。
依次执行: compute summer → compute winter → plot
"""

import matplotlib
matplotlib.use("Agg")

from compute_timevarying_sync import main as compute_main, parse_args as compute_parse
from plot_timevarying_sync import main as plot_main

# ═══════════════════════════════════════════════════════════════
# 配置 (在此修改参数，不需要命令行输入)
# ═══════════════════════════════════════════════════════════════
FREQ = "5min"            # 数据频率: "5min", "30min" 等
REALIZATIONS = 50        # Monte Carlo 次数
BASE_SEED = 42           # 随机种子 (可复现)
SYNTHETIC = True         # True=合成曲线, False=真实数据(需要 data/ 目录)
KAPPA = 5.0              # 耦合强度 κ
K = 4                    # WS 平均度
Q = 0.1                  # WS 重连概率


def run():
    # ── 第 1 步: 计算 Summer ──
    print("\n" + "█" * 60)
    print("  第 1/3 步: 计算 Summer")
    print("█" * 60)

    summer_args = ["--season", "summer",
                   "--freq", FREQ,
                   "--realizations", str(REALIZATIONS),
                   "--base-seed", str(BASE_SEED),
                   "--kappa", str(KAPPA),
                   "--K", str(K),
                   "--q", str(Q)]
    if SYNTHETIC:
        summer_args.append("--synthetic")

    import sys
    _backup = sys.argv
    sys.argv = ["compute_timevarying_sync.py"] + summer_args
    compute_main()

    # ── 第 2 步: 计算 Winter ──
    print("\n" + "█" * 60)
    print("  第 2/3 步: 计算 Winter")
    print("█" * 60)

    winter_args = ["--season", "winter",
                   "--freq", FREQ,
                   "--realizations", str(REALIZATIONS),
                   "--base-seed", str(BASE_SEED),
                   "--kappa", str(KAPPA),
                   "--K", str(K),
                   "--q", str(Q)]
    if SYNTHETIC:
        winter_args.append("--synthetic")

    sys.argv = ["compute_timevarying_sync.py"] + winter_args
    compute_main()

    # ── 第 3 步: 画图 ──
    print("\n" + "█" * 60)
    print("  第 3/3 步: 生成图表")
    print("█" * 60)

    sys.argv = ["plot_timevarying_sync.py"]
    plot_main()

    sys.argv = _backup

    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    run()
