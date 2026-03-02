#!/usr/bin/env python3
"""
run_diurnal.py — 一键运行昼夜稳定性剖面实验

直接点击运行即可，无需命令行参数。
依次执行: compute summer κ_c → compute winter κ_c → 生成图表
"""

import matplotlib
matplotlib.use("Agg")

from compute_diurnal_stability import main as compute_main
from plot_diurnal_stability import main as plot_main

# ═══════════════════════════════════════════════════════════════
# 配置 (在此修改参数)
# ═══════════════════════════════════════════════════════════════
REALIZATIONS = 30        # Monte Carlo 次数
BASE_SEED = 42           # 随机种子
SYNTHETIC = False        # True=合成曲线, False=真实数据(LCL+PV)
K = 4                    # WS 平均度
Q = 0.1                  # WS 重连概率
WORKERS = 0              # 0=使用全部CPU核心, 1=串行
SWEEP = False            # 是否运行 κ 扫描热力图 (耗时较长)


def run():
    import sys
    _backup = sys.argv

    # ── 第 1 步: 计算 Summer κ_c ──
    print("\n" + "█" * 60)
    print("  第 1/3 步: 计算 Summer κ_c(h)")
    print("█" * 60)

    args = ["--season", "summer",
            "--realizations", str(REALIZATIONS),
            "--base-seed", str(BASE_SEED),
            "--K", str(K),
            "--q", str(Q),
            "--workers", str(WORKERS)]
    if SYNTHETIC:
        args.append("--synthetic")
    if SWEEP:
        args.append("--sweep")

    sys.argv = ["compute_diurnal_stability.py"] + args
    compute_main()

    # ── 第 2 步: 计算 Winter κ_c ──
    print("\n" + "█" * 60)
    print("  第 2/3 步: 计算 Winter κ_c(h)")
    print("█" * 60)

    args[1] = "winter"
    sys.argv = ["compute_diurnal_stability.py"] + args
    compute_main()

    # ── 第 3 步: 生成图表 ──
    print("\n" + "█" * 60)
    print("  第 3/3 步: 生成昼夜稳定性图表")
    print("█" * 60)

    sys.argv = ["plot_diurnal_stability.py"]
    plot_main()

    sys.argv = _backup

    print("\n" + "=" * 60)
    print("  全部完成!")
    print("  图表: timevarying_sync/outputs/figures/")
    print("  统计: timevarying_sync/outputs/diurnal_stats.json")
    print("=" * 60)


if __name__ == "__main__":
    run()
