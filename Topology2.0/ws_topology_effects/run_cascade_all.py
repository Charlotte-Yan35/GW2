"""
run_cascade_all.py — 运行三个 ratio 的 swing cascade bisection 计算
用法: MPLBACKEND=Agg python run_cascade_all.py
"""

from ws_cascade_compute import compute_and_cache_bisection

ratios = ["balanced", "gen_heavy", "load_heavy"]

for i, ratio in enumerate(ratios, 1):
    print(f"\n{'='*60}")
    print(f"  [{i}/3] {ratio}")
    print(f"{'='*60}")
    compute_and_cache_bisection(ratio)

print("\n✅ All three done.")
