"""
run_all.py — Entry point: compute & plot for all ratio configurations.
"""

from ws_config import RATIO_CONFIGS
from ws_compute import compute_all_for_ratio
from ws_plots import (
    plot_kappa_c_map,
    plot_lorenz_curves,
    plot_gini_vs_q,
    plot_cascade_size_vs_q,
)


def main() -> None:
    for ratio_name in RATIO_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Ratio config: {ratio_name}")
        print(f"{'='*60}")

        # ── compute ──
        compute_all_for_ratio(ratio_name)

        # ── plot ──
        plot_kappa_c_map(ratio_name)
        plot_lorenz_curves(ratio_name)
        plot_gini_vs_q(ratio_name)
        plot_cascade_size_vs_q(ratio_name)

    print("\nAll done.")


if __name__ == "__main__":
    main()
