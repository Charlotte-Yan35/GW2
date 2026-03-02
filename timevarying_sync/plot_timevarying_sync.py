#!/usr/bin/env python3
"""
plot_timevarying_sync.py — 时变同步结果可视化 (只画图/统计，不计算)

报告 §4.3.1: "时变 P_k(t) 对同步的影响（基于 data-driven 的设定）"

用法:
  python timevarying_sync/plot_timevarying_sync.py \\
      --summer-cache timevarying_sync/cache/results_summer.npz \\
      --winter-cache timevarying_sync/cache/results_winter.npz
  python timevarying_sync/plot_timevarying_sync.py --help
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── 项目路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    FIGURES_DIR, OUTPUT_DIR,
    MORNING_WINDOW, EVENING_WINDOW,
    DPI, FONT_FAMILY,
    R_COLLAPSE_THRESHOLD,
)

# ── 全局绘图设置 ──
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

# 颜色方案
C_SUMMER = "#e74c3c"
C_WINTER = "#3498db"
C_MORNING = "#f39c12"
C_EVENING = "#8e44ad"


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_cache(path):
    """加载 .npz 缓存文件，返回 dict。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"缓存文件不存在: {p}")
    data = np.load(p, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    if "metadata_json" in result:
        result["metadata"] = json.loads(str(result["metadata_json"]))
    return result


# ═══════════════════════════════════════════════════════════════
# 图 1: 24h 同步序参量演化
# ═══════════════════════════════════════════════════════════════

def _plot_r_time_single(ax, t_grid, r_ensemble, color, label):
    """在 ax 上画 r(t) 均值曲线 + 10-90 分位置信带。"""
    hours = t_grid / 3600.0
    r_mean = np.nanmean(r_ensemble, axis=0)
    r_p10 = np.nanpercentile(r_ensemble, 10, axis=0)
    r_p90 = np.nanpercentile(r_ensemble, 90, axis=0)

    ax.fill_between(hours, r_p10, r_p90, alpha=0.2, color=color)
    ax.plot(hours, r_mean, color=color, lw=1.8, label=label)


def _add_risk_shading(ax):
    """添加高风险时窗阴影。"""
    ax.axvspan(*MORNING_WINDOW, alpha=0.08, color=C_MORNING, label="Morning window")
    ax.axvspan(*EVENING_WINDOW, alpha=0.08, color=C_EVENING, label="Evening window")


def _finalize_r_ax(ax, title):
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"$r(t) = |\langle e^{i\theta}\rangle|$")
    ax.set_title(title)
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(0, 25, 4))
    ax.axhline(R_COLLAPSE_THRESHOLD, ls="--", lw=0.8, color="gray",
               label=f"$r_{{collapse}}={R_COLLAPSE_THRESHOLD}$")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)


def plot_r_time(summer, winter):
    """生成 fig_r_time_summer / winter / overlay。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for data, season, color in [
        (summer, "summer", C_SUMMER),
        (winter, "winter", C_WINTER),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _plot_r_time_single(ax, data["t_grid"], data["r_ensemble"], color, season)
        _add_risk_shading(ax)
        _finalize_r_ax(ax, f"Synchronization order parameter — {season}")
        fig.savefig(FIGURES_DIR / f"fig_r_time_{season}.png")
        plt.close(fig)
        print(f"  ✓ fig_r_time_{season}.png")

    # overlay
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot_r_time_single(ax, summer["t_grid"], summer["r_ensemble"],
                        C_SUMMER, "Summer")
    _plot_r_time_single(ax, winter["t_grid"], winter["r_ensemble"],
                        C_WINTER, "Winter")
    _add_risk_shading(ax)
    _finalize_r_ax(ax, "Synchronization order parameter — Summer vs Winter")
    fig.savefig(FIGURES_DIR / "fig_r_time_overlay.png")
    plt.close(fig)
    print("  ✓ fig_r_time_overlay.png")


# ═══════════════════════════════════════════════════════════════
# 图 2: 崩溃时刻分布直方图
# ═══════════════════════════════════════════════════════════════

def plot_collapse_hist(summer, winter):
    """t_collapse 的小时值直方图。"""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bins = np.linspace(0, 24, 25)
    for data, label, color in [
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]:
        tc_h = data["t_collapse"] / 3600.0
        valid = tc_h[~np.isnan(tc_h)]
        if len(valid) > 0:
            ax.hist(valid, bins=bins, alpha=0.5, color=color, label=label,
                    edgecolor="white", lw=0.5)

    _add_risk_shading(ax)
    ax.set_xlabel("Collapse hour")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of collapse times")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_collapse_hist.png")
    plt.close(fig)
    print("  ✓ fig_collapse_hist.png")


# ═══════════════════════════════════════════════════════════════
# 图 3: 冬季傍晚高风险显著性检验
# ═══════════════════════════════════════════════════════════════

def plot_evening_significance(summer, winter):
    """比较冬季/夏季崩溃落入 evening_window 的比例，Fisher exact test。

    选择 Fisher exact test 的理由:
      - 比较的是两个独立二项比例 (崩溃是否在窗口内)
      - 样本可能较小 (<100), Fisher 精确检验不依赖大样本近似
      - 比 two-proportion z-test 更稳健
    """
    ev_lo, ev_hi = EVENING_WINDOW

    stats_dict = {}
    table_2x2 = np.zeros((2, 2), dtype=int)  # rows=season, cols=in_window/out

    for i, (data, label) in enumerate([
        (summer, "summer"),
        (winter, "winter"),
    ]):
        tc_h = data["t_collapse"] / 3600.0
        valid = tc_h[~np.isnan(tc_h)]
        n_total = len(valid)
        n_in_window = int(np.sum((valid >= ev_lo) & (valid <= ev_hi)))
        n_out = n_total - n_in_window
        table_2x2[i, 0] = n_in_window
        table_2x2[i, 1] = n_out
        ratio = n_in_window / n_total if n_total > 0 else 0.0
        stats_dict[label] = {
            "n_collapse_total": n_total,
            "n_in_evening": n_in_window,
            "ratio_in_evening": round(ratio, 4),
        }

    # Fisher exact test
    if table_2x2.sum() > 0:
        odds_ratio, p_value = stats.fisher_exact(table_2x2)
    else:
        odds_ratio, p_value = np.nan, np.nan
    stats_dict["fisher_exact"] = {
        "odds_ratio": round(float(odds_ratio), 4) if np.isfinite(odds_ratio) else None,
        "p_value": round(float(p_value), 6) if np.isfinite(p_value) else None,
        "contingency_table": table_2x2.tolist(),
        "test_description": "Fisher exact test comparing proportion of collapses "
                            "within evening window (16-20h) between summer and winter",
    }

    # 保存 stats.json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"  ✓ stats.json  (p={p_value:.4g})")

    # 绘图: 文字面板 + 条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    # 条形图
    seasons = ["Summer", "Winter"]
    ratios = [stats_dict["summer"]["ratio_in_evening"],
              stats_dict["winter"]["ratio_in_evening"]]
    bars = ax1.bar(seasons, ratios, color=[C_SUMMER, C_WINTER],
                   edgecolor="white", width=0.5)
    ax1.set_ylabel("Fraction of collapses\nin evening window")
    ax1.set_title(f"Evening ({ev_lo}:00–{ev_hi}:00) collapse ratio")
    ax1.set_ylim(0, max(ratios) * 1.4 + 0.05)
    for bar, r in zip(bars, ratios):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{r:.2%}", ha="center", va="bottom", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)

    # 文字面板
    ax2.axis("off")
    txt_lines = [
        "Fisher Exact Test",
        "-" * 30,
        "H0: evening collapse ratio",
        "    same across seasons",
        "",
        f"Summer: {stats_dict['summer']['n_in_evening']}"
        f"/{stats_dict['summer']['n_collapse_total']} in window",
        f"Winter: {stats_dict['winter']['n_in_evening']}"
        f"/{stats_dict['winter']['n_collapse_total']} in window",
        "",
        f"Odds ratio = {odds_ratio:.3f}" if np.isfinite(odds_ratio) else "Odds ratio = N/A",
        f"p-value = {p_value:.4g}" if np.isfinite(p_value) else "p-value = N/A",
        "",
        "Significant (p<0.05)" if (np.isfinite(p_value) and p_value < 0.05) else "Not significant",
    ]
    ax2.text(0.1, 0.95, "\n".join(txt_lines), transform=ax2.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_evening_significance.png")
    plt.close(fig)
    print("  ✓ fig_evening_significance.png")


# ═══════════════════════════════════════════════════════════════
# 图 4: 崩溃前兆指标可观测性
# ═══════════════════════════════════════════════════════════════

def plot_early_warning(summer, winter):
    """Var[r] 和 AC1[r] 随时间的 ensemble 曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for col, (data, label, color) in enumerate([
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]):
        hours = data["t_grid"] / 3600.0

        # Var[r]
        ax = axes[0, col]
        var_mean = np.nanmean(data["var_r"], axis=0)
        var_p10 = np.nanpercentile(data["var_r"], 10, axis=0)
        var_p90 = np.nanpercentile(data["var_r"], 90, axis=0)
        ax.fill_between(hours, var_p10, var_p90, alpha=0.2, color=color)
        ax.plot(hours, var_mean, color=color, lw=1.5)
        ax.set_ylabel(r"$\mathrm{Var}[r]$")
        ax.set_title(f"Sliding-window variance — {label}")
        _add_risk_shading(ax)
        ax.grid(alpha=0.25)

        # AC1[r]
        ax = axes[1, col]
        ac_mean = np.nanmean(data["ac1_r"], axis=0)
        ac_p10 = np.nanpercentile(data["ac1_r"], 10, axis=0)
        ac_p90 = np.nanpercentile(data["ac1_r"], 90, axis=0)
        ax.fill_between(hours, ac_p10, ac_p90, alpha=0.2, color=color)
        ax.plot(hours, ac_mean, color=color, lw=1.5)
        ax.set_ylabel(r"$\mathrm{AC}_1[r]$")
        ax.set_title(f"Sliding-window lag-1 autocorrelation — {label}")
        ax.set_xlabel("Hour of day")
        _add_risk_shading(ax)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))

    fig.suptitle("Early warning indicators before collapse", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_early_warning.png")
    plt.close(fig)
    print("  ✓ fig_early_warning.png")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="时变同步结果可视化 (§4.3.1) — 只画图/统计，不计算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_cache = Path(__file__).resolve().parent / "cache"
    p.add_argument("--summer-cache",
                   default=str(default_cache / "results_summer.npz"),
                   help="Summer 结果缓存路径")
    p.add_argument("--winter-cache",
                   default=str(default_cache / "results_winter.npz"),
                   help="Winter 结果缓存路径")
    return p.parse_args(argv)


def main():
    args = parse_args()

    print("=" * 50)
    print("时变 P_k(t) 同步 — 图表生成")
    print("=" * 50)

    summer = load_cache(args.summer_cache)
    winter = load_cache(args.winter_cache)

    s_meta = summer.get("metadata", {})
    w_meta = winter.get("metadata", {})
    print(f"  Summer: {summer['r_ensemble'].shape[0]} realizations, "
          f"{summer['r_ensemble'].shape[1]} time points")
    print(f"  Winter: {winter['r_ensemble'].shape[0]} realizations, "
          f"{winter['r_ensemble'].shape[1]} time points")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 24h 同步序参量演化")
    plot_r_time(summer, winter)

    print("[2/4] 崩溃时刻分布直方图")
    plot_collapse_hist(summer, winter)

    print("[3/4] 冬季傍晚高风险显著性检验")
    plot_evening_significance(summer, winter)

    print("[4/4] 崩溃前兆指标")
    plot_early_warning(summer, winter)

    print(f"\n所有图表已保存至: {FIGURES_DIR}")
    print(f"统计结果已保存至: {OUTPUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
