#!/usr/bin/env python3
"""
plot_diurnal_stability.py — 昼夜稳定性剖面可视化

图表 (夏冬对比):
  1. fig_diurnal_kappa_c.png     — κ_c(h) vs 小时, 红=夏/蓝=冬 + 置信带
  2. fig_stability_margin.png    — 稳定裕度 κ_op/κ_c(h)
  3. fig_hourly_kc_boxplot.png   — 逐小时箱线图, 夏冬并排
  4. fig_power_vs_kc.png         — 双轴: Σ|P_k| 与 κ_c
  5. fig_kappa_sweep_heatmap.png — (可选) κ × hour → r 热力图
  6. fig_diurnal_summary.png     — 2×2 组合发表级图

统计输出: outputs/diurnal_stats.json

用法:
  python timevarying_sync/plot_diurnal_stability.py
  python timevarying_sync/plot_diurnal_stability.py --summer-cache ... --winter-cache ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ── 项目路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    FIGURES_DIR, OUTPUT_DIR,
    MORNING_WINDOW, EVENING_WINDOW,
    DPI, FONT_FAMILY, KAPPA,
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

# 颜色方案 (与 plot_timevarying_sync.py 一致)
C_SUMMER = "#e74c3c"
C_WINTER = "#3498db"
C_MORNING = "#f39c12"
C_EVENING = "#8e44ad"


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_diurnal_cache(path):
    """加载 diurnal_kc_{season}.npz"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"缓存文件不存在: {p}")
    data = np.load(p, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    if "metadata_json" in result:
        result["metadata"] = json.loads(str(result["metadata_json"]))
    return result


def _add_peak_shading(ax):
    """添加早晚高峰时窗阴影。"""
    ax.axvspan(*MORNING_WINDOW, alpha=0.08, color=C_MORNING,
               label="Morning peak (6-10h)")
    ax.axvspan(*EVENING_WINDOW, alpha=0.08, color=C_EVENING,
               label="Evening peak (16-20h)")


# ═══════════════════════════════════════════════════════════════
# 图 1: κ_c(h) 曲线 + 置信带
# ═══════════════════════════════════════════════════════════════

def plot_kappa_c_curve(summer, winter):
    """κ_c(h) vs 小时, 红=夏/蓝=冬, 10-90% 置信带 + κ_op 参考线。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    hours = np.arange(24)

    for data, label, color in [
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]:
        kc = data["kappa_c"]  # (R, 24)
        kc_mean = np.mean(kc, axis=0)
        kc_p10 = np.percentile(kc, 10, axis=0)
        kc_p90 = np.percentile(kc, 90, axis=0)

        ax.fill_between(hours, kc_p10, kc_p90, alpha=0.2, color=color)
        ax.plot(hours, kc_mean, color=color, lw=2, marker="o", ms=4,
                label=f"{label} (mean)")

    # 操作 κ 参考线
    ax.axhline(KAPPA, ls="--", lw=1.5, color="gray",
               label=f"Operating $\\kappa = {KAPPA}$")

    _add_peak_shading(ax)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"Critical coupling $\kappa_c$")
    ax.set_title(r"Diurnal critical coupling $\kappa_c(h)$ — Summer vs Winter")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_diurnal_kappa_c.png")
    plt.close(fig)
    print("  -> fig_diurnal_kappa_c.png")


# ═══════════════════════════════════════════════════════════════
# 图 2: 稳定裕度 κ_op / κ_c(h)
# ═══════════════════════════════════════════════════════════════

def plot_stability_margin(summer, winter):
    """稳定裕度 = κ_op / κ_c(h)。 margin > 1 为安全。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    hours = np.arange(24)
    kappa_op = KAPPA

    for data, label, color in [
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]:
        kc = data["kappa_c"]  # (R, 24)
        margin = kappa_op / kc  # (R, 24)
        margin_mean = np.mean(margin, axis=0)
        margin_p10 = np.percentile(margin, 10, axis=0)
        margin_p90 = np.percentile(margin, 90, axis=0)

        ax.fill_between(hours, margin_p10, margin_p90, alpha=0.2, color=color)
        ax.plot(hours, margin_mean, color=color, lw=2, marker="s", ms=4,
                label=f"{label} (mean)")

    # margin = 1 临界线
    ax.axhline(1.0, ls="--", lw=1.5, color="black", label="Critical margin = 1")

    _add_peak_shading(ax)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"Stability margin $\kappa_{op} / \kappa_c(h)$")
    ax.set_title(f"Stability margin (operating $\\kappa = {kappa_op}$)")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_stability_margin.png")
    plt.close(fig)
    print("  -> fig_stability_margin.png")


# ═══════════════════════════════════════════════════════════════
# 图 3: 逐小时箱线图
# ═══════════════════════════════════════════════════════════════

def plot_hourly_boxplot(summer, winter):
    """逐小时 κ_c 箱线图, 夏冬并排。"""
    fig, ax = plt.subplots(figsize=(14, 5))

    hours = np.arange(24)
    width = 0.35

    # Summer
    kc_s = summer["kappa_c"]  # (R, 24)
    bp_s = ax.boxplot([kc_s[:, h] for h in range(24)],
                      positions=hours - width / 2, widths=width,
                      patch_artist=True, showfliers=False)
    for patch in bp_s["boxes"]:
        patch.set_facecolor(C_SUMMER)
        patch.set_alpha(0.6)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp_s[element]:
            line.set_color(C_SUMMER)

    # Winter
    kc_w = winter["kappa_c"]
    bp_w = ax.boxplot([kc_w[:, h] for h in range(24)],
                      positions=hours + width / 2, widths=width,
                      patch_artist=True, showfliers=False)
    for patch in bp_w["boxes"]:
        patch.set_facecolor(C_WINTER)
        patch.set_alpha(0.6)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp_w[element]:
            line.set_color(C_WINTER)

    # 操作 κ 参考线
    ax.axhline(KAPPA, ls="--", lw=1.5, color="gray",
               label=f"Operating $\\kappa = {KAPPA}$")

    _add_peak_shading(ax)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"$\kappa_c$")
    ax.set_title(r"Hourly $\kappa_c$ distribution — Summer (red) vs Winter (blue)")
    ax.set_xlim(-0.8, 23.8)
    ax.set_xticks(range(0, 24, 2))

    # 自定义图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_SUMMER, alpha=0.6, label="Summer"),
        Patch(facecolor=C_WINTER, alpha=0.6, label="Winter"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_hourly_kc_boxplot.png")
    plt.close(fig)
    print("  -> fig_hourly_kc_boxplot.png")


# ═══════════════════════════════════════════════════════════════
# 图 4: 功率 vs κ_c 双轴图
# ═══════════════════════════════════════════════════════════════

def plot_power_vs_kc(summer, winter):
    """双轴: Σ|P_k| (上) 与 κ_c (下), 展示功率-稳定性关联。"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    hours = np.arange(24)

    for data, label, color in [
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]:
        # 上图: 功率 L1 范数
        ax = axes[0]
        pl1 = data["power_l1"]  # (R, 24)
        pl1_mean = np.mean(pl1, axis=0)
        pl1_p10 = np.percentile(pl1, 10, axis=0)
        pl1_p90 = np.percentile(pl1, 90, axis=0)
        ax.fill_between(hours, pl1_p10, pl1_p90, alpha=0.15, color=color)
        ax.plot(hours, pl1_mean, color=color, lw=2, label=f"{label}")

        # 下图: κ_c
        ax = axes[1]
        kc = data["kappa_c"]
        kc_mean = np.mean(kc, axis=0)
        kc_p10 = np.percentile(kc, 10, axis=0)
        kc_p90 = np.percentile(kc, 90, axis=0)
        ax.fill_between(hours, kc_p10, kc_p90, alpha=0.15, color=color)
        ax.plot(hours, kc_mean, color=color, lw=2, label=f"{label}")

    # 上图装饰
    axes[0].set_ylabel(r"$\sum |P_k|$ (normalized)")
    axes[0].set_title("Total power injection vs Critical coupling")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)
    _add_peak_shading(axes[0])

    # 下图装饰
    axes[1].axhline(KAPPA, ls="--", lw=1.5, color="gray",
                    label=f"$\\kappa_{{op}} = {KAPPA}$")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel(r"$\kappa_c$")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].set_xlim(-0.5, 23.5)
    axes[1].set_xticks(range(0, 24, 2))
    _add_peak_shading(axes[1])

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_power_vs_kc.png")
    plt.close(fig)
    print("  -> fig_power_vs_kc.png")


# ═══════════════════════════════════════════════════════════════
# 图 5: (可选) κ 扫描热力图
# ═══════════════════════════════════════════════════════════════

def plot_kappa_sweep_heatmap(summer, winter):
    """(κ, hour) → mean r(t) 热力图, 夏冬并排。"""
    has_sweep_s = "sweep_r_mean" in summer
    has_sweep_w = "sweep_r_mean" in winter
    if not has_sweep_s and not has_sweep_w:
        print("  -- 无 κ 扫描数据, 跳过热力图")
        return

    n_cols = has_sweep_s + has_sweep_w
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    col = 0
    for data, label in [(summer, "Summer"), (winter, "Winter")]:
        if "sweep_r_mean" not in data:
            continue
        ax = axes[col]
        r_mean = data["sweep_r_mean"]  # (n_kappa, 24)
        kappa_vals = data["sweep_kappa_values"]

        im = ax.imshow(r_mean, aspect="auto", origin="lower",
                       extent=[-0.5, 23.5, kappa_vals[0], kappa_vals[-1]],
                       cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(r"$\kappa$")
        ax.set_title(f"Mean $r(\\kappa, h)$ — {label}")
        ax.set_xticks(range(0, 24, 4))

        # κ_c 叠加
        kc_mean = np.mean(data["kappa_c"], axis=0)
        ax.plot(np.arange(24), kc_mean, "k--", lw=1.5, label=r"$\kappa_c(h)$")
        ax.axhline(KAPPA, ls=":", lw=1, color="white",
                   label=f"$\\kappa_{{op}}={KAPPA}$")
        ax.legend(fontsize=8, loc="upper right")

        plt.colorbar(im, ax=ax, label=r"$\langle r \rangle$", shrink=0.85)
        col += 1

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_kappa_sweep_heatmap.png")
    plt.close(fig)
    print("  -> fig_kappa_sweep_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# 图 6: 2×2 组合发表级图
# ═══════════════════════════════════════════════════════════════

def plot_summary_2x2(summer, winter):
    """2×2 组合图: (a) κ_c 曲线 (b) 稳定裕度 (c) 功率 L1 (d) 箱线图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    hours = np.arange(24)

    # (a) κ_c 曲线
    ax = axes[0, 0]
    for data, label, color in [
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]:
        kc = data["kappa_c"]
        kc_mean = np.mean(kc, axis=0)
        kc_p10 = np.percentile(kc, 10, axis=0)
        kc_p90 = np.percentile(kc, 90, axis=0)
        ax.fill_between(hours, kc_p10, kc_p90, alpha=0.2, color=color)
        ax.plot(hours, kc_mean, color=color, lw=2, marker="o", ms=3, label=label)

    ax.axhline(KAPPA, ls="--", lw=1.2, color="gray",
               label=f"$\\kappa_{{op}}={KAPPA}$")
    _add_peak_shading(ax)
    ax.set_ylabel(r"$\kappa_c$")
    ax.set_title("(a) Critical coupling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 4))

    # (b) 稳定裕度
    ax = axes[0, 1]
    for data, label, color in [
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]:
        kc = data["kappa_c"]
        margin = KAPPA / kc
        margin_mean = np.mean(margin, axis=0)
        margin_p10 = np.percentile(margin, 10, axis=0)
        margin_p90 = np.percentile(margin, 90, axis=0)
        ax.fill_between(hours, margin_p10, margin_p90, alpha=0.2, color=color)
        ax.plot(hours, margin_mean, color=color, lw=2, marker="s", ms=3, label=label)

    ax.axhline(1.0, ls="--", lw=1.2, color="black", label="Critical = 1")
    _add_peak_shading(ax)
    ax.set_ylabel(r"$\kappa_{op}/\kappa_c$")
    ax.set_title("(b) Stability margin")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 4))

    # (c) 功率 L1 范数
    ax = axes[1, 0]
    for data, label, color in [
        (summer, "Summer", C_SUMMER),
        (winter, "Winter", C_WINTER),
    ]:
        pl1 = data["power_l1"]
        pl1_mean = np.mean(pl1, axis=0)
        pl1_p10 = np.percentile(pl1, 10, axis=0)
        pl1_p90 = np.percentile(pl1, 90, axis=0)
        ax.fill_between(hours, pl1_p10, pl1_p90, alpha=0.15, color=color)
        ax.plot(hours, pl1_mean, color=color, lw=2, label=label)

    _add_peak_shading(ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"$\sum |P_k|$")
    ax.set_title(r"(c) Total power injection $\sum|P_k|$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 4))

    # (d) κ_c 箱线图 (简化版: 只画均值 + 误差棒)
    ax = axes[1, 1]
    for data, label, color, offset in [
        (summer, "Summer", C_SUMMER, -0.15),
        (winter, "Winter", C_WINTER, 0.15),
    ]:
        kc = data["kappa_c"]
        kc_mean = np.mean(kc, axis=0)
        kc_std = np.std(kc, axis=0)
        ax.errorbar(hours + offset, kc_mean, yerr=kc_std,
                    fmt="o", ms=4, color=color, capsize=2, lw=1.2,
                    label=label)

    ax.axhline(KAPPA, ls="--", lw=1.2, color="gray",
               label=f"$\\kappa_{{op}}={KAPPA}$")
    _add_peak_shading(ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"$\kappa_c$")
    ax.set_title(r"(d) $\kappa_c$ mean $\pm$ std")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.8, 23.8)
    ax.set_xticks(range(0, 24, 4))

    fig.suptitle("Diurnal Stability Profile — Summary", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_diurnal_summary.png")
    plt.close(fig)
    print("  -> fig_diurnal_summary.png")


# ═══════════════════════════════════════════════════════════════
# 统计输出
# ═══════════════════════════════════════════════════════════════

def compute_and_save_stats(summer, winter):
    """计算统计量并保存到 diurnal_stats.json。"""
    hours = np.arange(24)
    stats_dict = {}

    for data, label in [(summer, "summer"), (winter, "winter")]:
        kc = data["kappa_c"]  # (R, 24)
        kc_mean = np.mean(kc, axis=0)
        pl1 = data["power_l1"]
        pl1_mean = np.mean(pl1, axis=0)

        # 峰谷时段
        h_max = int(np.argmax(kc_mean))
        h_min = int(np.argmin(kc_mean))

        # Pearson 相关: Σ|P| vs κ_c
        r_corr, p_corr = sp_stats.pearsonr(pl1_mean, kc_mean)

        # 早晚高峰 vs 全天 Wilcoxon
        morning_mask = (hours >= MORNING_WINDOW[0]) & (hours < MORNING_WINDOW[1])
        evening_mask = (hours >= EVENING_WINDOW[0]) & (hours < EVENING_WINDOW[1])
        off_peak_mask = ~morning_mask & ~evening_mask

        kc_peak = kc[:, morning_mask | evening_mask].ravel()
        kc_off = kc[:, off_peak_mask].ravel()

        try:
            w_stat, w_pval = sp_stats.mannwhitneyu(kc_peak, kc_off,
                                                    alternative="greater")
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        stats_dict[label] = {
            "kc_mean_all": round(float(np.mean(kc_mean)), 4),
            "kc_std_all": round(float(np.std(kc_mean)), 4),
            "kc_max_hour": h_max,
            "kc_max_value": round(float(kc_mean[h_max]), 4),
            "kc_min_hour": h_min,
            "kc_min_value": round(float(kc_mean[h_min]), 4),
            "pearson_power_vs_kc": {
                "r": round(float(r_corr), 4),
                "p": round(float(p_corr), 6),
            },
            "peak_vs_offpeak_mannwhitney": {
                "U_statistic": round(float(w_stat), 2) if np.isfinite(w_stat) else None,
                "p_value": round(float(w_pval), 6) if np.isfinite(w_pval) else None,
                "test_description": "Mann-Whitney U test: κ_c(peak hours) > κ_c(off-peak)",
            },
        }

    # 夏冬差异检验: 逐小时配对 t 检验
    kc_s_mean = np.mean(summer["kappa_c"], axis=0)  # (24,)
    kc_w_mean = np.mean(winter["kappa_c"], axis=0)
    t_stat, t_pval = sp_stats.ttest_rel(kc_s_mean, kc_w_mean)
    stats_dict["summer_vs_winter"] = {
        "paired_ttest": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(t_pval), 6),
            "test_description": "Paired t-test on hourly mean κ_c: summer vs winter",
        },
        "mean_diff": round(float(np.mean(kc_s_mean - kc_w_mean)), 4),
    }

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "diurnal_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"  -> diurnal_stats.json")
    return stats_dict


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="昼夜稳定性剖面可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_cache = Path(__file__).resolve().parent / "cache"
    p.add_argument("--summer-cache",
                   default=str(default_cache / "diurnal_kc_summer.npz"),
                   help="Summer κ_c 缓存路径")
    p.add_argument("--winter-cache",
                   default=str(default_cache / "diurnal_kc_winter.npz"),
                   help="Winter κ_c 缓存路径")
    return p.parse_args(argv)


def main():
    args = parse_args()

    print("=" * 60)
    print("昼夜稳定性剖面 — 图表生成")
    print("=" * 60)

    summer = load_diurnal_cache(args.summer_cache)
    winter = load_diurnal_cache(args.winter_cache)

    s_meta = summer.get("metadata", {})
    w_meta = winter.get("metadata", {})
    print(f"  Summer: {summer['kappa_c'].shape[0]} realizations, "
          f"24 hours")
    print(f"  Winter: {winter['kappa_c'].shape[0]} realizations, "
          f"24 hours")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] κ_c(h) 曲线")
    plot_kappa_c_curve(summer, winter)

    print("[2/7] 稳定裕度")
    plot_stability_margin(summer, winter)

    print("[3/7] 逐小时箱线图")
    plot_hourly_boxplot(summer, winter)

    print("[4/7] 功率 vs κ_c")
    plot_power_vs_kc(summer, winter)

    print("[5/7] κ 扫描热力图")
    plot_kappa_sweep_heatmap(summer, winter)

    print("[6/7] 2x2 组合图")
    plot_summary_2x2(summer, winter)

    print("[7/7] 统计输出")
    stats = compute_and_save_stats(summer, winter)

    # 打印关键统计
    print()
    for label in ["summer", "winter"]:
        s = stats[label]
        print(f"  [{label}] κ_c: mean={s['kc_mean_all']:.3f}, "
              f"peak hour={s['kc_max_hour']}h ({s['kc_max_value']:.3f}), "
              f"trough hour={s['kc_min_hour']}h ({s['kc_min_value']:.3f})")
        print(f"           Pearson(|P| vs κ_c): r={s['pearson_power_vs_kc']['r']:.3f}, "
              f"p={s['pearson_power_vs_kc']['p']:.4g}")

    sw = stats["summer_vs_winter"]
    print(f"  [summer vs winter] mean diff={sw['mean_diff']:.4f}, "
          f"paired t-test p={sw['paired_ttest']['p_value']:.4g}")

    print(f"\n所有图表已保存至: {FIGURES_DIR}")
    print(f"统计结果已保存至: {OUTPUT_DIR / 'diurnal_stats.json'}")


if __name__ == "__main__":
    main()
