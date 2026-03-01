"""
generate_summary.py — Ratio Scan 结果文本摘要
读取 cache/ 下的 CSV 数据，生成结构化分析报告。

输出: results/summary.txt
"""

import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = _MODULE_DIR / "cache"
RESULTS_DIR = _MODULE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_VALUES = [4, 8]
Q_VALUES = [0.0, 0.15, 1.0]
N_HOUSEHOLDS = 49  # cascade 默认值，stability 从数据自动检测


def load_csv():
    """读取 cascade CSV，清洗无效行。"""
    csv_path = CACHE_DIR / "raw_results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # 确保数值列
    for col in ("alpha_star", "p_sync", "p_flow", "alpha_pas", "K", "q"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除 alpha_star 缺失的行
    df = df.dropna(subset=["alpha_star"])

    # 如果有 n_valid 列，删除 n_valid <= 0
    if "n_valid" in df.columns:
        df = df[df["n_valid"] > 0]

    return df.reset_index(drop=True)


def load_stability_agg():
    """读取 stability_agg.csv。"""
    csv_path = CACHE_DIR / "stability_agg.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    for col in ("kappa_c_mean", "kappa_c_std", "n_valid", "K", "q"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["kappa_c_mean"])
    return df.reset_index(drop=True)


def select_alpha_pas(df, target=1.0):
    """返回 target 对应的 alpha_pas 值；若不存在则回退到最大值。"""
    available = sorted(df["alpha_pas"].unique())
    for v in available:
        if np.isclose(v, target):
            return target
    fallback = max(available)
    return fallback


def section_header(title, char="="):
    return f"\n{char * 60}\n{title}\n{char * 60}\n"


def generate_summary():
    df_raw = load_csv()
    df_stab = load_stability_agg()

    out = StringIO()
    out.write(section_header("RATIO SCAN SUMMARY REPORT"))

    has_cascade = df_raw is not None and len(df_raw) > 0
    has_stability = df_stab is not None and len(df_stab) > 0

    # ── Section 0: Stability Summary ──────────────────────────────
    out.write(section_header("0. Stability Summary (kappa_c)"))

    if has_stability:
        stab_total = int(df_stab["ng"].iloc[0] + df_stab["nc"].iloc[0]
                         + df_stab["np"].iloc[0])
        out.write(f"节点总数: {stab_total}\n")
        out.write(f"数据条数: {len(df_stab)}\n")
        out.write(f"K 值: {sorted(df_stab['K'].unique())}\n")
        out.write(f"q 值: {sorted(df_stab['q'].unique())}\n")

        for K in sorted(df_stab["K"].unique()):
            for q in sorted(df_stab["q"].unique()):
                sub = df_stab[(df_stab["K"] == K) & (np.isclose(df_stab["q"], q))]
                if len(sub) == 0:
                    continue

                kc = sub["kappa_c_mean"]
                best_row = sub.loc[kc.idxmin()]
                worst_row = sub.loc[kc.idxmax()]

                out.write(f"\n  K={K}, q={q:g}  ({len(sub)} ratios)\n")
                out.write(f"    Mean kappa_c:  {kc.mean():.4f}\n")
                out.write(f"    Std kappa_c:   {kc.std():.4f}\n")
                out.write(f"    Min kappa_c:   {kc.min():.4f}  "
                          f"@ (ng={int(best_row['ng'])}, nc={int(best_row['nc'])}, "
                          f"np={int(best_row['np'])})\n")
                out.write(f"    Max kappa_c:   {kc.max():.4f}  "
                          f"@ (ng={int(worst_row['ng'])}, nc={int(worst_row['nc'])}, "
                          f"np={int(worst_row['np'])})\n")

        # Spearman 相关: kappa_c vs alpha* (如果两者都有)
        if has_cascade:
            out.write("\n  --- Spearman Correlation: kappa_c_mean vs alpha*_mean ---\n")

            alpha_pas_val = select_alpha_pas(df_raw)
            df_cascade = df_raw[np.isclose(df_raw["alpha_pas"], alpha_pas_val)].copy()

            # 按 (K, q, ng, nc, np) 聚合 cascade 数据
            cascade_agg = df_cascade.groupby(["K", "q", "ng", "nc", "np"]).agg(
                alpha_star_mean=("alpha_star", "mean"),
            ).reset_index()

            from scipy.stats import spearmanr

            # 按 (K, q) 分组计算相关
            for K in sorted(set(df_stab["K"].unique()) & set(cascade_agg["K"].unique())):
                for q in sorted(df_stab["q"].unique()):
                    stab_sub = df_stab[(df_stab["K"] == K) & np.isclose(df_stab["q"], q)]
                    casc_sub = cascade_agg[(cascade_agg["K"] == K) & np.isclose(cascade_agg["q"], q)]

                    if len(stab_sub) == 0 or len(casc_sub) == 0:
                        continue

                    merged = stab_sub.merge(casc_sub, on=["K", "q", "ng", "nc", "np"])
                    if len(merged) < 3:
                        continue

                    rho, pval = spearmanr(merged["kappa_c_mean"],
                                          merged["alpha_star_mean"])
                    out.write(f"    K={K}, q={q:g}: rho={rho:.3f}, "
                              f"p={pval:.4f} (n={len(merged)})\n")

            # 全局相关
            global_merged = df_stab.merge(cascade_agg,
                                          on=["K", "q", "ng", "nc", "np"])
            if len(global_merged) >= 3:
                rho, pval = spearmanr(global_merged["kappa_c_mean"],
                                      global_merged["alpha_star_mean"])
                out.write(f"    Global: rho={rho:.3f}, "
                          f"p={pval:.4f} (n={len(global_merged)})\n")
    else:
        out.write("  [无 stability 数据] 运行 python -m ratio_scan.run_stability_scan\n")

    # ── 以下为原有 cascade 报告 ───────────────────────────────────
    if not has_cascade:
        out.write(section_header("CASCADE DATA NOT FOUND"))
        out.write("  [无 cascade 数据] 运行 python -m ratio_scan.run_ratio_scan\n")
        report = out.getvalue()
        out_path = RESULTS_DIR / "summary.txt"
        out_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"\nSaved -> {out_path}")
        return

    alpha_pas_val = select_alpha_pas(df_raw)

    out.write(f"\n数据条数 (清洗后): {len(df_raw)}\n")
    out.write(f"K 值: {sorted(df_raw['K'].unique())}\n")
    out.write(f"q 值: {sorted(df_raw['q'].unique())}\n")
    out.write(f"alpha_pas 值: {sorted(df_raw['alpha_pas'].unique())}\n")
    out.write(f"选定 alpha_pas: {alpha_pas_val}")
    if not np.isclose(alpha_pas_val, 1.0):
        out.write(f"  (回退: 目标 1.0 不可用)")
    out.write("\n")

    # ── 1. 各 (K, q) 组合的 alpha* 统计 ─────────────────────────────
    out.write(section_header(f"1. alpha* Statistics per (K, q) — alpha_pas={alpha_pas_val}"))

    df1 = df_raw[np.isclose(df_raw["alpha_pas"], alpha_pas_val)].copy()
    df1["rg"] = df1["ng"] / N_HOUSEHOLDS
    df1["rc"] = df1["nc"] / N_HOUSEHOLDS
    df1["rp"] = df1["np"] / N_HOUSEHOLDS

    for K in sorted(df1["K"].unique()):
        for q in sorted(df1["q"].unique()):
            sub = df1[(df1["K"] == K) & (np.isclose(df1["q"], q))]
            if len(sub) == 0:
                continue

            alpha = sub["alpha_star"]
            best_row = sub.loc[alpha.idxmax()]
            worst_row = sub.loc[alpha.idxmin()]

            out.write(f"\n  K={K}, q={q:g}  ({len(sub)} ratios)\n")
            out.write(f"    Mean alpha*:  {alpha.mean():.4f}\n")
            out.write(f"    Std alpha*:   {alpha.std():.4f}\n")
            out.write(f"    Max alpha*:   {alpha.max():.4f}  "
                      f"@ (ng={int(best_row['ng'])}, nc={int(best_row['nc'])}, "
                      f"np={int(best_row['np'])}) "
                      f"[rg={best_row['rg']:.2f}, rc={best_row['rc']:.2f}, "
                      f"rp={best_row['rp']:.2f}]\n")
            out.write(f"    Min alpha*:   {alpha.min():.4f}  "
                      f"@ (ng={int(worst_row['ng'])}, nc={int(worst_row['nc'])}, "
                      f"np={int(worst_row['np'])}) "
                      f"[rg={worst_row['rg']:.2f}, rc={worst_row['rc']:.2f}, "
                      f"rp={worst_row['rp']:.2f}]\n")

    # ── 2. 跨拓扑比较 ─────────────────────────────────────────────
    out.write(section_header(f"2. Cross-Topology Comparisons — alpha_pas={alpha_pas_val}"))

    # K 比较
    for K in sorted(df1["K"].unique()):
        vals = df1[df1["K"] == K]["alpha_star"]
        out.write(f"  K={K}: mean alpha* = {vals.mean():.4f} (n={len(vals)})\n")

    if set(K_VALUES).issubset(df1["K"].unique()):
        k4_mean = df1[df1["K"] == 4]["alpha_star"].mean()
        k8_mean = df1[df1["K"] == 8]["alpha_star"].mean()
        out.write(f"  Delta(K=8 - K=4) = {k8_mean - k4_mean:+.4f}\n")

    out.write("\n")

    # q 比较
    for q in sorted(df1["q"].unique()):
        vals = df1[np.isclose(df1["q"], q)]["alpha_star"]
        out.write(f"  q={q:g}: mean alpha* = {vals.mean():.4f} (n={len(vals)})\n")

    # ── 3. Delta_alpha* 分析 (buffering) ────────────────────────────────────
    out.write(section_header("3. Delta_alpha* Analysis — Buffering Effect (alpha_pas=1.0 vs 0.1)"))

    has_1 = any(np.isclose(v, 1.0) for v in df_raw["alpha_pas"].unique())
    has_01 = any(np.isclose(v, 0.1) for v in df_raw["alpha_pas"].unique())

    if has_1 and has_01:
        df_hi = df_raw[np.isclose(df_raw["alpha_pas"], 1.0)].copy()
        df_lo = df_raw[np.isclose(df_raw["alpha_pas"], 0.1)].copy()

        for K in sorted(df_raw["K"].unique()):
            for q in sorted(df_raw["q"].unique()):
                hi = df_hi[(df_hi["K"] == K) & np.isclose(df_hi["q"], q)]
                lo = df_lo[(df_lo["K"] == K) & np.isclose(df_lo["q"], q)]

                if len(hi) == 0 or len(lo) == 0:
                    continue

                # 按 (ng, nc, np) 合并
                merge_cols = ["ng", "nc", "np"]
                merged = hi[merge_cols + ["alpha_star"]].merge(
                    lo[merge_cols + ["alpha_star"]],
                    on=merge_cols, suffixes=("_hi", "_lo"))

                if len(merged) == 0:
                    continue

                merged["delta_alpha"] = merged["alpha_star_hi"] - merged["alpha_star_lo"]

                out.write(f"\n  K={K}, q={q:g}:  ({len(merged)} matched ratios)\n")
                out.write(f"    mean Delta_alpha* = {merged['delta_alpha'].mean():.4f}\n")
                out.write(f"    std  Delta_alpha* = {merged['delta_alpha'].std():.4f}\n")

                # Top-5 ratios by delta_alpha
                top5 = merged.nlargest(5, "delta_alpha")
                out.write(f"    Top-5 ratios by Delta_alpha*:\n")
                for _, row in top5.iterrows():
                    out.write(f"      (ng={int(row['ng'])}, nc={int(row['nc'])}, "
                              f"np={int(row['np'])})  Delta_alpha*={row['delta_alpha']:.4f}\n")
    else:
        apas_list = sorted(df_raw["alpha_pas"].unique(), reverse=True)
        if len(apas_list) > 1 and has_1:
            out.write("  alpha_pas=0.1 不存在，使用一般性差异分析:\n")
            for K in sorted(df_raw["K"].unique()):
                for q in sorted(df_raw["q"].unique()):
                    base_sub = df_raw[(df_raw["K"] == K) & np.isclose(df_raw["q"], q)
                                      & np.isclose(df_raw["alpha_pas"], 1.0)]
                    if len(base_sub) == 0:
                        continue

                    out.write(f"\n  K={K}, q={q:g}:\n")
                    base_mean = base_sub["alpha_star"].mean()
                    out.write(f"    alpha_pas=1.0: mean alpha* = {base_mean:.4f}\n")

                    for ap in apas_list:
                        if np.isclose(ap, 1.0):
                            continue
                        comp = df_raw[(df_raw["K"] == K) & np.isclose(df_raw["q"], q)
                                      & np.isclose(df_raw["alpha_pas"], ap)]
                        if len(comp) == 0:
                            continue
                        comp_mean = comp["alpha_star"].mean()
                        delta = comp_mean - base_mean
                        out.write(f"    alpha_pas={ap}: mean alpha* = {comp_mean:.4f}  "
                                  f"(Delta = {delta:+.4f})\n")
        else:
            out.write("  仅有单一 alpha_pas 值，无法做差异分析。\n")

    # ── 4. 失效模式统计 ───────────────────────────────────────────
    out.write(section_header(f"4. Failure Mode Summary — alpha_pas={alpha_pas_val}"))

    for K in sorted(df1["K"].unique()):
        for q in sorted(df1["q"].unique()):
            sub = df1[(df1["K"] == K) & np.isclose(df1["q"], q)]
            if len(sub) == 0:
                continue

            n_total = len(sub)
            n_sync = (sub["p_sync"] > sub["p_flow"]).sum()
            n_flow = (sub["p_flow"] > sub["p_sync"]).sum()
            n_tie = (np.isclose(sub["p_sync"], sub["p_flow"])).sum()

            pct_sync = 100 * n_sync / n_total if n_total > 0 else 0

            out.write(f"\n  K={K}, q={q:g}:  "
                      f"sync-dominant={n_sync} ({pct_sync:.1f}%), "
                      f"flow-dominant={n_flow}, "
                      f"tie={n_tie}  (total={n_total})\n")

            mean_psync = sub["p_sync"].mean()
            mean_pflow = sub["p_flow"].mean()
            out.write(f"    mean p_sync={mean_psync:.3f}, "
                      f"mean p_flow={mean_pflow:.3f}\n")

    # ── 结束 ──────────────────────────────────────────────────────
    out.write(section_header("END OF REPORT", "-"))

    report = out.getvalue()
    out_path = RESULTS_DIR / "summary.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    generate_summary()
