"""
London Household Electricity Usage Pattern Analysis
12 months × 24 hours with P10–P90 range

Data: Small_LCL_Data.parquet (half-hourly kWh, ~5500 households)
Pipeline:
  1. Sum two half-hour readings → hourly kWh
  2. Per household × month × hour: mean hourly kWh
  3. Across households: mean (typical), P10/P90 (range)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Small_LCL_Data.parquet")

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load & clean ──────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_parquet(DATA_PATH, columns=["LCLid", "DateTime", "KWH/hh (per half hour)"])
df.rename(columns={"KWH/hh (per half hour)": "kwh", "LCLid": "hid"}, inplace=True)

n_before = len(df)
df.dropna(subset=["kwh"], inplace=True)
print(f"Dropped {n_before - len(df):,} null rows  ({len(df):,} remaining)")

df["month"] = df["DateTime"].dt.month
df["hour"] = df["DateTime"].dt.hour
df["date"] = df["DateTime"].dt.date

# ── 2. Half-hourly → hourly kWh (sum two readings per hour) ─────────────
print("Aggregating to hourly kWh …")
hourly_raw = df.groupby(["hid", "date", "month", "hour"])["kwh"].sum().reset_index()

# ── 3. Per household × month × hour: mean hourly kWh ────────────────────
print("Computing per-household monthly-hourly averages …")
hh_month_hour = hourly_raw.groupby(["hid", "month", "hour"])["kwh"].mean().reset_index()

# ── 4. Across households: mean, P10, P90 ────────────────────────────────
print("Computing cross-household statistics …")

def agg_stats(group):
    return pd.Series({
        "avg_kwh": group["kwh"].mean(),
        "p10_kwh": np.percentile(group["kwh"], 10),
        "p90_kwh": np.percentile(group["kwh"], 90),
    })

stats = hh_month_hour.groupby(["month", "hour"]).apply(agg_stats).reset_index()
stats["month_name"] = stats["month"].map(MONTH_NAMES)
stats = stats.sort_values(["month", "hour"]).reset_index(drop=True)

# ── 5. Save CSV ──────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "monthly_hourly_usage.csv")
stats[["month", "month_name", "hour", "avg_kwh", "p10_kwh", "p90_kwh"]].to_csv(csv_path, index=False)
print(f"Saved  {csv_path}")

# ── 6. Charts ────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
HOURS = list(range(24))

def plot_month(ax, month_data, title):
    """Draw 24h curve + P10-P90 shaded band on a given axes."""
    ax.plot(month_data["hour"], month_data["avg_kwh"],
            color="steelblue", linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(month_data["hour"], month_data["p10_kwh"], month_data["p90_kwh"],
                    color="steelblue", alpha=0.18)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylabel("kWh")
    ax.set_xlabel("Hour")

# 6a. 3×4 combined figure
print("Plotting 3×4 monthly profiles …")
fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
fig.suptitle("Monthly 24-Hour Electricity Usage Profiles (Mean ± P10–P90)",
             fontsize=14, fontweight="bold", y=0.98)

for idx, m in enumerate(range(1, 13)):
    ax = axes[idx // 4, idx % 4]
    md = stats[stats["month"] == m]
    plot_month(ax, md, MONTH_NAMES[m])

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUTPUT_DIR, "monthly_24h_profiles.png"), dpi=150)
plt.close(fig)
print("Saved  monthly_24h_profiles.png")

# 6b. Individual month figures
print("Plotting individual month figures …")
for m in range(1, 13):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    md = stats[stats["month"] == m]
    plot_month(ax, md, f"{MONTH_NAMES[m]} – 24-Hour Usage Profile")
    ax.set_xticks(range(24))
    ax.legend(["Mean", "P10–P90 range"], loc="upper left", fontsize=9)
    # manually add legend patch for shading
    from matplotlib.patches import Patch
    handles = [
        plt.Line2D([0], [0], color="steelblue", linewidth=1.8),
        Patch(facecolor="steelblue", alpha=0.18),
    ]
    ax.legend(handles, ["Mean", "P10–P90 range"], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"month_{m:02d}.png"), dpi=150)
    plt.close(fig)
print("Saved  month_01.png … month_12.png")

# 6c. Heatmap (hourly kWh)
print("Plotting heatmap …")
pivot = stats.pivot(index="month", columns="hour", values="avg_kwh")
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
    yticklabels=[MONTH_NAMES[m] for m in pivot.index],
    xticklabels=[f"{h:02d}" for h in pivot.columns],
    cbar_kws={"label": "Avg Hourly kWh"},
)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Month")
ax.set_title("London Households – Hourly Consumption Intensity (Month × Hour)")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "summary_heatmap.png"), dpi=150)
plt.close(fig)
print("Saved  summary_heatmap.png")

# ── 7. Markdown report ───────────────────────────────────────────────────
print("Writing markdown report …")
md_lines = []
md_lines.append("# Monthly 24-Hour Household Electricity Usage\n")
md_lines.append("## Data & Method\n")
md_lines.append("- **Source**: London Smart Meter dataset (`Small_LCL_Data.parquet`), ~5 500 households, 2011–2014")
md_lines.append("- **Granularity**: Half-hourly readings summed to hourly kWh")
md_lines.append("- **Aggregation**: Per household × month × hour mean → cross-household **mean**, **P10**, **P90**")
md_lines.append("- P10–P90 range captures the spread across households\n")

for m in range(1, 13):
    md = stats[stats["month"] == m]
    md_lines.append(f"## {MONTH_NAMES[m]}\n")
    md_lines.append("| Hour | Avg kWh | P10 kWh | P90 kWh |")
    md_lines.append("|-----:|--------:|--------:|--------:|")
    for _, row in md.iterrows():
        md_lines.append(f"| {int(row['hour']):2d} | {row['avg_kwh']:.4f} | {row['p10_kwh']:.4f} | {row['p90_kwh']:.4f} |")
    md_lines.append("")

# Key findings
peak_row = stats.loc[stats["avg_kwh"].idxmax()]
low_row = stats.loc[stats["avg_kwh"].idxmin()]
winter_avg = stats[stats["month"].isin([12, 1, 2])]["avg_kwh"].mean()
summer_avg = stats[stats["month"].isin([6, 7, 8])]["avg_kwh"].mean()

md_lines.append("## Key Findings\n")
md_lines.append(f"- **Peak consumption**: {MONTH_NAMES[int(peak_row['month'])]} at hour {int(peak_row['hour']):02d} "
                f"({peak_row['avg_kwh']:.4f} kWh)")
md_lines.append(f"- **Lowest consumption**: {MONTH_NAMES[int(low_row['month'])]} at hour {int(low_row['hour']):02d} "
                f"({low_row['avg_kwh']:.4f} kWh)")
md_lines.append(f"- **Winter avg** (Dec–Feb): {winter_avg:.4f} kWh/h")
md_lines.append(f"- **Summer avg** (Jun–Aug): {summer_avg:.4f} kWh/h")
md_lines.append(f"- Winter/summer ratio: {winter_avg / summer_avg:.2f}×")
md_lines.append("")

md_path = os.path.join(OUTPUT_DIR, "monthly_hourly_usage.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"Saved  {md_path}")

print("\nDone ✓")
