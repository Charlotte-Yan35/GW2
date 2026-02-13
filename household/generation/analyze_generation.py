"""分析家庭光伏产电量的季节性和日内变化模式"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
OUT_DIR = Path(__file__).parent
SEASON_ORDER = ['Spring', 'Summer', 'Autumn', 'Winter']
SEASON_COLORS = {'Spring': '#2ecc71', 'Summer': '#e74c3c',
                 'Autumn': '#e67e22', 'Winter': '#3498db'}
MONTH_TO_SEASON = {3: 'Spring', 4: 'Spring', 5: 'Spring',
                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                   9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
                   12: 'Winter', 1: 'Winter', 2: 'Winter'}

# ── 数据读取与预处理 ──────────────────────────────────
df = pd.read_pickle(OUT_DIR / 'hourly_data.pkl')
df['P_GEN_AVG'] = (df['P_GEN_MIN'] + df['P_GEN_MAX']) / 2
df['hour'] = df['t_h'].astype(int)
df['month'] = df['d_m'].astype(int)
df['season'] = df['month'].map(MONTH_TO_SEASON)
df['season'] = pd.Categorical(df['season'], categories=SEASON_ORDER, ordered=True)

print(f"数据量: {len(df)} 条, 日期范围: {df['t_date'].min().date()} ~ {df['t_date'].max().date()}")
print(f"季节分布:\n{df['season'].value_counts().sort_index()}\n")

# ── 图1: 日内产电模式（按季节） ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for season in SEASON_ORDER:
    sub = df[df['season'] == season].groupby('hour').agg(
        avg=('P_GEN_AVG', 'mean'),
        lo=('P_GEN_MIN', 'mean'),
        hi=('P_GEN_MAX', 'mean'),
    )
    ax.plot(sub.index, sub['avg'], label=season, color=SEASON_COLORS[season], lw=2)
    ax.fill_between(sub.index, sub['lo'], sub['hi'],
                    color=SEASON_COLORS[season], alpha=0.15)
ax.set(xlabel='Hour of Day', ylabel='Generation (kW)',
       title='Diurnal Generation Pattern by Season', xticks=range(0, 24))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig1_diurnal_by_season.png')
print("✓ fig1_diurnal_by_season.png")

# ── 图2: 季节性产电量箱线图 ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
# 仅保留白天有产电的记录以获得更好的分布可视化
day_df = df[(df['hour'] >= 6) & (df['hour'] <= 20)]
sns.boxplot(data=day_df, x='season', y='P_GEN_AVG', hue='season',
            order=SEASON_ORDER, hue_order=SEASON_ORDER,
            palette=SEASON_COLORS, ax=ax, fliersize=1, legend=False)
ax.set(xlabel='Season', ylabel='Generation (kW)',
       title='Generation Distribution by Season (6:00–20:00)')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_seasonal_boxplot.png')
print("✓ fig2_seasonal_boxplot.png")

# ── 图3: 月度平均产电量柱状图 ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
monthly = df.groupby('month')['P_GEN_AVG'].mean()
monthly = monthly.reindex(range(1, 13))
colors = [SEASON_COLORS[MONTH_TO_SEASON[m]] for m in monthly.index]
ax.bar(monthly.index, monthly.values, color=colors, edgecolor='white', lw=0.5)
ax.set(xlabel='Month', ylabel='Avg Generation (kW)',
       title='Monthly Average Generation', xticks=range(1, 13))
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_monthly_bar.png')
print("✓ fig3_monthly_bar.png")

# ── 图4: 热力图 (小时 × 月份) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
pivot = df.pivot_table(values='P_GEN_AVG', index='hour', columns='month', aggfunc='mean')
pivot = pivot.reindex(columns=range(1, 13))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
            cbar_kws={'label': 'Avg Generation (kW)'})
ax.set(xlabel='Month', ylabel='Hour of Day',
       title='Generation Heatmap (Hour × Month)')
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig4_heatmap.png')
print("✓ fig4_heatmap.png")

# ── 表5: 24h 各季节产电区间表（P25 ~ P75） ─────────────
rows = []
for season in SEASON_ORDER:
    sub = df[df['season'] == season].groupby('hour')['P_GEN_AVG'].quantile([0.25, 0.75]).unstack()
    sub.columns = ['P25', 'P75']
    sub['season'] = season
    rows.append(sub)
range_df = pd.concat(rows).reset_index()
range_df.columns = ['hour', 'P25', 'P75', 'season']

# 负值截断为0（夜间微小负值无意义）
range_df['P25'] = range_df['P25'].clip(lower=0).round(3)
range_df['P75'] = range_df['P75'].clip(lower=0).round(3)

# 透视为宽表：每季节两列（P25, P75）
wide = range_df.pivot(index='hour', columns='season')
# 按季节顺序重排列
col_order = []
for s in SEASON_ORDER:
    col_order.append(('P25', s))
    col_order.append(('P75', s))
wide = wide[col_order]
# 扁平化列名
wide.columns = [f'{s}_{stat}' for stat, s in wide.columns]
wide.index.name = 'Hour'

# 保存CSV
csv_path = OUT_DIR / 'generation_range_by_season.csv'
wide.to_csv(csv_path)
print(f"✓ generation_range_by_season.csv")

# 打印表格
print("\n" + "=" * 90)
print("24h 各季节产电功率区间 (P25 ~ P75, kW)")
print("=" * 90)
header = f"{'Hour':>4}"
for s in SEASON_ORDER:
    header += f" | {s:^17}"
print(header)
print("-" * 90)
for h in range(24):
    row = f"{h:4d}"
    for s in SEASON_ORDER:
        p25 = wide.loc[h, f'{s}_P25']
        p75 = wide.loc[h, f'{s}_P75']
        row += f" | {p25:6.3f} ~ {p75:6.3f}"
    print(row)

print("\n全部图表已保存至:", OUT_DIR)
