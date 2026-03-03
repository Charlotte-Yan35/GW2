"""
shared_diurnal_utils.py — 数据加载与注入向量构建。

功能:
  1. 加载 PV 发电 24h 均值曲线 (summer / winter)
  2. 加载 LCL 家庭用电数据 → 各户各小时平均用电 (带缓存)
  3. 构建注入向量 P(h)：复刻 reference_code 的 House.get_house_power + MicroGrid.get_power_vec 逻辑

符号约定 (同 reference_code):
  - 发电 → 正 (+)
  - 消费 → 负 (-)
  - PCC 吸收/补偿剩余功率: P[0] = -sum(P[1:])
"""

import pickle
import numpy as np
import pandas as pd

from config import (
    PARQUET_PATH, PV_CSV_PATH,
    CACHE_DIR, SEASONS,
    N, PCC_NODE, N_HOUSEHOLDS, PENETRATION,
)


# ====================================================================
# 1. PV 发电加载
# ====================================================================

def load_pv_generation() -> dict[str, np.ndarray]:
    """从 generation_range_by_season.csv 加载 24h PV 均值曲线。

    Returns
    -------
    dict  {"summer": ndarray(24,), "winter": ndarray(24,)}  单位: kW
    """
    df = pd.read_csv(PV_CSV_PATH)
    return {
        "summer": df["Summer_Mean"].values.astype(float),
        "winter": df["Winter_Mean"].values.astype(float),
    }


# ====================================================================
# 2. 家庭用电加载 (从 parquet，带缓存)
# ====================================================================

def load_household_demand_profiles() -> dict[str, pd.DataFrame]:
    """加载 LCL 数据，计算各户各季各小时平均用电。

    缓存到 cache/household_demand_{season}.pkl

    Returns
    -------
    dict  {"summer": DataFrame[hid, hour, demand_kw],
           "winter": DataFrame[hid, hour, demand_kw]}
    """
    result = {}
    all_cached = True

    for season, months in SEASONS.items():
        cache_path = CACHE_DIR / f"household_demand_{season}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                result[season] = pickle.load(f)
            print(f"  [cache hit] {cache_path.name}")
        else:
            all_cached = False

    if all_cached:
        return result

    # 需要从 parquet 加载原始数据
    print("  Loading LCL parquet (this may take a moment)...")
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=["LCLid", "DateTime", "KWH/hh (per half hour)"],
    )
    df.rename(columns={"KWH/hh (per half hour)": "kwh", "LCLid": "hid"}, inplace=True)
    df.dropna(subset=["kwh"], inplace=True)
    df["month"] = df["DateTime"].dt.month
    df["hour"] = df["DateTime"].dt.hour
    df["date"] = df["DateTime"].dt.date

    for season, months in SEASONS.items():
        cache_path = CACHE_DIR / f"household_demand_{season}.pkl"
        if cache_path.exists():
            continue

        print(f"  Processing {season} (months={months})...")
        # 按月筛选
        mask = df["month"].isin(months)
        df_season = df.loc[mask]

        # 半小时 → 小时: groupby(hid, date, hour).sum()
        hourly = (
            df_season
            .groupby(["hid", "date", "hour"])["kwh"]
            .sum()
            .reset_index()
        )

        # 各户各小时均值: groupby(hid, hour).mean()
        hh_hour = (
            hourly
            .groupby(["hid", "hour"])["kwh"]
            .mean()
            .reset_index()
            .rename(columns={"kwh": "demand_kw"})
        )

        result[season] = hh_hour
        with open(cache_path, "wb") as f:
            pickle.dump(hh_hour, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved {cache_path.name} ({hh_hour['hid'].nunique()} households × 24h)")

    return result


def preprocess_demand_pools(
    profiles: dict[str, pd.DataFrame],
) -> dict[str, dict[int, np.ndarray]]:
    """转为紧凑格式 {season: {hour: ndarray(n_households,)}}，方便多进程采样。"""
    pools = {}
    for season, df in profiles.items():
        hour_dict = {}
        for h in range(24):
            vals = df.loc[df["hour"] == h, "demand_kw"].values
            hour_dict[h] = vals.astype(np.float64)
        pools[season] = hour_dict
    return pools


# ====================================================================
# 3. 注入向量构建
# ====================================================================

def build_injection_vector(
    hour: int,
    season: str,
    pv_profile: np.ndarray,
    demand_pool: dict[int, np.ndarray],
    penetration: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """构建 N 节点功率注入向量 P(h)。

    算法 (严格遵循 reference_code 的 House.get_house_power + MicroGrid.get_power_vec):
      1. 从 demand_pool[hour] 中随机采样 N_HOUSEHOLDS 户用电量
      2. 前 penetration 户有 PV: P[k+1] = pv[hour] - demand[k]
      3. 其余户无 PV: P[k+1] = -demand[k]
      4. PCC 平衡: P[0] = -sum(P[1:])
      5. 不归一化: 直接使用原始 kW 值

    Parameters
    ----------
    hour : int          0-23
    season : str        "summer" or "winter"
    pv_profile : ndarray  (24,) kW
    demand_pool : dict    {hour: ndarray(n_households_in_pool,)}
    penetration : int     有 PV 的家庭数
    rng : np.random.Generator

    Returns
    -------
    P : ndarray (N,)  单位: kW
    """
    pool = demand_pool[hour]
    n_pool = len(pool)

    # 随机采样 N_HOUSEHOLDS 户
    indices = rng.integers(0, n_pool, size=N_HOUSEHOLDS)
    demands = pool[indices]  # (N_HOUSEHOLDS,) kW

    P = np.zeros(N)
    pv_gen = pv_profile[hour]

    # 前 penetration 户有 PV
    for k in range(penetration):
        P[k + 1] = pv_gen - demands[k]  # 发电 - 消费 = 净注入

    # 其余户无 PV (纯消费)
    for k in range(penetration, N_HOUSEHOLDS):
        P[k + 1] = -demands[k]

    # PCC 平衡
    P[PCC_NODE] = -np.sum(P[1:])

    return P


# ====================================================================
# 验证
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Validating shared_diurnal_utils.py")
    print("=" * 60)

    # 1. PV 数据
    print("\n--- PV Generation ---")
    pv = load_pv_generation()
    for s in ["summer", "winter"]:
        peak_h = np.argmax(pv[s])
        print(f"  {s}: peak = {pv[s][peak_h]:.3f} kW at hour {peak_h}")
    assert abs(pv["summer"][13] - 1.347) < 0.01, f"Summer h13 expected ~1.347, got {pv['summer'][13]}"
    assert abs(pv["winter"][13] - 0.662) < 0.01, f"Winter h13 expected ~0.662, got {pv['winter'][13]}"
    print("  PV validation passed!")

    # 2. LCL 数据
    print("\n--- Household Demand ---")
    profiles = load_household_demand_profiles()
    pools = preprocess_demand_pools(profiles)
    for s in ["summer", "winter"]:
        n_hh = profiles[s]["hid"].nunique()
        print(f"  {s}: {n_hh} households × 24 hours")
        assert n_hh > 5000, f"Expected >5000 households, got {n_hh}"
    print("  LCL validation passed!")

    # 3. 注入向量
    print("\n--- Injection Vector ---")
    rng = np.random.default_rng(42)
    for s in ["summer", "winter"]:
        P = build_injection_vector(13, s, pv[s], pools[s], PENETRATION, rng)
        balance = np.abs(np.sum(P))
        n_pos = np.sum(P[1:] > 0)
        n_neg = np.sum(P[1:] < 0)
        print(f"  {s} h=13: sum(P)={balance:.2e}, "
              f"P[0]={P[0]:.3f} kW, "
              f"pos={n_pos}, neg={n_neg}, "
              f"max={P.max():.3f}, min={P.min():.3f}")
        assert balance < 1e-10, f"Power balance violated: sum(P) = {balance}"
    print("  Injection vector validation passed!")

    print("\nAll validations passed!")
