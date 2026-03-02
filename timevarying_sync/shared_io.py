"""
shared_io.py — 从 reference_code 对齐的 data-driven 数据加载模块

功能：
1. locate_reference_data_config()  — 定位数据源路径并验证
2. load_datadriven_profile()       — 加载 LCL/PV 典型日曲线
3. build_node_injections()         — 构建 P_k(t) 注入矩阵

数据加载流程严格对齐 reference_code/scripts/powerreader.py 与
reference_code/scripts/powerclasses.py 中的方式：
  - LCL CSV: LCLid / DateTime / KWH/hh (per half hour)
  - PV  CSV: Substation / datetime / P_GEN_MAX / P_GEN_MIN
  - 按月过滤 → 按 day-of-week 分组 → 取均值 → 插值
  - PCC (最后一个节点) 平衡: Pvec[-1] = -sum(Pvec[:-1])
"""

import glob
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from config import (
    PROJECT_ROOT, LCL_DATA_DIR, LCL_PARQUET, PV_DATA_FILE,
    SEASON_TO_MONTH, N, N_HOUSEHOLDS, PCC_NODE, PV_PENETRATION,
    T_TOTAL,
    LCL_COL_ID, LCL_COL_DATETIME, LCL_COL_POWER,
    PV_COL_ID, PV_COL_DATETIME, PV_COL_GEN_MAX, PV_COL_GEN_MIN,
)


# ═══════════════════════════════════════════════════════════════
# 1. 数据定位与验证
# ═══════════════════════════════════════════════════════════════

def locate_reference_data_config():
    """定位并验证 reference_code 对齐的数据源路径。

    reference_code 原始路径映射：
      powerdata/data/*.csv                          → data/Small LCL Data/*.csv
      powerdata/PV/HourlyData/CustomerEndpoints.csv → data/PV Data/.../Customer Endpoints.csv

    Returns
    -------
    dict  包含 lcl_files, pv_file 等信息

    Raises
    ------
    FileNotFoundError  找不到必要数据时抛出，并提示所需文件
    """
    errors = []

    # --- LCL ---
    lcl_files = sorted(LCL_DATA_DIR.glob("*.csv")) if LCL_DATA_DIR.exists() else []
    has_parquet = LCL_PARQUET.exists()
    if not lcl_files and not has_parquet:
        errors.append(
            f"LCL 数据未找到:\n"
            f"  CSV 目录: {LCL_DATA_DIR}  (需含 LCL-June2015v2_*.csv)\n"
            f"  Parquet:  {LCL_PARQUET}\n"
            f"  对应 reference_code 'powerdata/data/*.csv'"
        )

    # --- PV ---
    if not PV_DATA_FILE.exists():
        errors.append(
            f"PV 数据未找到: {PV_DATA_FILE}\n"
            f"  对应 reference_code 'powerdata/PV/HourlyData/CustomerEndpoints.csv'"
        )

    if errors:
        raise FileNotFoundError(
            "数据定位失败:\n" + "\n".join(errors) + "\n"
            "请确保 data/ 目录中有完整的 LCL 和 PV 数据集。\n"
            "若无数据可用 --synthetic 标志切换为合成曲线模式。"
        )

    return {
        "lcl_dir": LCL_DATA_DIR,
        "lcl_files": lcl_files,
        "lcl_parquet": LCL_PARQUET if has_parquet else None,
        "pv_file": PV_DATA_FILE,
    }


# ═══════════════════════════════════════════════════════════════
# 2. 单条曲线构建 (对齐 powerreader.py)
# ═══════════════════════════════════════════════════════════════

def _strip_to_datevidandpower(raw_df):
    """对齐 powerreader.strip_to_datevidandpower"""
    df = raw_df.copy()
    # 兼容列名可能有/无尾部空格
    power_col = None
    for c in df.columns:
        if "KWH" in c.upper() or "kwh" in c.lower():
            power_col = c
            break
    if power_col is None:
        raise KeyError(f"LCL CSV 中未找到 KWH 列, 可用列: {list(df.columns)}")

    id_col = LCL_COL_ID if LCL_COL_ID in df.columns else df.columns[0]
    dt_col = LCL_COL_DATETIME if LCL_COL_DATETIME in df.columns else df.columns[2]

    df["date"] = pd.to_datetime(df[dt_col], errors="coerce")
    data = df[[id_col, power_col]].copy()
    data.index = df["date"]
    data.columns = ["LCLid", "power"]
    data["power"] = pd.to_numeric(data["power"], errors="coerce")
    return data


def _make_random_day_profile_lcl(month, rng, lcl_files):
    """构建 LCL 单户典型日负荷曲线。

    严格对齐 powerreader.make_random_week_profiles():
    1. 随机选取 CSV 文件 + 随机选取 household
    2. 按月过滤 → day-of-week 分组 → 同一 DOW 取均值
    3. 提取 Tuesday (DOW=1) 作为典型日
       (对齐 powerexperiments.py: tday_sample = t[48:96] → 86400s~171000s → 1970-01-02 = Tue)
    4. 返回 (secs_from_midnight, values) 半小时分辨率
    """
    file_idx = rng.integers(0, len(lcl_files))
    raw_df = pd.read_csv(lcl_files[file_idx], header=0)
    data = _strip_to_datevidandpower(raw_df)

    # 随机选 household
    house_ids = list(data["LCLid"].dropna().unique())
    if not house_ids:
        return None, None, False
    house = house_ids[rng.integers(0, len(house_ids))]
    selection = data.loc[data["LCLid"] == house].copy()
    selection.index = pd.to_datetime(selection.index, errors="coerce")
    selection = selection.dropna(subset=["power"])

    # 按月过滤
    month_data = selection[selection.index.month == month]
    if len(month_data) < 48:
        return None, None, False

    month_data = month_data.copy()
    month_data["dow"] = month_data.index.dayofweek
    month_data = month_data.sort_index()

    # 需要完整 7 天 (对齐 reference_code 的 check)
    if len(month_data["dow"].unique()) < 7:
        return None, None, False

    # 取 Tuesday (dow=1)，将日期统一映射到 1970-01-02
    # 对齐 powerexperiments.py: tday_sample = t[48:96] → 第 2 天 = Tuesday
    target_dow = 1
    dow_data = month_data[month_data["dow"] == target_dow].copy()
    if len(dow_data) < 10:
        return None, None, False

    new_idx = [t.replace(year=1970, month=1, day=2) for t in dow_data.index]
    dow_data.index = pd.DatetimeIndex(new_idx)

    # 按时间戳取均值 (对齐 reference_code 同逻辑)
    grouped = dow_data.groupby(dow_data.index)["power"].mean()

    # 转为秒轴 (当日 00:00 起的秒数)
    secs = np.array([(t.hour * 3600 + t.minute * 60 + t.second)
                     for t in grouped.index])
    values = grouped.values.astype(float)

    # 排序 + 去重
    order = np.argsort(secs)
    secs, values = secs[order], values[order]
    uniq_mask = np.concatenate([[True], np.diff(secs) > 0])
    secs, values = secs[uniq_mask], values[uniq_mask]

    if len(secs) < 10:
        return None, None, False

    # 缺失值处理: ffill (对齐 reference_code 中 interpolation 的隐含处理)
    nan_mask = np.isnan(values)
    if nan_mask.all():
        return None, None, False
    if nan_mask.any():
        good = ~nan_mask
        values = np.interp(secs, secs[good], values[good])

    return secs, values, True


def _make_random_day_profile_pv(month, rng, pv_file):
    """构建 PV 单户典型日发电曲线。

    严格对齐 powerreader.make_random_week_profiles_PV():
    1. 随机选取 Substation
    2. 按月过滤 → DOW 分组 → P_GEN = (P_GEN_MAX + P_GEN_MIN) / 2 取均值
    3. 提取 Tuesday (DOW=1)，与 LCL 一致
    """
    raw_df = pd.read_csv(pv_file, header=0)
    raw_df[PV_COL_DATETIME] = pd.to_datetime(raw_df[PV_COL_DATETIME], errors="coerce")
    for col in [PV_COL_GEN_MAX, PV_COL_GEN_MIN]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    # 随机选 PV panel
    pv_ids = list(raw_df[PV_COL_ID].dropna().unique())
    if not pv_ids:
        return None, None, False
    pv_id = pv_ids[rng.integers(0, len(pv_ids))]

    sel = raw_df[raw_df[PV_COL_ID] == pv_id].copy()
    sel.index = pd.to_datetime(sel[PV_COL_DATETIME], errors="coerce")
    sel = sel.dropna(subset=[PV_COL_GEN_MAX, PV_COL_GEN_MIN])

    month_data = sel[sel.index.month == month]
    if len(month_data) < 10:
        return None, None, False

    month_data = month_data.copy()
    month_data["dow"] = month_data.index.dayofweek
    if len(month_data["dow"].unique()) < 7:
        return None, None, False

    dow_data = month_data[month_data["dow"] == 1].copy()
    if len(dow_data) < 5:
        return None, None, False

    new_idx = [t.replace(year=1970, month=1, day=2) for t in dow_data.index]
    dow_data.index = pd.DatetimeIndex(new_idx)

    # P_GEN = (P_GEN_MAX + P_GEN_MIN) / 2  (对齐 reference_code)
    dow_data["P_GEN"] = (dow_data[PV_COL_GEN_MAX] + dow_data[PV_COL_GEN_MIN]) / 2.0
    grouped = dow_data.groupby(dow_data.index)["P_GEN"].mean()

    secs = np.array([(t.hour * 3600 + t.minute * 60 + t.second)
                     for t in grouped.index])
    values = grouped.values.astype(float)

    order = np.argsort(secs)
    secs, values = secs[order], values[order]
    uniq_mask = np.concatenate([[True], np.diff(secs) > 0])
    secs, values = secs[uniq_mask], values[uniq_mask]

    if len(secs) < 5:
        return None, None, False

    nan_mask = np.isnan(values)
    if nan_mask.all():
        return None, None, False
    if nan_mask.any():
        good = ~nan_mask
        values = np.interp(secs, secs[good], values[good])
    values = np.maximum(values, 0.0)   # 发电量不为负

    return secs, values, True


# ═══════════════════════════════════════════════════════════════
# 3. 合成曲线 (fallback)
# ═══════════════════════════════════════════════════════════════

def _synthetic_demand_24h(rng):
    """合成 UK 家庭日负荷曲线 (kW), 基于 household/node/household_microgrid.py 模式"""
    hours = np.linspace(0, 24, 49)    # 半小时分辨率
    base = 0.25 + 0.08 * rng.standard_normal()
    morning = 0.45 * np.exp(-((hours - 8.0) ** 2) / 3.0)
    evening = 0.65 * np.exp(-((hours - 19.0) ** 2) / 4.0)
    noise = 0.03 * np.abs(rng.standard_normal(len(hours)))
    profile = np.maximum(0.05, base + morning + evening + noise)
    secs = hours * 3600.0
    return secs, profile


def _synthetic_pv_24h(season, rng):
    """合成 PV 日发电曲线 (kW)"""
    hours = np.linspace(0, 24, 49)
    if season == "summer":
        peak = 1.8 + 0.5 * rng.standard_normal()
        width = 5.0
    else:
        peak = 0.5 + 0.2 * rng.standard_normal()
        width = 3.0
    cloud = 0.55 + 0.4 * rng.random()
    profile = np.maximum(0.0, peak * cloud *
                         np.exp(-((hours - 13.0) ** 2) / (2 * width ** 2)))
    secs = hours * 3600.0
    return secs, profile


# ═══════════════════════════════════════════════════════════════
# 4. 公开接口
# ═══════════════════════════════════════════════════════════════

def build_daily_profiles_for_realization(season, rng, use_synthetic=False):
    """为一次 realization 构建所有节点的 24h 负荷 / 发电曲线。

    Parameters
    ----------
    season : str  "summer" | "winter"
    rng : np.random.Generator
    use_synthetic : bool  强制使用合成曲线

    Returns
    -------
    dict  demand_profiles: list[(secs, vals)] × N_HOUSEHOLDS
          gen_profiles   : list[(secs, vals)] × PV_PENETRATION
    """
    if use_synthetic:
        return _build_synthetic(season, rng)

    try:
        config = locate_reference_data_config()
        return _build_datadriven(season, rng, config)
    except (FileNotFoundError, RuntimeError) as exc:
        warnings.warn(
            f"无法加载真实数据 ({exc}), 回退到合成曲线模式",
            RuntimeWarning, stacklevel=2,
        )
        return _build_synthetic(season, rng)


def _build_datadriven(season, rng, config):
    month = SEASON_TO_MONTH[season]
    lcl_files = config["lcl_files"]
    pv_file = config["pv_file"]
    max_retries = 300

    demand_profiles = []
    for i in range(N_HOUSEHOLDS):
        for _ in range(max_retries):
            secs, vals, ok = _make_random_day_profile_lcl(month, rng, lcl_files)
            if ok:
                demand_profiles.append((secs, vals))
                break
        else:
            raise RuntimeError(
                f"节点 {i}: 加载 LCL 负荷曲线失败 (season={season}, month={month}), "
                f"尝试 {max_retries} 次均无有效数据"
            )

    gen_profiles = []
    for i in range(PV_PENETRATION):
        for _ in range(max_retries):
            secs, vals, ok = _make_random_day_profile_pv(month, rng, pv_file)
            if ok:
                gen_profiles.append((secs, vals))
                break
        else:
            raise RuntimeError(
                f"节点 {i}: 加载 PV 发电曲线失败 (season={season}, month={month})"
            )

    return {"demand_profiles": demand_profiles, "gen_profiles": gen_profiles}


def _build_synthetic(season, rng):
    demand_profiles = [_synthetic_demand_24h(rng) for _ in range(N_HOUSEHOLDS)]
    gen_profiles = [_synthetic_pv_24h(season, rng) for _ in range(PV_PENETRATION)]
    return {"demand_profiles": demand_profiles, "gen_profiles": gen_profiles}


def build_node_injections(profiles, t_grid_sec, balance_at_pcc=True):
    """构建归一化 P_k(t) 注入矩阵。

    对齐 powerclasses.MicroGrid.get_power_vec():
      P[i](t) = gen_i(t) - demand_i(t)   (i = 0 .. N_HOUSEHOLDS-1)
      P[PCC](t) = -sum(P[0..N_HOUSEHOLDS-1])

    归一化: 使 max_t Σ|P_k(t)| = 2.0  (与静态 PMAX=1.0 情形一致)

    Parameters
    ----------
    profiles : dict  from build_daily_profiles_for_realization
    t_grid_sec : ndarray (T,)  秒
    balance_at_pcc : bool

    Returns
    -------
    P_matrix : ndarray (T, N)  归一化功率
    P_raw    : ndarray (T, N)  原始 kW (诊断用)
    """
    T = len(t_grid_sec)
    P_raw = np.zeros((T, N))

    demand_profiles = profiles["demand_profiles"]
    gen_profiles = profiles["gen_profiles"]

    for k in range(N_HOUSEHOLDS):
        # --- 负荷插值 (对齐 House.make_power_interpolator) ---
        d_s, d_v = demand_profiles[k]
        d_s, d_v = _ensure_24h_coverage(d_s, d_v)
        demand_f = interp1d(d_s, d_v, kind="linear", fill_value="extrapolate")
        demand_t = demand_f(t_grid_sec)

        # --- 发电插值 (对齐 House.make_generation_interpolator) ---
        if k < len(gen_profiles):
            g_s, g_v = gen_profiles[k]
            g_s, g_v = _ensure_24h_coverage(g_s, g_v)
            gen_f = interp1d(g_s, g_v, kind="linear", fill_value="extrapolate")
            gen_t = np.maximum(gen_f(t_grid_sec), 0.0)
        else:
            gen_t = np.zeros(T)

        # P = generation - demand  (对齐 House.get_house_power)
        P_raw[:, k] = gen_t - demand_t

    # PCC 平衡 (对齐 MicroGrid.get_power_vec: Pvec[-1] = -surplus)
    if balance_at_pcc:
        P_raw[:, PCC_NODE] = -np.sum(P_raw[:, :PCC_NODE], axis=1)

    # 归一化: max_t Σ|P_k| = 2.0 (与静态 gen=1, con=1 等效)
    l1_norms = np.sum(np.abs(P_raw), axis=1)
    max_l1 = np.max(l1_norms)
    if max_l1 > 1e-12:
        P_matrix = P_raw * (2.0 / max_l1)
    else:
        P_matrix = P_raw.copy()

    # 归一化后重新平衡 PCC
    if balance_at_pcc:
        P_matrix[:, PCC_NODE] = -np.sum(P_matrix[:, :PCC_NODE], axis=1)

    return P_matrix, P_raw


def _ensure_24h_coverage(secs, vals):
    """确保时间轴覆盖 [0, 86400] (对齐 interp1d 边界要求)"""
    secs = np.asarray(secs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if secs[0] > 0:
        secs = np.concatenate([[0.0], secs])
        vals = np.concatenate([[vals[0]], vals])
    if secs[-1] < T_TOTAL:
        secs = np.concatenate([secs, [T_TOTAL]])
        vals = np.concatenate([vals, [vals[-1]]])
    return secs, vals
