"""
复现 Reference3 Figure 4 — Panels D, E, F, G

从真实数据 (LCL 用电 + PV 发电 CSV) 自主生成微电网轨迹，
不依赖老师预计算的 pickle 文件。

Panel D: Winter (month=1), 50% PV uptake  (penetration=24)
Panel E: Winter (month=1), 100% PV uptake (penetration=49)
Panel F: Summer (month=7), 50% PV uptake  (penetration=24)
Panel G: Summer (month=7), 100% PV uptake (penetration=49)

每个 panel: 50 个微电网 ensemble 的周轨迹投影到三角 simplex 上
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import glob as globmod
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
LCL_DIR = DATA_ROOT / "Small LCL Data"
PV_FILE = (DATA_ROOT / "PV Data" / "2014-11-28 Cleansed and Processed"
           / "EXPORT HourlyData" / "EXPORT HourlyData - Customer Endpoints.csv")
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================
# 物理参数
# ============================================================
N_NODES = 50          # 微电网节点数
N_ENSEMBLE = 50       # ensemble 成员数
H_TRI = np.sqrt(3) / 2

# 时间向量 (与 reference 一致)
_t_full = np.linspace(0, 604800 - 1800, 336)[:-24]  # 312 个点
TWEEK_SAMPLE = _t_full[48:]  # 跳过第1天, 264 个点

# 颜色
martaRed = "#c24c51"
TRAJ_COLOR = martaRed
FIRST_COLOR = "#404040"


# ============================================================
# Step 1: 真实数据读取
# ============================================================

# ---- 内存缓存: 避免反复读取同一 CSV ----

_LCL_FILE_LIST = None          # LCL 文件路径列表
_LCL_CACHE = {}                # filepath → processed DataFrame
_PV_DF_CACHE = None            # PV 数据 (只读一次)


def _get_lcl_filelist():
    global _LCL_FILE_LIST
    if _LCL_FILE_LIST is None:
        _LCL_FILE_LIST = sorted(globmod.glob(str(LCL_DIR / "LCL-June2015v2_*.csv")))
        if not _LCL_FILE_LIST:
            raise FileNotFoundError(f"No LCL CSV files in {LCL_DIR}")
    return _LCL_FILE_LIST


def _load_lcl_file(filepath):
    """读取并缓存一个 LCL CSV 文件"""
    if filepath not in _LCL_CACHE:
        raw_df = pd.read_csv(filepath, header=0)
        df = raw_df.copy()
        df["date"] = pd.to_datetime(df["DateTime"])
        data = df.loc[:, ["LCLid", "KWH/hh (per half hour) "]]
        data = data.set_index(df["date"])
        data["KWH/hh (per half hour) "] = pd.to_numeric(
            data["KWH/hh (per half hour) "], downcast="float", errors="coerce"
        )
        _LCL_CACHE[filepath] = data
    return _LCL_CACHE[filepath]


def _load_pv_df():
    """读取并缓存 PV 数据"""
    global _PV_DF_CACHE
    if _PV_DF_CACHE is None:
        raw_df = pd.read_csv(PV_FILE, header=0)
        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
        df = raw_df.set_index("datetime")
        df["P_GEN_MAX"] = pd.to_numeric(df["P_GEN_MAX"], downcast="float", errors="coerce")
        df["P_GEN_MIN"] = pd.to_numeric(df["P_GEN_MIN"], downcast="float", errors="coerce")
        _PV_DF_CACHE = df
    return _PV_DF_CACHE


def _build_week_curve(month_df, value_col, day_of_week_col="day_of_week"):
    """
    通用: 将月份数据按 day_of_week 聚合为周曲线 (secs, means)。
    month_df 必须已有 day_of_week 列, index 为 datetime。
    返回: (secs, means) 或 None
    """
    # 重映射日期到 1970-01-01~07
    indexaslist = month_df.index.tolist()
    dow_vals = month_df[day_of_week_col].values
    for p in range(len(indexaslist)):
        indexaslist[p] = indexaslist[p].replace(year=1970, month=1, day=int(dow_vals[p]) + 1)
    month_df = month_df.copy()
    month_df.index = indexaslist

    dates, means = [], []
    for dow in range(7):
        pows = month_df.loc[month_df[day_of_week_col] == dow]
        for t in sorted(pows.index.unique()):
            vals = pows.loc[pows.index == t, value_col].to_numpy().astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                means.append(np.mean(vals))
                dates.append(t)

    if len(dates) == 0:
        return None

    z = sorted(zip(dates, means), key=lambda pair: pair[0])
    dates, means = zip(*z)

    epoch = pd.Timestamp("1970-01-01")
    secs = np.array([(d - epoch).total_seconds() for d in dates])
    means = np.array(means, dtype=float)
    return secs, means


def make_random_week_profiles(monthchoice, rng=None):
    """
    从 LCL 用电数据随机选取一户，构建指定月份的平均周功率曲线。
    返回: (secs, means), True) 或 (None, False)
    """
    if rng is None:
        rng = np.random.default_rng()

    filelist = _get_lcl_filelist()
    f = filelist[rng.integers(0, len(filelist))]
    df = _load_lcl_file(f)

    # 随机选住户
    houselist = list(df.LCLid.unique())
    house = houselist[rng.integers(0, len(houselist))]

    selection = df.loc[df["LCLid"] == house].copy()
    selection.index = pd.to_datetime(selection.index)

    # 筛选目标月份
    month = selection.loc[selection.index.month == monthchoice].copy()
    if month.empty:
        return None, False

    month["day_of_week"] = month.index.dayofweek
    month = month.sort_index()

    if len(month.day_of_week.unique()) < 7:
        return None, False

    result = _build_week_curve(month, "KWH/hh (per half hour) ")
    if result is None:
        return None, False
    return result, True


def make_random_week_profiles_PV(monthchoice, rng=None):
    """
    从 PV 发电数据随机选取一个面板，构建指定月份的平均周发电曲线。
    返回: ((secs, means), True) 或 (None, False)
    """
    if rng is None:
        rng = np.random.default_rng()

    df = _load_pv_df()

    pvlist = list(df.Substation.unique())
    pv = pvlist[rng.integers(0, len(pvlist))]

    selection = df.loc[df["Substation"] == pv].copy()
    selection.index = pd.to_datetime(selection.index)

    month = selection.loc[selection.index.month == monthchoice].copy()
    if month.empty:
        return None, False

    # 计算 (P_GEN_MAX + P_GEN_MIN)/2 作为发电功率
    month["P_GEN_AVG"] = (month["P_GEN_MAX"] + month["P_GEN_MIN"]) / 2.0
    month["day_of_week"] = month.index.dayofweek
    month = month.sort_index()

    if len(month.day_of_week.unique()) < 7:
        return None, False

    result = _build_week_curve(month, "P_GEN_AVG")
    if result is None:
        return None, False
    return result, True


# ============================================================
# Step 2: 微电网构建 & Step 3: 轨迹计算
# ============================================================

def continuoussourcesinkcounter(Pvec):
    """计算连续 source/sink/passive 密度 (与 reference 一致)"""
    largest_source = np.max(Pvec)
    n = len(Pvec)
    largest_sink = np.abs(np.min(Pvec))

    if largest_source == 0 or largest_sink == 0:
        # 退化情况: 全部为 source 或全部为 sink
        return 0.0, 0.0, 1.0

    source_terms = Pvec[Pvec > 0.0]
    sink_terms = Pvec[Pvec < 0.0]

    sigma_s = np.sum(source_terms) / (n * largest_source)
    sigma_d = np.sum(np.abs(sink_terms)) / (n * largest_sink)
    sigma_p = 1.0 - sigma_s - sigma_d

    return sigma_s, sigma_d, sigma_p


def _assign_profile_with_retry(profile_func, monthchoice, rng, max_tries=200):
    """反复尝试获取一个有效的周功率曲线, 返回 (secs, means) 元组"""
    for _ in range(max_tries):
        result, ok = profile_func(monthchoice, rng=rng)
        if ok:
            return result  # (secs, means) tuple
    raise RuntimeError(
        f"Failed to get a valid week profile after {max_tries} tries "
        f"(month={monthchoice})"
    )


def compute_one_ensemble(month, penetration, rng):
    """
    构建一个随机微电网并计算其周轨迹的 sigma 序列。

    返回: list of (sigma_s, sigma_d, sigma_p), 长度 = len(TWEEK_SAMPLE)
    """
    n = N_NODES

    # --- 为每个节点分配用电曲线插值器 ---
    consumption_interps = []
    for i in range(n - 1):  # 前 49 个节点
        secs, vals = _assign_profile_with_retry(make_random_week_profiles, month, rng)
        consumption_interps.append(interp1d(secs, vals, bounds_error=False, fill_value=0.0))

    # --- 为前 penetration 个节点分配 PV 发电曲线插值器 ---
    generation_interps = []
    for i in range(penetration):
        secs, vals = _assign_profile_with_retry(make_random_week_profiles_PV, month, rng)
        generation_interps.append(interp1d(secs, vals, bounds_error=False, fill_value=0.0))

    # --- 时间演化, 每步计算功率向量 → sigmas ---
    sigmas = []
    Pvec = np.zeros(n)

    for t in TWEEK_SAMPLE:
        Pvec[:] = 0.0
        for i in range(n - 1):
            demand = consumption_interps[i](t)
            if i < penetration:
                gen = generation_interps[i](t)
                Pvec[i] = gen - demand
            else:
                Pvec[i] = -demand
        # slack bus
        Pvec[-1] = -np.sum(Pvec[:-1])

        sigma = continuoussourcesinkcounter(Pvec)
        sigmas.append(sigma)

    return sigmas


def compute_ensemble_sigmas(month, penetration, n_ensemble=N_ENSEMBLE, seed=42):
    """
    计算 n_ensemble 个随机微电网的 sigma 轨迹，带 npz 缓存。

    返回: list of list of (sigma_s, sigma_d, sigma_p)
    """
    cache_file = CACHE_DIR / f"sigmas_m{month}_p{penetration}_n{n_ensemble}.npz"

    if cache_file.exists():
        print(f"  Loading cache: {cache_file.name}")
        data = np.load(cache_file)
        all_sigmas = []
        for i in range(n_ensemble):
            arr = data[f"ensemble_{i}"]  # shape (T, 3)
            all_sigmas.append([tuple(row) for row in arr])
        return all_sigmas

    rng = np.random.default_rng(seed)
    all_sigmas = []

    for z in range(n_ensemble):
        print(f"  Ensemble {z+1}/{n_ensemble} (month={month}, pen={penetration})")
        sigmas = compute_one_ensemble(month, penetration, rng)
        all_sigmas.append(sigmas)

    # 保存缓存
    save_dict = {}
    for i, sigmas in enumerate(all_sigmas):
        save_dict[f"ensemble_{i}"] = np.array(sigmas)
    np.savez_compressed(cache_file, **save_dict)
    print(f"  Cached to {cache_file.name}")

    return all_sigmas


# ============================================================
# Step 4: 绘图 (复用已有逻辑)
# ============================================================

def sigma_to_simplex_point(sigma, n):
    """sigma (source, sink, passive) → simplex point (ns, nd, ne)"""
    s = sigma[0] * n
    d = sigma[1] * n
    p = n - s - d
    return (s, d, p)


def simplex_to_cart(ns, nd, ne, scale):
    """Simplex point → Cartesian (x, y)
    与 python-ternary 坐标一致, reference 的 remap:
        (ns, nd, ne) → (a=nd-1, b=ne, c=ns-1)
    """
    a = nd - 1
    b = ne
    s = scale - 2
    x = (a + b / 2.0) / s
    y = (b * H_TRI) / s
    return x, y


def sigmas_to_cart_path(sigmas, n):
    """将 sigma 序列转为 (x, y) 路径"""
    xs, ys = [], []
    for sigma in sigmas:
        sp = sigma_to_simplex_point(sigma, n)
        x, y = simplex_to_cart(sp[0], sp[1], sp[2], n)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def compute_mean_trajectory(all_sigmas):
    """对所有 ensemble 成员的 sigma 序列逐点取均值"""
    n_steps = min(len(s) for s in all_sigmas)
    mean_sigmas = []
    for t in range(n_steps):
        s_sum, d_sum = 0.0, 0.0
        count = 0
        for member in all_sigmas:
            if t < len(member):
                s_sum += member[t][0]
                d_sum += member[t][1]
                count += 1
        s_mean = s_sum / count
        d_mean = d_sum / count
        p_mean = 1.0 - s_mean - d_mean
        mean_sigmas.append((s_mean, d_mean, p_mean))
    return mean_sigmas


def draw_ternary_frame(ax, scale):
    """绘制三角形边框和网格线"""
    n = scale
    s = n - 2

    bl = np.array([(-1) / s, 0.0])
    br = np.array([(n - 1) / s, 0.0])
    top = np.array([(-1 + n / 2) / s, n * H_TRI / s])

    corners = np.array([bl, br, top, bl])
    ax.plot(corners[:, 0], corners[:, 1], 'k-', lw=1.5)

    multiple = 6
    n_lines = s // multiple
    for k in range(1, n_lines + 1):
        frac = k * multiple / s

        p0 = bl + frac * (top - bl)
        p1 = br + frac * (top - br)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.3, alpha=0.5)

        p0 = bl + frac * (br - bl)
        p1 = top + frac * (br - top)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.3, alpha=0.5)

        p0 = br + frac * (bl - br)
        p1 = top + frac * (bl - top)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.3, alpha=0.5)

    mid_left_x = (bl[0] + top[0]) / 2 - 0.08
    mid_left_y = (bl[1] + top[1]) / 2
    ax.text(mid_left_x, mid_left_y, r'$\leftarrow\,\eta_+$',
            fontsize=11, rotation=60, ha='center', va='center')

    mid_right_x = (br[0] + top[0]) / 2 + 0.08
    mid_right_y = (br[1] + top[1]) / 2
    ax.text(mid_right_x, mid_right_y, r'$\eta_p\,\rightarrow$',
            fontsize=11, rotation=-60, ha='center', va='center')

    mid_bot_x = (bl[0] + br[0]) / 2
    mid_bot_y = bl[1] - 0.06
    ax.text(mid_bot_x, mid_bot_y, r'$\eta_-\,\rightarrow$',
            fontsize=11, ha='center', va='top')


def plot_panel(ax, all_sigmas, label):
    """绘制 Fig4 D-G 单个三角形 panel"""
    n = N_NODES

    draw_ternary_frame(ax, n)

    for i, sigmas in enumerate(all_sigmas):
        xs, ys = sigmas_to_cart_path(sigmas, n)
        color = FIRST_COLOR if i == 0 else TRAJ_COLOR
        ax.plot(xs, ys, color=color, alpha=0.02, lw=0.9)

    mean_sigmas = compute_mean_trajectory(all_sigmas)
    xs, ys = sigmas_to_cart_path(mean_sigmas, n)
    ax.plot(xs, ys, color=TRAJ_COLOR, alpha=0.8, lw=2.8)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(label, loc='left', fontweight='bold', fontsize=16)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, H_TRI * n / (n - 2) + 0.05)


# ============================================================
# 主函数
# ============================================================

def main():
    font = {'size': 14}
    matplotlib.rc('font', **font)

    print("Computing trajectories from real data...")

    # D: Winter, 50% PV
    print("Panel D: Winter, 50% PV uptake")
    sigmas_d = compute_ensemble_sigmas(month=1, penetration=24)

    # E: Winter, 100% PV
    print("Panel E: Winter, 100% PV uptake")
    sigmas_e = compute_ensemble_sigmas(month=1, penetration=49)

    # F: Summer, 50% PV
    print("Panel F: Summer, 50% PV uptake")
    sigmas_f = compute_ensemble_sigmas(month=7, penetration=24)

    # G: Summer, 100% PV
    print("Panel G: Summer, 100% PV uptake")
    sigmas_g = compute_ensemble_sigmas(month=7, penetration=49)

    print("Plotting...")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    plot_panel(axes[0], sigmas_d, 'D')
    plot_panel(axes[1], sigmas_e, 'E')
    plot_panel(axes[2], sigmas_f, 'F')
    plot_panel(axes[3], sigmas_g, 'G')

    fig.tight_layout(pad=1.0)
    fig.savefig(OUTPUT_DIR / "figure4_DEFG.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "figure4_DEFG.png", dpi=200, bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR / 'figure4_DEFG.pdf'} and .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
