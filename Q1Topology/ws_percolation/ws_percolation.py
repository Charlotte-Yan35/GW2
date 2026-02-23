"""
W-S 拓扑结构 bond percolation 分析
研究不同 rewiring probability q_ws、平均度 k、距离衰减指数 α 对渗流阈值 q_c 的影响。
"""

import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
from pathlib import Path
import random
import sys
import os

# 把 ws_kappa_c 加入路径，复用其函数
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ws_kappa_c"))
from ws_kappa_c import integrate_swing, find_kappa_c, assign_powers

# ============================================================
# 全局参数
# ============================================================
N = 50
K_BAR = 4
PMAX = 5.0
REALIZATIONS = 50       # 网络实现数
N_PERC_TRIALS = 20      # 每个网络的渗流重复数
Q_PERC_VALUES = np.linspace(0, 1, 201)  # bond removal probability

Q_WS_LIST = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]
ALPHA_LIST = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 网络生成
# ============================================================

def generate_ws_network(n, k, q_ws):
    """标准 Watts-Strogatz 网络（保证连通）。"""
    return nx.connected_watts_strogatz_graph(n, k, q_ws)


def generate_ws_network_alpha(n, k, alpha, seed=None):
    """
    距离依赖的 Watts-Strogatz 网络。
    从环形格子出发，以距离衰减概率 rewire：P(d) ∝ d^{-α}。
    α=0 → 均匀 rewiring（标准 W-S, q=1）
    α→∞ → 几乎不 rewire（类似环形格子）
    """
    rng = np.random.default_rng(seed)

    # 起始：环形格子，每个节点连 k/2 个最近邻（每侧）
    G = nx.watts_strogatz_graph(n, k, 0, seed=int(rng.integers(0, 10**9)))

    edges_to_rewire = list(G.edges())
    rng.shuffle(edges_to_rewire)

    for u, v in edges_to_rewire:
        # 计算到所有非邻居节点的环形距离
        candidates = [w for w in range(n) if w != u and not G.has_edge(u, w)]
        if not candidates:
            continue

        distances = np.array([min(abs(u - w), n - abs(u - w)) for w in candidates])
        distances = np.maximum(distances, 1)

        if alpha == 0:
            probs = np.ones(len(candidates))
        else:
            probs = distances.astype(float) ** (-alpha)

        probs /= probs.sum()

        new_v = rng.choice(candidates, p=probs)
        G.remove_edge(u, v)
        G.add_edge(u, new_v)

    # 保证连通
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            u = rng.choice(list(components[0]))
            v = rng.choice(list(components[i]))
            G.add_edge(u, v)
            components[0] = components[0] | components[i]

    return G


# ============================================================
# Bond percolation 核心
# ============================================================

def bond_percolation_single(G, q_remove, rng):
    """
    以概率 q_remove 移除边，返回 (S1, S2)。
    S1 = 最大连通分量 / N
    S2 = 第二大连通分量 / N
    """
    n = G.number_of_nodes()
    if q_remove == 0:
        return 1.0, 0.0
    if q_remove >= 1.0:
        return 1.0 / n, 0.0

    H = G.copy()
    edges = list(H.edges())
    mask = rng.random(len(edges)) < q_remove
    edges_to_remove = [e for e, m in zip(edges, mask) if m]
    H.remove_edges_from(edges_to_remove)

    components = sorted(nx.connected_components(H), key=len, reverse=True)
    S1 = len(components[0]) / n
    S2 = len(components[1]) / n if len(components) > 1 else 0.0
    return S1, S2


def bond_percolation_sweep(G, q_values, n_trials, rng):
    """
    对网络 G 扫描 bond removal probability，返回 S1, S2 数组。
    形状: (len(q_values),) 取 n_trials 次平均。
    """
    nq = len(q_values)
    S1_arr = np.zeros(nq)
    S2_arr = np.zeros(nq)

    for trial in range(n_trials):
        for qi, q in enumerate(q_values):
            s1, s2 = bond_percolation_single(G, q, rng)
            S1_arr[qi] += s1
            S2_arr[qi] += s2

    S1_arr /= n_trials
    S2_arr /= n_trials
    return S1_arr, S2_arr


def find_qc(q_values, S2):
    """S_2 峰值对应的 q 为渗流阈值 q_c。"""
    idx = np.argmax(S2)
    return q_values[idx]


# ============================================================
# 并行工作函数
# ============================================================

def _worker_percolation_qws(args):
    """单次实现：标准 W-S 网络 + bond percolation。"""
    n, k, q_ws, q_perc_values, n_perc_trials, seed = args
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    G = generate_ws_network(n, k, q_ws)
    S1, S2 = bond_percolation_sweep(G, q_perc_values, n_perc_trials, rng)
    qc = find_qc(q_perc_values, S2)
    return S1, S2, qc


def _worker_percolation_alpha(args):
    """单次实现：α-W-S 网络 + bond percolation。"""
    n, k, alpha, q_perc_values, n_perc_trials, seed = args
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    G = generate_ws_network_alpha(n, k, alpha, seed=seed)
    S1, S2 = bond_percolation_sweep(G, q_perc_values, n_perc_trials, rng)
    qc = find_qc(q_perc_values, S2)
    return S1, S2, qc


def _worker_kappa_c_qws(args):
    """单次实现：标准 W-S 网络 + κ_c。"""
    n, k, q_ws, n_plus, n_minus, Pmax, seed = args
    np.random.seed(seed)
    random.seed(seed)
    A = nx.to_numpy_array(generate_ws_network(n, k, q_ws))
    P = assign_powers(n, n_plus, n_minus, Pmax)
    return find_kappa_c(A, P, n)


def _worker_kappa_c_alpha(args):
    """单次实现：α-W-S 网络 + κ_c。"""
    n, k, alpha, n_plus, n_minus, Pmax, seed = args
    np.random.seed(seed)
    random.seed(seed)
    G = generate_ws_network_alpha(n, k, alpha, seed=seed)
    A = nx.to_numpy_array(G)
    P = assign_powers(n, n_plus, n_minus, Pmax)
    return find_kappa_c(A, P, n)


# ============================================================
# 集成计算
# ============================================================

def compute_percolation_vs_qws(n=N, k=K_BAR, q_ws_list=None, realizations=REALIZATIONS,
                                n_perc_trials=N_PERC_TRIALS, q_perc_values=None):
    """
    扫描 W-S rewiring probability，返回渗流数据。
    返回: dict with keys 'q_ws_list', 'q_perc_values',
          'S1_mean', 'S2_mean' (shape: len(q_ws_list) x len(q_perc_values)),
          'qc_mean', 'qc_std' (shape: len(q_ws_list),)
    """
    if q_ws_list is None:
        q_ws_list = Q_WS_LIST
    if q_perc_values is None:
        q_perc_values = Q_PERC_VALUES

    cache_file = CACHE_DIR / f"perc_vs_qws_n{n}_k{k}_R{realizations}.npz"
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file.name}")
        return dict(np.load(cache_file, allow_pickle=True))

    nq_ws = len(q_ws_list)
    nq_perc = len(q_perc_values)
    S1_all = np.zeros((nq_ws, nq_perc))
    S2_all = np.zeros((nq_ws, nq_perc))
    qc_all = np.zeros((nq_ws, realizations))

    rng_master = np.random.default_rng(42)
    n_workers = min(cpu_count(), 8)

    with Pool(n_workers) as pool:
        for qi, q_ws in enumerate(q_ws_list):
            print(f"  q_ws={q_ws:.2f} ...")
            seeds = rng_master.integers(0, 10**9, size=realizations)
            args_list = [(n, k, q_ws, q_perc_values, n_perc_trials, int(s))
                         for s in seeds]
            results = pool.map(_worker_percolation_qws, args_list)

            S1_stack = np.array([r[0] for r in results])
            S2_stack = np.array([r[1] for r in results])
            qc_stack = np.array([r[2] for r in results])

            S1_all[qi] = S1_stack.mean(axis=0)
            S2_all[qi] = S2_stack.mean(axis=0)
            qc_all[qi] = qc_stack

    result = {
        'q_ws_list': np.array(q_ws_list),
        'q_perc_values': q_perc_values,
        'S1_mean': S1_all,
        'S2_mean': S2_all,
        'qc_mean': np.mean(qc_all, axis=1),
        'qc_std': np.std(qc_all, axis=1),
        'qc_all': qc_all,
    }
    np.savez(cache_file, **result)
    print(f"  Cached → {cache_file.name}")
    return result


def compute_percolation_vs_alpha(n=N, k=K_BAR, alpha_list=None, realizations=REALIZATIONS,
                                  n_perc_trials=N_PERC_TRIALS, q_perc_values=None):
    """
    扫描 distance decay exponent α，返回渗流数据。
    """
    if alpha_list is None:
        alpha_list = ALPHA_LIST
    if q_perc_values is None:
        q_perc_values = Q_PERC_VALUES

    cache_file = CACHE_DIR / f"perc_vs_alpha_n{n}_k{k}_R{realizations}.npz"
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file.name}")
        return dict(np.load(cache_file, allow_pickle=True))

    n_alpha = len(alpha_list)
    nq_perc = len(q_perc_values)
    S1_all = np.zeros((n_alpha, nq_perc))
    S2_all = np.zeros((n_alpha, nq_perc))
    qc_all = np.zeros((n_alpha, realizations))

    rng_master = np.random.default_rng(42)
    n_workers = min(cpu_count(), 8)

    with Pool(n_workers) as pool:
        for ai, alpha in enumerate(alpha_list):
            print(f"  α={alpha:.1f} ...")
            seeds = rng_master.integers(0, 10**9, size=realizations)
            args_list = [(n, k, alpha, q_perc_values, n_perc_trials, int(s))
                         for s in seeds]
            results = pool.map(_worker_percolation_alpha, args_list)

            S1_stack = np.array([r[0] for r in results])
            S2_stack = np.array([r[1] for r in results])
            qc_stack = np.array([r[2] for r in results])

            S1_all[ai] = S1_stack.mean(axis=0)
            S2_all[ai] = S2_stack.mean(axis=0)
            qc_all[ai] = qc_stack

    result = {
        'alpha_list': np.array(alpha_list),
        'q_perc_values': q_perc_values,
        'S1_mean': S1_all,
        'S2_mean': S2_all,
        'qc_mean': np.mean(qc_all, axis=1),
        'qc_std': np.std(qc_all, axis=1),
        'qc_all': qc_all,
    }
    np.savez(cache_file, **result)
    print(f"  Cached → {cache_file.name}")
    return result


def compute_kappa_c_vs_qws(n=N, k=K_BAR, q_ws_list=None, realizations=REALIZATIONS,
                            n_plus=25, n_minus=25):
    """κ_c vs q_ws（与渗流对比用）。"""
    if q_ws_list is None:
        q_ws_list = Q_WS_LIST

    cache_file = CACHE_DIR / f"kappa_c_vs_qws_n{n}_k{k}_R{realizations}.npz"
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file.name}")
        return dict(np.load(cache_file, allow_pickle=True))

    rng_master = np.random.default_rng(42)
    kappa_c_mean = np.zeros(len(q_ws_list))
    kappa_c_std = np.zeros(len(q_ws_list))
    n_workers = min(cpu_count(), 8)

    with Pool(n_workers) as pool:
        for qi, q_ws in enumerate(q_ws_list):
            print(f"  κ_c: q_ws={q_ws:.2f} ...")
            seeds = rng_master.integers(0, 10**9, size=realizations)
            args_list = [(n, k, q_ws, n_plus, n_minus, PMAX, int(s)) for s in seeds]
            results = pool.map(_worker_kappa_c_qws, args_list)
            values = np.array([r for r in results if not np.isnan(r)])
            kappa_c_mean[qi] = np.mean(values) if len(values) > 0 else np.nan
            kappa_c_std[qi] = np.std(values) if len(values) > 0 else np.nan
            print(f"    κ_c = {kappa_c_mean[qi]:.4f} ± {kappa_c_std[qi]:.4f}")

    result = {
        'q_ws_list': np.array(q_ws_list),
        'kappa_c_mean': kappa_c_mean,
        'kappa_c_std': kappa_c_std,
    }
    np.savez(cache_file, **result)
    print(f"  Cached → {cache_file.name}")
    return result


def compute_kappa_c_vs_alpha(n=N, k=K_BAR, alpha_list=None, realizations=REALIZATIONS,
                              n_plus=25, n_minus=25):
    """κ_c vs α。"""
    if alpha_list is None:
        alpha_list = ALPHA_LIST

    cache_file = CACHE_DIR / f"kappa_c_vs_alpha_n{n}_k{k}_R{realizations}.npz"
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file.name}")
        return dict(np.load(cache_file, allow_pickle=True))

    rng_master = np.random.default_rng(42)
    kappa_c_mean = np.zeros(len(alpha_list))
    kappa_c_std = np.zeros(len(alpha_list))
    n_workers = min(cpu_count(), 8)

    with Pool(n_workers) as pool:
        for ai, alpha in enumerate(alpha_list):
            print(f"  κ_c: α={alpha:.1f} ...")
            seeds = rng_master.integers(0, 10**9, size=realizations)
            args_list = [(n, k, alpha, n_plus, n_minus, PMAX, int(s)) for s in seeds]
            results = pool.map(_worker_kappa_c_alpha, args_list)
            values = np.array([r for r in results if not np.isnan(r)])
            kappa_c_mean[ai] = np.mean(values) if len(values) > 0 else np.nan
            kappa_c_std[ai] = np.std(values) if len(values) > 0 else np.nan
            print(f"    κ_c = {kappa_c_mean[ai]:.4f} ± {kappa_c_std[ai]:.4f}")

    result = {
        'alpha_list': np.array(alpha_list),
        'kappa_c_mean': kappa_c_mean,
        'kappa_c_std': kappa_c_std,
    }
    np.savez(cache_file, **result)
    print(f"  Cached → {cache_file.name}")
    return result


# ============================================================
# 命令行入口
# ============================================================

def main():
    print("=" * 60)
    print("Bond Percolation 分析: W-S 拓扑结构")
    print(f"N={N}, K_bar={K_BAR}, R={REALIZATIONS}, perc_trials={N_PERC_TRIALS}")
    print("=" * 60)

    print("\n[1/4] Percolation vs q_ws ...")
    data_qws = compute_percolation_vs_qws()

    print("\n[2/4] Percolation vs α ...")
    data_alpha = compute_percolation_vs_alpha()

    print("\n[3/4] κ_c vs q_ws ...")
    data_kappa_qws = compute_kappa_c_vs_qws()

    print("\n[4/4] κ_c vs α ...")
    data_kappa_alpha = compute_kappa_c_vs_alpha()

    print("\n✓ 所有计算完成。数据已缓存至 cache/")
    print("运行 ws_figure.py 生成组合图。")


if __name__ == "__main__":
    main()
