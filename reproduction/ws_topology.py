"""
W-S 网络拓扑指标分析
检验平均路径长度 L(q)、聚类系数 C(q)、代数连通度 λ₂(q) 随 k 和 q 的变化趋势

输出：ws_topology_q.pdf / ws_topology_q.png
      ws_topology_k.pdf / ws_topology_k.png
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────────────────────────
N = 100                          # 节点数
K_VALUES = [4, 6, 8, 10]        # 平均度 ⟨k⟩ 候选值
K_REF = 4                        # 固定 k 时参考值 (q 扫描)

# q 轴使用对数间隔（更清晰地展示小世界区域）
Q_VALUES = np.concatenate([
    np.logspace(-3, -0.3, 30),
    np.linspace(0.5, 1.0, 10),
])
Q_VALUES = np.unique(np.clip(Q_VALUES, 0.001, 1.0))

REALIZATIONS = 50               # 每组参数的网络实现数
SEED = 42


# ── 核心计算 ──────────────────────────────────────────────────────────────────

def compute_metrics_single(n, k, q, seed):
    """生成一个 W-S 网络，计算三个拓扑指标。"""
    rng = np.random.default_rng(seed)
    nx_seed = int(rng.integers(0, 2**31 - 1))
    G = nx.connected_watts_strogatz_graph(n, k, q, seed=nx_seed)

    # 平均路径长度
    L = nx.average_shortest_path_length(G)

    # 聚类系数（平均）
    C = nx.average_clustering(G)

    # 代数连通度 λ₂：Laplacian 第二小特征值
    L_mat = nx.laplacian_matrix(G).toarray().astype(float)
    eigvals = np.linalg.eigvalsh(L_mat)
    eigvals_sorted = np.sort(eigvals)
    lambda2 = eigvals_sorted[1]   # 最小非零特征值

    return L, C, lambda2


def compute_sweep_q(n, k, q_values, realizations, seed_base):
    """固定 k，扫描 q，返回各指标均值和标准差。"""
    L_mean, L_std = [], []
    C_mean, C_std = [], []
    lam_mean, lam_std = [], []

    for qi, q in enumerate(q_values):
        Ls, Cs, lams = [], [], []
        for r in range(realizations):
            seed = seed_base + qi * 1000 + r
            L, C, lam = compute_metrics_single(n, k, q, seed)
            Ls.append(L); Cs.append(C); lams.append(lam)

        L_mean.append(np.mean(Ls)); L_std.append(np.std(Ls))
        C_mean.append(np.mean(Cs)); C_std.append(np.std(Cs))
        lam_mean.append(np.mean(lams)); lam_std.append(np.std(lams))

    return (np.array(L_mean), np.array(L_std),
            np.array(C_mean), np.array(C_std),
            np.array(lam_mean), np.array(lam_std))


def compute_sweep_k(n, k_values, q_values, realizations, seed_base):
    """扫描多个 k 值，返回字典 {k: (L_mean, L_std, C_mean, C_std, lam_mean, lam_std)}。"""
    results = {}
    for ki, k in enumerate(k_values):
        print(f"  k={k} …")
        results[k] = compute_sweep_q(n, k, q_values, realizations,
                                     seed_base + ki * 100000)
    return results


# ── 缓存层 ────────────────────────────────────────────────────────────────────

def cache_path_q(n, k, realizations):
    return CACHE_DIR / f"ws_topo_q_n{n}_k{k}_r{realizations}.npz"


def cache_path_k(n, k_values_str, realizations):
    return CACHE_DIR / f"ws_topo_k_n{n}_ks{k_values_str}_r{realizations}.npz"


def load_or_compute_sweep_q(n, k, q_values, realizations, seed_base):
    cp = cache_path_q(n, k, realizations)
    if cp.exists():
        print(f"  [cache] loaded {cp.name}")
        data = np.load(cp)
        return (data['q_values'],
                data['L_mean'], data['L_std'],
                data['C_mean'], data['C_std'],
                data['lam_mean'], data['lam_std'])

    print(f"  computing k={k}, {len(q_values)} q-points × {realizations} realizations …")
    L_m, L_s, C_m, C_s, lam_m, lam_s = compute_sweep_q(
        n, k, q_values, realizations, seed_base)
    np.savez(cp, q_values=q_values,
             L_mean=L_m, L_std=L_s,
             C_mean=C_m, C_std=C_s,
             lam_mean=lam_m, lam_std=lam_s)
    print(f"  [cache] saved {cp.name}")
    return q_values, L_m, L_s, C_m, C_s, lam_m, lam_s


def load_or_compute_all_k(n, k_values, q_values, realizations, seed_base):
    results = {}
    for ki, k in enumerate(k_values):
        q_vals, L_m, L_s, C_m, C_s, lam_m, lam_s = load_or_compute_sweep_q(
            n, k, q_values, realizations, seed_base + ki * 100000)
        results[k] = (q_vals, L_m, L_s, C_m, C_s, lam_m, lam_s)
    return results


# ── 绘图 ──────────────────────────────────────────────────────────────────────

COLORS = ['#e8850c', '#4a7ebb', '#2ca02c', '#d62728']
LINESTYLES = ['-', '--', '-.', ':']


def plot_normalized_vs_q(results, title_suffix=""):
    """
    绘制归一化指标 L(q)/L(0)、C(q)/C(0)、λ₂(q)/λ₂(0) vs q
    每条曲线对应不同 k 值
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax_L, ax_C, ax_lam = axes

    for ki, (k, color, ls) in enumerate(zip(K_VALUES, COLORS, LINESTYLES)):
        q_vals, L_m, L_s, C_m, C_s, lam_m, lam_s = results[k]

        # 归一化：除以最小 q 点（接近规则环网）处的值
        L0 = L_m[0]; C0 = C_m[0]; lam0 = lam_m[0]

        label = rf"$\langle k\rangle = {k}$"

        ax_L.semilogx(q_vals, L_m / L0, color=color, ls=ls, lw=2.0, label=label)
        ax_L.fill_between(q_vals,
                          (L_m - L_s) / L0, (L_m + L_s) / L0,
                          color=color, alpha=0.15)

        ax_C.semilogx(q_vals, C_m / C0, color=color, ls=ls, lw=2.0, label=label)
        ax_C.fill_between(q_vals,
                          (C_m - C_s) / C0, (C_m + C_s) / C0,
                          color=color, alpha=0.15)

        ax_lam.semilogx(q_vals, lam_m / lam0, color=color, ls=ls, lw=2.0, label=label)
        ax_lam.fill_between(q_vals,
                            (lam_m - lam_s) / lam0, (lam_m + lam_s) / lam0,
                            color=color, alpha=0.15)

    for ax, ylabel, title in [
        (ax_L,   r"$L(q)\,/\,L(0)$",                   r"Average Path Length $L(q)$"),
        (ax_C,   r"$C(q)\,/\,C(0)$",                   r"Clustering Coefficient $C(q)$"),
        (ax_lam, r"$\lambda_2(q)\,/\,\lambda_2(0)$",   r"Algebraic Connectivity $\lambda_2(q)$"),
    ]:
        ax.set_xlabel(r"Rewiring probability $q$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(q_vals[0], 1.0)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, which='both', alpha=0.3)

    # highlight small-world regime
    for ax in [ax_L, ax_C]:
        ax.axvspan(0.01, 0.1, alpha=0.08, color='gray',
                   label='Small-world regime' if ax is ax_L else None)

    fig.suptitle(
        rf"W-S Network Topology Metrics vs Rewiring Probability $q$  ($n={N}$, varying $\langle k\rangle$)",
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    return fig


def plot_raw_vs_q(results):
    """绘制原始（未归一化）指标，3行×(len K_VALUES)列 对比"""
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax_L, ax_C, ax_lam = axes

    for ki, (k, color, ls) in enumerate(zip(K_VALUES, COLORS, LINESTYLES)):
        q_vals, L_m, L_s, C_m, C_s, lam_m, lam_s = results[k]
        label = rf"$\langle k\rangle = {k}$"

        ax_L.semilogx(q_vals, L_m, color=color, ls=ls, lw=2.0, label=label)
        ax_L.fill_between(q_vals, L_m - L_s, L_m + L_s, color=color, alpha=0.15)

        ax_C.semilogx(q_vals, C_m, color=color, ls=ls, lw=2.0, label=label)
        ax_C.fill_between(q_vals, C_m - C_s, C_m + C_s, color=color, alpha=0.15)

        ax_lam.semilogx(q_vals, lam_m, color=color, ls=ls, lw=2.0, label=label)
        ax_lam.fill_between(q_vals, lam_m - lam_s, lam_m + lam_s, color=color, alpha=0.15)

    ax_L.set_ylabel(r"$L(q)$", fontsize=13)
    ax_C.set_ylabel(r"$C(q)$", fontsize=13)
    ax_lam.set_ylabel(r"$\lambda_2(q)$", fontsize=13)
    ax_lam.set_xlabel(r"Rewiring probability $q$", fontsize=13)

    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(q_vals[0], 1.0)
        ax.axvspan(0.01, 0.1, alpha=0.07, color='gray')

    ax_L.set_title(
        rf"W-S Network Topology Metrics vs $q$  ($n={N}$, varying $\langle k\rangle$)",
        fontsize=12
    )
    fig.tight_layout()
    return fig


# ── 打印数值摘要 ──────────────────────────────────────────────────────────────

def print_summary(results, k_ref):
    q_vals, L_m, _, C_m, _, lam_m, _ = results[k_ref]
    print(f"\n{'='*60}")
    print(f"数值摘要  (k={k_ref}, n={N})")
    print(f"{'='*60}")
    print(f"{'q':>10} {'L(q)':>10} {'C(q)':>10} {'λ₂(q)':>12}")
    print(f"{'-'*45}")
    # 选取代表性 q 点
    target_qs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    for tq in target_qs:
        idx = np.argmin(np.abs(q_vals - tq))
        print(f"{q_vals[idx]:>10.4f} {L_m[idx]:>10.4f} {C_m[idx]:>10.4f} {lam_m[idx]:>12.6f}")
    print(f"\n小世界特性验证 (q ≈ 0.05):")
    idx_sw = np.argmin(np.abs(q_vals - 0.05))
    idx_0  = 0
    print(f"  L(q)/L(0) = {L_m[idx_sw]/L_m[idx_0]:.3f}  (应 << 1)")
    print(f"  C(q)/C(0) = {C_m[idx_sw]/C_m[idx_0]:.3f}  (应 ≈ 1)")
    print(f"  → 小世界: 短路径 + 高聚类 ✓" if
          L_m[idx_sw]/L_m[idx_0] < 0.5 and C_m[idx_sw]/C_m[idx_0] > 0.5
          else "  → 小世界特性未完全体现，检查参数")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("W-S 网络拓扑指标分析")
    print(f"n={N}, k_values={K_VALUES}, realizations={REALIZATIONS}")
    print("=" * 60)

    print("\n[1/2] 计算各 k 值下的拓扑指标 …")
    results = load_or_compute_all_k(
        n=N, k_values=K_VALUES, q_values=Q_VALUES,
        realizations=REALIZATIONS, seed_base=SEED
    )

    print_summary(results, k_ref=K_REF)

    print("\n[2/2] 绘图 …")

    # 图1：归一化曲线（横排3面板）
    fig_norm = plot_normalized_vs_q(results)
    p_norm_png = OUTPUT_DIR / "ws_topology_normalized.png"
    p_norm_pdf = OUTPUT_DIR / "ws_topology_normalized.pdf"
    fig_norm.savefig(p_norm_png, dpi=200, bbox_inches='tight')
    fig_norm.savefig(p_norm_pdf, dpi=300, bbox_inches='tight')
    print(f"  saved {p_norm_png.name}")

    # 图2：原始值曲线（纵排3面板）
    fig_raw = plot_raw_vs_q(results)
    p_raw_png = OUTPUT_DIR / "ws_topology_raw.png"
    p_raw_pdf = OUTPUT_DIR / "ws_topology_raw.pdf"
    fig_raw.savefig(p_raw_png, dpi=200, bbox_inches='tight')
    fig_raw.savefig(p_raw_pdf, dpi=300, bbox_inches='tight')
    print(f"  saved {p_raw_png.name}")

    plt.show()
    print("\n完成。")


if __name__ == "__main__":
    main()
