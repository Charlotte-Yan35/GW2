"""
shared_utils.py — ratio_scan 包的公共函数与常量。

自包含模块，不 import Topology2.0/ 中任何内容。
所有函数从源文件逐一提取，保持语义一致。
"""

import numpy as np
import networkx as nx


# ====================================================================
# 共享常量
# ====================================================================

N = 50
N_HOUSEHOLDS = 49
PCC_NODE = 0
PMAX = 1.0
I_INERTIA = 1.0
D_DAMP = 1.0


# ====================================================================
# 1. 网络生成
# ====================================================================

def generate_ws_network(N: int, K: int, q: float,
                        seed: int | None = None) -> nx.Graph:
    """生成连通的 Watts-Strogatz 小世界图。

    Parameters
    ----------
    N : int   节点数
    K : int   每个节点连接到 K 个最近邻
    q : float 重连概率 (0=环, 1≈随机)
    seed : int, optional  随机种子

    Returns
    -------
    nx.Graph
    """
    return nx.connected_watts_strogatz_graph(N, K, q, seed=seed)


# ====================================================================
# 2. Swing 方程核心函数
# ====================================================================

def fswing(psi, A, P, n, I, D, kappa):
    """Swing equation 右端函数。

    psi = [omega_0..omega_{n-1}, theta_0..theta_{n-1}]
    """
    omega = psi[:n]
    theta = psi[n:]
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    domega = (P - D * omega - kappa * coupling) / I
    return np.concatenate([domega, omega])


def fsteadystate(theta, A, P, kappa):
    """稳态残差: P - kappa * sum_j A_ij * sin(theta_i - theta_j)"""
    diff = theta[:, None] - theta[None, :]
    coupling = np.sum(A * np.sin(diff), axis=1)
    return P - kappa * coupling


def fsteadystate_gauge(theta_reduced, A, P, kappa, pcc_node=PCC_NODE):
    """带 gauge fixing 的稳态残差。

    Gauge fixing: theta[pcc_node] = 0，从残差中移除 PCC 节点方程。

    Parameters
    ----------
    theta_reduced : ndarray (N-1,)
        不含 PCC 节点的相角
    A : ndarray (N, N)
        邻接矩阵
    P : ndarray (N,)
        功率注入
    kappa : float
        耦合强度
    pcc_node : int
        被固定的节点

    Returns
    -------
    residual : ndarray (N-1,)
    """
    n = len(P)
    # 重建完整 theta: 在 pcc_node 处插入 0
    theta_full = np.insert(theta_reduced, pcc_node, 0.0)
    # 计算完整残差
    full_resid = fsteadystate(theta_full, A, P, kappa)
    # 移除 PCC 节点方程
    reduced_resid = np.delete(full_resid, pcc_node)
    return reduced_resid


def edgepower(theta, E, kappa):
    """边功率流: kappa * sin(E^T * theta)"""
    return kappa * np.sin(E.T @ theta)


# ====================================================================
# 3. 关联矩阵与图工具
# ====================================================================

def build_incidence_matrix(G):
    """从 networkx 图构建关联矩阵 E (n x m)。"""
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    E = np.zeros((n, m))
    for j, (u, v) in enumerate(edges):
        E[u, j] = 1.0
        E[v, j] = -1.0
    return E


def adjacencymatrix_from_incidence(E):
    """邻接矩阵 A = diag(deg) - E * E^T"""
    degrees = np.sum(np.abs(E), axis=1)
    L = E @ E.T
    return np.diag(degrees) - L


def connected_components(A):
    """BFS 连通分量检测。

    Returns
    -------
    n_comp : int
    components : list of list[int]
    """
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for u in range(n):
        if not visited[u]:
            visited[u] = True
            queue = [u]
            comp = [u]
            while queue:
                v = queue.pop(0)
                for w in range(n):
                    if A[v, w] > 0.5 and not visited[w]:
                        visited[w] = True
                        queue.append(w)
                        comp.append(w)
            components.append(comp)
    return len(components), components


# ====================================================================
# 4. 比例网格
# ====================================================================

def build_ratio_grid(step=5, total=None):
    """生成三元单纯形网格 (ng, nc, np_count)，整数节点数。

    ng + nc + np_count = total
    ng >= 1, nc >= 1, np_count >= 0
    step: 分母 (e.g. 5 -> 0.2 步长)
    total: 节点总数。默认 N_HOUSEHOLDS (49)，cascade 向后兼容。
    """
    if total is None:
        total = N_HOUSEHOLDS
    pts = []
    for ig in range(1, step + 1):
        for ic in range(1, step + 1 - ig):
            ip = step - ig - ic
            if ip < 0:
                continue
            rg = ig / step
            rc = ic / step
            ng = max(1, round(rg * total))
            nc = max(1, round(rc * total))
            np_count = total - ng - nc
            if np_count < 0:
                if ng >= nc:
                    ng += np_count
                else:
                    nc += np_count
                np_count = 0
            pts.append((ng, nc, np_count))
    # 去重
    return sorted(set(pts))


# ====================================================================
# 5. 角色分配
# ====================================================================

def assign_roles(ng, nc, seed=0):
    """分配功率向量 P 和节点类型。

    Returns
    -------
    P : ndarray (N,)
        P[0]=0 (PCC), gen +Pmax/ng, con -Pmax/nc, pas 0
    node_types : ndarray (N,) int
        0=pcc, 1=gen, 2=con, 3=pas
    """
    rng = np.random.default_rng(seed)
    P = np.zeros(N)
    node_types = np.full(N, 3, dtype=int)  # 默认 passive
    node_types[PCC_NODE] = 0               # PCC

    perm = rng.permutation(np.arange(1, N))
    gen_nodes = perm[:ng]
    con_nodes = perm[ng:ng + nc]

    if ng > 0:
        P[gen_nodes] = PMAX / ng
        node_types[gen_nodes] = 1
    if nc > 0:
        P[con_nodes] = -PMAX / nc
        node_types[con_nodes] = 2

    return P, node_types


def assign_powers(n, n_plus, n_minus, Pmax=PMAX, seed=0):
    """功率分配 (无 PCC，与 figure1.py 一致)。所有 n 节点参与。

    Parameters
    ----------
    n : int       节点总数
    n_plus : int  发电节点数
    n_minus : int 消费节点数
    Pmax : float  归一化最大功率
    seed : int    随机种子

    Returns
    -------
    P : ndarray (n,)
    """
    rng = np.random.default_rng(seed)
    P = np.zeros(n)
    if n_plus == 0 or n_minus == 0:
        return P
    indices = rng.permutation(n)
    P[indices[:n_plus]] = Pmax / n_plus
    P[indices[n_plus:n_plus + n_minus]] = -Pmax / n_minus
    return P
