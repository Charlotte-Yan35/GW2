"""
plot_topology_structures.py — 2x2 panel showing one representative network
per topology family (WS, RGG, SBM, Core-Periphery).

Custom layouts:
  WS  — circular (ring + shortcuts)
  RGG — true geometric coordinates
  SBM — block-separated (4 communities in quadrants)
  CP  — radial (core centre, periphery outer ring)

PCC node is red and larger; household nodes are blue.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from compute_ratio_simplex_kappa import (
    generate_ws, generate_rgg, generate_sbm, generate_cp,
    household_mean_degree as _hmd,
    _ensure_connected,
    PCC_NODE, N, K_TARGET,
)
from scipy.spatial.distance import pdist, squareform

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 123


# ====================================================================
# Display-only CP generator (more balanced edge distribution for visual)
# ====================================================================

def _generate_cp_display(seed: int, n: int = N, k_target: int = K_TARGET,
                         f_core: float = 0.3) -> nx.Graph:
    """CP network tuned for visual clarity: larger core fraction (0.3),
    less extreme probability ratios (1 : 0.5 : 0.15).
    Only used for the structure figure — does not affect experiments.

    Uses rejection sampling: binary-search p_cc, then retry with different
    seeds until household mean degree rounds to exactly k_target.
    """
    n_hh = n - 1
    n_core = max(2, int(f_core * n_hh))
    n_periph = n_hh - n_core
    sizes = [n_core, n_periph]

    # Outer loop: try different seeds to beat SBM randomness
    for attempt in range(50):
        rng = np.random.default_rng(seed + attempt)
        sbm_seed = int(rng.integers(2**31))

        p_lo, p_hi = 0.001, 1.0
        best_G = None
        best_diff = float('inf')

        for _ in range(60):
            p_cc = (p_lo + p_hi) / 2
            p_cp = p_cc * 0.5
            p_pp = p_cc * 0.15

            probs = [[p_cc, p_cp], [p_cp, p_pp]]
            G_hh = nx.stochastic_block_model(sizes, probs, seed=sbm_seed)
            G_hh.remove_edges_from(nx.selfloop_edges(G_hh))
            mapping = {i: i + 1 for i in range(n_hh)}
            G_hh = nx.relabel_nodes(G_hh, mapping)

            md = _hmd(G_hh)
            diff = md - k_target
            if abs(diff) < best_diff:
                best_diff = abs(diff)
                best_G = G_hh.copy()
            if diff < 0:
                p_lo = p_cc
            else:
                p_hi = p_cc
            if best_diff < 0.01:
                break

        if best_diff < 0.05:
            break

    G = best_G
    G.graph["core_nodes"] = list(range(1, n_core + 1))
    G.add_node(PCC_NODE)
    G.add_edge(PCC_NODE, 1)
    G.add_edge(PCC_NODE, n_core + 1)
    G = _ensure_connected(G)
    return G


# ====================================================================
# Layout helpers
# ====================================================================

def get_pos_ws(G: nx.Graph) -> dict:
    """Circular layout for household nodes; PCC below the ring."""
    household = sorted(n for n in G.nodes() if n != PCC_NODE)
    circ = nx.circular_layout(G.subgraph(household), scale=1.0)
    pos = {n: circ[n] for n in household}
    pos[PCC_NODE] = np.array([0.0, -1.25])
    return pos


def get_pos_rgg(G: nx.Graph) -> dict:
    """Use stored geometric positions from G.graph['pos'].

    Rescales [0,1]^2 coordinates to [-1,1]^2 for consistent axis limits.
    PCC placed at centroid of household positions.
    """
    if 'pos' not in G.graph:
        raise ValueError(
            "RGG graph missing G.graph['pos']. "
            "Store node positions at generation time in generate_rgg()."
        )
    raw = G.graph['pos']
    # Rescale from [0,1] to [-1.5, 1.5] for visual spread
    pos = {}
    hh_xs, hh_ys = [], []
    for node in G.nodes():
        if node == PCC_NODE:
            continue
        x, y = raw[node]
        px, py = (x - 0.5) * 3.0, (y - 0.5) * 3.0
        pos[node] = np.array([px, py])
        hh_xs.append(px)
        hh_ys.append(py)
    # PCC at centroid
    pos[PCC_NODE] = np.array([np.mean(hh_xs), np.mean(hh_ys)])
    return pos


def get_pos_sbm(G: nx.Graph) -> dict:
    """Block-separated layout: 4 communities placed in quadrants.

    Requires G.nodes[i]["block"] for household nodes.
    """
    # Check metadata
    household = [n for n in G.nodes() if n != PCC_NODE]
    for n in household:
        if "block" not in G.nodes[n]:
            raise ValueError(
                f"SBM graph missing 'block' attribute on node {n}. "
                "Store block assignment in generate_sbm() via "
                "G.nodes[i]['block'] = b."
            )

    block_centers = {
        0: np.array([-1.0,  1.0]),
        1: np.array([ 1.0,  1.0]),
        2: np.array([-1.0, -1.0]),
        3: np.array([ 1.0, -1.0]),
    }

    # Group household nodes by block
    blocks = {}
    for n in household:
        b = G.nodes[n]["block"]
        blocks.setdefault(b, []).append(n)

    pos = {}
    for b, nodes in blocks.items():
        sub = G.subgraph(nodes)
        local = nx.spring_layout(sub, seed=SEED, scale=0.35)
        center = block_centers.get(b, np.array([0.0, 0.0]))
        for n in nodes:
            pos[n] = local[n] + center

    pos[PCC_NODE] = np.array([0.0, -1.65])
    return pos


def get_pos_cp(G: nx.Graph) -> dict:
    """Radial layout: core nodes on inner circle, periphery on outer circle.

    Requires G.graph["core_nodes"].
    """
    if "core_nodes" not in G.graph:
        raise ValueError(
            "CP graph missing G.graph['core_nodes']. "
            "Store core node list in generate_cp() via "
            "G.graph['core_nodes'] = list(range(1, n_core+1))."
        )

    core_set = set(G.graph["core_nodes"])
    household = sorted(n for n in G.nodes() if n != PCC_NODE)
    core_nodes = sorted(n for n in household if n in core_set)
    peri_nodes = sorted(n for n in household if n not in core_set)

    r_core = 0.55
    r_peri = 1.55
    pos = {}

    for i, n in enumerate(core_nodes):
        angle = 2 * np.pi * i / len(core_nodes) - np.pi / 2
        pos[n] = np.array([r_core * np.cos(angle), r_core * np.sin(angle)])

    for i, n in enumerate(peri_nodes):
        angle = 2 * np.pi * i / len(peri_nodes) - np.pi / 2
        pos[n] = np.array([r_peri * np.cos(angle), r_peri * np.sin(angle)])

    pos[PCC_NODE] = np.array([0.0, 0.0])
    return pos


# ====================================================================
# Main figure
# ====================================================================

FAMILIES = {
    "Watts-Strogatz":    (generate_ws,           get_pos_ws),
    "Random Geometric":  (generate_rgg,          get_pos_rgg),
    "Stochastic Block":  (generate_sbm,          get_pos_sbm),
    "Core-Periphery":    (_generate_cp_display,  get_pos_cp),
}


def household_mean_degree(G):
    return _hmd(G)


def plot_topology_structures():
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()

    for idx, (name, (gen_func, pos_func)) in enumerate(FAMILIES.items()):
        ax = axes[idx]
        G = gen_func(seed=SEED)
        pos = pos_func(G)

        # Node properties
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == PCC_NODE:
                node_colors.append('#e74c3c')
                node_sizes.append(200)
            else:
                node_colors.append('#3498db')
                node_sizes.append(40)

        # Draw nodes first (behind edges for WS so ring edges are visible)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, edgecolors='k',
                               linewidths=0.5)
        # Draw edges on top so short ring edges in WS are not hidden
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.8,
                               edge_color='gray')
        # Label PCC
        nx.draw_networkx_labels(G, pos, labels={PCC_NODE: "PCC"}, ax=ax,
                                font_size=7, font_weight='bold',
                                font_color='white')

        ax.set_title(name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Stats annotation
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        avg_deg = household_mean_degree(G)
        ax.text(0.02, 0.02,
                f"N={n_nodes}, E={n_edges}, <k>={avg_deg:.1f}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                          alpha=0.8))

    fig.suptitle("Representative Network Topologies (50 nodes, target <k>=4)",
                 fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                        wspace=0.1, hspace=0.15)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=12, label='PCC (node 0)', markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=8, label='Household nodes', markeredgecolor='k'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, frameon=True, fancybox=True)

    out = FIGURES_DIR / "topology_structures_2x2.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_topology_structures()
