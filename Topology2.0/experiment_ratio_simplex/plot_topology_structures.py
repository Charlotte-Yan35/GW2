"""
plot_topology_structures.py — 2x2 panel showing one representative network
per topology family (WS, RGG, SBM, Core-Periphery).

Custom layouts:
  WS  — circular (ring + shortcuts)
  RGG — true geometric coordinates
  SBM — block-separated (4 communities in quadrants)
  CP  — radial (core centre, periphery outer ring)

Nodes coloured by structural role within each topology.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from pathlib import Path

from compute_ratio_simplex_kappa import (
    generate_ws, generate_rgg, generate_sbm, generate_cp,
    household_mean_degree as _hmd,
    _ensure_connected, _attach_pcc,
    PCC_NODE, N, K_TARGET,
)
from scipy.spatial.distance import pdist, squareform

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 123
SEED_SBM = 122  # chosen so all 4 blocks are internally connected

# ── Colour palette ───────────────────────────────────────────────
PCC_COLOR = '#d62728'          # red
PCC_SIZE = 220

# Per-topology household palettes
PAL_WS   = '#1f77b4'          # uniform blue
PAL_RGG  = '#2ca02c'          # green (spatial)
PAL_SBM  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']  # 4 block colours
PAL_CP_CORE = '#e377c2'       # pink for core
PAL_CP_PERI = '#17becf'       # cyan for periphery

EDGE_COLOR_DEFAULT = '#888888'
EDGE_COLOR_SHORTCUT = '#d62728'   # WS shortcuts in red
EDGE_COLOR_INTER = '#aaaaaa'      # SBM inter-block edges lighter
EDGE_COLOR_PCC = '#FFD700'        # gold/yellow for PCC links (all panels)

HH_SIZE = 45                  # household node size

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 250,
})


# ====================================================================
# Display-only SBM generator (PCC connects to all 4 blocks)
# ====================================================================

def _generate_sbm_display(seed: int, n: int = N, k_target: int = K_TARGET,
                          m: int = 4, eta: float = 8.0) -> nx.Graph:
    """SBM network for display: PCC connects to one node in each block.
    Uses rejection sampling to find a seed with all blocks internally connected.
    """
    n_hh = n - 1
    block_sizes = [n_hh // m] * m
    for i in range(n_hh % m):
        block_sizes[i] += 1

    for attempt in range(50):
        rng = np.random.default_rng(seed + attempt)
        sbm_seed = int(rng.integers(2**31))
        s_lo, s_hi = 0.001, 1.0
        best_G, best_diff = None, float('inf')

        for _ in range(60):
            s = (s_lo + s_hi) / 2
            p_in, p_out = min(s, 1.0), min(s / eta, 1.0)
            probs = [[p_out] * m for _ in range(m)]
            for i in range(m):
                probs[i][i] = p_in
            G_hh = nx.stochastic_block_model(block_sizes, probs, seed=sbm_seed)
            G_hh.remove_edges_from(nx.selfloop_edges(G_hh))
            mapping = {i: i + 1 for i in range(n_hh)}
            G_hh = nx.relabel_nodes(G_hh, mapping)
            md = _hmd(G_hh)
            diff = md - k_target
            if abs(diff) < best_diff:
                best_diff, best_G = abs(diff), G_hh.copy()
            if diff < 0:
                s_lo = s
            else:
                s_hi = s
            if best_diff < 0.01:
                break

        # Check all blocks internally connected
        G = best_G
        offset = 1
        all_connected = True
        for b in range(m):
            bs = block_sizes[b]
            nodes_b = list(range(offset, offset + bs))
            for nd in nodes_b:
                G.nodes[nd]["block"] = b
            sub = G.subgraph(nodes_b)
            if not nx.is_connected(sub):
                all_connected = False
                break
            offset += bs

        if all_connected and best_diff < 0.1:
            break

    # SBM-specific PCC attachment: one node per block
    G.add_node(PCC_NODE)
    rng = np.random.default_rng(seed + 999999)
    offset = 1
    for b in range(m):
        bs = block_sizes[b]
        nodes_b = list(range(offset, offset + bs))
        chosen = rng.choice(nodes_b)
        G.add_edge(PCC_NODE, int(chosen))
        offset += bs
    G = _ensure_connected(G)
    return G


# ====================================================================
# Display-only CP generator (more balanced edge distribution)
# ====================================================================

def _generate_cp_display(seed: int, n: int = N, k_target: int = K_TARGET,
                         f_core: float = 0.3) -> nx.Graph:
    """CP network tuned for visual clarity.
    Only used for the structure figure — does not affect experiments.
    """
    n_hh = n - 1
    n_core = max(2, int(f_core * n_hh))
    n_periph = n_hh - n_core
    sizes = [n_core, n_periph]

    for attempt in range(50):
        rng = np.random.default_rng(seed + attempt)
        sbm_seed = int(rng.integers(2**31))
        p_lo, p_hi = 0.001, 1.0
        best_G, best_diff = None, float('inf')

        for _ in range(60):
            p_cc = (p_lo + p_hi) / 2
            p_cp, p_pp = p_cc * 0.5, p_cc * 0.15
            probs = [[p_cc, p_cp], [p_cp, p_pp]]
            G_hh = nx.stochastic_block_model(sizes, probs, seed=sbm_seed)
            G_hh.remove_edges_from(nx.selfloop_edges(G_hh))
            mapping = {i: i + 1 for i in range(n_hh)}
            G_hh = nx.relabel_nodes(G_hh, mapping)
            md = _hmd(G_hh)
            diff = md - k_target
            if abs(diff) < best_diff:
                best_diff, best_G = abs(diff), G_hh.copy()
            if diff < 0:
                p_lo = p_cc
            else:
                p_hi = p_cc
            if best_diff < 0.01:
                break
        if best_diff < 0.05:
            break

    G = best_G
    core_nodes_list = list(range(1, n_core + 1))
    G.graph["core_nodes"] = core_nodes_list
    # CP-specific PCC attachment: only connect to Core nodes
    G.add_node(PCC_NODE)
    rng = np.random.default_rng(seed + 999999)
    chosen = rng.choice(core_nodes_list, size=min(4, len(core_nodes_list)), replace=False)
    for v in chosen:
        G.add_edge(PCC_NODE, int(v))
    G = _ensure_connected(G)
    return G


# ====================================================================
# Layout helpers
# ====================================================================

def get_pos_ws(G):
    household = sorted(n for n in G.nodes() if n != PCC_NODE)
    circ = nx.circular_layout(G.subgraph(household), scale=1.0)
    pos = {n: circ[n] for n in household}
    pos[PCC_NODE] = np.array([0.0, 0.0])
    return pos


def get_pos_rgg(G):
    if 'pos' not in G.graph:
        raise ValueError("RGG graph missing G.graph['pos'].")
    raw = G.graph['pos']
    pos = {}
    hh_xs, hh_ys = [], []
    for node in G.nodes():
        if node == PCC_NODE:
            continue
        x, y = raw[node]
        px, py = (x - 0.5) * 3.0, (y - 0.5) * 3.0
        pos[node] = np.array([px, py])
        hh_xs.append(px); hh_ys.append(py)
    pos[PCC_NODE] = np.array([np.mean(hh_xs), np.mean(hh_ys)])
    return pos


def get_pos_sbm(G):
    household = [n for n in G.nodes() if n != PCC_NODE]
    for n in household:
        if "block" not in G.nodes[n]:
            raise ValueError(f"SBM node {n} missing 'block' attribute.")
    block_centers = {
        0: np.array([-1.0,  1.0]), 1: np.array([ 1.0,  1.0]),
        2: np.array([-1.0, -1.0]), 3: np.array([ 1.0, -1.0]),
    }
    blocks = {}
    for n in household:
        blocks.setdefault(G.nodes[n]["block"], []).append(n)
    pos = {}
    for b, nodes in blocks.items():
        sub = G.subgraph(nodes)
        local = nx.spring_layout(sub, seed=SEED, scale=0.55)
        center = block_centers.get(b, np.array([0.0, 0.0]))
        for n in nodes:
            pos[n] = local[n] + center
    pos[PCC_NODE] = np.array([0.0, 0.0])
    return pos


def get_pos_cp(G):
    if "core_nodes" not in G.graph:
        raise ValueError("CP graph missing G.graph['core_nodes'].")
    core_set = set(G.graph["core_nodes"])
    household = sorted(n for n in G.nodes() if n != PCC_NODE)
    core_nodes = sorted(n for n in household if n in core_set)
    peri_nodes = sorted(n for n in household if n not in core_set)
    r_core, r_peri = 0.55, 1.55
    pos = {}
    for i, n in enumerate(core_nodes):
        a = 2 * np.pi * i / len(core_nodes) - np.pi / 2
        pos[n] = np.array([r_core * np.cos(a), r_core * np.sin(a)])
    for i, n in enumerate(peri_nodes):
        a = 2 * np.pi * i / len(peri_nodes) - np.pi / 2
        pos[n] = np.array([r_peri * np.cos(a), r_peri * np.sin(a)])
    pos[PCC_NODE] = np.array([0.0, 0.0])
    return pos


# ====================================================================
# Per-topology drawing helpers
# ====================================================================

def _draw_ws(G, pos, ax):
    """WS: ring edges grey, shortcut edges red to highlight small-world property."""
    household = sorted(n for n in G.nodes() if n != PCC_NODE)
    ring_set = set()
    for i, n in enumerate(household):
        for d in [1, 2]:  # k=4 → 2 neighbours each side
            nb = household[(i + d) % len(household)]
            ring_set.add((min(n, nb), max(n, nb)))

    ring_edges, shortcut_edges, pcc_edges = [], [], []
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if u == PCC_NODE or v == PCC_NODE:
            pcc_edges.append((u, v))
        elif key in ring_set:
            ring_edges.append((u, v))
        else:
            shortcut_edges.append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=ring_edges, ax=ax,
                           alpha=0.35, width=0.7, edge_color=EDGE_COLOR_DEFAULT)
    nx.draw_networkx_edges(G, pos, edgelist=shortcut_edges, ax=ax,
                           alpha=0.6, width=1.0, edge_color=EDGE_COLOR_SHORTCUT,
                           style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=pcc_edges, ax=ax,
                           alpha=0.8, width=1.8, edge_color=EDGE_COLOR_PCC)

    # Nodes: all household same colour
    hh_nodes = [n for n in G.nodes() if n != PCC_NODE]
    nx.draw_networkx_nodes(G, pos, nodelist=hh_nodes, ax=ax,
                           node_color=PAL_WS, node_size=HH_SIZE,
                           edgecolors='white', linewidths=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=[PCC_NODE], ax=ax,
                           node_color=PCC_COLOR, node_size=PCC_SIZE,
                           edgecolors='k', linewidths=1.0)

    # Legend entries for this panel
    return [
        Line2D([0], [0], color=EDGE_COLOR_DEFAULT, lw=0.8, label='Ring edge'),
        Line2D([0], [0], color=EDGE_COLOR_SHORTCUT, lw=1.0, ls='--', label='Shortcut'),
        Line2D([0], [0], color=EDGE_COLOR_PCC, lw=2.0, label='PCC link'),
    ]


def _draw_rgg(G, pos, ax):
    """RGG: edges grey, nodes green, degree mapped to size. PCC edges yellow."""
    hh_nodes = [n for n in G.nodes() if n != PCC_NODE]
    degs = dict(G.degree())
    sizes = [HH_SIZE + (degs[n] - 2) * 12 for n in hh_nodes]

    hh_edges = [(u, v) for u, v in G.edges() if u != PCC_NODE and v != PCC_NODE]
    pcc_edges = [(u, v) for u, v in G.edges() if u == PCC_NODE or v == PCC_NODE]

    nx.draw_networkx_edges(G, pos, edgelist=hh_edges, ax=ax, alpha=0.3,
                           width=0.6, edge_color=EDGE_COLOR_DEFAULT)
    nx.draw_networkx_edges(G, pos, edgelist=pcc_edges, ax=ax,
                           alpha=0.8, width=1.8, edge_color=EDGE_COLOR_PCC)
    nx.draw_networkx_nodes(G, pos, nodelist=hh_nodes, ax=ax,
                           node_color=PAL_RGG, node_size=sizes,
                           edgecolors='white', linewidths=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=[PCC_NODE], ax=ax,
                           node_color=PCC_COLOR, node_size=PCC_SIZE,
                           edgecolors='k', linewidths=1.0)
    return [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PAL_RGG,
               markersize=5, label='Size $\\propto$ degree', markeredgecolor='w'),
        Line2D([0], [0], color=EDGE_COLOR_PCC, lw=2.0, label='PCC link'),
    ]


def _draw_sbm(G, pos, ax):
    """SBM: nodes coloured by block, intra-block edges coloured, inter-block lighter."""
    hh_nodes = [n for n in G.nodes() if n != PCC_NODE]
    blocks = {}
    for n in hh_nodes:
        blocks.setdefault(G.nodes[n]["block"], []).append(n)

    # Classify edges
    intra_edges, inter_edges, pcc_edges = [], [], []
    for u, v in G.edges():
        if u == PCC_NODE or v == PCC_NODE:
            pcc_edges.append((u, v))
        elif G.nodes[u].get("block") == G.nodes[v].get("block"):
            intra_edges.append((u, v))
        else:
            inter_edges.append((u, v))

    # Draw nodes FIRST (underneath)
    for b, nodes in sorted(blocks.items()):
        col = PAL_SBM[b % len(PAL_SBM)]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                               node_color=col, node_size=HH_SIZE,
                               edgecolors='white', linewidths=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=[PCC_NODE], ax=ax,
                           node_color=PCC_COLOR, node_size=PCC_SIZE,
                           edgecolors='k', linewidths=1.0)

    # Draw edges ON TOP so intra-block edges are visible
    nx.draw_networkx_edges(G, pos, edgelist=inter_edges, ax=ax,
                           alpha=0.45, width=0.8, edge_color=EDGE_COLOR_INTER)
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, ax=ax,
                           alpha=0.55, width=1.0, edge_color=EDGE_COLOR_DEFAULT)
    nx.draw_networkx_edges(G, pos, edgelist=pcc_edges, ax=ax,
                           alpha=0.8, width=1.8, edge_color=EDGE_COLOR_PCC)
    return [
        mpatches.Patch(color=PAL_SBM[i], label=f'Block {i+1}')
        for i in range(min(4, len(blocks)))
    ] + [
        Line2D([0], [0], color=EDGE_COLOR_PCC, lw=2.0, label='PCC link'),
    ]


def _draw_cp(G, pos, ax):
    """CP: core pink, periphery cyan, core-core edges darker."""
    core_set = set(G.graph["core_nodes"])
    hh_nodes = [n for n in G.nodes() if n != PCC_NODE]
    core_nodes = [n for n in hh_nodes if n in core_set]
    peri_nodes = [n for n in hh_nodes if n not in core_set]

    # Classify edges
    cc_edges, cp_edges, pp_edges, pcc_edges = [], [], [], []
    for u, v in G.edges():
        if u == PCC_NODE or v == PCC_NODE:
            pcc_edges.append((u, v))
        elif u in core_set and v in core_set:
            cc_edges.append((u, v))
        elif u in core_set or v in core_set:
            cp_edges.append((u, v))
        else:
            pp_edges.append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=pp_edges, ax=ax,
                           alpha=0.15, width=0.4, edge_color=EDGE_COLOR_INTER)
    nx.draw_networkx_edges(G, pos, edgelist=cp_edges, ax=ax,
                           alpha=0.25, width=0.6, edge_color=EDGE_COLOR_DEFAULT)
    nx.draw_networkx_edges(G, pos, edgelist=cc_edges, ax=ax,
                           alpha=0.5, width=1.0, edge_color='#555555')
    nx.draw_networkx_edges(G, pos, edgelist=pcc_edges, ax=ax,
                           alpha=0.8, width=1.8, edge_color=EDGE_COLOR_PCC)

    nx.draw_networkx_nodes(G, pos, nodelist=peri_nodes, ax=ax,
                           node_color=PAL_CP_PERI, node_size=HH_SIZE,
                           edgecolors='white', linewidths=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, ax=ax,
                           node_color=PAL_CP_CORE, node_size=HH_SIZE + 30,
                           edgecolors='white', linewidths=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=[PCC_NODE], ax=ax,
                           node_color=PCC_COLOR, node_size=PCC_SIZE,
                           edgecolors='k', linewidths=1.0)
    return [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PAL_CP_CORE,
               markersize=8, label='Core', markeredgecolor='w'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PAL_CP_PERI,
               markersize=6, label='Periphery', markeredgecolor='w'),
        Line2D([0], [0], color=EDGE_COLOR_PCC, lw=2.0, label='PCC link'),
    ]


# ====================================================================
# Main figure
# ====================================================================

FAMILIES = [
    ("(a)  Watts-Strogatz",    generate_ws,           get_pos_ws,  _draw_ws,  SEED),
    ("(b)  Random Geometric",  generate_rgg,          get_pos_rgg, _draw_rgg, SEED),
    ("(c)  Stochastic Block",  _generate_sbm_display,  get_pos_sbm, _draw_sbm, SEED_SBM),
    ("(d)  Core-Periphery",    _generate_cp_display,  get_pos_cp,  _draw_cp,  SEED),
]


def plot_topology_structures():
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    axes = axes.flatten()

    for idx, (title, gen_func, pos_func, draw_func, seed) in enumerate(FAMILIES):
        ax = axes[idx]
        G = gen_func(seed=seed)
        pos = pos_func(G)

        # Topology-specific drawing (edges, nodes, colours)
        legend_entries = draw_func(G, pos, ax)

        # PCC label
        nx.draw_networkx_labels(G, pos, labels={PCC_NODE: "PCC"}, ax=ax,
                                font_size=6.5, font_weight='bold',
                                font_color='white')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=8, loc='left')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2.1, 2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Stats box (no <k>)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        stats = f"$N={n_nodes}$,  $E={n_edges}$"
        ax.text(0.5, 0.01, stats, transform=ax.transAxes, fontsize=8.5,
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                          edgecolor='#cccccc', alpha=0.9))

        # Per-panel legend (structural roles)
        pcc_handle = Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=PCC_COLOR, markersize=9,
                            label='PCC', markeredgecolor='k', markeredgewidth=0.7)
        handles = [pcc_handle] + legend_entries
        ax.legend(handles=handles, loc='lower left', fontsize=7,
                  framealpha=0.85, edgecolor='#cccccc', fancybox=True,
                  handlelength=1.5, handletextpad=0.4, borderpad=0.4)

    fig.suptitle(
        "Representative Network Topologies  ($N=50$)",
        fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03,
                        wspace=0.08, hspace=0.12)

    out = FIGURES_DIR / "topology_structures_2x2.png"
    fig.savefig(out, dpi=250, bbox_inches='tight')
    print(f"Saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_topology_structures()
