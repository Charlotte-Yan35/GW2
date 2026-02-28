"""
plot_ratio_simplex_kappa.py — 2x2 ternary simplex heatmaps of κ_c
for WS, RGG, SBM, and Core-Periphery topologies.

Uses barycentric-to-Cartesian transform (no external ternary library).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib import ticker
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

FAMILIES = ["WS", "RGG", "SBM", "CP"]
FAMILY_LABELS = {"WS": "Watts-Strogatz", "RGG": "Random Geometric",
                 "SBM": "Stochastic Block", "CP": "Core-Periphery"}


# ====================================================================
# Barycentric → Cartesian
# ====================================================================

def bary_to_cart(rg, rc, rp):
    """Convert barycentric (rg, rc, rp) to 2D Cartesian for an equilateral triangle.

    Vertices:
        Generator (rg=1) → top          (0.5, sqrt(3)/2)
        Consumer  (rc=1) → bottom-left  (0, 0)
        Passive   (rp=1) → bottom-right (1, 0)
    """
    x = 0.5 * rg + rp
    y = (np.sqrt(3) / 2) * rg
    return x, y


# ====================================================================
# Load aggregated data
# ====================================================================

def load_agg_data(agg_path: Path) -> dict:
    """Load agg_results.csv → dict keyed by family → (rg, rc, rp, kc_mean) arrays."""
    data = {f: {"rg": [], "rc": [], "rp": [], "kc_mean": []}
            for f in FAMILIES}

    with open(agg_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam = row["family"]
            if fam not in data:
                continue
            kc = row["kappa_c_mean"]
            if kc == "NaN":
                continue
            data[fam]["rg"].append(float(row["rg"]))
            data[fam]["rc"].append(float(row["rc"]))
            data[fam]["rp"].append(float(row["rp"]))
            data[fam]["kc_mean"].append(float(kc))

    # Convert to numpy arrays
    for fam in FAMILIES:
        for key in data[fam]:
            data[fam][key] = np.array(data[fam][key])

    return data


# ====================================================================
# Draw simplex outline + axis labels
# ====================================================================

def draw_simplex_frame(ax):
    """Draw equilateral triangle outline and vertex labels."""
    # Triangle vertices
    verts = np.array([
        [0.5, np.sqrt(3) / 2],  # Generator (top)
        [0.0, 0.0],              # Consumer (bottom-left)
        [1.0, 0.0],              # Passive (bottom-right)
    ])
    triangle = plt.Polygon(verts, fill=False, edgecolor='k', linewidth=1.2)
    ax.add_patch(triangle)

    # Vertex labels
    offset = 0.06
    ax.text(verts[0, 0], verts[0, 1] + offset, "Generator",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(verts[1, 0] - offset, verts[1, 1] - offset, "Consumer",
            ha='center', va='top', fontsize=9, fontweight='bold')
    ax.text(verts[2, 0] + offset, verts[2, 1] - offset, "Passive",
            ha='center', va='top', fontsize=9, fontweight='bold')

    # Grid lines (optional — 10% increments)
    for frac in np.arange(0.1, 1.0, 0.1):
        # Lines of constant rg
        x0, y0 = bary_to_cart(frac, 1 - frac, 0)
        x1, y1 = bary_to_cart(frac, 0, 1 - frac)
        ax.plot([x0, x1], [y0, y1], 'k-', lw=0.3, alpha=0.3)
        # Lines of constant rc
        x0, y0 = bary_to_cart(0, frac, 1 - frac)
        x1, y1 = bary_to_cart(1 - frac, frac, 0)
        ax.plot([x0, x1], [y0, y1], 'k-', lw=0.3, alpha=0.3)
        # Lines of constant rp
        x0, y0 = bary_to_cart(1 - frac, 0, frac)
        x1, y1 = bary_to_cart(0, 1 - frac, frac)
        ax.plot([x0, x1], [y0, y1], 'k-', lw=0.3, alpha=0.3)

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')


# ====================================================================
# Main plotting
# ====================================================================

def plot_simplex_heatmaps(agg_path: Path = None):
    """Create 2x2 ternary simplex heatmaps of log10(κ_c)."""
    if agg_path is None:
        agg_path = CACHE_DIR / "agg_results.csv"

    data = load_agg_data(agg_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()

    # Shared color scale: compute global min/max of log10(kc)
    all_kc = []
    for fam in FAMILIES:
        kc = data[fam]["kc_mean"]
        if len(kc) > 0:
            all_kc.extend(kc[kc > 0].tolist())

    if not all_kc:
        print("No valid κ_c data found. Exiting.")
        return

    log_kc_all = np.log10(np.array(all_kc) + 1e-6)
    vmin = np.percentile(log_kc_all, 2)
    vmax = np.percentile(log_kc_all, 98)

    for idx, fam in enumerate(FAMILIES):
        ax = axes[idx]
        draw_simplex_frame(ax)
        ax.set_title(FAMILY_LABELS[fam], fontsize=13, fontweight='bold', pad=10)

        rg = data[fam]["rg"]
        rc = data[fam]["rc"]
        rp = data[fam]["rp"]
        kc = data[fam]["kc_mean"]

        if len(kc) < 3:
            ax.text(0.5, 0.4, "Insufficient data", ha='center', fontsize=10,
                    color='red')
            continue

        # Convert to Cartesian
        x, y = bary_to_cart(rg, rc, rp)
        z = np.log10(kc + 1e-6)

        # Triangulate and interpolate
        tri = Triangulation(x, y)
        interp = LinearTriInterpolator(tri, z)

        # Fine grid for smooth rendering
        xi = np.linspace(-0.05, 1.05, 200)
        yi = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, 200)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = interp(Xi, Yi)

        # Mask points outside the triangle
        # Point (x,y) is inside if rg,rc,rp >= 0
        # rg = 2y/sqrt(3), rp = x - y/sqrt(3), rc = 1 - rg - rp
        Rg = 2 * Yi / np.sqrt(3)
        Rp = Xi - Yi / np.sqrt(3)
        Rc = 1 - Rg - Rp
        mask = (Rg < -0.01) | (Rc < -0.01) | (Rp < -0.01)
        Zi = np.ma.masked_where(mask, Zi)

        levels = np.linspace(vmin, vmax, 20)
        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='viridis',
                         extend='both')

        # Scatter original data points
        ax.scatter(x, y, c='white', s=8, edgecolors='k', linewidths=0.3,
                   zorder=5, alpha=0.6)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'$\log_{10}(\kappa_c)$', fontsize=12)

    fig.suptitle(r'Critical Coupling $\kappa_c$ across Topology Families',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.05,
                        wspace=0.15, hspace=0.2)

    # Save
    for ext in ['png', 'pdf']:
        out = FIGURES_DIR / f"ratio_simplex_kappa_2x2.{ext}"
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_simplex_heatmaps()
