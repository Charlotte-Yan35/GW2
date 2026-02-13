import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from figure1 import plot_panel_c, N

cache = np.load(Path(__file__).parent / "cache/v4_panel_c_n50_k4_q0.0_r100_s2.npz")
fig, ax = plt.subplots(figsize=(5, 4.5))
plot_panel_c(ax, cache['configs'], cache['kappa_c_vals'], n=N)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "panel_c_preview.png", dpi=200)
print("Saved panel_c_preview.png")
