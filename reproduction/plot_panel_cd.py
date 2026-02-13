import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from figure1 import plot_panel_c, plot_panel_d, N

cache_dir = Path(__file__).parent / "cache"

# 加载缓存
c_data = np.load(cache_dir / "v4_panel_c_n50_k4_q0.0_r100_s2.npz")
d_data = np.load(cache_dir / "v4_panel_d_n50_k4_np15_r100.npz", allow_pickle=True)

fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(10, 4.5))

plot_panel_c(ax_c, c_data['configs'], c_data['kappa_c_vals'], n=N)
plot_panel_d(ax_d, d_data['q_values'], d_data['n_minus_range'],
             d_data['kappa_c_mean'], d_data['kappa_c_std'])

fig.tight_layout()
out = Path(__file__).parent / "figure1_CD.png"
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved {out}")
