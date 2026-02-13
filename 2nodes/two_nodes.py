import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# 2-node reduced model (Delta theta, omega)
#   d(Delta)/dt = omega
#   domega/dt   = 2P - gamma*omega - 2*kappa*sin(Delta)
# -----------------------------

# 基于 S14和S15, 其中 Delta = theta_1 - theta_2
def swing_2node_rhs(t, y, P, kappa, gamma):
    Delta, omega = y
    dDelta = omega
    domega = 2.0 * P - gamma * omega - 2.0 * kappa * np.sin(Delta)
    return [dDelta, domega]

def fixed_points(P, kappa):
    """
    Return the two fixed points (Delta*, omega*=0) if they exist (|P| <= kappa).
    """
    if abs(P) > kappa:
        return None
    a = np.arcsin(P / kappa)  # principal value in [-pi/2, pi/2]
    fp1 = (a, 0.0)
    fp2 = (np.pi - a, 0.0)
    return fp1, fp2

def wrap_to_pi(x):
    """
    Wrap angle to (-pi, pi] for nicer phase plots. 将相位差包裹到 (-π, π] 范围内。
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi

# -----------------------------
# Parameters (edit these)
# -----------------------------
P = 0.5         # power injection magnitude
kappa = 1.0     # coupling strength
gamma = 1.0     # damping

# Initial condition: [Delta(0), omega(0)]
y0 = [1.57, 0.0]

t0, t1 = 0.0, 60.0
t_eval = np.linspace(t0, t1, 4000)

# -----------------------------
# Solve ODE
# -----------------------------
sol = solve_ivp(
    fun=lambda t, y: swing_2node_rhs(t, y, P=P, kappa=kappa, gamma=gamma),
    t_span=(t0, t1),
    y0=y0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-12,
    method="RK45"
)

if not sol.success:
    raise RuntimeError(sol.message)

t = sol.t
Delta = sol.y[0]
omega = sol.y[1]

# Optional: wrap Delta for visualization (does not change dynamics)
Delta_wrapped = wrap_to_pi(Delta) #用来画相位图

# -----------------------------
# Fixed points (if exist)
# -----------------------------
fps = fixed_points(P, kappa)

print("Parameters:")
print(f"  P = {P}, kappa = {kappa}, gamma = {gamma}")
if fps is None:
    print("No fixed points (|P| > kappa) -> likely desynchronization / running solution.")
else:
    (D1, w1), (D2, w2) = fps
    print("Fixed points (Delta*, omega*):")
    print(f"  FP1 = ({D1:.6f}, {w1:.6f})")
    print(f"  FP2 = ({D2:.6f}, {w2:.6f})")

# -----------------------------
# Plot: time series + phase portrait
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# (1) Time series
ax = axes[0]
ax.plot(t, Delta, label=r"$\Delta\theta(t)$")
ax.plot(t, omega, label=r"$\omega(t)$")
ax.set_xlabel("t")
ax.set_ylabel("state")
ax.set_title("2-node swing model: time series")
ax.legend()

# (2) Phase portrait
ax = axes[1]
ax.plot(Delta_wrapped, omega, linewidth=1.0)
ax.set_xlabel(r"wrapped $\Delta\theta$")
ax.set_ylabel(r"$\omega$")
ax.set_title(r"Phase portrait ($\omega$ vs $\Delta\theta$)")

# Mark fixed points on phase portrait (also wrapped)
if fps is not None:
    for (D, w) in fps:
        ax.scatter([wrap_to_pi(D)], [w], s=40)
  
  
# -----------------------------
# Plot: Figure S1 (A) + (B)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

# =========
# Panel A: bifurcation diagram (Delta* vs P/kappa)
# =========
ax = axes[0]
s = np.linspace(0.0, 1.0, 600)  # s = P/kappa, assume P/kappa in [0,1]
Delta_stable = np.arcsin(s)
Delta_saddle = np.pi - np.arcsin(s)

ax.plot(s, Delta_saddle, lw=2.0, alpha=0.35)   # saddle (light)
ax.plot(s, Delta_stable, lw=2.5, alpha=0.90)   # stable (dark)
ax.scatter([1.0], [np.pi/2], s=35, zorder=5)   # critical point

ax.set_xlim(0.0, 1.02)
ax.set_ylim(0.0, np.pi)
ax.set_xlabel(r"$P/\kappa$")
ax.set_ylabel(r"$\Delta\theta$")
ax.set_yticks([0, np.pi])
ax.set_yticklabels([r"$0$", r"$\pi$"])
ax.set_title("A", loc="left", fontweight="bold")


# =========
# Panel B: phase portrait (vector field + your single trajectory)
# =========
ax = axes[1]

# (1) vector field grid
Delta_grid = np.linspace(-np.pi, np.pi, 240)
omega_grid = np.linspace(-3.2, 3.2, 240)
D, W = np.meshgrid(Delta_grid, omega_grid)

U = W
V = 2.0*P - gamma*W - 2.0*kappa*np.sin(D)
speed = np.sqrt(U**2 + V**2)

# background (optional) + streamlines
ax.pcolormesh(D, W, speed, shading="auto", alpha=0.55)
ax.streamplot(Delta_grid, omega_grid, U, V, density=1.2, linewidth=0.9, arrowsize=0.9)

# (2) overlay YOUR solved trajectory (no extra solve_ivp)
#ax.plot(Delta_wrapped, omega, lw=1.5, alpha=0.95)

# (3) mark fixed points (if exist)
if fps is not None:
    for (D0, w0) in fps:
        ax.scatter([wrap_to_pi(D0)], [w0], s=45, zorder=6)

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r"$\Delta\theta$")
ax.set_ylabel(r"$\omega$")
ax.set_title("B", loc="left", fontweight="bold")

plt.show()      
        
plt.show()