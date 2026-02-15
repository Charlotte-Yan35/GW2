
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. 系统定义
# ============================================================

def two_node_system(t, y, P, kappa, gamma):
    """
    两节点系统的ODE

    参数:
        y[0] = Δθ (相位差)
        y[1] = ω (相位差的导数)
        P: 功率 (发电机输出P, 消费者消耗-P)
        kappa: 耦合强度
        gamma: 阻尼系数
    """
    delta_theta, omega = y

    d_delta_theta = omega
    d_omega = 2*P - gamma*omega - 2*kappa*np.sin(delta_theta)

    return [d_delta_theta, d_omega]

#这里加了随机扰动
def two_node_system_with_noise(t, y, P, kappa, gamma, noise_func=None):
    """
    带随机扰动的两节点系统

    noise_func: 返回 [noise_theta, noise_omega] 的函数
    """
    delta_theta, omega = y

    d_delta_theta = omega
    d_omega = 2*P - gamma*omega - 2*kappa*np.sin(delta_theta)

    # 添加随机扰动
    if noise_func is not None:
        noise = noise_func(t)
        d_delta_theta += noise[0]
        d_omega += noise[1]

    return [d_delta_theta, d_omega]


# ============================================================
# 2. 不动点和稳定性分析
# ============================================================

def find_fixed_points(P, kappa):
    """
    计算不动点

    返回:
        fixed_point_1: (arcsin(P/κ), 0) - 稳定点或稳定螺旋
        fixed_point_2: (π - arcsin(P/κ), 0) - 鞍点
    """
    if P/kappa > 1:
        return None, None  # 无不动点（系统不稳定）

    delta_theta_1 = np.arcsin(P/kappa)
    delta_theta_2 = np.pi - np.arcsin(P/kappa)

    return (delta_theta_1, 0), (delta_theta_2, 0)


def compute_jacobian(delta_theta, P, kappa, gamma):
    """
    计算雅可比矩阵

    J = [[0, 1], [x, -γ]]
    其中 x = -2κcos(Δθ)
    """
    x = -2 * kappa * np.cos(delta_theta)
    J = np.array([[0, 1], [x, -gamma]])
    return J


def analyze_stability(P, kappa, gamma):
    """
    分析两个不动点的稳定性
    """
    fp1, fp2 = find_fixed_points(P, kappa)

    if fp1 is None:
        print(f"P/κ = {P/kappa:.3f} > 1: 系统无稳定不动点!")
        return None

    results = {}

    # 分析第一个不动点
    J1 = compute_jacobian(fp1[0], P, kappa, gamma)
    eigenvalues_1 = np.linalg.eigvals(J1)
    trace_1 = np.trace(J1)
    det_1 = np.linalg.det(J1)

    results['fp1'] = {
        'position': fp1,
        'jacobian': J1,
        'eigenvalues': eigenvalues_1,
        'trace': trace_1,
        'determinant': det_1,
        'type': classify_fixed_point(trace_1, det_1, eigenvalues_1)
    }

    # 分析第二个不动点
    J2 = compute_jacobian(fp2[0], P, kappa, gamma)
    eigenvalues_2 = np.linalg.eigvals(J2)
    trace_2 = np.trace(J2)
    det_2 = np.linalg.det(J2)

    results['fp2'] = {
        'position': fp2,
        'jacobian': J2,
        'eigenvalues': eigenvalues_2,
        'trace': trace_2,
        'determinant': det_2,
        'type': classify_fixed_point(trace_2, det_2, eigenvalues_2)
    }

    return results


def classify_fixed_point(trace, det, eigenvalues):
    """根据迹和行列式分类不动点"""
    if det < 0:
        return "鞍点 (Saddle)"
    elif det > 0:
        if trace < 0:
            if np.isreal(eigenvalues[0]) and np.isreal(eigenvalues[1]):
                return "稳定节点 (Stable Node)"
            else:
                return "稳定螺旋 (Stable Spiral)"
        elif trace > 0:
            if np.isreal(eigenvalues[0]) and np.isreal(eigenvalues[1]):
                return "不稳定节点 (Unstable Node)"
            else:
                return "不稳定螺旋 (Unstable Spiral)"
        else:
            return "中心 (Center)"
    else:
        return "退化情况"


# ============================================================
# 3. 绘图函数
# ============================================================

def plot_bifurcation_diagram(kappa=1.0, gamma=1.0, save_path=None):
    """
    绘制分岔图 (复现 Fig S1A)

    横轴: P/κ
    纵轴: Δθ*
    """
    P_kappa_ratio = np.linspace(0.01, 0.999, 200)

    delta_theta_stable = np.arcsin(P_kappa_ratio)
    delta_theta_saddle = np.pi - np.arcsin(P_kappa_ratio)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 稳定分支（蓝色实线）
    ax.plot(P_kappa_ratio, delta_theta_stable, 'b-', linewidth=2, label='Stable fixed point')
    # 不稳定分支（浅蓝色虚线）
    ax.plot(P_kappa_ratio, delta_theta_saddle, 'c--', linewidth=2, label='Saddle point')

    # 标记分岔点
    ax.plot(1, np.pi/2, 'ko', markersize=10, label=f'Bifurcation at P/κ = 1')

    ax.set_xlabel(r'$P/\kappa$', fontsize=14)
    ax.set_ylabel(r'$\Delta\theta^*$', fontsize=14)
    ax.set_title('Bifurcation Diagram: Two-Node System', fontsize=14)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, np.pi)
    ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"分岔图已保存到: {save_path}")
    plt.show()

    return fig


def plot_phase_portrait(P=0.5, kappa=1.0, gamma=1.0, save_path=None):
    """
    绘制相图 (复现 Fig S1B)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 创建网格
    delta_theta_range = np.linspace(-3, 3, 20)
    omega_range = np.linspace(-3, 3, 20)
    DT, OM = np.meshgrid(delta_theta_range, omega_range)

    # 计算向量场
    U = OM  # d(Δθ)/dt = ω
    V = 2*P - gamma*OM - 2*kappa*np.sin(DT)  # dω/dt

    # 绘制流线
    ax.streamplot(DT, OM, U, V, density=1.5, color='purple',
                  linewidth=0.5, arrowsize=1, arrowstyle='->')

    # 标记不动点
    fp1, fp2 = find_fixed_points(P, kappa)
    if fp1:
        ax.plot(fp1[0], fp1[1], 'go', markersize=12, label=f'Stable: ({fp1[0]:.2f}, 0)')
        ax.plot(fp2[0], fp2[1], 'ro', markersize=12, label=f'Saddle: ({fp2[0]:.2f}, 0)')

    ax.set_xlabel(r'$\Delta\theta$', fontsize=14)
    ax.set_ylabel(r'$\omega$', fontsize=14)
    ax.set_title(f'Phase Portrait: P={P}, κ={kappa}, γ={gamma}', fontsize=14)
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"相图已保存到: {save_path}")
    plt.show()

    return fig


def plot_time_series(P=0.5, kappa=1.0, gamma=1.0,
                     initial_conditions=None, t_span=(0, 50),
                     save_path=None):
    """
    绘制时间序列
    """
    if initial_conditions is None:
        initial_conditions = [
            [0.1, 0.0],   # 接近稳定点
            [2.5, 0.0],   # 接近鞍点
            [1.5, 1.0],   # 中间位置
        ]

    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    fig, axes = plt.subplots(len(initial_conditions), 2, figsize=(12, 3*len(initial_conditions)))
    if len(initial_conditions) == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(initial_conditions)))

    for i, y0 in enumerate(initial_conditions):
        sol = solve_ivp(two_node_system, t_span, y0,
                       args=(P, kappa, gamma),
                       t_eval=t_eval, method='RK45')

        axes[i, 0].plot(sol.t, sol.y[0], color=colors[i], linewidth=1.5)
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel(r'$\Delta\theta$')
        axes[i, 0].set_title(f'Initial: Δθ₀={y0[0]:.1f}, ω₀={y0[1]:.1f}')
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(sol.t, sol.y[1], color=colors[i], linewidth=1.5)
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel(r'$\omega$')
        axes[i, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Time Series: P={P}, κ={kappa}, γ={gamma}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"时间序列图已保存到: {save_path}")
    plt.show()

    return fig


# ============================================================
# 4. 随机扰动模拟
# ============================================================

def simulate_with_random_events(P=0.5, kappa=1.0, gamma=1.0,
                                 y0=[0.1, 0.0], t_span=(0, 100),
                                 event_times=None, event_magnitudes=None,
                                 save_path=None):
    """
    模拟带随机事件的系统

    随机事件可以模拟:
    - 突然的负载变化
    - 线路故障
    - 发电机跳闸

    参数:
        event_times: 事件发生的时间列表
        event_magnitudes: 对应的扰动幅度 [(delta_P, delta_omega), ...]
    """
    if event_times is None:
        # 随机生成3-5个事件
        n_events = np.random.randint(3, 6)
        event_times = np.sort(np.random.uniform(10, t_span[1]-10, n_events))
        event_magnitudes = [(np.random.uniform(-0.3, 0.3),
                            np.random.uniform(-0.5, 0.5)) for _ in range(n_events)]

    # 分段求解
    t_all = []
    y_all = []

    current_y = y0.copy()
    current_t = t_span[0]

    all_event_times = list(event_times) + [t_span[1]]

    for i, t_end in enumerate(all_event_times):
        t_eval = np.linspace(current_t, t_end, int((t_end - current_t) * 20))

        sol = solve_ivp(two_node_system, (current_t, t_end), current_y,
                       args=(P, kappa, gamma), t_eval=t_eval, method='RK45')

        t_all.extend(sol.t.tolist())
        y_all.append(sol.y)

        if i < len(event_magnitudes):
            # 应用扰动
            current_y = [sol.y[0, -1] + event_magnitudes[i][0],
                        sol.y[1, -1] + event_magnitudes[i][1]]
            current_t = t_end

    # 合并结果
    t_all = np.array(t_all)
    y_all = np.hstack(y_all)

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_all, y_all[0], 'b-', linewidth=1)
    axes[0].set_ylabel(r'$\Delta\theta$', fontsize=12)
    axes[0].set_title(f'System Response with Random Events (P={P}, κ={kappa}, γ={gamma})')

    # 标记事件
    for t_event in event_times:
        axes[0].axvline(t_event, color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(t_event, color='r', linestyle='--', alpha=0.5)

    axes[1].plot(t_all, y_all[1], 'g-', linewidth=1)
    axes[1].set_ylabel(r'$\omega$', fontsize=12)
    axes[1].set_xlabel('Time', fontsize=12)

    # 添加不动点参考线
    fp1, _ = find_fixed_points(P, kappa)
    if fp1:
        axes[0].axhline(fp1[0], color='b', linestyle=':', alpha=0.5, label=f'Stable Δθ*={fp1[0]:.2f}')
        axes[0].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"随机事件模拟图已保存到: {save_path}")
    plt.show()

    return fig, t_all, y_all, event_times, event_magnitudes


def simulate_cascading_failure(P_initial=0.5, kappa=1.0, gamma=1.0,
                                y0=[0.1, 0.0], t_span=(0, 100),
                                failure_time=30, P_after_failure=0.9,
                                save_path=None):
    """
    模拟级联故障场景

    场景: 在 failure_time 时刻，功率需求突然增加到 P_after_failure
    如果 P_after_failure/kappa > 1，系统将失去稳定性
    """
    # 故障前
    t_eval_before = np.linspace(0, failure_time, 500)
    sol_before = solve_ivp(two_node_system, (0, failure_time), y0,
                          args=(P_initial, kappa, gamma),
                          t_eval=t_eval_before, method='RK45')

    # 故障后（从故障前的最终状态继续）
    y_at_failure = [sol_before.y[0, -1], sol_before.y[1, -1]]
    t_eval_after = np.linspace(failure_time, t_span[1], 1000)
    sol_after = solve_ivp(two_node_system, (failure_time, t_span[1]), y_at_failure,
                         args=(P_after_failure, kappa, gamma),
                         t_eval=t_eval_after, method='RK45')

    # 合并
    t_all = np.concatenate([sol_before.t, sol_after.t])
    y_all = np.hstack([sol_before.y, sol_after.y])

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Δθ
    axes[0].plot(t_all, y_all[0], 'b-', linewidth=1.5)
    axes[0].axvline(failure_time, color='r', linestyle='--', linewidth=2, label='Failure Event')
    axes[0].set_ylabel(r'$\Delta\theta$', fontsize=12)
    axes[0].legend()

    # ω
    axes[1].plot(t_all, y_all[1], 'g-', linewidth=1.5)
    axes[1].axvline(failure_time, color='r', linestyle='--', linewidth=2)
    axes[1].set_ylabel(r'$\omega$ (frequency deviation)', fontsize=12)

    # Power flow
    power_flow = kappa * np.sin(y_all[0])
    axes[2].plot(t_all, power_flow, 'm-', linewidth=1.5)
    axes[2].axvline(failure_time, color='r', linestyle='--', linewidth=2)
    axes[2].axhline(kappa, color='orange', linestyle=':', label=f'Max capacity κ={kappa}')
    axes[2].set_ylabel(r'Power flow $\kappa\sin(\Delta\theta)$', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].legend()

    # 判断是否失稳
    is_stable = P_after_failure / kappa <= 1
    status = "STABLE" if is_stable else "UNSTABLE (P/κ > 1)"

    axes[0].set_title(f'Cascading Failure Simulation\n'
                      f'P: {P_initial} → {P_after_failure} at t={failure_time} | Status: {status}',
                      fontsize=14)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"级联故障模拟图已保存到: {save_path}")
    plt.show()

    return fig


# ============================================================
# 5. 参数扫描
# ============================================================

def parameter_sweep(P_range=(0.1, 0.95), gamma_range=(0.1, 2.0),
                    kappa=1.0, resolution=20, save_path=None):
    """
    参数扫描：研究P和γ对稳定性的影响
    """
    P_values = np.linspace(P_range[0], P_range[1], resolution)
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], resolution)

    # 存储收敛时间（作为稳定性度量）
    convergence_time = np.zeros((resolution, resolution))

    for i, P in enumerate(P_values):
        for j, gamma in enumerate(gamma_values):
            # 从扰动状态开始
            y0 = [0.5, 0.5]
            t_span = (0, 100)
            t_eval = np.linspace(0, 100, 1000)

            sol = solve_ivp(two_node_system, t_span, y0,
                           args=(P, kappa, gamma), t_eval=t_eval)

            # 计算收敛到稳态的时间
            fp1, _ = find_fixed_points(P, kappa)
            if fp1:
                threshold = 0.01
                converged = np.abs(sol.y[0] - fp1[0]) < threshold
                if np.any(converged):
                    conv_idx = np.argmax(converged)
                    convergence_time[j, i] = sol.t[conv_idx]
                else:
                    convergence_time[j, i] = 100  # 未收敛
            else:
                convergence_time[j, i] = np.nan

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(convergence_time, origin='lower', aspect='auto',
                   extent=[P_range[0], P_range[1], gamma_range[0], gamma_range[1]],
                   cmap='RdYlGn_r')

    plt.colorbar(im, label='Convergence Time')
    ax.set_xlabel(r'$P$', fontsize=14)
    ax.set_ylabel(r'$\gamma$', fontsize=14)
    ax.set_title(f'Parameter Sweep: Convergence Time (κ={kappa})\n'
                 f'Green=Fast convergence, Red=Slow/No convergence', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"参数扫描图已保存到: {save_path}")
    plt.show()

    return fig, convergence_time


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("两节点电网稳定性分析")
    print("=" * 60)

    # ----- 参数设置 -----
    # 推荐参数范围（基于论文）:
    # - κ (耦合强度): 通常设为1，实际物理值约18.3 kW
    # - γ (阻尼系数): 通常在0.1-2之间，论文中常用γ=1
    # - P (功率): 必须满足 P/κ < 1 才有稳定解

    P = 0.5      # 功率
    kappa = 1.0  # 耦合强度
    gamma = 1.0  # 阻尼系数

    print(f"\n参数设置: P={P}, κ={kappa}, γ={gamma}")
    print(f"P/κ = {P/kappa:.3f} (必须 < 1 才稳定)")

    # ----- 稳定性分析 -----
    print("\n" + "-" * 40)
    print("稳定性分析")
    print("-" * 40)

    results = analyze_stability(P, kappa, gamma)

    if results:
        for name, info in results.items():
            print(f"\n{name.upper()}:")
            print(f"  位置: Δθ* = {info['position'][0]:.4f}, ω* = {info['position'][1]:.4f}")
            print(f"  特征值: {info['eigenvalues']}")
            print(f"  迹 = {info['trace']:.4f}, 行列式 = {info['determinant']:.4f}")
            print(f"  类型: {info['type']}")

    # ----- 生成图形 -----
    print("\n" + "-" * 40)
    print("生成图形...")
    print("-" * 40)

    # 1. 分岔图
    plot_bifurcation_diagram(kappa, gamma, save_path=OUT_DIR / 'bifurcation_diagram.png')

    # 2. 相图
    plot_phase_portrait(P, kappa, gamma, save_path=OUT_DIR / 'phase_portrait.png')

    # 3. 时间序列
    plot_time_series(P, kappa, gamma, save_path=OUT_DIR / 'time_series.png')

    # 4. 随机事件模拟
    print("\n" + "-" * 40)
    print("随机事件模拟")
    print("-" * 40)
    simulate_with_random_events(P, kappa, gamma, save_path=OUT_DIR / 'random_events.png')

    # 5. 级联故障模拟
    print("\n" + "-" * 40)
    print("级联故障模拟")
    print("-" * 40)
    # 测试接近临界的情况
    simulate_cascading_failure(P_initial=0.5, P_after_failure=0.95,
                               kappa=kappa, gamma=gamma,
                               save_path=OUT_DIR / 'cascading_failure_stable.png')

    # 测试超过临界的情况（系统失稳）
    simulate_cascading_failure(P_initial=0.5, P_after_failure=1.2,
                               kappa=kappa, gamma=gamma,
                               save_path=OUT_DIR / 'cascading_failure_unstable.png')

    # 6. 参数扫描
    print("\n" + "-" * 40)
    print("参数扫描")
    print("-" * 40)
    parameter_sweep(save_path=OUT_DIR / 'parameter_sweep.png')

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
