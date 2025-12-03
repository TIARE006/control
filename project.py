import numpy as np
from numpy.linalg import eig, matrix_rank
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import matplotlib.pyplot as plt

# =============================
# 模型参数：A, B
# =============================
A = np.array([
    [-0.0366,  0.0271,  0.0188, -0.4555],
    [ 0.0482, -1.0100,  0.0024, -4.0208],
    [ 0.1002,  0.3681, -0.7070,  1.4200],
    [ 0.0,     0.0,     1.0,     0.0   ]
])

B = np.array([
    [ 0.4422,  0.1761],
    [ 3.5446, -7.5922],
    [-5.5200,  4.4900],
    [ 0.0,     0.0   ]
])

# 单输入：只用 u1
B1 = B[:, [0]]  # 4x1
B2 = B[:, [1]]  # 4x1

# 输出（只能量到 pitch angle x4）
C = np.array([[0.0, 0.0, 0.0, 1.0]])

# 初始状态
x0 = np.array([0.85, 0.15, 0.0, -0.05])
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 1001)

labels = [r"$x_1$ (horiz vel)", r"$x_2$ (vert vel)",
          r"$x_3$ (pitch rate)", r"$x_4$ (pitch angle)"]

# =====================================================
# 一些辅助函数：可控性、可观性、能量、仿真
# =====================================================
def ctrb(A, B):
    n = A.shape[0]
    P = B
    for i in range(1, n):
        P = np.hstack((P, np.linalg.matrix_power(A, i) @ B))
    return P

def obsv(A, C):
    n = A.shape[0]
    Q = C
    for i in range(1, n):
        Q = np.vstack((Q, C @ np.linalg.matrix_power(A, i)))
    return Q

def control_energy(u, t):
    # ∫ u(t)^2 dt，数值积分
    return np.trapz(u ** 2, t)

def closed_loop_dyn(A, B1, K):
    """返回一个 f(t, x) 给 solve_ivp 用，u = -Kx."""
    def f(t, x):
        u = float(-K @ x)          # 标量
        xdot = A @ x + (B1.flatten() * u)
        return xdot
    return f

def simulate_closed_loop(A, B1, K, x0, t_span, t_eval, label_prefix,
                         plot=True):
    """仿真全状态反馈闭环，返回 (x(t), u(t))"""
    dyn = closed_loop_dyn(A, B1, K)
    sol = solve_ivp(dyn, t_span, x0, t_eval=t_eval)
    x = sol.y
    # 计算 u(t)
    u = np.zeros_like(t_eval)
    for k in range(len(t_eval)):
        u[k] = float(-K @ x[:, k])

    if plot:
        plt.figure()
        for i in range(4):
            plt.plot(t_eval, x[i, :], label=labels[i])
        plt.xlabel("t [s]")
        plt.ylabel("States")
        plt.title(f"{label_prefix} closed-loop state trajectories")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(t_eval, u)
        plt.xlabel("t [s]")
        plt.ylabel("u1(t)")
        plt.title(f"{label_prefix} control input u1(t)")
        plt.grid(True)

    return x, u

def simulate_observer(A, B1, C, K, L, x0, t_span, t_eval, label_prefix):
    """
    全阶观测器 + 状态反馈：
        x_dot     = A x + B u
        xhat_dot  = A xhat + B u + L (y - C xhat)
        u = -K xhat
    返回 (x, x_hat, e, u)
    """
    def aug_dyn(t, z):
        x = z[0:4]
        x_hat = z[4:8]

        y = C @ x                 # 1x1
        u = float(-K @ x_hat)     # 标量

        x_dot = A @ x + (B1.flatten() * u)
        xhat_dot = (A @ x_hat
                    + (B1.flatten() * u)
                    + (L @ (y - C @ x_hat)).flatten())
        return np.concatenate((x_dot, xhat_dot))

    z0 = np.concatenate((x0, np.zeros(4)))
    sol = solve_ivp(aug_dyn, t_span, z0, t_eval=t_eval)
    z = sol.y
    x = z[0:4, :]
    x_hat = z[4:8, :]
    e = x - x_hat

    # 控制输入
    u_traj = np.zeros_like(t_eval)
    for k in range(len(t_eval)):
        u_traj[k] = float(-K @ x_hat[:, k])

    # 画图
    plt.figure()
    for i in range(4):
        plt.plot(t_eval, x[i, :], label=f"x{i+1}")
    plt.xlabel("t [s]")
    plt.ylabel("States")
    plt.title(f"{label_prefix}: true states")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t_eval, u_traj)
    plt.xlabel("t [s]")
    plt.ylabel("u1(t)")
    plt.title(f"{label_prefix}: control input u1(t)")
    plt.grid(True)

    plt.figure()
    for i in range(4):
        plt.plot(t_eval, e[i, :], label=f"e{i+1}")
    plt.xlabel("t [s]")
    plt.ylabel("Estimation error")
    plt.title(f"{label_prefix}: estimation error e = x - x_hat")
    plt.legend()
    plt.grid(True)

    return x, x_hat, e, u_traj


# =====================================================
# Question 1: Stability
# =====================================================
eigvals, eigvecs = eig(A)
print("==== Q1: Eigenvalues of A ====")
print(eigvals)
# 报告上：看所有特征值的实部，判断开环稳定性


# =====================================================
# Question 2: Controllability / Reachability / Stabilizability
# =====================================================
print("\n==== Q2: Controllability / Stabilizability ====")

for name, Bb in [("u1", B1), ("u2", B2)]:
    P = ctrb(A, Bb)
    r = matrix_rank(P)
    print(f"\nInput {name}: rank(ctrb) = {r}")

    n = A.shape[0]

    if r == n:
        print(f"  -> 对输入 {name}：系统对所有状态都是可控/可达的")
        stabilizable = True
    else:
        print(f"  -> 对输入 {name}：系统整体不可控")
        # 检查可稳定化：所有不稳定极点是否可控（PBH 条件）
        stabilizable = True
        for lam in eigvals:
            if lam.real >= 0:   # 不稳定或边界稳定
                M = np.hstack((lam * np.eye(n) - A, Bb))
                if matrix_rank(M) < n:
                    stabilizable = False
                    break
    if stabilizable:
        print(f"  -> 对输入 {name}：系统是可稳定化的 (stabilizable)")
    else:
        print(f"  -> 对输入 {name}：系统不是可稳定化的")


# =====================================================
# Question 3: Open-loop simulation (u=0)
# =====================================================
print("\n==== Q3: Open-loop simulation (u=0) ====")

def dyn_open_loop(t, x):
    return A @ x  # u=0

sol_open = solve_ivp(dyn_open_loop, t_span, x0, t_eval=t_eval)
x_ol = sol_open.y  # shape (4, len(t_eval))

plt.figure()
for i in range(4):
    plt.plot(t_eval, x_ol[i, :], label=labels[i])
plt.xlabel("t [s]")
plt.ylabel("States")
plt.title("Open-loop state trajectories (u=0)")
plt.legend()
plt.grid(True)


# =====================================================
# Question 4(a): State-feedback via pole placement (single input u1)
# =====================================================
print("\n==== Q4(a): Pole-placement state feedback (u1 only) ====")
# 期望极点：-2 ± j，重复两次
p_des = [-2 + 1j, -2 - 1j, -2 + 1j*1.001, -2 - 1j*1.001]  # 小扰动避免完全重复

res_pp = place_poles(A, B1, p_des)
K_pp = res_pp.gain_matrix  # 1x4
print("K (pole placement) =", K_pp)

A_cl_pp = A - B1 @ K_pp
eigvals_pp, _ = eig(A_cl_pp)
print("Closed-loop eigenvalues (PP):", eigvals_pp)


# =====================================================
# Question 4(b): LQR (Q = 10 I, R = 1)
# =====================================================
print("\n==== Q4(b): LQR with Q=10I, R=1 ====")
Q = 10.0 * np.eye(4)
R = np.array([[1.0]])

P_lqr = solve_continuous_are(A, B1, Q, R)
K_lqr = np.linalg.inv(R) @ (B1.T @ P_lqr)  # 1x4
print("K (LQR) =", K_lqr)

A_cl_lqr = A - B1 @ K_lqr
eigvals_lqr, _ = eig(A_cl_lqr)
print("Closed-loop eigenvalues (LQR):", eigvals_lqr)


# =====================================================
# Question 4(c): Closed-loop simulations (PP vs LQR)
# =====================================================
print("\n==== Q4(c): Closed-loop simulations ====")
x_pp,  u_pp  = simulate_closed_loop(A, B1, K_pp,  x0, t_span, t_eval,
                                    "Pole placement", plot=True)
x_lqr, u_lqr = simulate_closed_loop(A, B1, K_lqr, x0, t_span, t_eval,
                                    "LQR (Q=10I, R=1)", plot=True)


# =====================================================
# Question 4(d): Tune LQR such that |x1(t)| <= 0.01 for t>=5
#                and energy less than PP controller
# =====================================================
print("\n==== Q4(d): Tuning LQR (full state) ====")

def evaluate_LQR_full(Q_diag, R_scalar=1.0):
    Q_tune = np.diag(Q_diag)
    R_tune = np.array([[R_scalar]])

    P_tune = solve_continuous_are(A, B1, Q_tune, R_tune)
    K_tune = np.linalg.inv(R_tune) @ (B1.T @ P_tune)

    x_tune, u_tune = simulate_closed_loop(
        A, B1, K_tune, x0, t_span, t_eval,
        label_prefix=f"LQR tuned Q={Q_diag}, R={R_scalar}",
        plot=False
    )
    mask = t_eval >= 5.0
    max_x1 = np.max(np.abs(x_tune[0, mask]))
    Eu = control_energy(u_tune, t_eval)
    return max_x1, Eu, K_tune, Q_tune, R_tune

Eu_pp_full = control_energy(u_pp, t_eval)
print(f"PP controller energy Eu = {Eu_pp_full:.4f}")

best = None

q1_candidates = [10, 50, 100, 200, 500, 1000, 2000]
for q1 in q1_candidates:
    Q_diag = [q1, 10, 10, 10]
    max_x1, Eu, K_tune, Q_tune, R_tune = evaluate_LQR_full(Q_diag, 1.0)
    print(f"Q_diag={Q_diag}, max|x1| (t>=5) = {max_x1:.4e}, Eu = {Eu:.4f}")
    if (max_x1 <= 0.01) and (Eu < Eu_pp_full):
        if (best is None) or (Eu < best["Eu"]):
            best = {
                "Eu": Eu,
                "Q": Q_tune,
                "R": R_tune,
                "K": K_tune,
                "max_x1": max_x1
            }

if best is not None:
    print("\n[Chosen tuned LQR (Q4d)]")
    print("Q =\n", best["Q"])
    print("R =\n", best["R"])
    print("max|x1(t)| for t>=5 =", best["max_x1"])
    print("Eu (tuned LQR)     =", best["Eu"])
    # 可以画一下对应轨迹
    x_best, u_best = simulate_closed_loop(
        A, B1, best["K"], x0, t_span, t_eval,
        label_prefix="Tuned LQR (Q4d)", plot=True
    )
else:
    print("\n找不到同时满足 |x1|<=0.01 且 Eu < Eu_PP 的候选 Q，请扩大搜索范围再试。")


# =====================================================
# Question 5(a): Observability / reconstructible / detectable
# =====================================================
print("\n==== Q5(a): Observability / reconstructibility / detectability ====")
O = obsv(A, C)
rO = matrix_rank(O)
print("rank(Observability matrix) =", rO)

n = A.shape[0]
observable = (rO == n)

# PBH 检查：对于所有不稳定极点，检查 [lam I - A; C] 的秩
detectable = True
for lam in eigvals:
    if lam.real >= 0:
        M = np.vstack((lam * np.eye(n) - A, C))
        if matrix_rank(M) < n:
            detectable = False
            break

if observable:
    print("-> 系统是可观的 (observable)")
    print("-> 也因此是可重构 / 可检测的")
else:
    print("-> 系统不可完全观测")
    if detectable:
        print("-> 但所有不稳定模态可观，因此是可重构/可检测的 (reconstructible/detectable)")
    else:
        print("-> 且存在不稳定不可观模态，因此不可检测 (not detectable)")


# =====================================================
# Question 5(b): Full-order observer with pole placement
# =====================================================
print("\n==== Q5(b): Full-order observer via pole placement ====")
p_obs = [-4.0, -4.1, -4.2, -4.3]  # 一组在 -4 附近的极点

res_L = place_poles(A.T, C.T, p_obs)
L_pp = res_L.gain_matrix.T  # 4x1
print("L (observer gain) =\n", L_pp)


# =====================================================
# Question 5(c): LQR (same as Q4b, but配合观测器使用)
# =====================================================
print("\n==== Q5(c): LQR used with observer (Q=10I, R=1) ====")
print("K_lqr 已在 Q4(b) 计算，将和观测器 L_pp 一起使用。")


# =====================================================
# Question 5(d): Simulate controller + observer
# =====================================================
print("\n==== Q5(d): Simulations with controller + observer ====")

# 极点配置控制器 + 观测器
x_pp_obs, xhat_pp, e_pp, u_pp_obs = simulate_observer(
    A, B1, C, K_pp, L_pp, x0, t_span, t_eval,
    "PP controller + observer"
)

# LQR 控制器 + 观测器
x_lqr_obs, xhat_lqr, e_lqr, u_lqr_obs = simulate_observer(
    A, B1, C, K_lqr, L_pp, x0, t_span, t_eval,
    "LQR controller + observer"
)


# =====================================================
# Question 5(e): Control energy with observer
# =====================================================
print("\n==== Q5(e): Control energy with observers ====")
Eu_pp_obs  = control_energy(u_pp_obs,  t_eval)
Eu_lqr_obs = control_energy(u_lqr_obs, t_eval)
print("Eu (PP + observer)  =", Eu_pp_obs)
print("Eu (LQR + observer) =", Eu_lqr_obs)


# =====================================================
# Question 5(f): Tune LQR (with observer) for |x1|<=0.01, Eu<L_PP
# =====================================================
print("\n==== Q5(f): Tuning LQR with observer ====")

Eu_target = Eu_pp_obs  # 要求 LQR 能量少于此值

def evaluate_LQR_with_observer(Q_diag, R_scalar=1.0):
    Q_tune = np.diag(Q_diag)
    R_tune = np.array([[R_scalar]])

    P_tune = solve_continuous_are(A, B1, Q_tune, R_tune)
    K_tune = np.linalg.inv(R_tune) @ (B1.T @ P_tune)

    x_tune, xhat_tune, e_tune, u_tune = simulate_observer(
        A, B1, C, K_tune, L_pp, x0, t_span, t_eval,
        label_prefix=f"LQR+Obs tuned Q={Q_diag}, R={R_scalar}"
    )

    # 约束：|x1(t)| <= 0.01 for t>=5
    mask = t_eval >= 5.0
    max_x1 = np.max(np.abs(x_tune[0, mask]))
    Eu = control_energy(u_tune, t_eval)
    return max_x1, Eu, K_tune, Q_tune, R_tune

best_obs = None
q1_candidates_obs = [10, 50, 100, 200, 500, 1000, 2000]

for q1 in q1_candidates_obs:
    Q_diag = [q1, 10, 10, 10]
    max_x1, Eu, K_tune, Q_tune, R_tune = evaluate_LQR_with_observer(
        Q_diag, 1.0
    )
    print(f"[Obs] Q_diag={Q_diag}, max|x1| (t>=5) = {max_x1:.4e}, Eu = {Eu:.4f}")
    if (max_x1 <= 0.01) and (Eu < Eu_target):
        if (best_obs is None) or (Eu < best_obs["Eu"]):
            best_obs = {
                "Eu": Eu,
                "Q": Q_tune,
                "R": R_tune,
                "K": K_tune,
                "max_x1": max_x1
            }

if best_obs is not None:
    print("\n[Chosen tuned LQR + observer (Q5f)]")
    print("Q =\n", best_obs["Q"])
    print("R =\n", best_obs["R"])
    print("max|x1(t)| for t>=5 =", best_obs["max_x1"])
    print("Eu (tuned LQR+Obs)  =", best_obs["Eu"])
else:
    print("\n找不到同时满足 |x1|<=0.01 且 Eu < Eu_PP+Obs 的候选 Q，请扩大搜索范围再试。")

plt.show()
