import numpy as np
from numpy.linalg import eig, matrix_rank
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import matplotlib.pyplot as plt

# =============================
# 模型参数
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

B1 = B[:, [0]]
B2 = B[:, [1]]

C = np.array([[0,0,0,1]])

x0 = np.array([0.85, 0.15, 0, -0.05])
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1001)

labels = ["x1","x2","x3","x4"]

def ctrb(A,B):
    n=A.shape[0]
    P=B
    for i in range(1,n):
        P=np.hstack((P, np.linalg.matrix_power(A,i)@B))
    return P

def obsv(A,C):
    n=A.shape[0]
    O=C
    for i in range(1,n):
        O=np.vstack((O, C@np.linalg.matrix_power(A,i)))
    return O

def control_energy(u,t):
    return np.trapz(u*u,t)

# ========== Q1 特征值 ==========
eigvals,_=eig(A)
print("\n===== Q1: Eigenvalues(A) =====")
print(eigvals)

# ========== Q2 可控性 ==========
print("\n===== Q2: Controllability =====")
for name,Bb in [("u1",B1),("u2",B2)]:
    r=matrix_rank(ctrb(A,Bb))
    print(f"rank(ctrb) for {name} =",r)

# ========== Q3 开环图 ==========
def open_loop(t,x):
    return A@x

sol=solve_ivp(open_loop,t_span,x0,t_eval=t_eval)
plt.figure()
for i in range(4):
    plt.plot(t_eval,sol.y[i],label=labels[i])
plt.legend(); plt.grid()
plt.title("Q3: Open-loop states")
plt.show()

# ========== Q4(a) 极点配置（注意 u=Kx，所以对(A,-B1)放极点） ==========
p_des=[-2+1j,-2-1j,-2+1.001j,-2-1.001j]
res_pp=place_poles(A,-B1,p_des)
K_pp=res_pp.gain_matrix
print("\n===== Q4(a): K_pp =====")
print(K_pp)

A_cl_pp=A + B1@K_pp
print("Closed-loop eigenvalues =", eig(A_cl_pp)[0])

def simulate(A,B1,K,label):
    def f(t,x): return A@x + B1.flatten()*float(K@x)
    sol=solve_ivp(f,t_span,x0,t_eval=t_eval)
    x=sol.y
    u=np.array([float(K@x[:,k]) for k in range(len(t_eval))])

    plt.figure()
    for i in range(4): plt.plot(t_eval,x[i],label=labels[i])
    plt.legend(); plt.grid()
    plt.title(label+" states")

    plt.figure()
    plt.plot(t_eval,u); plt.grid()
    plt.title(label+" control input")

    return x,u

# ========== Q4(b) LQR ==========
Q=10*np.eye(4)
R=np.array([[1]])
P=solve_continuous_are(A,B1,Q,R)
K_lqr = -np.linalg.inv(R)@(B1.T@P)
print("\n===== Q4(b): K_lqr =====")
print(K_lqr)

A_cl_lqr=A + B1@K_lqr
print("Closed-loop eigenvalues =", eig(A_cl_lqr)[0])

# ========== Q4(c) 仿真图 ==========
x_pp,u_pp = simulate(A,B1,K_pp,"Q4(c) Pole placement")
x_lqr,u_lqr = simulate(A,B1,K_lqr,"Q4(c) LQR")

Eu_pp = control_energy(u_pp,t_eval)
Eu_lqr= control_energy(u_lqr,t_eval)

print("\n===== Q4(c): Control Energy =====")
print("Eu_pp  =", Eu_pp)
print("Eu_lqr =", Eu_lqr)

# ========== Q4(d) 调参 LQR ==========
print("\n===== Q4(d): tuning LQR =====")
Eu_target = Eu_pp   # 目标: LQR 能量必须小于 PP

best = None

Q11_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, 1e8]

for q1 in Q11_list:

    Q_t = np.diag([q1, 10, 10, 10])
    P_t = solve_continuous_are(A, B1, Q_t, R)
    K_t = -np.linalg.inv(R) @ (B1.T @ P_t)

    # simulate
    def f(t,x): return A@x + B1.flatten()*float(K_t@x)
    sol = solve_ivp(f, t_span, x0, t_eval=t_eval)
    x_t = sol.y
    u_t = np.array([float(K_t @ x_t[:,k]) for k in range(len(t_eval))])

    max_x1 = np.max(np.abs(x_t[0, t_eval>=5]))
    Eu = control_energy(u_t, t_eval)

    print(f"Q11={q1:.1e}, max|x1|={max_x1:.4e}, Eu={Eu:.4f}")

    if max_x1 <= 0.01 and Eu < Eu_target:
        best = (Q_t, K_t, Eu, max_x1)
        break

if best is None:
    print("\nNo solution found. Try increasing Q11 range.")
else:
    print("\n===== Q4(d) Final Tuned LQR =====")
    print("Q =\n", best[0])
    print("max|x1| =", best[3])
    print("Energy =", best[2])

    # Plot final tuned LQR
    simulate(A,B1,best[1],"Q4(d) Final tuned LQR")


# 最终 tuned LQR 图
simulate(A,B1,best[1],"Q4(d) Final tuned LQR")

# ========================= Q5: Observer =========================

# Q5(a)
O=obsv(A,C)
print("\n===== Q5(a): rank(O) =====")
print(matrix_rank(O))

# Q5(b) 观测器极点
p_obs=[-4,-4.1,-4.2,-4.3]
L = place_poles(A.T,C.T,p_obs).gain_matrix.T
print("\n===== Q5(b): L =====")
print(L)

def simulate_observer(A,B1,C,K,L,label):
    def f(t,z):
        x=z[0:4]
        xh=z[4:8]
        y=C@x
        u=float(K@xh)
        dx=A@x + B1.flatten()*u
        dxh=A@xh + B1.flatten()*u + (L@(y - C@xh)).flatten()
        return np.concatenate((dx,dxh))

    z0=np.concatenate((x0, np.zeros(4)))
    sol=solve_ivp(f,t_span,z0,t_eval=t_eval)
    x=sol.y[0:4]; xh=sol.y[4:8]
    e=x-xh
    u=np.array([float(K@xh[:,k]) for k in range(len(t_eval))])

    # 状态图
    plt.figure()
    for i in range(4): plt.plot(t_eval,x[i],label=labels[i])
    plt.grid(); plt.legend()
    plt.title(label+" states")

    # 控制输入
    plt.figure()
    plt.plot(t_eval,u); plt.grid()
    plt.title(label+" control input")

    # 误差
    plt.figure()
    for i in range(4): plt.plot(t_eval,e[i],label="e"+str(i+1))
    plt.grid(); plt.legend()
    plt.title(label+" estimation error")

    return x,xh,e,u

# Q5(d) 两种控制器 + observer
x_pp_obs,_,_,u_pp_obs = simulate_observer(A,B1,C,K_pp,L,"Q5(d) PP+Observer")
x_lqr_obs,_,_,u_lqr_obs = simulate_observer(A,B1,C,K_lqr,L,"Q5(d) LQR+Observer")

# Q5(e) 能量
Eu_pp_obs = control_energy(u_pp_obs,t_eval)
Eu_lqr_obs= control_energy(u_lqr_obs,t_eval)

print("\n===== Q5(e): energies =====")
print("PP+obs =",Eu_pp_obs)
print("LQR+obs=",Eu_lqr_obs)

# Q5(f) tuned LQR + observer
print("\n===== Q5(f): tuning LQR+observer =====")

best2=None
for q1 in [50,100,200,500,1000,2000]:
    Q_t=np.diag([q1,10,10,10])
    P_t=solve_continuous_are(A,B1,Q_t,R)
    K_t=-np.linalg.inv(R)@(B1.T@P_t)
    x_t,_,_,u_t = simulate_observer(A,B1,C,K_t,L,"temp")
    plt.close("all")
    max_x1=np.max(np.abs(x_t[0,t_eval>=5]))
    Eu=control_energy(u_t,t_eval)
    if (max_x1<=0.01) and (Eu<Eu_pp_obs):
        best2=(Q_t,K_t,Eu,max_x1)
        break

print("Chosen Q =\n",best2[0])
print("max|x1| =",best2[3])
print("Energy =",best2[2])

simulate_observer(A,B1,C,best2[1],L,"Q5(f) Final tuned LQR+Observer")

plt.show()

