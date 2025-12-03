'''第二题
# python3.7
'''

import numpy as np
import matplotlib.pyplot as plt


# 试函数选帽函数

# phi_i phi_j 内积
def cdot_hatBasis(i, j, h):
    if i == j:
        return 2/3 * h
    elif j-1<=i<=j+1:
        return 1/6 * h
    else:
        return 0

# phi_i' phi_j' 内积
def cdot_DhatBasis(i, j, h):
    if i == j:
        return 2 / h
    elif j-1<=i<=j+1:
        return -1 / h
    else:
        return 0
    
def get_matrix_neumann(kappa, n):
    h = 1.0 / (n - 1)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            # 用节点 i/j 的中点作为积分点
            x_mid = (i + j) / 2.0 * h
            kp = kappa(x_mid)
            A[i, j] += kp * cdot_DhatBasis(i, j, h)
    A[0,0] *= 1/2
    A[-1,-1] *= 1/2
    return A


def get_b_neumann(f, n):
    h = 1.0 / (n - 1)
    b = np.zeros(n)
    # 体积分：简单近似 ∫ f φ_j ≈ f(x_j) * h
    for j in range(n):
        xj = j * h
        b[j] += f(xj) * h
    b[0] *= 1/2
    b[-1] *= 1/2
    # 右端 Neumann 边界：q1 = κ(1) u'(1)
    q1 = np.exp(1.0) * (np.cos(1.0) - np.sin(1.0))
    b[-1] += q1
    return b


def solve_neumann_example2(kappa, f, n):
    A = get_matrix_neumann(kappa, n)
    b = get_b_neumann(f, n)
    # 纯 Neumann → A 奇异 → 选一个解：u(0)=0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0   # 与精确解一致
    u = np.linalg.solve(A, b)
    h = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    return u, x

kappa = lambda x: np.exp(x)
f = lambda x: -np.exp(x) * (np.cos(x) - 2*np.sin(x) - x*np.cos(x) - x*np.sin(x))

for n in [1000]:
    u, x = solve_neumann_example2(kappa, f, n)
    u_real = x * np.cos(x)
    mse = np.mean((u - u_real)**2)
    print(n, mse)
    plt.plot(x, u, label=f'n={n}')
    plt.plot(x, u_real, label=f'exact')
    plt.legend()
    plt.show()