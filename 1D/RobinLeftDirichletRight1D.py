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

def get_matrix(kappa, beta, n):
    '''
    A[i,j]
    \sum <\phi_i', \phi_j'>_{\kappa} u_j
    + \int_{\partial \Omega} b \phi_j \phi_i
    '''
    # n: num of basis
    h = 1/(n-1)
    A = np.zeros((n, n))
    A[0,0] = cdot_DhatBasis(0,0,h)*1/2 * kappa(1/2 * h) + beta(0) # * cdot_hatBasis(0,0,h) * 1/2
    A[0,1] = cdot_DhatBasis(0,1,h) * kappa(1/2 * h)
    A[-1,-1] = cdot_DhatBasis(0,0,h)*1/2 * kappa(1 - 1/2 * h) # + beta(1) # * cdot_hatBasis(0,0,h) * 1/2
    A[-1,-2] = cdot_DhatBasis(0,1,h) * kappa(1 - 1/2 * h)
    for i in range(1,n-1):
        for j in range(i-1,i+2):
            x = (i+j)/2 * h
            # kp = (kappa(x+h/3.46) + kappa(x-h/3.46)) / 2
            kp = kappa(x)
            # bx = b(x)
            A[i,j] = cdot_DhatBasis(i,j,h) * kp
    return A

def get_b(f, g, n):
    '''
    - \int_{\Omega} f * \phi_j + \int_{\partial \Omega} g * \phi_j
    '''
    h = 1/(n-1)
    b = np.zeros(n)
    b[0] = g(0) - f(0) * 1/2 * h
    b[-1] = - f(1) * 1/2 * h
    for i in range(1,n-1):
        x = i * h
        b[i] = - f(x) * h
    return b

def solve_robin_dirichlet_01(kappa, g, beta, f, u1, n):
    x = np.linspace(0,1,n)
    A = get_matrix(kappa, beta, n)
    b = get_b(f, g, n)
   
    A[-1,:] = 0
    A[-1,-1] = 1
    b[-1] = u1
    
    u = np.linalg.solve(A, b)
    # u = np.zeros(n)
    # u[0] = 0
    # u[1] = (b[0] - u[0]*A[0,0])/A[0,1]
    # for i in range(2,n):
    #     u[i] = (b[i-1] - u[i-1]*A[i-1,i-1] - u[i-2]*A[i-1,i-2])/A[i-1,i]

    return u, x, A, b



beta = lambda x:  - 1
g = lambda x: - 1
kappa = lambda x: np.exp(x)
f = lambda x: np.exp(x) * (np.cos(x) - 2*np.sin(x) - x*np.cos(x) - x*np.sin(x))
u1 = np.cos(1)

solve_real = lambda x: x * np.cos(x)
x_ref = np.linspace(0,1,100)
u_ref = solve_real(x_ref)

n = 1000
u, x, A, b = solve_robin_dirichlet_01(kappa, g, beta, f, u1, n) # 过0,0点

plt.plot(x, u, label='u')
plt.plot(x_ref, u_ref, label='u_ref')
plt.legend()
plt.title(f"n {n}")
plt.show()



'''
import sympy as sp

A = sp.Matrix(A)
b = sp.Matrix(b)

M = A.row_join(b)
R, pivots = M.rref()

v = R[:,-2]
u_s = -v
u_s[-1] = 1
u_s = u_s - u_s[0]
u_s = np.array(u_s)
u_s = u_s[:,0]
u_s = u_s - u_s[0]

# plt.plot(x, u_s, label='u_s')

'''