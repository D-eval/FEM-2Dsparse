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
    A[0,0] = cdot_DhatBasis(0,0,h)*1/2 * kappa(0) + beta(0) # * cdot_hatBasis(0,0,h) * 1/2
    A[0,1] = cdot_DhatBasis(0,1,h) * kappa(1/2 * h)
    A[-1,-1] = cdot_DhatBasis(0,0,h)*1/2 * kappa(1) + beta(1) # * cdot_hatBasis(0,0,h) * 1/2
    A[-1,-2] = cdot_DhatBasis(0,1,h) * kappa(1 - 1/2 * h)
    for i in range(1,n-1):
        for j in range(i-1,i+2):
            x = (i+j)/2 * h
            kp = (kappa(x+h/3.46) + kappa(x-h/3.46)) / 2
            # kp = kappa(x)
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
    b[-1] = g(1) - f(1) * 1/2 * h
    for i in range(1,n-1):
        x = i * h
        b[i] = - f(x) * h
    return b

def solve_robin01(kappa, g, beta, f, n):
    x = np.linspace(0,1,n)
    A = get_matrix(kappa, beta, n)
    b = get_b(f, g, n)
    
    # 选择经过的点
    # A[C[0], :] = 0.0
    # A[C[0], C[0]] = 1
    # b[C[0]] = C[1]
    
    u = np.linalg.solve(A, b)
    return u, x, A, b


