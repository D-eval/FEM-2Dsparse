'''
# python3.9
对于第一题
令
kappa = e^x
g = cos(x)
f = e^x * (cos(x) - 2*sin(x) - x*cos(x) - x*sin(x))

u' = u - g
nabla_kappa_nabla_g = - (sin(x) + cos(x)) * e^x
f' = f - nabla_kappa_nabla_g
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

def get_matrix(kappa, n):
    '''
    A[i,j]
    \int \kappa <\phi_i, \phi_j> u_j
    '''
    # n: num of basis
    h = 1/(n-1)
    A = np.zeros((n, n))
    A[0,0] = 1#cdot_DhatBasis(0,0,h) * kappa(0)
    A[0,1] = 0#cdot_DhatBasis(0,1,h) * kappa(1/2 * h)
    A[-1,-1] = 1#cdot_DhatBasis(0,0,h) * kappa(1)
    A[-1,-2] = 0#cdot_DhatBasis(0,1,h) * kappa(1 - 1/2 * h)
    for i in range(1,n-1):
        for j in range(i-1,i+2):
            x = (i+j)/2 * h
            kp = kappa(x)
            A[i,j] = cdot_DhatBasis(i,j,h) * kp
    return A

def get_b(f, n):
    '''
    - \int_{\Omega} f * \phi_j
    '''
    h = 1/(n-1)
    b = np.zeros(n)
    b[0] = 0#f(0) * 1/2 * h
    b[-1] = 0#f(1) * 1/2 * h
    for i in range(1,n-1):
        x = i * h
        b[i] = f(x) * h
    b *= -1
    return b



def solve_dirichlet01(kappa, g, f, nabla_kappa_nabla_g, n):
    f_prime = lambda x: f(x) - nabla_kappa_nabla_g(x)
    x = np.linspace(0,1,n)
    A = get_matrix(kappa, n)
    b = get_b(f_prime, n)
    u_prime = np.linalg.solve(A, b)
    u = u_prime + g(x)
    return u, x


