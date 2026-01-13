'''
xyt
ode问题
计算时间采样点，
init后
Point2Value = {Point: [u0]}
迭代后
Point2Value = {Point: [u0,u1,...,un]}
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mesh_div import get_tri
import torch
import torch.nn.functional as F

import os

# {(tri,point_idx): Unit}
AllUnit = {}
Point2Unit = {} # {Point: {Unit: idx}}
class Unit:
    def __init__(self, tri, point_idx):
        AllUnit[(tri,point_idx)] = self
        self.tri = tri
        self.point_idx = point_idx
        point = tri.points[point_idx]
        if Point2Unit.get(tri.points[point_idx]) is None:
            Point2Unit[point] = {}
            Point2Unit[point][self] = point_idx
        else:
            Point2Unit[point][self] = point_idx
    def getPoint(self):
        p_idx0 = self.point_idx
        p_idx1 = (self.point_idx+1)%3
        p_idx2 = (self.point_idx+2)%3
        return self.tri.points[p_idx0], self.tri.points[p_idx1], self.tri.points[p_idx2]


# 首先需要一个函数，给定一个点，返回所有相邻点
# 这将决定 Phi 的相邻 Phi
def get_neighbor_points(point):
    # 邻居点包含自身
    unit_dict = Point2Unit[point]
    points = []
    for unit, idx in unit_dict.items():
        tri, _ = unit.tri, unit.point_idx
        idx1 = (idx+1)%3
        idx2 = (idx+2)%3
        p1 = tri.points[idx1]
        p2 = tri.points[idx2]
        if p1 not in points:
            points.append(p1)
        if p2 not in points:
            points.append(p2)
    points.append(point)
    return points

def get_nabla(unit):
    # return: (2,)
    idx = unit.point_idx
    tri = unit.tri
    n = tri.getNByIdx(idx)
    H = tri.getHByIdx(idx)
    nabla = n / H
    return nabla

# 我们先定义一个函数，计算两个单位元的D内积
def get_Dphi_dot(unit1, unit2, kappa):
    assert unit1.tri == unit2.tri
    tri = unit1.tri
    nabla1 = get_nabla(unit1)
    nabla2 = get_nabla(unit2)
    area = tri.area
    kp = kappa(tri.center)
    dot = kp * area * (nabla1 @ nabla2)
    return dot

def get_phi_dot(unit1, unit2):
    assert unit1.tri == unit2.tri
    tri = unit1.tri
    P1 = unit1.getPoint()[0]
    P2 = unit2.getPoint()[0]
    if P1 != P2:
        P0 = tri.getAnotherPoint(P1,P2)
        p1 = P1.xy - P0.xy
        p2 = P2.xy - P0.xy
        det = np.abs(p1[0]*p2[1] - p2[1]*p1[0])
        return 1/24 * det
    else:
        # P1: (1,0)
        P0, P2 = tri.getAnotherTwoPoint(P2)
        p1 = P1.xy - P0.xy
        p2 = P2.xy - P0.xy
        det = np.abs(p1[0]*p2[1] - p2[1]*p1[0])
        return 1/12 * det


def get_Phi_dot(p1, p2):
    if p1 == p2:
        dot = 0
        for unit1, idx1 in Point2Unit[p1].items():
            dot += get_phi_dot(unit1, unit1)
        return dot
    unit_dict1 = Point2Unit[p1]
    tri_units = {} # {tri: (unit1, unit2)}
    for unit1, idx1 in unit_dict1.items():
        tri1 = unit1.tri
        if p2 in tri1.points:
            unit2 = [unit2 for unit2 in Point2Unit[p2].keys() if unit2.tri == tri1][0]
            tri_units[tri1] = (unit1, unit2)
        else:
            pass
    dot = 0
    for tri, (unit1, unit2) in tri_units.items():
        dot += get_phi_dot(unit1, unit2)
    return dot
    
    

def get_partial_phi_dot(unit1, unit2, dim_choice):
    '''
    get_partial_phi_dot 的 Docstring
    Example:
    :param unit1: Unit
    :param unit2: Unit
    :param dim_choice: Tuple (dim1,dim2) : $\partial_{dim1}\phi_1 * \partial_{dim2}\phi_2$
    '''
    assert unit1.tri == unit2.tri
    tri = unit1.tri
    area = tri.area
    nabla1 = get_nabla(unit1)
    nabla2 = get_nabla(unit2)
    d_phi1 = nabla1[dim_choice[0]]
    d_phi2 = nabla2[dim_choice[1]]
    dot = area * d_phi1 * d_phi2
    return dot

# 需要一个函数，给定两个点（Phi）返回它们的内积
def get_DPhi_dot(p1, p2, kappa):
    if p1 == p2:
        dot = 0
        for unit1, idx1 in Point2Unit[p1].items():
            dot += get_Dphi_dot(unit1, unit1, kappa)
        return dot
    unit_dict1 = Point2Unit[p1]
    tri_units = {} # {tri: (unit1, unit2)}
    for unit1, idx1 in unit_dict1.items():
        tri1 = unit1.tri
        if p2 in tri1.points:
            unit2 = [unit2 for unit2 in Point2Unit[p2].keys() if unit2.tri == tri1][0]
            tri_units[tri1] = (unit1, unit2)
        else:
            pass
    dot = 0
    for tri, (unit1, unit2) in tri_units.items():
        dot += get_Dphi_dot(unit1, unit2, kappa)
    return dot



def get_partial_Phi_dot(p1, p2, dim_choice):
    if p1 == p2:
        dot = 0
        for unit1, idx1 in Point2Unit[p1].items():
            dot += get_partial_phi_dot(unit1, unit1, dim_choice)
        return dot
    unit_dict1 = Point2Unit[p1]
    tri_units = {} # {tri: (unit1, unit2)}
    for unit1, idx1 in unit_dict1.items():
        tri1 = unit1.tri
        if p2 in tri1.points:
            unit2 = [unit2 for unit2 in Point2Unit[p2].keys() if unit2.tri == tri1][0]
            tri_units[tri1] = (unit1, unit2)
        else:
            pass
    dot = 0
    for tri, (unit1, unit2) in tri_units.items():
        dot += get_partial_phi_dot(unit1, unit2, dim_choice)
    return dot

# 为了获得常数项，我们定义一个获得Phi总面积的函数
def get_Phi_area(point):
    unit_dict = Point2Unit[point]
    area = 0
    for unit in unit_dict.keys():
        area += unit.tri.area
    return area

def get_units(all_tri):
    for tri in all_tri:
        tri.calVertical()
        tri.calArea()
        tri.calCenter()
        for i in range(3):
            _ = Unit(tri, i)
            

def is_boundary(point):
    tris = [unit.tri for unit in Point2Unit[point].keys()]
    # tris 围绕 point 是闭合的，则不是边界
    for tri in tris:
        p1,p2 = tri.getAnotherTwoPoint(point)
        tri1 = tri.getTri(p1)
        tri2 = tri.getTri(p2)
        if tri1 is None or tri2 is None:
            return True
    return False

def approx0(x, eps=1e-10):
    return -eps <= x <= eps


def ColEqRaw(AllEq):
    all_x = []
    for point_j in AllEq.keys():
        for point_i in AllEq[point_j].keys():
            if point_i == 'const':
                continue
            if point_i not in all_x:
                all_x.append(point_i)
    equation_num = len(all_x)
    return equation_num==len(all_x)


def solve_sparse(AllEq):
    '''
    能把AllEq消元成对角
    注意 AllEq 不包含边界
    :param AllEq: {point_j: {point_i: coef, const: b}}
    '''
    assert ColEqRaw(AllEq), "行列数不相等"
    print("开始消元")
    # gauss消元法
    AllPoint = list(AllEq.keys())
    for point_j in AllPoint:
        c_jj = AllEq[point_j][point_j]
        assert c_jj > 0
        # 先把 Eq[j][j] 化成 1
        for point_i in AllEq[point_j].keys():
            # 这里包括了 const
            AllEq[point_j][point_i] /= c_jj
        # 然后去消去这一列其他人
        for point_j2 in AllPoint:
            if point_j2 == point_j:
                continue
            # 如果不为0，把 point_j 那一行 加到 Point_j2 行
            if not AllEq[point_j2].get(point_j):
                continue
            if approx0(AllEq[point_j2][point_j]):
                AllEq[point_j2].pop(point_j)
                continue
            c_j2j = AllEq[point_j2][point_j]
            # 只要操作 AllEq[point_j] 这些列
            for point_i in AllEq[point_j]:
                if AllEq[point_j2].get(point_i):
                    AllEq[point_j2][point_i] -=\
                        AllEq[point_j][point_i] * c_j2j
                else:
                    AllEq[point_j2][point_i] =\
                        - AllEq[point_j][point_i] * c_j2j
    print("消元完成")

def success_solve(AllEq):
    '''
    检查是否成功对角化
    '''
    for point_j in AllEq.keys():
        for point_i in AllEq[point_j].keys():
            if point_i == 'const':
                continue
            if point_i==point_j:
                if AllEq[point_j][point_i] != 1:
                    return False
                else:
                    continue
            if AllEq[point_j][point_i] != 0:
                return False
    return True

def ode_2D1T(c, f, g, u0, dt, t1, theta,
             dh, xy_boundary):
    '''
    ode_2D1T 的 Docstring
    
    :param c: lambda xy, t: float
    :param f: lambda xy, t: float
    :param g: lambda xy, t: float
    :param u0: lambda xy: float
    :param dt: float
    :param t1: float
    '''
    AllUnit.clear()
    # 网格
    all_tri, all_point = get_tri(xy_boundary, dh, need_disp=False)
    get_units(all_tri)

    t = 0
    Point2AllValue = {point:[(t, u0(point.xy))] for point in all_point}

    while t < t1:
        AllEq = {} # 未知数是 u^{n+1}
        for point_j in all_point:
            AllEq[point_j] = {}
            AllEq[point_j]['const'] =\
                get_Phi_area(point_j)/3 * (\
                theta*f(point_j.xy,t+dt)+\
                (1-theta)*f(point_j.xy,t)\
                ) * dt
            points_neighbor = get_neighbor_points(point_j)
            for point_i in points_neighbor:
                dot_D = get_DPhi_dot(point_i, point_j, lambda _: 1)
                dot = get_Phi_dot(point_i, point_j)
                c_i = c(point_i.xy, t)
                
                AllEq[point_j][point_i] = dot + dt*theta* dot_D* c_i
                AllEq[point_j]['const'] +=\
                    Point2AllValue[point_i][-1][1] * (\
                    dot - dt*(1-theta)* c_i * dot_D \
                    )
        for point_j in all_point:
            if is_boundary(point_j):
                AllEq.pop(point_j)
        for point_j in AllEq.keys():
            AllEqKeys = list(AllEq[point_j].keys())
            for point_i in AllEqKeys:
                if point_i == 'const':
                    continue
                if is_boundary(point_i):
                    AllEq[point_j]['const'] -=\
                        AllEq[point_j][point_i] *\
                            g(point_i.xy,t+dt)
                    AllEq[point_j].pop(point_i)
                    
        solve_sparse(AllEq)
        assert success_solve(AllEq), AllEq
        
        for point_j in all_point:
            if is_boundary(point_j):
                Point2AllValue[point_j].append((t+dt, g(point_j.xy,t+dt)))
            else:
                Point2AllValue[point_j].append((t+dt, AllEq[point_j]['const']))
        t += dt
        print(f'solved t = {t}, dt = {dt}, t1 = {t1}')

    return Point2AllValue


def plot_solve(Point2Value):
    # point 的 位移 (u1,u2)
    xs, ys, zs = [], [], []

    for point, value in Point2Value.items():
        x, y = point.xy
        xs.append(x)
        ys.append(y)
        zs.append(value)

    # 2 个 subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter(xs, ys, zs, color='blue', s=10, label='cal u')
    ax.legend()

    

def plot_real(Point2Value, u_real):
    ax = plt.figure().add_subplot(projection='3d')
    xs, ys, zs = [], [], []
    loss = 0
    
    for point, value in Point2Value.items():
        x, y = point.xy
        xs.append(x)
        ys.append(y)
        
        z_pred = value
        z_real = u_real(point.xy)

        loss += (z_pred - z_real)**2
        
        zs.append(z_real)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='red', s=10, label='real u')
    ax.legend()

    return loss / len(Point2Value)


def eval_solve(Point2AllValue, u_real):
    Point2AllValue_real = {}
    l2 = 0
    for point, list_t_u in Point2AllValue.items():
        Point2AllValue_real[point] = []
        for t, u in list_t_u:
            Point2AllValue_real[point].append((t, u_real(point.xy, t)))
            l2 += (u - u_real(point.xy, t))**2


    l2 /= len(Point2AllValue) * len(list_t_u)

    print(f'l2 = {l2}')

    assert len(list_t_u) == int(t1//dt)+1

    energy_solved_list = []
    energy_real_list = []

    for t in range(int(t1//dt)+1):
        energy_solved = 0
        energy_real = 0
        for point, list_t_u in Point2AllValue.items():
            energy_solved += list_t_u[t][1] ** 2
            energy_real += Point2AllValue_real[point][t][1] ** 2
        energy_solved_list.append(energy_solved)
        energy_real_list.append(energy_real)

    show_idx = -1
    plt.subplot(121)
    plt.plot(np.arange(0,t1+1e-5,dt)[:show_idx], energy_solved_list[:show_idx], label='energy solved')
    plt.legend()
    plt.subplot(122)
    plt.plot(np.arange(0,t1+1e-5,dt)[:show_idx], energy_real_list[:show_idx], label='energy real')
    plt.legend()
    plt.savefig(os.path.join('output', f'xyt_dh{dh}_dt{dt}_theta{theta}.png'))
    plt.close()


u_real = lambda xy, t: np.exp(xy[0]+xy[1]+t)

c = lambda xy, t: 2
f = lambda xy, t: - np.exp(xy[0]+xy[1]+t)

u0 = lambda xy: np.exp(xy[0]+xy[1])

g = lambda xy, t: np.exp(xy[0]+xy[1]+t)

dh = 1/16

theta = 1
dt = 4 * dh**2

xy_boundary = np.array([[0,0], [1,0], [1,1], [0,1]])
t1 = 1

Point2AllValue = ode_2D1T(c, f, g, u0, dt, t1, theta,
             dh, xy_boundary)

eval_solve(Point2AllValue, u_real)