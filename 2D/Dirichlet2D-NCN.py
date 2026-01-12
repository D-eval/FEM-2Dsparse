'''
Dirichlet2D-NCN
NCN:
\nabla c \nabla u = f
c 是矩阵
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
            

# xy: (2,)
def get_matrix(xy_boundary, f, dh, c, g):
    AllUnit.clear()
    # 网格
    all_tri, all_point = get_tri(xy_boundary, dh, need_disp=False)
    # plt.show()
    get_units(all_tri)
    # 每个 Point 对应一个方程
    # 由于方程的系数是稀疏的，用字典保存
    # 方程的key是 Point_neighbor 的分量，value 是 Point_neighbor 分量的系数
    # {Point: {Point: coef}}
    AllEq = {}
    # 方程数: point
    # 未知数个数: point
    for point, unit_dict in Point2Unit.items():
        # 这表示一个 Phi
        AllEq[point] = {}
        xy_i = point.xy
        f_xi = f(xy_i)
        area = get_Phi_area(point)
        volume = area / 3
        AllEq[point]['const'] = - f_xi * volume

        points_neighbor = get_neighbor_points(point)



        for point_neighbor in points_neighbor:
            xy_j = point_neighbor.xy
            xy = (xy_i + xy_j) / 2
            dot_11 = get_partial_Phi_dot(point_neighbor, point, (0,0))
            dot_22 = get_partial_Phi_dot(point_neighbor, point, (1,1))
            dot_12 = get_partial_Phi_dot(point_neighbor, point, (0,1))
            dot_21 = get_partial_Phi_dot(point_neighbor, point, (1,0))
            # assert dot_12 == dot_21
            coef =  c[0][0](xy) * dot_11 +\
                c[1][0](xy) * dot_12 +\
                c[0][1](xy) * dot_21 +\
                c[1][1](xy) * dot_22
            AllEq[point][point_neighbor] = coef


    apply_dirichlet(AllEq, g)
    return AllEq, all_point, all_tri


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

def apply_dirichlet(AllEq, g):
    # Dirichlet 边界条件
    for point in AllEq.keys():
        if is_boundary(point):
            xy = point.xy
            AllEq[point]["const"] = g(xy)
            AllEq[point][point] = 1


def solve_matrix_torch(AllEq, all_point, num_epoch=5000, lr=3e-2, device='cpu', show_train=True):
    """
    使用 PyTorch（Adam）求解 AllEq 定义的线性系统 A x = b
    AllEq: { unit_j : { unit_i: coef, ..., 'const': b_j } }
    """
    tau = 0.9999
    Units = list(AllEq.keys())
    m = len(Units)
    n = len(all_point)
    # 批量映射 Index
    Point2Idx = {p: i for i, p in enumerate(all_point)}
    # 初始 x（解）
    x = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    # b 向量
    b = torch.zeros(n, dtype=torch.float64, device=device)
    mask_bool = torch.ones(n, dtype=bool)
    for p, ip in Point2Idx.items():
        b[ip] = AllEq[p]['const']
        if is_boundary(p):
            x.data[ip] = AllEq[p]['const']
            mask_bool[ip] = 0
    optimizer = torch.optim.Adam([x], lr=lr)
    for epoch in range(num_epoch):
        # --- 构造 Ax（稀疏计算，不建立矩阵）---
        Ax = torch.zeros_like(b)
        for p, ip in Point2Idx.items():
            if is_boundary(p):
                x.data[ip] = AllEq[p]['const']
            val = 0.0
            for pj, coef in AllEq[p].items():
                if pj == 'const':
                    continue
                jp = Point2Idx[pj]
                val += coef * x[jp]
            # if is_boundary(p):
            #     assert val == b[ip], f"{val}, {b[ip]}"
            Ax[ip] = val
        # --- loss = 1/2 * ||Ax - b||^2 ---
        error = (Ax - b)**2
        # choice = F.softmax(error * 100, dim=0)
        # loss = torch.sum(error * choice)
        loss = torch.mean(error[mask_bool])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 and show_train:
            print(f"Epoch {epoch}, loss = {loss.item()}")
            # print(f"choiceMax: {choice.max().item()}")
        # lr *= tau
    # ===== 把Unit上的解合并成 Point 值 =====
    Point2Value = {}


    for point, idx in Point2Idx.items():
        val = x[idx].item()
        Point2Value[point] = val
    # print(len(Point2Value))
    # print(len(all_point))
    return Point2Value

'''
def plot_solve(Point2Value):
    ax = plt.figure().add_subplot(projection='3d')
    for point, value in Point2Value.items():
        x,y = point.xy
        # plot3
        ax.scatter(x, y, value, color='blue', s=10, label='FEM solution')
    # plt.show()

def plot_real(Point2Value, u_real):
    ax = plt.figure().add_subplot(projection='3d')
    for point, value in Point2Value.items():
        x,y = point.xy
        z = u_real(point.xy)
        # plot3
        ax.scatter(x, y, z, color='red', s=10, label='Exact solution')
    # plt.show()
'''

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

def solve_Dirichlet(xy_boundary, g, f, dh, c, num_epoch, show_train=False):
    AllEq, all_point, all_tri = get_matrix(xy_boundary, f, dh, c, g)
    Point2Value = solve_matrix_torch(AllEq, all_point, num_epoch=num_epoch, show_train=show_train)
    return Point2Value, all_point, all_tri


'''

c = [[
    lambda xy: 1,
    lambda xy: 2],[
    lambda xy: 3,
    lambda xy: 4
    ]]

f = lambda xy: 10 * np.exp(xy[0]+xy[1])

g = lambda xy: np.exp(xy[0]+xy[1])

xy_boundary = np.array([[0,0], [1,0], [1,1], [0,1]])
dh = 1/8

Point2Value, all_point, all_tri = solve_Dirichlet(xy_boundary, g, f, dh, c, num_epoch=1000)

plot_solve(Point2Value)

plt.savefig(os.path.join(".","output","D2NCN_solve.png"))

u_real = lambda xy: np.exp(xy[0]+xy[1])


l2 = plot_real(Point2Value, u_real)
plt.savefig(os.path.join(".","output","D2NCN_real.png"))

print(f"L2 loss: {l2}")

'''