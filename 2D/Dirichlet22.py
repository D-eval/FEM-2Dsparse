import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mesh_div import get_tri
import torch
import torch.nn.functional as F

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
def get_matrix(xy_boundary, f1, f2, dh,
               lam, mu, g1, g2):
    '''
    Example:
    kappa = lambda xy: 2
    g = lambda xy: np.exp(xy[0]+xy[1])
    f = lambda xy: np.exp(xy[0]+xy[1])
    # 定义边界
    xy_boundary = np.array([[0,0], [1,0], [1,1], [0,1]])
    # 网格半径
    dh = 0.1
    '''
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
        AllEq[point] = {0:{},1:{}}
        f1_xi = f1(point.xy)
        f2_xi = f2(point.xy)
        area = get_Phi_area(point)
        volume = area / 3
        AllEq[point][0]['const'] = - f1_xi * volume
        AllEq[point][1]['const'] = - f2_xi * volume
        
        points_neighbor = get_neighbor_points(point)
        
        for point_neighbor in points_neighbor:
            dot_11 = get_partial_Phi_dot(point_neighbor, point, (0,0))
            dot_22 = get_partial_Phi_dot(point_neighbor, point, (1,1))
            dot_12 = get_partial_Phi_dot(point_neighbor, point, (0,1))
            dot_21 = get_partial_Phi_dot(point_neighbor, point, (1,0))
            # assert dot_12 == dot_21
            AllEq[point][0][point_neighbor] = {} # malloc
            AllEq[point][1][point_neighbor] = {} # malloc
            
            AllEq[point][0][point_neighbor][0] = (2*mu+lam)*dot_11 + mu*dot_22
            AllEq[point][0][point_neighbor][1] = lam*dot_21 + mu*dot_12
            AllEq[point][1][point_neighbor][1] = (2*mu+lam)*dot_22 + mu*dot_11
            AllEq[point][1][point_neighbor][0] = lam*dot_12 + mu*dot_21
 
    #apply_dirichlet(AllEq, g1, g2)
    return AllEq, all_point, all_tri

def apply_dirichlet(AllEq, g1, g2):
    # Dirichlet 边界条件
    for point in AllEq.keys():
        # is_boundary = True if len([1 for tri in [unit.tri for unit in Point2Unit[point].keys()] if None in tri.neighborTri]) >= 0 else False
        tris = [unit.tri for unit in Point2Unit[point].keys()]
        # 这个点只要参与过带 boundary 边的三角形，就视为边界点
        is_boundary = any(None in tri.neighborTri for tri in tris)
        if is_boundary:
            AllEq[point][0] = {}
            AllEq[point][0][point] = {0:1,
                                      1:0}
            AllEq[point][0]['const'] = g1(point.xy)
            
            AllEq[point][1] = {}
            AllEq[point][1][point] = {0:0,
                                      1:1}
            AllEq[point][1]['const'] = g2(point.xy)
            

def solve_matrix_torch(AllEq, all_point, num_epoch=3000, lr=4e-4, device='cpu'):
    """
    return: Point2Value: {Point: (val1, val2)}
    """
    tau = 0.9999
    Units = list(AllEq.keys())
    n = len(all_point)
    # 批量映射 Index
    Point2Idx = {p: i for i, p in enumerate(all_point)}
    # 初始 x（解）
    x = torch.zeros((n,2), dtype=torch.float64, device=device, requires_grad=True)
    # b 向量
    b = torch.zeros((n,2), dtype=torch.float64, device=device)
    for p, ip in Point2Idx.items():
        b[ip,0] = AllEq[p][0]['const']
        b[ip,1] = AllEq[p][1]['const']
    # Adam 优化器（超快收敛）
    optimizer = torch.optim.Adam([x], lr=lr)
    for epoch in range(num_epoch):
        # --- 构造 Ax（稀疏计算，不建立矩阵）---
        Ax = torch.zeros_like(b) # (n,2)
        for p, ip in Point2Idx.items():
            for d1 in [0,1]:
                val = 0
                for pj, coef_d2 in AllEq[p][d1].items():
                    if pj == 'const':
                        continue
                    coef1 = coef_d2[0]
                    coef2 = coef_d2[1]
                    jp = Point2Idx[pj]
                    val += coef1 * x[jp,0] + coef1 * x[jp,1]
                Ax[ip,d1] = val
        # --- loss = 1/2 * ||Ax - b||^2 ---
        error = (Ax - b)**2
        error = error.view(-1)
        choice = F.softmax(error * 100, dim=0)
        loss = torch.sum(error * choice)
        # loss = torch.mean(error)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss = {loss.item()}")
            # print(f"choiceMax: {choice.max().item()}")
        # lr *= tau
    # ===== 把Unit上的解合并成 Point 值 =====
    Point2Value = {}


    for point, idx in Point2Idx.items():
        val1 = x[idx,0].item()
        val2 = x[idx,1].item()
        Point2Value[point] = (val1, val2)
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
    xs, ys, z1s, z2s = [], [], [], []

    for point, value in Point2Value.items():
        x, y = point.xy
        xs.append(x)
        ys.append(y)
        z1s.append(value[0])
        z2s.append(value[1])

    # 2 个 subplot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')    
    ax.scatter(xs, ys, z1s, color='blue', s=10, label='cal u1')
    ax.legend()
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(xs, ys, z2s, color='blue', s=10, label='cal u2')
    ax.legend()

    

def plot_real(Point2Value, u1_real, u2_real):
    ax = plt.figure().add_subplot(projection='3d')
    xs, ys, z1s, z2s = [], [], [], []
    loss = 0
    
    for point, value in Point2Value.items():
        x, y = point.xy
        xs.append(x)
        ys.append(y)
        
        z1_pred = value[0]
        z2_pred = value[1]
        z1_real = u1_real(point.xy)
        z2_real = u2_real(point.xy)

        loss += (z1_pred - z1_real)**2 + (z2_pred - z2_real)**2
        
        z1s.append(z1_real)
        z2s.append(z2_real)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(xs, ys, z1s, color='red', s=10, label='real u1')
    ax.legend()
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(xs, ys, z2s, color='red', s=10, label='real u2')
    ax.legend()

    return loss / len(Point2Value)

def solve_Dirichlet(xy_boundary, g1, g2, f1, f2, lam, mu, dh, num_epoch):
    AllEq, all_point, all_tri = get_matrix(xy_boundary, f1, f2, dh, lam, mu, g1, g2)
    Point2Value = solve_matrix_torch(AllEq, all_point, num_epoch=num_epoch)
    return Point2Value, all_point, all_tri



g1 = lambda xy: 0
g2 = lambda xy: 0
lam = 1
mu = 2

pi = np.pi
sin = np.sin
cos = np.cos
f1 = lambda xy: (lam+3*mu) * (-pi**2 * sin(pi*xy[0]) * sin(pi*xy[1])) + (lam+mu) * (2*xy[0]-1) * (2*xy[1]-1)
f2 = lambda xy: (lam+2*mu) * 2*(xy[0]**2-xy[0]) + mu * 2*(xy[1]**2-xy[1]) + (lam+mu) * pi**2 * cos(pi*xy[0]) * cos(pi*xy[1])

xy_boundary = np.array([[0,0], [1,0], [1,1], [0,1]])
dh = 0.1

Point2Value, all_point, all_tri = solve_Dirichlet(xy_boundary, g1, g2, f1, f2, lam, mu, dh, num_epoch=3000)

plot_solve(Point2Value)
plt.savefig('./output/D22_solve.png')

u1_real = lambda xy: sin(pi*xy[0]) * sin(pi*xy[1])
u2_real = lambda xy: xy[0]*(xy[0]-1)*xy[1]*(xy[1]-1)
l2 = plot_real(Point2Value, u1_real, u2_real)
plt.savefig('./output/D22_real.png')

print(f"L2 loss: {l2}")
