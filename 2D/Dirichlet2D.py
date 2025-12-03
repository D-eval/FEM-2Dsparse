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
    return points

# 我们先定义一个函数，计算两个单位元的D内积
def get_Dphi_dot(unit1, unit2, kappa):
    assert unit1.tri == unit2.tri
    tri = unit1.tri
    idx1 = unit1.point_idx
    idx2 = unit2.point_idx
    n1 = tri.getNByIdx(idx1)
    H1 = tri.getHByIdx(idx1)
    nabla1 = n1 / H1
    n2 = tri.getNByIdx(idx2)
    H2 = tri.getHByIdx(idx2)
    nabla2 = n2 / H2
    area = tri.area
    kp = kappa(tri.center)
    dot = kp * area * (nabla1 @ nabla2)
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
def get_matrix(xy_boundary, kappa, g, f, dh):
    '''
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
        AllEq[point] = {}
        points_neighbor = get_neighbor_points(point)
        AllEq[point][point] = get_DPhi_dot(point, point, kappa)
        for point_neighbor in points_neighbor:
            AllEq[point][point_neighbor] = get_DPhi_dot(point, point_neighbor, kappa)
        area = get_Phi_area(point)
        f_xi = f(point.xy)
        volume = area / 3
        AllEq[point]['const'] = - f_xi * volume
    apply_dirichlet(AllEq, g)
    return AllEq, all_point, all_tri

def apply_dirichlet(AllEq, g):
    # Dirichlet 边界条件
    for point in AllEq.keys():
        # is_boundary = True if len([1 for tri in [unit.tri for unit in Point2Unit[point].keys()] if None in tri.neighborTri]) >= 0 else False
        tris = [unit.tri for unit in Point2Unit[point].keys()]
        # 这个点只要参与过带 boundary 边的三角形，就视为边界点
        is_boundary = any(None in tri.neighborTri for tri in tris)
        if is_boundary:
            AllEq[point] = {point: 1, "const": g(point.xy)}

# 迭代
# x = x + k*(Ax - b)

def solve_matrix(AllEq, num_epoch=5000):
    """
    梯度下降 解 稀疏线性方程组
    """
    # global AllUnit
    temp_solve = np.random.rand(len(AllUnit))
    b = np.zeros(len(AllUnit))
    for idx_j, unit_j in enumerate(AllUnit.values()):
        b_j = AllEq[unit_j]['const']
        b[idx_j] = b_j
    Unit2Idx = {unit: idx for idx, unit in enumerate(AllUnit.values())}
    Ax = np.zeros(len(AllUnit))
    # 迭代法求解 稀疏线性方程组
    for epoch in range(num_epoch):
        for idx_j, unit_j in enumerate(AllUnit.values()):
            temp_sum = 0
            for unit_i, coef in AllEq[unit_j].items():
                if unit_i == 'const':
                    continue
                temp_sum += coef * temp_solve[Unit2Idx[unit_i]]
            Ax[idx_j] = temp_sum
        error = Ax - b
        temp_solve -= 0.0001 * error
        print(error.mean())
    Point2Value = {}
    for unit, idx in Unit2Idx.items():
        point,_,_ = unit.getPoint()
        if Point2Value.get(point) is None:
            Point2Value[point] = temp_solve[idx]
        else:
            Point2Value[point] += temp_solve[idx]
    return Point2Value


def solve_matrix_torch(AllEq, all_point, num_epoch=5000, lr=4e-3, device='cpu'):
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
    for p, ip in Point2Idx.items():
        b[ip] = AllEq[p]['const']
    # Adam 优化器（超快收敛）
    optimizer = torch.optim.Adam([x], lr=lr)
    for epoch in range(num_epoch):
        # --- 构造 Ax（稀疏计算，不建立矩阵）---
        Ax = torch.zeros_like(b)
        for p, ip in Point2Idx.items():
            val = 0.0
            for pj, coef in AllEq[p].items():
                if pj == 'const':
                    continue
                jp = Point2Idx[pj]
                val += coef * x[jp]
            Ax[ip] = val
        # --- loss = 1/2 * ||Ax - b||^2 ---
        error = (Ax - b)**2
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
    ax = plt.figure().add_subplot(projection='3d')
    xs, ys, zs = [], [], []

    for point, value in Point2Value.items():
        x, y = point.xy
        xs.append(x)
        ys.append(y)
        zs.append(value)

    ax.scatter(xs, ys, zs, color='blue', s=10, label='FEM solution')
    ax.legend()
    return ax

def plot_real(Point2Value, u_real):
    ax = plt.figure().add_subplot(projection='3d')
    xs, ys, zs = [], [], []

    for point, value in Point2Value.items():
        x, y = point.xy
        xs.append(x)
        ys.append(y)
        zs.append(u_real(point.xy))

    ax.scatter(xs, ys, zs, color='red', s=10, label='Exact solution')
    ax.legend()
    return ax

def plot_compare(Point2Value, u_real):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zfem, zreal = [], [], [], []
    point2L1 = {}
    for p, v in Point2Value.items():
        x, y = p.xy
        xs.append(x)
        ys.append(y)
        zfem.append(v)
        z_real = u_real(p.xy)
        zreal.append(z_real)
        point2L1[p] = np.abs(v - z_real)

    ax.scatter(xs, ys, zfem, color='blue',  s=12, label='FEM')
    ax.scatter(xs, ys, zreal, color='red', s=12, label='Exact')

    ax.legend()
    return point2L1, ax




def solve_Dirichlet(xy_boundary, kappa, g, f, dh, num_epoch):
    AllEq, all_point, all_tri = get_matrix(xy_boundary, kappa, g, f, dh)
    Point2Value = solve_matrix_torch(AllEq, all_point, num_epoch=num_epoch)
    return Point2Value, all_point, all_tri


