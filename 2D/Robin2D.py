import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mesh_div import get_tri


# {(tri,point_idx): Unit}
AllUnit = {}
class Unit:
    def __init__(self, tri, point_idx):
        AllUnit[(tri,point_idx)] = self
        self.tri = tri
        self.point_idx = point_idx
    def getPoint(self):
        p_idx0 = self.point_idx
        p_idx1 = (self.point_idx+1)%3
        p_idx2 = (self.point_idx+2)%3
        return self.tri.points[p_idx0], self.tri.points[p_idx1], self.tri.points[p_idx2]


# xy: (2,)
def get_matrix(xy_boundary, kappa, b, g, f, dh):
    '''
    kappa = lambda xy: 2
    b = lambda xy: 1
    g = lambda xy: np.exp(xy[0]+xy[1])
    f = lambda xy: np.exp(xy[0]+xy[1])
    # 定义边界
    xy_boundary = np.array([[0,0], [1,0], [1,1], [0,1]])
    # 网格半径
    dh = 0.1
    '''

    # 网格
    all_tri = get_tri(xy_boundary, dh, need_disp=False)

    # 三角形上一次元
    # 每个三角形有三个基函数

    for tri in all_tri:
        tri.calVertical()
        tri.calArea()
        tri.calCenter()
        for i in range(3):
            _ = Unit(tri, i)

    # 每个Unit_j对应一个方程
    # 由于方程的系数是稀疏的，用字典保存
    # 方程的key是Unit_i的分量，value是Unit_i分量的系数
    # {Unit_j: {Unit_i: coef}}
    AllEq = {}
    for k,unit_j in AllUnit.items():
        AllEq[unit_j] = {}
        tri,idx_j = k
        idx1 = (idx_j+1)%3
        idx2 = (idx_j+2)%3
        # 其它两个unit
        unit1 = AllUnit[(tri,idx1)]
        unit2 = AllUnit[(tri,idx2)]
        
        # target 1
        tri.calVertical()
        tri.calArea()
        kp = kappa(tri.center)
        area = tri.area
        n_j = [tri.n1, tri.n2, tri.n3][idx_j]
        H_j = [tri.H1, tri.H2, tri.H3][idx_j]
        # 还要和自己！
        for unit_i in [unit_j,unit1,unit2]:
            AllEq[unit_j][unit_i] = 0
            
            idx_i = unit_i.point_idx
            n_i = [tri.n1, tri.n2, tri.n3][idx_i]
            H_i = [tri.H1, tri.H2, tri.H3][idx_i]
            coef1 = kp * area * (n_j @ n_i) / (H_j * H_i)
            AllEq[unit_j][unit_i] += coef1
            
            coef2 = 0
            if None in tri.neighborTri:
                bound_idx = tri.neighborTri.index(None)
                if idx_i == bound_idx or idx_j == bound_idx:
                    pass
                else:
                    # 都在边界上
                    P1,P2 = tri.getAnotherPointByIdx(bound_idx)
                    P1P2 = P2.xy - P1.xy
                    P1P2_mag = np.linalg.norm(P1P2)
                    P1P2_center = (P1.xy+P2.xy)/2
                    b_xi = b(P1P2_center)
                    if idx_i == idx_j:
                        coef2 = b_xi * (P1P2_mag**2/3)
                    else:
                        coef2 = b_xi * (P1P2_mag/2 - P1P2_mag**2/3)
            AllEq[unit_j][unit_i] += coef2
            
        AllEq[unit_j]['const'] = 0
        
        coef3 = 0
        if None in tri.neighborTri:
            bound_idx = tri.neighborTri.index(None)
            if idx_j == bound_idx:
                pass
            else:
                # 自己在边界上
                P1,P2 = tri.getAnotherPointByIdx(bound_idx)
                P1P2 = P2.xy - P1.xy
                P1P2_mag = np.linalg.norm(P1P2)
                P1P2_center = (P1.xy+P2.xy)/2
                g_xi = g(P1P2_center)
                coef3 = g_xi * P1P2_mag / 2
        
        AllEq[unit_j]['const'] += coef3
        
        f_xi = f(tri.center)
        coef4 = f_xi * tri.area / 3
        AllEq[unit_j]['const'] += coef4
    return AllEq


# 迭代
# x = x + k*(Ax - b)
def solve_matrix(AllEq, num_epoch=1000):
    # global AllUnit
    temp_solve = np.zeros(len(AllUnit))
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
        loss = Ax - b
        temp_solve -= 0.1 * loss
        print(loss.mean())
    Point2Value = {}
    for unit, idx in Unit2Idx.items():
        point,_,_ = unit.getPoint()
        if Point2Value.get(point) is None:
            Point2Value[point] = temp_solve[idx]
        else:
            Point2Value[point] += temp_solve[idx]
    return Point2Value


def plot_solve(Point2Value):
    ax = plt.figure().add_subplot(projection='3d')
    for point, value in Point2Value.items():
        x,y = point.xy
        # plot3
        ax.scatter(x, y, value, zdir='z', label='points in (x, z)')
    # plt.show()
    

kappa = lambda xy: 1
b = lambda xy: 1
g = lambda xy: np.exp(xy[0]+xy[1])
f = lambda xy: np.exp(xy[0]+xy[1])
# 定义边界
xy_boundary = np.array([[0,0], [1,0], [1,1], [0,1]])
# 网格半径
dh = 0.1



AllEq = get_matrix(xy_boundary, kappa, b, g, f, dh)
Point2Value = solve_matrix(AllEq, num_epoch=1000)
plot_solve(Point2Value)
plt.show()