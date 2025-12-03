import numpy as np
import matplotlib.pyplot as plt


def epsEq(a1,a2,eps):
    return a1-eps<=a2<=a1+eps

def allDimEpsEq(a1,a2,eps):
    # a1, a2: (n,)
    res = True
    for i in range(a1.shape[0]):
        if not epsEq(a1[i],a2[i],eps):
            res = False
            break
    return res


# 定义边界形状，逆时针 (n_boundary,2)
rot_90 = np.array([[0,-1], [1,0]])
# 网格划分
def angleSum(ps, xy_boundary):
    # ps : (n,2)
    # xy_boundary : (n_boundary,2)
    PA = ps[:,None,:] - xy_boundary[None,:,:] # (n,n_boundary,2)
    PA_magnitude = np.linalg.norm(PA,axis=-1,keepdims=True) # (n,n_boundary)
    PA = PA / PA_magnitude # (n,n_boundary,2)
    # 逆时针转一个
    PB = np.roll(PA,1,axis=1) # (n,n_boundary,2)
    PC_magnitude = (PA * PB).sum(-1) # (n,n_boundary)
    PC = PC_magnitude[:,:,None] * PA # (n,n_boundary,2)
    CB = PB - PC # (n,n_boundary,2)
    PA90 = PA @ rot_90.T # (n,n_boundary,2)
    CB_magnitude = (CB * PA90).sum(-1) # (n,n_boundary)
    theta = np.arctan2(CB_magnitude, PC_magnitude) # (n,n_boundary)
    theta_sum = theta.sum(1) # (n,)
    return theta_sum

def inner_choice(ps, xy_boundary):
    thetas = angleSum(ps, xy_boundary)
    n_surrounding = np.round(thetas / (2*np.pi))
    inner_idx = n_surrounding == -1
    ps = ps[inner_idx]
    return ps



AllPoint = []
class Point:
    def __init__(self, x, y):
        self.xy = np.array([x,y])
        AllPoint.append(self)

ori_point1 = Point(-3,-3)
ori_point2 = Point(3,-3)
ori_point3 = Point(1,5)

AllTriangle = []
class Triangle:
    def __init__(self, point1, point2, point3):
        assert point1 != point2 and point1 != point3 and point2 != point3
        self.points = [point1, point2, point3]
        self.neighborTri = [None, None, None] # 顶点的对边为公共边的三角形
        self.sort()
        self.get_heart()
        _ = self.getEdges()
        AllTriangle.append(self)
    def sort(self):
        # 三点呈逆时针排列
        P1,P2,P3 = self.points
        P1P2 = P2.xy - P1.xy
        P1Q2 = rot_90 @ P1P2
        P1P3 = P3.xy - P1.xy
        prod = (P1Q2 * P1P3).sum()
        assert prod != 0
        if prod < 0:
            self.points[1],self.points[2] = self.points[2],self.points[1]
            self.neighborTri[1],self.neighborTri[2] = self.neighborTri[2],self.neighborTri[1]
    def connect(self, other, Pa, Pb):
        # other : Triangle
        # Pa, Pb: 指定的公共点
        id_a = self.points.index(Pa)
        id_b = self.points.index(Pb)
        id_c = 3 - id_a - id_b
        self.neighborTri[id_c] = other
    def get_heart(self):
        # 获取外心和半径
        P1,P2,P3 = self.points
        C3 = (P1.xy+P2.xy) * 0.5
        C2 = (P1.xy+P3.xy) * 0.5
        P1P2 = P2.xy - P1.xy
        P1P2_unit = P1P2 / np.linalg.norm(P1P2)
        P3P1 = P1.xy - P3.xy
        P3P1_unit = P3P1 / np.linalg.norm(P3P1)
        
        
        n3 = rot_90 @ P1P2_unit # (2,)
        n2 = rot_90 @ P3P1_unit

        # C3 + k1*n3 = C2 + k2*n2
        # [n3 -n2] k = C2 - C3
        # k = [n3 -n2]^{-1} @ (C2 - C3)
        A = np.stack([n3, -n2],axis=1)
        # print(A)
        k = np.linalg.inv(A) @ (C2 - C3).T
        k1,k2 = k
        heart = C3 + k1*n3
        _heart = C2 + k2*n2
        if not allDimEpsEq(heart, _heart, 1e-6):
            print(heart, _heart)
            print(n2, C2)
            print(n3, C3)
            print(k)
            print(A)
            raise ValueError("外心计算错误")
        radius = np.linalg.norm(heart - P1.xy)
        self.heart = heart
        self.radius = radius
        
        P2P3 = P3.xy - P2.xy
        P2P3_unit = P2P3 / np.linalg.norm(P2P3)
        n1 = rot_90 @  P2P3_unit
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
    def ExistInYourHeart(self, point):
        # 判断点是否在三角形的外心内
        heart = self.heart
        radius = self.radius
        xy = point.xy
        return np.linalg.norm(xy - heart) < radius
    def getAnotherTwoPoint(self, point):
        idx_p = self.points.index(point)
        idx_p1 = (idx_p+1) % 3
        idx_p2 = (idx_p+2) % 3
        return self.points[idx_p1], self.points[idx_p2]
    def getAnotherPoint(self, point1, point2):
        idx_p1 = self.points.index(point1)
        idx_p2 = self.points.index(point2)
        idx_p3 = 3 - idx_p1 - idx_p2
        return self.points[idx_p3]
    def getTri(self, point):
        idx_p = self.points.index(point)
        return self.neighborTri[idx_p]
    def getEdges(self):
        edges = [
            {self.points[1], self.points[2]},
            {self.points[2], self.points[0]},
            {self.points[0], self.points[1]}
        ]
        self.edges = edges
        return edges
    def calCenter(self):
        P1,P2,P3 = self.points
        self.center = (P1.xy+P2.xy+P3.xy) / 3.0
        return self.center
    def calVertical(self):
        n1,n2,n3 = self.n1,self.n2,self.n3
        P1,P2,P3 = self.points
        P1P2 = P2.xy - P1.xy
        P2P3 = P3.xy - P2.xy
        P3P1 = P1.xy - P3.xy
        
        H1 = -P1P2 @ n1
        H2 = -P2P3 @ n2
        H3 = -P3P1 @ n3
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
    def getHByIdx(self, idx):
        return [self.H1, self.H2, self.H3][idx]
    def calArea(self):
        P1,P2,P3 = self.points
        P1P2 = P2.xy - P1.xy
        H3 = self.H3
        P1P2_mag = np.linalg.norm(P1P2)
        area = 0.5 * P1P2_mag * H3
        self.area = area    
    def getAnotherPointByIdx(self,idx):
        idx1 = (idx+1) % 3
        idx2 = (idx+2) % 3
        return self.points[idx1], self.points[idx2]
    def getNByIdx(self, idx):
        return [self.n1, self.n2, self.n3][idx]

def AllTriWhoLoveMe(point):
    myFriends = []
    for tri in AllTriangle:
        if tri.ExistInYourHeart(point):
            myFriends.append(tri)
    if len(myFriends) >= 3:
        ValueError("I don't need so many friends")
    if len(myFriends) == 0:
        ValueError("I don't have any friends")
    return myFriends


def ConnectEach(tri1,tri2,point1,point2):
    if tri1 is None or tri2 is None:
        return
    tri1.connect(tri2, point1, point2)
    tri2.connect(tri1, point1, point2)




def OneOfUsAdoptThisChild(tri1, tri2, point):
    # 交换公共点
    tri1_in_tri2 = tri2.neighborTri.index(tri1)
    P4 = tri2.points[tri1_in_tri2]
    tri2_in_tri1 = tri1.neighborTri.index(tri2)
    P2 = tri1.points[tri2_in_tri1]
    
    P3, P1 = tri1.getAnotherTwoPoint(P2)
    
    new_tri1 = Triangle(P1, P2, point)
    new_tri2 = Triangle(P2, P3, point)
    new_tri3 = Triangle(P3, P4, point)
    new_tri4 = Triangle(P4, P1, point)
    
    tri34 = tri2.getTri(P1)
    tri23 = tri1.getTri(P1)
    tri24 = tri1.getTri(P3)
    
    AllTriangle.remove(tri1)
    AllTriangle.remove(tri2)
    del tri1, tri2
    
    ConnectEach(new_tri1, new_tri2, P2, point)
    ConnectEach(new_tri2, new_tri3, P3, point)
    ConnectEach(new_tri3, new_tri4, P4, point)
    ConnectEach(new_tri4, new_tri1, P1, point)
    
    ConnectEach(new_tri1, tri24, P1, P2)
    ConnectEach(new_tri2, tri23, P2, P3)
    ConnectEach(new_tri3, tri34, P3, P4)
    ConnectEach(new_tri4, tri24, P4, P1)
    

def Reorganize(point, tri):
    # 破碎，重组
    point1, point2, point3 = tri.points
    tri1 = Triangle(point2, point3, point)
    tri2 = Triangle(point3, point1, point)
    tri3 = Triangle(point1, point2, point)
    ConnectEach(tri1, tri2, point3, point)
    ConnectEach(tri2, tri3, point1, point)
    ConnectEach(tri3, tri1, point2, point)
    AllTriangle.remove(tri)

def DrawAllPoint():
    for point in AllPoint:
        plt.plot(point.xy[0], point.xy[1], '*')


def DrawAllTri():
    for tri in AllTriangle:
        # 给这个三角形生成一个随机颜色
        color = np.random.rand(3,)  # RGB in [0,1]
        # 三点
        xs = [p.xy[0] for p in tri.points]
        ys = [p.xy[1] for p in tri.points]
        # 闭合
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs, ys, color=color)


def ori_in(tri):
    cond1 = ori_point1 in tri.points
    cond2 = ori_point2 in tri.points
    cond3 = ori_point3 in tri.points
    return cond1 or cond2 or cond3


# 删除影响三角形的公共边
def delCommonEdge(tris, point):
    all_edges = []
    for tri_idx in range(len(tris)):
        tri = tris[tri_idx]
        temp_edges = tri.getEdges()
        for edge in temp_edges:
            if edge in all_edges:
                all_edges.remove(edge)
            else:
                all_edges.append(edge)
    for tri_idx in range(len(tris)):
        tri = tris[tri_idx]
        AllTriangle.remove(tri)
        del tri
    for edge in all_edges:
        point1, point2 = edge
        tri = Triangle(point1, point2, point)

# 始祖三角形
def create_mesh(all_point):
    global ori_point1, ori_point2, ori_point3
    ori_triangle = Triangle(ori_point1, ori_point2, ori_point3)
    for point_idx in range(all_point.shape[0]):
        xy = all_point[point_idx]
        point = Point(xy[0], xy[1]) # 创建
        myFriends = AllTriWhoLoveMe(point)
        delCommonEdge(myFriends, point)
        # disp_tri()

        
    tris_to_del = []
    for tri in AllTriangle:
        if ori_in(tri):
            tris_to_del.append(tri)

    for tri in tris_to_del:
        AllTriangle.remove(tri)
        del tri

    AllPoint.remove(ori_point1)
    AllPoint.remove(ori_point2)
    AllPoint.remove(ori_point3)
    del ori_point1, ori_point2, ori_point3

def list_common(lst1, lst2):
    common_elements = []
    for element in lst1:
        if element in lst2:
            common_elements.append(element)
    return common_elements


def connect_all_tri():
    for tri1_idx in range(len(AllTriangle)):
        tri1 = AllTriangle[tri1_idx]
        for tri2_idx in range(tri1_idx+1, len(AllTriangle)):
            tri2 = AllTriangle[tri2_idx]
            cap_edge = list_common(tri1.edges,tri2.edges)
            if len(cap_edge) >= 2:
                raise ValueError("两个三角形有多个公共边")
            if len(cap_edge) == 0:
                continue
            edge = cap_edge.pop()
            edge_idx1 = tri1.edges.index(edge)
            edge_idx2 = tri2.edges.index(edge)
            tri1.neighborTri[edge_idx1] = tri2
            tri2.neighborTri[edge_idx2] = tri1

    # 边界点个数
    boundary_num = 0
    for tri in AllTriangle:
        if None in tri.neighborTri:
            boundary_num += 1
    print("边界点个数：", boundary_num)
    print("总点数：", len(AllPoint))


# dh_default = 0.1
# xy_boundary_default = np.array([[0,0], [1,0], [0.5,1]])

def disp_tri():
    DrawAllPoint()
    DrawAllTri()
    # plt.show()

def get_tri(xy_boundary, dh, need_disp=False):
    assert dh <= 0.5
    assert abs(xy_boundary.max()) <= 1
    _ = np.arange(0,1,dh)
    x,y = np.meshgrid(_,_)
    # for i in range(x.shape[0]):
    #     if i % 2 == 0:
    #         x[i,:] += dh / 2
    # x += np.random.randn(*x.shape) * 0.01
    # y += np.random.randn(*y.shape) * 0.01
    ps = np.stack([x.flatten(),y.flatten()],axis=-1)
    ps = inner_choice(ps, xy_boundary)

    new_boundary = []

    edge = np.roll(xy_boundary,1,axis=0) - xy_boundary
    edge_mag = np.linalg.norm(edge,axis=-1)
    num_subdiv = np.ceil(edge_mag / dh).astype(int)
    for i in range(num_subdiv.shape[0]):
        k = np.arange(num_subdiv[i]) / num_subdiv[i]
        subdiv_edge_temp = xy_boundary[i] + k[:,None] * edge[i]
        new_boundary.append(subdiv_edge_temp)

    new_boundary = np.concatenate(new_boundary,axis=0)

    all_point = np.concatenate([ps,new_boundary],axis=0)

    # all_point = ps

    create_mesh(all_point)
    connect_all_tri()
    # 不能有两个边都在外面
    global AllTriangle
    newTriangle = []
    for tri in AllTriangle:
        if tri not in newTriangle:
            newTriangle.append(tri)
    AllTriangle = newTriangle
    while 1:
        for idx_tri,tri in enumerate(AllTriangle):
            if tri.neighborTri.count(None) == 2:
                boundary_edges = [i for i, nb in enumerate(tri.neighborTri) if nb is None]
                ano_idx = 3 - boundary_edges[0] - boundary_edges[1]
                tri_neighbor = tri.neighborTri[ano_idx]
                # 交换对角线
                P4 = tri.points[ano_idx]
                P1 = tri.points[(ano_idx+1)%3]
                P3 = tri.points[(ano_idx+2)%3]
                P2 = tri_neighbor.getAnotherPoint(P1, P3)
                t1 = tri_neighbor.getTri(P1)
                t3 = tri_neighbor.getTri(P3)
                tri1 = Triangle(P4, P2, P3)
                tri2 = Triangle(P4, P1, P2)
                ConnectEach(tri1,tri2,P2,P4)
                ConnectEach(tri1,t1,P2,P3)
                ConnectEach(tri2,t3,P2,P1)
                AllTriangle.remove(tri)
                AllTriangle.remove(tri_neighbor)
                del tri, tri_neighbor
                # AllTriangle.append(tri1)
                # AllTriangle.append(tri2)
                break
            elif tri.neighborTri.count(None) == 3:
                raise ValueError("wtf")
            else:
                pass
        if idx_tri == len(AllTriangle)-1:
            break
    newTriangle = []
    for tri in AllTriangle:
        if tri not in newTriangle:
            newTriangle.append(tri)
    AllTriangle = newTriangle
    if need_disp:
        disp_tri()
        
    global AllPoint
    newAllPoint = []
    print(len(AllPoint))
    for point in AllPoint:
        if point not in newAllPoint:
            newAllPoint.append(point)
    AllPoint = newAllPoint
    print("三角形创建完成")
    print(len(AllPoint))
    return AllTriangle, AllPoint

# 删除始祖点

'''
plt.plot(new_boundary[:,0], new_boundary[:,1], '*')
plt.plot(ps[:,0], ps[:,1], '*')
'''

'''
boundary = np.array([[0,0], [1,0], [1,1], [0,1]])
AllTriangle = get_tri(boundary, 0.25)
disp_tri()
plt.show()
plt.close()
'''
