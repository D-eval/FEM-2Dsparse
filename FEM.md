# Dirichlet

$$
\begin{aligned}
&\nabla (\kappa \nabla u) = f & u \in \Omega \\
& u = g & u \in \partial \Omega
\end{aligned}
$$

只须令 $u' = u - g $，则 $u'$ 在边界取 $0$，有

$$
\nabla (\kappa \nabla u') = f - \nabla (\kappa \nabla g)
$$

令 $f' = f - \nabla (\kappa \nabla g)$

我们只要关心 $\nabla (\kappa \nabla u) = f$ 的情形。

$$
\forall j,
\sum_{i} u_i \int \kappa (\nabla \phi_i) \cdot (\nabla \phi_j)
= - \int f \phi_j
$$

# 第一类Green 公式
小定理1

$$
\nabla \cdot (p\vec{q}) = (\nabla p) \cdot \vec{q} + p (\nabla\cdot \vec{q})
$$

小定理2

$$
\int_{\Omega} \vec \nabla \cdot \vec{F} = \int_{\partial \Omega} \vec{F} \cdot \vec{n}
$$

由此这两个小定理可得第一类Green公式（首先小定理1两边积分，然后左边rw小定理2即可）

$$
\int_{\Omega} p \nabla \cdot \vec{q} = 
- \int_{\Omega} (\nabla p) \cdot \vec{q} +
\int_{\partial \Omega} p \vec{n} \cdot \vec{q}
$$

# Robin 边界条件 弱形式

```
先推弱形式，再组装系数矩阵
```

## 条件

求解 $u$ 使得

$$
\forall x\in \Omega,\nabla\cdot (\kappa \nabla u) = f
$$

边界条件
$$
\forall x \in \partial \Omega, (\kappa \vec{n}\cdot \vec \nabla) u + b u = g
$$

或者，规定一系列基函数 $\varphi$，然后求解各个系数 $c$

$$
u = \sum_i c_i \varphi_i
$$

## 解

### 1. 弱形式

$$
\int_{\Omega} v \nabla \cdot (\kappa \nabla u) = \int_{\Omega} v f
$$

对于左边用第一类格林公式，得到

$$
-\int_{\Omega} (\nabla v) \cdot (\kappa \nabla u)
+ \int_{\partial \Omega} v (\kappa \vec{n} \cdot \nabla u)
= \int_{\Omega} v f
$$

根据边界条件，替换上式左边第二项
$$
-\int_{\Omega} (\nabla v) \cdot (\kappa \nabla u)
+ \int_{\partial \Omega} v (g - bu)
= \int_{\Omega} v f
$$

### 2. 代入基函数

把 $u = \sum_i c_i \varphi_i$ 和 $v = \varphi_j$ 代入上式，把$\kappa$提前，得到

$$
\sum_i 
(\int_{\Omega} \kappa   (\nabla \varphi_j) \cdot (\nabla \varphi_i)
+ \int_{\partial \Omega} b \varphi_j \varphi_i
)
c_i
=  (\int_{\partial \Omega} \varphi_j g
- \int_{\Omega} \varphi_j f)
$$

接下来的目标就是计算下面这几项，算好之后就可以获得$c_i$的系数
1. $\int_{\Omega} \kappa   (\nabla \varphi_j) \cdot (\nabla \varphi_i)$
2. $\int_{\partial \Omega} b \varphi_j \varphi_i$
3. $\int_{\partial \Omega} \varphi_j g$
4. $\int_{\Omega} \varphi_j f$

考虑$\Omega$为二维的情况，我们把网格划分成三角形，选取一次基函数，也就是，对于三个顶点，一个基函数只在某个顶点处取$1$，在其他顶点取$0$。

现在每个面有个编号，每个面有三个顶点$P_1,P_2,P_3$，也有三个基函数$f_1,f_2,f_3$，每个基函数只在相应的顶点上取$1$，在其它顶点上取$0$。这显然是一个平面，我们过三个顶点做三个垂线，垂足为$H_1,H_2,H_3$，并得到$3$个又垂足指向顶点的单位法向量$\vec{n_1},\vec{n_2},\vec{n_3}$，现在对于三角形内部的一个点$P$，我们可以轻松的写出

$$
f_1(P) = \frac{\overrightarrow{H_1P}\cdot \vec{n_1}}{|H_1P_1|}
$$

并且显然，$\nabla f_1$方向一定是$\vec{n_1}$（你找到一个观察角度，让$P_2P_3$重合就可以发现，而在这个观察角度中，平面就退化为直线，于是，$\nabla f_1$的大小就是$\frac{1}{|H_1P_1|}$），从而

$$
\nabla f_1 =  \frac{\vec{n_1}}{|H_1P_1|}
$$

于是我们就能推导出目标1的表达式，当$\varphi_i,\varphi_j$不在同一个三角形上时，取值为$0$，当他们在一个三角单元上时，比如说为$f_1, f_2$，于是获得目标1的表达式

Target 1:

$$
\kappa (\xi) S_{P_1P_2P_3} \frac{\vec{n_1}\cdot \vec{n_2}}{|H_1P_1||H_2P_2|}
$$

其中$\xi$通过积分中值定理分离出来，可以近似为三角形的中心。为了减小这一近似的误差，我们应当把网格尽量细分。并且在创建三角单元类时，我们就应当计算并储存$H,n,S_{P_1P_2P_3},|HP|$

接下来我们来看目标2，对于边界单元，我们应当事先储存作为边界的顶点索引$i,j$，这个积分也只是在$P_iP_j$线段上的积分，积分中$b$同样通过积分中值定理分离出来，然而这次应当选取的$\xi$应当是$P_iP_j$的中点，于是目标2可以化为

$$
b(\xi) \int_{P_1P_2} f_1 f_2 \mathrm{d} p
$$

进一步，在这条直线上，$f_1$和$f_2$的表达式都很简单

$f_1 = \frac{|PP_2|}{|P_1P_2|}$

$f_2 = \frac{|PP_1|}{|P_1P_2|}$

于是右边的乘数化为

$$
\int_0^{|P_1P_2|} \frac{p}{|P_1P_2|} (1 - \frac{p}{|P_1P_2|}) \mathrm{d} p = \frac{|P_1P_2|}{2} - \frac{|P_1P_2|^2}{3}
$$

于是

Target 2:

$$
b(\xi) (\frac{|P_1P_2|}{2} - \frac{|P_1P_2|^2}{3})
$$

至于目标3

$$
\int_{\partial \Omega} \varphi_j g = 
g(\xi) \int_0^{|P_1P_2|} \frac{p}{|P_1P_2|}\mathrm{d} p=g(\xi) \frac{|P_1P_2|}{2}
$$

Target 3:
$$
g(\xi) \frac{|P_1P_2|}{2}
$$

至于目标4

$$
\int_{\Omega} \varphi_j f = 
f(\xi) \int_{P_1P_2P_3} \varphi_j
$$

积分可以看作一个三棱锥的体积，它显然等于 $\frac{1}{3} S_{P_1P_2P_3}$

Target 4:
$$
f(\xi) \frac{1}{3} S_{P_1P_2P_3}
$$

综上，我们能计算出各个系数。

### 3. 网格划分

二维问题边界的保存格式可以是形状为$(n,2)$的Array，（逆时针顺序）。

我们把网格划分成三角形

### 4. 基函数选取

划分好三角形后，