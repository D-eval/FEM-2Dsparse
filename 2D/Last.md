
$$
\begin{aligned}
&
\sigma^{T} \nabla = f
\\&
u = 0 & \forall x \in \partial \Omega
\end{aligned}
$$

where

$$
\begin{aligned}
&
\epsilon = 
\begin{bmatrix}
\partial_1 u_1 & \frac{1}{2}(\partial_2 u_1 + \partial_1 u_2) \\
\frac{1}{2}(\partial_2 u_1 + \partial_1 u_2) & \partial_2 u_2
\end{bmatrix}
\\&
\sigma = (\lambda \nabla \cdot u) I + 2\mu \epsilon
\end{aligned}
$$

Then

$$
\sigma =
\lambda
\begin{bmatrix}
\partial_1 u_1 + \partial_2 u_2 &\\
& \partial_1 u_1 + \partial_2 u_2
\end{bmatrix}
+ \mu
\begin{bmatrix}
2\partial_1 u_1 & \partial_2 u_1 + \partial_1 u_2 \\
\partial_2 u_1 + \partial_1 u_2 & 2 \partial_2 u_2
\end{bmatrix}
$$

$$
\sigma^T \nabla = 
\lambda
\begin{bmatrix}
\partial_{11} u_1 + \partial_{12} u_2 \\
\partial_{22} u_2 + \partial_{12} u_1
\end{bmatrix}
+ \mu
\begin{bmatrix}
2\partial_{11} u_1 + \partial_{22} u_1 + \partial_{12} u_2 \\
2\partial_{22} u_2 + \partial_{11} u_2 + \partial_{12} u_1
\end{bmatrix}
=
\begin{bmatrix}
f_1 \\ f_2
\end{bmatrix}
$$

We consider this first equation

$$
\lambda (
    \partial_{11} u_1 +
    \partial_{12} u_2
) +
\mu (
    2 \partial_{11} u_1 +
    \partial_{22} u_1 +
    \partial_{12} u_2
) = f_1
$$

Weaky form

$$
\begin{aligned}
&
(\lambda + 2\mu)(-<\partial_1 u_1, v>_{\partial_1} + <\partial_1 u_1, \partial_1 v>) +
\\&
(\lambda + \mu)(-<\partial_2 u_2, v>_{\partial_1} + <\partial_2 u_2, \partial_1 v>) +
\\&
\mu (-<\partial_2 u_1, v>_{\partial_2} + <\partial_2 u_1, \partial_2 v>)
\\&
= -<f_1, v>
\end{aligned}
$$

Let

$v=\Phi_i$

$u_1 = \sum c_{1j} \Phi_j$

$u_2 = \sum c_{2j} \Phi_j$

Then

$$
\begin{aligned}
&
(\lambda + 2\mu) (
    \sum c_{1j} (<\partial_1 \Phi_j, \partial_1 \Phi_i> -
    <\partial_1 \Phi_j, \Phi_i>_{\partial_1}
    )
) +
\\&
(\lambda + \mu) (
    \sum c_{2j} (
        <\partial_2 \Phi_j, \partial_1 \Phi_i> -
        <\partial_2 \Phi_j, \Phi_i>_{\partial_1}
    )
) +
\\&
\mu (
    \sum c_{1j} (
        <\partial_2 \Phi_j, \partial_2 \Phi_i> -
        <\partial_2 \Phi_j, \Phi_i>_{\partial_2}
    )
)
\\&
= - <f_1, \Phi_i>
\end{aligned}
$$

and his brother


$$
\begin{aligned}
&
(\lambda + 2\mu) (
    \sum c_{2j} (<\partial_2 \Phi_j, \partial_2 \Phi_i> -
    <\partial_2 \Phi_j, \Phi_i>_{\partial_2}
    )
) +
\\&
(\lambda + \mu) (
    \sum c_{1j} (
        <\partial_1 \Phi_j, \partial_2\Phi_i> -
        <\partial_1 \Phi_j, \Phi_i>_{\partial_2}
    )
) +
\\&
\mu (
    \sum c_{2j} (
        <\partial_1 \Phi_j, \partial_1 \Phi_i> -
        <\partial_1 \Phi_j, \Phi_i>_{\partial_1}
    )
)
\\&
= - <f_2, \Phi_i>
\end{aligned}
$$

In case of triangle unit, we just have to use
`for j in Neighbor(i)` to write coef into the parse equation. So in the $i$-th 2 equation, the coef of $c_{1j}$ and $c_{2j}$ is

$$
\begin{aligned}
&
\text{malloc Equation:(n,2,n,2) and (n,2) by key 'const'}
\\&
\text{for i in range(n):}
\\&
\mathrm{Equation[i][0]['const']} = 
- <f_1, \Phi_i>
\\&
\mathrm{Equation[i][1]['const']} = 
- <f_2, \Phi_i>
\\&\text{for j in Neighbor(i):}
\\&
\mathrm{Equation[i][0][j][0]} =
(\lambda+2\mu) (<\partial_1 \Phi_j, \partial_1 \Phi_i> -
<\partial_1 \Phi_j, \Phi_i>_{\partial_1}
)
+ \mu (<\partial_2 \Phi_j, \partial_2 \Phi_i> -
<\partial_2 \Phi_j, \Phi_i>_{\partial_2}
)
\\&
\mathrm{Equation[i][0][j][1]} =
\lambda (<\partial_2 \Phi_j, \partial_1 \Phi_i>
- <\partial_2 \Phi_j, \Phi_i>_{\partial_1}
) +
\mu (<\partial_1 \Phi_j, \partial_2 \Phi_i> -
<\partial_1 \Phi_j, \Phi_i>_{\partial_2}
)
\\&
\mathrm{Equation[i][1][j][1]} =
(\lambda+2\mu) (<\partial_2 \Phi_j, \partial_2 \Phi_i> -
<\partial_2 \Phi_j, \Phi_i>_{\partial_2}
)
+ \mu (<\partial_1 \Phi_j, \partial_1 \Phi_i> -
<\partial_1 \Phi_j, \Phi_i>_{\partial_1}
)
\\&
\mathrm{Equation[i][1][j][0]} =
\lambda (<\partial_1 \Phi_j, \partial_2 \Phi_i> -
<\partial_1 \Phi_j, \Phi_i>_{\partial_2}
) +
\mu (<\partial_2 \Phi_j, \partial_1 \Phi_i> -
<\partial_2 \Phi_j, \Phi_i>_{\partial_1}
)
\end{aligned}
$$

Then malloc a init solve `X: (n,2)`, then we can get L2

$$
\begin{aligned}
&
\text{malloc X:(n,2), L:(n,2)}
\\&
\text{for i in range(n):}
\\&
\text{tempL} = \text{sum Equation[i][0][:][:] * X}
\\&
\text{tempL} = \text{tempL} - \text{Equation[i][0]['const']}
\\&
\text{L[i][0]} = \text{tempL} ^ 2

\\&
\text{tempL} = \text{sum Equation[i][1][:][:] * X}
\\&
\text{tempL} = \text{tempL} - \text{Equation[i][1]['const']}
\\&
\text{L[i][1]} = \text{tempL} ^ 2
\end{aligned}
$$

Then

$$
\text{L2} = \text{sum L[:][:]}
$$

To minimize `L2` to zero (by auto grad), we can get `X`.

