# CP04 — Bound Tightness

状态：完成（2026-07-28）

## 目标

区分 bound validity 与 bound usefulness，并识别从完整误差结构压缩到 supremum bound 时丢失的信息。

## 基准函数

令

\[
f(x)=\frac1{1-x},\qquad a=0.
\]

其 \(n\) 阶 Taylor polynomial 是有限几何和：

\[
P_n(x)=1+x+\cdots+x^n
=\frac{1-x^{n+1}}{1-x}.
\]

因此可以直接得到 exact remainder：

\[
R_n(x)
=
f(x)-P_n(x)
=
\frac{x^{n+1}}{1-x}.
\]

## 通用 Lagrange bound

因为

\[
f^{(n+1)}(t)
=
\frac{(n+1)!}{(1-t)^{n+2}},
\]

对 \(0\le t\le x<1\)，有

\[
|f^{(n+1)}(t)|
\le
\frac{(n+1)!}{(1-x)^{n+2}}.
\]

代入 Lagrange bound：

\[
|R_n(x)|
\le
\frac{x^{n+1}}{(1-x)^{n+2}}.
\]

它与 exact remainder 的比值为

\[
\frac{\text{Lagrange bound}}{|R_n(x)|}
=
\frac1{(1-x)^{n+1}}.
\]

## 关键案例：\(x=1/2\)

actual error 为

\[
R_n(1/2)=2^{-n},
\]

随 \(n\) 指数衰减。但上述 Lagrange bound 恒为

\[
2,
\]

完全无法认证这种收敛。

这个 bound 始终正确，但随着 \(n\) 增加而越来越无用。原因是 supremum bound 把靠近 \(t=x\) 的最大导数与整个 kernel mass 相乘，忽略了 integral kernel \((x-t)^n\) 恰好在 \(t=x\) 处为零。

## 隐藏中间点与最右端替换

Lagrange 形式中的真实中间点满足

\[
R_n(x)=\frac{f^{(n+1)}(\xi_n)}{(n+1)!}x^{n+1}.
\]

对本例与 exact remainder 比较可得

\[
\xi_n=1-(1-x)^{1/(n+2)}.
\]

固定 \(0<x<1\) 时，\(n\to\infty\) 有 \(\xi_n\to0\)，而不是趋向区间最右端 \(x\)。用右端点替换 \(\xi_n\) 虽然保持了有效性，却把导数放大因子

\[
(1-\xi_n)^{-(n+2)}
\]

粗暴替换为更大的

\[
(1-x)^{-(n+2)}.
\]

该放大还会随阶数累积，因此相对松弛度发散。

## 结构性解释

- 直接原因：对导数做区间 supremum，把局部最坏值当成整个积分区间都同时达到最坏情况；
- 几何原因：导数最大的位置恰好是 integral kernel 为零的位置，二者不能独立最大化；
- 深层原因：\(x=1\) 的奇点使高阶导数在右端附近快速增长，放大了信息压缩造成的损失；
- 更紧方法：保留几何级数结构可直接得到 exact remainder，无需 Laurent 展开。

## 检查点结论

一个误差界至少需要分别评价：

1. **有效性**：是否始终覆盖 actual error；
2. **松弛度**：\(S=B/E\) 是否可接受；
3. **渐近质量**：当阶数或尺度变化时，界能否显示真实收敛；
4. **信息来源**：界是在何处用 supremum、三角不等式或独立最坏情况压缩了结构。

因此，“界正确”只是最低要求，并不意味着它足以指导误差控制或算法选择。
