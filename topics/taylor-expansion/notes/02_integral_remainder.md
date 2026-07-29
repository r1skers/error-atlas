# 02 — Integral Remainder

状态：完成（2026-07-16）

## 目标

用一个不含未知中间点的精确积分表示 remainder，并理解它与 Lagrange remainder、显式上界之间的关系。

## 定理

在足以使用微积分基本定理和分部积分的光滑性条件下，

\[
R_n(x)
=
\frac{1}{n!}
\int_a^x f^{(n+1)}(t)(x-t)^n\,dt.
\]

这个公式保留了 \(f^{(n+1)}\) 在整个区间上的信息，而不是把它压缩成某个未知点 \(\xi\) 处的取值。

## 与 Lagrange bound 的联系

若在 \(a\) 与 \(x\) 之间有

\[
|f^{(n+1)}(t)|\le M,
\]

则

\[
|R_n(x)|
\le
\frac{M}{n!}
\int_a^x |x-t|^n\,|dt|
=
\frac{M}{(n+1)!}|x-a|^{n+1}.
\]

## 当前基准例

对 \(f(x)=e^x\)、\(a=0\)、\(n=2\)，

\[
R_2(x)
=
\frac12\int_0^x e^t(x-t)^2\,dt.
\]

下一步将从 \(f(x)-f(a)=\int_a^x f'(t)\,dt\) 出发，通过分部积分推导一般公式。

## 阶段结论

- integral remainder 把 \(f^{(n+1)}\) 在区间上的局部贡献通过 kernel \((x-t)^n/n!\) 累积起来；
- 被积函数的符号可以判断 Taylor polynomial 是高估还是低估；
- 对一阶近似，\(f''\ge0\) 推出切线低估 convex function，\(f''\le0\) 则推出切线高估；
- 对非负 kernel 使用积分中值定理，会把完整积分压缩成 Lagrange remainder；
- 从 integral representation 到 Lagrange representation，再到 supremum bound，信息逐步减少而可操作性增强。
