# CP01 — Lagrange Remainder

状态：完成（2026-07-16）

## 目标

把 remainder 的定义、存在性表示、显式上界和主导项连接起来。

## 充分假设版本

设 \(f\in C^{n+1}(I)\)，其中区间 \(I\) 包含展开点 \(a\) 和评价点 \(x\)。Taylor theorem 保证存在某个位于 \(a\) 与 \(x\) 之间的 \(\xi\)，使得

\[
R_n(x)
=
\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}.
\]

这是一个精确的存在性表示，但 \(\xi\) 通常未知，因此它本身未必能直接算出 actual error。

若在连接 \(a\) 与 \(x\) 的区间上

\[
|f^{(n+1)}(t)|\le M,
\]

则立即得到可计算上界

\[
|R_n(x)|
\le
\frac{M}{(n+1)!}|x-a|^{n+1}.
\]

## 基准例：small-angle approximation

对 \(f(\theta)=\sin\theta\)、\(a=0\)、\(n=2\)，有

\[
P_2(\theta)=\theta.
\]

Lagrange remainder 给出某个介于 \(0\) 与 \(\theta\) 之间的 \(\xi\)，使得

\[
R_2(\theta)
=
\sin\theta-\theta
=
-\frac{\cos\xi}{6}\theta^3.
\]

因为 \(|\cos\xi|\le1\)，所以

\[
|R_2(\theta)|\le\frac{|\theta|^3}{6}.
\]

当 \(\theta\to0\) 时，夹在 \(0\) 与 \(\theta\) 之间的 \(\xi\to0\)，于是

\[
\frac{R_2(\theta)}{\theta^3}
=
-\frac{\cos\xi}{6}
\to
-\frac16.
\]

因此

\[
R_2(\theta)
=
-\frac{\theta^3}{6}+o(\theta^3),
\]

其中 \(-\theta^3/6\) 是主导项。

## \(\xi\) 的来源

固定 \(x\ne a\)，令

\[
K=\frac{R_n(x)}{(x-a)^{n+1}}
\]

并构造

\[
F(t)=f(t)-P_n(t)-K(t-a)^{n+1}.
\]

这个选择使 \(F(x)=0\)，而 Taylor polynomial 的导数匹配使

\[
F(a)=F'(a)=\cdots=F^{(n)}(a)=0.
\]

反复使用 Rolle theorem，存在介于 \(a\) 和 \(x\) 之间的 \(\xi\)，满足

\[
F^{(n+1)}(\xi)=0.
\]

因此

\[
f^{(n+1)}(\xi)-K(n+1)!=0,
\]

代回 \(K\) 即得到 Lagrange remainder。

## Checkpoint conclusion

- \(\xi\) 是 Rolle theorem 保证存在的点，并不是直接计算出的参数；
- 对每个 \(x\)，对应的 \(\xi_x\) 可能不同，但总位于 \(a\) 与 \(x\) 之间；
- 若 \(|f^{(n+1)}|\le M\)，则 \(|R_n(x)|\le M|x-a|^{n+1}/(n+1)!\)；
- 若 \(f^{(n+1)}\) 在 \(a\) 连续，则

\[
R_n(x)=\frac{f^{(n+1)}(a)}{(n+1)!}(x-a)^{n+1}
+o((x-a)^{n+1}).
\]

- 对 \(f(x)=\cos x\)、\(a=0\)、\(n=1\)，有

\[
R_1(x)=-\frac{\cos\xi_x}{2}x^2
=-\frac{x^2}{2}+o(x^2).
\]

## 已完成检查题

对 \(f(x)=e^x\)、\(a=0\)、\(n=2\)：

1. 写出 \(P_2(x)\)；
2. 写出 Lagrange remainder；
3. 当 \(0\le x\le0.1\) 时，为 \(|R_2(x)|\) 找到一个不含未知 \(\xi\) 的显式上界。
