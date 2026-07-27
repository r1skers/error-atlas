# CP03 — Peano Remainder

状态：完成（2026-07-17）

## 目标

理解在不要求 \((n+1)\) 阶导数控制的情况下，Taylor polynomial 仍能提供什么局部渐近保证。

## 一阶情形

函数在 \(a\) 可微的定义是

\[
\lim_{h\to0}
\frac{f(a+h)-f(a)}{h}
=f'(a).
\]

等价地，

\[
f(a+h)=f(a)+f'(a)h+o(h).
\]

因此，一阶 Peano remainder 不是额外技巧，而是 differentiability 本身的另一种写法。

## 一般形式

在标准的 \(n\) 阶可微假设下，

\[
f(a+h)
=
\sum_{k=0}^{n}
\frac{f^{(k)}(a)}{k!}h^k
+o(h^n).
\]

Peano form 提供局部 asymptotic statement，但通常不提供实际余项的符号、未知中间点表示或可计算的显式上界。

## 区分性案例

令

\[
f(x)=|x|^{3/2},\qquad a=0.
\]

有 \(f(0)=0\) 且 \(f'(0)=0\)，所以

\[
P_1(x)=0,
\qquad
R_1(x)=|x|^{3/2}.
\]

由于

\[
\frac{|x|^{3/2}}{|x|}=|x|^{1/2}\to0,
\]

所以 \(R_1=o(|x|)\)。但

\[
\frac{|x|^{3/2}}{x^2}=\frac1{|x|^{1/2}}\to\infty,
\]

因此 \(R_1\ne O(x^2)\)。这个例子展示了 Peano conclusion 可以成立，而二阶 Lagrange-style bound 并不存在。

## Induction proof

令

\[
r_n(h)
=
f(a+h)-
\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}h^k.
\]

当 \(n=1\) 时，\(r_1(h)=o(h)\) 正是 \(f\) 在 \(a\) 可微的定义。

假设 \(n-1\) 阶结论已经成立。对 \(r_n\) 求导：

\[
r_n'(h)
=
f'(a+h)-
\sum_{k=1}^{n}
\frac{f^{(k)}(a)}{(k-1)!}h^{k-1}.
\]

右侧是 \(f'\) 的 \(n-1\) 阶 Taylor remainder，因此由 induction hypothesis，

\[
r_n'(h)=o(h^{n-1}).
\]

又因为 \(r_n(0)=0\)，中值定理给出一个位于 \(0\) 与 \(h\) 之间的 \(c_h\)，使得

\[
r_n(h)=r_n'(c_h)h.
\]

由 \(|c_h|\le|h|\) 以及 \(c_h\to0\)，得到

\[
r_n(h)=h\,o(h^{n-1})=o(h^n).
\]

## Checkpoint conclusion

- 一阶 Peano form 就是 differentiability 的等价写法；
- 一般形式通过对 \(f'\) 使用低一阶结论和中值定理归纳得到；
- Peano form 使用较弱的局部可微性信息，结论是 \(R_n=o(h^n)\)；
- 它通常不给余项符号、显式常数或 \(O(h^{n+1})\) 保证；
- \(O(h^{n+1})\) 能推出 \(o(h^n)\)，反向一般不成立。
