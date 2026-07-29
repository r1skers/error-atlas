# Softmax

> Active — directional propagation and finite-precision stability.

## Current question

研究 logits 的方向性误差怎样经过 Softmax normalization，并继续连接
finite precision、cross-entropy 与可调精度。

## Research card

- **Object**：binary Softmax and cross-entropy；下一步推广到多分类。
- **Reference**：实数算术下的 Softmax 或 fused cross-entropy。
- **Metric**：stored logit-difference error、概率绝对误差与输出
  \(2\)-norm。
- **Sources**：输入量化、exp approximation、overflow、underflow、求和、
  除法、舍入顺序与不稳定组合。
- **Propagation**：Jacobian 的奇异方向区分共同平移和 relative contrast。
- **Control**：subtract-max、sign-aware sigmoid、fused loss 与计算 dtype。
- **Optimization**：后续比较 precision、dynamic range 和计算成本。
- **Verification**：精确小例、运行前预测、边界测试、CSV、metadata 与
  closed-book rewrite。

## Established results

二分类 Jacobian 为

\[
J_s(\mathbf z)
=p_1p_2
\begin{pmatrix}
1&-1\\
-1&1
\end{pmatrix}.
\]

共同平移方向 \((1,1)^T/\sqrt2\) 的奇异值为 \(0\)，contrast direction
\((1,-1)^T/\sqrt2\) 的奇异值为 \(2p_1p_2\)。因此

\[
\|J_s(\mathbf z)\|_2
=2p_1p_2
\le\frac12.
\]

Binary Softmax 只依赖 \(d=z_1-z_2\)。对 one-hot cross-entropy，

\[
\nabla_{\mathbf z}L=\mathbf p-\mathbf y,
\qquad
\nabla_{\mathbf z}^2L=J_s(\mathbf z).
\]

subtract-max 利用精确 shift invariance 控制正指数 overflow；fused
cross-entropy 在浮点求值前保留解析抵消。但这些稳定化不能恢复
normalization 之前已经丢失的输入差异。

## Verified experiment

固定

\[
\mathbf z(M)=(M+1,M),
\qquad
p_1^{\mathrm{ref}}=\sigma(1).
\]

先把 logits 存为 FP32，再执行 subtract-max。实验得到

\[
M=2^{23}
\Rightarrow
\widehat d=1,\quad
\widehat p_1\approx0.7310586,
\]

\[
M=2^{24}
\Rightarrow
\widehat d=0,\quad
\widehat p_1=0.5.
\]

误差发生在 input quantization，而不是 stable normalization。量化与
中心化一般不交换：

\[
Q\!\left(\mathbf z-m\mathbf1\right)
\ne
Q(\mathbf z)-\max(Q(\mathbf z))\mathbf1.
\]

实验预测、实测表、边界、源码、测试、CSV、metadata 与 closed-book
evidence 统一保存在 experiments 目录。

## Research workflow

本 topic 不复制 Taylor 的教程式 checkpoint 结构，只复用同一研究纪律：

- framework/error_analysis_protocol.md 规定研究循环与 evidence level；
- framework/implementation_learning_protocol.md 规定主动输出和代码
  ownership；
- 运行前记录 direction、scale、boundary 与 failure signature；
- 运行后区分理论、实现、浮点和输入量化误差；
- 结论必须能回到测试与版本化 artifact。

## Current status

- 二分类方向性、稳定求值和首个 FP32 输入量化实验已收口；
- 原实现与 closed-book rewrite 在三个注册 probe 上完全一致；
- 下一入口：三分类可手算案例；
- 暂不拆分额外理论文档，只有出现新的推导或实验边界时再扩展。
