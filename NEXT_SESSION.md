# Next Session

记录日期：2026-07-29

## Resume point

- Active topic：Softmax
- Completed checkpoint：binary directionality and FP32 shift-resolution probe
- Topic summary：topics/softmax/README.md
- Experiment：topics/softmax/experiments/fp32_shift_resolution.py
- Registry：TOPICS.md

## 已建立的核心结论

- 对二维线性映射 \(A=\operatorname{diag}(3,0.5)\)，相同长度的误差会因
  方向不同而具有 \(0.5\) 到 \(3\) 的放大倍数；
- operator \(2\)-norm 只保留最大奇异值，完整 SVD 还需要 singular
  vectors 才能同时给出倍率与方向；
- 非线性映射的 Jacobian 是位置相关的局部线性传播器，并带有 Taylor
  remainder；
- 二分类 Softmax Jacobian 为

\[
J_s(\mathbf z)
=p_1p_2
\begin{pmatrix}
1&-1\\
-1&1
\end{pmatrix},
\]

  其共同平移方向奇异值为 \(0\)，contrast direction 奇异值为
  \(2p_1p_2\le1/2\)；
- subtract-max 利用精确 shift invariance 避免正指数 overflow，但不能恢复
  已在输入量化中丢失的相对差异；
- 分开计算 softmax 后再取 log 可能破坏解析抵消，融合 cross-entropy 保留
  \(\nabla_{\mathbf z}L=\mathbf p-\mathbf y\)；
- FP32 实验验证：\(M=2^{23}\) 时单位差仍保留，\(M=2^{24}\) 时 stored
  difference 变为零，第一概率从约 \(0.731\) 跳到 \(0.5\)；
- 量化与中心化一般不交换：

\[
Q\!\left(\mathbf z-m\mathbf1\right)
\ne
Q(\mathbf z)-\max(Q(\mathbf z))\mathbf1;
\]

- closed-book rewrite 独立恢复 quantize-before-stabilize 机制，并与原实现
  在三个注册 probe 上完全一致。

## 下一阶段

从二分类推广到 \(K\) 分类 Softmax：

\[
J_s(\mathbf z)=\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^T.
\]

第一组问题：

1. 为什么 \(J_s\mathbf1=0\) 在任意类别数下仍成立？
2. 除共同平移方向外，其余方向的局部放大由什么决定？
3. operator norm 的全局上界能否达到，在哪些概率分布上达到？
4. 二分类的 contrast direction 如何推广到概率单纯形的切空间？

先完成一个三分类、可手算的最小案例和 explain-back，再决定下一组实验。

## 验证命令

    python -m unittest discover -s topics/softmax/experiments -p "test_*.py" -v
    python topics/softmax/experiments/fp32_shift_resolution.py

Taylor 第一轮验证命令仍保留在 topics/taylor-expansion/experiments/README.md。
