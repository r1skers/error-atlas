# Next Session

记录日期：2026-07-28

## Resume point

- Completed topic：Taylor expansion — first pass
- Completed checkpoint：CP06 — Error Control and Optimization
- Summary：topics/taylor-expansion/CP06_control_and_optimization.md
- Registry：TOPICS.md

## 已建立的核心结论

- Taylor remainder 的表示、渐近阶、上界和 bound tightness 已完成第一轮；
- 数值微分中的截断误差、舍入误差与稳定表示已经连接；
- Richardson extrapolation 利用跨尺度误差的主导结构消除首项；
- 中心差分利用对称性把截断误差提高到二阶；
- 相关噪声经过差分和 \(N\) 次平均后的方差为

\[
\frac{\sigma^2(1-\rho)}{2Nh^2};
\]

- MSE 的偏差—方差分解解释了 U 形误差曲线和最优步长；
- Monte Carlo 实验验证了 \(h^{-1}\) 与 \(h^2\) 两个主导区；
- 浮点求值顺序可以在理论精确抵消后留下机器误差；
- closed-book rewrite 成功恢复相关 Gaussian 噪声构造，并补上非法调用不得推进 RNG 的实现约束。

## 下一阶段

从标量局部传播

\[
\Delta y\approx f'(x)\Delta x
\]

推广到向量映射

\[
\Delta\mathbf y\approx J_f(\mathbf x)\Delta\mathbf x.
\]

第一组问题：

1. 为什么相同范数的输入误差会因方向不同而被放大或压缩？
2. Jacobian 的 operator norm 与 singular values 分别提供什么信息？
3. Softmax 为什么会完全消掉共同平移方向？
4. 这种方向性如何连接到 finite precision、subtract-max 与后续 loss？

在正式建立 Softmax topic 目录前，先完成一个二维线性映射的最小案例和 explain-back。

## Taylor 验证命令

    python -m unittest discover -s topics/taylor-expansion/experiments -p "test_*.py" -v
    python topics/taylor-expansion/experiments/finite_difference.py
    python topics/taylor-expansion/experiments/statistical_noise.py
