# 06 — Error Control and Optimization

状态：完成（2026-07-28）

## 实践问题

用带噪中心差分估计

\[
A=f'(0)=e^0=1.
\]

一次带噪观测为

\[
D_i(h)
=
\frac{[e^h+\varepsilon_{i,+}]-[e^{-h}+\varepsilon_{i,-}]}{2h},
\]

实际估计量是 \(N\) 次观测的平均：

\[
\bar D_{h,N}=\frac1N\sum_{i=1}^N D_i(h).
\]

外层再把整个估计器重复 \(M\) 次，用于测量它的经验偏差、方差和 RMSE。\(M\) 不参与单个估计量的定义，也不改变理论最优步长。

## 截断偏差

无噪声中心差分为

\[
\frac{e^h-e^{-h}}{2h}
=\frac{\sinh h}{h}.
\]

因此 exact bias 是

\[
b(h)=\frac{\sinh h}{h}-1,
\]

小步长下

\[
b(h)=\frac{h^2}{6}+O(h^4).
\]

## 相关噪声与平均

用两个独立标准正态变量 \(Z_1,Z_2\) 构造

\[
\varepsilon_+=\sigma Z_1,
\qquad
\varepsilon_-=
\sigma\left(\rho Z_1+\sqrt{1-\rho^2}Z_2\right).
\]

这样两侧噪声的标准差均为 \(\sigma\)，相关系数为 \(\rho\)。经过中心差分并平均 \(N\) 次后，

\[
V(h,N,\rho)
=
\operatorname{Var}(\bar D_{h,N})
=
\frac{\sigma^2(1-\rho)}{2Nh^2}.
\]

\(N\) 决定单次估计器的随机精度，标准差按 \(N^{-1/2}\) 下降；\(M\) 只决定 Monte Carlo 性能测量的精度。

## MSE 分解

将

\[
\bar D_{h,N}=A+b(h)+\xi
\]

写成确定性偏差与零均值随机项，得到

\[
\operatorname{MSE}
=
\mathbb E[(\bar D_{h,N}-A)^2]
=
b(h)^2+V(h,N,\rho).
\]

于是理论 RMSE 为

\[
\boxed{
\operatorname{RMSE}(h,N,\rho)
=
\sqrt{
\left(\frac{\sinh h}{h}-1\right)^2
+
\frac{\sigma^2(1-\rho)}{2Nh^2}
}.
}
\]

经验指标使用外层 \(M\) 个结果计算，并以分母 \(M\) 定义经验方差，使有限数据也严格满足

\[
\operatorname{RMSE}_{\mathrm{emp}}^2
=
\operatorname{bias}_{\mathrm{emp}}^2
+
\operatorname{variance}_{\mathrm{emp}}
\]

至浮点舍入误差。

## U 形与最优步长

以 \(C=1/6\) 近似 \(b(h)\approx Ch^2\)，令

\[
K=\frac{\sigma^2(1-\rho)}{2N},
\]

则

\[
\operatorname{MSE}(h)\approx C^2h^4+\frac{K}{h^2}.
\]

求导得到

\[
\boxed{
h_*=
\left(
\frac{\sigma^2(1-\rho)}
{4NC^2}
\right)^{1/6}.
}
\]

最优点来自两项斜率抵消，不要求两项数值相等。最优点满足

\[
\frac{K}{h_*^2}=2C^2h_*^4,
\]

即随机方差是偏差平方的两倍。

本实验参数

\[
\sigma=10^{-3},\qquad N=100,\qquad \rho=0
\]

给出

\[
h_*\approx0.06694.
\]

## 运行前预测

- \(h\ll h_*\)：随机标准差主导，RMSE 按 \(h^{-1}\) 变化；
- \(h\gg h_*\)：截断偏差主导，RMSE 按 \(h^2\) 变化；
- 最小值出现在 \(h\approx0.067\)；
- \(N\) 增加四倍时，随机标准差减半；
- \(h\) 减半时，截断偏差约降至四分之一，随机标准差约增至两倍。

## 实验结果

对 \(h\in[10^{-3},1]\) 的 41 个对数间隔点，每个配置使用 \(N=100\)、\(M=2000\)：

- 左侧理论 log-log 斜率：\(-1.0000\)；
- 右侧理论 log-log 斜率：\(2.0541\)；
- 理论和经验网格最优点均为 \(h=0.06310\)；
- 经验网格最小 RMSE：\(1.277\times10^{-3}\)；
- 理论网格最小 RMSE：\(1.302\times10^{-3}\)；
- 经验与理论 RMSE 的中位相对差：约 \(0.60\%\)；
- 最大相对差：约 \(3.18\%\)。

结果支持截断偏差、相关噪声传播、\(N^{-1/2}\) 平均律和 MSE 最优步长模型。

## 额外发现：理论抵消后的浮点残余

当 \(\rho=1\) 时，\(\varepsilon_+=\varepsilon_-\)，随机观测噪声在实数算术中应精确抵消。但程序实际计算

\[
\operatorname{fl}(e^h+\varepsilon)
\quad\text{和}\quad
\operatorname{fl}(e^{-h}+\varepsilon)
\]

时，两次加法在不同的浮点网格位置舍入，留下不同的舍入痕迹。相减后公共噪声消失，但舍入误差之差仍存在，实验观察到约 \(10^{-16}\) 量级的残余波动。

因此本实践不仅验证了统计误差模型，也再次验证：

\[
\boxed{\text{代数等价或理论精确抵消，不保证浮点实现逐位等价。}}
\]

## 产物

- experiments/statistical_noise.py：用户实现研究核心，agent 补实验编排；
- experiments/rewrite_correlated_noise.py：独立 closed-book rewrite；
- experiments/results/statistical_noise_comparison.csv：逐步长原始指标；
- experiments/results/statistical_noise_metadata.json：参数与运行环境；
- experiments/results/statistical_noise_error.png：理论与经验误差曲线；
- tests/test_statistical_noise.py：核心 invariant 与回归测试。

## 当前限制

- 噪声假设为加性、Gaussian、同方差；
- 不同样本之间假设独立；
- 当前只扫描固定的 \(\sigma,N,\rho\)；
- 理论统计模型不单独描述浮点舍入，实际实现会叠加机器误差；
- 经验偏差远小于随机标准差时，需要更大的 \(M\) 才能精确测量。

## Closed-book rewrite

在不查看原实现的情况下，于 rewrite_correlated_noise.py 中重新实现相关 Gaussian 噪声构造。重写成功恢复了三个核心 invariant：

\[
\operatorname{Var}(\varepsilon_+)
=
\operatorname{Var}(\varepsilon_-)
=\sigma^2,
\qquad
\operatorname{Corr}(\varepsilon_+,\varepsilon_-)=\rho.
\]

第一次 review 发现参数验证发生在随机抽样之后：非法调用虽然抛出异常，却会推进 RNG 状态。将验证移动到抽样之前后，失败调用不再具有随机副作用。

最终验证覆盖：

- \(\rho=1\) 时 common-mode noise；
- \(\rho=-1\) 时 opposite noise；
- \(\sigma=0\) 时零噪声；
- 非法输入不推进 RNG；
- 每次有效调用恰好消耗两个 Gaussian draws；
- 经验边际方差和相关系数符合理论；
- 重写版与原版具有一致的分布结构。

这次重写说明掌握的不只是公式表面，而是“两个独立变量如何构造指定协方差结构”及其可复现性要求。

## 阶段结论

Taylor expansion 第一轮的退出标准已满足：理论模型、运行前预测、用户主写实现、实验验证、误差归因、持久化测试与闭卷恢复均已完成。
