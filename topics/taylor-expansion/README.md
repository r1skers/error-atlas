# Topic — Taylor Expansion

## 当前问题

从 Taylor remainder 出发，研究一个精确误差对象怎样逐步获得表示、渐近阶、上界、传播规律和可优化的误差模型。

核心对象是

\[
P_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k,
\qquad
R_n(x)=f(x)-P_n(x).
\]

需要始终记录函数 \(f\)、展开点 \(a\)、阶数 \(n\) 和评价点 \(x\)。任何余项公式还必须附带自己的 smoothness 与 interval assumptions。

## Error map

- **Reference**：精确数学值 \(f(x)\)。
- **Approximation**：Taylor polynomial \(P_n(x)\) 或它的浮点实现 \(\widehat P_n(x)\)。
- **Primary error**：\(R_n(x)=f(x)-P_n(x)\)。
- **Additional errors**：系数近似、浮点舍入、求值顺序与 cancellation。
- **Controls**：阶数 \(n\)、展开点 \(a\)、步长或评价距离、表示方法与计算精度。
- **Optimization question**：在给定成本下，怎样选择这些控制量使总误差最小？

## Checkpoints

### CP00 — 误差语言（完成）

- 区分 actual error、representation、asymptotic order 与 bound；
- 明确 big-\(O\) 和 little-\(o\) 提供什么信息。

### CP01 — 对象、假设与 Lagrange remainder（完成）

- 明确展开点、阶数、区间和光滑性假设；
- 区分 remainder 的定义与各类 remainder theorem。

### CP02 — Integral remainder（完成）

- Lagrange remainder；
- integral remainder；
- 比较区间积分、未知中间点与最坏情况上界保留的信息。

### CP03 — Peano remainder（完成）

- 从 differentiability 的定义理解 \(o(h)\)；
- 推广到 \(R_n=o(h^n)\)；
- 比较 Peano 与 Lagrange/integral 所需假设和所得结论。

### CP04 — 界的质量（完成）

- 从 integral remainder 推出统一上界；
- 比较 actual error 与 bound；
- 用 \(e^x\)、\(\sin x\) 和 \(1/(1-x)\) 检查 tightness 与边界。

### CP05 — 误差传播（完成）

- Taylor remainder 作为局部线性化遗漏项；
- 复合函数中的扰动传播；
- 计算链中的下游放大、抵消与相关误差。

### CP06 — 控制与优化（完成）

- 用阶数 \(n\)、步长 \(h\) 和展开点控制 truncation error；
- 加入 floating-point roundoff 与 cancellation；
- 研究数值微分中的最优步长。

## 首个应用：数值微分

前向差分

\[
D_hf(x)=\frac{f(x+h)-f(x)}{h}
\]

将用于把 Taylor truncation error 与 floating-point cancellation 放入同一个模型：

\[
E(h)\approx C_1h+C_2\frac{u}{h}.
\]

目标是解释最优步长为什么存在，并说明 \(C_1,C_2\) 与函数尺度、具体计算模型之间的关系。

随后用带相关噪声的中心差分把确定性偏差与随机方差统一为

\[
\operatorname{MSE}(h)
=
\left(\frac{\sinh h}{h}-1\right)^2
+
\frac{\sigma^2(1-\rho)}{2Nh^2}.
\]

这一步用于验证偏差—方差分解、\(N^{-1/2}\) 平均律、相关噪声抵消和统计意义下的最优步长。

### Implementation ownership

该实验进入代码阶段时采用 `framework/implementation_learning_protocol.md`：

- 用户先 explain-back 总误差模型和实验伪代码；
- agent 搭建函数签名、测试接口、数据记录与绘图 skeleton；
- 用户实现 Taylor/finite-difference core；
- 改变 \(h\) 或 precision 前，用户先预测误差方向、量级与曲线形状；
- review 后 closed-book 重写 1–2 个关键 evaluator，再用 diff 和测试比较。

## Topic 退出标准

完成本 topic 的第一轮研究时，应当能够：

1. 不混淆等式、渐近式和上界地陈述 Taylor remainder；
2. 从假设出发推导至少一种余项表示；
3. 判断一个误差界是否有效、是否紧；
4. 写出数值微分的 truncation–roundoff error model；
5. 用理论推导和可复现实验解释最优步长为何存在；
6. 区分单个估计器的采样数 \(N\) 与用于评估估计器的 Monte Carlo 重复数 \(M\)。

## 当前状态（2026-07-28）

- CP00–CP06 已完成并归档；
- 理论、实现、运行前预测、Monte Carlo 验证和误差归因已完成；
- 两个实验均保留原始 CSV、metadata 和图像；
- 持久化测试与 closed-book rewrite 已完成；
- 本 topic 第一轮正式收口；
- 下一阶段入口：向量误差传播、Jacobian 与 Softmax。

## 2026-07-22 历史状态

Checkpoint confirmed：2026-07-22。下次从根目录 `NEXT_SESSION.md` 指向的 CP04 继续。

- 已完成：`CP00_orientation.md`，区分 \(R_n\)、absolute error、big-\(O\) 与 little-\(o\)；
- 已完成：`CP01_lagrange_remainder.md`，从存在性表示推出显式上界与主导项；
- 已完成：`CP02_integral_remainder.md`，理解区间加权、误差符号与信息压缩；
- 已完成：`CP03_peano_remainder.md`，从 differentiability 与 induction 得到局部渐近余项；
- 当前入口：`CP04_bound_tightness.md`；
- 下一步：比较 exact remainder 与通用 Lagrange bound；
- 尚未开始：误差传播、数值实验与结论归档。
