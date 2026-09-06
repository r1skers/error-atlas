# 固定全局贡献 vs online 递推：最小消融 v1

2026-09-06。用户进一步提出：先区分普通 reduction 效应和 online 路径效应，
再决定是否扩展输出或优化分段树。confirmation/bootstrap 校准保持暂停；
[输出门槛设计](online_output_gate_v1.md) 暂作后续选项，本消融成为当前前置任务。
当前核心由 agent 按用户明确授权实现；下述判读规则在 64 族消融运行前写定。
运行已完成并通过完整重放，见[结果与下一入口](online_fixed_contribution_ablation_results_v1.md)。
bundle 中的 `protocol.md` 保留实际运行前的原文，不含本条事后链接。

## 需要检验的假说，而非既定结论

讨论提出的假说：当前 chain 劣势主要可由固定全局贡献上的普通 FP32 求和重现。
该假说尚未验证，不把引述内容自动记成用户自己的量化预测。

R 是冻结实际权重后的乘加/FMA 舍入，不只是普通加法舍入。重标定次数、位置和中间尺度
可以通过 R 影响误差。W 是 frozen reference 相对 real-exp 总分母的净偏差；
净 W 小也可能含路径间抵消。因此“W 未随块数增长”不能独立推出“online 没有新机制”。
即使消融差异出现在 R 中，也不能把它预先排除出 online 相关机制。
反过来，一次输出差异也不证明机制具有新颖性或工程重要性。

## 固定贡献怎样落到 FP32

对同一个 BlockFamily，M=max_b m_b、ell_star=sum_b n_b exp(m_b−M)。
主基线初始化为：

```
c_b = n_b * exp(m_b - M)           # 精确实数含义，指数差用精确有理数
c_hat_b = RN32(c_b)               # 整项只舍入一次
```

用 `real_exp_interval` 包住 exp，两个端点先乘 n_b，再各自 RN32；结果相同才接受。
固定精度不能证明时报告未判定/失败，不能用中点替代。保留所有零贡献与输入次序。
不能使用 `correctly_rounded_exp(fp32_sub(m_b,M))` 冒充这个初始化；那是另一条算术路径。
尤其不能先将 exp 下溢到零再乘质量，因为整项可能仍可表示。

准备好的 c_hat 位模式只计算一次，普通 chain/balanced 共用；普通合并只做 FP32 add，
不存在 separate/FMA 两种不同算术。online 的两种 FMA 设置分别与同一个普通基线配对。
基线预先知道最终 M，是离线机制参照，不能视为免费的在线替代算法。

## 不混淆叶量化与累加舍入

每条普通加法树保存：

```
C = sum_b c_hat_b                  # 精确 Fraction 求和
J = C - ell_star                   # 初始化误差，区间；与普通树形无关
S_T = fixed_root_T - C             # 纯加法树的净舍入，精确 Fraction
E_fixed_T = S_T + J                # 对共同 real-exp 分母的总误差
```

online 仍使用原有 E_online_T=R_T+W_T，不改变归因定义。
四种计算都对同一个 ell_star 测 A=abs(root−ell_star)/ell_star。
J 虽然与树形无关，但经过绝对值后不会必然在 A_chain−A_balanced 中抵消，
所以除了 A 和差距，还必须保存 J、S、R、W 的带符号结果及两个根的差。

若初始化路径足以解释差异，再增加一次“全局 FP32 初始化”桥接臂：
RN32(n_b * Exp32(RN32(m_b−M)))，内部树仍只相加。它不是首轮必须扩张的实验轴，
也不能将其结果与理想整项舍入混名。

## 小范围配对比较

先复用已有 expansion_v1 的 64 族保存输入、chain/balanced 图和 online 两种 FMA 结果，
新增固定贡献与普通加法结果。每份输入在普通/online 两条路径中必须完全相同。
这是事后机制消融，不重新命名为 confirmation。

首轮输出逐族：

- D_online = A_online_chain−A_online_balanced（每种 FMA 一列）。
- D_fixed = A_fixed_chain−A_fixed_balanced（一列）。
- H = D_online−D_fixed，以及同树 online_root−fixed_root 的精确差。
- J、S、R、W 和原有 max/gap/weight 轨迹，必要时按路径统计重标定次数。

H 是两条具体算法路径的配对差，不是自动隔离出的单个因果分量；绝对误差可能因符号抵消
而看起来相同。先报告精确数值外包与逐族分布，不以均值、比值或少数相同根代替所有对照。
本轮不设置等价性检验或“几乎相同”通过阈值，不输出等价/不等价的二元结论。
预先固定报告每格 mean D_online、mean D_fixed、mean H、mean abs(H)、H 范围与正负族数；
另报告同树根逐位相同的数量、带符号根差和 J/S/R/W。若普通臂也随块数表现出链劣势，
只说明普通加法在本样本中足以产生该现象；若 H 非零，再看其幅度与符号，不能据此宣称新机制。
不按结果选择样本、FMA 设置或阈值。旧结果已见过，这一轮明确属于事后消融。

Agent 的运行前判断（不是用户预测）：预计普通加法会重现长链劣势；动态路径仍可能改变
单族根与差距，因此不预计逐位全等，也不预言 H 的稳定正负方向。

确定性控制先于这批比较：

1. 全部块 max 相同：exp(0)=1、固定叶等于原质量，同一图的 ordinary/online 应逐位一致。
2. 同一小组 logits 的不同排列：max 靠前、靠后及递增；每种排列在两条路径中同步使用，
   固定原 multiset 与 mass，区别于重新生成输入。用于检查反复重标定的效应。
3. 一叶、零舍入贡献、初始化 rounding 无法判定等接口边界。

此次先不扫描 GPU approximate exp、FTZ、一般叶内 reduction 或 O；它们是后续机制候选，
不必等所有轴做完才解释当前结果。FMA 保留，因为已有 online 输入同时具备两臂。

## 结果如何决定去留

若当前范围的链劣势被普通基线充分重现，则将其归档为“当前观察可由已有求和机制解释”，
停止为了这个观察独立发展 online 理论。不能把有限范围的相近结果推广为 online 无任何
特有问题，也不能把链更差本身作为新的机制发现。

若差异明显，则先沿 J、S、R、W 和重标定轨迹定位，再决定是否增加某个硬件或输出轴。
即使找到区别，也须另查已有文献并通过工程重要性门槛；不自动进入 selector 或分段优化。

## 实现分工与当前入口

用户最新明确授权此次核心由 agent 完成，下一块真正推进研究的核心再交回用户。
`fixed_contribution_ablation.global_contributions` 负责精确初始化与最终舍入认证；
普通 add 树复用已有 rewrite oracle，`fixed_ablation_runner` 保存载荷、分量并执行配对审计。
保存新的初始化位模式、参考区间、J/S 与配对输出，旧 bundle 和源实现保持原样。
