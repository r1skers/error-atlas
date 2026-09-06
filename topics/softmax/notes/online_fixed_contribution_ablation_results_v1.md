# 固定贡献消融 v1：长链劣势被普通加法重现

2026-09-06。用户授权 agent 完成本轮核心及审计。此次是已有输入上的事后机制消融，
不是新增分布上的 confirmation，也不是新机制或工程收益的证明。

## 交付与证据边界

使用 [expansion_v1](../experiments/online/runs/expansion_v1/README.md) 的 64 族输入：
32/128/512/2048 块 × spread 8/25 × 每格 8 族；质量均为 32。
读取保存的 FP32 位模式和 chain/balanced 图，不重新采样。
新增普通加法 128 个根，并重算 online 的 256 个根及 128 个配对差。
原 online 根、A 区间端点及 D 区间端点全部精确一致。

新 bundle：[fixed_contribution_ablation_v1](../experiments/online/runs/fixed_contribution_ablation_v1/README.md)。
manifest SHA-256：`2188109ce74dd8440fdf017d0f47bf286ceb3e20d278de5fa6eb363f06309c8a`。
运行前判读规则保存在 bundle 的 `protocol.md`，包含 agent 的预测并明确不设等价性阈值。
源码、父 manifest、输入来源和算术设置均可核对；首轮计算约 98 秒。
之后完整重放返回 `exact_match`：64 族、128 个普通结果、256 个 online 结果和 128 个配对。
逐族原始分量、固定叶位模式、确定性控制与汇总文件均精确重建。

固定叶为 RN32(n_b × real_exp(m_b−M))，指数差精确、整项只舍入一次；区间端点舍入
一致才接受。普通树复用 rewrite 的 FP32 加法 oracle，且只准备一份叶供两种图共用。
它预先知道全局 M，是离线机制控制。online 的两种 FMA 臂共用同一个普通基线。

## 首要观察

定义 D=A_chain−A_balanced，H=D_online−D_fixed。以下是逐格均值近似值，
不是总体置信区间；精确数值外包、范围与逐族记录都保存在 bundle。

| 块数 | spread | 普通 mean D | online separate mean D | online FMA mean D |
| --- | --- | --- | --- | --- |
| 32 | 8 | 7.8634e-8 | 5.8759e-8 | 3.6002e-8 |
| 128 | 8 | 1.6388e-7 | 1.4854e-7 | 1.4849e-7 |
| 512 | 8 | 3.3158e-7 | 2.3395e-7 | 2.3555e-7 |
| 2048 | 8 | 4.8131e-7 | 4.7014e-7 | 4.9565e-7 |
| 32 | 25 | -1.0474e-8 | -4.3548e-8 | -3.9927e-8 |
| 128 | 25 | 1.0153e-7 | 1.3481e-7 | 9.6359e-8 |
| 512 | 25 | 4.4061e-7 | 3.8970e-7 | 4.1044e-7 |
| 2048 | 25 | 1.0315e-6 | 1.0935e-6 | 1.1116e-6 |

普通加法臂的链劣势也随块数表现出增长，已经足以在本样本中产生原来的主要现象。
2048 块下，两种 spread 各自的 8 族在普通臂和两种 online 臂中全部 D>0。
各块数格的均值不是某一个固定输入随 n 变化的轨迹，也不能外推到未测输入分布。
32 块、spread 25 的均值为负，保留这一结果，不只展示支持链劣势的格子。

## 相近均值不能代替逐族对照

2048 块、spread 25、separate：mean H=6.1950e-8，但 mean abs(H)=1.7395e-7；
H 范围约 [-1.8377e-7, 6.2692e-7]，8 族中 4 正、4 负。
FMA 对应 mean H=8.0115e-8、mean abs(H)=1.7442e-7，5 正、3 负。
因此均值差有抵消，不能用一个小的 mean H 宣称所有输入的路径等价。

所有 256 个 online/同树普通根对照中，149 个逐位一致；128 个 FMA/族配对中，
41 个同时满足 chain 和 balanced 的两臂根一致。这是共享输入的记录计数，
不是 256 次独立采样，也不是等价概率估计。

带符号分量亦已保留。以 2048/spread25/separate 为例，以下量均除以同族 real 分母：

| 量 | 逐族绝对值的均值（近似） |
| --- | --- |
| 初始化 J | 1.2511e-9 |
| 普通 chain 的 S | 1.0862e-6 |
| online chain 的 R | 1.1418e-6 |
| online chain 的 W | 6.3730e-9 |

这里普通 S 和 online R 都达到长链误差的主尺度。J 小是本格观察，不是经过绝对值后
自动抵消的代数性质；R 仍含重标定乘法/FMA，H 仍是两条完整算法路径之间的差。
这份消融没有将 H 进一步隔离为某一个单独的因果来源。

## 确定性控制与审计

- 全 max 相同：普通树与 online separate/FMA 根逐位一致；chain/balanced 本身仍可不同。
- 同一组 0,-1,...,-7 改为 max-first、max-last、ascending，质量均 1。
  固定贡献的 multiset 和共同参考不变。max-last 的 online FMA chain 根为 `3fca6cd3`，
  普通 chain 为 `3fca6cd2`；这只是 1 ULP 路径差异，不能当成主导新机制。
- 一叶、最终舍成零、先 exp 下溢再乘质量、非 FP32 精确差、无法认证舍入均有测试。
- 20 条核心/runner 测试通过；6/6 初始化变异体被杀，且检查了预定针对性用例。
- Softmax 全套回归运行 437 条通过；专项 20 条另覆盖新增的 R 残差交叉核对。
  维护检查 9 条通过，`git diff --check` 无错误。
- 保存/重放、篡改拒绝、旧结果不匹配拒绝、重复/缺失记录拒绝均有覆盖。
  R 另与逐节点带权残差之和独立交叉核对，S 与普通加法残差之和闭合。

```sh
python tools/run_tests.py --suite softmax -p "test_online_fixed*.py" -v
python tools/audit_online_fixed_ablation_mutants.py
python -m topics.softmax.experiments.online.fixed_ablation_runner replay topics/softmax/experiments/online/runs/expansion_v1 topics/softmax/experiments/online/runs/fixed_contribution_ablation_v1
```

## 当前决定与下一入口

本批“长 chain 更差”的现象已经能由普通 FP32 求和产生，不能继续把它单独包装成
online 特有机制。confirmation/bootstrap 保持暂停；不因此直接进入 selector 或分段优化。
同时不声称 online 没有特有问题，也不把有限对照的相近均值当作算法等价性证明。

若继续机制研究，下一块用户主写的核心应是刻意制造反复重标定的受控输入，
沿用普通/online 同输入、同图对照，检查是否出现需要进一步解释的路径差异；
不要继续只扩充随机样本来证明已有平均 D>0。GPU exp/FTZ 和输出 O 仍为后续候选轴。
