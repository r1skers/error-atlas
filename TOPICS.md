# Topic Registry

更新时间：2026-08-12

## 组织原则

每个 topic 是一个具体数学对象、算法或计算系统。它都接受同一组问题：

> reference and metric -> sources -> propagation -> estimation and bounds -> control -> optimization -> verification

topic 之间可以共享工具和结论，但不预设必须存在一条线性的学习顺序。

## Registry

| Topic | 当前问题 | 状态 |
| --- | --- | --- |
| Taylor expansion | remainder、bound quality、传播、稳定性与 bias–variance 优化 | Completed — first pass |
| Softmax | 输入扰动、有限精度故障与 graph-aware error prediction | Active — exact oracle accepted; statistical predictor validation not started |

## Completed：Taylor expansion

当前从 Taylor remainder 切入，依次研究：

- actual remainder、representation、asymptotic order 与 bound；
- Lagrange、integral 与 Peano remainder；
- error-bound tightness；
- Taylor remainder 作为非线性传播的遗漏项；
- 数值微分中的 truncation、cancellation 和 roundoff；
- Richardson extrapolation、确定性误差预算与统计误差预算；
- 带相关噪声的中心差分和最优步长；
- 理论模型与可复现 Monte Carlo 实验的相互验证。

详细计划见 `topics/taylor-expansion/README.md`。

当前理论、实验、数据、测试与 closed-book rewrite 均已归档。该 topic 第一轮已经 Completed，后续高级内容按新问题重新开启。

## Active：Softmax

Softmax 已从二维线性映射的方向性误差进入正式研究。当前已完成：

- 数学参考值与 finite-precision implementation 之间的误差；
- logit perturbation 通过 softmax Jacobian 的传播；
- 多分类 Jacobian 的切空间、概率加权方差表示与方向相关谱；
- 三分类均匀／非均匀点的可手算谱，以及局部 \(3/8\) 与全局 \(1/2\)
  operator-norm 界的区别；
- subtract-max 对 overflow 的控制及其输入量化边界；
- exp、求和与除法误差的一阶传播预算及下溢边界；
- 问题条件性、求值算法稳定性和输入表示误差的拆分；
- FP32 在 \(2^{24}\) 附近丢失单位 logit difference 的可复现实验；
- 顺序、固定平衡树、补偿求和与 FP64 accumulator 的受控处置矩阵；
- raw observation、repeatability/accuracy summary、consumer policy 与
  failure/warning code 分层；
- power/decimal tail 的双 layout、midpoint below/tie/above 对照、
  correctly-rounded policy 与 policy-free summary；
- 只依赖 stored FP32 input、显式二叉 reduction tree 与 RN-even 的 exact
  semantic oracle / label generator；当前 accepted evidence 为逐项审查的
  12 行 graph observations，
  完整 36 行 scaled-midpoint 结果保留为 provisional batch replication；
- 一个不同的 nonuniform positive case 已先冻结 preregistration，再执行两张图
  各一次；两行 prediction 均命中，且 correct-rounding decision 独立记录；
  repeatability 未测量，不另建与 failure-triage 重复的 repeat pipeline；
- cheap depth-margin screening score 已明确为运行前排序量而非概率或证书；
  head-depth family 在执行前降级为 known-mechanism calibration，随后一个固定
  $S_{\mathrm{leaf}},M,D_G,R_G$、只改变 sibling grouping 的预注册 pair
  产生相同 score 但不同 correct-rounding labels，推翻 universal ordering；
- 原实现、边界测试、CSV、metadata 与 closed-book rewrite。

当前只接受一个 preregistered nonuniform case，不声称 family generalization，
也不声称 repeatability。该 case 的 repeatability 若成为 consumer requirement，
应接入已有 failure-triage raw/summary/assessment pipeline。当前没有自动排定的
下一实验；depth-only proxy 不继续补丁式扩展。下一研究入口是先由用户冻结受控
分布、targets、split 与统计 metrics，再主写最小 metric 实现；真实 attention
输入只能作为受控验证之后的新选择，而不是既有承诺。GPU 基础与性能测试继续暂停。

## 新 Topic 的最小模板

每个新目录先只建立一个 `README.md`，回答：

1. **Object**：研究的数学或计算对象是什么？
2. **Reference**：什么被视为参考真值？
3. **Metric**：怎样衡量误差？
4. **Sources**：误差从哪些环节进入？
5. **Propagation**：哪些结构放大、衰减或重分配误差？
6. **Control**：有哪些可调机制？
7. **Optimization**：精度与成本之间怎样权衡？
8. **Verification**：用什么证明、反例或实验验证？

只有当推导、代码或实验实际出现后，才继续拆分 `derivations/`、`experiments/` 和 `findings/`。
