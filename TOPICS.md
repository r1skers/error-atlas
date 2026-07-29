# Topic Registry

更新时间：2026-07-29

## 组织原则

每个 topic 是一个具体数学对象、算法或计算系统。它都接受同一组问题：

> reference and metric -> sources -> propagation -> estimation and bounds -> control -> optimization -> verification

topic 之间可以共享工具和结论，但不预设必须存在一条线性的学习顺序。

## Registry

| Topic | 当前问题 | 状态 |
| --- | --- | --- |
| Taylor expansion | remainder、bound quality、传播、稳定性与 bias–variance 优化 | Completed — first pass |
| Softmax | 输入扰动与有限精度误差如何经过 normalization 传播 | Active — first FP32 probe verified |

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
- subtract-max 对 overflow 的控制及其输入量化边界；
- FP32 在 \(2^{24}\) 附近丢失单位 logit difference 的可复现实验；
- 原实现、边界测试、CSV、metadata 与 closed-book rewrite。

下一步将方向性结论推广到多分类 Softmax，再研究更多 finite-precision
source、后续 loss 传播和精度—成本权衡。

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
