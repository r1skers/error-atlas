# Topic Registry

更新时间：2026-07-28

## 组织原则

每个 topic 是一个具体数学对象、算法或计算系统。它都接受同一组问题：

> reference and metric -> sources -> propagation -> estimation and bounds -> control -> optimization -> verification

topic 之间可以共享工具和结论，但不预设必须存在一条线性的学习顺序。

## Registry

| Topic | 当前问题 | 状态 |
| --- | --- | --- |
| Taylor expansion | remainder、bound quality、传播、稳定性与 bias–variance 优化 | Active — first-round closure |
| Softmax | 输入扰动与有限精度误差如何经过 normalization 传播 | Candidate |

## Active：Taylor expansion

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

当前理论、实验、数据与测试已经归档，只剩一次 closed-book rewrite。完成后该 topic 第一轮转为 Completed，后续高级内容按新问题重新开启。

## Candidate：Softmax

softmax 暂时只登记研究动机，不提前建立 topic 目录。它可能包含：

- 数学参考值与 finite-precision implementation 之间的误差；
- logit perturbation 通过 softmax Jacobian 的传播；
- `exp` approximation、overflow、underflow、求和与除法误差；
- subtract-max 如何利用 shift invariance 改善数值范围；
- 误差继续传播到 cross-entropy、attention output 或 sampling；
- FP32、FP16、BF16 与 mixed-precision accumulation 的误差—成本权衡。

真正启动该 topic 时，再明确 reference、metric、问题边界与第一组实验。

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
