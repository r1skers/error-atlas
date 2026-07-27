# Error Analysis Protocol

每个 topic 都沿用这套协议，避免直接从公式跳到结论。

## 研究循环

> parse -> shrink -> compute -> conjecture -> prove or break -> boundary audit -> write

1. **Parse**：准确写出对象、reference、metric、变量和假设。
2. **Shrink**：先缩到一个能精确计算的最小案例。
3. **Compute**：先写预期结果，再求 actual error，并与近似式或上界比较；涉及代码时遵循 `implementation_learning_protocol.md`。
4. **Conjecture**：提出传播规律、主导项或最优控制量。
5. **Prove or break**：证明猜想，或构造反例击破它。
6. **Boundary audit**：检查端点、奇点、退化情形和假设失效处。
7. **Write**：只把已经澄清的结论写入正式笔记。

## Error Analysis Card

研究任何一个案例时，至少回答：

- **Target**：我们真正想计算或预测什么？
- **Reference**：误差相对于哪个对象定义？
- **Metric**：使用绝对误差、相对误差、范数还是概率风险？
- **Sources**：输入、模型、截断、迭代、舍入、实现误差分别在哪里？
- **Identity**：能否先写出一个精确误差恒等式？
- **Propagation**：误差经过 Jacobian、递推或组合后如何变化？
- **Bound**：界依赖哪些常数和假设？
- **Tightness**：界与实际误差之间相差多少，能否达到？
- **Control**：有哪些可调变量可以降低误差？
- **Cost**：降低误差需要付出什么计算、测量或复杂度成本？
- **Validation**：理论结论如何通过精确算例或实验复核？

代码实验还必须记录：运行前预测、核心实现 ownership、实际结果与预测偏差，以及偏差来自理论、实现还是测量。

## 证据等级

从弱到强区分：

1. 数值观察；
2. 渐近解释；
3. 带假设的显式上界；
4. 精确恒等式；
5. 可达到性、下界或反例。

“实验中没有失败”不能替代误差保证；“存在上界”也不代表该界足够紧。
