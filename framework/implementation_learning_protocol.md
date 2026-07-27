# Implementation Learning Protocol

## 目标

代码阶段不仅要得到正确 artifact，还要训练从空白文件构造核心实现的能力。默认避免把“能跟随解释和 review”误当成“能独立产出”。

## Ownership split

开始编码前先判断任务属于哪一类：

### Research core — 用户主写

承载研究机制、关键 invariant 或主要结论的部分，默认由用户从伪代码和空白骨架开始实现。例如：

- Taylor polynomial/remainder evaluator；
- finite-difference core；
- softmax normalization 与 online update invariant；
- 误差传播、error budget 或关键数值稳定化逻辑。

agent 的默认角色是追问、分级提示、审查、构造反例和验证，不直接替换核心实现。

### Scaffolding — agent 可协助

不承载主要研究判断的部分可由 agent 搭建，例如：

- 函数签名、类型、docstring 和 TODO；
- CLI、参数解析、测试 harness、CSV 输出和绘图样板；
- 重复性数据整理、环境配置和陌生库的最小调用示例。

### New API — agent 可先示范

第一次接触陌生 API 时，agent 可以给一个最小、可运行、逐行解释的示范。示范代码与研究核心分开，避免 API plumbing 掩盖算法 ownership。

用户的显式指令可以切换模式；以上是默认协议，不是禁止 agent 实现的硬规则。

## Active-output loop

核心实现使用以下顺序：

1. **Explain-back**：用户先说明目标、输入输出、关键 invariant、误差来源和伪代码。
2. **Predict**：运行前写出结果方向、粗略量级、曲线形状和预期失败点。
3. **Skeleton**：agent 只提供函数签名、注释、测试接口或空 TODO，核心逻辑留白。
4. **User implementation**：用户完成第一版；允许不完整，但要暴露真实卡点。
5. **Review before rewrite**：agent 先指出 correctness、assumption、numerical stability 和 evidence 问题，不静默改写核心代码。
6. **Run and compare**：实际运行，并逐项比较预测与观察。
7. **Error attribution**：将偏差归因到理论模型、实现缺陷、浮点效应、测试设计或测量噪声。
8. **Closed-book rewrite**：选 1–2 个 invariant-bearing 函数，在不看原实现的情况下从空白重写。
9. **Diff and record**：对比两个实现，将真正学到的 invariant、失败模式和测试证据写入 topic。

## Hint ladder

核心实现卡住时，agent 按最小充分帮助逐级增加信息：

1. 提一个定位问题；
2. 提醒关键 invariant 或边界条件；
3. 给一个手算小例子或反例；
4. 给函数 skeleton 与分段 TODO；
5. 给局部伪代码；
6. 只有在用户明确要求或该部分属于非核心 plumbing 时给完整实现。

默认停在解决当前卡点所需的最低一级。

## Prediction record

运行实验前至少写四项：

- **Direction**：结果应上升、下降、变号还是出现极小点？
- **Scale**：预计是 \(O(h)\)、\(O(h^2)\)、\(O(u/h)\) 还是其他量级？
- **Boundary**：哪个输入区间可能破坏假设？
- **Failure signature**：若理论或实现有误，最可能看到什么现象？

运行后不能只记录 pass/fail，还要写：预测是否命中、偏差多大、原因是什么。

## Closed-book rewrite safety

重写前保留原实现作为可比较证据：优先使用已提交版本、独立 rewrite 文件或临时练习分支。重写完成前不破坏唯一工作副本。目标是 retrieval practice 和差异分析，不是丢失 provenance。

## Completion criteria

核心函数只有同时满足以下条件才视为掌握：

1. 用户能从空白写出主要控制流和 invariant；
2. 能在运行前预测至少一个可检验结果；
3. 能解释测试为何覆盖关键误差机制；
4. 能指出实现在哪些假设或数值范围下会失败；
5. closed-book rewrite 保留相同核心机制，而不是仅复述表面语法。
