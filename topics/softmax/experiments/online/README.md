# Online normalizer 分区

新研究代码，实现 [算术合同](../../notes/online_normalizer_contract.md) 的 $(m,\ell)$ 部分。
与 [rewrite/](../rewrite/) 的区别：那里是闭卷复现，不产 artifact；这里是新阶段的研究核心。

按[实现学习协议](../../../../framework/implementation_learning_protocol.md)，
标 `USER-WRITTEN CORE` 的函数由用户主写，agent 提供骨架、测试、追问与反例。
当前协作约定：审计确认的缺陷由 agent 直接修复、验证并报告；新增研究核心仍由用户主写。

## 顺序与状态

| 步骤 | 模块 | 对照 | 状态 |
| --- | --- | --- | --- |
| 3 | [fp32_signed.py](fp32_signed.py)：有符号 RN-even 与四个受舍入算子 | `rewrite.fp32_oracle`（非负子集）、硬件 float32 | 通过差分测试（2026-09-04）；explain-back 待回填 |
| 1c | [schedules.py](schedules.py)：CUDA 可表达的 schedule 族 | 结构性质自检 | agent 脚手架，已通过 |
| 1b | [merge.py](merge.py)：(m, ℓ) 递推、`MergeDump`、frozen-weight 恒等式 | 独立重算的残差；解析权重负控制 | 通过（2026-09-05） |
| 1a | [fp32_exp.py](fp32_exp.py)：正确舍入的 FP32 exp，即 specified-exp reference | 独立区间参照；ULP 分布 | 已实现并通过测试；学习记录待回填 |
| pilot | [pilot.py](pilot.py)：块族、实数 exp 参照、相对误差指标 | 与 decimal 路线交叉核对；保守性抽样 | 核心已实现，探索性算例可运行 |
| 2a | dump 传输层与硬件 provenance | — | 未开始 |

## 为什么需要第 3 步

`rewrite/fp32_oracle.py` 的 `round_to_fp32` 只接受非负输入——那是旧线"非负 FP32 归约树"
合同的直接产物。online 设定里 $m$ 是有符号 logit，$\hat\Delta=m_a\ominus m_v$ 也有符号，
因此合同边界必须显式改写，不能靠 wrapper 绕过。

## 检查

```sh
python tools/run_tests.py --suite softmax -p test_online_fp32_signed.py -v
```

核心未实现时测试自动 skip；实现后必须与参照逐值精确一致。

## Pilot 的使用边界

`pilot.py` 中的 `denominator_interval` 与 `relative_error` 已实现。
运行 `python tools/run_tests.py --suite softmax -p test_online_pilot.py -v` 核对实现与确定性算例。
首轮 scratch 的解读修正见[合同 §12](../../notes/online_normalizer_contract.md#12-pilot-首轮设计已定)：
输出的 FP32 间距不能当作高精度参照的测量分辨率，32 块也能分出不同 schedule 的误差。
Explain-back 可补充当前理解；已有测量只能记为事后观察，下一轮 prediction record 必须在运行前写。

- real-exp 区间除了 FP32 舍入交叉核对，还会对照独立的 320 位 Decimal 区间。
  该参照直接用有向十进制除法包住有理指数，不经过 binary float；核对的是实数区间包含性。
- `Measurement.has_valid_error_interval` 只表示区间合法。同一 `BlockFamily` 的结果可用
  `error_difference_interval(other)` 取保守配对差，`error_order(other)` 返回 -1／1／0
  分别表示误差更低／更高／已证明相等；区间不足以判定时返回 `None`。
  区间相减可能比利用共同参照的联合分析更宽；相同计算输出的误差则必然相等。
- `absorbed_merges` 只计非零贡献的完全吸收；`exp_underflow_edges` 和
  `product_underflow_edges` 单列。FMA 没有独立乘法舍入，因此后一计数恒为零；用于
  判断吸收的单项舍入值只是反事实对照，不是 FMA 实际执行的中间步骤。
- 块族必须非空，块大小必须是非 bool 的整数，满足 `1 <= n_b <= 2**24`。

这些是维护测试和诊断口径，不是新的 pilot 效应或确认结果。
