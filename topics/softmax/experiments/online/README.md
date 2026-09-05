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
| pilot | [pilot.py](pilot.py)：块族、实数 exp 参照、相对误差指标 | 与 decimal 路线交叉核对；保守性抽样 | 骨架已建 |
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
