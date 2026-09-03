# 闭卷重写分区

目的：按[实现学习协议](../../../../framework/implementation_learning_protocol.md)，
由用户从空白骨架重写研究核心，并用旧实现做 differential test，独立验证既有结论。
这里的代码是 replication，不生成研究 artifact，也不替换任何冻结阶段的 source。

## 规则

- 旧模块（`summation_graph_predictor.py` 等）只作为测试里的参考真值，本包不导入它们。
- 用户主写标记为 `USER-WRITTEN CORE` 的函数；agent 只提供骨架、测试、追问与反例。
- 每个模块先填 docstring 里的 explain-back 和 prediction record，再写实现。
- 测试在函数仍抛 `NotImplementedError` 时自动 skip；实现后必须与旧实现逐值精确一致。
- 不写结果文件，不重跑任何 one-shot runner。

## 顺序与状态

| 步骤 | 模块 | 对照 | 状态 |
| --- | --- | --- | --- |
| 1 | [fp32_oracle.py](fp32_oracle.py)：RN-even 舍入与整树精确求值 | 旧 oracle 与硬件 float32 | 通过差分测试（2026-09-03） |
| 2 | [coherence.py](coherence.py)：A/C 分解，复现"C 主导"发现 | reduction_analysis 的 A/C；宽度 256 上 std C / std A > 1.5 | 通过差分与复现测试（2026-09-03） |
| 3 | [generators.py](generators.py)：树与受控输入生成器 | 旧生成器；冻结 v2 的 192 组 stored_leaf_bits 与 graph_sha256 | 骨架已建 |
| 4a | [macro_score.py](macro_score.py)：Q_8/12 分数与 shortlist | 冻结 v2 CSV 的 q_score/capture/shortlist/q_selected | 通过复现测试（2026-09-04） |
| 4b | B=3 cell beam（训练 probe，不重写） | 由 test_predictor_fixed_k8_beam_inference 守住 | 范围外 |
| 5 | 独立重算 v2 的 paired regret 与 bootstrap 区间 | 冻结 heldout 汇总 | 未开始 |

## 复现记录

步骤 2 用重写的 oracle 与分解独立重算 std(C)/std(A)，宽度 256、32 棵树，与旧实现逐值一致：

| 输入 seed | 重写实现 | 旧实现 |
| ---: | ---: | ---: |
| 1 | 3.531 | 3.531 |
| 2 | 2.550 | 2.550 |
| 3 | 3.120 | 3.120 |
| 22260821 | 3.609 | 3.609 |

这是 replication，确认既有代码在这一步没有实现错误；不构成新的研究证据。

## 检查

```sh
python tools/run_tests.py --suite softmax -p test_rewrite_fp32_oracle.py -v
```

每步通过后，把与旧实现的差异、真正学到的 invariant 和失败模式记到
[notes/](../../notes/)，再进入下一步。
