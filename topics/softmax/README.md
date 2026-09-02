# Softmax

研究输入扰动和有限精度误差如何经过 normalization 与 reduction 传播，
并区分问题条件性、算法稳定性、输入表示误差和统计预测能力。

## 导航

- [当前续接点](../../NEXT_SESSION.md)：当前状态和待冻结的下一步。
- [基础与 exact-oracle 笔记](notes/foundations.md)：Jacobian、方向性、浮点预算、P0–P5。
- [早期实验说明](notes/early_experiments.md)：shift resolution、summation 与 failure triage。
- [实验代码索引](experiments/README.md)：按模块角色查找入口。
- [结果索引](experiments/results/README.md)：按证据等级引用研究结论。
- [回归测试](tests/)：与实验实现分离。

## 研究阶段

| 阶段 | 已建立的范围 |
| --- | --- |
| 基础与 exact oracle | 显式非负 FP32 reduction tree 的精确语义；保留早期 accepted/provisional 划分 |
| Depth-margin baseline | 已归档 universal ordering 反例 |
| Calibration diagnostics | 结构特征、second moment、history/phase、beam 与成本探索；不是确认数据 |
| Energy beam v1 → fixed-K8 v2 | v1 primary negative；v2 在新受控输入上通过 pooled 确认 |
| Score-only → offline reuse | oracle-free prototype 仍贵；offline reuse 未过 balanced-FP32 门槛 |
| Online risk certificate | 最新完成阶段是 calibration，有统计信号，但未通过新确认或部署 |

最新方向是稀疏 exactness correction，不是继续堆叠全局拓扑分数。
具体结论与边界以 [NEXT_SESSION.md](../../NEXT_SESSION.md) 链接的 artifacts 为准。

## 回归与边界

从仓库根目录运行：

```sh
python tools/run_tests.py --suite softmax
```

测试成功不代表未测量的 repeatability、跨分布泛化或 GPU 性能已被验证。
研究核心的 ownership 继续遵循
[实现学习协议](../../framework/implementation_learning_protocol.md)。
