# Softmax / reduction-tree 研究阶段结题

2026-09-07。用户决定结束当前研究阶段，转入整理、提交和归档。
当前候选已完成从分母现象、机制归因、普通求和消融到标量输出诊断的检查，
尚未显示足以支持继续工程投入的收益。本次结题是研究资源决定，不是普遍无效定理。
唯一的当前状态入口为 [NEXT_SESSION](../../../NEXT_SESSION.md)。

## 可以保留的结论

1. **已有 reduction 结果保留原证据等级。** Fixed-K8/B3 在冻结受控分布上的窄赢已独立复现；
   offline reuse 未过 balanced-FP32 部署门槛；online risk certificate 只到校准。
   参见 [主题入口](../README.md)与[复现记录](rewrite_replication.md)。
2. **当前 long-chain 劣势无需诉诸新机制才能重现。**
   [固定贡献消融](online_fixed_contribution_ablation_results_v1.md)中，普通 FP32 加法也产生
   同量级树形差距。原样本的 R 主导与经典求和误差相容，不等于普通与 online 算法逐族等价。
   [已知性筛查](online_prior_work_screening_v1.md)保留相关基线及解释修正。
3. **不能把平坦 W 推成普遍自限。**
   [确定性路径反例](online_weight_path_audit_v1.md)在正确舍入 CPU exp 下得到 R=0、
   W 随 chain 长度增长；2048 块约为 256u。这否定原先过强的解释，
   尚不证明新理论贡献或真实应用中的重要性。
4. **分母差异不等于最终输出收益。**
   [标量输出结果](online_scalar_output_v1.md)中，恒 V=0/1 输出完全精确；
   非恒定 V 的 384 条 online 树形配对中，332 条 FP32 输出不同，
   FP16/BF16 存储后均无差异。普通求和对照仍有同量级 FP32 差距。
   这些配对共享输入和算术设置，不是 384 个独立样本；存储相同也不等于误差为零。

## 停止范围与未覆盖边界

本阶段停止新增 confirmation/统计校准、selector、分段方案、重标定压力扫描及
CUDA 租卡/dump 工作。未执行的历史计划撤出当前待办，现有学习代码和预检保留。
没有把看过数据后的阈值补写成事前工程判据。

输出证据仅覆盖保存的 64 族常值叶块、5 个标量 V 模式、正确舍入 CPU exp、
FP32 累加与最终 FP16/BF16 存储 cast。它不覆盖 QK/PV matmul、向量 V、训练/反向传播、
实际 GPU exp/FTZ 或其他分布，也未证明 FP32 下游应用可以忽略这些误差。
当前没有给定这样的应用容差或成本预算，故不能宣称生产收益或普遍安全。

将来只有新的明确问题才重开；工程方向须先定义应用、格式、容差与成本，
进行已知性筛查并冻结停止规则。这些条件是重开入口，不是当前待办。

## 可重放交付与验收

- [输出 bundle](../experiments/online/runs/scalar_output_v1/README.md)：64 族、320 份参考、
  1920 条测量、960 条配对；包含输入关联、协议、完整数值依赖源码及 SHA-256。
  manifest：`2c18dec51c4c3e73c4a253674825516a27e7b82f834b396b2ee6a1620133e944`。
- 完整输出重放已返回 `exact_match`，原分母 256 个根/A 与 128 个 D 同时精确复现。
- 数值交付验证：Softmax 475 条、输出专项 37 条通过，输出变异 8/8 被杀；
  维护测试 9 条通过。结题整理只更新说明和导航，不改冻结结果或数值源码。
- [结题前历史快照](../../../docs/history/2026-09-07-softmax-handoff.md)保留预测、
  审计修正和路线调整；不回填未完成的学习记录。

从仓库根目录复核（重放会重新计算，耗时数分钟）：

```sh
python tools/run_tests.py --suite softmax
python tools/run_tests.py --suite maintenance
python tools/audit_online_output_mutants.py
python -m topics.softmax.experiments.online.output_runner replay topics/softmax/experiments/online/runs/expansion_v1 topics/softmax/experiments/online/runs/scalar_output_v1
```

证据保持原路径和字节；只归档当前研究阶段，不删除实现，不覆盖旧 artifact。
