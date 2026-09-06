# Current research status

2026-09-07：**当前 Softmax / reduction-tree 研究阶段已结题归档，无待执行研究任务。**
用户决定在完成标量输出诊断后停止本轮研究。结题表示停止投入当前候选，
不表示证明所有 online softmax 误差都无关紧要，也不表示已有生产方案。

先读 [结题说明](topics/softmax/notes/online_research_closeout_2026-09-07.md)。
这仍是唯一的当前状态入口；详细过程移入
[结题前历史快照](docs/history/2026-09-07-softmax-handoff.md)。

## 保留的结论与证据

- 旧 reduction-only 确认、负结果与独立复现保留各自证据等级，见
  [Softmax 主题入口](topics/softmax/README.md)。
- [固定贡献消融](topics/softmax/notes/online_fixed_contribution_ablation_results_v1.md)：
  普通求和重现当前样本中的长链劣势，尚未建立 online 特有的新标度律。
- [W 路径审计](topics/softmax/notes/online_weight_path_audit_v1.md)：
  正确舍入 exp 下也存在 W 增长的构造反例，原样本的平坦 W 不是普遍定理。
- [标量输出诊断](topics/softmax/notes/online_scalar_output_v1.md)：
  64 族输入、5 个 V 模式；非恒定 V 的 384 条相关 online 树形配对中，
  FP32 输出 332 条不同，FP16/BF16 存储后均为 0 条不同。完整结果已精确重放。

## 归档范围

停止本轮 confirmation/统计校准、selector、分段优化、重标定扩扫及 CUDA 租卡/dump 计划。
历史协议和学习记录中的未完成项保留为历史，不自动续跑，也不事后补成运行前预测。
源码、测试、冻结输入与结果留在原位置，供学习和复核；归档不删除证据。

仅在用户另行提出明确问题时重开。若目标是工程收益，应先给出应用路径、输出格式、
容许误差和成本预算，并重新做已知性筛查及冻结停止规则。

## 维护与复核

- 代码：[实验索引](topics/softmax/experiments/README.md)。
- 复核：`python tools/run_tests.py`；输出重放命令见结题说明。
- 维护：[仓库规则](docs/maintenance.md)。不要批量执行发布型 one-shot runners，
  不要改写冻结 bundle 或其 source hashes。
