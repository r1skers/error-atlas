# Next Session

整理日期：2026-09-02。当前研究状态依据 `ad1fe87` 及其版本化 artifacts；该提交已推送至
远端分支 `research/predictor-validation-protocol`，远端 `main` 尚未合入。
这是唯一的当前续接入口；历史推导与旧待办已归入
[2026-08-12 handoff](docs/history/2026-08-12-softmax-handoff.md)。

## 当前落点

活跃主题是 Softmax 的 **reduction-only 风险预测**，不是“统计验证尚未开始”。

1. [Fixed-K8/B3 v2](topics/softmax/experiments/results/wide_range_fixed_k8_beam_v2/README.md)
   在已冻结受控分布上通过 pooled 确认；不能外推为跨分布或生产低成本结论。
2. [Score-only 实现与成本](topics/softmax/experiments/results/wide_range_fixed_k8_beam_score_only_v1/README.md)
   移除了 oracle instrumentation，但仍需遍历许多候选树。
3. [Offline tree reuse](topics/softmax/experiments/results/wide_range_offline_tree_reuse_v1/README.md)
   的随机固定树比较通过，balanced-FP32 部署门槛失败，结论是 no-go。
4. [Online risk certificate 校准](topics/softmax/experiments/results/wide_range_online_risk_certificate_calibration_v1/README.md)
   对 192 个输入完成 cross-fitted 校准；exactness-weighted energy 有风险区分信号，
   rigorous envelope 太松，尚无 confirmation 或 deployment claim。

上述 stages 的负结果、校准和确认等级不因代码整理而改变。

## 下一研究入口（尚未执行）

依据最新 artifact 的解释，候选是 **sparse exactness correction**：
全树维护粗尺度，只在预定 root band 或高能量预算内检查 exact/inexact。

先由用户说明目标与伪代码，再冻结新版本的预算、公式、输入和评价协议。
不能把旧校准的 192 个标签重新当作新候选的确认数据。
本次整理不授权生成新数据；GPU、未知 black-box graph、负数 cancellation
与端到端 Softmax 扩展仍需单独定义范围。

## 继续工作时

- 找代码：[Softmax 实验索引](topics/softmax/experiments/README.md)。
- 检查回归：`python tools/run_tests.py`。
- 改结构前读：[维护指南](docs/maintenance.md)。
- 实验运行前读对应 artifact README；不要批量执行 one-shot runners。
- 当前测试中的数值可移植性容差不是研究 policy，也不是新实验结果。
