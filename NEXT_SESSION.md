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

## 当前阶段：独立复现重写（2026-09-02 决定）

继承既有研究路线，用闭卷重写的实现重新走一遍核心链，并对照冻结 artifact 的数字。
目的是独立验证现有代码，不产生新的研究结论；结果记录为 replication，不是 confirmation。

- 范围：exact oracle、树生成器、受控输入、Q_8/12 分数、B=3 cell beam、oracle-free 推理，
  约 1,200 行核心。历史校准脚本、results 目录和两份 checkpoint 文档原样冻结，不重构。
- 方法：按[实现学习协议](framework/implementation_learning_protocol.md)由用户主写；
  每个模块用旧实现做 differential test；新包从一开始按带边权重、多误差通道的 trace 设计接口，
  以便后续阶段直接复用。
- 已完成：oracle 对照硬件 float32 的差分审计，见
  `topics/softmax/tests/test_oracle_hardware_differential.py`；逐对加法与整树状态零不匹配。
- 进行中：[rewrite 分区](topics/softmax/experiments/rewrite/README.md)已建骨架，第一步 oracle 待用户实现。
- 复现检查点：(A) oracle 对硬件；(B) 分数与 beam 在 192 个冻结 v2 组上精确复现决策与分数；
  (C) 独立重算 v2 的 paired regret 与 stratified group bootstrap 区间。

## 后续研究入口（复现完成后决定）

两个候选，尚未冻结：

1. **Online/blockwise softmax 的带重标定 reduction tree**：状态 (m, ℓ)，加权分解
   E² = A_w + C_w，w_v = e^{m_v − m_root}。冻结前必须先定义 exp 的参考语义；
   w_v ≤ 1 意味着重标定只衰减不放大，新 coherence 只能来自 exp/mul 通道或 max jump 触发的
   stagnation，这应作为可证伪的主假设写入预注册。exp/mul 节点恒为 inexact，
   certificate 的 Q_inexact 技巧不能直接继承。
2. **Sparse exactness correction**：全树维护粗尺度，只在 root band 或高能量预算内检查 exact/inexact。

目前倾向第一项。不能把旧校准的 192 个标签重新当作任一新候选的确认数据。
GPU、未知 black-box graph、负数 cancellation 与端到端 Softmax 扩展仍需单独定义范围。

## 继续工作时

- 找代码：[Softmax 实验索引](topics/softmax/experiments/README.md)。
- 检查回归：`python tools/run_tests.py`。
- 改结构前读：[维护指南](docs/maintenance.md)。
- 实验运行前读对应 artifact README；不要批量执行 one-shot runners。
- 当前测试中的数值可移植性容差不是研究 policy，也不是新实验结果。
