# Next Session

整理日期：2026-09-04。冻结证据仍依据 `ad1fe87` 及其版本化 artifacts；该提交已推送至
远端分支 `research/predictor-validation-protocol`，远端 `main` 尚未合入。新阶段的合同与
第一个模块在 `02ec506`（仅本地）。这是唯一的当前续接入口；历史推导与旧待办已归入
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

## 上一阶段：独立复现重写（2026-09-02 决定，2026-09-04 完成）

继承既有研究路线，用闭卷重写的实现重新走一遍核心链，并对照冻结 artifact 的数字。
目的是独立验证现有代码，不产生新的研究结论；结果记录为 replication，不是 confirmation。
详细笔记见 [rewrite 复现笔记](topics/softmax/notes/rewrite_replication.md)。

- 范围：exact oracle、A/C 分解、树与受控输入生成器、Q_8/12 分数与 shortlist、
  regret 与分层 bootstrap。B=3 cell beam（训练 probe）范围外，由既有 8-ULP 测试守住。
  历史校准脚本、results 目录和两份 checkpoint 文档原样冻结，不重构。
- 方法：按[实现学习协议](framework/implementation_learning_protocol.md)由用户主写；
  每个模块用旧实现或冻结 artifact 做 differential test。代码见
  [rewrite 分区](topics/softmax/experiments/rewrite/README.md)。
- **五步全部通过**：(1) oracle 对旧实现与硬件 float32；(2) A/C 与 C 主导；
  (3) 生成器从 seed 精确重建 192 组 stored_leaf_bits 与 12288 个 graph_sha256；
  (4) Q 分数/capture/shortlist/q_selected 逐值复现冻结 v2；(5) 重算 primary 与 95% CI
  逐位等于冻结的 +0.057699 与 [+0.018713, +0.097991]。
- 过程中在重写侧抓到 4 个实现 bug（ulp 返回值、root-band size 下标、手写求和 vs 内置 sum、
  target 平方浮点路径），全部由差分/冻结对照发现。另发现冻结 CSV 的 target 与当前
  `_beam_tree` 源码差 1 ULP（精确平方 vs 二次舍入），即冻结证据早于该源码状态。
- 结论：v2「beam 在受控分布上窄赢 Q」经独立复现成立，不依赖任何被检查的实现错误。

## 当前阶段：online normalizer 的误差几何（2026-09-04 选定，进行中）

候选一（带重标定的 reduction tree）已选定并开工，候选二（sparse exactness correction）
留在架上未启动。**目前只有设计与脚手架，没有预注册、没有研究假设、没有 artifact。**
算术合同见 [online normalizer 合同](topics/softmax/notes/online_normalizer_contract.md)，
代码见 [online 分区](topics/softmax/experiments/online/README.md)。

范围：只有 (m, ℓ)，不含 V、不含 O、不含 y = O/ℓ 的除法。主合同 FP32；低精度是
**压力测试与候选设计**，不是真实 kernel 的默认（FA2/FA3 的 (m, ℓ, O) 累加器保持 FP32）。

已确立的三条，都是设计结论不是研究结论：

1. **加权恒等式只在 frozen-weight 形式下成立**：
   `l_root_hat − l_root_frozen = Σ_v W_v (μ_a + μ_b + α_v)`，其中 `W_v = Π ŵ` 是**实际算出的**
   重标定因子之积，不是解析的 `e^{m_v − m_root}`。已逐位验证（chain/balanced × 三档 spread ×
   FMA 开关），解析权重版本作为负控制按预期失败。
2. **只有 frozen-weight 能保住 Fraction 逐位精确**，因为它把 ŵ 冻成数据，
   exp 的实现误差与 Δm 的舍入都被吸收进去。specified-exp 与 real-exp 含超越数，
   只能高精度核到 N 位。**这同时是它能跨 CPU/GPU 的原因。**
3. **旧 oracle 的非负合同不够用**：m 是有符号 logit，Δm 是相减。
   [fp32_signed.py](topics/softmax/experiments/online/fp32_signed.py) 已通过差分测试
   （对 rewrite oracle 非负子集、对硬件 float32、Sterbenz、两个方向的 tie 与溢出带）。

不能把旧校准的 192 个标签重新当作本阶段的确认数据。未知 black-box graph 与端到端
Softmax 扩展仍需单独定义范围。

## 下一步顺序

| | 内容 | 依赖 | 状态 |
| --- | --- | --- | --- |
| 3 | oracle 有符号合同扩展 | — | **已完成**（`02ec506`） |
| 1 | CUDA 可表达 schedule 族的精确模拟 | 3 | 下一项 |
| 2a | dump 传输层：位模式格式、硬件 provenance 块、CPU 侧核对 harness | — | 可并行 |
| 2b | 要测哪些 schedule 与哪些 exp 变体 | 1 | 未开始 |

CUDA 已决定走**远程租卡**。要点记在合同 §7–§8：

- 本机是 AMD，无 nvcc；VPS 也是 CPU-only。但**需要卡的只有两项**——exp 家族的实测 ULP
  分布，和跑 dump kernel 取真实 ŵ 与逐节点中间量。FMA 收缩与 FTZ 看 PTX 即可，不需要卡。
- 用 **CUDA C++ 不用 Triton**（Triton 抽掉了归约 schedule，且 `tl.exp` 走 `ex2.approx`）。
- dump 必须是**原始 uint32 位模式**，转十进制文本就丢掉了跨边界的逐位性。
- 租来的共享实例上，**数值测量可靠、性能测量不可靠**。任何 perf–error 结论要独占裸金属
  加锁频才算证据。
- artifact schema 需要新的硬件 provenance 块（GPU 型号、compute capability、驱动、
  toolkit 版本、nvcc flags）；判据是换一台租来的机器重跑能否得到同一答案。

## 冻结前仍未决

见合同 §10。其中会改变结论解读的两条：叶块 ℓ 本身如何计算并是否进入合同；
primary metric（v2 的 normalized regret 依赖大候选集，此处不适用）。
研究假设与效应量门槛**要在 pilot 之后再冻结**，现在不预设。

## 继续工作时

- 找代码：[Softmax 实验索引](topics/softmax/experiments/README.md)。
- 检查回归：`python tools/run_tests.py`。
- 改结构前读：[维护指南](docs/maintenance.md)。
- 实验运行前读对应 artifact README；不要批量执行 one-shot runners。
- 当前测试中的数值可移植性容差不是研究 policy，也不是新实验结果。
