# Next Session

整理日期：2026-09-06。冻结证据仍依据 `ad1fe87` 及其版本化 artifacts。
当前 online pilot 已在 `bf632c3` 实现；首轮解读的审计修正见合同 §12。
这是唯一的当前续接入口；历史推导与旧待办已归入
[2026-08-12 handoff](docs/history/2026-08-12-softmax-handoff.md)。

**2026-09-06 最新方向调整：先做普通固定贡献加法 vs online 递推的最小消融。**
当前入口见[固定贡献消融](topics/softmax/notes/online_fixed_contribution_ablation_v1.md)，
先判断已有链劣势是否主要被普通求和重现；不能只凭 W 未增长就排除 online 机制，
因为 R 也含重标定乘法/FMA 舍入。confirmation 与统计方法校准继续暂停。
[输出影响与分段合并](topics/softmax/notes/online_output_gate_v1.md) 保留为后续选项，
当前不扩展 O、不开发 selector 或分段优化。保留全部既有代码和记录。
用户授权本轮核心由 agent 实现，现已完成固定贡献准备、普通加法图适配与可重放 runner。
[64 族消融结果](topics/softmax/notes/online_fixed_contribution_ablation_results_v1.md)：
普通 FP32 加法重现了随块数增长的链劣势；2048/spread25 的 mean D 为普通 1.0315e-6、
online separate 1.0935e-6、FMA 1.1116e-6。逐族 H 仍有双向差异，不能宣称算法等价。
旧 online 256 个根/A 和 128 个 D 已精确核对。下一块研究核心再交回用户：
构造反复重标定的受控输入，保留同输入/同图普通求和基线；不继续平均 D confirmation。

## 当前落点

活跃主题是 **普通 reduction 与 online 重标定机制的区分**；输出影响和工程改进随后再判。
以下四项是此前 reduction-only 线的既有证据，不是当前待办：

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

## 当前阶段：online 与固定贡献求和的机制消融（2026-09-06 调整）

候选一（带重标定的 reduction tree）已选定并开工，候选二（sparse exactness correction）
留在架上未启动。**CPU 模拟与 pilot 指标已实现，已有探索性测量；尚无预注册的确认实验
或冻结的本阶段结果 artifact。**
算术合同见 [online normalizer 合同](topics/softmax/notes/online_normalizer_contract.md)，
代码见 [online 分区](topics/softmax/experiments/online/README.md)。

已实现范围：只有 (m, ℓ)，尚不含 V、O 和 y = O/ℓ 的除法。先完成固定贡献消融，
再决定是否扩展输出；不能将已有分母归因视为已证明输出重要性。主合同 FP32；低精度是
**压力测试与候选设计**，不是真实 kernel 的默认（FA2/FA3 的 (m, ℓ, O) 累加器保持 FP32）。

已确立的三条，都是设计结论不是研究结论：

1. **加权恒等式只在 frozen-weight 形式下成立**：
   `l_root_hat − l_root_frozen = Σ_v W_v (μ_a + μ_b + α_v)`，其中 `W_v = Π ŵ` 是**实际算出的**
   重标定因子之积，不是解析的 `e^{m_v − m_root}`。已逐位验证（chain/balanced × 三档 spread ×
   FMA 开关），解析权重版本作为负控制按预期失败。
2. **frozen-weight 恒等式可用 Fraction 逐位精确核对**，因为它把 ŵ 冻成数据，
   exp 的实现误差与 Δm 的舍入都被吸收进去。specified-exp 的输出也是 FP32 有理数，
   可精确比较，但确定正确舍入需要高精度计算；real-exp 的实数参照使用严格区间。
   不要把这三种 reference 混用，详见合同 §3。
3. **旧 oracle 的非负合同不够用**：m 是有符号 logit，Δm 是相减。
   [fp32_signed.py](topics/softmax/experiments/online/fp32_signed.py) 已通过差分测试
   （对 rewrite oracle 非负子集、对硬件 float32、Sterbenz、两个方向的 tie 与溢出带）。

不能把旧校准的 192 个标签重新当作本阶段的确认数据。未知 black-box graph 与端到端
Softmax 扩展仍需单独定义范围。

## 下一步顺序

| | 内容 | 依赖 | 状态 |
| --- | --- | --- | --- |
| 3 | oracle 有符号合同扩展 | — | **已完成**（`02ec506`） |
| 1 | CUDA 可表达 schedule 族的精确模拟与 merge | 3 | **已完成 CPU 模拟与测试** |
| pilot | 实数参照、相对误差区间与探索性扫描 | 1 | **64 族扩展 pilot 完成并精确重放**；仍为探索性 |
| fixed ablation | 固定全局贡献的普通加法 vs online，分开初始化误差 | pilot | **当前下一步**；固定贡献核心由用户主写 |
| output gate | 同步 O 传播、实数参照与最终 y 误差 | fixed ablation | 后续候选，暂不实现 |
| segmented | 组内 chain/组间 balanced 与固定方案比较 | output gate | 暂缓；旧 split_k 的顶层是 chain |
| confirmation/calibration | 平均 D 确认及统计方法校准 | — | **用户决定暂停**；既有工具与预检保留 |
| 2a | dump 传输层：位模式格式、硬件 provenance 块、CPU 侧核对 harness | — | 可并行 |
| 2b | 硬件测量的 schedule 与 exp 变体范围 | 1 | 待固定 |

首轮解读已修正：32 块也能由严格区间分辨 schedule 误差；输出 ULP 不是参照的
测量分辨率。保留小块数对照，把较大块数作为探索轴，不将 512 设为必需下限。
误差比值不是自动成立的下界。随机扫描须保存输入位模式、块大小、种子、实际 schedule、
FMA/exp 设置与逐族结果；不支持的 shfl 宽度记为不适用，不能以 balanced 冒充。
已有试跑只能补事后解释；下一轮预测在运行前记录。

2026-09-05 续接：用户对下一轮块数趋势的事前判断为“没有稳定方向吧”，随后补充
“就是很有可能是解耦的，无关联”。原话与 agent 预测分开保存在
[预测记录](topics/softmax/notes/online_pilot_prediction_v1.md)。
[runner](topics/softmax/experiments/online/pilot_runner.py) 保存实际输入位模式、块大小、种子、
完整调度、FMA/exp 设置、逐族精确结果、配对差以及源码/预测快照，可直接重放所存输入。
验收使用已知 32 块确定性算例，保存于
[smoke_v1](topics/softmax/experiments/online/runs/smoke_v1/README.md)，不作新的随机扫描结论。
按[扩展配置](topics/softmax/experiments/online/plans/expansion_v1.json) 完成 4 档块数 × 2 档 spread ×
每格 8 族、chain/balanced × 两种 FMA arm：64 族、256 次测量、128 条配对差全部精确重放。
[结果与预测对照](topics/softmax/notes/online_pilot_expansion_v1.md) 记录：四个 spread/FMA 层的
mean D 均随 n 单调增大，12 个相邻样本均值差均严格为正；spread=25 的 32 块对照中
chain 平均更准，到 2048 块转为更差。它挑战“无关联”的初步预测，不是总体确认。
[逐格表格与图](topics/softmax/experiments/online/runs/expansion_v1_summary/README.md) 已生成，
统计 CI 尚未实现。原运行前预测保留，不用结果回填。

事后机制归因已复用保存输入完成，区分 frozen-weight 带权舍入项和冻结权重相对
real-exp 参照的差异，观察符号抵消。两种树都有 n−1 次 merge，
不可只用总事件数或树深度给最终误差增长定因。常值叶块本轮确实没有块内 reduction 误差。
用户明确要求下一块核心仍由自己实现。已建立
[归因学习入口](topics/softmax/notes/online_attribution_learning.md) 和
[attribution.py](topics/softmax/experiments/online/attribution.py)，用户已完成 `decompose`
与 `cancellation_savings`。归因前预测原话：“merge 的带权舍入残差逐渐累积”。
16 项测试全部通过，无 skip；9 个设计的变异体全被杀死，
[审计记录](topics/softmax/experiments/online/runs/attribution_audit_v1.json) 含执行源码/测试 hashes。
只移除一条无用 sklearn 导入，数值函数保留用户实现。用户随后补了正负残差与抵消的解释。
[归因 v1 结果](topics/softmax/notes/online_attribution_v1.md) 已保存：256 次分解与旧根/A 区间、
128 条配对差精确核对，独立保存数据审计通过 336 个汇总区间及 128 组 FMA 权重不变性。
四个 spread/FMA 层内，chain 的归一化净舍入量随 n 增长，W 没有对应增长；大 n 的
平均 schedule 差距主要可由 R 的差异重建，支持用户分量预测。内部抵消仍很大，
不能把它解释为节点残差全同号，也不是低成本预测器或总体确认。
2026-09-06 用户选定确认轮首要问题：在固定输入分布和较大块数下，chain 的平均
相对误差高于 balanced；归因作为次要分析。[确认轮草案](topics/softmax/notes/online_confirmation_draft_v1.md)
提出 n=2048、spread=8/25 × 两种 FMA 的四条件共同命题，**这些具体参数尚是 agent 提案**。
样本量、统计区间方法及多条件判据未冻结，尚未生成新确认输入。
外部独立审计已落成仓库工具
[verify_online_pilot.py](tools/verify_online_pilot.py)，不 import `online` 任何模块：
硬件 float32 与自写精确有理 RN-even 互校、mpmath 700-bit 区间认证每个 exp 与 real-exp
参照、叶 maxima 从 seed 重生成。对 expansion_v1 通过并保存
[复核记录](topics/softmax/experiments/online/runs/expansion_v1_independent_audit.json)：
64 族重生成、256 个根位模式与舍入事件计数一致、256 个 A 区间包住严格区间、128 条配对差
与次序一致、260736 次硬件/精确运算零分歧、85769 个 exp 参数认证，另核对汇总 16 格与
12 条相邻变化。六项篡改对照（改根、改 seed、收窄 A 区间、换 balanced 配对、翻转
error_order、改事件计数）全部被拒绝，工具非空过。mpmath 已记入 requirements 的
audit-only 段，测试套件不依赖。这只证明交付代码算的是合同所写，合同本身有错不会被发现。
已直接从保存 JSON 复核其中三个设计事实：chain 根的 separate/FMA 在
57/64 个 family 相同，D 在 39/64 个 family 相同；spread=25、n=32 每个 arm 均为 4 个
chain 更准和 4 个平局。确认草案已据此强调 FMA 的重复测量属性，并区分合取检验与四格
同时置信界；2^-128 网格安全但会主导统计模块的数值外包宽度。
[confirmation_stats.py](topics/softmax/experiments/online/confirmation_stats.py) 已建立用户核心骨架：
先回填 Explain-back/预测，再写 `paired_mean` 和 `bootstrap_means`；后者只生成共同索引下的
均值分布，不产生 CI。用户已实现 `paired_mean`，索引、重复计数、缩放与区间返回经审查正确；
用户随后实现 `bootstrap_means` 的循环与结果收集；agent 删除非法的函数调用赋值，
将用户均值计算原样提取到 `_mean_from_grid` 复用，避免每次重采样重复量化输入。
当前专项测试 8 条全部通过，无 skip；[9 个变异体审计](topics/softmax/experiments/online/runs/confirmation_stats_audit_v1.json)
全部由测试断言杀死，无测试错误或 skip。可用 `python tools/audit_online_confirmation_mutants.py`
复核；源码、测试与审计脚本 hashes 已记录。学习记录仍待用户回填，不能补成事前预测。
下一入口是[统计学习说明](topics/softmax/notes/online_confirmation_learning.md)：先理解分位数
及单侧下界。用户已正确手算五个值在 p=3/8 时 h=1.5、插值结果为 2。
用户已完成 `linear_quantile` 的排序、精确位置、整数边界及加权插值；审查正确，
agent 仅移除已过期的 TODO。用户随后正确手算区间中位数外包 [1,4]，并完成
`quantile_interval`：拆出上下界两列，分别调用 `linear_quantile`。实现经审查正确，
agent 仅移除过期 TODO。均值与分位数专项合计 19 条通过，无 skip；
[v2 变异审计](topics/softmax/experiments/online/runs/confirmation_stats_audit_v2.json) 覆盖均值和
分位数的 18 个设计变异体，全部由断言杀死，无测试错误。v1 保留为旧均值审计记录。
数值工具已接通，并用真均值已知的合成分布完成了工程预检；
这条统计方法支线现已暂停，正式覆盖校准及确认方法/判据均未冻结。
已落地[校准设计 v1](topics/softmax/notes/online_calibration_design_v1.md) 与
[confirmation_calibration.py](topics/softmax/experiments/online/confirmation_calibration.py)：
7 个真均值已知的联合离散总体、可复现抽样和既有 percentile 工具拼装已给。
用户已实现 `assess_trial` 的覆盖/拒绝/误报与合取条件；判断逻辑正确，agent 修复
循环中布尔赋值覆盖前一 arm、随后 `tuple(bool)` 报错的问题，改为列表 append 收集。
用户事前预测原话：
“后者问题大一点？不太清楚”，指稀有负尾比对称零均值分布更可能失准。
评分核心已通过 [8 个变异体审计](topics/softmax/experiments/online/runs/calibration_score_audit_v1.json)，
全部由断言杀死。[calibration_runner.py](topics/softmax/experiments/online/calibration_runner.py)
已建并通过 7 条测试；加原确认统计相关 29 条均通过，无 skip，维护测试 9 条通过。
按已保存预检配置完成 280 次外层实验、全部精确重放，测量约 4.316 秒。
[预检说明](topics/softmax/notes/online_calibration_preflight_v1.md) 记录稀有负尾每批 32 族时
20 次中 9 次误报，逐条对应没抽到负值的常数正样本；另做了 280 条独立评分核对。
解析反例表明该总体下误报概率至少 (31/32)^32≈36.2%，不是用 9/20 估计得出的保证。
这里的 32 是每批独立族数，不是每个 softmax 输入的块数；不外推为真实 D 分布结论。
原计划先固定方法筛选规则再正式校准；这项待办随本次方向调整暂停。
当前 runner 仅支持 preflight；未选定新方法，未运行 softmax 确认数据。
原均值/分位数变异审计限定到其两份测试文件，新增校准骨架的 skip 不会混入旧核心审计；
调整测试选择后 18 个变异体仍全部被断言杀死。
输入验证、向外网格量化和重采样索引由 agent 提供。上述统计工具保留复用，
不再作为输出门槛实验的前置条件。原样本上的扩展保持事后探索标注，2a 随输出状态扩展。

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

见合同 §10–§12。首轮已采用精确初始化的常值块，指标为对 real-exp 参照的相对误差，
以同族 schedule 的配对差为 primary；一般块内计算仍待扩展。
当前先明确固定贡献的初始化合同、J/S 与 R/W 的对照和消融判读范围，
详见[固定贡献消融](topics/softmax/notes/online_fixed_contribution_ablation_v1.md)。
输出合同、V/输入范围、工程误差容差和固定分段成本仍是后续门槛。
平均 D 确认的分布/样本量/统计区间问题暂停，不阻塞当前机制探索；
后续工程结论仍须明确适用范围及验证规则。

## 继续工作时

- 找代码：[Softmax 实验索引](topics/softmax/experiments/README.md)。
- 检查回归：`python tools/run_tests.py`。
- 改结构前读：[维护指南](docs/maintenance.md)。
- 实验运行前读对应 artifact README；不要批量执行 one-shot runners。
- 当前测试中的数值可移植性容差不是研究 policy，也不是新实验结果。
