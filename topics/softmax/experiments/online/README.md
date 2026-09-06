# Online normalizer 分区

本阶段已于 2026-09-07 结题归档，见[结题说明](../../notes/online_research_closeout_2026-09-07.md)。
以下为实现与实验的历史索引，表中未完成项不再是当前待办。

新研究代码，实现 [算术合同](../../notes/online_normalizer_contract.md) 的 $(m,\ell)$ 部分，
并扩展到[标量 V 输出诊断](../../notes/online_output_gate_v1.md)。
与 [rewrite/](../rewrite/) 的区别：那里是闭卷复现，不产 artifact；这里是新阶段的研究核心。

按[实现学习协议](../../../../framework/implementation_learning_protocol.md)，
标 `USER-WRITTEN CORE` 的函数由用户主写，agent 提供骨架、测试、追问与反例。
当前协作约定：审计确认的缺陷由 agent 直接修复、验证并报告；新增研究核心仍由用户主写。
本轮固定贡献消融核心例外：用户明确授权 agent 完成，后续研究核心再交回用户。

当前[固定贡献消融](../../notes/online_fixed_contribution_ablation_results_v1.md)已完成，
先读[已知性筛查](../../notes/online_prior_work_screening_v1.md)与
[已完成的 CPU 输出诊断](../../notes/online_scalar_output_v1.md)。本批低精度存储对照未显示输出变化，
按预算暂停自动扩展；继续 FP32 工程方向前先明确用途和容差。
平均 D confirmation 和统计校准支线暂停，已有实现/预检保留；实际续接状态见
[NEXT_SESSION](../../../../NEXT_SESSION.md)。标量 V 探针不包含 QK/PV matmul，不能当成完整 attention 结果。

## 顺序与状态

| 步骤 | 模块 | 对照 | 状态 |
| --- | --- | --- | --- |
| 3 | [fp32_signed.py](fp32_signed.py)：有符号 RN-even 与四个受舍入算子 | `rewrite.fp32_oracle`（非负子集）、硬件 float32 | 通过差分测试（2026-09-04）；explain-back 待回填 |
| 1c | [schedules.py](schedules.py)：CUDA 可表达的 schedule 族 | 结构性质自检 | agent 脚手架，已通过 |
| 1b | [merge.py](merge.py)：(m, ℓ) 递推、`MergeDump`、frozen-weight 恒等式 | 独立重算的残差；解析权重负控制 | 通过（2026-09-05） |
| 1a | [fp32_exp.py](fp32_exp.py)：正确舍入的 FP32 exp，即 specified-exp reference | 独立区间参照；ULP 分布 | 已实现并通过测试；学习记录待回填 |
| pilot | [pilot.py](pilot.py)：块族、实数 exp 参照、相对误差指标 | 与 decimal 路线交叉核对；保守性抽样 | 核心已实现，探索性算例可运行 |
| fixed ablation | [fixed_contribution_ablation.py](fixed_contribution_ablation.py)：全局贡献初始化与普通加法适配 | 独立 Decimal、整项舍入与下溢边界 | agent 按用户授权完成；6/6 变异体被杀 |
| fixed runner | [fixed_ablation_runner.py](fixed_ablation_runner.py)：同输入同图普通/online 消融 | 原根/A/D 核对、J/S/R/W、位模式与源码保存 | [64 族结果](../../notes/online_fixed_contribution_ablation_results_v1.md)：普通求和重现长链劣势 |
| output probe | [output_probe.py](output_probe.py)：V 模式、局部分子叶和有符号 O 传播 | 全 1/0/负值控制、实际权重复用、FMA 边界 | 用户核心经审计修正图字段与 FMA 操作数；11/11 通过 |
| output reference | [output_reference.py](output_reference.py)：最终 RN32 除法、真实分子/输出区间 | 负系数端点、四角除法、独立 Decimal、常 V 相关性 | 用户两个区间核心均正确；13/13 通过 |
| output measure | [output_measure.py](output_measure.py)：输出误差/反事实与存储 cast | FP16 对 NumPy、BF16 中点奇偶/下溢、带符号记账 | agent 脚手架；8/8 输出相关变异体被杀 |
| output runner | [output_runner.py](output_runner.py)：保存输入的单轮 V 诊断 | 普通加法对照、原分母重放、逐族记录与完整重放 | 37 条输出专项测试通过；[64 族结果](../../notes/online_scalar_output_v1.md)已保存 |
| runner | [pilot_runner.py](pilot_runner.py)：输入/图/源码快照与精确重放 | 已知确定性算例；损坏记录拒绝 | agent 脚手架，使用方法见下 |
| summary | [pilot_summary.py](pilot_summary.py)：逐格描述性汇总 | 人工可算的合成均值、区间和分组对照 | agent 汇总脚手架，不输出统计 CI |
| attribution | [attribution.py](attribution.py)：带符号分解、归一化与抵消 | 独立参照、已知反例、区间边界 | 用户实现通过单测与变异检查；[学习入口](../../notes/online_attribution_learning.md) |
| attribution runner | [attribution_runner.py](attribution_runner.py)：保存输入的归因与配对重建 | 原输出/A/配对差精确核对；小输入重放 | agent 记录与汇总脚手架 |
| confirmation stats | [confirmation_stats.py](confirmation_stats.py)：配对均值与共同重采样 | 手算合成区间、重复索引、FMA 配对 | 用户核心已通过单测；[确认轮草案](../../notes/online_confirmation_draft_v1.md)未冻结 |
| calibration | [confirmation_calibration.py](confirmation_calibration.py)：已知真值合成总体与单次评分 | 精确总体均值、手算覆盖/误报 | [校准设计](../../notes/online_calibration_design_v1.md)；评分核心已通过单测 |
| calibration runner | [calibration_runner.py](calibration_runner.py)：合成校准预检保存与重放 | 输入、逐次评分、汇总精确重放；损坏/失败拒绝 | agent 脚手架；仅支持 preflight |
| 独立复核 | [tools/verify_online_pilot.py](../../../../tools/verify_online_pilot.py)：pilot bundle 的第二实现 | 硬件 float32、自写 RNE、mpmath 700-bit、seed 重生成 | 通过；见下「独立复核」 |
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

## 可重放 runner

从仓库根目录运行；每次必须选不存在的新输出目录：

```sh
python -m topics.softmax.experiments.online.pilot_runner run --plan topics/softmax/experiments/online/plans/smoke_v1.json --output topics/softmax/experiments/online/runs/smoke_v1
python -m topics.softmax.experiments.online.pilot_runner replay topics/softmax/experiments/online/runs/smoke_v1
python tools/run_tests.py --suite softmax -p test_online_pilot_runner.py -v
```

`run` 先复制配置、预测原文与源码，保存所有实际输入和图，再进行测量。正常完成后
`manifest.json` 才标为 `complete` 并记录文件 SHA-256；中断/失败留下 `running`，
不能充当完整结果，重试须使用新目录。当前不支持断点续跑。

- `inputs.json`：有序 FP32 十六进制位模式、逐块大小、family ID、cell ID 和种子。
- `graphs.json`：实际有序左右孩子表；不支持的 shfl 宽度单列 `not_applicable`。
- `measurements.json`：逐族、逐 schedule、逐 FMA arm 的输出位模式、事件计数及精确分数区间。
- `paired_differences.json`：同族同 FMA arm 的 `A_schedule - A_balanced` 区间与可判定次序。
  正数表示 schedule 更差；此文件是数值区间，不是统计 CI。
- `plan.json`、`prediction.md`、`sources/online/`、`manifest.json`：配置、预测、执行源码快照、
  Git/环境/算术设置和完整性校验。源码快照覆盖此 runner 的 online 本地依赖；依赖仅用标准库。

重放先验证文件 hashes 与当前源码一致，再直接读取保存的输入和图，不调用随机生成器。
源码快照以 `.py.txt` 存储，是归档文本，不是新研究模块。源码不一致时拒绝严格重放；
可在隔离目录恢复所存 `sources/online/`（去掉每个文件的 `.txt` 后缀），并在所记录 Python
环境中使用 `python -m online.pilot_runner replay ...`，不会自动执行记录里的代码。
hash 用于检测文件变化，不是防恶意篡改的签名。

CPU Fraction 合同将零统一为 +0；这里不是硬件 dump 位模式协议。exp 当前只支持默认
correctly-rounded 实现（Decimal 精度 60），不接受其他 exp 名字后悄悄回退。

[预测与范围草案](../../notes/online_pilot_prediction_v1.md) 区分用户原话、agent 预测和已知算例。
[扩展配置](plans/expansion_v1.json) 记录 64 族 pilot 的输入范围，不是确认实验。
当前实际进度与下一步只维护在 [NEXT_SESSION](../../../../NEXT_SESSION.md)。

## 独立复核

`pilot_runner replay` 是用同一份实现重放保存输入；下面这条是**第二实现**，不 import
`online` 任何模块：

```sh
python tools/verify_online_pilot.py topics/softmax/experiments/online/runs/expansion_v1 --summary topics/softmax/experiments/online/runs/expansion_v1_summary
```

它另写 binary32 的 RN-even（硬件 `numpy.float32` 与精确有理两路互校）、用 mpmath
区间在 700 bits 上认证每个 exp 与 real-exp 参照、并从 seed 重新生成叶 maxima，
从而对 `round_to_fp32`、`correctly_rounded_exp`、`merge_reduce`、`denominator_interval`、
`relative_error`、`schedules` 与 `uniform_spread` 各走一条独立路径。
需要 mpmath（仅此工具用，测试套件与实验不依赖）。

[expansion_v1 复核记录](runs/expansion_v1_independent_audit.json)：64 族从 seed 重生成、
256 个根位模式与舍入事件计数一致、256 个 A 区间包住 700-bit 严格区间、128 条配对差与
次序一致、260736 次硬件/精确 FP32 运算零分歧、85769 个 exp 参数逐个认证，
另交叉核对汇总的 16 格与 12 条相邻变化。

篡改对照已验证工具非空过：改根位模式、改 seed、把 A 区间收窄到真值以内、把 balanced
换成别的配对、翻转一个 `error_order`、改一个舍入事件计数——六项全部被拒绝。

这只证明交付代码算的就是合同所写、且保存数字由保存输入唯一推出；合同本身若有错，
这条路径会同样忠实地复现出来。

## 扩展轮的描述性汇总

```sh
python -m topics.softmax.experiments.online.pilot_summary topics/softmax/experiments/online/runs/expansion_v1 --output topics/softmax/experiments/online/runs/expansion_v1_summary --plot
```

该命令先严格重放完整 bundle，再在新目录输出 `summary.json`、可读表格和汇总源码快照。
范围为 uniform-spread 输入上的 chain/balanced；按 spread、块数与 FMA arm 分格，保存
各格平均 A、平均 D、逐族 D 的数值符号计数及相邻块数的 mean D 变化。
所有聚合先用精确分数，JSON 端点向外舍入到 50 位十进制有效数字，表格只显示近似中点。
不同块数使用不同种子；相邻变化是两个样本均值之差，不是同一输入的逐族配对差。
两种 FMA arm 共用输入，不能把它们当成独立的两倍样本量。
这里的区间只包住这些观测值的数值参照误差，不是总体置信区间，也不检验统计独立。
可选 `--plot` 用 Matplotlib 显示逐族 D 与格均值，不画统计误差棒；版本记入汇总 provenance。

## 在保存输入上做归因

```sh
python -m topics.softmax.experiments.online.attribution_runner run topics/softmax/experiments/online/runs/expansion_v1 topics/softmax/experiments/online/runs/attribution_v1 --prediction topics/softmax/notes/online_attribution_plan_v1.md
python -m topics.softmax.experiments.online.attribution_runner replay topics/softmax/experiments/online/runs/expansion_v1 topics/softmax/experiments/online/runs/attribution_v1
```

运行先检查父 bundle 的完整性和旧执行源码，再在同一遍重建中核对旧计算根、A 区间与
配对差，调用用户的归因核心。每八族打印进度。输出必须是新目录；失败不标 complete。
原始分数以十六进制分子/分母存储，避免巨整数十进制字符串长度限制，仍保持精确值。
`attributions.json` 保存分解、归一化区间、预算、抵消量与节点残差符号计数；
逐节点数组按需由父输入/图和源码重建。`paired_attributions.json` 保存 D、D_R 和 D−D_R。
`summary.json` 和 README 按 cell/schedule/FMA 分格，不跨格混池；无统计 CI。
完整性清单包含生成的 README，运行后解释应另写笔记，不能修改已存 bundle。

归因的两个新增核心函数在 `attribution.py`，runner 只做记录、已有结果核对及描述性汇总。
运行前范围见[归因记录](../../notes/online_attribution_plan_v1.md)。

## 确认轮统计核心的学习入口

本支线暂停。以下为已完成工具的使用说明，不是当前下一步。

[均值到下界的学习说明](../../notes/online_confirmation_learning.md) 衔接已完成的均值计算
与下一步分位数工具。当前均值实现可用 `python tools/audit_online_confirmation_mutants.py`
做独立于研究测量的变异审计。

[confirmation_stats.py](confirmation_stats.py) 中用户已实现 `paired_mean`、`bootstrap_means`、
`linear_quantile` 与 `quantile_interval`。解释记录可补当前理解；已运行测试不能倒填事前预测。
输入已是同族的 chain−balanced 误差差区间；
两个 FMA arm 使用共同的族索引，重复索引必须重复计入。区间端点保留精确分数。

```sh
python tools/run_tests.py --suite softmax -p test_online_confirmation_stats.py -v
python tools/run_tests.py --suite softmax -p test_online_confirmation_quantiles.py -v
python tools/run_tests.py --suite softmax -p test_online_confirmation_calibration.py -v
```

此模块产生均值、重采样均值分布及分位数的数值外包，不给统计 CI 或通过判定。
标量和区间分位数均已通过手算与端点测试；统计方法校准另行设计。
`confirmation_calibration.py` 的 `assess_trial` 对一次外层实验判断覆盖、拒绝零假设
与误报，已通过手算测试。其合成总体与候选拼装由 agent 提供，核心评分由用户写。
方法与样本量待按[确认轮草案](../../notes/online_confirmation_draft_v1.md)确定，
实际进度见 [NEXT_SESSION](../../../../NEXT_SESSION.md)。

## 合成校准预检 runner

本支线暂停，以下命令用于复核已有预检，不触发新的研究方向。

```sh
python tools/audit_online_confirmation_mutants.py --calibration
python tools/run_tests.py --suite softmax -p test_online_calibration_runner.py -v
python -m topics.softmax.experiments.online.calibration_runner replay topics/softmax/experiments/online/runs/calibration_preflight_v1
```

首次运行使用 `calibration_runner run --plan <配置路径> --output <新的输出目录>`。
[预检配置](plans/calibration_preflight_v1.json) 已执行，现有输出目录不能覆盖。
runner 在测量前保存协议、源码、总体和全部外层输入；逐次保存下界、评分、常数样本与
负值观测计数。失败标记 `failed`，不把半批结果当完整实验。
重放要求文件 hashes、当前源码及 Python 版本匹配，从保存的外层输入计算，
只根据已存种子重建内层抽样索引并核对索引 hash。源码 `.py.txt` 是归档文本，
如需恢复旧版本，应在隔离目录恢复源码及记录环境，不自动执行 bundle 中的代码。
此 runner 只支持工程预检，输出整数计数而不输出校准覆盖率或方法筛选结论。
