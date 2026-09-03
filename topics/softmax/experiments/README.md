# Softmax 实验代码索引

这里按用途组织实验脚本与共享分析包，原有脚本入口路径保持稳定。
测试已迁往 [../tests/](../tests/)，本目录只保留旧完整发现命令的兼容入口。
迁移边界与后续拆分计划见 [维护指南](../../../docs/maintenance.md)。
第一批 coherence 计算已从独立脚本抽到共享包，不再需要为每个分析重复 replay。

## 共享分析主干

已有输入和树时，一次 `replay(values, graph)` 得到轨迹，然后从同一个
`CoherenceAnalysis` 读取 `ac`、`structure` 和 `history`。
用法与边界见 [共享分析说明](reduction_analysis/README.md)。

| 模块 | 用途 |
| --- | --- |
| [public API](reduction_analysis/__init__.py) | 共享分析入口 |
| [trace](reduction_analysis/trace.py) | 包装已有 oracle 结果，不引入第二套舍入实现 |
| [topology](reduction_analysis/topology.py) | 共用 parent/depth 与祖先查询 |
| [coherence](reduction_analysis/coherence.py) | 惰性 A/C、结构分区与 ancestor/history |

## 先读这三个入口

- [NEXT_SESSION](../../../NEXT_SESSION.md)：最新状态与尚待冻结的下一研究入口。
- [结果索引](results/README.md)：确认、校准、负结果和 provisional evidence 的边界。
- [早期实验笔记](../notes/early_experiments.md)：量化、summation、failure triage、P0–P5 的推导与历史运行说明。

本目录的 [PREDICTOR_VALIDATION_PROTOCOL](PREDICTOR_VALIDATION_PROTOCOL.md) 和
[PREDICTOR_RESEARCH_DIRECTION](PREDICTOR_RESEARCH_DIRECTION.md) 是已有阶段记录，
本次不修改它们；具体阶段的 preregistration 以各 results 子目录为准。

## 日常检查（仓库根目录）

```sh
python tools/run_tests.py --suite softmax
python tools/run_tests.py --suite softmax -p "test_predictor_fixed_k8_beam_inference.py" -v
```

不要遍历执行所有 `*.py`。某些脚本会写结果，one-shot stages 还要求 clean committed
checkout；单元测试通过并不授权重新生成 artifacts。完整旧测试命令仍兼容：
`python -m unittest discover -s topics/softmax/experiments -p "test_*.py"`。

## 基础实现与 oracle

这些模块承载原有数值机制，保持原路径与 source bytes；新代码引用前先阅读其输入合同。

| 模块 | 用途 |
| --- | --- |
| [fp32_shift_resolution ](fp32_shift_resolution.py) | 输入量化边界 |
| [rewrite_fp32_shift_resolution ](rewrite_fp32_shift_resolution.py) | 学习者闭卷重写 |
| [fp32_summation_stress ](fp32_summation_stress.py) | 顺序 / pairwise / Kahan 求和 |
| [fp32_softmax_summation ](fp32_softmax_summation.py) | normalization 端到端误差 |
| [summation_graph_predictor ](summation_graph_predictor.py) | exact FP32 graph oracle / label generator |
| [softmax_failure_triage ](softmax_failure_triage.py) | case、observation、summary 和 policy 数据模型 |

## 输入、图与基线特征

研究用构造器和基线；显式 seed schedule、stored input identity 与 tree contract 是版本边界。

| 模块 | 用途 |
| --- | --- |
| [calibration_inputs ](predictor_calibration_inputs.py) | 受控 stored-FP32 输入 |
| [tree_generator ](predictor_tree_generator.py) | 随机二叉树 |
| [structural_features ](predictor_structural_features.py) | sibling scale mismatch |
| [dominant_exposure ](predictor_dominant_exposure.py) | dominant-leaf exposure |
| [second_moment_baseline ](predictor_second_moment_baseline.py) | partial-sum second-moment baseline |

## 冻结阶段与当前阅读入口

这里的 runner 用于 evidence generation，不是日常 smoke 命令。先读对应 [结果目录](results/README.md)；one-shot guard 不应被绕过。

| 模块 | 用途 |
| --- | --- |
| [online_risk_certificate_calibration ](predictor_online_risk_certificate_calibration.py) | 最新完成：online certificate 校准 |
| [wide_range_offline_tree_reuse_v1 ](predictor_wide_range_offline_tree_reuse_v1.py) | offline reuse 确认 / no-go |
| [wide_range_fixed_k8_beam_v2_heldout ](predictor_wide_range_fixed_k8_beam_v2_heldout.py) | fixed-K8/B3 v2 pooled 确认 |
| [wide_range_energy_beam_v1_heldout ](predictor_wide_range_energy_beam_v1_heldout.py) | energy beam v1 负结果 |
| [depth_margin_topology_challenge_v1_runner ](depth_margin_topology_challenge_v1_runner.py) | depth-margin 反例 |
| [nonuniform_graph_predictor_v1_runner ](nonuniform_graph_predictor_v1_runner.py) | P4 single-case |
| [summation_graph_predictor_validation ](summation_graph_predictor_validation.py) | 早期 accepted selector / provisional batch |
| [softmax_failure_triage_runner ](softmax_failure_triage_runner.py) | suite v3；结果只能定向输出到新 scratch 目录 |

## 推理原型与成本

score-only 不等于生产低成本；成本结论只能用于记录的 Python 环境。

| 模块 | 用途 |
| --- | --- |
| [fixed_k8_beam_inference ](predictor_fixed_k8_beam_inference.py) | oracle-free fixed-K8/B3 selector |
| [fixed_k8_beam_inference_benchmark ](predictor_fixed_k8_beam_inference_benchmark.py) | score-only fidelity / timing |
| [resource_benchmark ](predictor_resource_benchmark.py) | 基线资源诊断 |
| [candidate_resource_benchmark ](predictor_candidate_resource_benchmark.py) | 候选资源对照 |

## 历史校准：误差机理

以下是保留的 calibration / diagnostic code，不代表当前待办，也不自动构成新的 held-out evidence。

| 模块 | 用途 |
| --- | --- |
| [ranking_smoke ](predictor_ranking_smoke.py) | 早期 ranking smoke / 统计 helper |
| [target_variation_diagnostic ](predictor_target_variation_diagnostic.py) | target variation |
| [wide_range_theory_diagnostic ](predictor_wide_range_theory_diagnostic.py) | theory baseline 对照 |
| [wide_range_stagnation_diagnostic ](predictor_wide_range_stagnation_diagnostic.py) | stagnation |
| [wide_range_oracle_mechanism_diagnostic ](predictor_wide_range_oracle_mechanism_diagnostic.py) | oracle 机理分解 |
| [wide_range_shadow_phase_diagnostic ](predictor_wide_range_shadow_phase_diagnostic.py) | shadow phase |
| [wide_range_history_scale_diagnostic ](predictor_wide_range_history_scale_diagnostic.py) | history scale |
| [wide_range_history_transition_diagnostic ](predictor_wide_range_history_transition_diagnostic.py) | history transition |
| [wide_range_history_chain_correlation ](predictor_wide_range_history_chain_correlation.py) | history chain correlation |
| [wide_range_conditional_history_distribution ](predictor_wide_range_conditional_history_distribution.py) | conditional history |
| [wide_range_signed_history_distribution ](predictor_wide_range_signed_history_distribution.py) | signed history |
| [wide_range_ancestor_history_decomposition ](predictor_wide_range_ancestor_history_decomposition.py) | ancestor history；共享分析的兼容入口 |
| [wide_range_ac_decomposition ](predictor_wide_range_ac_decomposition.py) | A/C；共享分析的兼容入口 |
| [wide_range_coherence_structure ](predictor_wide_range_coherence_structure.py) | 结构分解；共享分析的兼容入口 |
| [wide_range_coherence_sparsity ](predictor_wide_range_coherence_sparsity.py) | coherence sparsity |

## 历史校准：局部模型与消融

这些脚本相互导入部分 private helpers；逻辑分组不表示依赖已经解耦。

| 模块 | 用途 |
| --- | --- |
| [boundary_aware_score_calibration ](predictor_boundary_aware_score_calibration.py) | boundary-aware score |
| [sparse_first_order_phase_score_calibration ](predictor_sparse_first_order_phase_score_calibration.py) | sparse first-order phase |
| [recursive_gaussian_moment_score_calibration ](predictor_recursive_gaussian_moment_score_calibration.py) | recursive Gaussian moment |
| [gaussian_ancestor_coherence_calibration ](predictor_gaussian_ancestor_coherence_calibration.py) | Gaussian ancestor coherence |
| [discrete_ancestor_phase_score_calibration ](predictor_discrete_ancestor_phase_score_calibration.py) | discrete ancestor phase |
| [reliability_weighted_coherence_calibration ](predictor_reliability_weighted_coherence_calibration.py) | reliability weighting |
| [first_order_history_failure_diagnostic ](predictor_first_order_history_failure_diagnostic.py) | first-order failure |
| [ancestor_transition_predictability_diagnostic ](predictor_ancestor_transition_predictability_diagnostic.py) | ancestor transition predictability |
| [signed_cell_shift_predictability_diagnostic ](predictor_signed_cell_shift_predictability_diagnostic.py) | signed-cell shift |
| [shadow_trajectory_failure_diagnostic ](predictor_shadow_trajectory_failure_diagnostic.py) | shadow trajectory failure |
| [shadow_sparse_repair_ablation ](predictor_shadow_sparse_repair_ablation.py) | sparse repair |
| [shadow_high_layer_mistake_ablation ](predictor_shadow_high_layer_mistake_ablation.py) | high-layer mistakes |

## 历史校准：选择预算与 beam

预算搜索、阈值校准与复杂度诊断；不得把已看过的校准结果重新标记为确认。

| 模块 | 用途 |
| --- | --- |
| [two_stage_cheap_score_calibration ](predictor_two_stage_cheap_score_calibration.py) | two-stage score |
| [ancestor_cell_beam_score_calibration ](predictor_ancestor_cell_beam_score_calibration.py) | ancestor-cell beam |
| [ancestor_cell_beam_cost_diagnostic ](predictor_ancestor_cell_beam_cost_diagnostic.py) | beam cost |
| [energy_mass_selection_calibration ](predictor_energy_mass_selection_calibration.py) | energy-mass selection |
| [width_aware_cascade_calibration ](predictor_width_aware_cascade_calibration.py) | width-aware cascade |
| [q_beam_shortlist_cascade_calibration ](predictor_q_beam_shortlist_cascade_calibration.py) | Q/beam shortlist |
| [ulp_energy_convergence_diagnostic ](predictor_ulp_energy_convergence_diagnostic.py) | ULP energy convergence |
| [ulp_energy_cost_pareto_diagnostic ](predictor_ulp_energy_cost_pareto_diagnostic.py) | cost / quality Pareto |

## 闭卷重写（进行中）

独立复现分区，用户主写核心，旧模块只作测试参考；规则与顺序见 [rewrite/README.md](rewrite/README.md)。

| 模块 | 用途 |
| --- | --- |
| [rewrite package](rewrite/__init__.py) | 重写分区入口 |
| [rewrite/fp32_oracle](rewrite/fp32_oracle.py) | RN-even 舍入与整树精确求值（已通过差分测试） |
| [rewrite/coherence](rewrite/coherence.py) | A/C 分解与 C 主导统计（已通过差分与复现测试） |
| [rewrite/macro_score](rewrite/macro_score.py) | Q_8/12 分数与 shortlist（已通过冻结复现） |
| [rewrite/regret_stats](rewrite/regret_stats.py) | regret 与分层 bootstrap，重算 v2 headline（已通过冻结复现） |
| [rewrite/generators](rewrite/generators.py) | 树与受控输入生成器，seed schedule 为冻结边界（已通过差分与冻结复现） |

## 新增代码的规则

新增 module 时同时更新本索引并添加测试。维护测试会检查 source coverage，
避免再出现大量没有入口说明的脚本。新功能不要继续从历史 runner 复制同名 helpers；
先做语义比较和 characterization tests，再进行有版本边界的公共模块提取。
