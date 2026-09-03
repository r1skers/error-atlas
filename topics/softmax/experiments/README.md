# Softmax experiment code index

Experiment scripts and the shared analysis package, organized by role; original
script entry paths stay stable. Tests have moved to [../tests/](../tests/); this
directory keeps only a compatibility entry for the old full-discovery command.
Migration boundaries and the deferred split plan are in the
[maintenance guide](../../../docs/maintenance.md). The first batch of coherence
computations has been extracted into a shared package, so each analysis no longer
replays the oracle separately.

## Shared analysis backbone

Given inputs and a tree, one `replay(values, graph)` yields a trace; then read `ac`,
`structure`, and `history` from the same `CoherenceAnalysis`. Usage and boundaries are
in the [shared analysis notes](reduction_analysis/README.md).

| Module | Role |
| --- | --- |
| [public API](reduction_analysis/__init__.py) | Shared analysis entry |
| [trace](reduction_analysis/trace.py) | Wraps existing oracle results; introduces no second rounding implementation |
| [topology](reduction_analysis/topology.py) | Shared parent/depth and ancestor queries |
| [coherence](reduction_analysis/coherence.py) | Lazy A/C, structural partition, and ancestor/history |

## Read these three entries first

- [NEXT_SESSION](../../../NEXT_SESSION.md) — current status and the next research entry to freeze.
- [Results index](results/README.md) — boundaries of confirmed, calibration, negative, and provisional evidence.
- [Early experiment notes](../notes/early_experiments.md) — quantization, summation, failure triage, and the P0–P5 derivation with historical run notes.

The [PREDICTOR_VALIDATION_PROTOCOL](PREDICTOR_VALIDATION_PROTOCOL.md) and
[PREDICTOR_RESEARCH_DIRECTION](PREDICTOR_RESEARCH_DIRECTION.md) in this directory are
existing stage records, unchanged here; per-stage preregistration lives in each results
subdirectory.

## Routine checks (repository root)

```sh
python tools/run_tests.py --suite softmax
python tools/run_tests.py --suite softmax -p "test_predictor_fixed_k8_beam_inference.py" -v
```

Do not execute every `*.py`. Some scripts write results, and one-shot stages also require
a clean committed checkout; passing unit tests does not authorize regenerating artifacts.
The full legacy test command still works:
`python -m unittest discover -s topics/softmax/experiments -p "test_*.py"`.

## Core implementation and oracle

These modules carry the original numerical machinery and keep their original paths and
source bytes; read their input contracts before referencing them.

| Module | Role |
| --- | --- |
| [fp32_shift_resolution](fp32_shift_resolution.py) | Input quantization boundary |
| [rewrite_fp32_shift_resolution](rewrite_fp32_shift_resolution.py) | Learner closed-book rewrite |
| [fp32_summation_stress](fp32_summation_stress.py) | Sequential / pairwise / Kahan summation |
| [fp32_softmax_summation](fp32_softmax_summation.py) | End-to-end normalization error |
| [summation_graph_predictor](summation_graph_predictor.py) | Exact FP32 graph oracle / label generator |
| [softmax_failure_triage](softmax_failure_triage.py) | Case, observation, summary, and policy data models |

## Inputs, graphs, and baseline features

Study constructors and baselines; the explicit seed schedule, stored input identity, and
tree contract are version boundaries.

| Module | Role |
| --- | --- |
| [calibration_inputs](predictor_calibration_inputs.py) | Controlled stored-FP32 inputs |
| [tree_generator](predictor_tree_generator.py) | Random binary trees |
| [structural_features](predictor_structural_features.py) | Sibling scale mismatch |
| [dominant_exposure](predictor_dominant_exposure.py) | Dominant-leaf exposure |
| [second_moment_baseline](predictor_second_moment_baseline.py) | Partial-sum second-moment baseline |

## Frozen stages and current reading entries

These runners generate evidence; they are not routine smoke commands. Read the matching
[results directory](results/README.md) first; the one-shot guard should not be bypassed.

| Module | Role |
| --- | --- |
| [online_risk_certificate_calibration](predictor_online_risk_certificate_calibration.py) | Latest completed: online certificate calibration |
| [wide_range_offline_tree_reuse_v1](predictor_wide_range_offline_tree_reuse_v1.py) | Offline reuse confirmation / no-go |
| [wide_range_fixed_k8_beam_v2_heldout](predictor_wide_range_fixed_k8_beam_v2_heldout.py) | Fixed-K8/B3 v2 pooled confirmation |
| [wide_range_energy_beam_v1_heldout](predictor_wide_range_energy_beam_v1_heldout.py) | Energy beam v1 negative result |
| [depth_margin_topology_challenge_v1_runner](depth_margin_topology_challenge_v1_runner.py) | Depth-margin counterexample |
| [nonuniform_graph_predictor_v1_runner](nonuniform_graph_predictor_v1_runner.py) | P4 single case |
| [summation_graph_predictor_validation](summation_graph_predictor_validation.py) | Early accepted selector / provisional batch |
| [softmax_failure_triage_runner](softmax_failure_triage_runner.py) | Suite v3; results only to a new scratch directory |

## Inference prototype and cost

Score-only does not mean production-cheap; cost conclusions apply only to the recorded
Python environment.

| Module | Role |
| --- | --- |
| [fixed_k8_beam_inference](predictor_fixed_k8_beam_inference.py) | Oracle-free fixed-K8/B3 selector |
| [fixed_k8_beam_inference_benchmark](predictor_fixed_k8_beam_inference_benchmark.py) | Score-only fidelity / timing |
| [resource_benchmark](predictor_resource_benchmark.py) | Baseline resource diagnostic |
| [candidate_resource_benchmark](predictor_candidate_resource_benchmark.py) | Candidate resource comparison |

## Historical calibration: error mechanism

Retained calibration / diagnostic code below; it is not a current to-do and does not
by itself constitute new held-out evidence.

| Module | Role |
| --- | --- |
| [ranking_smoke](predictor_ranking_smoke.py) | Early ranking smoke / statistics helper |
| [target_variation_diagnostic](predictor_target_variation_diagnostic.py) | Target variation |
| [wide_range_theory_diagnostic](predictor_wide_range_theory_diagnostic.py) | Theory baseline comparison |
| [wide_range_stagnation_diagnostic](predictor_wide_range_stagnation_diagnostic.py) | Stagnation |
| [wide_range_oracle_mechanism_diagnostic](predictor_wide_range_oracle_mechanism_diagnostic.py) | Oracle mechanism decomposition |
| [wide_range_shadow_phase_diagnostic](predictor_wide_range_shadow_phase_diagnostic.py) | Shadow phase |
| [wide_range_history_scale_diagnostic](predictor_wide_range_history_scale_diagnostic.py) | History scale |
| [wide_range_history_transition_diagnostic](predictor_wide_range_history_transition_diagnostic.py) | History transition |
| [wide_range_history_chain_correlation](predictor_wide_range_history_chain_correlation.py) | History chain correlation |
| [wide_range_conditional_history_distribution](predictor_wide_range_conditional_history_distribution.py) | Conditional history |
| [wide_range_signed_history_distribution](predictor_wide_range_signed_history_distribution.py) | Signed history |
| [wide_range_ancestor_history_decomposition](predictor_wide_range_ancestor_history_decomposition.py) | Ancestor history; shared-analysis compatibility entry |
| [wide_range_ac_decomposition](predictor_wide_range_ac_decomposition.py) | A/C; shared-analysis compatibility entry |
| [wide_range_coherence_structure](predictor_wide_range_coherence_structure.py) | Structural decomposition; shared-analysis compatibility entry |
| [wide_range_coherence_sparsity](predictor_wide_range_coherence_sparsity.py) | Coherence sparsity |

## Historical calibration: local models and ablations

These scripts import some private helpers from one another; the logical grouping does not
mean the dependencies are decoupled.

| Module | Role |
| --- | --- |
| [boundary_aware_score_calibration](predictor_boundary_aware_score_calibration.py) | Boundary-aware score |
| [sparse_first_order_phase_score_calibration](predictor_sparse_first_order_phase_score_calibration.py) | Sparse first-order phase |
| [recursive_gaussian_moment_score_calibration](predictor_recursive_gaussian_moment_score_calibration.py) | Recursive Gaussian moment |
| [gaussian_ancestor_coherence_calibration](predictor_gaussian_ancestor_coherence_calibration.py) | Gaussian ancestor coherence |
| [discrete_ancestor_phase_score_calibration](predictor_discrete_ancestor_phase_score_calibration.py) | Discrete ancestor phase |
| [reliability_weighted_coherence_calibration](predictor_reliability_weighted_coherence_calibration.py) | Reliability weighting |
| [first_order_history_failure_diagnostic](predictor_first_order_history_failure_diagnostic.py) | First-order failure |
| [ancestor_transition_predictability_diagnostic](predictor_ancestor_transition_predictability_diagnostic.py) | Ancestor transition predictability |
| [signed_cell_shift_predictability_diagnostic](predictor_signed_cell_shift_predictability_diagnostic.py) | Signed-cell shift |
| [shadow_trajectory_failure_diagnostic](predictor_shadow_trajectory_failure_diagnostic.py) | Shadow trajectory failure |
| [shadow_sparse_repair_ablation](predictor_shadow_sparse_repair_ablation.py) | Sparse repair |
| [shadow_high_layer_mistake_ablation](predictor_shadow_high_layer_mistake_ablation.py) | High-layer mistakes |

## Historical calibration: selection budget and beam

Budget search, threshold calibration, and complexity diagnostics; already-seen
calibration results must not be relabeled as confirmation.

| Module | Role |
| --- | --- |
| [two_stage_cheap_score_calibration](predictor_two_stage_cheap_score_calibration.py) | Two-stage score |
| [ancestor_cell_beam_score_calibration](predictor_ancestor_cell_beam_score_calibration.py) | Ancestor-cell beam |
| [ancestor_cell_beam_cost_diagnostic](predictor_ancestor_cell_beam_cost_diagnostic.py) | Beam cost |
| [energy_mass_selection_calibration](predictor_energy_mass_selection_calibration.py) | Energy-mass selection |
| [width_aware_cascade_calibration](predictor_width_aware_cascade_calibration.py) | Width-aware cascade |
| [q_beam_shortlist_cascade_calibration](predictor_q_beam_shortlist_cascade_calibration.py) | Q/beam shortlist |
| [ulp_energy_convergence_diagnostic](predictor_ulp_energy_convergence_diagnostic.py) | ULP energy convergence |
| [ulp_energy_cost_pareto_diagnostic](predictor_ulp_energy_cost_pareto_diagnostic.py) | Cost / quality Pareto |

## Closed-book rewrite (replication)

Independent replication partition: the user writes the core, legacy modules serve only as
test references. Rules and order are in [rewrite/README.md](rewrite/README.md).

| Module | Role |
| --- | --- |
| [rewrite package](rewrite/__init__.py) | Rewrite partition entry |
| [rewrite/fp32_oracle](rewrite/fp32_oracle.py) | RN-even rounding and exact whole-tree reduction (passed differential tests) |
| [rewrite/coherence](rewrite/coherence.py) | A/C decomposition and C-dominance statistic (passed differential and replication tests) |
| [rewrite/macro_score](rewrite/macro_score.py) | Q_8/12 score and shortlist (passed frozen replication) |
| [rewrite/regret_stats](rewrite/regret_stats.py) | Regret and stratified bootstrap, recomputing the v2 headline (passed frozen replication) |
| [rewrite/generators](rewrite/generators.py) | Tree and controlled-input generators, seed schedule is a frozen boundary (passed differential and frozen replication) |

## Rules for new code

When adding a module, update this index and add a test at the same time. The maintenance
test checks source coverage to prevent another pile of unexplained scripts. Do not keep
copying same-named helpers from historical runners for new features; do a semantic
comparison and characterization tests first, then extract shared modules with a version
boundary.
