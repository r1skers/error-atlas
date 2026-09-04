# Softmax

How input perturbation and finite-precision error propagate through normalization
and reduction, separating problem conditioning, algorithm stability, input
representation error, and statistical predictability.

## Navigation

- [Current resume point](../../NEXT_SESSION.md) — current status and the next step to freeze.
- [Foundations & exact-oracle notes](notes/foundations.md) — Jacobian, directionality, floating-point budget, P0–P5.
- [Early experiment notes](notes/early_experiments.md) — shift resolution, summation, and failure triage.
- [Replication notes](notes/rewrite_replication.md) — the independent blank-slate rewrite and what it verified.
- [Online normalizer contract](notes/online_normalizer_contract.md) — arithmetic contract, the frozen-weight weighted identity, and what a CUDA measurement would need.
- [Experiment code index](experiments/README.md) — find entry points by module role.
- [Results index](experiments/results/README.md) — cite conclusions by evidence grade.
- [Regression tests](tests/) — separate from the experiment implementations.

## Research stages

| Stage | Established scope |
| --- | --- |
| Foundations & exact oracle | Exact semantics of an explicit nonnegative FP32 reduction tree; early accepted/provisional split preserved |
| Depth-margin baseline | Archived universal-ordering counterexample |
| Calibration diagnostics | Structural features, second moment, history/phase, beam and cost exploration; not confirmation data |
| Energy beam v1 → fixed-K8 v2 | v1 primary negative; v2 passes pooled confirmation on fresh controlled inputs |
| Score-only → offline reuse | Oracle-free prototype still costly; offline reuse fails the balanced-FP32 gate |
| Online risk certificate | Latest completed stage is calibration: statistical signal, but no new confirmation or deployment |

The main finding is that the reduction error is dominated by the **sign coherence**
between local rounding errors, so a magnitude-only score cannot rank trees; a
coherence-aware beam wins only narrowly and not yet cheaply. The latest direction is
sparse exactness correction rather than stacking more global topology scores. For exact
conclusions and boundaries, defer to the artifacts linked from
[NEXT_SESSION.md](../../NEXT_SESSION.md).

## Regression and boundaries

From the repository root:

```sh
python tools/run_tests.py --suite softmax
```

Passing tests do not certify unmeasured repeatability, cross-distribution
generalization, or GPU performance. Ownership of the research core continues to follow
the [implementation-learning protocol](../../framework/implementation_learning_protocol.md).
