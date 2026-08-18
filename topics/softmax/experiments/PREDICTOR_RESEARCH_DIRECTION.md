# Predictor Research Direction — Second-Moment Theory Checkpoint

Status: **calibration research note; not a frozen validation protocol**

This note records a research-direction checkpoint reached during calibration. It does not
authorize held-out generation and does not change any frozen artifact or prior negative
result.

## Why this checkpoint exists

The first hand-designed structural predictors were intentionally simple:

- sibling scale mismatch;
- dominant-leaf exposure.

On the predeclared irregular stored-FP32 calibration inputs, increasing the random-tree
sample from K=16 to K=64 removed most of the large apparent rank correlations seen in the
smaller sample. The remaining within-input Spearman correlations are generally weak and
heterogeneous across input families. These results are calibration diagnostics only, but
they are sufficient to discourage adding more ad-hoc global summary features without a
stronger numerical model.

A closely related recent paper provides that stronger theoretical reference point:

> P. Sao, N. Miniskar, P. Valero-Lara, K. Teranishi, and S. Seal,
> “A Second-Moment Theory for Floating-Point Reduction Trees,” arXiv:2607.18758, 2026.

The paper derives a mean-square-error recurrence for binary reduction trees under
conditionally unbiased rounding. Its leading tree-dependent cost depends on partial sums
through a common-ancestor kernel, and for common statistical input models reduces to tree
statistics involving leaf depth and squared internal-subtree sizes. The paper also reports
that stagnation and bias in positive low-precision sums limit the model's applicability.
That limitation is directly relevant to Softmax-style reductions, whose leaves are
nonnegative after exponentiation.

## Revised research question

The immediate goal is no longer to invent additional scalar topology summaries until one
happens to correlate with the oracle. The next question is:

> How well does a second-moment / partial-sum theory baseline rank the deterministic
> round-to-nearest-even FP32 error of candidate reduction graphs for a fixed positive
> input, and what systematic residual structure remains in the Softmax-like regime?

The project remains focused on **input-specific within-input graph ranking**. Expected or
RMS behavior over an input distribution is useful as a baseline but is not interchangeable
with the deterministic target used by the exact oracle for one stored input and one graph.

## Immediate calibration plan

Before introducing a learned TreeRNN, GNN, or other flexible predictor:

1. Implement the smallest faithful theory-derived baseline that can be evaluated from the
   stored FP32 leaves and candidate tree without using the exact FP32 oracle as an input.
   A partial-sum/tree cost such as the paper's leading second-moment term is the intended
   starting point; the exact implemented formula must be checked against the paper before
   being named as that baseline.
2. Evaluate it on the same predeclared irregular calibration inputs and random-tree
   schedule already used by the current ranking smoke test. Do not select seeds or input
   rows using the resulting predictor metrics.
3. Compare its within-input rank correlation against the existing mismatch and dominant
   exposure baselines.
4. Inspect where the theory-derived score and deterministic RN-even oracle disagree,
   especially in positive-input cases with few distinct oracle error levels, stagnation,
   or systematic rounding bias.
5. Only after this comparison decide whether a learned local correction is justified.

## Candidate learned direction, if a residual gap remains

If the theory baseline leaves reproducible deterministic residual structure, learning
should first be used as a constrained correction rather than as an unrestricted black-box
replacement. One candidate form is a tree-local state or correction term conditioned on
cheap quantities such as partial-sum scale, exponent/mantissa alignment, subtree size,
depth, or other explicitly permitted pre-execution state.

A learned model is useful scientifically only if it helps identify information absent from
the theory baseline and remains cheap enough to support the eventual systems goal of graph
selection or construction. Predictor cost must therefore remain part of later evaluation.

## Boundary with the held-out protocol

Nothing in this note freezes the final predictor, target, graph mixture, K, generator, or
metric protocol. The current irregular inputs and K=64 runs remain calibration-only. No
held-out data should be generated or inspected because of this checkpoint.

Before held-out validation begins, any theory-derived or learned score promoted to the
primary predictor must be versioned and frozen in `PREDICTOR_VALIDATION_PROTOCOL.md` with
its allowed inputs, monotonic direction, preprocessing, and metric procedure.
