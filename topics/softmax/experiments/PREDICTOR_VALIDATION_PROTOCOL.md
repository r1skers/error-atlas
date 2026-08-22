# Predictor Validation Protocol — Scaffold

Status: **draft scaffold for the future broad multi-family stage**

Two narrow, stage-specific records now sit beside this scaffold:

- `wide_range_energy_beam_v1` is completed; its energy-mass primary comparison did not satisfy its
  preregistered positive-evidence rule.
- `wide_range_fixed_k8_beam_v2` freezes a fresh confirmation of the unchanged fixed-K8/B3 baseline
  that was promoted only after v1 was opened; it completed with positive pooled evidence.

An engineering follow-up, `wide_range_fixed_k8_beam_score_only_v1`, then removed `Fraction` and
oracle instrumentation from the inference path.  It reproduced every Q score, beam score,
shortlist, and selection on all 192 v2 groups exactly.  The remaining algorithm still performs 64
macro traversals plus four full shadow traversals per input, so this establishes a score-only
prototype—not a production-cheap per-vector selector.

Neither stage freezes, completes, or weakens this broader three-family scaffold.  The prohibitions
below continue to apply to held-out data not explicitly authorized by a stage-specific
preregistration.

This document is the working protocol for the next Softmax predictor-validation stage.
It deliberately separates protocol structure from research decisions that still need to be
made by the project owner. A field marked `TO FREEZE` is not an invitation to infer a
reasonable default during execution: it must be resolved explicitly before the relevant
stage can run.

This file does not modify the status of any existing preregistration, result artifact, or
negative result.

## 1. Scope and stage boundary

The next stage evaluates whether a cheap, pre-execution score is useful for ranking or
screening reduction-graph failures under a controlled stored-input distribution.

Until this protocol is frozen:

- do not generate a new held-out validation set;
- do not inspect held-out labels, oracle outputs, or held-out metric values;
- do not patch the cheap score in response to held-out behavior;
- do not overwrite or reinterpret existing frozen artifacts;
- do not expand the stage to GPU measurements or a new real-model data source.

Calibration/pilot work, if used, must be explicitly identified as calibration data and
must never later be relabeled as held-out evidence.

## 2. Experimental unit

**Decision recorded:** the independent experimental/sampling unit is one input group: one
ordered set of stored FP32 leaves. The graph observations evaluated on those same leaves
are paired observations within that input group, not independent samples.

The dataset schema must retain a stable `input_group_id` on every graph observation.
Calibration/held-out assignment is group-wise: graph rows from one input group must never
be split across the boundary.

Every input group used in the primary analysis must have the complete graph-observation
set required by the frozen graph protocol. If a required graph observation is missing or
invalid, preserve the group and failure metadata for diagnostics, but exclude the whole
group from the primary paired analysis rather than silently using the remaining rows.

The exact graph set remains `TO FREEZE` in Section 4.

## 3. Controlled stored-input distribution

This stage uses a **controlled synthetic stress distribution**, not a distribution intended
to estimate the prevalence of numerical failures in real Softmax workloads. Its purpose is
to provide broad and deliberately difficult numerical cases for predictor
ranking/screening validation. Failure prevalence measured here must not be reported as a
real-workload prevalence estimate.

### Input families

Use three approximately equally represented families:

1. **head + many small tails** — one dominant term with many smaller terms;
2. **same-scale random** — randomly varying terms of broadly comparable scale;
3. **wide-dynamic-range random** — terms spanning a deliberately broad exponent / scale
   range.

Family membership is a broad stratum, not a single fixed template. Each family must vary
its internal difficulty parameters so the dataset covers a range of numerical conditions.
The exact within-family parameter distributions remain `TO FREEZE`.

### Width strata

The regular validation width strata are:

- 256;
- 1,024;
- 4,096;
- 16,384;
- 65,536;
- 262,144.

Width 1,048,576 is reserved as an optional resource-stress stratum and is not part of the
regular approximately equal-allocation validation mixture unless a later protocol revision
explicitly promotes it before held-out generation.

This boundary was selected from an infrastructure-only VPS resource benchmark. On the
current 4-logical-CPU / approximately 3.7-GiB-RAM environment, width 262,144 required
about 14.9 seconds group total and about 203 MiB peak RSS for the benchmarked sequential
and balanced exact-oracle pair; width 1,048,576 required about 59.8 seconds and about
760 MiB. No swap was used in either run. These measurements constrain execution cost only
and were not selected using predictor, oracle-error, or failure metrics.

### Distribution health checks

Calibration may use input-only diagnostics to check whether the generated stress inputs
are diverse and whether the three families occupy useful, non-degenerate regions of input
feature space. Candidate diagnostics include exponent range, head dominance, top-k mass
fraction, coefficient of variation, pairwise feature distances, feature-histogram entropy,
family silhouette, and coverage bins.

Generator parameters may be revised during calibration in response to these **input-only**
distribution-health diagnostics. They must not be revised in response to held-out data or,
for distribution tuning, in response to oracle failure labels, failure prevalence,
predictor correlation, AUROC, or other predictor-performance metrics.

### Still to freeze

Before held-out generation, freeze:

- generator name and version;
- exact within-family parameters and their distributions;
- exact family and width allocation counts;
- stored dtype and materialization rules;
- ordering or layout rules;
- random-seed policy;
- exclusions and validity checks.

Generated records must distinguish generator/source values from the actual ordered stored
FP32 leaves consumed by the graph oracle.

## 4. Graphs and pairing

`TO FREEZE`

Specify the graph families evaluated for each input group and whether every group is
required to have a complete paired set of graph observations.

At minimum the protocol must state:

- graph names and graph-definition versions;
- whether graph mixture weights are fixed;
- how incomplete pairs are handled;
- whether any graph-specific metric is considered primary or diagnostic.

## 5. Targets and labels

### Continuous target

`TO FREEZE`

Freeze the exact formula, normalization, sign convention, and handling of degenerate
cases. Preserve signed graph error in the raw data even if the primary continuous target
uses an absolute value or normalization.

### Classification target

`TO FREEZE`

Freeze the failure definition before held-out generation. The failure label must be
computed from frozen oracle / policy logic, not from the cheap score being evaluated.

## 6. Cheap score

`TO FREEZE`

Before held-out generation, record:

- exact score formula;
- version / identifier;
- inputs the score is allowed to inspect;
- monotonic direction (larger means more risky, or smaller means more risky);
- deterministic tie behavior if needed;
- any preprocessing or normalization.

The score is a screening/ranking statistic unless a separate claim is preregistered. Do
not silently reinterpret it as a probability, rigorous error bound, or safety
certificate.

## 7. Calibration / pilot stage

`TO FREEZE`

If a calibration stage is used, predeclare what it may be used for. Candidate purposes
include checking generator health, pair completeness, rough failure prevalence, score
direction, and planning held-out sample size.

Before opening held-out data, record which research choices may still change based on
calibration and which choices are already frozen.

If additional calibration groups may be generated adaptively, define the stopping rule or
the limited set of reasons that authorize more calibration data. Do not keep generating
until a preferred metric value appears.

## 8. Held-out split and opening rule

`TO FREEZE`

Freeze:

- held-out group count or sample-size rule;
- split/generation procedure;
- seed or seed-generation policy;
- strata allocation;
- one-shot opening procedure;
- what metadata is recorded before the first held-out metric is computed.

Held-out data are opened only after Sections 2–9 contain no unresolved item required to
compute the preregistered metrics.

After opening, changes to the score, target, sampling unit, bootstrap scheme, or primary
metric require a new validation stage rather than an in-place rewrite of this one.

## 9. Metrics and uncertainty

`TO FREEZE`

The protocol must name the primary ranking metric(s), any secondary diagnostics, and the
uncertainty procedure before held-out generation.

For each metric, state:

- pooled versus graph-specific reporting;
- weighting / mixture convention;
- resampling unit;
- confidence-interval method and number of resamples if applicable;
- behavior when a stratum contains too few positive or negative labels for the metric to
  be defined;
- whether rare-failure diagnostics such as PR-AUC or recall at a fixed inspection budget
  are included.

Paired observations from the same input group must remain paired whenever the frozen
uncertainty procedure requires group-level resampling.

## 10. Prevalence and evidence sufficiency

`TO FREEZE`

Always report failure prevalence alongside classification metrics. Define in advance how
the report labels graph/stratum results that are undefined or too weakly supported to
interpret because failures (or non-failures) are too rare.

Do not convert an undefined graph-specific metric into an apparently valid numeric score
by silently pooling, smoothing, or dropping the problematic stratum.

## 11. Artifact boundary

Before execution, define the output directory and versioned schemas for at least:

- frozen protocol / configuration snapshot;
- input-group records;
- graph-observation records;
- score values;
- continuous targets and failure labels;
- metric summary;
- uncertainty / bootstrap summary;
- metadata and hashes needed to reproduce the run.

`TO FREEZE`: exact paths, schema versions, identifiers, and overwrite policy.

New held-out runners should refuse to overwrite an existing frozen held-out artifact
directory unless a separately documented recovery procedure explicitly allows it.

## 12. Ownership checkpoint before implementation

Before metric or generator implementation begins, the project owner should be able to
explain, in their own words:

1. the experimental unit and why paired graph rows are or are not independent;
2. the calibration-versus-held-out boundary;
3. the continuous target and failure label;
4. what the cheap score is allowed to know;
5. the primary metric and its monotonic direction;
6. the uncertainty / resampling unit;
7. what happens when failures are too rare for a requested metric;
8. what decisions become immutable when held-out data are opened.

The project owner should then sketch the minimal metric/data-flow pseudocode. Agent work
may review that design and provide tests, CLI/scaffolding, schema checks, and mechanical
refactoring without silently choosing unresolved research decisions.

## 13. Freeze checklist

This protocol may be marked **frozen for held-out generation** only when all applicable
`TO FREEZE` fields above have been resolved and reviewed.

Before changing the status line, verify that:

- the controlled distribution and graph pairing are versioned;
- targets and label definitions are fixed;
- the cheap score and direction are fixed;
- calibration is closed or its allowed continuation rule is explicit;
- held-out size/split/opening rules are fixed;
- metrics and uncertainty are fixed;
- artifact paths/schemas and no-overwrite behavior are fixed;
- no held-out outputs have been inspected while making these choices.

Changing the status line alone does not create evidence; evidence begins only when the
frozen protocol is executed and its artifacts are reviewed.
