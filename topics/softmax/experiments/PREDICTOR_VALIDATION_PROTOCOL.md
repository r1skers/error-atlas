# Predictor Validation Protocol — Scaffold

Status: **draft scaffold; not frozen; no new validation data authorized**

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

`TO FREEZE`

Record explicitly:

- what constitutes one input group;
- which graph observations are produced from one group;
- which observations share the same underlying stored leaves;
- the unit that is considered independent for sampling and uncertainty estimation.

The dataset schema must retain an `input_group_id` (or an equivalently explicit stable
identifier) so paired graph observations cannot be accidentally treated as unrelated
rows.

## 3. Controlled stored-input distribution

`TO FREEZE`

Freeze before held-out generation:

- generator name and version;
- all generator parameters and their distributions;
- stored dtype and materialization rules;
- shape / width strata;
- ordering or layout rules;
- random-seed policy;
- exclusions and validity checks.

Generated records must distinguish generator/source values from the actual ordered
stored FP32 leaves consumed by the graph oracle.

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
