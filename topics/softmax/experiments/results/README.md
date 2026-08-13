# Softmax Experiment Artifacts

This directory separates generated evidence by experiment. Source modules and
their tests remain adjacent in the parent directory so direct script execution
and the existing test imports stay simple.

| Directory | Question isolated | Artifacts |
| --- | --- | --- |
| `shift_resolution/` | When does FP32 input quantization erase a logit difference? | Probe CSV and provenance metadata |
| `summation_permutation/` | How do order, balanced trees, and compensation change FP32 summation? | Controlled permutation CSV and metadata |
| `softmax_summation/` | How does denominator summation error propagate through normalization? | End-to-end CSV and metadata |
| `failure_triage/` | Which mitigation passes tolerance or correct-rounding policy? | Checked-in legacy raw/summary/metadata; schema v2 adds case and assessment CSVs when intentionally regenerated |
| `graph_predictor_validation/` | Can stored inputs plus an explicit FP32 addition tree predict signed reduction error before execution? | 12-row accepted selector inside a retained 36-row provisional batch; see the directory README |
| `nonuniform_graph_predictor_v1/` | Does the predictor survive one preregistered nonuniform positive input? | Frozen preregistration, two single-run observation rows, provenance metadata, and evidence-status README; repeatability not measured |
| `graph_risk_proxy_v1/` | Does moving a large leaf across depths provide a new validation mechanism for a depth-margin screening score? | Frozen but unexecuted preregistration, reclassified before execution as known-mechanism calibration / optional replication |
| `depth_margin_topology_challenge_v1/` | Can equal depth-margin scores universally rank correct-rounding failure when only sibling grouping changes? | Frozen adversarial preregistration, two exact-oracle rows, provenance metadata, and a confirmed tied informative counterexample; no candidate execution |

CSV files are evidence rather than hand-maintained inputs. JSON metadata records
the execution environment, registered configuration, and source or artifact
hashes needed to interpret that evidence. Regenerate artifacts from the
repository root with the commands documented in the parent `README.md`.

The checked-in `failure_triage/` directory is the 2026-08-08 artifact snapshot
from the earlier suite. The current source and tests define suite version 3 and
artifact schema version 2, but those expanded artifacts have intentionally not
overwritten the earlier evidence. Generate them in a new scratch directory and
review them before replacing any checked-in artifact.

The graph-predictor batch also has an explicit evidence-status split.  Its raw
36-row CSV remains intact, while only the stepwise-reviewed selector is accepted
as the current milestone.  Read `graph_predictor_validation/README.md` before
citing the batch-wide result.
