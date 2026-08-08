# Softmax Experiment Artifacts

This directory separates generated evidence by experiment. Source modules and
their tests remain adjacent in the parent directory so direct script execution
and the existing test imports stay simple.

| Directory | Question isolated | Artifacts |
| --- | --- | --- |
| `shift_resolution/` | When does FP32 input quantization erase a logit difference? | Probe CSV and provenance metadata |
| `summation_permutation/` | How do order, balanced trees, and compensation change FP32 summation? | Controlled permutation CSV and metadata |
| `softmax_summation/` | How does denominator summation error propagate through normalization? | End-to-end CSV and metadata |
| `failure_triage/` | Which mitigation passes a consumer accuracy/repeatability policy? | Raw runs, derived summary, and metadata |

CSV files are evidence rather than hand-maintained inputs. JSON metadata records
the execution environment, registered configuration, and source or artifact
hashes needed to interpret that evidence. Regenerate artifacts from the
repository root with the commands documented in the parent `README.md`.
