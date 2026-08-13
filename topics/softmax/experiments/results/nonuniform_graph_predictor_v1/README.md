# Nonuniform Graph Predictor V1

Evidence status: **accepted single preregistered case; repeatability not
measured**.

This directory records one nonuniform positive-FP32 input and two explicit
reduction graphs.  It does not establish a nonuniform-input family result.

## Frozen preregistration

- Case: `nonuniform_positive_v1`
- Ordered exact leaves: `(4u, 2u, 2u, u, 1)`, with `u = 2**-27`
- Exact leaf sum: `134217737/134217728 = 1 + 9u`
- Correctly rounded FP32 target: `0x3f800001`
- Preregistration SHA-256:
  `0c8047f38363c02ef6a6995bcc58a3f890dfebe7395aa5eb62a5cb671d6e47a6`

The preregistration file was hashed before candidate execution and was not
modified afterward.

## Observation

| Graph | Predicted / actual bits | Predicted / actual signed error | Prediction match | Correctly rounded |
| --- | --- | --- | --- | --- |
| sequential left-to-right | `0x3f800001` | `7/134217728` | yes | yes |
| balanced contiguous floor-half | `0x3f800000` | `-9/134217728` | yes | no |

The observation CSV versions graph-output prediction and candidate
correct-rounding as separate fields.  This prevents a correct prediction of an
incorrect candidate output from being collapsed into one ambiguous pass/fail.

Each graph was executed once.  The artifacts contain no repeated raw runs,
bit-pattern counts, or repeatability summary, so they do not support a bitwise
repeatability claim.  Future regression-test executions also do not change this
evidence status because they are not recorded as repeatability observations.

If repeatability becomes a consumer requirement, this case should reuse the
failure-triage raw -> summary -> assessment pipeline rather than add a second
ad hoc repeat runner.

## Boundary

- The source values are exactly representable as stored FP32 leaves, so this
  case isolates reduction rounding rather than source quantization.
- The result validates only the two frozen graphs on this one ordered input.
- Single observations can falsify a graph prediction, but cannot establish
  within-process, cross-process, cross-machine, or GPU repeatability.
- It does not infer an unknown black-box graph, rank summation methods
  universally, or cover cancellation, Kahan operations, FMA, overflow, or GPU.
