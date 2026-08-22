# Wide-range offline tree reuse v1

Status: **completed; preregistered deployment gate failed**.

This stage tested a cost-first deployment contract for the frozen fixed-K8/B3 selector.  A fixed
64-tree catalog was shared across inputs at each width.  The unchanged score was evaluated on 32
fresh representative calibration inputs, one tree was selected per width by mean complete cascade
rank, and that tree was reused on 64 fresh confirmation inputs with no online selection work.

The one-shot run completed all 96 calibration and 192 confirmation groups.  Its opening provenance
is the clean local staging commit `aac78f7`, tree `ac53528868c9c20e7998d787e1fb4a2e596092fe`.
The Git tree, rather than the environment-local commit identity, is the reproducible frozen source
boundary.

## Frozen gates

| Comparison | Mean | 95% bootstrap CI | Gate |
| --- | ---: | ---: | --- |
| random fixed expected regret - score-static regret | +0.04296 | [+0.00697, +0.07692] | pass |
| balanced target - score-static target | -0.02429 | [-0.08459, +0.03568] | fail |
| balanced target - calibration-label static target | -0.03174 | [-0.10059, +0.03548] | fail |

The primary reuse signal is positive: mean normalized regret falls from `0.16730` for a random fixed
catalog tree to `0.12434` for the score-selected static tree, a `25.7%` relative reduction.  The
prespecified engineering decision is nevertheless **no-go** because the static tree does not beat
the simpler balanced FP32 reduction.  Even the tree chosen with calibration oracle labels does not
generalize better than balanced; “oracle-static” is a calibration-information ceiling, not a bound
on unseen-input performance.

## Quality and cost

| Method | Online selection | Mean squared root-ULP error | Correctly rounded |
| --- | ---: | ---: | ---: |
| score-static reuse | 0 passes | 0.22647 | 70.31% (135/192) |
| balanced FP32 | 0 passes | 0.20218 | 75.00% (144/192) |
| per-input K8/B3 | 68 tree-equivalent passes | 0.16363 | 80.73% (155/192) |
| Kahan FP32 | 0 selector passes; compensated loop | 0.07794 | 98.44% (189/192) |
| FP64 then FP32 | 0 selector passes; one FP64 pass | 0.07709 | 100% (192/192) |
| sequential FP32 | 0 passes | 32.20385 | 8.33% (16/192) |

FP64 and Kahan results are descriptive for this positive wide-range synthetic distribution, not
universal correct-rounding guarantees or target-hardware timing claims.

## Why reuse did not close the deployment gap

- At every width, the 32-input cascade aggregation and Q-only aggregation selected the same tree.
  The input-specific B3 phase correction did not survive cross-input averaging.
- The most frequent per-input beam winner captured only `12.5%` to `18.75%` of calibration groups;
  normalized winner entropy was `0.62` to `0.69`.  There is no dominant universal catalog tree.
- Increasing representative count was not monotone.  The frozen r=32 mean regret was `0.12434`,
  while r=1 was `0.12188` and the descriptively best r=8 point was `0.11489`.  r=8 may not be
  promoted after opening.
- The pooled random-baseline comparison passes, but none of the individual width intervals excludes
  zero.  The pooled result is the preregistered claim; width rows remain descriptive.

## Computation advice under the tested contract

For input-dependent one-off reductions, do not run the 68-pass selector.  Use FP64 accumulation and
cast when FP64 is acceptable; otherwise use compensated FP32 when its extra operations are
acceptable.  Under an exact `n-1` FP32-add budget, use the fixed balanced tree.  The current K8/B3
selector remains useful evidence that tree error is predictable, but neither online selection nor
offline reuse is the preferred deployment policy in this experiment.

Machine-readable selections, per-group metrics, raw paired graph observations, input bits, hashes,
and the complete summary are retained in `confirmation/`.
