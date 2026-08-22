# Fixed-K8/B3 score-only inference and cost v1

Status: **completed engineering validation**.

This milestone separates the frozen v2 selector from its evidence-generation runner.  The new
`predictor_fixed_k8_beam_inference.py` consumes only nonnegative finite FP32 leaf bits, explicit
candidate graphs, and the already-frozen innovation model.  It does not import or execute the exact
candidate oracle and performs no `Fraction` arithmetic.

The score itself is unchanged:

1. evaluate connected-root-band `Q_8 / 12` for all 64 candidates;
2. retain the four lowest-Q candidates with stable graph-index ties;
3. rerank only those four with the frozen 19-feature, width-three joint cell beam.

Exact subtree sums are maintained as integers on the binary32 `2**-149` lattice.  This preserves
the v2 exact phase/ULP semantics without replaying a candidate FP32 trajectory.

## Frozen-score fidelity

All 192 completed v2 groups were replayed solely as an implementation-fidelity check.  Held-out
targets were not used to tune the implementation or to estimate new efficacy:

| Check | Exact reproductions | Maximum absolute difference |
| --- | ---: | ---: |
| all 64 Q scores per group | 192 / 192 | 0 |
| all four beam scores per group | 192 / 192 | 0 |
| shortlist | 192 / 192 | — |
| Q-only selection | 192 / 192 | — |
| beam selection | 192 / 192 | — |

Because every frozen decision and score is identical, the completed v2 efficacy result is preserved
rather than re-estimated: best-tier hit `70.83% -> 83.33%`, and mean normalized regret
`0.13191 -> 0.07421` (a `43.7%` relative reduction).

## Prototype cost

The benchmark used four public v2 inputs per width and five selector repeats.  Graph generation and
model loading were excluded.  Times are medians from this Python environment and are not compiled
kernel claims.

| Width | Q-only selector | Q + B3 selector | Beam/Q cost | One Python FP32 tree | Selector/tree | Speedup vs Fraction oracle (64 trees) | Peak traced memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 4.816 ms | 10.220 ms | 2.12x | 0.112 ms | 91.1x | 27.1x | 1.84 MiB |
| 512 | 7.991 ms | 17.275 ms | 2.16x | 0.233 ms | 74.0x | 34.0x | 3.63 MiB |
| 1024 | 14.361 ms | 31.024 ms | 2.16x | 0.616 ms | 50.4x | 39.9x | 7.20 MiB |

The robust algorithmic count explains the wall-clock result: this online selector performs 64
full-width macro metadata traversals plus four full-width shadow traversals, or 68 full-tree passes
before the small K=8 beam work.  Removing research instrumentation therefore makes the selector
much cheaper than the exact `Fraction` oracle, but not cheap relative to executing one reduction.

## Cost-quality conclusion

The fixed-K8/B3 point is a real quality improvement over Q-only, but costs about `2.1x` the Q-only
selector in this prototype.  It is therefore Pareto-relevant when numerical ranking quality is the
objective.  It is not yet a low-overhead per-vector online selector:

- every legal tree still performs `n - 1` additions, so better numerical selection has no intrinsic
  runtime saving that repays selection cost;
- amortizing selection to 10% of one-tree execution would require roughly 504--911 reuses in these
  Python measurements;
- reuse across changing inputs is not validated, while the current score is input-dependent.

The defensible label is **score-only validated research prototype**.  A production-cheap claim
would require a different deployment contract (for example offline/representative-input reuse) or
an architectural cost reduction, plus compiled target-hardware measurements.

## Reproduction

From `topics/softmax/experiments`:

```text
python predictor_fixed_k8_beam_inference_benchmark.py \
  --groups-per-width 4 \
  --repeats 5 \
  --reduction-repeats 20
```

Machine-readable output and provenance are in `benchmark.json`.
