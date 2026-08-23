# Wide-range online risk certificate — calibration v1

Status: **completed calibration; positive statistical signal, no confirmation claim**

This experiment stopped ranking candidate trees.  It executed one fixed balanced FP32 tree and
carried an annotated state beside that reduction.  The target is reduction error relative to the
exact sum of already-stored positive FP32 leaves; input materialization and the other Softmax stages
remain outside the contract.

The frozen run contains 192 calibration inputs: 64 at each width 256, 512, and 1024.  Every
statistical prediction is five-fold cross-fitted.  Thus each row's error interval is calibrated on
the other four folds, but the entire experiment remains calibration-only rather than held-out
evidence.

## Main result

For the primary state, let

$$
Q_{\rm inexact}
=\frac1{12}\sum_{v:\,\mathrm{add}_v\;\mathrm{inexact}}
\left(\frac{U_v}{U_{\rm root}}\right)^2.
$$

A fold-fitted Gaussian model for $E/\sqrt{Q_{\rm inexact}}$ passed every preregistered calibration
criterion:

| Diagnostic | Frozen requirement | Observed |
| --- | ---: | ---: |
| pooled 90% interval coverage | at least 88% | 90.10% (173/192) |
| pooled 99% interval coverage | at least 98% | 100% (192/192) |
| per-width 90% coverage | at least 85% | 92.19%, 92.19%, 85.94% |
| per-width 99% coverage | at least 95% | 100% at every width |
| cross-fitted $|Z|$ p99 | at most 3.5 | 2.225 |
| correct-rounding Brier score | below prevalence baseline | 0.16945 vs 0.19065 |

The standardized width-specific standard deviations were 0.987, 0.943, and 1.065.  Fold-fitted
standardized means were only +0.028 to +0.065, and the frozen zero-mean variant produced essentially
the same Brier score and coverage.  This is the useful role for the earlier bell-shaped observation:
sign symmetry is suitable for integrating a tail/safe-cell probability even though it was harmful
when used to average away sign coherence for fine tree ranking.

The computed root was correctly rounded on 143/192 inputs (74.48%).  The primary model assigned
$P_{\rm safe}\ge0.90$ to 34/192 inputs; 33/34 of those were actually correct (97.06%).  Only one
input reached 0.99.  Therefore this is a promising risk gradient, not yet a broadly useful
high-confidence acceptance certificate.

## What the cheaper and more complicated variants say

| State | 90% coverage | 99% coverage | Brier | $P_{\rm safe}\ge0.90$ |
| --- | ---: | ---: | ---: | ---: |
| $Q_{\rm all}$ | 89.06% | 98.96% | 0.18308 | 0/192 |
| $Q_{\rm corr4,all}$ | 88.54% | 98.96% | 0.18141 | 0/192 |
| $Q_{\rm inexact}$ | 90.10% | 100% | **0.16945** | 34/192 |
| $Q_{\rm corr4,inexact}$ | 90.10% | 100% | 0.16956 | 35/192 |

$Q_{\rm all}$ needs no signed phase prediction and already gives a usable coarse statistical
interval, but its input-to-input probability range is too narrow for the frozen 0.90 acceptance
threshold.  The four-gap ancestor kernel adds no material value.  In particular,
$Q_{\rm corr4,inexact}$ reaches only 81.25% coverage at nominal 90% for width 1024, so the added
state is not justified by this run.

## Rigorous envelope: valid but too loose

The online recurrence also carried

$$
B_{\rm inexact}
=\sum_{v:\,\mathrm{add}_v\;\mathrm{inexact}}\frac{U_v}{2},
$$

with outward-rounded binary64 bookkeeping.  The exact identity $E=\sum_v\delta_v$ implies
$|E|\le B_{\rm inexact}$ under the frozen reduction-only contract.  All 192 observations were
covered, but the bound-to-actual-error ratio had median 10.06, p90 48.67, and p99 226.39.  Its
interval fit strictly inside the returned FP32 value's rounding cell on 0/192 inputs.  The rigorous
path therefore cannot currently certify correct rounding; it is a valid but operationally weak
upper envelope.

## Where the probability signal comes from

The fraction of inexact additions is nearly constant at about 91% for all widths.  The useful
variation is instead in their energy:

| Width | mean inexact-node fraction | mean inexact-energy fraction | mean top-8 energy share |
| ---: | ---: | ---: | ---: |
| 256 | 91.10% | 57.59% | 83.21% |
| 512 | 90.96% | 56.51% | 85.42% |
| 1024 | 90.92% | 48.59% | 87.08% |

This explains both the promise and the cost problem.  The primary Python trace detects exactness at
every node through aligned integer operand arithmetic.  That is much cheaper than scoring 64 trees,
but it is not a demonstrated low-overhead kernel—and once the full discarded-bit information is
available, accumulating signed local residuals would be close to computing the exact reduction
error directly.

At the same time, only a few high-energy nodes dominate $Q$.  The next scientifically clean
candidate is therefore a **sparse exactness correction**: maintain $Q_{\rm all}$ everywhere, inspect
exact/inexact status only for a fixed root band or top-energy budget, and remove only those proven
zero-noise terms.  The budget and formula must be frozen on a new calibration version; this run's
192 labels cannot be reused to promote it.

## Deployment interpretation

No deployment policy is approved.  Descriptively, if an ideal fallback corrected every selected
input, sending the lowest predicted-safety 20% to fallback would catch 18/49 balanced-tree failures
and raise correct rounding from 74.48% to 83.85%.  A 50% fallback budget would catch 37/49 and reach
93.75%.  Those rates are not competitive with the previously observed 98.44% for Kahan FP32 or
100% for FP64-then-FP32 on this synthetic distribution, and the fallback curve was not a frozen
primary metric.

The present conclusion is narrower:

1. the Gaussian idea is rehabilitated for **uncertainty integration**, not tree ranking;
2. $Q_{\rm all}$ is a very cheap coarse scale, while exactness-weighted energy adds a real risk
   gradient;
3. the rigorous half-ULP sum is too loose;
4. full-tree exactness inspection is not yet cheap enough to justify confirmation;
5. a new sparse high-energy exactness proxy is the next candidate, not another global coherence
   score.

Artifacts:

- `wide_range_online_risk_certificate_calibration_v1_preregistration.json` — immutable design;
- `calibration/observations.csv` — all 192 base traces and cross-fitted predictions;
- `calibration/model_summary.json` — model, coverage, probability, bound, and macro summaries;
- `calibration/metadata.json` — source/protocol hashes and pre-opening Git provenance.

Run the source only in a clean committed checkout and only when no `calibration/` directory exists:

```text
python topics/softmax/experiments/predictor_online_risk_certificate_calibration.py
```
