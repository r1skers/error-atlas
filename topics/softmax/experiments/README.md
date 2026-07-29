# Softmax Experiments

## FP32 shift resolution

The file fp32_shift_resolution.py isolates one error source: loss of a unit
logit difference when a large common offset consumes FP32 resolution.

The research-core function is learner-owned:

    fp32_softmax_probe(common_offset)

It quantizes the logits before subtract-max so that the experiment measures
input-representation error rather than a hypothetical higher-precision
pre-centering path.

## Prediction record

- **Direction**: the first probability changes from approximately \(0.731\) to
  \(0.5\) when the stored difference collapses from one to zero.
- **Scale**: the first-component absolute error approaches \(0.231\).
- **Boundary**: FP32 stops resolving consecutive integers at approximately
  \(2^{24}=16{,}777{,}216\).
- **Failure signature**: stored_difference is zero together with
  first_probability equal to \(0.5\).

## Observed result

| Common offset | FP32 ULP | Stored difference | First probability | Absolute probability error |
| ---: | ---: | ---: | ---: | ---: |
| \(2^{23}\) | 1 | 1 | 0.731058598 | \(1.89\times10^{-8}\) |
| \(2^{24}\) | 2 | 0 | 0.5 | 0.231058579 |
| \(2^{25}\) | 4 | 0 | 0.5 | 0.231058579 |

All four predictions were observed. At \(2^{24}\), the stored difference is
already zero before subtract-max, so the error is attributed to FP32 input
quantization rather than normalization.

## Boundary audit

Mathematical shift invariance does not imply finite-precision shift invariance.
If \(Q\) denotes FP32 quantization, then in general

\[
Q\!\left(\mathbf z-m\mathbf1\right)
\ne
Q(\mathbf z)-\max(Q(\mathbf z))\mathbf1.
\]

Centering in higher precision before quantization can preserve the unit
difference, but standard subtract-max cannot recover information already lost
in FP32 logits. The current experiment covers binary Softmax, a positive unit
gap, three powers-of-two offsets, and NumPy FP32; it does not yet establish a
general mixed-precision policy.

## Ownership and closed-book evidence

The first implementation was learner-written and reviewed before execution.
After one local reference demonstration, the learner independently recovered
the quantize-before-stabilize mechanism in rewrite_fp32_shift_resolution.py.
The original and rewritten functions match exactly on all three registered
probes.

## Run

From the repository root:

    python topics/softmax/experiments/fp32_shift_resolution.py
    python -m unittest discover -s topics/softmax/experiments -p "test_*.py" -v

Outputs:

- results/fp32_shift_resolution.csv
- results/fp32_shift_resolution_metadata.json

The CSV preserves measured evidence. The metadata records Python, NumPy,
platform, FP32 parameters, error settings, probe configuration, and a source
hash.
