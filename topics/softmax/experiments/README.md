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

- **Direction**: the first probability changes from approximately $0.731$ to
  $0.5$ when the stored difference collapses from one to zero.
- **Scale**: the first-component absolute error approaches $0.231$.
- **Boundary**: FP32 stops resolving consecutive integers at approximately
  $2^{24}=16{,}777{,}216$.
- **Failure signature**: stored_difference is zero together with
  first_probability equal to $0.5$.

## Observed result

| Common offset | FP32 ULP | Stored difference | First probability | Absolute probability error |
| ---: | ---: | ---: | ---: | ---: |
| $2^{23}$ | 1 | 1 | 0.731058598 | $1.89\times10^{-8}$ |
| $2^{24}$ | 2 | 0 | 0.5 | 0.231058579 |
| $2^{25}$ | 4 | 0 | 0.5 | 0.231058579 |

All four predictions were observed. At $2^{24}$, the stored difference is
already zero before subtract-max, so the error is attributed to FP32 input
quantization rather than normalization.

## Boundary audit

Mathematical shift invariance does not imply finite-precision shift invariance.
If $Q$ denotes FP32 quantization, then in general

$$
Q\!\left(\mathbf z-m\mathbf1\right)
\ne
Q(\mathbf z)-\max(Q(\mathbf z))\mathbf1.
$$

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

- `results/shift_resolution/fp32_shift_resolution.csv`
- `results/shift_resolution/fp32_shift_resolution_metadata.json`

The CSV preserves measured evidence. The metadata records Python, NumPy,
platform, FP32 parameters, error settings, probe configuration, and a source
hash.

## Failure-triage summation suite

`softmax_failure_triage_runner.py` registers two fast power-tail cases and one
explicit stress case.  Every case freezes the actual FP32 input

$$
q=(1,2^{-24},\ldots,2^{-24})
$$

before comparing sequential FP32, pairwise FP32, compensated FP32, and
sequential FP64 accumulation with FP32 output.  Each candidate is repeated
three times under one recorded environment and consumer policy.  Raw runs are
preserved separately from derived summaries and assessments.

From the repository root:

    python topics/softmax/experiments/softmax_failure_triage_runner.py --include-stress

Outputs:

- `results/failure_triage/softmax_failure_triage_runs.csv`
- `results/failure_triage/softmax_failure_triage_summary.csv`
- `results/failure_triage/softmax_failure_triage_metadata.json`

For the $2^{20}$-tail stress case, sequential FP32 returns $1$ and loses
the full tail mass $1/16$, giving absolute relative error $1/17$.  The
other three registered candidates return the exact FP32 value $17/16$ for
this case.  This establishes numerical eligibility only; performance and
resource ranking remain target-implementation measurements.

The complete artifact layout, including the two intermediate summation
experiments, is indexed in `results/README.md`.
