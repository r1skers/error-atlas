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

`softmax_failure_triage_runner.py` now implements suite version 3 and artifact
schema version 2.  A `CaseRecipe` freezes generator version, parameters, dtype,
shape, and ordered layout; `input_hash` independently audits the materialized
FP32 bytes.  The smoke registry contains 16 cases:

- power tails `(N, e) = (1,-24), (2,-24), (1024,-24), (1023,-34),
  (1024,-34), (1025,-34)`;
- decimal tails `(N,t) = (5,10^{-8}), (6,10^{-8})`;
- both `head_then_tail` and `tail_then_head` layouts for every parameter set.

The opt-in stress tier adds `N=2**20, e=-24` in both layouts.  Every case runs
four candidates: sequential FP32, the repository's fixed pairwise FP32 tree,
Kahan-compensated FP32, and sequential FP64 accumulation with FP32 output.

The evidence pipeline deliberately keeps four layers separate:

1. case rows preserve exact source/stored sums and the correctly rounded FP32
   target;
2. raw rows preserve every repeated output bit pattern and error;
3. summaries aggregate repeats without embedding a policy;
4. assessments apply each registered policy to the same summary.

The two default policies ask different questions. `consumer_tolerance` requires
absolute relative error at most `1e-6` plus bitwise repeatability;
`correct_rounding` requires every output bit pattern to equal the certified
FP32 target.  Both canonical policy JSON and `policy_id` are recorded.

With the default repeat count of three, smoke execution produces 16 case rows,
192 raw rows, 64 policy-free summaries, and 128 assessments.  CSV writing is
schema-strict: missing or unexpected fields stop artifact generation instead
of being silently discarded.

### Midpoint controls

For stored FP32 values near one, the rounding midpoint between `0x3f800000`
and `0x3f800001` is $1+2^{-24}$.  The small tie control
`N=1,e=-24` correctly rounds to `0x3f800000` by ties-to-even.  The scaled
boundary family fixes `e=-34`:

| Tail count | Exact position | Correct FP32 target |
| ---: | --- | --- |
| 1023 | below midpoint | `0x3f800000` |
| 1024 | exact midpoint | `0x3f800000` |
| 1025 | minimally above midpoint | `0x3f800001` |

At `N=1025`, sequential FP32 fails only for `head_then_tail`; tail-first
accumulation reaches the target.  The current fixed pairwise tree returns
`0x3f800000` in both layouts, while compensated FP32 and the FP64 accumulator
return `0x3f800001` in both layouts.  This is a statement about the registered
implementations and inputs, not a universal ranking of summation methods.

The decimal controls distinguish requested source values from stored FP32
inputs.  For `t=1e-8`, `N=5` lies below the midpoint and all candidates return
`0x3f800000`.  At `N=6`, the correctly rounded stored-sum target is
`0x3f800001`; wrong one-ULP outputs still satisfy the `1e-6` consumer tolerance
but fail the correct-rounding policy.  This is why tolerance and
correct-rounding decisions remain separate artifacts.

From the repository root in PowerShell:

    python topics/softmax/experiments/softmax_failure_triage_runner.py --include-stress --output-dir "$env:TEMP\error-atlas-softmax-triage-v3"

Outputs:

- `softmax_failure_triage_cases.csv`
- `softmax_failure_triage_runs.csv`
- `softmax_failure_triage_summary.csv`
- `softmax_failure_triage_assessments.csv`
- `softmax_failure_triage_metadata.json`

The paths above are filenames within the selected output directory.  Always use
a new scratch directory while auditing; do not overwrite checked-in evidence
implicitly.

For the $2^{20}$-tail stress case, sequential FP32 returns $1$ and loses
the full tail mass $1/16$, giving absolute relative error $1/17$.  The
other three registered candidates return the exact FP32 value $17/16$ for
this case.  This establishes numerical eligibility only; performance and
resource ranking remain target-implementation measurements.

## Exact graph semantic oracle

`summation_graph_predictor.py` defines an exact graph semantic oracle without
calling NumPy arithmetic.  It is intended as a label generator for later
predictor evaluation, not as a cheap engineering score.  For exact stored FP32
leaves and an explicit binary addition tree, it recursively computes

$$
a_v=y_{\ell(v)}+y_{r(v)},
\qquad
y_v=\operatorname{RN}_{32}(a_v),
\qquad
\rho_v=y_v-a_v.
$$

The primary oracle quantity is the exact signed forward error

$$
E_G(x)=y_{\mathrm{root}}-\sum_i x_i=\sum_{v\in G}\rho_v.
$$

It depends only on the ordered stored input, the explicit graph and FP32
round-to-nearest-even semantics.  Before a candidate runs, the oracle produces
both $E_G$ and the exact semantic output bits.  Either a bit mismatch or a
signed error mismatch falsifies the claimed graph/dtype/rounding contract.  The
sequential graph canonically removes the exact initial `0 + x[0]` identity
operation; every potentially inexact addition remains explicit.

### Accepted stepwise oracle-conformance validation

The research review now accepts four deliberately separated steps:

1. **P0 — definition**: distinguish the source sum, exact stored-leaf sum
   $S_{\mathrm{leaf}}$, correctly rounded target, predicted root, and signed
   forward error $E_G$.
2. **P1 — RN32 semantics**: audit normal odd/even ties, the subnormal tie,
   the subnormal/normal boundary, binade carry, and the finite-domain contract.
3. **P2 — minimal graph discriminator**: use $(N,e)=(9,-27)$ tail-first, then
   change only the layout to head-first.
4. **P3 — boundary and scale controls**: use the tail-first $N=7/8/9$ triple
   at $e=-27$, then the above-midpoint points $(129,-31)$ and $(2049,-35)$.

P1 is backed by the four targeted `ExactFP32RoundingTests`.  P2 and P3 select
12 rows from the batch CSV:

- `k=3`, tail-first, `N in {7,8,9}`, both graphs;
- `k=3`, head-first, `N=9`, both graphs;
- `(k,N)=(7,129)` and `(11,2049)`, tail-first, both graphs.

For these rows, predicted bits and exact signed errors match the observations.
The review also established that prediction agreement and candidate correctness
are separate facts: an oracle may accurately predict a candidate output that
fails correct rounding.

The current batch schema versions only prediction agreement.
`preregistered_sum_bits` is the graph-specific pre-run output prediction, not
the correctly rounded target.  A future correct-rounding assessment must add a
separate target and decision field; it must not reinterpret this column.

### Provisional batch evidence

The existing scaled-midpoint batch uses

$$
k\in\{3,7,11\},\qquad
u_k=2^{-(24+k)},\qquad
N\in\{2^k-1,2^k,2^k+1\},
$$

with both layouts and both graphs.  All 36 rows matched, but the 24 rows outside
the accepted selector above are **provisional batch replication**, not a single
completed research rung.  The raw CSV is retained rather than deleted or
rewritten.

Two nonuniform positive-tail permutations were also examined interactively.
They clarified how negative branch residuals and positive root residuals can
combine, and how one permutation can change fixed-pairwise correct rounding.
They are not present in a versioned artifact and must not be described as
preregistered accepted evidence.

### Accepted P4 single nonuniform case

After the pilot, a different case was preregistered before execution:

$$
u=2^{-27},
\qquad
x=(4u,2u,2u,u,1),
\qquad
S_{\mathrm{leaf}}=1+9u.
$$

The preregistration freezes exact leaves, explicit graph nodes, the correct
target `0x3f800001`, graph-specific output bits, signed errors, and expected
correct-rounding decisions.  Its SHA-256 is
`0c8047f38363c02ef6a6995bcc58a3f890dfebe7395aa5eb62a5cb671d6e47a6`.

One execution of each graph produced:

| Graph | Bits | $E_G$ | Prediction matched | Correctly rounded |
| --- | --- | --- | --- | --- |
| sequential left-to-right | `0x3f800001` | $+7u$ | yes | yes |
| balanced contiguous floor-half | `0x3f800000` | $-9u$ | yes | no |

The observation CSV now versions predicted graph output and correctly-rounded
target as independent fields.  This is accepted evidence for one case only;
no nonuniform-family generalization is claimed.

Each graph was executed once.  This is sufficient to expose a mismatch with
the frozen prediction if one occurs, but it does not measure repeatability.
Adding repeated raw runs, bit counts, and a repeatability decision here would
duplicate the existing failure-triage evidence layers.  If a consumer later
requires repeatability, route this case through that pipeline instead.  A
regression test that happens to call the candidate again is not a versioned
repeatability artifact.

Run only this preregistered case from the repository root:

    python topics/softmax/experiments/nonuniform_graph_predictor_v1_runner.py

Outputs are under `results/nonuniform_graph_predictor_v1/`.

The oracle is agent-authored for this requested validation; learner
closed-book mastery is not claimed.  Its current contract covers nonnegative
finite FP32 leaves, no overflow, one RN-even rounding per binary addition and
proper trees.  It does not yet cover Kahan operations, negative cancellation,
FMA, unknown library reductions or GPU graphs.  Matching a finite probe can
reject a graph hypothesis when it fails, but cannot uniquely infer a black-box
graph when it succeeds.

Run from the repository root:

    python topics/softmax/experiments/summation_graph_predictor_validation.py

This command reproduces the retained 36-row provisional batch.  It is an
artifact-reproduction command, not the next stepwise research action.

Outputs:

- `results/graph_predictor_validation/summation_graph_predictor_validation.csv`
- `results/graph_predictor_validation/summation_graph_predictor_validation_metadata.json`

### Depth-margin screening baseline: confirmed topology counterexample

A separate exploratory line asked whether the cheaper score

$$
D_G=\sum_i d_i|x_i|,
\qquad
R_G=\frac{\varepsilon_{32}D_G}{M},
$$

could rank correct-rounding failure before exact graph replay.  It is a
screening score only: it is not a probability, threshold, error bound, or
certificate.

The first frozen family moved the unit head between shallow and deep positions.
Before execution it was reclassified as known-mechanism calibration because
the predicted ordering merely restated the already-known head-depth effect.
Its immutable preregistration remains unexecuted under
`results/graph_risk_proxy_v1/`.

The next challenge held $S_{\mathrm{leaf}}$, margin $M$, every leaf depth,
$D_G$, and $R_G$ fixed while changing only sibling grouping.  For

$$
x=(1,32q,q/2,q/2),\qquad q=2^{-29},
$$

the two graphs both produced
$R_G=536870945/8388608$, but the exact semantic oracle confirmed

| Grouping | Oracle bits | $E_G$ | $F_G$ |
| --- | --- | --- | --- |
| $(1+32q)+(q/2+q/2)$ | `0x3f800000` | $-33q$ | 1 |
| $(1+q/2)+(32q+q/2)$ | `0x3f800001` | $+31q$ | 0 |

This is a tied informative pair, so it falsifies the registered strong claim
that every informative pair has $R_{\mathrm{failure}}>R_{\mathrm{correct}}$.
It establishes that leaf depth plus boundary margin discards relevant sibling-
grouping and rounding-phase information.  It does not measure population-level
ranking quality and does not weaken the exact semantic graph oracle.

Artifacts are under `results/depth_margin_topology_challenge_v1/`.  The one-shot
runner independently recomputed the proxy before calling the exact oracle and
did not execute a NumPy candidate.  It refuses to overwrite the recorded
artifacts and is not a routine regression command.

The checked-in `results/failure_triage/` files are the earlier 2026-08-08
artifact snapshot and have intentionally not been regenerated during this
source/test expansion.  They must not be cited as evidence for suite v3 until
an intentional regeneration and review replaces them.

The complete artifact layout, including the two intermediate summation
experiments, is indexed in `results/README.md`.
