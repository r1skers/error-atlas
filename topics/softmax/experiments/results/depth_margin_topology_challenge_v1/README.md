# Depth-margin topology challenge v1

## Result

This directory records a confirmed adversarial counterexample to the universal
matched-pair ordering claim for the depth-margin screening score

$$
R_G=\frac{\varepsilon_{32}\sum_i d_i|x_i|}{M}.
$$

The preregistration was frozen before machine proxy calculation or exact-oracle
execution. Its SHA-256 is
`63fbe76d7af9f5df128c1fdfade6ca49fc1d4e26c472ea9914888a3ee28f9949`.
The stored leaves are

$$
x=(1,32q,q/2,q/2),\qquad q=2^{-29},
$$

so both explicit trees have the same $S_{\mathrm{leaf}}=1+33q$, rounding
margin $M=q$, leaf depths $(2,2,2,2)$, depth exposure
$D_G=2(1+33q)$, and score

$$
R_G=\frac{536870945}{8388608}.
$$

Only sibling grouping differs:

| Graph | Grouping | $R_G$ | Oracle bits | $E_G$ | $F_G$ |
| --- | --- | --- | --- | --- | --- |
| `midpoint_then_remainder` | $(1+32q)+(q/2+q/2)$ | `536870945/8388608` | `0x3f800000` | $-33q$ | 1 |
| `half_q_then_midpoint_plus_half_q` | $(1+q/2)+(32q+q/2)$ | `536870945/8388608` | `0x3f800001` | $+31q$ | 0 |

The pair is informative because its correct-rounding labels differ, while the
registered proxy scores tie. This confirms the preregistered falsifier and
rejects the strong claim that every informative matched pair satisfies
$R_{\mathrm{failure}}>R_{\mathrm{correct}}$.

## Evidence boundary

- This is a constructed topology counterexample, not population-level
  validation or an estimate of average screening quality.
- It falsifies the universal ordering claim for this depth-only score. It does
  not show that every depth/margin feature is useless, nor does it falsify the
  exact semantic graph oracle.
- The independent proxy phase derived depths from the frozen raw edges and
  computed $D$, the fixed near-one $M$, and $R$ without calling FP32 rounding
  helpers. The exact semantic oracle was invoked only after those values matched
  the preregistration.
- No NumPy summation candidate was executed. Candidate conformance and
  repeatability are not claims of this artifact.
- `depth_margin_topology_challenge_v1_runner.py` refuses to overwrite the
  checked-in observation and metadata files; do not rerun it as a routine test.

The two-row CSV contains the proxy quantities and exact-oracle labels. Metadata
records the preregistration, source, and CSV hashes plus the confirmed
`pair_outcome=tie` and `strong_hypothesis_falsified=true` decisions.
