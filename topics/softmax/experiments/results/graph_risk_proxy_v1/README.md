# Depth-margin risk proxy v1

## Evidence status

`depth_margin_risk_v1_preregistration.json` is an immutable pre-execution
record. Its SHA-256 at registration was
`1667717ae158479dd8753563f4b3bab2bb95608dea220e49506b208962946c72`.

On 2026-08-12, before either the exact graph oracle or a NumPy candidate was
run on the registered cases, the family was reclassified as
**known-mechanism calibration / optional replication**. It is not the primary
statistical predictor-validation family.

The registered numeric inputs were unexecuted, but the tested mechanism was
not conceptually held out: moving the unit head between shallow and deep leaf
positions directly changes the dominant term in
`D_G = sum_i d_i * abs(x_i)`. Recomputing the frozen risk ordering would test
calculator conformance, while agreement with later labels would provide only
narrow replication of the already-known head-depth effect.

The preregistration remains `preregistered_not_executed`; do not edit it to
encode this later interpretation. This README is the separate decision record.
The current research line should instead hold `S_leaf`, rounding margin `M`,
and depth exposure `D_G` fixed while changing explicit sibling grouping. Such
a challenge targets graph information discarded by the depth-only proxy.

## Execution boundary

- No exact-oracle label has been obtained for either registered layout.
- No NumPy summation candidate has been executed for either registered layout.
- No observation CSV or validation metadata exists for this calibration family.
- If it is executed later, report it as calibration/replication rather than as
  the main predictor-validation milestone.
