# Wide-range fixed-K8 beam v2

Status: **completed**.

This stage promotes the fixed-K8/B3 baseline only after observing the completed v1 outcomes.  The
adaptation is disclosed in the preregistration, and the v1 held-out groups are not reused for
training, threshold selection, or v2 evidence.

The frozen confirmation is deliberately narrow:

- macro score: fixed connected root-band `Q_8 / 12` over all 64 trees;
- baseline: select the global minimum macro score;
- candidate: shortlist the four lowest macro scores and rerank them with the unchanged B=3 joint
  cell beam;
- held-out strata: widths 256, 512, and 1024, with 64 new input groups per width;
- primary estimand: paired normalized-regret improvement of beam over fixed-K8 Q-only.

The evaluation graph seeds use a new namespace, and the input seeds are explicitly frozen and
disjoint from calibration and v1 held-out inputs.  The runner refuses a dirty worktree and creates
`heldout/` exactly once.  An interrupted or partial directory is forensic evidence and must not be
deleted for a silent rerun.

This stage validates ranking utility, not production runtime.  Its evidence-generation runner still
contains exact `Fraction` and oracle instrumentation.  The subsequent
`wide_range_fixed_k8_beam_score_only_v1` milestone reproduced all 192 frozen decisions and scores
exactly without those operations, then measured the remaining inference cost separately.

## Frozen outcome

The preregistered overall comparison passed its positive-evidence rule:

- paired normalized-regret improvement, `regret(Q-only) - regret(beam)`: `+0.057699`;
- stratified group-bootstrap 95% interval: `[+0.018713, +0.097991]`;
- beam versus Q-only best-tier hit: `83.33%` versus `70.83%`;
- beam versus Q-only mean normalized regret: `0.07421` versus `0.13191`;
- beam versus Q-only severe-regret rate: `5.73%` versus `11.46%`.

The pooled, width-stratified comparison is the sole confirmatory claim.  Width 256 was individually
positive, while the descriptive width-512 and width-1024 intervals crossed zero.  The result does
not establish cross-family generality or production runtime cheapness.

## Engineering follow-up

The score-only implementation is 27--40x faster than evaluating all 64 candidates with the Python
`Fraction` oracle, but it still requires 64 macro traversals plus four full shadow traversals.  Its
measured cost was 50--91 Python FP32 tree reductions, depending on width.  The efficacy conclusion
therefore survives unchanged, while a per-vector production-cheap claim does not yet follow.  See
`../wide_range_fixed_k8_beam_score_only_v1/README.md` for fidelity, timing, and amortization limits.
