# Wide-range fixed-K8 beam v2

Status: **frozen, not opened**.

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

This stage validates ranking utility, not production runtime.  Its research implementation still
contains exact `Fraction` and oracle instrumentation; a score-only cost implementation is a
separate milestone if the confirmation succeeds.
