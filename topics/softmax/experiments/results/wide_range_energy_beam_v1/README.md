# Wide-range energy-beam v1

This directory contains the frozen preregistration for the narrow, wide-range-only validation of
the selected reduction-tree cascade.  It does not complete or supersede the broader multi-family
validation scaffold.

The frozen policy is:

- macro shortlist: lowest four 80%-energy-mass Q scores;
- micro reranker: joint ancestor-cell beam with width three;
- calibration model: width-256 seeds 22260821--22260824 only;
- held-out strata: widths 256, 512, and 1024, with 32 input groups per width;
- primary estimand: group-paired normalized-regret improvement over Q-only selection.

`predictor_wide_range_energy_beam_v1_heldout.py` refuses to run unless its output directory is
absent and the Git worktree is clean.  It creates `heldout/` once.  A partial output directory after
an interrupted run is intentionally blocking and must be preserved for forensic inspection; it may
not be deleted and rerun without a separately committed recovery note.
