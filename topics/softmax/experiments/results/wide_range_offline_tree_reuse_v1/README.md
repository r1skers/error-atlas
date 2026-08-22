# Wide-range offline tree reuse v1

Status: **preregistered, not opened**.

This stage tests a cost-first deployment contract for the frozen fixed-K8/B3 selector.  A fixed
64-tree catalog is shared across inputs at each width.  The score is evaluated on 32 fresh
representative calibration inputs, one tree is selected per width by mean complete cascade rank,
and that tree is reused on 64 fresh confirmation inputs with no online selection work.

The score and innovation model are unchanged.  The primary comparison is against the exact expected
regret of a random fixed catalog tree.  Balanced FP32 is a separate engineering gate, and an
oracle-static calibration policy is retained only as a ceiling.  Direct FP64-then-FP32 and Kahan
FP32 baselines determine whether tree selection is useful relative to simpler computation advice.

The one-shot runner must be executed only from the clean commit that freezes the accompanying
preregistration.  An existing partial confirmation directory may not be overwritten.
