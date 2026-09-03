# Error Atlas

> An experiment-driven study of how approximation and finite-precision error is
> defined, propagated, estimated, controlled, and traded off against cost.

## Abstract

Error Atlas investigates a single question across mathematical and computational
objects: how does error enter a computation, how does structure propagate it, and
what can be predicted or controlled before it happens. The main line of work studies
**FP32 rounding error in the summation reduction trees** that form the denominator of
a softmax. Given stored FP32 inputs and an explicit addition tree, an exact rational
oracle reproduces the hardware result bit-for-bit and attributes the final error to
each node, enabling controlled study of how the *shape* of the reduction changes the
error. Every stage is preregistered, its evidence is frozen and versioned, negative
results are recorded as first-class outcomes, and the headline confirmation has been
independently reproduced from a blank-slate reimplementation.

## Headline findings

- **The final reduction error is dominated by the sign coherence between local
  rounding errors, not by their magnitude.** Writing E² = A + C (local energy plus
  pairwise cross term), the tree-to-tree spread of the error is driven by C: across
  trees on a fixed input, the standard deviation of C is 2.5–3.6× that of A.
- **A cheap magnitude-only score cannot rank trees reliably**, because it estimates
  only A and is blind to the coherence term C that actually separates good trees from
  bad ones. This is a confirmed negative result, not a tuning failure.
- **A coherence-aware beam narrowly beats the cheap score** on a preregistered,
  frozen synthetic distribution: paired normalized-regret improvement +0.058, 95%
  bootstrap CI [+0.019, +0.098]. The win is genuine but narrow (per-width intervals
  for widths 512 and 1024 cross zero) and its inference cost is not yet cheap enough
  for production; an offline-reuse variant failed its preregistered deployment gate,
  and an online risk certificate reached calibration only.
- **Net:** magnitude-only tree selection is infeasible; making selection feasible
  requires paying to observe the coherence term, and that cost is not yet low enough
  to deploy. This motivated the shift from *ranking trees* toward *carrying a risk
  state* alongside a single reduction.

## What makes it rigorous

- **Exact oracle.** An integer/rational (`Fraction`) implementation of round-to-nearest,
  ties-to-even reproduces hardware binary32 addition exactly, including subnormals and
  carry, verified against NumPy float32 over hundreds of thousands of cases.
- **Preregistration and frozen evidence.** Each stage freezes its protocol, seeds, and
  budgets before execution; artifacts are versioned CSV/JSON with SHA-256 provenance and
  are never silently overwritten. See the [results index](topics/softmax/experiments/results/README.md).
- **Honest negatives.** Depth-margin screening, energy-beam v1, offline tree reuse, and
  the online certificate are all recorded with their exact evidence grade, including the
  ones that failed.
- **Independent replication.** The confirmed pipeline was re-implemented from a blank
  skeleton and reproduced the frozen headline bit-for-bit, catching four implementation
  bugs and one source-vs-artifact drift along the way. See the
  [replication notes](topics/softmax/notes/rewrite_replication.md).

## Why it matters for systems

The reduction-tree object is the kernel of online/blockwise softmax and, ultimately,
attention accumulators: the same rounding coherence that this work isolates in plain
FP32 summation reappears, weighted by online rescaling factors, in the (m, ℓ) state of
FlashAttention-style kernels. The exact-oracle-plus-preregistration method is designed
to extend to that setting.

## Repository guide

| Topic / stage | Status |
| --- | --- |
| [Taylor expansion](topics/taylor-expansion/README.md) | First pass complete: derivations, experiments, closed-book rewrite |
| [Softmax foundations & exact graph oracle](topics/softmax/notes/foundations.md) | First pass complete; early evidence grades preserved |
| Fixed-K8/B3 tree ranking | Confirmed on a controlled distribution; inference cost still high |
| Offline tree reuse | Beats a random fixed tree but fails the balanced-FP32 deployment gate (no-go) |
| Online risk certificate | Calibration complete; statistical signal, no confirmation or deployment claim |

```text
framework/                 research discipline and the implementation-learning protocol
docs/                      maintenance guide and historical handoffs
tools/                     the single test entry and its own tests
topics/<topic>/
    README.md              topic entry
    notes/                 theory and research notes
    experiments/           source, frozen protocols, results/ evidence, rewrite/ replication
    tests/                 regression tests
```

- [Softmax experiment index](topics/softmax/experiments/README.md) — find code by role.
- [Results index](topics/softmax/experiments/results/README.md) — confirmed results,
  negatives, and calibration observations, each with its evidence boundary.
- [Replication partition](topics/softmax/experiments/reduction_analysis/README.md) and
  the [rewrite package](topics/softmax/experiments/rewrite/README.md) — the independent
  reimplementation and its differential tests.
- [KNOWLEDGE_MAP.md](KNOWLEDGE_MAP.md) — a standalone teaching text; learning material,
  not a source of current research status.
- [NEXT_SESSION.md](NEXT_SESSION.md) — current status and the next research entry.

## Reproducing the checks

Requires Python 3.10+; dependencies in [requirements.txt](requirements.txt).

```sh
python -m pip install -r requirements.txt
python tools/run_tests.py
python tools/run_tests.py --suite softmax -v
```

The test entry runs regression and replication tests only; it never re-runs the
one-shot experiment CLIs that publish frozen artifacts. Before any intentional
reproduction, read the specific stage's results README and preregistration.

## Method conventions

Fix the reference, metric, assumptions, and error sources first; then study bounds,
propagation, and control. Follow the
[error-analysis protocol](framework/error_analysis_protocol.md) and the
[implementation-learning protocol](framework/implementation_learning_protocol.md):
record predictions before running, preserve raw data and provenance, and separate
implementation, numerical, measurement, and statistical error. Topic registry:
[TOPICS.md](TOPICS.md). Structure and evidence-preservation rules:
[maintenance guide](docs/maintenance.md).
