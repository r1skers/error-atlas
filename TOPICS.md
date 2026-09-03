# Topic Registry

Updated 2026-09-02. Current research status lives in
[NEXT_SESSION.md](NEXT_SESSION.md); this file is only the topic registry and does not
duplicate the experiment log.

| Topic | Object | Status / entry |
| --- | --- | --- |
| Taylor expansion | Remainder, error bounds, numerical differentiation, bias–variance | [First pass complete](topics/taylor-expansion/README.md) |
| Softmax | Input quantization, normalization, reduction error, risk prediction | [Active: after online-certificate calibration](topics/softmax/README.md) |

## Reading paths

- Taylor: [theory notes](topics/taylor-expansion/notes/00_error_language.md) →
  [experiment notes](topics/taylor-expansion/experiments/README.md).
- Softmax: [foundations & exact-oracle derivation](topics/softmax/notes/foundations.md) →
  [experiment code index](topics/softmax/experiments/README.md) →
  [staged evidence](topics/softmax/experiments/results/README.md).
- Learning primer: [full knowledge map](KNOWLEDGE_MAP.md); teaching text, not a
  substitute for frozen evidence.
- Old research handoff: [2026-08-12 snapshot](docs/history/2026-08-12-softmax-handoff.md).

## Minimal template for a new topic

Start with only the topic README, answering eight points:

1. Object: what is studied.
2. Reference: what error is defined against.
3. Metric: how it is measured.
4. Sources: where error enters.
5. Propagation: which structures change the error.
6. Control: the adjustable mechanisms.
7. Optimization: the accuracy–cost tradeoff.
8. Verification: proof, counterexample, or reproducible experiment.

Add notes, experiments, and tests only once the content actually exists. Research
discipline is in the [error-analysis protocol](framework/error_analysis_protocol.md).
