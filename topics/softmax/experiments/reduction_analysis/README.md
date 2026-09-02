# Shared reduction analysis

This package turns the first three coherence diagnostics into views of one trace:

- `trace.py`: call the existing exact oracle once and retain its NodePrediction records;
- `topology.py`: share immutable parent/depth information and proper-ancestor queries;
- `coherence.py`: lazily compute A/C, structural pair partitions and ancestor/history.

It does not generate inputs, pick seeds, execute a candidate, train a predictor, or write results.

## Compose the views

Inside an experiment script, with its existing stored `values` and explicit `graph`:

```python
from reduction_analysis import CoherenceAnalysis, replay

trace = replay(values, graph)
analysis = CoherenceAnalysis(trace, graph_family=family, graph_seed=seed)

energy = analysis.ac          # exact E² = A + C; linear prefix identity audit
structure = analysis.structure  # float parent / far-ancestor / disjoint partition
history = analysis.history   # exact C_ancestor = 2 sum(delta_v * H_v)
```

Retain the same analysis object when composing views. It caches A/C and each requested view;
the trace caches topology and residual accessors. There is no global cache across cases.
Requesting only A/C does not construct topology or run the pair scans.
Pair scans can be expensive: the current implementations enumerate node pairs and walk ancestry;
they are diagnostic algorithms, not a low-cost inference claim.

Use `replay()` to construct a trace. `trace.values` and `trace.graph` describe the actual case;
graph family and seed on the analysis are report labels, not identity validation.
`trace.value_at(i)` returns a stored leaf or rounded internal output. It is not an exact
unrounded shadow-subtree sum; future phase/history adapters must keep that distinction.

## Boundaries preserved

- RN32 semantics remain solely in [summation_graph_predictor.py](../summation_graph_predictor.py).
- The trace contains oracle labels. Do not feed it to the cheap predictor / score-only path.
- Candidate execution remains independent of the oracle; shared execution cannot validate itself.
- A/C and ancestor-history identities remain Fraction-exact.
- Structural partition accumulation preserves the legacy float order and tolerance.
- Historical ancestor statistics retain their zero-based average ranks and NaN conventions.
  This extraction does not silently standardize all repository statistics.
- Seed schedules, calibration/held-out splits and CLI/reporting remain in the experiment wrappers.
  The ancestor runner still uses a different seed namespace from the A/C and structure runners.

## Compatibility and version boundary

The original `diagnose_tree(values, graph, ...)` / `diagnose(values, graph, family)`
entry points and result types remain available in the three old script modules. They are
adapters to this package; existing CLI output remains unchanged.

The pre-extraction implementation is preserved by Git revision `ad1fe87`.
The [characterization fixture](../../tests/fixtures/coherence_pre_refactor.json) was captured
before replacing the implementations. It records 18 exact input/tree cases, Fraction outputs,
float hex values (including NaN conventions), and three deterministic CLI transcripts.
The fixture records its capture environment and original source Git blobs.
Its float-hex checks characterize that environment, not universal cross-platform bitwise
repeatability. It is regression data, not new calibration/confirmation evidence.

Run `python tools/run_tests.py --suite softmax -p test_reduction_analysis.py`
from the repository root. The tests also ensure one oracle call serves all three views.

No archived results or preregistrations were regenerated. Future experiments using this package
must record its source files in their provenance; an old wrapper's hash alone no longer captures
the implementation. To reproduce the old source boundary exactly, use the recorded Git revision.
