# Synthetic calibration preflight

Pipeline/timing preflight only. These small-sample counts are not coverage estimates suitable for method selection.
Both arms share each input family; they are not independent trials. No softmax confirmation data were generated.

| Population | n | Outer trials | False positives separate/FMA | Constant samples separate/FMA | Joint false positives |
| --- | --- | --- | --- | --- | --- |
| mixed_null_positive | 32 | 20 | 1/0 | 0/0 | 1 |
| mixed_null_positive | 128 | 20 | 0/0 | 0/0 | 0 |
| positive_constant | 32 | 20 | 0/0 | 20/20 | 0 |
| positive_constant | 128 | 20 | 0/0 | 20/20 | 0 |
| rare_negative | 32 | 20 | 9/9 | 9/9 | 9 |
| rare_negative | 128 | 20 | 0/0 | 0/0 | 0 |
| rare_positive | 32 | 20 | 0/0 | 9/9 | 0 |
| rare_positive | 128 | 20 | 0/0 | 0/0 | 0 |
| symmetric_corr08 | 32 | 20 | 1/0 | 0/0 | 0 |
| symmetric_corr08 | 128 | 20 | 0/1 | 0/0 | 0 |
| symmetric_identical | 32 | 20 | 0/0 | 0/0 | 0 |
| symmetric_identical | 128 | 20 | 0/0 | 0/0 | 0 |
| zero_constant | 32 | 20 | 0/0 | 20/20 | 0 |
| zero_constant | 128 | 20 | 0/0 | 20/20 | 0 |

Exact per-trial bounds and verdicts: trials.jsonl. Actual outer inputs: inputs.jsonl.
Plan, protocol, source snapshots and file hashes are saved before/with measurement.
Constant observed samples are retained. All planned trials must finish for status=complete.
Bootstrap draws are reconstructed by the recorded local Random/randrange algorithm and checked by SHA-256.
Integrity hashes detect file changes; they are not signatures against malicious rewriting.
