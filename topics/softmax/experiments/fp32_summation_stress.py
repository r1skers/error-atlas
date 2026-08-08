"""Isolate FP32 summation loss before end-to-end Softmax experiments.

Prediction record for ``q = (1, u, ..., u)``, with ``u = 2**-24``:

- Direction: explicit large-first FP32 accumulation underestimates the sum.
- Scale: for ``k`` tail terms, the absolute error is ``-k*u`` and the
  relative error is ``-k*u / (1 + k*u)``.
- Boundary: the prediction assumes a fixed left-to-right loop, an FP32
  accumulator, rounding after every addition, and round-to-nearest-even.
- Failure signature: every tail term is nonzero, the explicit FP32 sum stays
  at one, and the FP64 reference retains the full tail mass.

For ``q = (1, u, 1, u, ..., 1, u)`` with ``m`` adjacent pairs, the fixed tree
loses ``u`` at every lowest-level pair.  Its predicted absolute error is
``-m*u`` and its relative error is ``-u / (1 + u)``.  This probe breaks exact
tree summation without claiming to attain the worst-case tree-depth bound.

The controlled permutation pair ``(1, u, u)`` and ``(u, 1, u)`` keeps the
multiset, length, dtype, and fixed tree rule unchanged.  It isolates input
order: the first tree returns ``1 + 2*u`` while the second returns ``1``.

For compensated summation, ``(u, 1, u)`` is the minimal registered probe.  A
standard Kahan correction stores the first lost ``u`` as ``c = -u`` and feeds
it back through ``x - c``, so the predicted result is ``1 + 2*u``.

The explicit sequential accumulator is learner-owned research core.  NumPy
reductions will be measured later as separate implementation strategies rather
than treated as evidence for a particular internal reduction order.
"""

import csv
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FP32_UNIT_ROUNDOFF = np.float32(2.0**-24)
DEFAULT_TAIL_COUNT = 2**20

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "summation_permutation"
CSV_PATH = RESULTS_DIR / "fp32_summation_permutation.csv"
METADATA_PATH = RESULTS_DIR / "fp32_summation_permutation_metadata.json"
CSV_FIELDS = (
    "dataset",
    "method",
    "element_count",
    "reference_sum",
    "predicted_sum",
    "computed_sum",
    "absolute_error",
    "relative_error",
    "prediction_matched",
)


@dataclass(frozen=True)
class SequentialSummationProbe:
    """Measurements that distinguish summation loss from prior underflow."""

    tail_count: int
    nonzero_tail_count: int
    computed_sum: float
    reference_sum: float
    absolute_error: float
    relative_error: float


def adversarial_terms(tail_count: int) -> np.ndarray:
    """Return FP32 ``(1, u, ..., u)`` with ``tail_count`` copies of ``u``."""
    if tail_count < 0:
        raise ValueError("tail_count must be nonnegative")

    values = np.full(
        tail_count + 1,
        FP32_UNIT_ROUNDOFF,
        dtype=np.float32,
    )
    values[0] = np.float32(1.0)
    return values


def alternating_terms(pair_count: int) -> np.ndarray:
    """Return FP32 ``(1, u, 1, u, ..., 1, u)`` with fixed adjacent pairs."""
    if pair_count < 0:
        raise ValueError("pair_count must be nonnegative")

    values = np.empty(2 * pair_count, dtype=np.float32)
    values[0::2] = np.float32(1.0)
    values[1::2] = FP32_UNIT_ROUNDOFF
    return values


def sequential_sum_fp32(values: np.ndarray) -> np.float32:
    """Accumulate ``values`` left-to-right with FP32 rounding at every step.

    Research-core invariants:
    - Reject non-FP32 input rather than silently changing the experiment.
    - Keep the accumulator in FP32 throughout the loop.
    - Preserve input order and perform exactly one scalar addition per item.
    - Return an FP32 scalar so the arithmetic contract is inspectable.
    """
    if values.dtype != np.float32:
        raise TypeError("values must have dtype float32")
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    accumulator = np.float32(0.0)
    for value in values:
        accumulator = np.float32(accumulator + value)
    return accumulator


def pairwise_sum_fp32(values: np.ndarray) -> np.float32:
    """Sum a one-dimensional FP32 array with a fixed balanced binary tree.

    Research-core invariants:
    - Reject non-FP32 or non-vector input.
    - Preserve the contiguous input order.
    - Split every interval after its first ``length // 2`` elements, so the
      tree is fixed even when the number of leaves is odd.
    - Perform every internal-node addition in FP32.
    - Return FP32 zero for an empty input and an FP32 scalar otherwise.
    - Do not delegate the reduction to ``np.sum`` or another black box.
    """
    if values.dtype != np.float32:
        raise TypeError("values must have dtype float32")
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    size = values.size
    if size == 0:
        return np.float32(0.0)
    if size == 1:
        return values[0]

    mid = size // 2
    left_sum = pairwise_sum_fp32(values[:mid])
    right_sum = pairwise_sum_fp32(values[mid:])

    return np.float32(left_sum + right_sum)


def compensated_sum_fp32(values: np.ndarray) -> np.float32:
    """Sum a one-dimensional FP32 array with standard Kahan compensation.

    Research-core invariants:
    - Reject non-FP32 or non-vector input.
    - Keep the running sum, corrected input, temporary sum, and signed
      rounding-error state in FP32.
    - Interpret ``c`` as actual increment minus requested increment.
    - Compute ``c = (t - s) - y`` while ``s`` is still the old running sum;
      only then update ``s = t``.
    - Preserve input order and return an FP32 scalar.
    """
    if values.dtype != np.float32:
        raise TypeError("values must have dtype float32")
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    s = np.float32(0.0)
    c = np.float32(0.0)

    for x in values:
        y = np.float32(x - c)
        t = np.float32(s + y)
        c = np.float32((t - s) - y)
        s = t
    return s


def fp64_reference_sum(values: np.ndarray) -> float:
    """Return the higher-precision reference for already-stored FP32 terms."""
    if values.dtype != np.float32:
        raise TypeError("values must have dtype float32")
    return float(np.sum(values, dtype=np.float64))


def sequential_summation_probe(
    tail_count: int = DEFAULT_TAIL_COUNT,
) -> SequentialSummationProbe:
    """Measure the pre-registered large-first summation failure signature."""
    values = adversarial_terms(tail_count)
    computed_sum = float(sequential_sum_fp32(values))
    reference_sum = fp64_reference_sum(values)
    absolute_error = computed_sum - reference_sum

    return SequentialSummationProbe(
        tail_count=tail_count,
        nonzero_tail_count=int(np.count_nonzero(values[1:])),
        computed_sum=computed_sum,
        reference_sum=reference_sum,
        absolute_error=absolute_error,
        relative_error=absolute_error / reference_sum,
    )


def controlled_permutation_rows() -> list[dict[str, str | int | float | bool]]:
    """Return the six pre-registered same-multiset permutation probes."""
    one = np.float32(1.0)
    u = FP32_UNIT_ROUNDOFF
    exact_fp32_sum = float(np.float32(one + np.float32(2.0) * u))
    cases = {
        "favorable_1_u_u": np.array([one, u, u], dtype=np.float32),
        "unfavorable_u_1_u": np.array([u, one, u], dtype=np.float32),
    }
    methods = {
        "sequential": sequential_sum_fp32,
        "pairwise": pairwise_sum_fp32,
        "compensated": compensated_sum_fp32,
    }
    predictions = {
        ("favorable_1_u_u", "sequential"): 1.0,
        ("favorable_1_u_u", "pairwise"): exact_fp32_sum,
        ("favorable_1_u_u", "compensated"): exact_fp32_sum,
        ("unfavorable_u_1_u", "sequential"): 1.0,
        ("unfavorable_u_1_u", "pairwise"): 1.0,
        ("unfavorable_u_1_u", "compensated"): exact_fp32_sum,
    }

    rows = []
    for dataset, values in cases.items():
        reference_sum = fp64_reference_sum(values)
        for method, summation in methods.items():
            predicted_sum = predictions[(dataset, method)]
            computed_sum = float(summation(values))
            absolute_error = computed_sum - reference_sum
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "element_count": int(values.size),
                    "reference_sum": reference_sum,
                    "predicted_sum": predicted_sum,
                    "computed_sum": computed_sum,
                    "absolute_error": absolute_error,
                    "relative_error": absolute_error / reference_sum,
                    "prediction_matched": computed_sum == predicted_sum,
                }
            )
    return rows


def write_controlled_permutation_results(
    rows: list[dict[str, str | int | float | bool]],
) -> None:
    """Write controlled permutation evidence and provenance."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python topics/softmax/experiments/fp32_summation_stress.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "dtype": "float32",
        "unit_roundoff": float(FP32_UNIT_ROUNDOFF),
        "registered_datasets": [
            "favorable_1_u_u",
            "unfavorable_u_1_u",
        ],
        "methods": ["sequential", "pairwise", "compensated"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run and record the controlled same-multiset permutation experiment."""
    rows = controlled_permutation_rows()
    write_controlled_permutation_results(rows)

    print("dataset,method,reference,predicted,computed,relative_error,matched")
    for row in rows:
        print(
            f"{row['dataset']},{row['method']},{row['reference_sum']:.9g},"
            f"{row['predicted_sum']:.9g},{row['computed_sum']:.9g},"
            f"{row['relative_error']:.9g},{row['prediction_matched']}"
        )
    print(f"wrote {CSV_PATH}")
    print(f"wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
