"""End-to-end FP32 Softmax probes with inspectable summation strategies.

The direct ``q`` experiments isolate reduction error.  This module adds the
preceding logit quantization and exp stages while returning every intermediate
needed for error attribution.
"""

import csv
import hashlib
import json
import math
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fp32_summation_stress import (
    FP32_UNIT_ROUNDOFF,
    compensated_sum_fp32,
    fp64_reference_sum,
    pairwise_sum_fp32,
    sequential_sum_fp32,
)


FP32Summation = Callable[[np.ndarray], np.float32]

IDEAL_TAIL_LOGIT = -24.0 * math.log(2.0)
IDEAL_TAIL_EXPONENTIAL = 2.0**-24
IDEAL_DENOMINATOR = 1.0 + 2.0 * IDEAL_TAIL_EXPONENTIAL

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "softmax_summation"
CSV_PATH = RESULTS_DIR / "fp32_softmax_summation.csv"
METADATA_PATH = RESULTS_DIR / "fp32_softmax_summation_metadata.json"
CSV_FIELDS = (
    "dataset",
    "method",
    "ideal_tail_logit",
    "stored_tail_logit",
    "stored_tail_logit_error",
    "ideal_tail_exponential",
    "computed_tail_exponential",
    "tail_exp_relative_error",
    "nonzero_tail_count",
    "ideal_reference_sum",
    "stage_reference_sum",
    "predicted_sum",
    "computed_sum",
    "pre_reduction_error",
    "summation_error",
    "total_error",
    "probability_sum_high_precision",
    "mass_residual",
    "probability_l1_error",
    "total_variation",
    "prediction_matched",
)


@dataclass(frozen=True)
class SoftmaxStageProbe:
    """Inspectable stages of one finite-precision Softmax evaluation."""

    stored_logits: np.ndarray
    shifted_logits: np.ndarray
    exponentials: np.ndarray
    denominator: np.float32
    probabilities: np.ndarray


def softmax_with_summation_fp32(
    logits: np.ndarray,
    summation: FP32Summation,
) -> SoftmaxStageProbe:
    """Evaluate stable FP32 Softmax using the supplied FP32 summation method.

    Research-core invariants:
    - Reject non-FP32, non-vector, or empty logits.
    - Treat input values as already-stored logits; do not silently promote
      them before subtract-max.
    - Keep shifted logits, exponentials, denominator, and probabilities in
      FP32.
    - Apply subtract-max before exp.
    - Use ``summation(exponentials)`` exactly once for the denominator.
    - Return copies of all stages so later diagnostics do not infer them from
      the final probabilities.
    """
    if logits.dtype != np.float32:
        raise TypeError(f"Expected FP32 logits, got {logits.dtype}.")
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got {logits.ndim}D.")
    if logits.size == 0:
        raise ValueError("Expected non-empty logits.")
    stored_logits = logits.copy()
    shifted_logits = stored_logits - np.max(stored_logits)
    exponentials = np.exp(shifted_logits, dtype=np.float32)
    denominator = summation(exponentials)
    probabilities = exponentials / denominator
    return SoftmaxStageProbe(
        stored_logits=stored_logits,
        shifted_logits=shifted_logits,
        exponentials=exponentials,
        denominator=denominator,
        probabilities=probabilities,
    )


def controlled_end_to_end_rows() -> list[dict[str, str | int | float | bool]]:
    """Return the six pre-registered end-to-end permutation probes."""
    ideal_cases = {
        "favorable_0_tail_tail": (
            np.array([0.0, IDEAL_TAIL_LOGIT, IDEAL_TAIL_LOGIT]),
            (1, 2),
        ),
        "unfavorable_tail_0_tail": (
            np.array([IDEAL_TAIL_LOGIT, 0.0, IDEAL_TAIL_LOGIT]),
            (0, 2),
        ),
    }
    methods = {
        "sequential": sequential_sum_fp32,
        "pairwise": pairwise_sum_fp32,
        "compensated": compensated_sum_fp32,
    }
    recovered_sum = float(
        np.float32(1.0) + np.float32(2.0) * FP32_UNIT_ROUNDOFF
    )
    predictions = {
        ("favorable_0_tail_tail", "sequential"): 1.0,
        ("favorable_0_tail_tail", "pairwise"): recovered_sum,
        ("favorable_0_tail_tail", "compensated"): recovered_sum,
        ("unfavorable_tail_0_tail", "sequential"): 1.0,
        ("unfavorable_tail_0_tail", "pairwise"): 1.0,
        ("unfavorable_tail_0_tail", "compensated"): recovered_sum,
    }

    rows = []
    for dataset, (ideal_logits, tail_indices) in ideal_cases.items():
        stored_logits = np.asarray(ideal_logits, dtype=np.float32)
        ideal_weights = np.full(3, IDEAL_TAIL_EXPONENTIAL, dtype=np.float64)
        head_index = next(index for index in range(3) if index not in tail_indices)
        ideal_weights[head_index] = 1.0
        ideal_probabilities = ideal_weights / IDEAL_DENOMINATOR
        for method, summation in methods.items():
            probe = softmax_with_summation_fp32(stored_logits, summation)
            tail_exponential = float(probe.exponentials[tail_indices[0]])
            stage_reference_sum = fp64_reference_sum(probe.exponentials)
            computed_sum = float(probe.denominator)
            predicted_sum = predictions[(dataset, method)]
            pre_reduction_error = stage_reference_sum - IDEAL_DENOMINATOR
            summation_error = computed_sum - stage_reference_sum
            total_error = computed_sum - IDEAL_DENOMINATOR
            computed_probabilities = probe.probabilities.astype(np.float64)
            probability_sum = float(np.sum(computed_probabilities, dtype=np.float64))
            probability_l1_error = float(
                np.sum(
                    np.abs(computed_probabilities - ideal_probabilities),
                    dtype=np.float64,
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "ideal_tail_logit": IDEAL_TAIL_LOGIT,
                    "stored_tail_logit": float(
                        probe.stored_logits[tail_indices[0]]
                    ),
                    "stored_tail_logit_error": float(
                        probe.stored_logits[tail_indices[0]]
                    )
                    - IDEAL_TAIL_LOGIT,
                    "ideal_tail_exponential": IDEAL_TAIL_EXPONENTIAL,
                    "computed_tail_exponential": tail_exponential,
                    "tail_exp_relative_error": (
                        tail_exponential - IDEAL_TAIL_EXPONENTIAL
                    )
                    / IDEAL_TAIL_EXPONENTIAL,
                    "nonzero_tail_count": int(
                        np.count_nonzero(probe.exponentials[list(tail_indices)])
                    ),
                    "ideal_reference_sum": IDEAL_DENOMINATOR,
                    "stage_reference_sum": stage_reference_sum,
                    "predicted_sum": predicted_sum,
                    "computed_sum": computed_sum,
                    "pre_reduction_error": pre_reduction_error,
                    "summation_error": summation_error,
                    "total_error": total_error,
                    "probability_sum_high_precision": probability_sum,
                    "mass_residual": probability_sum - 1.0,
                    "probability_l1_error": probability_l1_error,
                    "total_variation": 0.5 * probability_l1_error,
                    "prediction_matched": computed_sum == predicted_sum,
                }
            )
    return rows


def write_end_to_end_results(
    rows: list[dict[str, str | int | float | bool]],
) -> None:
    """Write end-to-end stage-attribution evidence and provenance."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python topics/softmax/experiments/fp32_softmax_summation.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "dtype": "float32",
        "unit_roundoff": float(FP32_UNIT_ROUNDOFF),
        "ideal_tail_logit": IDEAL_TAIL_LOGIT,
        "ideal_tail_exponential": IDEAL_TAIL_EXPONENTIAL,
        "ideal_denominator": IDEAL_DENOMINATOR,
        "registered_datasets": [
            "favorable_0_tail_tail",
            "unfavorable_tail_0_tail",
        ],
        "methods": ["sequential", "pairwise", "compensated"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run and record the end-to-end controlled permutation experiment."""
    rows = controlled_end_to_end_rows()
    write_end_to_end_results(rows)

    print(
        "dataset,method,tail_exp,stage_reference,computed_sum,"
        "pre_reduction_error,summation_error,total_error,mass_residual,"
        "l1_error,tv,matched"
    )
    for row in rows:
        print(
            f"{row['dataset']},{row['method']},"
            f"{row['computed_tail_exponential']:.9g},"
            f"{row['stage_reference_sum']:.17g},"
            f"{row['computed_sum']:.9g},"
            f"{row['pre_reduction_error']:.9g},"
            f"{row['summation_error']:.9g},"
            f"{row['total_error']:.9g},"
            f"{row['mass_residual']:.9g},"
            f"{row['probability_l1_error']:.9g},"
            f"{row['total_variation']:.9g},"
            f"{row['prediction_matched']}"
        )
    print(f"wrote {CSV_PATH}")
    print(f"wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
