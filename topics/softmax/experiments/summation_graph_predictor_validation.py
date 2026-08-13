"""Check the exact graph oracle on a registered scaled-midpoint batch.

Pre-run prediction record
-------------------------

For ``k in {3, 7, 11}``, set ``u_k = 2**(-(24+k))`` and use one FP32 head
equal to one plus ``N in {2**k-1, 2**k, 2**k+1}`` copies of ``u_k``.
Neither these ``k`` values nor their resulting ``(N, exponent)`` pairs occur
in the failure-triage registry.

Direction:
    * sequential head-first and pairwise in either layout underestimate;
    * sequential tail-first underestimates below/at the midpoint and
      overestimates minimally above it after correct final rounding.
Scale:
    all signed errors have magnitude near ``2**-24``, independent of ``k``.
Boundary:
    predictions assume nonnegative stored FP32 leaves, one RN-even FP32
    rounding at every explicit binary-addition node, and no overflow.
Failure signature:
    any mismatch in output bits or exact signed error falsifies the claimed
    graph/dtype/rounding contract.  In every above-midpoint tail-first case,
    sequential must produce ``0x3f800001`` while pairwise must produce
    ``0x3f800000``; failure to separate them makes the predictor graph-blind.

This script runs the semantic predictor first and only then invokes the two
existing NumPy candidates for observation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from fp32_summation_stress import pairwise_sum_fp32, sequential_sum_fp32
from softmax_failure_triage import fp32_bits_hex, input_hash
from summation_graph_predictor import (
    balanced_reduction_graph,
    predict_fp32_tree_error,
    sequential_reduction_graph,
)


VALIDATION_VERSION = "1"
SCALE_EXPONENTS = (3, 7, 11)
LAYOUTS = ("head_then_tail", "tail_then_head")
GRAPH_NAMES = ("sequential", "pairwise")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "graph_predictor_validation"
CSV_PATH = RESULTS_DIR / "summation_graph_predictor_validation.csv"
METADATA_PATH = RESULTS_DIR / "summation_graph_predictor_validation_metadata.json"
CSV_FIELDS = (
    "validation_version",
    "scale_exponent_k",
    "tail_count",
    "tail_power_of_two_exponent",
    "layout",
    "graph",
    "leaf_count",
    "input_hash",
    "exact_input_sum_fraction",
    "predicted_sum_fraction",
    "predicted_sum_bits",
    "preregistered_sum_bits",
    "computed_sum_bits",
    "predicted_signed_error_fraction",
    "observed_signed_error_fraction",
    "local_error_sum_fraction",
    "inexact_addition_count",
    "predictor_matched_preregistered",
    "predictor_matched_observation",
)


def preregistered_sum_bits(
    *,
    scale_exponent_k: int,
    tail_count: int,
    layout: str,
    graph_name: str,
) -> str:
    """Return the prediction recorded above without executing either method."""
    midpoint_count = 2**scale_exponent_k
    if tail_count not in (
        midpoint_count - 1,
        midpoint_count,
        midpoint_count + 1,
    ):
        raise ValueError("tail_count is outside the registered boundary triple")
    if layout not in LAYOUTS:
        raise ValueError("layout is not registered")
    if graph_name not in GRAPH_NAMES:
        raise ValueError("graph_name is not registered")

    if (
        graph_name == "sequential"
        and layout == "tail_then_head"
        and tail_count == midpoint_count + 1
    ):
        return "0x3f800001"
    return "0x3f800000"


def _stored_fraction_values(
    *,
    tail_count: int,
    tail_power_of_two_exponent: int,
    layout: str,
) -> tuple[Fraction, ...]:
    head = Fraction(1)
    tail = Fraction(2) ** tail_power_of_two_exponent
    tails = (tail,) * tail_count
    if layout == "head_then_tail":
        return (head, *tails)
    return (*tails, head)


def _materialized_fp32_values(values: tuple[Fraction, ...]) -> np.ndarray:
    """Materialize registered power-of-two values after exact prechecks."""
    materialized = np.array([float(value) for value in values], dtype=np.float32)
    round_tripped = tuple(Fraction.from_float(float(value)) for value in materialized)
    if round_tripped != values:
        raise AssertionError("materialized FP32 input differs from exact leaves")
    return materialized


def validation_rows() -> list[dict[str, object]]:
    """Predict first, execute second, and return exact comparison rows."""
    rows: list[dict[str, object]] = []
    for scale_exponent_k in SCALE_EXPONENTS:
        midpoint_count = 2**scale_exponent_k
        tail_exponent = -(24 + scale_exponent_k)
        for tail_count in (
            midpoint_count - 1,
            midpoint_count,
            midpoint_count + 1,
        ):
            for layout in LAYOUTS:
                exact_values = _stored_fraction_values(
                    tail_count=tail_count,
                    tail_power_of_two_exponent=tail_exponent,
                    layout=layout,
                )
                leaf_count = len(exact_values)
                graph_candidates = (
                    (
                        "sequential",
                        sequential_reduction_graph(leaf_count),
                        sequential_sum_fp32,
                    ),
                    (
                        "pairwise",
                        balanced_reduction_graph(leaf_count),
                        pairwise_sum_fp32,
                    ),
                )
                materialized_values = _materialized_fp32_values(exact_values)
                for graph_name, graph, candidate in graph_candidates:
                    prediction = predict_fp32_tree_error(exact_values, graph)
                    expected_bits = preregistered_sum_bits(
                        scale_exponent_k=scale_exponent_k,
                        tail_count=tail_count,
                        layout=layout,
                        graph_name=graph_name,
                    )

                    computed_sum = candidate(materialized_values)
                    computed_bits = fp32_bits_hex(computed_sum)
                    observed_sum = Fraction.from_float(float(computed_sum))
                    observed_error = observed_sum - prediction.exact_input_sum
                    rows.append(
                        {
                            "validation_version": VALIDATION_VERSION,
                            "scale_exponent_k": scale_exponent_k,
                            "tail_count": tail_count,
                            "tail_power_of_two_exponent": tail_exponent,
                            "layout": layout,
                            "graph": graph_name,
                            "leaf_count": leaf_count,
                            "input_hash": input_hash(materialized_values),
                            "exact_input_sum_fraction": str(prediction.exact_input_sum),
                            "predicted_sum_fraction": str(prediction.predicted_sum),
                            "predicted_sum_bits": prediction.predicted_sum_bits,
                            "preregistered_sum_bits": expected_bits,
                            "computed_sum_bits": computed_bits,
                            "predicted_signed_error_fraction": str(
                                prediction.signed_error
                            ),
                            "observed_signed_error_fraction": str(observed_error),
                            "local_error_sum_fraction": str(prediction.local_error_sum),
                            "inexact_addition_count": (
                                prediction.inexact_addition_count
                            ),
                            "predictor_matched_preregistered": (
                                prediction.predicted_sum_bits == expected_bits
                            ),
                            "predictor_matched_observation": (
                                prediction.predicted_sum_bits == computed_bits
                                and prediction.signed_error == observed_error
                            ),
                        }
                    )
    return rows


def write_artifacts(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    """Write the retained batch observations and their provenance."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    experiments_dir = Path(__file__).resolve().parent
    sources = {
        "summation_graph_predictor.py": experiments_dir
        / "summation_graph_predictor.py",
        "summation_graph_predictor_validation.py": Path(__file__).resolve(),
        "fp32_summation_stress.py": experiments_dir / "fp32_summation_stress.py",
        "softmax_failure_triage.py": experiments_dir / "softmax_failure_triage.py",
    }
    metadata = {
        "validation_version": VALIDATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "python topics/softmax/experiments/summation_graph_predictor_validation.py"
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "predictor_definition": (
            "E_G = predicted_root - exact_input_sum = sum(local_residuals)"
        ),
        "predictor_ownership": (
            "agent-authored independent semantic oracle for the requested "
            "validation; learner closed-book mastery is not claimed"
        ),
        "rounding": "IEEE binary32 round-to-nearest, ties-to-even",
        "registered_scale_exponents_k": list(SCALE_EXPONENTS),
        "registered_layouts": list(LAYOUTS),
        "registered_graphs": list(GRAPH_NAMES),
        "row_count": len(rows),
        "all_predictions_matched_preregistered": all(
            bool(row["predictor_matched_preregistered"]) for row in rows
        ),
        "all_predictions_matched_observation": all(
            bool(row["predictor_matched_observation"]) for row in rows
        ),
        "sources": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()
        },
        "artifacts": {CSV_PATH.name: hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()},
    }
    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return CSV_PATH, METADATA_PATH


def main() -> None:
    rows = validation_rows()
    paths = write_artifacts(rows)
    matched = sum(bool(row["predictor_matched_observation"]) for row in rows)
    print(f"matched {matched}/{len(rows)} predictor/observation rows")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
