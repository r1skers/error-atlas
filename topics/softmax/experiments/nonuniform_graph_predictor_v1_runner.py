"""Execute exactly one preregistered nonuniform graph-predictor case."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from fp32_summation_stress import pairwise_sum_fp32, sequential_sum_fp32
from softmax_failure_triage import fp32_bits_hex, input_hash
from summation_graph_predictor import (
    BinaryReductionGraph,
    balanced_reduction_graph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
    sequential_reduction_graph,
)


CASE_NAME = "nonuniform_positive_v1"
PREREGISTRATION_SHA256 = (
    "0c8047f38363c02ef6a6995bcc58a3f890dfebe7395aa5eb62a5cb671d6e47a6"
)
EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "nonuniform_graph_predictor_v1"
PREREGISTRATION_PATH = RESULTS_DIR / f"{CASE_NAME}_preregistration.json"
OBSERVATION_PATH = RESULTS_DIR / f"{CASE_NAME}_observations.csv"
METADATA_PATH = RESULTS_DIR / f"{CASE_NAME}_metadata.json"
OBSERVATION_COLUMNS = (
    "case_name",
    "preregistration_sha256",
    "input_hash",
    "graph_name",
    "leaf_count",
    "exact_leaf_sum_fraction",
    "correctly_rounded_target_bits",
    "preregistered_output_bits",
    "semantic_predicted_output_bits",
    "actual_output_bits",
    "preregistered_signed_error_fraction",
    "semantic_predicted_signed_error_fraction",
    "actual_signed_error_fraction",
    "prediction_matched_preregistration",
    "prediction_matched_observation",
    "preregistered_candidate_correctly_rounded",
    "observed_candidate_correctly_rounded",
)


@dataclass(frozen=True)
class CaseExecution:
    """Rows plus the frozen identity shared by both graph observations."""

    rows: tuple[dict[str, object], ...]
    input_hash: str
    preregistration_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_preregistration() -> dict[str, object]:
    actual_hash = _sha256(PREREGISTRATION_PATH)
    if actual_hash != PREREGISTRATION_SHA256:
        raise ValueError(
            "preregistration hash changed before execution: "
            f"expected {PREREGISTRATION_SHA256}, got {actual_hash}"
        )
    record = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    if record["case_name"] != CASE_NAME:
        raise ValueError("preregistration has an unexpected case_name")
    if record["status"] != "preregistered_not_executed":
        raise ValueError("preregistration status is not the frozen pre-run state")
    if record["preregistered_before_execution"] is not True:
        raise ValueError("record does not assert preregistration before execution")
    return record


def _graph_record(graph: BinaryReductionGraph) -> dict[str, object]:
    return {
        "name": graph.name,
        "leaf_indices": list(range(graph.leaf_count)),
        "nodes": [
            {
                "index": graph.leaf_count + offset,
                **asdict(node),
            }
            for offset, node in enumerate(graph.nodes)
        ],
        "root": graph.root,
    }


def execute_preregistered_case() -> CaseExecution:
    """Validate the frozen prediction, then execute each candidate once."""
    preregistration = _load_frozen_preregistration()
    input_record = preregistration["inputs"]
    exact_values = tuple(
        Fraction(value) for value in input_record["ordered_leaf_fractions"]
    )
    exact_leaf_sum = sum(exact_values, start=Fraction(0))
    if exact_leaf_sum != Fraction(preregistration["exact_leaf_sum_fraction"]):
        raise ValueError("ordered leaves do not reproduce exact_leaf_sum_fraction")

    materialized = np.array(
        [float(value) for value in exact_values],
        dtype=np.float32,
    )
    round_tripped = tuple(Fraction.from_float(float(value)) for value in materialized)
    if round_tripped != exact_values:
        raise ValueError("materialized FP32 leaves differ from preregistration")
    materialized_hash = input_hash(materialized)

    correct_target = round_nonnegative_fraction_to_fp32(exact_leaf_sum).bits_hex
    if correct_target != preregistration["correctly_rounded_target_bits"]:
        raise ValueError("independent correct target differs from preregistration")

    preregistered_predictions = {
        prediction["graph"]["name"]: prediction
        for prediction in preregistration["predictions"]
    }
    candidates = (
        (
            sequential_reduction_graph(len(exact_values)),
            sequential_sum_fp32,
        ),
        (
            balanced_reduction_graph(len(exact_values)),
            pairwise_sum_fp32,
        ),
    )

    rows: list[dict[str, object]] = []
    for graph, candidate in candidates:
        preregistered = preregistered_predictions[graph.name]
        if _graph_record(graph) != preregistered["graph"]:
            raise ValueError(
                f"materialized graph {graph.name} differs from preregistration"
            )

        semantic_prediction = predict_fp32_tree_error(exact_values, graph)
        prediction_matched_preregistration = (
            semantic_prediction.predicted_sum_bits
            == preregistered["predicted_output_bits"]
            and semantic_prediction.predicted_sum
            == Fraction(preregistered["predicted_output_fraction"])
            and semantic_prediction.signed_error
            == Fraction(preregistered["predicted_signed_error_fraction"])
        )
        if not prediction_matched_preregistration:
            raise ValueError(
                f"semantic predictor differs from preregistered {graph.name} values"
            )

        actual = candidate(materialized)
        actual_bits = fp32_bits_hex(actual)
        actual_fraction = Fraction.from_float(float(actual))
        actual_error = actual_fraction - exact_leaf_sum
        observed_correct_rounding = actual_bits == correct_target
        rows.append(
            {
                "case_name": CASE_NAME,
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "input_hash": materialized_hash,
                "graph_name": graph.name,
                "leaf_count": graph.leaf_count,
                "exact_leaf_sum_fraction": str(exact_leaf_sum),
                "correctly_rounded_target_bits": correct_target,
                "preregistered_output_bits": preregistered["predicted_output_bits"],
                "semantic_predicted_output_bits": (
                    semantic_prediction.predicted_sum_bits
                ),
                "actual_output_bits": actual_bits,
                "preregistered_signed_error_fraction": preregistered[
                    "predicted_signed_error_fraction"
                ],
                "semantic_predicted_signed_error_fraction": str(
                    semantic_prediction.signed_error
                ),
                "actual_signed_error_fraction": str(actual_error),
                "prediction_matched_preregistration": (
                    prediction_matched_preregistration
                ),
                "prediction_matched_observation": (
                    semantic_prediction.predicted_sum_bits == actual_bits
                    and semantic_prediction.signed_error == actual_error
                ),
                "preregistered_candidate_correctly_rounded": preregistered[
                    "candidate_correctly_rounded"
                ],
                "observed_candidate_correctly_rounded": (observed_correct_rounding),
            }
        )

    return CaseExecution(
        rows=tuple(rows),
        input_hash=materialized_hash,
        preregistration_sha256=PREREGISTRATION_SHA256,
    )


def write_observation_artifacts(execution: CaseExecution) -> tuple[Path, Path]:
    """Write two observations and provenance without changing preregistration."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OBSERVATION_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OBSERVATION_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(execution.rows)

    source_paths = {
        "nonuniform_graph_predictor_v1_runner.py": Path(__file__).resolve(),
        "summation_graph_predictor.py": (
            EXPERIMENTS_DIR / "summation_graph_predictor.py"
        ),
        "fp32_summation_stress.py": (EXPERIMENTS_DIR / "fp32_summation_stress.py"),
        "softmax_failure_triage.py": (EXPERIMENTS_DIR / "softmax_failure_triage.py"),
    }
    metadata = {
        "schema_version": "1",
        "case_name": CASE_NAME,
        "evidence_status": "single_preregistered_case_observed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "python topics/softmax/experiments/nonuniform_graph_predictor_v1_runner.py"
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "input_hash": execution.input_hash,
        "preregistration": {
            "path": PREREGISTRATION_PATH.name,
            "sha256": execution.preregistration_sha256,
        },
        "observation_row_count": len(execution.rows),
        "all_predictions_matched_observation": all(
            bool(row["prediction_matched_observation"]) for row in execution.rows
        ),
        "all_correct_rounding_predictions_matched": all(
            row["preregistered_candidate_correctly_rounded"]
            == row["observed_candidate_correctly_rounded"]
            for row in execution.rows
        ),
        "sources": {name: _sha256(path) for name, path in source_paths.items()},
        "artifacts": {
            OBSERVATION_PATH.name: _sha256(OBSERVATION_PATH),
        },
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
    return OBSERVATION_PATH, METADATA_PATH


def main() -> None:
    execution = execute_preregistered_case()
    paths = write_observation_artifacts(execution)
    for row in execution.rows:
        print(
            f"{row['graph_name']}: predicted={row['semantic_predicted_output_bits']} "
            f"actual={row['actual_output_bits']} "
            f"E={row['actual_signed_error_fraction']} "
            f"prediction_matched={row['prediction_matched_observation']} "
            f"correctly_rounded={row['observed_candidate_correctly_rounded']}"
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
