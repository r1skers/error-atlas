"""Run the registered Softmax summation triage and preserve raw evidence."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fp32_summation_stress import (
    compensated_sum_fp32,
    pairwise_sum_fp32,
    sequential_sum_fp32,
)
from softmax_failure_triage import (
    CaseRecipe,
    EnvironmentSnapshot,
    ExecutionConfig,
    RunAcceptancePolicy,
    RunObservation,
    RunSummary,
    assess_run_summary,
    config_id,
    environment_id,
    head_then_power_tail_recipe,
    observe_power_tail_summation,
    source_file_sha256,
    summarize_runs,
)


SUITE_NAME = "power_tail_summation_mitigation"
SUITE_VERSION = "1"
SMOKE_TAIL_COUNTS = (2, 2**10)
STRESS_TAIL_COUNTS = (2**20,)
TAIL_POWER_OF_TWO_EXPONENT = -24
DEFAULT_REPEAT_COUNT = 3
DEFAULT_TOLERANCE = 1e-6


RAW_COLUMNS = (
    "suite_name",
    "suite_version",
    "suite_tier",
    "tail_count",
    "tail_power_of_two_exponent",
    "candidate_name",
    "case_id",
    "input_hash",
    "config_id",
    "implementation_hash",
    "environment_id",
    "run_index",
    "computed_sum",
    "computed_sum_bits",
    "reference_sum",
    "signed_error",
    "absolute_error",
    "relative_error",
    "absolute_relative_error",
)


SUMMARY_COLUMNS = (
    "suite_name",
    "suite_version",
    "suite_tier",
    "tail_count",
    "tail_power_of_two_exponent",
    "candidate_name",
    "case_id",
    "input_hash",
    "config_id",
    "implementation_hash",
    "environment_id",
    "run_count",
    "finite_output_count",
    "nan_output_count",
    "positive_infinity_count",
    "negative_infinity_count",
    "has_nonfinite_output",
    "unique_output_count",
    "all_runs_bitwise_equal",
    "output_bit_counts_json",
    "finite_min",
    "finite_max",
    "finite_mean",
    "finite_population_std",
    "max_finite_absolute_relative_error",
    "max_absolute_relative_error_tolerance",
    "require_bitwise_repeatability",
    "accuracy_requirement_passed",
    "repeatability_requirement_passed",
    "overall_passed",
    "failure_reason_codes_json",
    "warning_reason_codes_json",
)


@dataclass(frozen=True)
class CandidateSpec:
    """One controlled summation implementation and its recorded identity."""

    name: str
    config: ExecutionConfig
    implementation_source: Path
    summation: Callable[[np.ndarray], np.float32]

    @property
    def implementation_hash(self) -> str:
        return source_file_sha256(self.implementation_source)


def sequential_fp64_accumulator_to_fp32(values: np.ndarray) -> np.float32:
    """Accumulate left-to-right in FP64 and cast the final result to FP32."""
    if values.dtype != np.float32:
        raise TypeError("values must have dtype float32")
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    accumulator = 0.0
    for value in values:
        accumulator += float(value)
    return np.float32(accumulator)


def registered_cases(*, include_stress: bool) -> tuple[tuple[str, CaseRecipe], ...]:
    """Return smoke cases and, when requested, the explicit stress tier."""
    registered = [
        (
            "smoke",
            head_then_power_tail_recipe(
                tail_count=tail_count,
                tail_power_of_two_exponent=TAIL_POWER_OF_TWO_EXPONENT,
            ),
        )
        for tail_count in SMOKE_TAIL_COUNTS
    ]
    if include_stress:
        registered.extend(
            (
                "stress",
                head_then_power_tail_recipe(
                    tail_count=tail_count,
                    tail_power_of_two_exponent=TAIL_POWER_OF_TWO_EXPONENT,
                ),
            )
            for tail_count in STRESS_TAIL_COUNTS
        )
    return tuple(registered)


def registered_candidates() -> tuple[CandidateSpec, ...]:
    """Return the baseline and three single-change mitigation candidates."""
    experiments_dir = Path(__file__).resolve().parent
    fp32_source = experiments_dir / "fp32_summation_stress.py"
    runner_source = Path(__file__).resolve()

    return (
        CandidateSpec(
            name="sequential_fp32",
            config=ExecutionConfig(
                implementation_name="explicit_left_to_right_loop",
                implementation_version="1",
                reduction_method="sequential",
                accumulator_dtype="float32",
                output_dtype="float32",
                deterministic=True,
                method_parameters={"rounding": "nearest_even"},
            ),
            implementation_source=fp32_source,
            summation=sequential_sum_fp32,
        ),
        CandidateSpec(
            name="pairwise_fp32",
            config=ExecutionConfig(
                implementation_name="fixed_balanced_binary_tree",
                implementation_version="1",
                reduction_method="pairwise",
                accumulator_dtype="float32",
                output_dtype="float32",
                deterministic=True,
                method_parameters={"split": "first_length_floor_half"},
            ),
            implementation_source=fp32_source,
            summation=pairwise_sum_fp32,
        ),
        CandidateSpec(
            name="compensated_fp32",
            config=ExecutionConfig(
                implementation_name="standard_kahan_loop",
                implementation_version="1",
                reduction_method="compensated",
                accumulator_dtype="float32",
                output_dtype="float32",
                deterministic=True,
                method_parameters={"compensation": "kahan"},
            ),
            implementation_source=fp32_source,
            summation=compensated_sum_fp32,
        ),
        CandidateSpec(
            name="sequential_fp64_accumulator",
            config=ExecutionConfig(
                implementation_name="explicit_left_to_right_loop",
                implementation_version="1",
                reduction_method="sequential",
                accumulator_dtype="float64",
                output_dtype="float32",
                deterministic=True,
                method_parameters={"final_cast": "float32"},
            ),
            implementation_source=runner_source,
            summation=sequential_fp64_accumulator_to_fp32,
        ),
    )


def capture_environment() -> EnvironmentSnapshot:
    """Capture the local CPU/NumPy execution environment used by this runner."""
    processor = platform.processor() or "unknown"
    return EnvironmentSnapshot(
        platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        backend="numpy_cpu",
        device=f"{platform.machine()}:{processor}",
        runtime_versions={
            "python_implementation": platform.python_implementation(),
        },
    )


def _raw_row(
    *,
    tier: str,
    recipe: CaseRecipe,
    candidate: CandidateSpec,
    observation: RunObservation,
) -> dict[str, object]:
    return {
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "suite_tier": tier,
        "tail_count": recipe.parameters["tail_count"],
        "tail_power_of_two_exponent": recipe.parameters[
            "tail_power_of_two_exponent"
        ],
        "candidate_name": candidate.name,
        **dataclasses.asdict(observation),
    }


def _summary_row(
    *,
    tier: str,
    recipe: CaseRecipe,
    candidate: CandidateSpec,
    summary: RunSummary,
    policy: RunAcceptancePolicy,
) -> dict[str, object]:
    assessment = assess_run_summary(summary, policy)
    summary_values = dataclasses.asdict(summary)
    output_bit_counts = summary_values.pop("output_bit_counts")
    return {
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "suite_tier": tier,
        "tail_count": recipe.parameters["tail_count"],
        "tail_power_of_two_exponent": recipe.parameters[
            "tail_power_of_two_exponent"
        ],
        "candidate_name": candidate.name,
        **summary_values,
        "output_bit_counts_json": json.dumps(
            output_bit_counts,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "max_absolute_relative_error_tolerance": (
            policy.max_absolute_relative_error_tolerance
        ),
        "require_bitwise_repeatability": (
            policy.require_bitwise_repeatability
        ),
        **dataclasses.asdict(assessment),
        "failure_reason_codes_json": json.dumps(
            assessment.failure_reason_codes,
            separators=(",", ":"),
        ),
        "warning_reason_codes_json": json.dumps(
            assessment.warning_reason_codes,
            separators=(",", ":"),
        ),
    }


def run_suite(
    *,
    include_stress: bool,
    repeat_count: int,
    policy: RunAcceptancePolicy,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Execute all registered case/candidate pairs and preserve every run."""
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
        raise TypeError("repeat_count must be an integer")
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")

    environment = capture_environment()
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cases = registered_cases(include_stress=include_stress)
    candidates = registered_candidates()

    for tier, recipe in cases:
        for candidate in candidates:
            observations = [
                observe_power_tail_summation(
                    recipe=recipe,
                    config=candidate.config,
                    environment=environment,
                    implementation_hash=candidate.implementation_hash,
                    run_index=run_index,
                    summation=candidate.summation,
                )
                for run_index in range(repeat_count)
            ]
            raw_rows.extend(
                _raw_row(
                    tier=tier,
                    recipe=recipe,
                    candidate=candidate,
                    observation=observation,
                )
                for observation in observations
            )
            summary_rows.append(
                _summary_row(
                    tier=tier,
                    recipe=recipe,
                    candidate=candidate,
                    summary=summarize_runs(observations),
                    policy=policy,
                )
            )

    metadata = {
        "schema_version": "1",
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "include_stress": include_stress,
        "repeat_count": repeat_count,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": dataclasses.asdict(policy),
        "environment": dataclasses.asdict(environment),
        "environment_id": environment_id(environment),
        "case_count": len(cases),
        "candidate_count": len(candidates),
        "raw_row_count": len(raw_rows),
        "summary_row_count": len(summary_rows),
        "reference_method": "fp64_sum_certified_against_exact_fraction",
        "pipeline_sources": {
            "softmax_failure_triage.py": source_file_sha256(
                Path(__file__).resolve().with_name("softmax_failure_triage.py")
            ),
            "softmax_failure_triage_runner.py": source_file_sha256(
                Path(__file__).resolve()
            ),
        },
        "cases": [
            {
                "suite_tier": tier,
                "recipe": dataclasses.asdict(recipe),
            }
            for tier, recipe in cases
        ],
        "candidates": [
            {
                "name": candidate.name,
                "config": dataclasses.asdict(candidate.config),
                "config_id": config_id(candidate.config),
                "implementation_source": candidate.implementation_source.name,
                "implementation_hash": candidate.implementation_hash,
            }
            for candidate in candidates
        ],
    }
    return raw_rows, summary_rows, metadata


def _write_csv(
    path: Path,
    *,
    columns: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    output_dir: Path,
    *,
    include_stress: bool,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    policy: RunAcceptancePolicy | None = None,
) -> tuple[Path, Path, Path]:
    """Run the suite and write raw, summary, and metadata artifacts."""
    selected_policy = policy or RunAcceptancePolicy(
        max_absolute_relative_error_tolerance=DEFAULT_TOLERANCE,
        require_bitwise_repeatability=True,
    )
    raw_rows, summary_rows, metadata = run_suite(
        include_stress=include_stress,
        repeat_count=repeat_count,
        policy=selected_policy,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "softmax_failure_triage_runs.csv"
    summary_path = output_dir / "softmax_failure_triage_summary.csv"
    metadata_path = output_dir / "softmax_failure_triage_metadata.json"
    _write_csv(raw_path, columns=RAW_COLUMNS, rows=raw_rows)
    _write_csv(summary_path, columns=SUMMARY_COLUMNS, rows=summary_rows)

    metadata["artifacts"] = {
        raw_path.name: source_file_sha256(raw_path),
        summary_path.name: source_file_sha256(summary_path),
    }
    metadata_path.write_text(
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
    return raw_path, summary_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Include the explicit 2**20-tail stress case.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "failure_triage",
    )
    arguments = parser.parse_args()

    paths = write_artifacts(
        arguments.output_dir,
        include_stress=arguments.include_stress,
        repeat_count=arguments.repeat_count,
        policy=RunAcceptancePolicy(
            max_absolute_relative_error_tolerance=arguments.tolerance,
            require_bitwise_repeatability=True,
        ),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
