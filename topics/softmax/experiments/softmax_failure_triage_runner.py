"""Run the registered Softmax summation triage and preserve raw evidence."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
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
    DECIMAL_TAIL_GENERATOR_NAME,
    DECIMAL_TAIL_GENERATOR_VERSION,
    POWER_TAIL_GENERATOR_NAME,
    POWER_TAIL_GENERATOR_VERSION,
    CaseRecipe,
    EnvironmentSnapshot,
    ExecutionConfig,
    RunAcceptancePolicy,
    RunObservation,
    RunSummary,
    assess_run_summary,
    canonical_policy_bytes,
    canonical_recipe_bytes,
    case_id,
    config_id,
    environment_id,
    exact_power_tail_sum,
    exact_uniform_decimal_tail_sums,
    fraction_to_exact_binary64,
    fp32_bits_hex,
    head_then_power_tail_recipe,
    observe_power_tail_summation,
    observe_uniform_decimal_tail_summation,
    policy_id,
    source_file_sha256,
    summarize_runs,
    uniform_decimal_tail_recipe,
)


SUITE_NAME = "softmax_denominator_summation_triage"
SUITE_VERSION = "3"
ARTIFACT_SCHEMA_VERSION = "2"
SMOKE_POWER_TAIL_SPECS = (
    (1, -24),
    (2, -24),
    (2**10, -24),
    (1023, -34),
    (1024, -34),
    (1025, -34),
)
SMOKE_DECIMAL_TAIL_SPECS = ((5, "1e-8"), (6, "1e-8"))
STRESS_POWER_TAIL_SPECS = ((2**20, -24),)
DEFAULT_REPEAT_COUNT = 3
DEFAULT_TOLERANCE = 1e-6
REGISTERED_LAYOUTS = ("head_then_tail", "tail_then_head")
REGISTERED_CASE_OBSERVERS: Mapping[
    tuple[str, str],
    Callable[..., RunObservation],
] = {
    (
        POWER_TAIL_GENERATOR_NAME,
        POWER_TAIL_GENERATOR_VERSION,
    ): observe_power_tail_summation,
    (
        DECIMAL_TAIL_GENERATOR_NAME,
        DECIMAL_TAIL_GENERATOR_VERSION,
    ): observe_uniform_decimal_tail_summation,
}

_CASE_CONTEXT_COLUMNS = (
    "suite_name",
    "suite_version",
    "suite_tier",
    "generator_name",
    "generator_version",
    "tail_count",
    "tail_power_of_two_exponent",
    "tail_source_decimal",
    "layout",
    "case_recipe_json",
)

CASE_COLUMNS = (
    *_CASE_CONTEXT_COLUMNS,
    "case_id",
    "input_hash",
    "source_sum_fraction",
    "stored_sum_fraction",
    "input_quantization_error_fraction",
    "correctly_rounded_stored_sum",
    "correctly_rounded_stored_sum_bits",
)

RAW_COLUMNS = (
    *_CASE_CONTEXT_COLUMNS,
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
    "correctly_rounded_reference_bits",
    "signed_error",
    "absolute_error",
    "relative_error",
    "absolute_relative_error",
)


SUMMARY_COLUMNS = (
    *_CASE_CONTEXT_COLUMNS,
    "candidate_name",
    "case_id",
    "input_hash",
    "config_id",
    "implementation_hash",
    "environment_id",
    "reference_sum",
    "correctly_rounded_reference_bits",
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
)


ASSESSMENT_COLUMNS = (
    *_CASE_CONTEXT_COLUMNS,
    "candidate_name",
    "case_id",
    "input_hash",
    "config_id",
    "implementation_hash",
    "environment_id",
    "policy_name",
    "policy_id",
    "policy_json",
    "max_absolute_relative_error_tolerance",
    "require_bitwise_repeatability",
    "require_correct_rounding",
    "accuracy_requirement_passed",
    "repeatability_requirement_passed",
    "correct_rounding_requirement_passed",
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
    implementation_hash: str
    summation: Callable[[np.ndarray], np.float32]


@dataclass(frozen=True)
class PolicySpec:
    """One human-readable label bound to a semantic policy configuration."""

    name: str
    policy: RunAcceptancePolicy


def _observe_registered_case(
    *,
    recipe: CaseRecipe,
    candidate: CandidateSpec,
    environment: EnvironmentSnapshot,
    run_index: int,
) -> RunObservation:
    """Dispatch one recipe only through its registered name/version pair."""
    generator_identity = (
        recipe.generator_name,
        recipe.generator_version,
    )
    try:
        observer = REGISTERED_CASE_OBSERVERS[generator_identity]
    except KeyError as exc:
        raise ValueError(
            "No observer registered for generator identity "
            f"{generator_identity}."
        ) from exc

    return observer(
        recipe=recipe,
        config=candidate.config,
        environment=environment,
        implementation_hash=candidate.implementation_hash,
        run_index=run_index,
        summation=candidate.summation,
    )


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
    registered: list[tuple[str, CaseRecipe]] = []
    for tail_count, tail_exponent in SMOKE_POWER_TAIL_SPECS:
        for layout in REGISTERED_LAYOUTS:
            recipe = head_then_power_tail_recipe(
                tail_count=tail_count,
                tail_power_of_two_exponent=tail_exponent,
                layout=layout,
            )
            registered.append(("smoke", recipe))

    for tail_count, tail_source_decimal in SMOKE_DECIMAL_TAIL_SPECS:
        for layout in REGISTERED_LAYOUTS:
            recipe = uniform_decimal_tail_recipe(
                tail_count=tail_count,
                tail_source_decimal=tail_source_decimal,
                layout=layout,
            )
            registered.append(("smoke", recipe))

    if include_stress:
        for tail_count, tail_exponent in STRESS_POWER_TAIL_SPECS:
            for layout in REGISTERED_LAYOUTS:
                recipe = head_then_power_tail_recipe(
                    tail_count=tail_count,
                    tail_power_of_two_exponent=tail_exponent,
                    layout=layout,
                )
                registered.append(("stress", recipe))

    return tuple(registered)


def registered_candidates() -> tuple[CandidateSpec, ...]:
    """Return the baseline and three single-change mitigation candidates."""
    experiments_dir = Path(__file__).resolve().parent
    fp32_source = experiments_dir / "fp32_summation_stress.py"
    runner_source = Path(__file__).resolve()
    fp32_implementation_hash = source_file_sha256(fp32_source)
    runner_implementation_hash = source_file_sha256(runner_source)

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
            implementation_hash=fp32_implementation_hash,
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
            implementation_hash=fp32_implementation_hash,
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
            implementation_hash=fp32_implementation_hash,
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
            implementation_hash=runner_implementation_hash,
            summation=sequential_fp64_accumulator_to_fp32,
        ),
    )


def registered_policies(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[PolicySpec, ...]:
    """Return orthogonal consumer-tolerance and correct-rounding policies."""
    return (
        PolicySpec(
            name="consumer_tolerance",
            policy=RunAcceptancePolicy(
                max_absolute_relative_error_tolerance=tolerance,
                require_bitwise_repeatability=True,
                require_correct_rounding=False,
            ),
        ),
        PolicySpec(
            name="correct_rounding",
            policy=RunAcceptancePolicy(
                max_absolute_relative_error_tolerance=tolerance,
                require_bitwise_repeatability=False,
                require_correct_rounding=True,
            ),
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


def _case_row_values(recipe: CaseRecipe) -> dict[str, object]:
    """Return common query columns plus the complete canonical recipe."""
    return {
        "generator_name": recipe.generator_name,
        "generator_version": recipe.generator_version,
        "tail_count": recipe.parameters.get("tail_count"),
        "tail_power_of_two_exponent": recipe.parameters.get(
            "tail_power_of_two_exponent"
        ),
        "tail_source_decimal": recipe.parameters.get("tail_source_decimal"),
        "layout": recipe.layout,
        "case_recipe_json": canonical_recipe_bytes(recipe).decode("utf-8"),
    }


def _case_reference_values(recipe: CaseRecipe) -> dict[str, object]:
    """Return exact case-level sums and the correctly rounded FP32 target."""
    generator_identity = (
        recipe.generator_name,
        recipe.generator_version,
    )
    if generator_identity == (
        POWER_TAIL_GENERATOR_NAME,
        POWER_TAIL_GENERATOR_VERSION,
    ):
        source_sum = exact_power_tail_sum(
            tail_count=recipe.parameters["tail_count"],
            tail_power_of_two_exponent=recipe.parameters[
                "tail_power_of_two_exponent"
            ],
        )
        stored_sum = source_sum
        certified_sum = fraction_to_exact_binary64(stored_sum)
        with np.errstate(over="ignore"):
            rounded_sum = np.float32(certified_sum)
        rounded_sum_value = float(rounded_sum)
        rounded_sum_bits = fp32_bits_hex(rounded_sum)
    elif generator_identity == (
        DECIMAL_TAIL_GENERATOR_NAME,
        DECIMAL_TAIL_GENERATOR_VERSION,
    ):
        references = exact_uniform_decimal_tail_sums(
            tail_count=recipe.parameters["tail_count"],
            tail_source_decimal=recipe.parameters["tail_source_decimal"],
        )
        source_sum = references.source_sum
        stored_sum = references.stored_sum
        rounded_sum_value = references.correctly_rounded_stored_sum
        rounded_sum_bits = references.correctly_rounded_stored_sum_bits
    else:
        raise ValueError(
            "No case reference registered for generator identity "
            f"{generator_identity}."
        )

    return {
        "source_sum_fraction": str(source_sum),
        "stored_sum_fraction": str(stored_sum),
        "input_quantization_error_fraction": str(stored_sum - source_sum),
        "correctly_rounded_stored_sum": rounded_sum_value,
        "correctly_rounded_stored_sum_bits": rounded_sum_bits,
    }


def _case_row(
    *,
    tier: str,
    recipe: CaseRecipe,
    materialized_input_hash: str,
) -> dict[str, object]:
    return {
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "suite_tier": tier,
        **_case_row_values(recipe),
        "case_id": case_id(recipe),
        "input_hash": materialized_input_hash,
        **_case_reference_values(recipe),
    }


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
        **_case_row_values(recipe),
        "candidate_name": candidate.name,
        **dataclasses.asdict(observation),
    }


def _summary_row(
    *,
    tier: str,
    recipe: CaseRecipe,
    candidate: CandidateSpec,
    summary: RunSummary,
) -> dict[str, object]:
    summary_values = dataclasses.asdict(summary)
    output_bit_counts = summary_values.pop("output_bit_counts")
    return {
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "suite_tier": tier,
        **_case_row_values(recipe),
        "candidate_name": candidate.name,
        **summary_values,
        "output_bit_counts_json": json.dumps(
            output_bit_counts,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _assessment_row(
    *,
    tier: str,
    recipe: CaseRecipe,
    candidate: CandidateSpec,
    summary: RunSummary,
    policy_spec: PolicySpec,
) -> dict[str, object]:
    policy = policy_spec.policy
    assessment_values = dataclasses.asdict(
        assess_run_summary(summary, policy)
    )
    failure_reason_codes = assessment_values.pop("failure_reason_codes")
    warning_reason_codes = assessment_values.pop("warning_reason_codes")
    return {
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "suite_tier": tier,
        **_case_row_values(recipe),
        "candidate_name": candidate.name,
        "case_id": summary.case_id,
        "input_hash": summary.input_hash,
        "config_id": summary.config_id,
        "implementation_hash": summary.implementation_hash,
        "environment_id": summary.environment_id,
        "policy_name": policy_spec.name,
        "policy_id": policy_id(policy),
        "policy_json": canonical_policy_bytes(policy).decode("utf-8"),
        **dataclasses.asdict(policy),
        **assessment_values,
        "failure_reason_codes_json": json.dumps(
            failure_reason_codes,
            separators=(",", ":"),
        ),
        "warning_reason_codes_json": json.dumps(
            warning_reason_codes,
            separators=(",", ":"),
        ),
    }


def run_suite(
    *,
    include_stress: bool,
    repeat_count: int,
    policies: Sequence[PolicySpec] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Execute all registered case/candidate pairs and preserve every run."""
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
        raise TypeError("repeat_count must be an integer")
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")

    selected_policies = tuple(
        registered_policies() if policies is None else policies
    )
    if not selected_policies:
        raise ValueError("Expected at least one policy spec.")
    if any(not isinstance(spec, PolicySpec) for spec in selected_policies):
        raise TypeError("Expected every policy entry to be a PolicySpec.")
    if len({spec.name for spec in selected_policies}) != len(selected_policies):
        raise ValueError("Policy names must be unique.")
    if len({policy_id(spec.policy) for spec in selected_policies}) != len(
        selected_policies
    ):
        raise ValueError("Policy identities must be unique.")

    environment = capture_environment()
    case_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    assessment_rows: list[dict[str, object]] = []
    cases = registered_cases(include_stress=include_stress)
    candidates = registered_candidates()

    for tier, recipe in cases:
        case_input_hash: str | None = None
        for candidate in candidates:
            observations = [
                _observe_registered_case(
                    recipe=recipe,
                    candidate=candidate,
                    environment=environment,
                    run_index=run_index,
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
            summary = summarize_runs(observations)
            if summary.case_id != case_id(recipe):
                raise ValueError("Summary case_id does not match its recipe.")
            if case_input_hash is None:
                case_input_hash = summary.input_hash
                case_rows.append(
                    _case_row(
                        tier=tier,
                        recipe=recipe,
                        materialized_input_hash=case_input_hash,
                    )
                )
            elif summary.input_hash != case_input_hash:
                raise ValueError(
                    "Candidates materialized different inputs for one case."
                )
            summary_rows.append(
                _summary_row(
                    tier=tier,
                    recipe=recipe,
                    candidate=candidate,
                    summary=summary,
                )
            )
            assessment_rows.extend(
                _assessment_row(
                    tier=tier,
                    recipe=recipe,
                    candidate=candidate,
                    summary=summary,
                    policy_spec=policy_spec,
                )
                for policy_spec in selected_policies
            )

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "suite_name": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "include_stress": include_stress,
        "repeat_count": repeat_count,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": dataclasses.asdict(environment),
        "environment_id": environment_id(environment),
        "case_count": len(cases),
        "case_row_count": len(case_rows),
        "candidate_count": len(candidates),
        "policy_count": len(selected_policies),
        "raw_row_count": len(raw_rows),
        "summary_row_count": len(summary_rows),
        "assessment_row_count": len(assessment_rows),
        "reference_methods": {
            POWER_TAIL_GENERATOR_NAME: (
                "fp64_sum_certified_against_exact_fraction"
            ),
            DECIMAL_TAIL_GENERATOR_NAME: (
                "exact_stored_fraction_certified_as_binary64_then_rounded_once"
            ),
        },
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
        "policies": [
            {
                "name": spec.name,
                "policy": dataclasses.asdict(spec.policy),
                "policy_id": policy_id(spec.policy),
            }
            for spec in selected_policies
        ],
    }
    return case_rows, raw_rows, summary_rows, assessment_rows, metadata


def _write_csv(
    path: Path,
    *,
    columns: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    expected_columns = set(columns)
    if len(expected_columns) != len(columns):
        raise ValueError("CSV columns must be unique.")
    for row_index, row in enumerate(rows):
        actual_columns = set(row)
        if actual_columns != expected_columns:
            missing = sorted(expected_columns - actual_columns)
            unexpected = sorted(actual_columns - expected_columns)
            raise ValueError(
                f"CSV row {row_index} does not match its schema; "
                f"missing={missing}, unexpected={unexpected}."
            )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    output_dir: Path,
    *,
    include_stress: bool,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    policies: Sequence[PolicySpec] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write case, raw, summary, assessment, and metadata artifacts."""
    case_rows, raw_rows, summary_rows, assessment_rows, metadata = run_suite(
        include_stress=include_stress,
        repeat_count=repeat_count,
        policies=policies,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / "softmax_failure_triage_cases.csv"
    raw_path = output_dir / "softmax_failure_triage_runs.csv"
    summary_path = output_dir / "softmax_failure_triage_summary.csv"
    assessment_path = output_dir / "softmax_failure_triage_assessments.csv"
    metadata_path = output_dir / "softmax_failure_triage_metadata.json"
    _write_csv(case_path, columns=CASE_COLUMNS, rows=case_rows)
    _write_csv(raw_path, columns=RAW_COLUMNS, rows=raw_rows)
    _write_csv(summary_path, columns=SUMMARY_COLUMNS, rows=summary_rows)
    _write_csv(
        assessment_path,
        columns=ASSESSMENT_COLUMNS,
        rows=assessment_rows,
    )

    metadata["artifacts"] = {
        case_path.name: source_file_sha256(case_path),
        raw_path.name: source_file_sha256(raw_path),
        summary_path.name: source_file_sha256(summary_path),
        assessment_path.name: source_file_sha256(assessment_path),
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
    return (
        case_path,
        raw_path,
        summary_path,
        assessment_path,
        metadata_path,
    )


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
        policies=registered_policies(tolerance=arguments.tolerance),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
