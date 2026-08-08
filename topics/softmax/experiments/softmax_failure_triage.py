"""Reproducible identities for staged Softmax failure-triage datasets.

This module starts with input identity only.  Execution configurations,
repeated run observations, and pathological generators are added in later
steps so recipe drift is not confused with reduction nondeterminism.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from fp32_summation_stress import fp64_reference_sum


POWER_TAIL_GENERATOR_NAME = "head_then_power_tail"
POWER_TAIL_GENERATOR_VERSION = "1"


@dataclass(frozen=True)
class CaseRecipe:
    """Semantic recipe for one ordered input case.

    ``case_id`` identifies this requested recipe.  A separate ``input_hash``
    identifies the dtype, shape, and ordered bytes actually materialized from
    it.
    """

    generator_name: str
    generator_version: str
    seeds: tuple[int, ...]
    parameters: Mapping[str, object]
    dtype: str
    shape: tuple[int, ...]
    layout: str


@dataclass(frozen=True)
class ExecutionConfig:
    """Semantic configuration for one reduction execution strategy."""

    implementation_name: str
    implementation_version: str
    reduction_method: str
    accumulator_dtype: str
    output_dtype: str
    deterministic: bool
    method_parameters: Mapping[str, object]


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Actual software and hardware environment used for one execution."""

    platform: str
    python_version: str
    numpy_version: str
    backend: str
    device: str
    runtime_versions: Mapping[str, str]


@dataclass(frozen=True)
class RunObservation:
    """One raw FP32 denominator result plus consistently derived errors."""

    case_id: str
    input_hash: str
    config_id: str
    implementation_hash: str
    environment_id: str
    run_index: int
    computed_sum: float
    computed_sum_bits: str
    reference_sum: float
    signed_error: float
    absolute_error: float
    relative_error: float
    absolute_relative_error: float


@dataclass(frozen=True)
class RunSummary:
    """Repeatability and finite-value statistics for one controlled run group."""

    case_id: str
    input_hash: str
    config_id: str
    implementation_hash: str
    environment_id: str
    run_count: int
    finite_output_count: int
    nan_output_count: int
    positive_infinity_count: int
    negative_infinity_count: int
    has_nonfinite_output: bool
    unique_output_count: int
    all_runs_bitwise_equal: bool
    output_bit_counts: Mapping[str, int]
    finite_min: float | None
    finite_max: float | None
    finite_mean: float | None
    finite_population_std: float | None
    max_finite_absolute_relative_error: float | None


@dataclass(frozen=True)
class RunAcceptancePolicy:
    """Consumer-owned numerical tolerance and repeatability requirement."""

    max_absolute_relative_error_tolerance: float
    require_bitwise_repeatability: bool

    def __post_init__(self) -> None:
        tolerance = self.max_absolute_relative_error_tolerance
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "max_absolute_relative_error_tolerance "
                "must be finite and nonnegative"
            )

        if not isinstance(self.require_bitwise_repeatability, bool):
            raise TypeError(
                "require_bitwise_repeatability must be a bool"
            )


@dataclass(frozen=True)
class RunAssessment:
    """Policy-dependent decision derived from a policy-free run summary."""

    accuracy_requirement_passed: bool
    repeatability_requirement_passed: bool
    overall_passed: bool
    failure_reason_codes: tuple[str, ...]
    warning_reason_codes: tuple[str, ...]




def _canonical_json_bytes(value: object) -> bytes:
    """Return canonical UTF-8 bytes for a JSON-compatible value.

    Learner-owned research core.  Required invariants:

    - Sort mapping keys recursively so insertion order cannot change identity.
    - Use compact JSON separators and reject NaN/Infinity.
    - Return encoded UTF-8 bytes, not a Python ``repr``.
    """
    json_bytes = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return json_bytes


def canonical_recipe_bytes(recipe: CaseRecipe) -> bytes:
    """Return the canonical UTF-8 representation used to derive ``case_id``.

    Learner-owned research core.  Required invariants:

    - Include every semantic ``CaseRecipe`` field, but no timestamps or
      execution configuration.
    - Normalize tuples and mappings into JSON-compatible values.
    - Sort mapping keys recursively so insertion order cannot change identity.
    - Use compact JSON separators and reject NaN/Infinity.
    - Return encoded UTF-8 bytes, not a Python ``repr``.
    """
    recipe_dict = dataclasses.asdict(recipe)
    recipe_bytes = _canonical_json_bytes(recipe_dict)
    return recipe_bytes


def case_id(recipe: CaseRecipe) -> str:
    """Return the SHA-256 identity of a canonical case recipe."""
    return hashlib.sha256(canonical_recipe_bytes(recipe)).hexdigest()


def canonical_config_bytes(config: ExecutionConfig) -> bytes:
    """Return canonical UTF-8 bytes for an execution configuration.

    Learner-owned research core.  Extract the canonical JSON mechanics shared
    with ``canonical_recipe_bytes`` into one private helper instead of copying
    the serialization block.
    """
    config_dict = dataclasses.asdict(config)
    config_bytes = _canonical_json_bytes(config_dict)
    return config_bytes


def config_id(config: ExecutionConfig) -> str:
    """Return the SHA-256 identity of a canonical execution configuration."""
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def environment_id(environment: EnvironmentSnapshot) -> str:
    """Return the SHA-256 identity of an actual execution environment."""
    environment_dict = dataclasses.asdict(environment)
    return hashlib.sha256(_canonical_json_bytes(environment_dict)).hexdigest()


def source_file_sha256(path: str | Path) -> str:
    """Return the SHA-256 identity of one implementation source file."""
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Expected an existing source file, got {source_path}.")
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def fp32_bits_hex(value: np.float32) -> str:
    """Return one FP32 scalar's exact bits as ``0x`` plus eight hex digits.

    Learner-owned research core.  Reject Python ``float`` and NumPy scalars of
    any other dtype so the bit width is never inferred silently.
    """
    if not isinstance(value, np.float32):
        raise TypeError(f"Expected np.float32, got {type(value)}.")
    view_bits = value.view(np.uint32)
    return f"0x{view_bits:08x}"


def observe_fp32_sum(
    *,
    case_identity: str,
    materialized_input_hash: str,
    config_identity: str,
    implementation_hash: str,
    environment_identity: str,
    run_index: int,
    computed_sum: np.float32,
    reference_sum: float,
) -> RunObservation:
    """Construct one internally consistent FP32 summation observation.

    Learner-owned research core.  Validate that ``computed_sum`` is exactly an
    FP32 scalar, ``reference_sum`` is finite and positive, and ``run_index`` is
    nonnegative.  Derive the bit pattern and all four error fields here; do not
    accept any derived metric from the caller.
    """
    if run_index < 0:
        raise ValueError(f"Expected nonnegative run_index, got {run_index}.")

    if not isinstance(computed_sum, np.float32):
        raise TypeError(f"Expected np.float32, got {type(computed_sum)}.")
    if not math.isfinite(reference_sum) or reference_sum <= 0.0:
        raise ValueError(
            f"Expected finite positive reference_sum, got {reference_sum}."
        )

    computed_sum_bits = fp32_bits_hex(computed_sum)
    computed_sum_float = float(computed_sum)

    signed = computed_sum_float - reference_sum
    absolute = abs(signed)
    relative = signed / reference_sum
    absolute_relative = abs(relative)

    return RunObservation(
        case_id=case_identity,
        input_hash=materialized_input_hash,
        config_id=config_identity,
        implementation_hash=implementation_hash,
        environment_id=environment_identity,
        run_index=run_index,
        computed_sum=computed_sum_float,
        computed_sum_bits=computed_sum_bits,
        reference_sum=reference_sum,
        signed_error=signed,
        absolute_error=absolute,
        relative_error=relative,
        absolute_relative_error=absolute_relative,
    )


def head_then_power_tail_recipe(
    *,
    tail_count: int,
    tail_power_of_two_exponent: int = -24,
) -> CaseRecipe:
    """Return a validated recipe for ``(1, 2**e, ..., 2**e)`` in FP32."""
    if isinstance(tail_count, bool) or not isinstance(tail_count, int):
        raise TypeError("tail_count must be an integer")
    if tail_count < 0:
        raise ValueError("tail_count must be nonnegative")
    if (
        isinstance(tail_power_of_two_exponent, bool)
        or not isinstance(tail_power_of_two_exponent, int)
    ):
        raise TypeError("tail_power_of_two_exponent must be an integer")
    if not -149 <= tail_power_of_two_exponent <= 0:
        raise ValueError(
            "tail exponent must produce a nonzero FP32 value no larger than one"
        )

    return CaseRecipe(
        generator_name=POWER_TAIL_GENERATOR_NAME,
        generator_version=POWER_TAIL_GENERATOR_VERSION,
        seeds=(),
        parameters={
            "head_power_of_two_exponent": 0,
            "tail_power_of_two_exponent": tail_power_of_two_exponent,
            "tail_count": tail_count,
        },
        dtype="float32",
        shape=(tail_count + 1,),
        layout="head_then_tail",
    )


def exact_power_tail_sum(
    *,
    tail_count: int,
    tail_power_of_two_exponent: int,
) -> Fraction:
    """Return the exact rational sum ``1 + tail_count * 2**exponent``.

    Learner-owned research core.  Reuse ``head_then_power_tail_recipe`` for
    parameter validation, but do not materialize an array or convert the exact
    result to float.
    """
    head_then_power_tail_recipe(
        tail_count=tail_count,
        tail_power_of_two_exponent=tail_power_of_two_exponent,
    )
    return (
        Fraction(1, 1)
        + tail_count * Fraction(2, 1) ** tail_power_of_two_exponent
    )


def materialize_head_then_power_tail(recipe: CaseRecipe) -> np.ndarray:
    """Materialize one validated power-of-two head/tail recipe.

    Learner-owned research core.  Reject a recipe whose generator identity,
    dtype, layout, seeds, parameters, or shape do not match this generator.
    Construct every value directly in FP32, preserve head-first order, and
    return an independent one-dimensional array.
    """
    if (
        recipe.generator_name != POWER_TAIL_GENERATOR_NAME
        or recipe.generator_version != POWER_TAIL_GENERATOR_VERSION
    ):
        raise ValueError("Recipe has the wrong power-tail generator identity.")

    required_parameters = {
        "head_power_of_two_exponent",
        "tail_count",
        "tail_power_of_two_exponent",
    }
    if set(recipe.parameters) != required_parameters:
        raise ValueError("Recipe has invalid power-tail parameters.")

    tail_count = recipe.parameters["tail_count"]
    tail_power_of_two_exponent = recipe.parameters["tail_power_of_two_exponent"]

    expected_recipe = head_then_power_tail_recipe(
        tail_count=tail_count,
        tail_power_of_two_exponent=tail_power_of_two_exponent,
    )


    if recipe != expected_recipe:
        raise ValueError("Recipe does not match expected parameters")

    tail = np.float32(2 ** tail_power_of_two_exponent)
    values = np.full(tail_count + 1, tail, dtype=np.float32)
    values[0] = np.float32(1.0)

    return values


def observe_power_tail_summation(
    *,
    recipe: CaseRecipe,
    config: ExecutionConfig,
    environment: EnvironmentSnapshot,
    implementation_hash: str,
    run_index: int,
    summation: Callable[[np.ndarray], np.float32],
) -> RunObservation:
    """Materialize one direct-``q`` case and isolate its summation error."""
    values = materialize_head_then_power_tail(recipe)
    reference_sum = fp64_reference_sum(values)
    exact_sum = exact_power_tail_sum(
        tail_count=recipe.parameters["tail_count"],
        tail_power_of_two_exponent=recipe.parameters[
            "tail_power_of_two_exponent"
        ],
    )
    if Fraction.from_float(reference_sum) != exact_sum:
        raise ValueError("FP64 reference is not exact for this case.")

    computed_sum = summation(values)
    return observe_fp32_sum(
        case_identity=case_id(recipe),
        materialized_input_hash=input_hash(values),
        config_identity=config_id(config),
        implementation_hash=implementation_hash,
        environment_identity=environment_id(environment),
        run_index=run_index,
        computed_sum=computed_sum,
        reference_sum=reference_sum,
    )


def summarize_runs(observations: Sequence[RunObservation]) -> RunSummary:
    """Summarize one identity-controlled group without hiding nonfinite runs.

    Learner-owned research core.  Reject empty input, mixed identity groups,
    and duplicate ``run_index`` values.  Count exact output bit patterns and
    nonfinite categories separately.  Compute min/max/mean/population-standard-
    deviation only over finite outputs; use ``None`` for all four fields when
    no finite output exists.

    Need read again
    """
    if not observations:
        raise ValueError("Expected non-empty observations.")

    seen_run_indices : set[int] = set()

    for obs in observations:
        if not isinstance(obs, RunObservation):
            raise TypeError(
                f"Expected RunObservation, got {type(obs)}: {obs}"
            )

        if obs.case_id != observations[0].case_id:
            raise ValueError(
                f"All observations must have the same case_id, "
                f"but got {obs.case_id} and {observations[0].case_id}"
            )
        if obs.config_id != observations[0].config_id:
            raise ValueError(
                f"All observations must have the same config_id, "
                f"but got {obs.config_id} and {observations[0].config_id}"
            )
        if obs.input_hash != observations[0].input_hash:
            raise ValueError(
                f"All observations must have the same input_hash, "
                f"but got {obs.input_hash} and {observations[0].input_hash}"
            )
        if obs.implementation_hash != observations[0].implementation_hash:
            raise ValueError(
                f"All observations must have the same implementation_hash, "
                f"but got {obs.implementation_hash} and "
                f"{observations[0].implementation_hash}"
            )
        if obs.environment_id != observations[0].environment_id:
            raise ValueError(
                f"All observations must have the same environment_id, "
                f"but got {obs.environment_id} and "
                f"{observations[0].environment_id}"
            )
        if obs.run_index in seen_run_indices:
            raise ValueError(f"Duplicate run_index {obs.run_index} found.")
        seen_run_indices.add(obs.run_index)

    output_bit_counts: dict[str, int] = {}
    finite_values: list[float] = []
    nan_output_count = 0
    positive_infinity_count = 0
    negative_infinity_count = 0

    for obs in observations:
        bits = obs.computed_sum_bits
        output_bit_counts[bits] = output_bit_counts.get(bits, 0) + 1

        value = obs.computed_sum
        if math.isnan(value):
            nan_output_count += 1
        elif math.isinf(value):
            if value > 0:
                positive_infinity_count += 1
            else:
                negative_infinity_count += 1
        else:
            finite_values.append(value)

    finite_output_count = len(finite_values)
    if finite_values:
        finite_min = min(finite_values)
        finite_max = max(finite_values)
        finite_mean = math.fsum(finite_values) / finite_output_count
        finite_variance = (
            math.fsum(
                (value - finite_mean) ** 2 for value in finite_values
            )
            / finite_output_count
        )
        finite_population_std = math.sqrt(finite_variance)
    else:
        finite_min = None
        finite_max = None
        finite_mean = None
        finite_population_std = None

    unique_output_count = len(output_bit_counts)
    has_nonfinite_output = (
        nan_output_count
        + positive_infinity_count
        + negative_infinity_count
        > 0
    )
    first = observations[0]

    finite_absolute_relative_errors = [
        obs.absolute_relative_error
        for obs in observations
        if math.isfinite(obs.computed_sum)
    ]
    max_finite_absolute_relative_error = (
        max(finite_absolute_relative_errors)
        if finite_absolute_relative_errors
        else None
    )

    return RunSummary(
        case_id=first.case_id,
        input_hash=first.input_hash,
        config_id=first.config_id,
        implementation_hash=first.implementation_hash,
        environment_id=first.environment_id,
        run_count=len(observations),
        finite_output_count=finite_output_count,
        nan_output_count=nan_output_count,
        positive_infinity_count=positive_infinity_count,
        negative_infinity_count=negative_infinity_count,
        has_nonfinite_output=has_nonfinite_output,
        unique_output_count=unique_output_count,
        all_runs_bitwise_equal=unique_output_count == 1,
        output_bit_counts=output_bit_counts,
        finite_min=finite_min,
        finite_max=finite_max,
        finite_mean=finite_mean,
        finite_population_std=finite_population_std,
        max_finite_absolute_relative_error=max_finite_absolute_relative_error,
    )


def assess_run_summary(
    summary: RunSummary,
    policy: RunAcceptancePolicy,
) -> RunAssessment:
    """Apply one consumer policy without changing the underlying evidence.

    Learner-owned research core.  A numerical pass requires no nonfinite
    output, an available finite error metric, and a worst finite absolute
    relative error within tolerance.  A repeatability pass is automatic when
    the policy does not require bitwise equality; otherwise every run must
    have the same output bit pattern.  The overall decision requires both.
    """
    accuracy_requirement_passed = (
        not summary.has_nonfinite_output
        and summary.max_finite_absolute_relative_error is not None
        and summary.max_finite_absolute_relative_error
        <= policy.max_absolute_relative_error_tolerance
    )
    repeatability_requirement_passed = (
        not policy.require_bitwise_repeatability
        or summary.all_runs_bitwise_equal
    )
    overall_passed = (
        accuracy_requirement_passed and repeatability_requirement_passed
    )

    failure_reason_codes: list[str] = []
    warning_reason_codes: list[str] = []
    if not summary.all_runs_bitwise_equal:
        if policy.require_bitwise_repeatability:
            failure_reason_codes.append(
                "bitwise_repeatability_required"
            )
        else:
            warning_reason_codes.append(
                "bitwise_nonrepeatable"
            )

    if summary.has_nonfinite_output:
        failure_reason_codes.append("nonfinite_output")
    elif (
        summary.max_finite_absolute_relative_error is not None
        and summary.max_finite_absolute_relative_error
        > policy.max_absolute_relative_error_tolerance
    ):
        failure_reason_codes.append("accuracy_tolerance_exceeded")
    return RunAssessment(
        accuracy_requirement_passed=accuracy_requirement_passed,
        repeatability_requirement_passed=repeatability_requirement_passed,
        overall_passed=overall_passed,
        failure_reason_codes=tuple(failure_reason_codes),
        warning_reason_codes=tuple(warning_reason_codes),
    )



def input_hash(values: np.ndarray) -> str:
    """Hash dtype, shape, and logical C-order bytes of a materialized array."""
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a NumPy array")
    if values.dtype.hasobject:
        raise TypeError("object arrays do not have portable raw-byte identity")

    header = {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "byte_order": "C",
    }
    header_bytes = _canonical_json_bytes(header)
    ordered_bytes = np.ascontiguousarray(values).tobytes(order="C")

    digest = hashlib.sha256()
    digest.update(header_bytes)
    digest.update(b"\0")
    digest.update(ordered_bytes)
    return digest.hexdigest()
