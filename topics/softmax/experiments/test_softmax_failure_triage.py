"""Tests for reproducible Softmax failure-triage input identities."""

import hashlib
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from fp32_summation_stress import pairwise_sum_fp32, sequential_sum_fp32

from softmax_failure_triage import (
    CaseRecipe,
    EnvironmentSnapshot,
    ExecutionConfig,
    RunAcceptancePolicy,
    RunAssessment,
    RunObservation,
    RunSummary,
    assess_run_summary,
    case_id,
    config_id,
    environment_id,
    exact_power_tail_sum,
    fp32_bits_hex,
    head_then_power_tail_recipe,
    input_hash,
    materialize_head_then_power_tail,
    observe_fp32_sum,
    observe_power_tail_summation,
    source_file_sha256,
    summarize_runs,
)


def base_recipe() -> CaseRecipe:
    return CaseRecipe(
        generator_name="head_tail_permutation",
        generator_version="1",
        seeds=(17,),
        parameters={"head": 1.0, "tail": 2.0**-24, "tail_count": 8},
        dtype="float32",
        shape=(9,),
        layout="seeded_permutation",
    )


def base_config() -> ExecutionConfig:
    return ExecutionConfig(
        implementation_name="fixed_pairwise_sum",
        implementation_version="1",
        reduction_method="pairwise",
        accumulator_dtype="float32",
        output_dtype="float32",
        deterministic=True,
        method_parameters={"block_size": 128, "items_per_thread": 4},
    )


def base_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        platform="test-platform",
        python_version="3.test",
        numpy_version="2.test",
        backend="numpy",
        device="cpu-test",
        runtime_versions={"blas": "test-blas-1"},
    )


class CaseRecipeIdentityTests(unittest.TestCase):
    def test_mapping_insertion_order_does_not_change_case_id(self) -> None:
        first = base_recipe()
        reordered = replace(
            first,
            parameters={"tail_count": 8, "tail": 2.0**-24, "head": 1.0},
        )

        self.assertEqual(case_id(first), case_id(reordered))

    def test_each_semantic_change_changes_case_id(self) -> None:
        recipe = base_recipe()
        variants = (
            replace(recipe, generator_version="2"),
            replace(recipe, seeds=(18,)),
            replace(recipe, parameters={**recipe.parameters, "tail_count": 9}),
            replace(recipe, dtype="float64"),
            replace(recipe, shape=(3, 3)),
            replace(recipe, layout="head_then_tail"),
        )

        original_id = case_id(recipe)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(original_id, case_id(variant))


class ExecutionConfigIdentityTests(unittest.TestCase):
    def test_parameter_insertion_order_does_not_change_config_id(self) -> None:
        first = base_config()
        reordered = replace(
            first,
            method_parameters={"items_per_thread": 4, "block_size": 128},
        )

        self.assertEqual(config_id(first), config_id(reordered))

    def test_each_execution_change_changes_config_id(self) -> None:
        config = base_config()
        variants = (
            replace(config, implementation_name="numpy_sum"),
            replace(config, implementation_version="2"),
            replace(config, reduction_method="sequential"),
            replace(config, accumulator_dtype="float64"),
            replace(config, output_dtype="float64"),
            replace(config, deterministic=False),
            replace(
                config,
                method_parameters={**config.method_parameters, "block_size": 256},
            ),
        )

        original_id = config_id(config)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(original_id, config_id(variant))


class EnvironmentIdentityTests(unittest.TestCase):
    def test_runtime_mapping_order_does_not_change_environment_id(self) -> None:
        first = replace(
            base_environment(),
            runtime_versions={"driver": "1", "runtime": "2"},
        )
        reordered = replace(
            first,
            runtime_versions={"runtime": "2", "driver": "1"},
        )

        self.assertEqual(environment_id(first), environment_id(reordered))

    def test_environment_change_changes_environment_id(self) -> None:
        environment = base_environment()
        variants = (
            replace(environment, platform="other-platform"),
            replace(environment, python_version="3.other"),
            replace(environment, numpy_version="2.other"),
            replace(environment, backend="cuda"),
            replace(environment, device="gpu-test"),
            replace(environment, runtime_versions={"blas": "test-blas-2"}),
        )

        original_id = environment_id(environment)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(original_id, environment_id(variant))


class SourceFileIdentityTests(unittest.TestCase):
    def test_hash_tracks_exact_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "candidate.py"
            original_bytes = b"def candidate():\n    return 1\n"
            source_path.write_bytes(original_bytes)

            original_hash = source_file_sha256(source_path)
            repeated_hash = source_file_sha256(str(source_path))
            source_path.write_bytes(original_bytes + b"# changed\n")
            changed_hash = source_file_sha256(source_path)

        self.assertEqual(
            original_hash,
            hashlib.sha256(original_bytes).hexdigest(),
        )
        self.assertEqual(repeated_hash, original_hash)
        self.assertNotEqual(changed_hash, original_hash)


class MaterializedInputHashTests(unittest.TestCase):
    def test_hash_includes_element_order(self) -> None:
        u = np.float32(2.0**-24)
        first = np.array([1.0, u, u], dtype=np.float32)
        reordered = np.array([u, 1.0, u], dtype=np.float32)

        self.assertNotEqual(input_hash(first), input_hash(reordered))

    def test_hash_includes_dtype_and_shape(self) -> None:
        flat = np.array([1.0, 2.0], dtype=np.float32)
        wider = flat.astype(np.float64)
        reshaped = flat.reshape(1, 2)

        self.assertNotEqual(input_hash(flat), input_hash(wider))
        self.assertNotEqual(input_hash(flat), input_hash(reshaped))

    def test_equivalent_views_have_same_logical_hash(self) -> None:
        base = np.arange(6, dtype=np.float32).reshape(2, 3)
        noncontiguous = base.T.copy().T

        self.assertFalse(noncontiguous.flags.c_contiguous)
        self.assertTrue(np.array_equal(base, noncontiguous))
        self.assertEqual(input_hash(base), input_hash(noncontiguous))

    def test_object_array_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            input_hash(np.array([object()], dtype=object))


class FP32BitPatternTests(unittest.TestCase):
    def test_one_and_its_successor_have_adjacent_bit_patterns(self) -> None:
        one = np.float32(1.0)
        successor = np.nextafter(one, np.float32(np.inf), dtype=np.float32)

        self.assertEqual(fp32_bits_hex(one), "0x3f800000")
        self.assertEqual(fp32_bits_hex(successor), "0x3f800001")

    def test_signed_zero_patterns_remain_distinct(self) -> None:
        self.assertEqual(fp32_bits_hex(np.float32(0.0)), "0x00000000")
        self.assertEqual(fp32_bits_hex(np.float32(-0.0)), "0x80000000")

    def test_non_fp32_scalars_are_rejected(self) -> None:
        for value in (1.0, np.float64(1.0), np.int32(1)):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    fp32_bits_hex(value)  # type: ignore[arg-type]


class RunObservationTests(unittest.TestCase):
    def test_lost_tail_has_negative_signed_and_nonnegative_absolute_errors(
        self,
    ) -> None:
        observation = observe_fp32_sum(
            case_identity="case-a",
            materialized_input_hash="input-a",
            config_identity="config-a",
            implementation_hash="implementation-a",
            environment_identity="environment-a",
            run_index=0,
            computed_sum=np.float32(1.0),
            reference_sum=1.0625,
        )

        self.assertIsInstance(observation, RunObservation)
        self.assertEqual(observation.case_id, "case-a")
        self.assertEqual(observation.input_hash, "input-a")
        self.assertEqual(observation.config_id, "config-a")
        self.assertEqual(observation.implementation_hash, "implementation-a")
        self.assertEqual(observation.environment_id, "environment-a")
        self.assertEqual(observation.run_index, 0)
        self.assertEqual(observation.computed_sum, 1.0)
        self.assertEqual(observation.computed_sum_bits, "0x3f800000")
        self.assertEqual(observation.reference_sum, 1.0625)
        self.assertEqual(observation.signed_error, -0.0625)
        self.assertEqual(observation.absolute_error, 0.0625)
        self.assertEqual(observation.relative_error, -1.0 / 17.0)
        self.assertEqual(observation.absolute_relative_error, 1.0 / 17.0)

    def test_invalid_raw_observation_inputs_are_rejected(self) -> None:
        common = {
            "case_identity": "case-a",
            "materialized_input_hash": "input-a",
            "config_identity": "config-a",
            "implementation_hash": "implementation-a",
            "environment_identity": "environment-a",
        }
        invalid = (
            {"run_index": -1, "computed_sum": np.float32(1.0), "reference_sum": 1.0},
            {"run_index": 0, "computed_sum": 1.0, "reference_sum": 1.0},
            {"run_index": 0, "computed_sum": np.float32(1.0), "reference_sum": 0.0},
            {
                "run_index": 0,
                "computed_sum": np.float32(1.0),
                "reference_sum": float("inf"),
            },
        )

        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    observe_fp32_sum(**common, **arguments)  # type: ignore[arg-type]


class HeadThenPowerTailGeneratorTests(unittest.TestCase):
    def test_small_recipe_materializes_exact_order_and_fp32_values(self) -> None:
        recipe = head_then_power_tail_recipe(
            tail_count=3,
            tail_power_of_two_exponent=-24,
        )
        first = materialize_head_then_power_tail(recipe)
        second = materialize_head_then_power_tail(recipe)
        u = np.float32(2.0**-24)

        self.assertEqual(recipe.shape, (4,))
        self.assertEqual(first.dtype, np.dtype(np.float32))
        self.assertEqual(first.ndim, 1)
        np.testing.assert_array_equal(
            first,
            np.array([np.float32(1.0), u, u, u], dtype=np.float32),
        )
        self.assertEqual(int(np.count_nonzero(first[1:])), 3)
        self.assertEqual(input_hash(first), input_hash(second))
        self.assertIsNot(first, second)

    def test_recipe_factory_rejects_unrepresentable_or_invalid_inputs(self) -> None:
        invalid = (
            {"tail_count": -1, "tail_power_of_two_exponent": -24},
            {"tail_count": True, "tail_power_of_two_exponent": -24},
            {"tail_count": 1, "tail_power_of_two_exponent": -150},
            {"tail_count": 1, "tail_power_of_two_exponent": 1},
        )

        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    head_then_power_tail_recipe(**arguments)  # type: ignore[arg-type]

    def test_materializer_rejects_recipe_shape_drift(self) -> None:
        recipe = replace(head_then_power_tail_recipe(tail_count=3), shape=(3,))

        with self.assertRaises(ValueError):
            materialize_head_then_power_tail(recipe)

    def test_materializer_rejects_foreign_generator_before_parameters(self) -> None:
        with self.assertRaises(ValueError):
            materialize_head_then_power_tail(base_recipe())

    def test_exact_reference_preserves_tail_below_fp64_resolution(self) -> None:
        registered = exact_power_tail_sum(
            tail_count=2**20,
            tail_power_of_two_exponent=-24,
        )
        extreme = exact_power_tail_sum(
            tail_count=1000,
            tail_power_of_two_exponent=-149,
        )

        self.assertEqual(registered, Fraction(17, 16))
        self.assertEqual(
            extreme,
            Fraction(1, 1) + 1000 * (Fraction(2, 1) ** -149),
        )
        self.assertNotEqual(extreme, Fraction(1, 1))

        with self.assertRaises(ValueError):
            exact_power_tail_sum(
                tail_count=-1,
                tail_power_of_two_exponent=-24,
            )


class PowerTailSummationIntegrationTests(unittest.TestCase):
    def test_direct_q_pipeline_attributes_only_sequential_summation_error(
        self,
    ) -> None:
        recipe = head_then_power_tail_recipe(tail_count=2)
        config = replace(
            base_config(),
            implementation_name="explicit_left_to_right_loop",
            reduction_method="sequential",
            method_parameters={"rounding": "nearest_even"},
        )
        observation = observe_power_tail_summation(
            recipe=recipe,
            config=config,
            environment=base_environment(),
            implementation_hash="test-implementation-hash",
            run_index=0,
            summation=sequential_sum_fp32,
        )
        u = float(np.float32(2.0**-24))
        reference_sum = 1.0 + 2.0 * u

        self.assertEqual(observation.case_id, case_id(recipe))
        self.assertEqual(observation.config_id, config_id(config))
        self.assertEqual(observation.computed_sum, 1.0)
        self.assertEqual(observation.reference_sum, reference_sum)
        self.assertEqual(observation.signed_error, -2.0 * u)
        self.assertEqual(observation.absolute_error, 2.0 * u)
        self.assertEqual(
            observation.relative_error,
            (-2.0 * u) / reference_sum,
        )

    def test_uncertified_fp64_reference_stops_before_target_summation(
        self,
    ) -> None:
        def should_not_run(values: np.ndarray) -> np.float32:
            self.fail("target summation ran with an uncertified reference")

        for exponent in (-53, -149):
            with self.subTest(exponent=exponent):
                recipe = head_then_power_tail_recipe(
                    tail_count=1000,
                    tail_power_of_two_exponent=exponent,
                )

                with self.assertRaises(ValueError):
                    observe_power_tail_summation(
                        recipe=recipe,
                        config=base_config(),
                        environment=base_environment(),
                        implementation_hash="test-implementation-hash",
                        run_index=0,
                        summation=should_not_run,
                    )


def summary_observation(
    *,
    run_index: int,
    computed_sum: np.float32,
    case_identity: str = "case-a",
) -> RunObservation:
    return observe_fp32_sum(
        case_identity=case_identity,
        materialized_input_hash="input-a",
        config_identity="config-a",
        implementation_hash="implementation-a",
        environment_identity="environment-a",
        run_index=run_index,
        computed_sum=computed_sum,
        reference_sum=1.0,
    )


class RunSummaryTests(unittest.TestCase):
    def test_identical_nan_bits_are_repeatable_but_nonfinite(self) -> None:
        observations = [
            summary_observation(run_index=index, computed_sum=np.float32(np.nan))
            for index in range(3)
        ]

        summary = summarize_runs(observations)

        self.assertIsInstance(summary, RunSummary)
        self.assertEqual(summary.run_count, 3)
        self.assertEqual(summary.finite_output_count, 0)
        self.assertEqual(summary.nan_output_count, 3)
        self.assertTrue(summary.has_nonfinite_output)
        self.assertEqual(summary.unique_output_count, 1)
        self.assertTrue(summary.all_runs_bitwise_equal)
        self.assertEqual(sum(summary.output_bit_counts.values()), 3)
        self.assertIsNone(summary.max_finite_absolute_relative_error)
        self.assertIsNone(summary.finite_min)
        self.assertIsNone(summary.finite_max)
        self.assertIsNone(summary.finite_mean)
        self.assertIsNone(summary.finite_population_std)

    def test_mixed_outputs_keep_finite_statistics_and_failure_counts(self) -> None:
        values = (
            np.float32(1.0),
            np.float32(1.0),
            np.float32(np.nan),
            np.float32(np.inf),
            np.float32(-np.inf),
        )
        observations = [
            summary_observation(run_index=index, computed_sum=value)
            for index, value in enumerate(values)
        ]

        summary = summarize_runs(observations)

        self.assertEqual(summary.run_count, 5)
        self.assertEqual(summary.finite_output_count, 2)
        self.assertEqual(summary.nan_output_count, 1)
        self.assertEqual(summary.positive_infinity_count, 1)
        self.assertEqual(summary.negative_infinity_count, 1)
        self.assertEqual(summary.unique_output_count, 4)
        self.assertFalse(summary.all_runs_bitwise_equal)
        self.assertEqual(summary.finite_min, 1.0)
        self.assertEqual(summary.finite_max, 1.0)
        self.assertEqual(summary.finite_mean, 1.0)
        self.assertEqual(summary.finite_population_std, 0.0)
        self.assertEqual(summary.max_finite_absolute_relative_error, 0.0)

    def test_accuracy_gate_metric_keeps_the_worst_finite_run(self) -> None:
        values = (np.float32(1.0), np.float32(1.25), np.float32(0.5))
        observations = [
            summary_observation(run_index=index, computed_sum=value)
            for index, value in enumerate(values)
        ]

        summary = summarize_runs(observations)

        self.assertEqual(summary.max_finite_absolute_relative_error, 0.5)

    def test_empty_mixed_group_and_duplicate_run_index_are_rejected(self) -> None:
        valid = summary_observation(run_index=0, computed_sum=np.float32(1.0))
        mixed = summary_observation(
            run_index=1,
            computed_sum=np.float32(1.0),
            case_identity="case-b",
        )
        duplicate = summary_observation(run_index=0, computed_sum=np.float32(1.0))

        invalid_groups = ([], [valid, mixed], [valid, duplicate])
        for group in invalid_groups:
            with self.subTest(group=group):
                with self.assertRaises(ValueError):
                    summarize_runs(group)


class RunAssessmentTests(unittest.TestCase):
    def test_policy_rejects_invalid_tolerance_at_construction(self) -> None:
        for tolerance in (-1.0, float("inf"), float("-inf"), float("nan")):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    RunAcceptancePolicy(
                        max_absolute_relative_error_tolerance=tolerance,
                        require_bitwise_repeatability=False,
                    )

        for tolerance in (0.0, 1e-6, 123.0):
            with self.subTest(tolerance=tolerance):
                policy = RunAcceptancePolicy(
                    max_absolute_relative_error_tolerance=tolerance,
                    require_bitwise_repeatability=False,
                )
                self.assertEqual(
                    policy.max_absolute_relative_error_tolerance,
                    tolerance,
                )

    def test_policy_rejects_non_boolean_repeatability_requirement(self) -> None:
        for requirement in ("false", 0, 1, None):
            with self.subTest(requirement=requirement):
                with self.assertRaises(TypeError):
                    RunAcceptancePolicy(
                        max_absolute_relative_error_tolerance=1e-6,
                        require_bitwise_repeatability=requirement,
                    )

    def test_nonrequired_bitwise_equality_does_not_block_accuracy(self) -> None:
        values = (
            np.float32(1.0),
            np.nextafter(np.float32(1.0), np.float32(np.inf)),
        )
        summary = summarize_runs(
            [
                summary_observation(run_index=index, computed_sum=value)
                for index, value in enumerate(values)
            ]
        )
        policy = RunAcceptancePolicy(
            max_absolute_relative_error_tolerance=1e-6,
            require_bitwise_repeatability=False,
        )

        assessment = assess_run_summary(summary, policy)

        self.assertIsInstance(assessment, RunAssessment)
        self.assertTrue(assessment.accuracy_requirement_passed)
        self.assertTrue(assessment.repeatability_requirement_passed)
        self.assertTrue(assessment.overall_passed)
        self.assertEqual(assessment.failure_reason_codes, ())
        self.assertEqual(
            assessment.warning_reason_codes,
            ("bitwise_nonrepeatable",),
        )
        self.assertFalse(summary.all_runs_bitwise_equal)

    def test_each_required_gate_can_block_the_overall_decision(self) -> None:
        nonrepeatable_summary = summarize_runs(
            [
                summary_observation(
                    run_index=0,
                    computed_sum=np.float32(1.0),
                ),
                summary_observation(
                    run_index=1,
                    computed_sum=np.nextafter(
                        np.float32(1.0), np.float32(np.inf)
                    ),
                ),
            ]
        )
        repeatability_required = RunAcceptancePolicy(
            max_absolute_relative_error_tolerance=1e-6,
            require_bitwise_repeatability=True,
        )
        inaccurate_summary = summarize_runs(
            [summary_observation(run_index=0, computed_sum=np.float32(1.5))]
        )
        strict_accuracy = RunAcceptancePolicy(
            max_absolute_relative_error_tolerance=0.1,
            require_bitwise_repeatability=False,
        )

        repeatability_failure = assess_run_summary(
            nonrepeatable_summary,
            repeatability_required,
        )
        accuracy_failure = assess_run_summary(
            inaccurate_summary,
            strict_accuracy,
        )

        self.assertTrue(repeatability_failure.accuracy_requirement_passed)
        self.assertFalse(
            repeatability_failure.repeatability_requirement_passed
        )
        self.assertFalse(repeatability_failure.overall_passed)
        self.assertEqual(
            repeatability_failure.failure_reason_codes,
            ("bitwise_repeatability_required",),
        )
        self.assertEqual(repeatability_failure.warning_reason_codes, ())
        self.assertFalse(accuracy_failure.accuracy_requirement_passed)
        self.assertTrue(accuracy_failure.repeatability_requirement_passed)
        self.assertFalse(accuracy_failure.overall_passed)
        self.assertEqual(
            accuracy_failure.failure_reason_codes,
            ("accuracy_tolerance_exceeded",),
        )
        self.assertEqual(accuracy_failure.warning_reason_codes, ())

    def test_nonfinite_output_has_a_cause_neutral_failure_code(self) -> None:
        summary = summarize_runs(
            [
                summary_observation(
                    run_index=0,
                    computed_sum=np.float32(np.nan),
                )
            ]
        )
        policy = RunAcceptancePolicy(
            max_absolute_relative_error_tolerance=1e-6,
            require_bitwise_repeatability=False,
        )

        assessment = assess_run_summary(summary, policy)

        self.assertFalse(assessment.accuracy_requirement_passed)
        self.assertFalse(assessment.overall_passed)
        self.assertEqual(
            assessment.failure_reason_codes,
            ("nonfinite_output",),
        )
        self.assertEqual(assessment.warning_reason_codes, ())


class SummationMitigationMatrixTests(unittest.TestCase):
    def test_one_u_u_case_filters_only_the_inaccurate_candidate(self) -> None:
        def sequential_fp64_accumulator_to_fp32(
            values: np.ndarray,
        ) -> np.float32:
            total = 0.0
            for value in values:
                total += float(value)
            return np.float32(total)

        recipe = head_then_power_tail_recipe(tail_count=2)
        policy = RunAcceptancePolicy(
            max_absolute_relative_error_tolerance=0.0,
            require_bitwise_repeatability=False,
        )
        candidates = {
            "sequential_fp32": sequential_sum_fp32,
            "pairwise_fp32": pairwise_sum_fp32,
            "sequential_fp64_accumulator": (
                sequential_fp64_accumulator_to_fp32
            ),
        }

        assessments = {}
        for name, summation in candidates.items():
            config = replace(
                base_config(),
                implementation_name=name,
                reduction_method=name.split("_fp", maxsplit=1)[0],
                accumulator_dtype=("float64" if "fp64" in name else "float32"),
            )
            observation = observe_power_tail_summation(
                recipe=recipe,
                config=config,
                environment=base_environment(),
                implementation_hash=f"test-{name}",
                run_index=0,
                summation=summation,
            )
            assessments[name] = assess_run_summary(
                summarize_runs([observation]),
                policy,
            )

        self.assertFalse(assessments["sequential_fp32"].overall_passed)
        self.assertTrue(assessments["pairwise_fp32"].overall_passed)
        self.assertTrue(
            assessments["sequential_fp64_accumulator"].overall_passed
        )


if __name__ == "__main__":
    unittest.main()
