"""Tests for the reproducible Softmax failure-triage suite runner."""

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from softmax_failure_triage import (
    CaseRecipe,
    case_id,
    head_then_power_tail_recipe,
    policy_id,
    source_file_sha256,
    uniform_decimal_tail_recipe,
)
from softmax_failure_triage_runner import (
    ARTIFACT_SCHEMA_VERSION,
    ASSESSMENT_COLUMNS,
    CASE_COLUMNS,
    RAW_COLUMNS,
    SUMMARY_COLUMNS,
    SUITE_VERSION,
    _observe_registered_case,
    capture_environment,
    registered_candidates,
    registered_cases,
    registered_policies,
    run_suite,
    sequential_fp64_accumulator_to_fp32,
    write_artifacts,
)


class RegisteredSuiteTests(unittest.TestCase):
    def test_stress_case_is_opt_in_and_does_not_change_smoke_cases(self) -> None:
        smoke = registered_cases(include_stress=False)
        with_stress = registered_cases(include_stress=True)

        self.assertEqual(
            [
                (
                    tier,
                    recipe.generator_name,
                    recipe.parameters["tail_count"],
                    recipe.parameters.get("tail_power_of_two_exponent"),
                    recipe.layout,
                )
                for tier, recipe in smoke
            ],
            [
                ("smoke", "head_then_power_tail", 1, -24, "head_then_tail"),
                ("smoke", "head_then_power_tail", 1, -24, "tail_then_head"),
                ("smoke", "head_then_power_tail", 2, -24, "head_then_tail"),
                ("smoke", "head_then_power_tail", 2, -24, "tail_then_head"),
                (
                    "smoke",
                    "head_then_power_tail",
                    2**10,
                    -24,
                    "head_then_tail",
                ),
                (
                    "smoke",
                    "head_then_power_tail",
                    2**10,
                    -24,
                    "tail_then_head",
                ),
                ("smoke", "head_then_power_tail", 1023, -34, "head_then_tail"),
                ("smoke", "head_then_power_tail", 1023, -34, "tail_then_head"),
                ("smoke", "head_then_power_tail", 1024, -34, "head_then_tail"),
                ("smoke", "head_then_power_tail", 1024, -34, "tail_then_head"),
                ("smoke", "head_then_power_tail", 1025, -34, "head_then_tail"),
                ("smoke", "head_then_power_tail", 1025, -34, "tail_then_head"),
                ("smoke", "uniform_decimal_tail", 5, None, "head_then_tail"),
                ("smoke", "uniform_decimal_tail", 5, None, "tail_then_head"),
                ("smoke", "uniform_decimal_tail", 6, None, "head_then_tail"),
                ("smoke", "uniform_decimal_tail", 6, None, "tail_then_head"),
            ],
        )
        self.assertEqual(with_stress[: len(smoke)], smoke)
        self.assertEqual(
            [
                (
                    tier,
                    recipe.parameters["tail_count"],
                    recipe.parameters["tail_power_of_two_exponent"],
                    recipe.layout,
                )
                for tier, recipe in with_stress[len(smoke) :]
            ],
            [
                ("stress", 2**20, -24, "head_then_tail"),
                ("stress", 2**20, -24, "tail_then_head"),
            ],
        )
        self.assertEqual(
            len({case_id(recipe) for _, recipe in with_stress}),
            len(with_stress),
        )

    def test_candidate_configs_are_distinct_and_fp64_loop_recovers_tail(
        self,
    ) -> None:
        candidates = registered_candidates()
        values = np.array(
            [1.0, 2.0**-24, 2.0**-24],
            dtype=np.float32,
        )

        self.assertEqual(len(candidates), 4)
        self.assertEqual(len({candidate.name for candidate in candidates}), 4)
        self.assertEqual(
            sequential_fp64_accumulator_to_fp32(values),
            np.float32(1.0 + 2.0**-23),
        )

    def test_observer_dispatch_uses_generator_name_and_version(self) -> None:
        candidate = registered_candidates()[0]
        decimal_recipe = uniform_decimal_tail_recipe(
            tail_count=6,
            tail_source_decimal="1e-8",
            layout="tail_then_head",
        )
        observation = _observe_registered_case(
            recipe=decimal_recipe,
            candidate=candidate,
            environment=capture_environment(),
            run_index=0,
        )

        self.assertEqual(observation.case_id, case_id(decimal_recipe))
        self.assertEqual(observation.computed_sum_bits, "0x3f800001")

        unknown_version = replace(
            head_then_power_tail_recipe(tail_count=2),
            generator_version="unknown",
        )
        with self.assertRaises(ValueError):
            _observe_registered_case(
                recipe=unknown_version,
                candidate=candidate,
                environment=capture_environment(),
                run_index=0,
            )

    def test_registered_policies_are_distinct_and_orthogonal(self) -> None:
        policies = registered_policies()

        self.assertEqual(
            [spec.name for spec in policies],
            ["consumer_tolerance", "correct_rounding"],
        )
        self.assertEqual(
            len({policy_id(spec.policy) for spec in policies}),
            len(policies),
        )
        consumer, correct_rounding = policies
        self.assertTrue(consumer.policy.require_bitwise_repeatability)
        self.assertFalse(consumer.policy.require_correct_rounding)
        self.assertFalse(
            correct_rounding.policy.require_bitwise_repeatability
        )
        self.assertTrue(correct_rounding.policy.require_correct_rounding)


class SuiteExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.case_rows,
            cls.raw_rows,
            cls.summary_rows,
            cls.assessment_rows,
            cls.metadata,
        ) = run_suite(
            include_stress=False,
            repeat_count=2,
        )
        cls.by_case_candidate_policy = {
            (
                row["case_id"],
                row["candidate_name"],
                row["policy_name"],
            ): row
            for row in cls.assessment_rows
        }
        cls.by_case_candidate_summary = {
            (row["case_id"], row["candidate_name"]): row
            for row in cls.summary_rows
        }

    @classmethod
    def assessment(
        cls,
        recipe: CaseRecipe,
        candidate_name: str,
        policy_name: str,
    ) -> dict[str, object]:
        return cls.by_case_candidate_policy[
            (case_id(recipe), candidate_name, policy_name)
        ]

    @classmethod
    def candidate_summary(
        cls,
        recipe: CaseRecipe,
        candidate_name: str,
    ) -> dict[str, object]:
        return cls.by_case_candidate_summary[
            (case_id(recipe), candidate_name)
        ]

    def test_smoke_suite_preserves_each_evidence_layer(self) -> None:
        self.assertEqual(len(self.case_rows), 16)
        self.assertEqual(len(self.raw_rows), 16 * 4 * 2)
        self.assertEqual(len(self.summary_rows), 16 * 4)
        self.assertEqual(len(self.assessment_rows), 16 * 4 * 2)
        self.assertEqual(
            self.metadata["raw_row_count"],
            len(self.raw_rows),
        )
        self.assertEqual(
            self.metadata["case_row_count"],
            len(self.case_rows),
        )
        self.assertEqual(
            self.metadata["summary_row_count"],
            len(self.summary_rows),
        )
        self.assertEqual(
            self.metadata["assessment_row_count"],
            len(self.assessment_rows),
        )
        self.assertEqual(
            set(self.metadata["pipeline_sources"]),
            {
                "softmax_failure_triage.py",
                "softmax_failure_triage_runner.py",
            },
        )

    def test_power_tail_smoke_controls_preserve_expected_decisions(self) -> None:
        self.assertFalse(
            self.assessment(
                head_then_power_tail_recipe(tail_count=2**10),
                "sequential_fp32",
                "consumer_tolerance",
            )["overall_passed"]
        )
        self.assertTrue(
            self.assessment(
                head_then_power_tail_recipe(
                    tail_count=2**10,
                    layout="tail_then_head",
                ),
                "sequential_fp32",
                "consumer_tolerance",
            )["overall_passed"]
        )
        for layout in ("head_then_tail", "tail_then_head"):
            for candidate_name in (
                "sequential_fp32",
                "pairwise_fp32",
                "compensated_fp32",
                "sequential_fp64_accumulator",
            ):
                tie_row = self.assessment(
                    head_then_power_tail_recipe(
                        tail_count=1,
                        layout=layout,
                    ),
                    candidate_name,
                    "correct_rounding",
                )
                self.assertTrue(tie_row["overall_passed"])
        tie_case_rows = [
            row
            for row in self.case_rows
            if row["generator_name"] == "head_then_power_tail"
            and row["tail_count"] == 1
        ]
        self.assertEqual(len(tie_case_rows), 2)
        self.assertTrue(
            all(
                row["correctly_rounded_stored_sum_bits"]
                == "0x3f800000"
                for row in tie_case_rows
            )
        )
        for candidate_name in (
            "pairwise_fp32",
            "compensated_fp32",
            "sequential_fp64_accumulator",
        ):
            for layout in ("head_then_tail", "tail_then_head"):
                self.assertTrue(
                    self.assessment(
                        head_then_power_tail_recipe(
                            tail_count=2**10,
                            layout=layout,
                        ),
                        candidate_name,
                        "consumer_tolerance",
                    )["overall_passed"]
                )

    def test_large_midpoint_boundary_family_matches_predictions(self) -> None:
        boundary_targets = {
            1023: "0x3f800000",
            1024: "0x3f800000",
            1025: "0x3f800001",
        }
        for tail_count, target_bits in boundary_targets.items():
            for layout in ("head_then_tail", "tail_then_head"):
                recipe = head_then_power_tail_recipe(
                    tail_count=tail_count,
                    tail_power_of_two_exponent=-34,
                    layout=layout,
                )
                expected_outputs = {
                    "sequential_fp32": (
                        "0x3f800000"
                        if layout == "head_then_tail"
                        else target_bits
                    ),
                    "pairwise_fp32": "0x3f800000",
                    "compensated_fp32": target_bits,
                    "sequential_fp64_accumulator": target_bits,
                }
                for candidate_name, expected_output_bits in (
                    expected_outputs.items()
                ):
                    summary = self.candidate_summary(recipe, candidate_name)
                    self.assertEqual(
                        summary["correctly_rounded_reference_bits"],
                        target_bits,
                    )
                    self.assertEqual(
                        json.loads(summary["output_bit_counts_json"]),
                        {expected_output_bits: 2},
                    )
                    correct_rounding_row = self.assessment(
                        recipe,
                        candidate_name,
                        "correct_rounding",
                    )
                    self.assertEqual(
                        correct_rounding_row["overall_passed"],
                        expected_output_bits == target_bits,
                    )

    def test_decimal_tail_controls_separate_policy_decisions(self) -> None:
        expected_correct_rounding = {
            ("head_then_tail", "sequential_fp32"): False,
            ("tail_then_head", "sequential_fp32"): True,
            ("head_then_tail", "pairwise_fp32"): False,
            ("tail_then_head", "pairwise_fp32"): False,
            ("head_then_tail", "compensated_fp32"): True,
            ("tail_then_head", "compensated_fp32"): True,
            ("head_then_tail", "sequential_fp64_accumulator"): True,
            ("tail_then_head", "sequential_fp64_accumulator"): True,
        }
        for layout in ("head_then_tail", "tail_then_head"):
            for candidate_name in (
                "sequential_fp32",
                "pairwise_fp32",
                "compensated_fp32",
                "sequential_fp64_accumulator",
            ):
                for policy_name in (
                    "consumer_tolerance",
                    "correct_rounding",
                ):
                    row = self.assessment(
                        uniform_decimal_tail_recipe(
                            tail_count=5,
                            tail_source_decimal="1e-8",
                            layout=layout,
                        ),
                        candidate_name,
                        policy_name,
                    )
                    self.assertTrue(row["overall_passed"])
                    self.assertNotIn(
                        "not_correctly_rounded",
                        json.loads(row["warning_reason_codes_json"]),
                    )
        for key, correctly_rounded in expected_correct_rounding.items():
            layout, candidate_name = key
            recipe = uniform_decimal_tail_recipe(
                tail_count=6,
                tail_source_decimal="1e-8",
                layout=layout,
            )
            consumer_row = self.assessment(
                recipe,
                candidate_name,
                "consumer_tolerance",
            )
            correct_rounding_row = self.assessment(
                recipe,
                candidate_name,
                "correct_rounding",
            )
            self.assertTrue(consumer_row["overall_passed"])
            warnings = json.loads(
                consumer_row["warning_reason_codes_json"]
            )
            self.assertEqual(
                "not_correctly_rounded" in warnings,
                not correctly_rounded,
            )
            self.assertEqual(
                correct_rounding_row["overall_passed"],
                correctly_rounded,
            )


class WrittenArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        (
            cls.case_path,
            cls.raw_path,
            cls.summary_path,
            cls.assessment_path,
            cls.metadata_path,
        ) = write_artifacts(
            Path(cls.temporary_directory.name),
            include_stress=False,
            repeat_count=1,
        )
        cls.case_rows = cls._read_csv(cls.case_path)
        cls.raw_rows = cls._read_csv(cls.raw_path)
        cls.summary_rows = cls._read_csv(cls.summary_path)
        cls.assessment_rows = cls._read_csv(cls.assessment_path)
        cls.metadata = json.loads(
            cls.metadata_path.read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_schemas_keep_evidence_and_policy_layers_separate(self) -> None:
        self.assertEqual(tuple(self.case_rows[0]), CASE_COLUMNS)
        self.assertEqual(tuple(self.raw_rows[0]), RAW_COLUMNS)
        self.assertEqual(tuple(self.summary_rows[0]), SUMMARY_COLUMNS)
        self.assertEqual(
            tuple(self.assessment_rows[0]),
            ASSESSMENT_COLUMNS,
        )
        self.assertEqual(len(self.case_rows), 16)
        self.assertEqual(len(self.raw_rows), 64)
        self.assertEqual(len(self.summary_rows), 64)
        self.assertEqual(len(self.assessment_rows), 128)
        self.assertEqual(SUITE_VERSION, "3")
        self.assertEqual(ARTIFACT_SCHEMA_VERSION, "2")
        self.assertEqual(self.metadata["suite_version"], "3")
        self.assertEqual(self.metadata["schema_version"], "2")
        self.assertTrue(
            all(
                row["correctly_rounded_reference_bits"]
                for row in self.raw_rows
            )
        )
        self.assertTrue(
            all(
                row["correctly_rounded_reference_bits"]
                for row in self.summary_rows
            )
        )
        self.assertTrue(
            all(
                "require_correct_rounding" not in row
                and "overall_passed" not in row
                for row in self.summary_rows
            )
        )
        self.assertEqual(
            {row["policy_name"] for row in self.assessment_rows},
            {"consumer_tolerance", "correct_rounding"},
        )

    def test_decimal_case_rows_preserve_exact_reference_split(self) -> None:
        decimal_cases = [
            row
            for row in self.case_rows
            if row["generator_name"] == "uniform_decimal_tail"
        ]
        self.assertEqual(len(decimal_cases), 4)
        for row in decimal_cases:
            source_sum = Fraction(row["source_sum_fraction"])
            stored_sum = Fraction(row["stored_sum_fraction"])
            input_error = Fraction(
                row["input_quantization_error_fraction"]
            )
            self.assertEqual(input_error, stored_sum - source_sum)
            self.assertLess(input_error, 0)
            self.assertEqual(row["tail_source_decimal"], "1E-8")
            self.assertEqual(
                row["correctly_rounded_stored_sum_bits"],
                {
                    "5": "0x3f800000",
                    "6": "0x3f800001",
                }[row["tail_count"]],
            )

    def test_canonical_json_recomputes_policy_and_case_ids(self) -> None:
        for row in self.assessment_rows:
            policy_json = row["policy_json"]
            self.assertEqual(
                hashlib.sha256(policy_json.encode("utf-8")).hexdigest(),
                row["policy_id"],
            )
        for row in (
            *self.case_rows,
            *self.raw_rows,
            *self.summary_rows,
            *self.assessment_rows,
        ):
            recipe_json = row["case_recipe_json"]
            self.assertEqual(
                hashlib.sha256(recipe_json.encode("utf-8")).hexdigest(),
                row["case_id"],
            )
            recipe_values = json.loads(recipe_json)
            self.assertEqual(
                row["generator_name"],
                recipe_values["generator_name"],
            )
            self.assertEqual(
                row["generator_version"],
                recipe_values["generator_version"],
            )
            self.assertEqual(
                row["tail_source_decimal"],
                recipe_values["parameters"].get("tail_source_decimal", ""),
            )

    def test_metadata_hashes_every_written_csv(self) -> None:
        for path in (
            self.case_path,
            self.raw_path,
            self.summary_path,
            self.assessment_path,
        ):
            self.assertEqual(
                self.metadata["artifacts"][path.name],
                source_file_sha256(path),
            )


if __name__ == "__main__":
    unittest.main()
