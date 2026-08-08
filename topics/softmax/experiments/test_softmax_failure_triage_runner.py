"""Tests for the reproducible Softmax failure-triage suite runner."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from softmax_failure_triage import RunAcceptancePolicy, source_file_sha256
from softmax_failure_triage_runner import (
    RAW_COLUMNS,
    SUMMARY_COLUMNS,
    registered_candidates,
    registered_cases,
    run_suite,
    sequential_fp64_accumulator_to_fp32,
    write_artifacts,
)


class RegisteredSuiteTests(unittest.TestCase):
    def test_stress_case_is_opt_in_and_does_not_change_smoke_cases(self) -> None:
        smoke = registered_cases(include_stress=False)
        with_stress = registered_cases(include_stress=True)

        self.assertEqual(
            [recipe.parameters["tail_count"] for _, recipe in smoke],
            [2, 2**10],
        )
        self.assertEqual(with_stress[: len(smoke)], smoke)
        self.assertEqual(with_stress[-1][0], "stress")
        self.assertEqual(
            with_stress[-1][1].parameters["tail_count"],
            2**20,
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


class SuiteExecutionTests(unittest.TestCase):
    def test_smoke_suite_preserves_runs_and_crosses_the_tolerance_boundary(
        self,
    ) -> None:
        raw_rows, summary_rows, metadata = run_suite(
            include_stress=False,
            repeat_count=2,
            policy=RunAcceptancePolicy(
                max_absolute_relative_error_tolerance=1e-6,
                require_bitwise_repeatability=True,
            ),
        )

        self.assertEqual(len(raw_rows), 2 * 4 * 2)
        self.assertEqual(len(summary_rows), 2 * 4)
        self.assertEqual(metadata["raw_row_count"], len(raw_rows))
        self.assertEqual(metadata["summary_row_count"], len(summary_rows))
        self.assertEqual(
            set(metadata["pipeline_sources"]),
            {
                "softmax_failure_triage.py",
                "softmax_failure_triage_runner.py",
            },
        )

        by_case_candidate = {
            (row["tail_count"], row["candidate_name"]): row
            for row in summary_rows
        }
        self.assertTrue(by_case_candidate[(2, "sequential_fp32")]["overall_passed"])
        self.assertFalse(
            by_case_candidate[(2**10, "sequential_fp32")]["overall_passed"]
        )
        for candidate_name in (
            "pairwise_fp32",
            "compensated_fp32",
            "sequential_fp64_accumulator",
        ):
            self.assertTrue(
                by_case_candidate[(2**10, candidate_name)]["overall_passed"]
            )

    def test_written_artifacts_keep_raw_and_derived_rows_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path, summary_path, metadata_path = write_artifacts(
                Path(directory),
                include_stress=False,
                repeat_count=1,
            )
            with raw_path.open(encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with summary_path.open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(tuple(raw_rows[0]), RAW_COLUMNS)
            self.assertEqual(tuple(summary_rows[0]), SUMMARY_COLUMNS)
            self.assertEqual(len(raw_rows), 8)
            self.assertEqual(len(summary_rows), 8)
            self.assertEqual(
                metadata["artifacts"][raw_path.name],
                source_file_sha256(raw_path),
            )
            self.assertEqual(
                metadata["artifacts"][summary_path.name],
                source_file_sha256(summary_path),
            )


if __name__ == "__main__":
    unittest.main()
