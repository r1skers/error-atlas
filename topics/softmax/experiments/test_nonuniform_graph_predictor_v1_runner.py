"""Regression tests for the single preregistered nonuniform case."""

import unittest

from nonuniform_graph_predictor_v1_runner import (
    PREREGISTRATION_SHA256,
    _sha256,
    execute_preregistered_case,
    PREREGISTRATION_PATH,
)


class NonuniformGraphPredictorV1Tests(unittest.TestCase):
    def test_preregistration_hash_remains_frozen(self) -> None:
        self.assertEqual(_sha256(PREREGISTRATION_PATH), PREREGISTRATION_SHA256)

    def test_two_graph_observations_match_the_frozen_predictions(self) -> None:
        execution = execute_preregistered_case()
        self.assertEqual(len(execution.rows), 2)
        by_graph = {row["graph_name"]: row for row in execution.rows}

        sequential = by_graph["sequential_left_to_right"]
        self.assertEqual(sequential["actual_output_bits"], "0x3f800001")
        self.assertEqual(
            sequential["actual_signed_error_fraction"],
            "7/134217728",
        )
        self.assertTrue(sequential["prediction_matched_observation"])
        self.assertTrue(sequential["observed_candidate_correctly_rounded"])

        pairwise = by_graph["balanced_contiguous_floor_half"]
        self.assertEqual(pairwise["actual_output_bits"], "0x3f800000")
        self.assertEqual(
            pairwise["actual_signed_error_fraction"],
            "-9/134217728",
        )
        self.assertTrue(pairwise["prediction_matched_observation"])
        self.assertFalse(pairwise["observed_candidate_correctly_rounded"])


if __name__ == "__main__":
    unittest.main()
