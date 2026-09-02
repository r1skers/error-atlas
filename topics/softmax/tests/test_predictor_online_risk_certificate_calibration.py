"""Tests for the calibration-only online reduction risk certificate."""

import csv
import json
import math
import unittest
from fractions import Fraction

from predictor_online_risk_certificate_calibration import (
    EXPECTED_GROUPS_PER_WIDTH,
    EXPECTED_WIDTHS,
    OUTPUT_DIRECTORY,
    _cross_fit,
    _derived_seed,
    _load_and_validate_preregistration,
    _normal_cell_probability,
    _root_error_cell,
    trace_online_certificate,
)
from predictor_wide_range_fixed_k8_beam_v2_heldout import _sha256
from predictor_shadow_sparse_repair_ablation import _fp32_ulp_fraction
from summation_graph_predictor import (
    balanced_reduction_graph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


class OnlineRiskCertificateCalibrationTests(unittest.TestCase):
    def test_frozen_preregistration_matches_runner(self) -> None:
        config = _load_and_validate_preregistration()
        self.assertEqual(config["data_boundary"]["widths"], [256, 512, 1024])
        self.assertEqual(config["frozen_variance_proxies"]["primary"], "q_inexact")

    def test_derived_seeds_are_unique(self) -> None:
        seeds = [
            _derived_seed(width, index)
            for width in EXPECTED_WIDTHS
            for index in range(EXPECTED_GROUPS_PER_WIDTH)
        ]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_all_exact_balanced_tree_has_only_all_node_energy(self) -> None:
        values = (Fraction(1), Fraction(1), Fraction(1), Fraction(1))
        bits = tuple(round_nonnegative_fraction_to_fp32(value).bits for value in values)
        state = trace_online_certificate(bits, balanced_reduction_graph(4))
        self.assertEqual(state.bits, 0x40800000)
        self.assertEqual(state.internal_count, 3)
        self.assertEqual(state.inexact_count, 0)
        self.assertAlmostEqual(state.q_all, 1.5 / 12.0)
        self.assertEqual(state.q_inexact, 0.0)
        self.assertEqual(state.b_inexact, 0.0)
        self.assertEqual(state.top_energy_inexact, ())

    def test_midpoint_add_is_inexact_and_bound_is_tight(self) -> None:
        values = (Fraction(1), Fraction(1, 1 << 24))
        bits = tuple(round_nonnegative_fraction_to_fp32(value).bits for value in values)
        graph = balanced_reduction_graph(2)
        state = trace_online_certificate(bits, graph)
        oracle = predict_fp32_tree_error(values, graph)
        root_ulp = Fraction(1, 1 << 23)
        error = float(oracle.signed_error / root_ulp)
        self.assertEqual(state.bits, 0x3F800000)
        self.assertEqual(state.inexact_count, 1)
        self.assertAlmostEqual(state.q_inexact, 1.0 / 12.0)
        self.assertAlmostEqual(state.b_inexact, 0.5)
        self.assertEqual(error, -0.5)
        self.assertLessEqual(abs(error), state.b_inexact)

    def test_online_root_matches_oracle_and_bound_on_irregular_values(self) -> None:
        values = (
            Fraction(1),
            Fraction(1, 1 << 24),
            Fraction(3, 1 << 25),
            Fraction(5, 1 << 28),
            Fraction(7, 1 << 30),
        )
        bits = tuple(round_nonnegative_fraction_to_fp32(value).bits for value in values)
        graph = balanced_reduction_graph(len(values))
        state = trace_online_certificate(bits, graph)
        oracle = predict_fp32_tree_error(values, graph)
        self.assertEqual(state.bits, int(oracle.predicted_sum_bits, 16))
        root_ulp = Fraction(1, 1 << 23)
        error = abs(float(oracle.signed_error / root_ulp))
        self.assertLessEqual(error, state.b_all)
        self.assertLessEqual(error, state.b_inexact)
        self.assertLessEqual(state.q_inexact, state.q_all)
        self.assertLessEqual(state.b_inexact, state.b_all)

        direct_q_all = sum(
            float(_fp32_ulp_fraction(node.exact_addend_sum) / root_ulp) ** 2
            / 12.0
            for node in oracle.node_predictions
        )
        direct_q_inexact = sum(
            float(_fp32_ulp_fraction(node.exact_addend_sum) / root_ulp) ** 2
            / 12.0
            for node in oracle.node_predictions
            if node.local_rounding_error
        )
        direct_b_inexact = sum(
            0.5 * float(_fp32_ulp_fraction(node.exact_addend_sum) / root_ulp)
            for node in oracle.node_predictions
            if node.local_rounding_error
        )
        self.assertAlmostEqual(state.q_all, direct_q_all)
        self.assertAlmostEqual(state.q_inexact, direct_q_inexact)
        self.assertGreaterEqual(state.b_inexact, direct_b_inexact)

    def test_root_cell_at_one_has_asymmetric_binade_margins(self) -> None:
        low, high = _root_error_cell(0x3F800000, -23)
        self.assertEqual(low, -0.5)
        self.assertEqual(high, 0.25)

    def test_normal_cell_probability_matches_symmetric_erf_case(self) -> None:
        probability = _normal_cell_probability(-0.5, 0.5, 0.0, 0.4)
        expected = math.erf(0.5 / (math.sqrt(2.0) * 0.4))
        self.assertAlmostEqual(probability, expected)

    def test_cross_fit_does_not_use_test_fold_label(self) -> None:
        rows = []
        for index in range(10):
            rows.append(
                {
                    "fold": index % 5,
                    "q_all": 1.0,
                    "q_inexact": 1.0,
                    "q_corr4_all": 1.0,
                    "q_corr4_inexact": 1.0,
                    "signed_error_root_ulp": float(index + 1),
                    "cell_error_low_root_ulp": -0.5,
                    "cell_error_high_root_ulp": 0.5,
                    "correctly_rounded": index % 2 == 0,
                }
            )
        predictions, fits = _cross_fit(rows, "q_inexact", "bias_aware")
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual(len(fits), 5)
        fold_zero_train = [1, 2, 3, 4, 6, 7, 8, 9]
        expected_beta = sum(index + 1 for index in fold_zero_train) / len(
            fold_zero_train
        )
        self.assertAlmostEqual(fits[0]["beta_standardized"], expected_beta)
        self.assertAlmostEqual(predictions[0].mu, expected_beta)

    def test_completed_artifact_hashes_match_metadata(self) -> None:
        with (OUTPUT_DIRECTORY / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["status"], "completed_calibration_only")
        self.assertEqual(
            metadata["git_commit_before_opening"],
            "54245d610e1a9bc3356bddb1619ce2c5a02ef3f3",
        )
        for name, record in metadata["artifacts"].items():
            path = OUTPUT_DIRECTORY / name
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(_sha256(path), record["sha256"])

    def test_completed_artifact_cardinality_and_primary_decision(self) -> None:
        with (OUTPUT_DIRECTORY / "observations.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            len(rows), len(EXPECTED_WIDTHS) * EXPECTED_GROUPS_PER_WIDTH
        )
        self.assertEqual(
            {int(row["width"]) for row in rows}, set(EXPECTED_WIDTHS)
        )
        self.assertEqual(sum(row["correctly_rounded"] == "True" for row in rows), 143)

        with (OUTPUT_DIRECTORY / "model_summary.json").open(
            encoding="utf-8"
        ) as handle:
            summary = json.load(handle)
        primary = summary["models"]["q_inexact__bias_aware"]
        self.assertAlmostEqual(primary["coverage"]["90"], 173 / 192)
        self.assertEqual(primary["coverage"]["99"], 1.0)
        self.assertTrue(summary["decision"]["primary_gaussian_viable"])
        self.assertTrue(summary["decision"]["primary_probability_threshold_gate"])
        self.assertFalse(summary["decision"]["rigorous_cell_certificate_nonzero"])


if __name__ == "__main__":
    unittest.main()
