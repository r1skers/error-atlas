"""Tests for the frozen fixed-K8/B3 confirmation without opening held-out data."""

import tempfile
import unittest
from pathlib import Path

from predictor_wide_range_fixed_k8_beam_v2_heldout import (
    EXPECTED_GROUPS_PER_WIDTH,
    EXPECTED_WIDTHS,
    PREREGISTRATION,
    _derived_seed,
    _load_and_validate_preregistration,
    _metric_block,
    _reserve_output_directory,
    _stable_min,
)


class WideRangeFixedK8BeamV2HeldoutTests(unittest.TestCase):
    def test_frozen_preregistration_is_self_consistent(self) -> None:
        config = _load_and_validate_preregistration(PREREGISTRATION)
        self.assertEqual(config["predictor"]["root_band_budget"], 8)
        self.assertEqual(config["predictor"]["shortlist_size"], 4)
        self.assertEqual(config["predictor"]["beam_width"], 3)
        self.assertEqual(config["uncertainty"]["resamples"], 20_000)
        self.assertEqual(
            config["metrics"]["primary"].split(":", 1)[0],
            "mean paired normalized-regret improvement",
        )

    def test_all_frozen_seeds_match_policy_and_are_unique(self) -> None:
        config = _load_and_validate_preregistration(PREREGISTRATION)
        observed = []
        for width in EXPECTED_WIDTHS:
            expected = [
                _derived_seed(width, index)
                for index in range(EXPECTED_GROUPS_PER_WIDTH)
            ]
            self.assertEqual(config["heldout_seeds"][str(width)], expected)
            observed.extend(expected)
        self.assertEqual(len(observed), len(set(observed)))

    def test_output_reservation_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "heldout"
            _reserve_output_directory(output)
            with self.assertRaises(FileExistsError):
                _reserve_output_directory(output)

    def test_stable_min_uses_graph_index_for_ties(self) -> None:
        self.assertEqual(_stable_min((3, 1, 2), {1: 0.5, 2: 0.5, 3: 1.0}), 1)

    def test_metric_block_primary_direction_and_tail_diagnostics(self) -> None:
        rows = [
            {
                "width": width,
                "q_regret": 0.75,
                "beam_regret": 0.25,
                "q_best_hit": 0.0,
                "beam_best_hit": 1.0,
                "shortlist_best_tier_coverage": 1.0,
                "best_prevalence": 0.25,
                "random_expected_regret": 0.5,
                "beam_pairwise_accuracy": 0.75,
                "beam_rho": 0.5,
            }
            for width in EXPECTED_WIDTHS
            for _ in range(2)
        ]
        summary = _metric_block(rows, resamples=20, seed=1)
        self.assertEqual(summary["primary_fixed_q_minus_beam_regret"], 0.5)
        self.assertEqual(summary["primary_95_ci"], [0.5, 0.5])
        self.assertTrue(summary["positive_evidence"])
        self.assertEqual(summary["beam_benefit_rate"], 1.0)
        self.assertEqual(summary["beam_harm_rate"], 0.0)
        self.assertEqual(summary["beam_regret_p90"], 0.25)
        self.assertEqual(summary["q_severe_regret_rate"], 1.0)

    def test_metric_block_counts_harm_and_exact_ties(self) -> None:
        base = {
            "width": 256,
            "q_best_hit": 0.0,
            "beam_best_hit": 0.0,
            "shortlist_best_tier_coverage": 1.0,
            "best_prevalence": 0.25,
            "random_expected_regret": 0.5,
            "beam_pairwise_accuracy": None,
            "beam_rho": None,
        }
        rows = [
            {**base, "q_regret": 0.0, "beam_regret": 1.0},
            {**base, "q_regret": 0.5, "beam_regret": 0.5},
        ]
        summary = _metric_block(rows, resamples=20, seed=2)
        self.assertEqual(summary["beam_benefit_rate"], 0.0)
        self.assertEqual(summary["beam_harm_rate"], 0.5)
        self.assertEqual(summary["beam_q_tie_rate"], 0.5)
        self.assertEqual(summary["beam_pairwise_defined_groups"], 0)
        self.assertEqual(summary["beam_spearman_defined_groups"], 0)


if __name__ == "__main__":
    unittest.main()
