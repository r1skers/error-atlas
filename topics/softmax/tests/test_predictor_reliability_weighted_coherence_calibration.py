"""Tests for reliability-weighted sparse coherence."""

import unittest

from predictor_reliability_weighted_coherence_calibration import (
    GainPair,
    NodeRecord,
    ReliabilityTable,
    TreeSample,
    _fit_reliability,
    _rank_bucket,
    _reliability_score,
    _selection_utility,
)


class ReliabilityWeightedCoherenceCalibrationTests(unittest.TestCase):
    def test_rank_buckets_match_absolute_root_band_ranges(self) -> None:
        self.assertEqual(
            [_rank_bucket(rank) for rank in (0, 3, 4, 7, 8, 15, 16, 31)],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )
        with self.assertRaises(ValueError):
            _rank_bucket(-1)

    def test_fitted_gains_are_clipped_to_unit_interval(self) -> None:
        records = tuple(
            NodeRecord(
                rank=index,
                event=False,
                large_drift=False,
                predicted=1.0,
                actual=2.0,
                energy=1.0,
            )
            for index in range(32)
        )
        sample = TreeSample(
            family="test",
            target=0.0,
            full_q=2.0,
            q_budget={4: 1.0},
            unweighted_phase={4: 1.0},
            nodes=records,
        )

        table = _fit_reliability([sample])

        self.assertEqual(table.global_gain.sign, 1.0)
        self.assertEqual(table.global_gain.ols, 1.0)

    def test_negative_relationship_shrinks_to_zero(self) -> None:
        records = tuple(
            NodeRecord(
                rank=index,
                event=False,
                large_drift=False,
                predicted=1.0,
                actual=-1.0,
                energy=1.0,
            )
            for index in range(32)
        )
        sample = TreeSample("test", 0.0, 2.0, {4: 1.0}, {4: 1.0}, records)

        table = _fit_reliability([sample])

        self.assertEqual(table.global_gain.sign, 0.0)
        self.assertEqual(table.global_gain.ols, 0.0)

    def test_zero_gain_returns_macro_baseline(self) -> None:
        record = NodeRecord(0, True, True, 0.25, -0.25, 1.0)
        sample = TreeSample("test", 0.0, 3.0, {1: 1.0}, {1: 1.0}, (record,))
        table = ReliabilityTable(GainPair(0.0, 0.0, 100), {})

        score = _reliability_score(sample, 1, table, "sign", binned=False)

        self.assertEqual(score, sample.full_q / 12.0)

    def test_invalid_mode_is_rejected(self) -> None:
        record = NodeRecord(0, False, False, 0.25, 0.25, 1.0)
        table = ReliabilityTable(GainPair(0.5, 0.5, 100), {})
        with self.assertRaises(ValueError):
            table.gain(record, "bad", binned=False)

    def test_selection_utility_ignores_target_ties(self) -> None:
        utility = _selection_utility(
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0],
        )

        self.assertEqual(utility.pairwise_accuracy, 1.0)
        self.assertEqual(utility.best_tier_hit, 1.0)
        self.assertEqual(utility.normalized_regret, 0.0)

    def test_selection_utility_rejects_mismatched_vectors(self) -> None:
        with self.assertRaises(ValueError):
            _selection_utility([1.0], [])


if __name__ == "__main__":
    unittest.main()
