"""Tests for the frozen offline tree-reuse validation without opening confirmation."""

import tempfile
import unittest
from pathlib import Path

from predictor_fixed_k8_beam_inference import SelectionResult
from predictor_wide_range_fixed_k8_beam_v2_heldout import _reserve_output_directory
from predictor_wide_range_offline_tree_reuse_v1 import (
    EXPECTED_CALIBRATION_GROUPS,
    EXPECTED_CONFIRMATION_GROUPS,
    EXPECTED_REPRESENTATIVE_COUNTS,
    EXPECTED_WIDTHS,
    PREREGISTRATION,
    _cascade_order,
    _catalog,
    _derived_seed,
    _fp64_then_fp32_bits,
    _kahan_fp32_bits,
    _load_and_validate_preregistration,
    _minimum_mean_index,
    _normalized_regrets,
    _rank_vector,
)


class WideRangeOfflineTreeReuseV1Tests(unittest.TestCase):
    def test_frozen_preregistration_is_self_consistent(self) -> None:
        config = _load_and_validate_preregistration(PREREGISTRATION)
        self.assertEqual(config["offline_policy"]["representative_counts"], [1, 2, 4, 8, 16, 32])
        self.assertEqual(config["graphs"]["candidate_count"], 64)
        self.assertEqual(config["online_cost_contract"]["score_static_selection_passes"], 0)

    def test_new_seed_splits_are_unique(self) -> None:
        calibration = [
            _derived_seed("calibration", width, index)
            for width in EXPECTED_WIDTHS
            for index in range(EXPECTED_CALIBRATION_GROUPS)
        ]
        confirmation = [
            _derived_seed("confirmation", width, index)
            for width in EXPECTED_WIDTHS
            for index in range(EXPECTED_CONFIRMATION_GROUPS)
        ]
        self.assertEqual(len(calibration), len(set(calibration)))
        self.assertEqual(len(confirmation), len(set(confirmation)))
        self.assertFalse(set(calibration) & set(confirmation))

    def test_catalog_is_fixed_and_interleaves_families(self) -> None:
        catalog = _catalog(256)
        self.assertEqual(len(catalog), 64)
        self.assertEqual([family for family, _ in catalog[:4]], ["contiguous", "pair_merge", "contiguous", "pair_merge"])
        self.assertIs(catalog, _catalog(256))

    def test_cascade_rank_places_beam_shortlist_first(self) -> None:
        result = SelectionResult(
            selected_index=2,
            q_selected_index=1,
            shortlist_indices=(1, 2),
            q_scores=(3.0, 1.0, 2.0, 4.0),
            beam_scores=(None, 0.5, 0.25, None),
        )
        order = _cascade_order(result)
        self.assertEqual(order, (2, 1, 0, 3))
        self.assertEqual(_rank_vector(order, 4), (2, 1, 0, 3))

    def test_minimum_mean_selection_is_stable(self) -> None:
        self.assertEqual(_minimum_mean_index([(0.0, 2.0), (2.0, 0.0)]), 0)
        self.assertEqual(_minimum_mean_index([(5.0, 1.0), (3.0, 1.0)]), 1)

    def test_normalized_regret_handles_ties(self) -> None:
        self.assertEqual(_normalized_regrets((2.0, 2.0)), (0.0, 0.0))
        self.assertEqual(_normalized_regrets((1.0, 2.0, 3.0)), (0.0, 0.5, 1.0))

    def test_direct_sum_helpers_preserve_exact_small_sum(self) -> None:
        one = 0x3F800000
        two = 0x40000000
        self.assertEqual(_fp64_then_fp32_bits((one, one)), two)
        self.assertEqual(_kahan_fp32_bits((one, one)), two)

    def test_output_reservation_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "confirmation"
            _reserve_output_directory(output)
            with self.assertRaises(FileExistsError):
                _reserve_output_directory(output)

    def test_representative_counts_end_at_full_calibration(self) -> None:
        self.assertEqual(EXPECTED_REPRESENTATIVE_COUNTS[-1], EXPECTED_CALIBRATION_GROUPS)


if __name__ == "__main__":
    unittest.main()
