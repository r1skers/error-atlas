"""Tests for the selected-root-band joint cell beam."""

import math
import unittest
from dataclasses import replace

from predictor_ancestor_cell_beam_score_calibration import (
    FP32_MAX_FINITE_BITS,
    _beam_root_states,
    _beam_tree,
    _fp32_bits_to_fraction,
)
from predictor_ancestor_transition_predictability_diagnostic import (
    SHADOW_DIMENSION,
)
from predictor_calibration_inputs import wide_range_random
from predictor_signed_cell_shift_predictability_diagnostic import _fit_probe
from summation_graph_predictor import (
    FP32_MAX_FINITE,
    balanced_reduction_graph,
    round_nonnegative_fraction_to_fp32,
)


class AncestorCellBeamScoreCalibrationTests(unittest.TestCase):
    def test_positive_finite_bits_round_trip(self) -> None:
        for bits in (0, 1, 0x3F800000, 0x40000000, FP32_MAX_FINITE_BITS):
            value = _fp32_bits_to_fraction(bits)
            self.assertEqual(round_nonnegative_fraction_to_fp32(value).bits, bits)
        self.assertEqual(_fp32_bits_to_fraction(FP32_MAX_FINITE_BITS), FP32_MAX_FINITE)

    def test_full_selected_tree_has_zero_oracle_innovation(self) -> None:
        values = wide_range_random(8, seed=22260821).values
        graph = balanced_reduction_graph(8)

        tree = _beam_tree(values, graph, "balanced", 7)

        self.assertEqual(len(tree.transitions), 7)
        self.assertTrue(all(sample.innovation_shift == 0 for sample in tree.transitions))

    def test_beam_probabilities_are_normalized(self) -> None:
        values = wide_range_random(16, seed=22260821).values
        graph = balanced_reduction_graph(16)
        tree = _beam_tree(values, graph, "balanced", 8)
        model = _fit_probe(
            list(tree.transitions) * 8,
            SHADOW_DIMENSION,
            label="innovation_shift",
        )

        states = _beam_root_states(tree, model, 3)

        self.assertLessEqual(len(states), 3)
        self.assertAlmostEqual(sum(state.probability for state in states), 1.0)
        self.assertTrue(
            all(
                0 <= state.bits <= FP32_MAX_FINITE_BITS
                and math.isfinite(state.probability)
                and state.probability > 0.0
                for state in states
            )
        )

    def test_evaluation_labels_do_not_change_beam(self) -> None:
        values = wide_range_random(16, seed=22260821).values
        graph = balanced_reduction_graph(16)
        tree = _beam_tree(values, graph, "balanced", 8)
        model = _fit_probe(
            list(tree.transitions) * 8,
            SHADOW_DIMENSION,
            label="innovation_shift",
        )
        original = _beam_root_states(tree, model, 5)
        corrupted = replace(
            tree,
            target=999.0,
            transitions=tuple(
                replace(
                    sample,
                    crossing=1 - sample.crossing,
                    sign_flip=1 - sample.sign_flip,
                    wrong_cell=1 - sample.wrong_cell,
                    cell_shift=99,
                    innovation_shift=-99,
                )
                for sample in tree.transitions
            ),
        )

        self.assertEqual(_beam_root_states(corrupted, model, 5), original)


if __name__ == "__main__":
    unittest.main()
