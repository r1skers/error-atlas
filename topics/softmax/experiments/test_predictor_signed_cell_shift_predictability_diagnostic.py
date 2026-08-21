"""Tests for signed FP32 cell-shift predictability."""

import unittest

import numpy as np

from predictor_ancestor_transition_predictability_diagnostic import (
    SHADOW_DIMENSION,
    TransitionSample,
)
from predictor_signed_cell_shift_predictability_diagnostic import (
    SHIFT_STATES,
    _clip_shift,
    _fit_probe,
    _metrics,
    _predict_probe,
    _softmax,
)


def _sample(value: float, shift: int, index: int) -> TransitionSample:
    return TransitionSample(
        family="test",
        node_index=index,
        ancestor_index=100 + index,
        gap=1,
        features=(value,) + (0.0,) * (SHADOW_DIMENSION - 1),
        crossing=int(shift != 0),
        sign_flip=int(shift < 0),
        wrong_cell=int(shift != 0),
        cell_shift=shift,
        innovation_shift=shift,
        predicted_crossing=0,
        predicted_sign_flip=0,
    )


class SignedCellShiftPredictabilityDiagnosticTests(unittest.TestCase):
    def test_shift_tail_clipping(self) -> None:
        self.assertEqual([_clip_shift(value) for value in (-9, -2, -1, 0, 1, 2, 9)], [-2, -2, -1, 0, 1, 2, 2])

    def test_softmax_rows_sum_to_one(self) -> None:
        probability = _softmax(np.asarray([[1.0, 2.0, 3.0], [-10.0, 0.0, 10.0]]))

        np.testing.assert_allclose(probability.sum(axis=1), np.ones(2))
        self.assertTrue((probability > 0.0).all())

    def test_multiclass_probe_learns_ordered_separable_states(self) -> None:
        samples = [
            _sample(float(state) + jitter, state, index)
            for index, (state, jitter) in enumerate(
                (state, jitter)
                for state in SHIFT_STATES
                for jitter in (-0.1, 0.0, 0.1)
            )
        ]

        model = _fit_probe(samples, 1)
        probability = _predict_probe(model, samples, 1)
        metric = _metrics(samples, probability, model.train_prevalence.copy())

        self.assertGreater(metric.accuracy, metric.majority_accuracy)
        self.assertGreater(metric.expected_shift_rho, 0.9)

    def test_baseline_probability_has_zero_information_gain(self) -> None:
        samples = [_sample(float(index), shift, index) for index, shift in enumerate((-1, 0, 0, 1))]
        baseline = np.asarray([0.0, 0.25, 0.5, 0.25, 0.0])
        probability = np.tile(baseline, (len(samples), 1))

        metric = _metrics(samples, probability, baseline.copy())

        self.assertAlmostEqual(metric.log_gain_bits, 0.0)


if __name__ == "__main__":
    unittest.main()
