"""Tests for the score-only fidelity/cost benchmark helpers."""

import json
import unittest

import numpy as np

from predictor_fixed_k8_beam_inference import InnovationModel, select_tree
from predictor_fixed_k8_beam_inference_benchmark import (
    HELDOUT,
    _execute_graph_fp32,
    _heldout_graphs,
    _percentile,
    _profile_selector,
)


class FixedK8BeamInferenceBenchmarkTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(_percentile([1.0], 0.9), 1.0)
        self.assertEqual(_percentile([0.0, 10.0], 0.5), 5.0)

    def test_profile_matches_public_selector(self) -> None:
        with (HELDOUT / "input_groups.jsonl").open(encoding="utf-8") as handle:
            record = json.loads(next(handle))
        graphs = _heldout_graphs(record["width"], 0)[:4]
        model = InnovationModel.from_json(HELDOUT / "calibration_model.json")

        profile, measured = _profile_selector(record["stored_leaf_bits"], graphs, model)
        public = select_tree(record["stored_leaf_bits"], graphs, model)

        self.assertEqual(measured, public)
        self.assertEqual(
            set(profile),
            {"input_ms", "macro_ms", "q_total_ms", "beam_ms", "total_ms"},
        )
        self.assertTrue(all(value >= 0.0 for value in profile.values()))

    def test_graph_execution_keeps_fp32_state(self) -> None:
        with (HELDOUT / "input_groups.jsonl").open(encoding="utf-8") as handle:
            record = json.loads(next(handle))
        values = np.asarray(record["stored_leaf_bits"], dtype=np.uint32).view(np.float32)
        graph = _heldout_graphs(record["width"], 0)[0]

        result = _execute_graph_fp32(values, graph)

        self.assertIsInstance(result, np.float32)
        self.assertTrue(np.isfinite(result))


if __name__ == "__main__":
    unittest.main()
