"""Tests for the oracle-free fixed-K8/B3 inference path."""

import csv
import json
import unittest
from pathlib import Path

from predictor_fixed_k8_beam_inference import (
    DEFAULT_MODEL_PATH,
    FP32_MAX_FINITE_BITS,
    InnovationModel,
    _bits_to_units,
    _round_exact_plus_float_to_fp32_bits,
    _round_units_to_fp32_bits,
    select_tree,
)
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)


HERE = Path(__file__).resolve().parent
HELDOUT = HERE / "results" / "wide_range_fixed_k8_beam_v2" / "heldout"


def _heldout_graphs(width: int, group_index: int):
    graph_group = 1000 + group_index
    graphs = []
    for graph_index in range(32):
        graphs.append(
            random_contiguous_split_graph(
                width,
                seed=45_000_000 + graph_group * 10_000 + graph_index,
            )
        )
        graphs.append(
            random_pair_merge_graph(
                width,
                seed=46_000_000 + graph_group * 10_000 + graph_index,
            )
        )
    return graphs


class FixedK8BeamInferenceTests(unittest.TestCase):
    def test_binary32_lattice_round_trip(self) -> None:
        for bits in (
            0,
            1,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0x40000000,
            FP32_MAX_FINITE_BITS,
        ):
            self.assertEqual(_round_units_to_fp32_bits(_bits_to_units(bits)), bits)

    def test_exact_plus_float_uses_ties_to_even(self) -> None:
        one = _bits_to_units(0x3F800000)
        half_ulp = 1 << (126 - 1)
        self.assertEqual(
            _round_exact_plus_float_to_fp32_bits(one, 2.0**-24),
            0x3F800000,
        )
        self.assertEqual(
            _round_units_to_fp32_bits(one + half_ulp + 1),
            0x3F800001,
        )

    def test_model_record_shape_is_frozen(self) -> None:
        model = InnovationModel.from_json(DEFAULT_MODEL_PATH)
        self.assertEqual(model.feature_mean.shape, (19,))
        self.assertEqual(model.feature_scale.shape, (19,))
        self.assertEqual(model.weights.shape, (20, 5))

    def test_first_v2_group_reproduces_every_frozen_score(self) -> None:
        with (HELDOUT / "input_groups.jsonl").open(encoding="utf-8") as handle:
            input_record = json.loads(next(handle))
        rows = []
        with (HELDOUT / "graph_observations.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                if row["input_group_id"] == input_record["input_group_id"]:
                    rows.append(row)
                elif rows:
                    break

        result = select_tree(
            input_record["stored_leaf_bits"],
            _heldout_graphs(input_record["width"], 0),
            InnovationModel.from_json(HELDOUT / "calibration_model.json"),
        )
        expected_shortlist = tuple(
            sorted(
                (index for index, row in enumerate(rows) if row["shortlisted"] == "1"),
                key=lambda index: (float(rows[index]["fixed_q_score"]), index),
            )
        )
        expected_q = next(
            index for index, row in enumerate(rows) if row["q_selected"] == "1"
        )
        expected_beam = next(
            index for index, row in enumerate(rows) if row["beam_selected"] == "1"
        )

        self.assertEqual(result.shortlist_indices, expected_shortlist)
        self.assertEqual(result.q_selected_index, expected_q)
        self.assertEqual(result.selected_index, expected_beam)
        self.assertEqual(
            result.q_scores,
            tuple(float(row["fixed_q_score"]) for row in rows),
        )
        for index in result.shortlist_indices:
            self.assertEqual(result.beam_scores[index], float(rows[index]["beam_score"]))


if __name__ == "__main__":
    unittest.main()
