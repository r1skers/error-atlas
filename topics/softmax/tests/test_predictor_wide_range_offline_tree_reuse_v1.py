"""Tests for the frozen offline tree-reuse validation and completed artifacts."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from predictor_fixed_k8_beam_inference import SelectionResult
from predictor_wide_range_fixed_k8_beam_v2_heldout import (
    _reserve_output_directory,
    _sha256,
)
from predictor_wide_range_offline_tree_reuse_v1 import (
    EXPECTED_CALIBRATION_GROUPS,
    EXPECTED_CONFIRMATION_GROUPS,
    EXPECTED_REPRESENTATIVE_COUNTS,
    EXPECTED_WIDTHS,
    OUTPUT_DIRECTORY,
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
        self.assertEqual(
            config["offline_policy"]["representative_counts"],
            [1, 2, 4, 8, 16, 32],
        )
        self.assertEqual(config["graphs"]["candidate_count"], 64)
        self.assertEqual(
            config["online_cost_contract"]["score_static_selection_passes"],
            0,
        )

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
        self.assertEqual(
            [family for family, _ in catalog[:4]],
            ["contiguous", "pair_merge", "contiguous", "pair_merge"],
        )
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

    def test_completed_artifact_hashes_match_metadata(self) -> None:
        with (OUTPUT_DIRECTORY / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["status"], "completed")
        for name, record in metadata["artifacts"].items():
            path = OUTPUT_DIRECTORY / name
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(_sha256(path), record["sha256"])

    def test_completed_artifact_cardinality_and_seed_policy(self) -> None:
        with (OUTPUT_DIRECTORY / "calibration_inputs.jsonl").open(
            encoding="utf-8"
        ) as handle:
            calibration = [json.loads(line) for line in handle]
        with (OUTPUT_DIRECTORY / "confirmation_inputs.jsonl").open(encoding="utf-8") as handle:
            confirmation = [json.loads(line) for line in handle]
        self.assertEqual(
            len(calibration),
            len(EXPECTED_WIDTHS) * EXPECTED_CALIBRATION_GROUPS,
        )
        self.assertEqual(
            len(confirmation),
            len(EXPECTED_WIDTHS) * EXPECTED_CONFIRMATION_GROUPS,
        )
        for split, rows in (
            ("calibration", calibration),
            ("confirmation", confirmation),
        ):
            positions = {width: 0 for width in EXPECTED_WIDTHS}
            for row in rows:
                width = int(row["width"])
                self.assertEqual(
                    row["seed"],
                    _derived_seed(split, width, positions[width]),
                )
                positions[width] += 1

        with (OUTPUT_DIRECTORY / "calibration_graph_observations.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            calibration_graphs = sum(1 for _ in csv.DictReader(handle))
        with (OUTPUT_DIRECTORY / "confirmation_graph_observations.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            confirmation_graphs = sum(1 for _ in csv.DictReader(handle))
        self.assertEqual(calibration_graphs, len(calibration) * 64)
        self.assertEqual(confirmation_graphs, len(confirmation) * 64)

    def test_completed_frozen_decision_is_no_go(self) -> None:
        with (OUTPUT_DIRECTORY / "metric_summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)["overall"]
        gates = summary["engineering_gates"]
        self.assertTrue(gates["reuse_signal_over_random_fixed_catalog"])
        self.assertFalse(gates["score_static_beats_balanced_fp32"])
        self.assertFalse(gates["oracle_static_ceiling_beats_balanced_fp32"])
        self.assertFalse(gates["offline_reuse_deployment_go"])


if __name__ == "__main__":
    unittest.main()
