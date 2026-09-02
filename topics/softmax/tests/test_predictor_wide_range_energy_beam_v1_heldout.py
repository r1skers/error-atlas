"""Tests for the frozen wide-range held-out runner without opening held-out data."""

import tempfile
import unittest
from pathlib import Path

from predictor_wide_range_energy_beam_v1_heldout import (
    _bootstrap_interval,
    _global_regret,
    _load_and_validate_preregistration,
    _percentile,
    _reserve_output_directory,
)


class WideRangeEnergyBeamV1HeldoutTests(unittest.TestCase):
    def test_frozen_preregistration_is_self_consistent(self) -> None:
        from predictor_wide_range_energy_beam_v1_heldout import PREREGISTRATION

        config = _load_and_validate_preregistration(PREREGISTRATION)
        self.assertEqual(config["predictor"]["energy_mass"], 0.8)
        self.assertEqual(config["metrics"]["primary"].split(":", 1)[0],
                         "mean paired normalized-regret improvement")

    def test_output_reservation_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "heldout"
            _reserve_output_directory(output)
            with self.assertRaises(FileExistsError):
                _reserve_output_directory(output)

    def test_global_regret_uses_all_candidates(self) -> None:
        self.assertEqual(_global_regret([1.0, 2.0, 5.0], 1), 0.25)
        self.assertEqual(_global_regret([3.0, 3.0], 0), 0.0)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(_percentile([0.0, 10.0], 0.25), 2.5)

    def test_bootstrap_preserves_width_strata(self) -> None:
        rows = [
            {"width": 1, "value": 0.0},
            {"width": 1, "value": 0.0},
            {"width": 2, "value": 2.0},
            {"width": 2, "value": 2.0},
        ]
        interval = _bootstrap_interval(
            rows,
            lambda row: row["value"],
            resamples=20,
            seed=1,
            stratify=True,
        )
        self.assertEqual(interval, (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
