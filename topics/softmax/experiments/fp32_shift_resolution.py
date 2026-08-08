"""Probe loss of a unit logit difference under FP32 input quantization.

The research core is intentionally left for the learner.  This first scaffold
measures the stored logit difference and the first Softmax probability near the
FP32 consecutive-integer boundary.  Surrounding scaffolding records CSV evidence
and JSON provenance.
"""

import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


COMMON_OFFSETS = (2**23, 2**24, 2**25)
REFERENCE_DIFFERENCE = 1.0
REFERENCE_FIRST_PROBABILITY = 1.0 / (1.0 + math.exp(-REFERENCE_DIFFERENCE))

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "shift_resolution"
CSV_PATH = RESULTS_DIR / "fp32_shift_resolution.csv"
METADATA_PATH = RESULTS_DIR / "fp32_shift_resolution_metadata.json"
CSV_FIELDS = (
    "common_offset",
    "log2_common_offset",
    "ulp_at_common_offset",
    "reference_difference",
    "stored_difference",
    "absolute_difference_error",
    "reference_first_probability",
    "computed_first_probability",
    "absolute_probability_error",
)


def fp32_softmax_probe(common_offset: float) -> tuple[float, float]:
    """Return ``(stored_difference, first_probability)`` for ``(M + 1, M)``.

    Core invariants:
    - Construct the two ideal logits from ``common_offset`` and a unit gap.
    - Quantize both logits to FP32 *before* applying subtract-max.
    - Evaluate Softmax with subtract-max and FP32 intermediates.
    - Return ordinary Python ``float`` values for a stable public interface.

    Expected boundary behavior:
    - At ``M = 2**23``, the stored difference remains one.
    - At ``M = 2**24``, round-to-nearest-even collapses the unit difference.
    """

    logits = np.array(
        [common_offset + 1.0, common_offset],
        dtype=np.float32,
    )
    stored_difference = logits[0] - logits[1]

    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / np.sum(exponentials)

    return float(stored_difference), float(probabilities[0])


def experiment_rows() -> list[dict[str, float | int]]:
    """Return the three pre-registered probes with reference errors."""
    rows = []
    for common_offset in COMMON_OFFSETS:
        stored_difference, first_probability = fp32_softmax_probe(common_offset)
        rows.append(
            {
                "common_offset": common_offset,
                "log2_common_offset": int(math.log2(common_offset)),
                "ulp_at_common_offset": float(
                    np.spacing(np.float32(common_offset))
                ),
                "reference_difference": REFERENCE_DIFFERENCE,
                "stored_difference": stored_difference,
                "absolute_difference_error": abs(
                    stored_difference - REFERENCE_DIFFERENCE
                ),
                "reference_first_probability": REFERENCE_FIRST_PROBABILITY,
                "computed_first_probability": first_probability,
                "absolute_probability_error": abs(
                    first_probability - REFERENCE_FIRST_PROBABILITY
                ),
            }
        )
    return rows


def write_results(rows: list[dict[str, float | int]]) -> None:
    """Write versioned CSV evidence and JSON provenance."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    float32_info = np.finfo(np.float32)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python topics/softmax/experiments/fp32_shift_resolution.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "dtype": "float32",
        "float32": {
            "precision_bits_including_hidden_bit": int(float32_info.nmant + 1),
            "eps": float(float32_info.eps),
            "tiny": float(float32_info.tiny),
            "max": float(float32_info.max),
        },
        "numpy_error_settings": np.geterr(),
        "common_offsets": list(COMMON_OFFSETS),
        "reference_difference": REFERENCE_DIFFERENCE,
        "reference_first_probability": REFERENCE_FIRST_PROBABILITY,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run, print, and record the three pre-registered boundary probes."""
    rows = experiment_rows()
    write_results(rows)

    print("common_offset,stored_difference,first_probability,absolute_error")
    for row in rows:
        print(
            f"{row['common_offset']},"
            f"{row['stored_difference']:.9g},"
            f"{row['computed_first_probability']:.9g},"
            f"{row['absolute_probability_error']:.9g}"
        )
    print(f"wrote {CSV_PATH}")
    print(f"wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
