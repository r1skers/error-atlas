"""Independent exact accounting audit of saved attribution data, without importing its core.

Checks serialized values and summaries, not a second numerical reconstruction of dumps.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def decode(value):
    if isinstance(value, dict) and set(value) == {"n_hex", "d_hex"}:
        return F(int(value["n_hex"], 16), int(value["d_hex"], 16))
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(decode(item) for item in value)
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enclosure(stored, actual):
    require(stored[0] <= actual[0] <= actual[1] <= stored[1], "Invalid enclosure")


def extrema_abs(low, high):
    return (F(0) if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high)))


def verify(parent, bundle):
    for directory in (parent, bundle):
        manifest = read(directory / "manifest.json")
        require(manifest["status"] == "complete", "Incomplete bundle")
        for name, digest in manifest["files"].items():
            require(sha(directory / name) == digest, f"File hash mismatch: {name}")
    require(read(bundle / "manifest.json")["parent_manifest_sha256"] == sha(parent / "manifest.json"), "Parent mismatch")
    plan = read(parent / "plan.json")
    rows = decode(read(bundle / "attributions.json"))
    pairs = decode(read(bundle / "paired_attributions.json"))
    summary = read(bundle / "summary.json")
    indexed, groups, pair_groups = {}, defaultdict(list), defaultdict(list)
    for row in rows:
        t = row["terms"]
        c, frozen, r, budget = t["computed"], t["frozen"], t["rounding_error"], t["rounding_budget"]
        low, high = t["reference"]
        require(0 < low <= high, "Invalid reference")
        require(r == c - frozen and budget >= abs(r), "Invalid R or budget")
        require(t["weight_error"] == (frozen - high, frozen - low), "Invalid W")
        require(t["total_error"] == (c - high, c - low), "Invalid E")
        require(t["relative_rounding"] == tuple(sorted((r / low, r / high))), "Invalid normalized R")
        require(t["relative_weight"] == (frozen / high - 1, frozen / low - 1), "Invalid normalized W")
        require(t["relative_total"] == (c / high - 1, c / low - 1), "Invalid normalized E")
        for key, field in (("abs_R", "relative_rounding"), ("abs_W", "relative_weight"), ("A", "relative_total")):
            require(row[key] == extrema_abs(*t[field]), f"Invalid magnitude {key}")
        require(row["budget_R"] == (budget / high, budget / low), "Invalid normalized budget")
        require(row["internal_savings"] == ((budget - abs(r)) / high, (budget - abs(r)) / low), "Invalid internal savings")
        # On each segment S(l)/l is a+b/l, hence extrema occur at endpoints or knots.
        reference_candidates = [low, high] + [point for point in (frozen, c) if low <= point <= high]
        values = [(abs(r) + abs(frozen - point) - abs(c - point)) / point for point in reference_candidates]
        enclosure(row["between_savings"], (min(values), max(values)))
        require(row["between_savings"][0] >= 0, "Negative cancellation savings")
        require(sum(row["residual_sign_counts"].values()) == plan["cells"][row["cell_index"]]["count"] - 1, "Wrong node count")
        key = (row["family_id"], row["schedule"], row["fused"])
        require(key not in indexed, "Duplicate attribution")
        indexed[key] = row
        groups[row["cell_index"], row["schedule"], row["fused"]].append(row)
    weight_invariance = 0
    for (family, schedule, fused), row in indexed.items():
        if fused and (family, schedule, False) in indexed:
            separate = indexed[family, schedule, False]
            for field in ("frozen", "reference", "weight_error", "relative_weight"):
                require(row["terms"][field] == separate["terms"][field], "FMA changed frozen weights")
            weight_invariance += 1
    for pair in pairs:
        chain, balanced = (indexed[pair["family_id"], name, pair["fused"]] for name in ("chain", "balanced"))
        low, high = chain["terms"]["reference"]
        numerator = abs(chain["terms"]["rounding_error"]) - abs(balanced["terms"]["rounding_error"])
        expected = tuple(sorted((numerator / low, numerator / high)))
        require(pair["D_R"] == expected, "Invalid rounding-only gap")
        a, b = chain["A"], balanced["A"]
        c, d = chain["terms"]["computed"], balanced["terms"]["computed"]
        gap = (F(0), F(0)) if c == d else (a[0] - b[1], a[1] - b[0])
        require(pair["D"] == gap, "Invalid original gap")
        candidates = [low, high] + [point for point in (c, d) if low <= point <= high]
        values = [(abs(c - point) - abs(d - point) - numerator) / point for point in candidates]
        enclosure(pair["weight_correction"], (min(values), max(values)))
        pair_groups[pair["cell_index"], pair["fused"]].append(pair)
    checked_means = 0
    for kind, grouped in (("cells", groups), ("comparisons", pair_groups)):
        for cell in summary[kind]:
            key = (cell["cell_index"], cell["schedule"], cell["fused"]) if kind == "cells" else (cell["cell_index"], cell["fused"])
            group = grouped[key]
            require(len(group) == len(plan["cells"][cell["cell_index"]]["seeds"]), "Incomplete group")
            for name, interval in cell.items():
                if not name.startswith("mean_"):
                    continue
                field = name.removeprefix("mean_")
                if field.startswith("signed_"):
                    data = [row["terms"]["relative_" + field.removeprefix("signed_")] for row in group]
                else:
                    data = [row[field] for row in group]
                average = tuple(sum((value[k] for value in data), F(0)) / len(data) for k in (0, 1))
                enclosure((F(interval["low"]), F(interval["high"])), average)
                checked_means += 1
    require(len(rows) == len(read(parent / "measurements.json")), "Measurement coverage mismatch")
    require(len(pairs) == len(read(parent / "paired_differences.json")), "Pair coverage mismatch")
    return {"status": "passed", "audit": "independent serialized accounting, not numerical dump replay",
            "measurements": len(rows), "pairs": len(pairs), "summary_enclosures_checked": checked_means,
            "FMA_weight_invariance_pairs": weight_invariance, "bundle_manifest_sha256": sha(bundle / "manifest.json"),
            "audit_source_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.parent, args.bundle)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
