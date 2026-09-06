"""Plumbing for the USER-WRITTEN attribution core over a saved pilot bundle.

This is a post-hoc diagnostic. It verifies the old root/error intervals in the same
pass that computes attribution, avoiding a separate redundant numerical replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from pathlib import Path

from . import attribution, pilot, pilot_runner, schedules
from .merge import merge_reduce

HERE = Path(__file__).resolve().parent
SOURCES = (*pilot_runner.SOURCE_NAMES, "attribution.py", "attribution_runner.py")
SCHEMA = "online-attribution-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(value):
    """Exact hexadecimal integers avoid Python's decimal integer-string size limit."""
    if isinstance(value, Fraction):
        return {"n_hex": hex(value.numerator), "d_hex": hex(value.denominator)}
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [pack(item) for item in value]
    return value


def unpack(value):
    if isinstance(value, dict) and set(value) == {"n_hex", "d_hex"}:
        return Fraction(int(value["n_hex"], 16), int(value["d_hex"], 16))
    if isinstance(value, dict):
        return {key: unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(unpack(item) for item in value)
    return value


def old_fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def absolute_interval(interval):
    low, high = interval
    return (Fraction(0) if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high)))


def divide_constant(value, reference):
    endpoints = (value / reference[0], value / reference[1])
    return min(endpoints), max(endpoints)


def mean(intervals):
    if not intervals or any(low > high for low, high in intervals):
        raise ValueError("Need nonempty ordered intervals")
    return tuple(sum((interval[k] for interval in intervals), Fraction(0)) / len(intervals) for k in (0, 1))


def decimal_interval(interval):
    result = {}
    for name, value, rounding in zip(("low", "high"), interval, (ROUND_FLOOR, ROUND_CEILING)):
        context = Context(prec=50, rounding=rounding)
        result[name] = str(context.divide(Decimal(value.numerator), Decimal(value.denominator)))
    return result


def load_verified_bundle(bundle):
    """Verify the frozen v1 inventory without modifying the old runner's source boundary."""
    manifest = read(bundle / "manifest.json")
    expected = {"plan.json", "prediction.md", "inputs.json", "graphs.json", "measurements.json",
                "paired_differences.json", *(f"sources/online/{name}.txt" for name in pilot_runner.SOURCE_NAMES)}
    if (manifest["schema"] != pilot_runner.SCHEMA or manifest["status"] != "complete"
            or set(manifest["files"]) != expected):
        raise ValueError("Need a complete supported pilot bundle")
    for name, checksum in manifest["files"].items():
        if digest(bundle / name) != checksum:
            raise ValueError(f"Input bundle integrity mismatch: {name}")
    for name in pilot_runner.SOURCE_NAMES:
        if (bundle / "sources/online" / (name + ".txt")).read_bytes() != (HERE / name).read_bytes():
            raise ValueError(f"Old measurement source mismatch: {name}")
    plan = read(bundle / "plan.json")
    pilot_runner.validate_plan(plan)
    if plan["schedules"] != ["chain", "balanced"]:
        raise ValueError("This diagnostic is specified for chain/balanced")
    return plan, read(bundle / "inputs.json"), read(bundle / "graphs.json"), read(bundle / "measurements.json"), read(bundle / "paired_differences.json")


def analyze(bundle, *, progress=False):
    plan, inputs, graphs, measurements, old_pairs = load_verified_bundle(bundle)
    expected = {(row["family_id"], row["graph_id"], row["fused"]): row for row in measurements}
    pair_map = {(row["family_id"], row["fused"]): row for row in old_pairs}
    if len(expected) != len(measurements) or len(pair_map) != len(old_pairs):
        raise ValueError("Duplicate saved measurements")
    rows, pairs, visited = [], [], set()
    for index, source in enumerate(inputs):
        family = pilot.BlockFamily(tuple(pilot_runner.from_word(word) for word in source["maxima_fp32"]),
                                   tuple(source["masses"]))
        group = {}
        for graph in graphs:
            if graph["leaf_count"] != len(family.maxima):
                continue
            schedule = schedules.Schedule(graph["leaf_count"], tuple(tuple(pair) for pair in graph["nodes"]), graph["kind"])
            for fused in plan["fused"]:
                key = (source["family_id"], graph["graph_id"], fused)
                if key in visited:
                    raise ValueError("Duplicate graph/input combination")
                visited.add(key)
                dump = merge_reduce(family.leaf_max, family.leaf_ell, schedule, fused=fused)
                result = attribution.decompose(family, dump)
                error = pilot.relative_error(result.computed, result.reference)
                saved = expected[key]
                if (saved["status"] != "ok" or saved["computed_fp32"] != pilot_runner.fp32_word(result.computed)
                        or error != (old_fraction(saved["error_low"]), old_fraction(saved["error_high"]))):
                    raise ValueError(f"Original root/error replay mismatch: {key}")
                raw_savings = attribution.cancellation_savings(result.rounding_error, result.weight_error)
                terms = asdict(result)
                terms.pop("residuals")  # reconstructible per-node array; save the budget and sign counts
                signs = Counter((value > 0) - (value < 0) for value in result.residuals)
                row = {"family_id": source["family_id"], "cell_index": source["cell_index"],
                       "schedule": graph["requested"], "fused": fused, "terms": terms,
                       "residual_sign_counts": {str(sign): signs[sign] for sign in (-1, 0, 1)},
                       "A": error, "abs_R": absolute_interval(result.relative_rounding),
                       "abs_W": absolute_interval(result.relative_weight),
                       "budget_R": divide_constant(result.rounding_budget, result.reference),
                       "internal_savings": divide_constant(result.rounding_budget - abs(result.rounding_error), result.reference),
                       "between_savings": (raw_savings[0] / result.reference[1], raw_savings[1] / result.reference[0])}
                rows.append(row)
                group[graph["requested"], fused] = result
        for fused in plan["fused"]:
            chain, balanced = group["chain", fused], group["balanced", fused]
            if chain.reference != balanced.reference:
                raise ValueError("Paired decomposition lost its common reference")
            original = pair_map[source["family_id"], fused]
            gap = (old_fraction(original["difference_low"]), old_fraction(original["difference_high"]))
            chain_A = pilot.relative_error(chain.computed, chain.reference)
            balanced_A = pilot.relative_error(balanced.computed, balanced.reference)
            checked_gap = ((Fraction(0), Fraction(0)) if chain.computed == balanced.computed else
                           (chain_A[0] - balanced_A[1], chain_A[1] - balanced_A[0]))
            if gap != checked_gap:
                raise ValueError("Original paired-error replay mismatch")
            rounding_gap = divide_constant(abs(chain.rounding_error) - abs(balanced.rounding_error), chain.reference)
            correction = (gap[0] - rounding_gap[1], gap[1] - rounding_gap[0])
            pairs.append({"family_id": source["family_id"], "cell_index": source["cell_index"],
                          "fused": fused, "D": gap, "D_R": rounding_gap, "weight_correction": correction})
        if progress and (index + 1) % 8 == 0:
            print(f"Attributed {index + 1}/{len(inputs)} families; original roots and A intervals match", flush=True)
    if visited != set(expected) or len(pairs) != len(old_pairs):
        raise ValueError("Incomplete attribution coverage")
    return plan, rows, pairs


def summarize(plan, rows, pairs):
    grouped, paired = defaultdict(list), defaultdict(list)
    for row in rows:
        grouped[row["cell_index"], row["schedule"], row["fused"]].append(row)
    for row in pairs:
        paired[row["cell_index"], row["fused"]].append(row)
    cells, comparisons = [], []
    for (cell_index, schedule, fused), group in sorted(grouped.items()):
        cell = plan["cells"][cell_index]
        if len(group) != len(cell["seeds"]):
            raise ValueError("Incomplete cell")
        entry = {"cell_index": cell_index, "count": cell["count"], "spread": cell["spread"],
                 "schedule": schedule, "fused": fused, "families": len(group)}
        for name in ("A", "abs_R", "abs_W", "budget_R", "internal_savings", "between_savings"):
            entry["mean_" + name] = decimal_interval(mean([row[name] for row in group]))
        for name in ("rounding", "weight", "total"):
            entry["mean_signed_" + name] = decimal_interval(mean([row["terms"]["relative_" + name] for row in group]))
        entry["abs_R_gt_abs_W"] = sum(row["abs_R"][0] > row["abs_W"][1] for row in group)
        entry["abs_W_gt_abs_R"] = sum(row["abs_W"][0] > row["abs_R"][1] for row in group)
        entry["between_cancellation_proved"] = sum(row["between_savings"][0] > 0 for row in group)
        cells.append(entry)
    for (cell_index, fused), group in sorted(paired.items()):
        cell = plan["cells"][cell_index]
        entry = {"cell_index": cell_index, "count": cell["count"], "spread": cell["spread"], "fused": fused}
        for name in ("D", "D_R", "weight_correction"):
            entry["mean_" + name] = decimal_interval(mean([row[name] for row in group]))
        comparisons.append(entry)
    return {"stage": "post_hoc_exploratory_attribution", "cells": cells, "comparisons": comparisons,
            "units": "All displayed components are normalized by the same real l_* within a family.",
            "intervals": "Numerical enclosures, not population confidence intervals; 50 decimal digits rounded outward."}


def markdown(summary):
    def show(interval):
        return f"{(Decimal(interval['low']) + Decimal(interval['high'])) / 2:.3E}"
    lines = ["# 归因 v1：描述性汇总", "", "所有量均已除以同族 real-exp 分母。显示的是数值区间中点近似，不是总体 CI。", "",
             "| spread | FMA | n | schedule | mean A | mean abs(R)/l* | mean abs(W)/l* | mean B_R/l* | 节点间抵消 | R/W 间抵消 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in sorted(summary["cells"], key=lambda r: (r["spread"], r["fused"], r["count"], r["schedule"])):
        values = " | ".join(show(row["mean_" + name]) for name in
                            ("A", "abs_R", "abs_W", "budget_R", "internal_savings", "between_savings"))
        lines.append(f"| {row['spread']} | {row['fused']} | {row['count']} | {row['schedule']} | {values} |")
    lines += ["", "## 同族 schedule 差距的重建", "", "D=A_chain−A_balanced；D_R=(abs(R_chain)−abs(R_balanced))/l*。", "",
              "| spread | FMA | n | mean D | mean D_R | mean (D−D_R) |", "| --- | --- | --- | --- | --- | --- |"]
    for row in sorted(summary["comparisons"], key=lambda r: (r["spread"], r["fused"], r["count"])):
        values = " | ".join(show(row["mean_" + name]) for name in ("D", "D_R", "weight_correction"))
        lines.append(f"| {row['spread']} | {row['fused']} | {row['count']} | {values} |")
    lines += ["", "节点间抵消=(B_R−abs(R))/l*；R/W 间抵消=S/l*。后者是保守外包，未要求紧区间。",
              "各格族数见 summary.json；FMA 两 arm 共用输入。W 包括差值、exp 和复合权重效应，不是纯 exp 误差。",
              "分量分解不是独立因果干预；结果只解释本批 CPU 常值块样本。", ""]
    return "\n".join(lines)


def run(bundle, output, prediction):
    load_verified_bundle(bundle)
    prediction_bytes = prediction.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"schema": SCHEMA, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "parent_manifest_sha256": digest(bundle / "manifest.json"), "python": sys.version,
                "stage": "post_hoc_exploratory_attribution", "residual_arrays": "reconstruct from saved parent inputs/graphs and source snapshots"}
    write(output / "manifest.json", manifest)
    (output / "prediction.md").write_bytes(prediction_bytes)
    source_dir = output / "sources/online"
    source_dir.mkdir(parents=True)
    for name in SOURCES:
        (source_dir / (name + ".txt")).write_bytes((HERE / name).read_bytes())
    plan, rows, pairs = analyze(bundle, progress=True)
    write(output / "attributions.json", pack(rows))
    write(output / "paired_attributions.json", pack(pairs))
    summary = summarize(plan, rows, pairs)
    write(output / "summary.json", summary)
    (output / "README.md").write_text(markdown(summary), encoding="utf-8")
    for name in SOURCES:
        if (source_dir / (name + ".txt")).read_bytes() != (HERE / name).read_bytes():
            raise ValueError(f"Source changed during attribution: {name}")
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started, measurements=len(rows), pairs=len(pairs),
                    original_root_error_replay="exact_match",
                    files={path.relative_to(output).as_posix(): digest(path) for path in sorted(output.rglob("*"))
                           if path.is_file() and path.name != "manifest.json"})
    write(output / "manifest.json", manifest)
    return {key: manifest[key] for key in ("status", "measurements", "pairs", "original_root_error_replay")}


def replay(bundle, output):
    manifest = read(output / "manifest.json")
    expected_files = {"prediction.md", "attributions.json", "paired_attributions.json", "summary.json", "README.md",
                      *(f"sources/online/{name}.txt" for name in SOURCES)}
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete" or set(manifest["files"]) != expected_files:
        raise ValueError("Need a complete supported attribution bundle")
    if manifest["parent_manifest_sha256"] != digest(bundle / "manifest.json"):
        raise ValueError("Wrong parent bundle")
    for name, checksum in manifest["files"].items():
        if digest(output / name) != checksum:
            raise ValueError(f"Attribution integrity mismatch: {name}")
    for name in SOURCES:
        if (output / "sources/online" / (name + ".txt")).read_bytes() != (HERE / name).read_bytes():
            raise ValueError(f"Attribution source mismatch: {name}")
    plan, rows, pairs = analyze(bundle, progress=True)
    if pack(rows) != read(output / "attributions.json") or pack(pairs) != read(output / "paired_attributions.json"):
        raise ValueError("Attribution replay mismatch")
    summary = summarize(plan, rows, pairs)
    if summary != read(output / "summary.json") or markdown(summary) != (output / "README.md").read_text(encoding="utf-8"):
        raise ValueError("Attribution summary replay mismatch")
    return {"status": "exact_match", "measurements": len(rows), "pairs": len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "replay"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.prediction is None:
        parser.error("run requires --prediction")
    print(json.dumps(run(args.bundle, args.output, args.prediction) if args.command == "run" else replay(args.bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
