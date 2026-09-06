"""One bounded, replayable scalar-output diagnostic on saved pilot inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import sys
import time

from . import attribution_runner as io, fixed_ablation_runner, fixed_contribution_ablation as fixed
from . import output_measure as measurement, output_probe as probe, output_reference as reference
from . import pilot, pilot_runner, schedules
from .merge import merge_reduce

HERE = Path(__file__).resolve().parent
SCHEMA = "online-scalar-output-v1"
SOURCES = (*fixed_ablation_runner.SOURCES, "online/output_probe.py", "online/output_reference.py",
           "online/output_measure.py", "online/output_runner.py")
ARTIFACTS = {"protocol.md", "results.json", "summary.json", "README.md"}


def prepare_references(family, prepared):
    """Share denominator preparation; use the user's signed core for nonconstant V.

    Constant V is handled by its exact proportionality to ell, matching reference_output.
    This avoids repeatedly evaluating every exp just to reproduce the same reference.
    """
    result = {}
    for name in probe.PROBE_NAMES:
        values = probe.probe_values(family, name)
        if all(v == values[0] for v in values):
            numerator = tuple(sorted(values[0] * x for x in prepared.reference))
            output = values[0], values[0]
        else:
            numerator = reference.numerator_interval(family, values)
            output = reference.quotient_interval(numerator, prepared.reference)
        result[name] = (values, reference.OutputReference(numerator, prepared.reference, output))
    return result


def analyze(bundle, *, progress=False):
    plan, inputs, graphs, old_rows, old_pairs = io.load_verified_bundle(bundle)
    if plan["fused"] != [False, True]:
        raise ValueError("The output diagnostic requires both saved FMA arms")
    expected = {(r["family_id"], r["graph_id"], r["fused"]): r for r in old_rows}
    expected_pairs = {(r["family_id"], r["fused"]): r for r in old_pairs}
    if len(expected) != len(old_rows) or len(expected_pairs) != len(old_pairs):
        raise ValueError("Duplicate parent records")
    result = {"families": [], "references": [], "measurements": [], "pairs": []}
    visited, paired_keys = set(), set()
    for index, source in enumerate(inputs):
        family = pilot.BlockFamily(tuple(pilot_runner.from_word(w) for w in source["maxima_fp32"]), tuple(source["masses"]))
        selected = [g for g in graphs if g["leaf_count"] == len(family.maxima)]
        if len(selected) != 2 or {g["requested"] for g in selected} != {"chain", "balanced"}:
            raise ValueError("Need the saved chain and balanced graphs")
        prepared = fixed.prepare(family)
        refs = prepare_references(family, prepared)
        common = {"family_id": source["family_id"], "cell_index": source["cell_index"]}
        result["families"].append({**common, "fixed_leaves_fp32": [pilot_runner.fp32_word(v) for v in prepared.values],
                                   "denominator_reference": prepared.reference})
        for name, (values, ref) in refs.items():
            result["references"].append({**common, "probe": name, "V_fp32": [pilot_runner.fp32_word(v) for v in values],
                                         **asdict(ref)})
        collected, old_replay = {}, {}
        for graph in selected:
            name = graph["requested"]
            tree = schedules.Schedule(graph["leaf_count"], tuple(tuple(p) for p in graph["nodes"]), graph["kind"])
            # All maxima equal makes the current merge exactly an ordinary add tree.
            # Cross-check its denominator with the independent original add oracle.
            fixed_dump = merge_reduce((family.global_max,) * tree.leaf_count, prepared.values, tree)
            ordinary_root, _ = fixed.ordinary_reduce(prepared.values, tree)
            if fixed_dump.ell_at(tree.root) != ordinary_root or any(w != 1 for w in fixed_dump.weight_left + fixed_dump.weight_right):
                raise AssertionError("Fixed arm does not reproduce ordinary addition")
            dumps = [("fixed", None, fixed_dump)]
            for fused in plan["fused"]:
                key = source["family_id"], graph["graph_id"], fused
                if key in visited or key not in expected:
                    raise ValueError("Unexpected or duplicate parent measurement")
                visited.add(key)
                dump = merge_reduce(family.leaf_max, family.leaf_ell, tree, fused=fused)
                root = dump.ell_at(tree.root)
                error = pilot.relative_error(root, prepared.reference)
                saved = expected[key]
                if (saved["status"] != "ok" or saved["computed_fp32"] != pilot_runner.fp32_word(root) or
                        error != (io.old_fraction(saved["error_low"]), io.old_fraction(saved["error_high"]))):
                    raise ValueError("Original denominator root/error replay mismatch")
                old_replay[name, fused] = {"root": root, "A": error}
                dumps.append(("online", fused, dump))
            for arm, fused, dump in dumps:
                for probe_name, (values, ref) in refs.items():
                    # Prescribed first-round V is 0 or +/-1, so multiplication here is exact.
                    leaves = (tuple(c * v for c, v in zip(prepared.values, values)) if arm == "fixed"
                              else probe.leaf_numerators(family, values))
                    computed = reference.finish_output(dump, leaves)
                    row = {**common, "probe": probe_name, "schedule": name, "graph_id": graph["graph_id"],
                           "arm": arm, "fused": fused, **measurement.record(computed, ref, max(map(abs, values)))}
                    if probe_name in ("zero", "one"):
                        target = Fraction(probe_name == "one")
                        if any(v["value"] != target or v["abs_error"] != (0, 0) for v in row["formats"].values()):
                            raise AssertionError("Constant-V output control failed")
                    if arm == "online":
                        control = collected[probe_name, name, "fixed", None]
                        row["same_tree_fixed_delta"] = {
                            dtype: row["formats"][dtype]["value"] - control["formats"][dtype]["value"]
                            for dtype in measurement.DTYPES}
                    collected[probe_name, name, arm, fused] = row
                    result["measurements"].append(row)
        for fused in plan["fused"]:
            key = source["family_id"], fused
            if key in paired_keys or key not in expected_pairs:
                raise ValueError("Unexpected or duplicate parent pair")
            paired_keys.add(key)
            old = expected_pairs[key]
            d = fixed_ablation_runner.gap(old_replay["chain", fused], old_replay["balanced", fused])
            if d != (io.old_fraction(old["difference_low"]), io.old_fraction(old["difference_high"])):
                raise ValueError("Original denominator pair replay mismatch")
        for probe_name in probe.PROBE_NAMES:
            for arm, fused in (("fixed", None), ("online", False), ("online", True)):
                result["pairs"].append({**common, "probe": probe_name, "arm": arm, "fused": fused,
                                        "formats": measurement.paired(collected[probe_name, "chain", arm, fused],
                                                                       collected[probe_name, "balanced", arm, fused])})
        if progress and (index + 1) % 8 == 0:
            print(f"Output {index + 1}/{len(inputs)} families; controls and original denominator roots/A/D verified", flush=True)
    if visited != set(expected) or paired_keys != set(expected_pairs):
        raise ValueError("Incomplete parent coverage")
    return plan, result


def summarize(plan, result):
    groups, pair_groups = defaultdict(list), defaultdict(list)
    for row in result["measurements"]:
        groups[row["cell_index"], row["probe"], row["arm"], str(row["fused"]), row["schedule"]].append(row)
    for row in result["pairs"]:
        pair_groups[row["cell_index"], row["probe"], row["arm"], str(row["fused"])].append(row)
    cells, comparisons = [], []
    for (cell_index, probe_name, arm, _, schedule), rows in sorted(groups.items()):
        cell = plan["cells"][cell_index]
        if len(rows) != len(cell["seeds"]):
            raise ValueError("Incomplete output cell")
        for dtype in measurement.DTYPES:
            errors = [r["formats"][dtype]["abs_error"] for r in rows]
            entry = {"cell_index": cell_index, "count": cell["count"], "spread": cell["spread"],
                     "probe": probe_name, "arm": arm, "fused": rows[0]["fused"], "schedule": schedule,
                     "dtype": dtype, "families": len(rows), "mean_abs_error": io.decimal_interval(io.mean(errors)),
                     "max_abs_error": io.decimal_interval((max(e[0] for e in errors), max(e[1] for e in errors)))}
            if arm == "online":
                entry["equal_fixed_outputs"] = sum(r["same_tree_fixed_delta"][dtype] == 0 for r in rows)
            if dtype == "fp32":
                entry["mean_denominator_only_abs_error"] = io.decimal_interval(io.mean([r["denominator_only_abs_error"] for r in rows]))
            cells.append(entry)
    for (cell_index, probe_name, arm, _), rows in sorted(pair_groups.items()):
        cell = plan["cells"][cell_index]
        if len(rows) != len(cell["seeds"]):
            raise ValueError("Incomplete output pair cell")
        for dtype in measurement.DTYPES:
            data = [r["formats"][dtype] for r in rows]
            comparisons.append({"cell_index": cell_index, "count": cell["count"], "spread": cell["spread"],
                                "probe": probe_name, "arm": arm, "fused": rows[0]["fused"], "dtype": dtype,
                                "families": len(rows), "changed_outputs": sum(not r["outputs_equal"] for r in data),
                                "chain_worse": sum(r["D"][0] > 0 for r in data),
                                "balanced_worse": sum(r["D"][1] < 0 for r in data),
                                "mean_D": io.decimal_interval(io.mean([r["D"] for r in data])),
                                "max_abs_output_delta": io.decimal_interval((max(abs(r["output_delta"]) for r in data),) * 2)})
    return {"stage": "post_hoc_scalar_output_diagnostic", "cells": cells, "comparisons": comparisons,
            "probes": list(probe.PROBE_NAMES), "casts": "FP32 result -> RN-even FP16/BF16, gradual underflow, canonical +0.",
            "intervals": "Numerical enclosures, not population confidence intervals.",
            "limits": "Scalar controlled V, saved CPU inputs; no application tolerance, GPU or task-quality conclusion."}


def markdown(summary):
    def show(x):
        return f"{(Decimal(x['low']) + Decimal(x['high'])) / 2:.4E}"
    lines = ["# Scalar output diagnostic v1", "", "D = abs(error_chain) - abs(error_balanced).",
             "Numerical interval midpoint approximations; no population CI or engineering threshold.",
             "Rows below show FP32 at the largest saved block count. All cells/dtypes are in summary.json.", "",
             "| n | spread | V probe | arm | FMA | mean D | changed outputs / families |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    largest = max(r["count"] for r in summary["comparisons"])
    for row in summary["comparisons"]:
        if row["dtype"] == "fp32" and row["count"] == largest:
            lines.append(f"| {row['count']} | {row['spread']} | {row['probe']} | {row['arm']} | {row['fused']} | {show(row['mean_D'])} | {row['changed_outputs']}/{row['families']} |")
    lines += ["", "Original denominator roots, A and D replay exactly. Constant V=0/1 controls are exact in every arm/dtype.",
              "The fixed arm uses the same globally prepared c_hat as the denominator ablation, then sums c_hat*V with ordinary FP32 adds.",
              "Raw references, V bits, signed errors, division residuals and counterfactuals are in results.json.",
              "Storage casts occur after FP32 division; no claim about actual GPU instructions or task impact.", ""]
    return "\n".join(lines)


def run(bundle, output, protocol):
    io.load_verified_bundle(bundle)
    protocol_bytes = protocol.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    parent_hash = io.digest(bundle / "manifest.json")
    manifest = {"schema": SCHEMA, "status": "running", "parent_manifest_sha256": parent_hash,
                "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "probes": list(probe.PROBE_NAMES), "dtypes": list(measurement.DTYPES)}
    io.write(output / "manifest.json", manifest)
    (output / "protocol.md").write_bytes(protocol_bytes)
    for name in SOURCES:
        p = output / "sources" / (name + ".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((HERE.parent / name).read_bytes())
    plan, result = analyze(bundle, progress=True)
    io.write(output / "results.json", io.pack(result))
    summary = summarize(plan, result)
    io.write(output / "summary.json", summary)
    (output / "README.md").write_text(markdown(summary), encoding="utf-8")
    for name in SOURCES:
        if (output / "sources" / (name + ".txt")).read_bytes() != (HERE.parent / name).read_bytes():
            raise ValueError(f"Source changed during output diagnostic: {name}")
    io.load_verified_bundle(bundle)
    if parent_hash != io.digest(bundle / "manifest.json"):
        raise ValueError("Parent changed during output diagnostic")
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - start,
                    counts={key: len(value) for key, value in result.items()},
                    original_denominator_replay="exact_match", constant_V_controls="exact_match",
                    files={p.relative_to(output).as_posix(): io.digest(p) for p in sorted(output.rglob("*"))
                           if p.is_file() and p.name != "manifest.json"})
    io.write(output / "manifest.json", manifest)
    return {k: manifest[k] for k in ("status", "counts", "original_denominator_replay", "constant_V_controls")}


def replay(bundle, output):
    manifest = io.read(output / "manifest.json")
    expected = ARTIFACTS | {"sources/" + name + ".txt" for name in SOURCES}
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete" or set(manifest["files"]) != expected:
        raise ValueError("Need a complete supported output bundle")
    if manifest["parent_manifest_sha256"] != io.digest(bundle / "manifest.json"):
        raise ValueError("Wrong parent bundle")
    for name, checksum in manifest["files"].items():
        if io.digest(output / name) != checksum:
            raise ValueError(f"Output artifact integrity mismatch: {name}")
    for name in SOURCES:
        if (output / "sources" / (name + ".txt")).read_bytes() != (HERE.parent / name).read_bytes():
            raise ValueError(f"Output source mismatch: {name}")
    plan, result = analyze(bundle, progress=True)
    if io.pack(result) != io.read(output / "results.json"):
        raise ValueError("Output numerical replay mismatch")
    summary = summarize(plan, result)
    if summary != io.read(output / "summary.json") or markdown(summary) != (output / "README.md").read_text(encoding="utf-8"):
        raise ValueError("Output summary replay mismatch")
    if manifest["counts"] != {key: len(value) for key, value in result.items()}:
        raise ValueError("Output count mismatch")
    return {"status": "exact_match", "counts": manifest["counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "replay"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.protocol is None:
        parser.error("run requires --protocol")
    print(run(args.bundle, args.output, args.protocol) if args.command == "run" else replay(args.bundle, args.output))


if __name__ == "__main__":
    main()
