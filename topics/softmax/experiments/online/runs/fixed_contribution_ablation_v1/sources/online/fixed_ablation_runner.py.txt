"""Replay saved pilot inputs through ordinary fixed-contribution and online trees."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import sys
import time

from . import attribution_runner as transport, fixed_contribution_ablation as core
from . import pilot, pilot_runner, schedules
from .merge import merge_reduce, frozen_weight_reference

HERE = Path(__file__).resolve().parent
SCHEMA = "online-fixed-contribution-ablation-v1"
SOURCES = tuple("online/" + name for name in
                (*transport.SOURCES, "fixed_contribution_ablation.py", "fixed_ablation_runner.py")) + (
                    "rewrite/__init__.py", "rewrite/fp32_oracle.py")
ARTIFACTS = {"protocol.md", "results.json", "summary.json", "README.md"}


def subtract(left, right):
    return left[0] - right[1], left[1] - right[0]


def gap(chain, balanced):
    return (Fraction(0), Fraction(0)) if chain["root"] == balanced["root"] else subtract(chain["A"], balanced["A"])


def compare(family, graphs, fused_settings):
    """One certified preparation shared by every graph and arithmetic arm."""
    prepared = core.prepare(family)
    reference, total = prepared.reference, prepared.exact_sum
    fixed, online = {}, {}
    for name, schedule in graphs.items():
        root, rounding = core.ordinary_reduce(prepared.values, schedule)
        fixed[name] = {"root": root, "root_fp32": pilot_runner.fp32_word(root), "S": rounding,
                       "E": (root - reference[1], root - reference[0]),
                       "A": pilot.relative_error(root, reference)}
        for fused in fused_settings:
            dump = merge_reduce(family.leaf_max, family.leaf_ell, schedule, fused=fused)
            online_root = dump.ell_at(schedule.root)
            frozen = frozen_weight_reference(dump)
            online[name, fused] = {
                "root": online_root, "root_fp32": pilot_runner.fp32_word(online_root),
                "R": online_root - frozen, "W": (frozen - reference[1], frozen - reference[0]),
                "E": (online_root - reference[1], online_root - reference[0]),
                "A": pilot.relative_error(online_root, reference), "root_delta": online_root - root,
                "rescaled_edges": sum(w != 1 for w in (*dump.weight_left, *dump.weight_right))}
    return {"fixed_leaves_fp32": [pilot_runner.fp32_word(v) for v in prepared.values],
            "reference": reference, "C": total, "J": prepared.initialization_error}, fixed, online


def controls():
    """Deterministic controls are reported separately from the saved sample."""
    maxima = tuple(Fraction(-i) for i in range(8))
    families = {"max_first": pilot.BlockFamily(maxima, (1,) * 8),
                "max_last": pilot.BlockFamily(maxima[1:] + maxima[:1], (1,) * 8),
                "ascending": pilot.BlockFamily(tuple(reversed(maxima)), (1,) * 8),
                "equal_max": pilot.BlockFamily((Fraction(0),) * 8, (2**24, 1, 1, 1, 1, 1, 1, 1))}
    rows = []
    for label, family in families.items():
        prep, fixed, online = compare(family, {"chain": schedules.sequential_chain(8),
                                              "balanced": schedules.balanced_pairwise(8)}, (False, True))
        if label == "equal_max" and any(row["root_delta"] != 0 for row in online.values()):
            raise AssertionError("Equal-max control differs from plain addition")
        rows.append({"label": label, "maxima_fp32": [pilot_runner.fp32_word(v) for v in family.maxima],
                     "masses": family.masses, **prep, "fixed": fixed,
                     "online": [{"schedule": name, "fused": fused, **row}
                                for (name, fused), row in online.items()]})
    return rows


def analyze(bundle, *, progress=False):
    plan, inputs, graphs, measurements, old_pairs = transport.load_verified_bundle(bundle)
    expected = {(r["family_id"], r["graph_id"], r["fused"]): r for r in measurements}
    expected_pairs = {(r["family_id"], r["fused"]): r for r in old_pairs}
    if len(expected) != len(measurements) or len(expected_pairs) != len(old_pairs):
        raise ValueError("Duplicate parent records")
    result = {"families": [], "fixed": [], "online": [], "pairs": [], "controls": controls()}
    visited, paired = set(), set()
    for index, source in enumerate(inputs):
        family = pilot.BlockFamily(tuple(pilot_runner.from_word(w) for w in source["maxima_fp32"]),
                                   tuple(source["masses"]))
        selected = [g for g in graphs if g["leaf_count"] == len(family.maxima)]
        trees = {g["requested"]: schedules.Schedule(g["leaf_count"], tuple(tuple(p) for p in g["nodes"]),
                                                   g["kind"]) for g in selected}
        if len(selected) != 2 or set(trees) != {"chain", "balanced"}:
            raise ValueError("Need exactly the saved chain and balanced graphs")
        identifiers = {g["requested"]: g["graph_id"] for g in selected}
        prep, fixed, online = compare(family, trees, plan["fused"])
        common = {"family_id": source["family_id"], "cell_index": source["cell_index"]}
        result["families"].append({**common, **prep})
        for name, row in fixed.items():
            result["fixed"].append({**common, "schedule": name, "graph_id": identifiers[name], **row})
        for (name, fused), row in online.items():
            key = (source["family_id"], identifiers[name], fused)
            if key in visited or key not in expected:
                raise ValueError("Duplicate or unexpected online measurement")
            visited.add(key)
            old = expected[key]
            if (old["status"] != "ok" or row["root_fp32"] != old["computed_fp32"] or
                    row["A"] != (transport.old_fraction(old["error_low"]), transport.old_fraction(old["error_high"]))):
                raise ValueError(f"Original root/error replay mismatch: {key}")
            result["online"].append({**common, "schedule": name, "graph_id": identifiers[name], "fused": fused, **row})
        fixed_gap = gap(fixed["chain"], fixed["balanced"])
        for fused in plan["fused"]:
            key = source["family_id"], fused
            if key in paired or key not in expected_pairs:
                raise ValueError("Duplicate or unexpected pair")
            paired.add(key)
            old = expected_pairs[key]
            online_gap = gap(online["chain", fused], online["balanced", fused])
            if online_gap != (transport.old_fraction(old["difference_low"]), transport.old_fraction(old["difference_high"])):
                raise ValueError("Original paired-error replay mismatch")
            same_roots = all(online[name, fused]["root"] == fixed[name]["root"] for name in trees)
            h = (Fraction(0), Fraction(0)) if same_roots else subtract(online_gap, fixed_gap)
            result["pairs"].append({**common, "fused": fused, "D_online": online_gap,
                                    "D_fixed": fixed_gap, "H": h, "both_roots_equal": same_roots})
        if progress and (index + 1) % 8 == 0:
            print(f"Compared {index + 1}/{len(inputs)} families; saved online roots/A/D match", flush=True)
    if visited != set(expected) or paired != set(expected_pairs):
        raise ValueError("Incomplete parent coverage")
    return plan, result


def summarize(plan, result):
    refs = {r["family_id"]: r["reference"] for r in result["families"]}
    grouped, pairs = defaultdict(list), defaultdict(list)
    for arm in ("fixed", "online"):
        for row in result[arm]:
            grouped[row["cell_index"], arm, row["schedule"], row.get("fused", False)].append(row)
    for row in result["pairs"]:
        pairs[row["cell_index"], row["fused"]].append(row)
    cells, comparisons = [], []
    for (cell_index, arm, schedule, fused), rows in sorted(grouped.items()):
        cell = plan["cells"][cell_index]
        if len(rows) != len(cell["seeds"]):
            raise ValueError("Incomplete cell")
        entry = {"cell_index": cell_index, "count": cell["count"], "spread": cell["spread"],
                 "arm": arm, "schedule": schedule, "fused": fused if arm == "online" else None,
                 "families": len(rows), "mean_A": transport.decimal_interval(transport.mean([r["A"] for r in rows]))}
        for term in (("S",) if arm == "fixed" else ("R", "root_delta")):
            entry["mean_signed_" + term] = transport.decimal_interval(transport.mean([
                transport.divide_constant(r[term], refs[r["family_id"]]) for r in rows]))
        if arm == "online":
            entry["equal_fixed_roots"] = sum(r["root_delta"] == 0 for r in rows)
            entry["mean_abs_root_delta"] = transport.decimal_interval(transport.mean([
                transport.divide_constant(abs(r["root_delta"]), refs[r["family_id"]]) for r in rows]))
        cells.append(entry)
    for (cell_index, fused), rows in sorted(pairs.items()):
        cell = plan["cells"][cell_index]
        if len(rows) != len(cell["seeds"]):
            raise ValueError("Incomplete paired cell")
        entry = {"cell_index": cell_index, "count": cell["count"], "spread": cell["spread"],
                 "fused": fused, "families": len(rows), "both_roots_equal": sum(r["both_roots_equal"] for r in rows)}
        for term in ("D_online", "D_fixed", "H"):
            entry["mean_" + term] = transport.decimal_interval(transport.mean([r[term] for r in rows]))
            entry[term + "_positive"] = sum(r[term][0] > 0 for r in rows)
            entry[term + "_negative"] = sum(r[term][1] < 0 for r in rows)
            entry[term + "_range"] = transport.decimal_interval((min(r[term][0] for r in rows), max(r[term][1] for r in rows)))
        entry["mean_abs_H"] = transport.decimal_interval(transport.mean([transport.absolute_interval(r["H"]) for r in rows]))
        comparisons.append(entry)
    return {"stage": "post_hoc_mechanism_ablation", "cells": cells, "comparisons": comparisons,
            "intervals": "Numerical enclosures, not population confidence intervals. Means over saved families only.",
            "decision": "No equivalence threshold; no equivalence or online-novelty claim from this comparison alone."}


def markdown(summary):
    def show(interval):
        return f"{(Decimal(interval['low']) + Decimal(interval['high'])) / 2:.4E}"
    lines = ["# 固定贡献消融 v1", "", "D = A_chain − A_balanced；H = D_online − D_fixed。",
             "数值外包中点近似；每格族数见 summary，不是总体置信区间。普通加法臂由两种 FMA 设置共用。", "",
             "| n | spread | FMA | mean D_online | mean D_fixed | mean H | mean abs(H) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary["comparisons"]:
        values = " | ".join(show(row["mean_" + term]) for term in ("D_online", "D_fixed", "H", "abs_H"))
        lines.append(f"| {row['count']} | {row['spread']} | {row['fused']} | {values} |")
    lines += ["", "逐族固定叶位模式、共同参考、J/S/R/W、带符号根差和确定性排列对照见 results.json。",
              "原 online 根、A、D 在本次计算中逐项重放核对。固定臂预知全局最大值，是离线机制控制。",
              "相近误差差距不自动证明机制等价；两臂差异也不是单一因果分量或新颖性证明。", ""]
    return "\n".join(lines)


def run(bundle, output, protocol):
    transport.load_verified_bundle(bundle)
    protocol_bytes = protocol.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent_hash = transport.digest(bundle / "manifest.json")
    manifest = {"schema": SCHEMA, "status": "running", "parent_manifest_sha256": parent_hash,
                "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "trajectories": "Reconstruct deterministically from parent inputs/graphs and frozen sources."}
    transport.write(output / "manifest.json", manifest)
    (output / "protocol.md").write_bytes(protocol_bytes)
    for name in SOURCES:
        target = output / "sources" / (name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((HERE.parent / name).read_bytes())
    plan, result = analyze(bundle, progress=True)
    transport.write(output / "results.json", transport.pack(result))
    summary = summarize(plan, result)
    transport.write(output / "summary.json", summary)
    (output / "README.md").write_text(markdown(summary), encoding="utf-8")
    for name in SOURCES:
        if (output / "sources" / (name + ".txt")).read_bytes() != (HERE.parent / name).read_bytes():
            raise ValueError(f"Source changed during run: {name}")
    transport.load_verified_bundle(bundle)
    if parent_hash != transport.digest(bundle / "manifest.json"):
        raise ValueError("Parent changed during run")
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    families=len(result["families"]), fixed=len(result["fixed"]), online=len(result["online"]),
                    pairs=len(result["pairs"]), original_root_error_pair_replay="exact_match",
                    files={p.relative_to(output).as_posix(): transport.digest(p) for p in sorted(output.rglob("*"))
                           if p.is_file() and p.name != "manifest.json"})
    transport.write(output / "manifest.json", manifest)
    return {k: manifest[k] for k in ("status", "families", "fixed", "online", "pairs")}


def replay(bundle, output):
    manifest = transport.read(output / "manifest.json")
    expected = ARTIFACTS | {"sources/" + name + ".txt" for name in SOURCES}
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete" or set(manifest["files"]) != expected:
        raise ValueError("Need a complete supported fixed-ablation bundle")
    if manifest["parent_manifest_sha256"] != transport.digest(bundle / "manifest.json"):
        raise ValueError("Wrong parent bundle")
    for name, checksum in manifest["files"].items():
        if transport.digest(output / name) != checksum:
            raise ValueError(f"Ablation integrity mismatch: {name}")
    for name in SOURCES:
        if (output / "sources" / (name + ".txt")).read_bytes() != (HERE.parent / name).read_bytes():
            raise ValueError(f"Ablation source mismatch: {name}")
    plan, result = analyze(bundle, progress=True)
    if transport.pack(result) != transport.read(output / "results.json"):
        raise ValueError("Ablation numerical replay mismatch")
    summary = summarize(plan, result)
    if summary != transport.read(output / "summary.json") or markdown(summary) != (output / "README.md").read_text(encoding="utf-8"):
        raise ValueError("Ablation summary replay mismatch")
    return {"status": "exact_match", **{k: len(result[k]) for k in ("families", "fixed", "online", "pairs")}}


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
