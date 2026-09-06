"""Replayable synthetic preflight for the user's calibration core; no coverage claim.

Only stage=preflight is supported. A statistically interpretable study needs a separate
frozen plan including Monte Carlo uncertainty and method selection rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from . import confirmation_calibration as cal
from .confirmation_stats import FamilySample, resample_indices

HERE = Path(__file__).resolve().parent
SCHEMA = "online-calibration-preflight-v1"
SOURCES = ("__init__.py", "confirmation_stats.py", "confirmation_calibration.py", "calibration_runner.py")
FILES = {"plan.json", "protocol.md", "populations.json", "inputs.jsonl", "trials.jsonl",
         "summary.json", "README.md", *(f"sources/{name}.txt" for name in SOURCES)}


def encode(value):
    if isinstance(value, Fraction):
        return {"$fraction": [str(value.numerator), str(value.denominator)]}
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def decode(value):
    if set(value) == {"$fraction"}:
        numerator, denominator = value["$fraction"]
        return Fraction(int(numerator), int(denominator))
    return value


def canonical(value):
    return json.dumps(value, default=encode, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path, value):
    path.write_text(json.dumps(value, default=encode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_hook=decode)


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line, object_hook=decode)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(plan):
    required = {"schema", "stage", "method", "population_names", "counts", "outer_trials",
                "bootstrap_repeats", "alpha", "seed_namespace", "protocol"}
    if set(plan) != required or plan["schema"] != SCHEMA or plan["stage"] != "preflight":
        raise ValueError("Need the exact preflight plan schema")
    if plan["method"] != "percentile":
        raise ValueError("Only the percentile candidate is implemented")
    known = {population.name for population in cal.populations()}
    names, counts = plan["population_names"], plan["counts"]
    if not isinstance(names, list) or not names or any(name not in known for name in names) or len(set(names)) != len(names):
        raise ValueError("Need unique supported population names")
    if not isinstance(counts, list) or not counts or any(type(n) is not int or n < 1 for n in counts) or len(set(counts)) != len(counts):
        raise ValueError("Need unique positive integer counts")
    for key in ("outer_trials", "bootstrap_repeats"):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    alpha = plan["alpha"]
    if (not isinstance(alpha, list) or len(alpha) != 2 or any(type(x) is not int for x in alpha)
            or not 0 < alpha[0] < alpha[1]):
        raise ValueError("alpha must be an integer numerator/denominator pair strictly between 0 and 1")
    for key in ("seed_namespace", "protocol"):
        if not isinstance(plan[key], str) or not plan[key]:
            raise ValueError(f"{key} must be nonempty text")


def trial_keys(plan):
    for population in plan["population_names"]:
        for count in plan["counts"]:
            for trial in range(plan["outer_trials"]):
                yield population, count, trial


def seed_for(plan, key, role):
    """SHA-256 of canonical JSON [namespace,population,n,trial,role], as a big-endian int."""
    payload = canonical([plan["seed_namespace"], *key, role]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def input_record(plan, key, population):
    input_seed = seed_for(plan, key, "input")
    return {"population": key[0], "count": key[1], "trial": key[2],
            "input_seed": input_seed, "resample_seed": seed_for(plan, key, "resample"),
            "truth": population.truth,
            "families": [asdict(row) for row in cal.draw_families(population, key[1], input_seed)]}


def evaluate_record(plan, row, population):
    """Reconstruct saved families; only bootstrap indices are regenerated on replay."""
    key = row["population"], row["count"], row["trial"]
    if (row["input_seed"] != seed_for(plan, key, "input")
            or row["resample_seed"] != seed_for(plan, key, "resample")
            or tuple(row["truth"]) != population.truth):
        raise ValueError("Saved seed or truth mismatch")
    samples = tuple(FamilySample(item["family_id"], tuple(item["separate"]), tuple(item["fused"]))
                    for item in row["families"])
    if len(samples) != row["count"]:
        raise ValueError("Saved family count mismatch")
    for sample in samples:
        if (sample.separate[0] != sample.separate[1] or sample.fused[0] != sample.fused[1]
                or (sample.separate[0], sample.fused[0]) not in population.outcomes):
            raise ValueError("Saved sample is outside the declared joint population")
    draws = resample_indices(len(samples), plan["bootstrap_repeats"], row["resample_seed"])
    bounds = cal.percentile_candidate(samples, draws, Fraction(*plan["alpha"]))
    verdict = cal.assess_trial(bounds, population.truth)
    columns = ([sample.separate[0] for sample in samples], [sample.fused[0] for sample in samples])
    return {"population": key[0], "count": key[1], "trial": key[2],
            "draws_sha256": hashlib.sha256(canonical(draws).encode("utf-8")).hexdigest(),
            "lower_bounds": bounds, "verdict": asdict(verdict),
            "constant_sample": tuple(len(set(column)) == 1 for column in columns),
            "negative_observations": tuple(sum(value < 0 for value in column) for column in columns)}


def summarize(records):
    """Descriptive outer-trial counts only; no interval, acceptance or coverage claim."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["population"], record["count"])].append(record)
    cells = []
    for (population, count), rows in sorted(groups.items()):
        arms = []
        for arm in (0, 1):
            counts = {field: sum(row["verdict"][field][arm] for row in rows)
                      for field in ("covered", "rejected", "false_positive")}
            counts.update(constant_samples=sum(row["constant_sample"][arm] for row in rows),
                          trials_without_negative=sum(row["negative_observations"][arm] == 0 for row in rows))
            arms.append(counts)
        cells.append({"population": population, "count": count, "outer_trials": len(rows), "arms": arms,
                      "conjunction_rejected": sum(row["verdict"]["conjunction_rejected"] for row in rows),
                      "conjunction_false_positive": sum(row["verdict"]["conjunction_false_positive"] for row in rows)})
    return {"stage": "preflight", "interpretation": "Descriptive counts, not calibrated coverage or method acceptance",
            "total_outer_trials": sum(len(rows) for rows in groups.values()), "cells": cells}


def render_readme(summary):
    lines = ["# Synthetic calibration preflight", "",
             "Pipeline/timing preflight only. These small-sample counts are not coverage estimates suitable for method selection.",
             "Both arms share each input family; they are not independent trials. No softmax confirmation data were generated.", "",
             "| Population | n | Outer trials | False positives separate/FMA | Constant samples separate/FMA | Joint false positives |",
             "| --- | --- | --- | --- | --- | --- |"]
    for cell in summary["cells"]:
        a, b = cell["arms"]
        lines.append(f"| {cell['population']} | {cell['count']} | {cell['outer_trials']} | "
                     f"{a['false_positive']}/{b['false_positive']} | {a['constant_samples']}/{b['constant_samples']} | "
                     f"{cell['conjunction_false_positive']} |")
    lines += ["", "Exact per-trial bounds and verdicts: trials.jsonl. Actual outer inputs: inputs.jsonl.",
              "Plan, protocol, source snapshots and file hashes are saved before/with measurement.",
              "Constant observed samples are retained. All planned trials must finish for status=complete.",
              "Bootstrap draws are reconstructed by the recorded local Random/randrange algorithm and checked by SHA-256.",
              "Integrity hashes detect file changes; they are not signatures against malicious rewriting.", ""]
    return "\n".join(lines)


def run(plan_path, output, *, progress=False):
    plan_path, output = Path(plan_path), Path(output)
    plan = read_json(plan_path)
    validate_plan(plan)
    protocol = (plan_path.parent / plan["protocol"]).resolve().read_bytes()
    population_map = {p.name: p for p in cal.populations() if p.name in plan["population_names"]}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "rng": "random.Random(seed).randrange; input weighted ticket, inner original-size index draws",
                "seed_derivation": "SHA-256 big-endian integer of canonical JSON [namespace,population,n,trial,role]"}
    write_json(output / "manifest.json", manifest)
    try:
        write_json(output / "plan.json", plan)
        (output / "protocol.md").write_bytes(protocol)
        (output / "sources").mkdir()
        for name in SOURCES:
            (output / "sources" / (name + ".txt")).write_bytes((HERE / name).read_bytes())
        write_json(output / "populations.json", [asdict(population_map[name]) for name in plan["population_names"]])
        started = time.perf_counter()
        with (output / "inputs.jsonl").open("x", encoding="utf-8") as stream:
            for key in trial_keys(plan):
                stream.write(canonical(input_record(plan, key, population_map[key[0]])) + "\n")
        manifest["input_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        records = []
        with (output / "trials.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(read_rows(output / "inputs.jsonl"), 1):
                result = evaluate_record(plan, row, population_map[row["population"]])
                stream.write(canonical(result) + "\n")
                stream.flush()
                records.append(result)
                if progress and index % 20 == 0:
                    print(f"Measured {index} outer trials", flush=True)
        manifest["measurement_seconds"] = time.perf_counter() - started
        summary = summarize(records)
        write_json(output / "summary.json", summary)
        (output / "README.md").write_text(render_readme(summary), encoding="utf-8")
        manifest.update(status="complete", outer_trials=len(records),
                        files={name: digest(output / name) for name in sorted(FILES)})
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(output / "manifest.json", manifest)
        raise


def replay(bundle, *, progress=False):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete" or set(manifest.get("files", {})) != FILES:
        raise ValueError("Need a complete preflight bundle with the exact inventory")
    for name, checksum in manifest["files"].items():
        if digest(bundle / name) != checksum:
            raise ValueError(f"File integrity mismatch: {name}")
    if manifest["python"] != sys.version:
        raise ValueError("Strict replay requires the recorded Python version")
    for name in SOURCES:
        if (bundle / "sources" / (name + ".txt")).read_bytes() != (HERE / name).read_bytes():
            raise ValueError(f"Current source mismatch: {name}")
    plan = read_json(bundle / "plan.json")
    validate_plan(plan)
    population_map = {p.name: p for p in cal.populations() if p.name in plan["population_names"]}
    if canonical(read_json(bundle / "populations.json")) != canonical([asdict(population_map[name]) for name in plan["population_names"]]):
        raise ValueError("Population definitions mismatch")
    records = []
    for key, row, expected in zip(trial_keys(plan), read_rows(bundle / "inputs.jsonl"), read_rows(bundle / "trials.jsonl"), strict=True):
        if (row["population"], row["count"], row["trial"]) != key:
            raise ValueError("Input trial inventory/order mismatch")
        result = evaluate_record(plan, row, population_map[key[0]])
        if canonical(result) != canonical(expected):
            raise ValueError(f"Numerical replay mismatch: {key}")
        records.append(result)
        if progress and len(records) % 20 == 0:
            print(f"Replayed {len(records)} outer trials", flush=True)
    summary = summarize(records)
    if len(records) != manifest["outer_trials"] or canonical(summary) != canonical(read_json(bundle / "summary.json")):
        raise ValueError("Summary/count replay mismatch")
    if (bundle / "README.md").read_text(encoding="utf-8") != render_readme(summary):
        raise ValueError("Readable summary replay mismatch")
    return {"status": "verified", "outer_trials": len(records), "manifest_sha256": digest(bundle / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.plan, args.output, progress=True)
        print(f"Complete: {result['outer_trials']} trials, measurement {result['measurement_seconds']:.3f}s")
    else:
        print(canonical(replay(args.bundle, progress=True)))


if __name__ == "__main__":
    main()
