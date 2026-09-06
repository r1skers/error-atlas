"""Replayable CPU pilot plumbing; no new metric, statistical inference or GPU claims.

Run from the repository root with ``python -m
topics.softmax.experiments.online.pilot_runner run --plan PLAN --output NEW_DIRECTORY``.
Replay uses saved FP32 words and ordered graphs, never a regenerated random family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from . import pilot, schedules
from .fp32_exp import DEFAULT_PRECISION
from .fp32_signed import is_stored_fp32

SCHEMA = "online-pilot-v1"
EXP = "online.fp32_exp.correctly_rounded_exp:default_precision"
MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[3]
SOURCE_NAMES = ("__init__.py", "pilot_runner.py", "pilot.py", "schedules.py",
                "merge.py", "fp32_exp.py", "fp32_signed.py")
SCHEDULE_NAMES = ("chain", "balanced", "shfl_down", "split_k_4")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _write(path: Path, value) -> None:
    path.write_text(_json(value), encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fp32_word(value: Fraction) -> str:
    """No rounding here: binary64 represents every finite FP32 value exactly.

    Fraction canonicalizes zero; this CPU contract records +0, not hardware sign bits.
    """
    if not is_stored_fp32(value):
        raise ValueError("Expected an already stored finite FP32 value")
    return struct.pack(">f", float(value)).hex()


def from_word(word: str) -> Fraction:
    if not isinstance(word, str) or len(word) != 8:
        raise ValueError("Expected an eight-digit FP32 hexadecimal word")
    value = struct.unpack(">f", bytes.fromhex(word))[0]
    if not math.isfinite(value) or word.lower() == "80000000":
        raise ValueError("CPU Fraction contract requires finite values and canonical +0")
    return Fraction(value)


def _fraction(value: Fraction) -> dict:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _integer(value, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_plan(plan: dict) -> None:
    if plan["schema"] != SCHEMA or plan["stage"] not in ("smoke", "exploratory"):
        raise ValueError("Only versioned smoke/exploratory plans are supported")
    if plan["exp"] != EXP:
        raise ValueError("This runner currently supports only correctly-rounded exp")
    names, arms = plan["schedules"], plan["fused"]
    if (not names or len(set(names)) != len(names) or "balanced" not in names
            or any(name not in SCHEDULE_NAMES for name in names)):
        raise ValueError("Schedules must be unique supported names including balanced")
    if not arms or any(type(arm) is not bool for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError("fused must contain distinct Boolean arms")
    if not plan["cells"] or not plan["prediction_record"]:
        raise ValueError("Provide cells and a prediction record")
    for cell in plan["cells"]:
        count = _integer(cell["count"], "count")
        mass = _integer(cell["mass"], "mass")
        if mass > 2**24:
            raise ValueError("Block mass exceeds the exact-initialization contract")
        if cell["kind"] == "max_at_position":
            position = _integer(cell["position"], "position", 0)
            if position >= count or Fraction(cell["gap"]) <= 0:
                raise ValueError("Invalid maximum position or gap")
        elif cell["kind"] == "uniform_spread":
            if not math.isfinite(float(cell["spread"])) or float(cell["spread"]) <= 0:
                raise ValueError("spread must be positive and finite")
            seeds = cell["seeds"]
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("Each cell requires distinct explicit seeds")
            for seed in seeds:
                _integer(seed, "seed", 0)
        else:
            raise ValueError("Unsupported family generator")


def build_inputs(plan: dict) -> list[dict]:
    validate_plan(plan)
    rows = []
    for cell_index, cell in enumerate(plan["cells"]):
        seeds = cell["seeds"] if cell["kind"] == "uniform_spread" else [None]
        for replicate, seed in enumerate(seeds):
            if seed is None:
                family = pilot.max_at_position(cell["count"], cell["mass"],
                                               Fraction(cell["gap"]), cell["position"])
            else:
                family = pilot.uniform_spread(cell["count"], cell["mass"],
                                             float(cell["spread"]), random.Random(seed))
            rows.append({"family_id": f"cell_{cell_index:03d}_rep_{replicate:03d}",
                         "cell_index": cell_index, "seed": seed,
                         "maxima_fp32": [fp32_word(v) for v in family.maxima],
                         "masses": list(family.masses)})
    return rows


def build_graphs(plan: dict) -> list[dict]:
    rows = []
    for count in sorted({cell["count"] for cell in plan["cells"]}):
        for name in plan["schedules"]:
            row = {"graph_id": f"{name}:{count}", "requested": name, "leaf_count": count}
            if name == "shfl_down" and (count > 32 or count & (count - 1)):
                row.update(status="not_applicable", reason="requires 1/2/4/8/16/32 lanes")
            elif name == "split_k_4" and count < 4:
                row.update(status="not_applicable", reason="requires at least four blocks")
            else:
                factory = {"chain": schedules.sequential_chain,
                           "balanced": schedules.balanced_pairwise,
                           "shfl_down": schedules.warp_shfl_down,
                           "split_k_4": lambda n: schedules.split_k(n, 4)}[name]
                graph = factory(count)
                row.update(status="ok", kind=graph.kind,
                           nodes=[list(pair) for pair in graph.nodes])
            rows.append(row)
    return rows


def evaluate(inputs: list[dict], graphs: list[dict], arms: list[bool]):
    """Serialize existing measurements and paired intervals without float conversion."""
    results, pairs = [], []
    for row in inputs:
        family = pilot.BlockFamily(tuple(from_word(v) for v in row["maxima_fp32"]),
                                   tuple(row["masses"]))
        measured = {}
        for graph in graphs:
            if graph["leaf_count"] != len(family.maxima):
                continue
            for fused in arms:
                key = {"family_id": row["family_id"], "graph_id": graph["graph_id"],
                       "fused": fused, "exp": EXP}
                if graph["status"] == "not_applicable":
                    results.append(dict(key, status="not_applicable", reason=graph["reason"]))
                    continue
                schedule = schedules.Schedule(graph["leaf_count"],
                                               tuple(tuple(p) for p in graph["nodes"]),
                                               graph["kind"])
                schedules.check_schedule(schedule)
                value = pilot.measure(family, schedule, fused=fused)
                measured[graph["requested"], fused] = value
                results.append(dict(key, status="ok", computed_fp32=fp32_word(value.computed),
                                    error_low=_fraction(value.error_low),
                                    error_high=_fraction(value.error_high),
                                    absorbed_merges=value.absorbed_merges,
                                    exp_underflow_edges=value.exp_underflow_edges,
                                    product_underflow_edges=value.product_underflow_edges))
        for (name, fused), value in measured.items():
            if name == "balanced":
                continue
            baseline = measured["balanced", fused]
            low, high = value.error_difference_interval(baseline)
            pairs.append({"family_id": row["family_id"], "schedule": name,
                          "baseline": "balanced", "fused": fused,
                          "difference_low": _fraction(low), "difference_high": _fraction(high),
                          "error_order": value.error_order(baseline)})
    return results, pairs


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(["git", *arguments], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(plan_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    plan = _read(plan_path)
    validate_plan(plan)
    prediction = (plan_path.parent / plan["prediction_record"]).read_bytes()
    if not prediction.strip():
        raise ValueError("Prediction record must not be empty")
    # Reject an existing destination before generation or measurement. Never overwrite evidence.
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": SCHEMA, "stage": plan["stage"], "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": _git("rev-parse", "HEAD"),
                "git_status": _git("status", "--porcelain"),
                "python": sys.version, "platform": platform.platform(),
                "arithmetic": {"exp": EXP, "rounding": "RN-even", "ftz": False,
                               "exp_decimal_precision": DEFAULT_PRECISION,
                               "reference_taylor_terms": pilot.TAYLOR_TERMS,
                               "reference_grid_bits": pilot.BRACKET_GRID.bit_length() - 1,
                               "zero": "canonical +0", "hardware": "CPU exact simulation"},
                "summary_rule": "per-family A_schedule - A_balanced within each FMA arm; "
                                "exact numerical intervals only, no statistical CI or pooled ratio"}
    _write(output / "manifest.json", metadata)
    _write(output / "plan.json", plan)
    (output / "prediction.md").write_bytes(prediction)
    source_dir = output / "sources" / "online"
    source_dir.mkdir(parents=True)
    for name in SOURCE_NAMES:
        (source_dir / (name + ".txt")).write_bytes((MODULE_DIR / name).read_bytes())
    inputs, graphs = build_inputs(plan), build_graphs(plan)
    _write(output / "inputs.json", inputs)
    _write(output / "graphs.json", graphs)
    results, pairs = evaluate(inputs, graphs, plan["fused"])
    _write(output / "measurements.json", results)
    _write(output / "paired_differences.json", pairs)
    # Refuse a completed run if its executing source files were edited during measurement.
    for name in SOURCE_NAMES:
        if (source_dir / (name + ".txt")).read_bytes() != (MODULE_DIR / name).read_bytes():
            raise ValueError(f"Source changed during measurement: {name}")
    metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started,
                    family_count=len(inputs), measurement_count=len(results),
                    files={path.relative_to(output).as_posix(): _sha(path.read_bytes())
                           for path in sorted(output.rglob("*"))
                           if path.is_file() and path.name != "manifest.json"})
    _write(output / "manifest.json", metadata)
    return metadata


def replay(output: Path) -> dict:
    metadata = _read(output / "manifest.json")
    if metadata["schema"] != SCHEMA or metadata["status"] != "complete":
        raise ValueError("Replay requires a complete supported bundle")
    expected = {"plan.json", "prediction.md", "inputs.json", "graphs.json", "measurements.json",
                "paired_differences.json", *(f"sources/online/{name}.txt" for name in SOURCE_NAMES)}
    if set(metadata["files"]) != expected:
        raise ValueError("Bundle file inventory mismatch")
    for name, digest in metadata["files"].items():
        if _sha((output / name).read_bytes()) != digest:
            raise ValueError(f"Bundle integrity mismatch: {name}")
    for name in SOURCE_NAMES:
        if (output / "sources" / "online" / (name + ".txt")).read_bytes() != (MODULE_DIR / name).read_bytes():
            raise ValueError(f"Source mismatch: {name}; use the recorded version for strict replay")
    plan = _read(output / "plan.json")
    validate_plan(plan)
    inputs, graphs = _read(output / "inputs.json"), _read(output / "graphs.json")
    results, pairs = evaluate(inputs, graphs, plan["fused"])
    if results != _read(output / "measurements.json"):
        raise ValueError("Measurement replay mismatch")
    if pairs != _read(output / "paired_differences.json"):
        raise ValueError("Paired-difference replay mismatch")
    return {"status": "exact_match", "families": len(inputs), "measurements": len(results),
            "paired_differences": len(pairs), "stage": plan["stage"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("replay")
    command.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.plan, args.output) if args.command == "run" else replay(args.bundle)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"pilot runner: {error}\n")
    if args.command == "run":
        result = {key: result[key] for key in ("status", "stage", "family_count", "measurement_count")}
    print(_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
