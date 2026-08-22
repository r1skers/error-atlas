"""Verify frozen-score fidelity and benchmark the score-only fixed-K8/B3 selector.

This is an engineering validation, not a new predictor-efficacy experiment.  The completed v2
artifacts are read only to check that removing research instrumentation changes no Q score, beam
score, shortlist, or selected graph.  No oracle target is used to tune or choose implementation
parameters.

Timing keeps model loading and graph construction outside the measured region.  The optional exact
oracle timing is an isolated comparison baseline; it is never called by the inference module.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from statistics import median

import numpy as np

from predictor_fixed_k8_beam_inference import (
    FROZEN_SHORTLIST_SIZE,
    InnovationModel,
    SelectionResult,
    _beam_score,
    _bits_to_units,
    _macro_trace,
    _shortlist_indices,
    _stable_min,
    select_tree,
)
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import predict_fp32_tree_error


HERE = Path(__file__).resolve().parent
HELDOUT = HERE / "results" / "wide_range_fixed_k8_beam_v2" / "heldout"
DEFAULT_GROUPS_PER_WIDTH = 4
DEFAULT_REPEATS = 5
DEFAULT_REDUCTION_REPEATS = 20


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
    return tuple(graphs)


def _input_records() -> list[dict]:
    with (HELDOUT / "input_groups.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _score_records() -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    with (HELDOUT / "graph_observations.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row["input_group_id"], []).append(row)
    return groups


def _group_index(record: dict) -> int:
    return int(record["input_group_id"].rsplit("g", 1)[1])


def _fidelity(model: InnovationModel, limit: int | None) -> dict:
    inputs = _input_records()
    if limit is not None:
        inputs = inputs[:limit]
    stored = _score_records()
    overall = {
        "groups": 0,
        "q_score_exact_groups": 0,
        "beam_score_exact_groups": 0,
        "shortlist_exact_groups": 0,
        "q_selection_exact_groups": 0,
        "beam_selection_exact_groups": 0,
        "max_abs_q_score_difference": 0.0,
        "max_abs_beam_score_difference": 0.0,
    }
    widths: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for position, record in enumerate(inputs, 1):
        width = int(record["width"])
        result = select_tree(
            record["stored_leaf_bits"],
            _heldout_graphs(width, _group_index(record)),
            model,
        )
        rows = stored[record["input_group_id"]]
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
        q_differences = [
            abs(score - float(row["fixed_q_score"]))
            for score, row in zip(result.q_scores, rows, strict=True)
        ]
        beam_differences = [
            abs(result.beam_scores[index] - float(rows[index]["beam_score"]))
            for index in result.shortlist_indices
        ]
        q_exact = all(difference == 0.0 for difference in q_differences)
        beam_exact = all(difference == 0.0 for difference in beam_differences)
        shortlist_exact = result.shortlist_indices == expected_shortlist
        q_selection_exact = result.q_selected_index == expected_q
        beam_selection_exact = result.selected_index == expected_beam

        overall["groups"] += 1
        overall["q_score_exact_groups"] += int(q_exact)
        overall["beam_score_exact_groups"] += int(beam_exact)
        overall["shortlist_exact_groups"] += int(shortlist_exact)
        overall["q_selection_exact_groups"] += int(q_selection_exact)
        overall["beam_selection_exact_groups"] += int(beam_selection_exact)
        overall["max_abs_q_score_difference"] = max(
            overall["max_abs_q_score_difference"],
            max(q_differences),
        )
        overall["max_abs_beam_score_difference"] = max(
            overall["max_abs_beam_score_difference"],
            max(beam_differences),
        )
        width_row = widths.setdefault(
            str(width),
            {
                "groups": 0,
                "score_exact_groups": 0,
                "selection_exact_groups": 0,
            },
        )
        width_row["groups"] += 1
        width_row["score_exact_groups"] += int(q_exact and beam_exact)
        width_row["selection_exact_groups"] += int(
            shortlist_exact and q_selection_exact and beam_selection_exact
        )
        if position % 32 == 0:
            print(f"fidelity progress {position}/{len(inputs)}", flush=True)
    overall["elapsed_seconds"] = time.perf_counter() - started
    overall["all_scores_exact"] = bool(
        overall["q_score_exact_groups"] == overall["groups"]
        and overall["beam_score_exact_groups"] == overall["groups"]
    )
    overall["all_decisions_exact"] = bool(
        overall["shortlist_exact_groups"] == overall["groups"]
        and overall["q_selection_exact_groups"] == overall["groups"]
        and overall["beam_selection_exact_groups"] == overall["groups"]
    )
    return {"overall": overall, "by_width": widths}


def _profile_selector(
    leaf_bits: list[int],
    graphs,
    model: InnovationModel,
) -> tuple[dict[str, float], SelectionResult]:
    started = time.perf_counter_ns()
    leaf_units = tuple(_bits_to_units(bits) for bits in leaf_bits)
    after_input = time.perf_counter_ns()
    traces = tuple(_macro_trace(leaf_units, graph) for graph in graphs)
    q_scores = tuple(trace.q_score for trace in traces)
    shortlist = _shortlist_indices(q_scores)
    after_macro = time.perf_counter_ns()
    beam_scores: list[float | None] = [None] * len(graphs)
    for index in shortlist:
        beam_scores[index] = _beam_score(traces[index], model)
    after_beam = time.perf_counter_ns()
    result = SelectionResult(
        selected_index=_stable_min(shortlist, beam_scores),
        q_selected_index=_stable_min(range(len(graphs)), q_scores),
        shortlist_indices=shortlist,
        q_scores=q_scores,
        beam_scores=tuple(beam_scores),
    )
    scale = 1.0e-6
    return (
        {
            "input_ms": (after_input - started) * scale,
            "macro_ms": (after_macro - after_input) * scale,
            "beam_ms": (after_beam - after_macro) * scale,
            "total_ms": (after_beam - started) * scale,
        },
        result,
    )


def _execute_graph_fp32(values: np.ndarray, graph) -> np.float32:
    states = np.empty(graph.leaf_count + len(graph.nodes), dtype=np.float32)
    states[: graph.leaf_count] = values
    for offset, node in enumerate(graph.nodes):
        states[graph.leaf_count + offset] = np.float32(
            states[node.left] + states[node.right]
        )
    return states[graph.root]


def _leaf_float32(bits: list[int]) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _leaf_fractions(bits: list[int]) -> tuple[Fraction, ...]:
    denominator = 1 << 149
    return tuple(Fraction(_bits_to_units(value), denominator) for value in bits)


def _time_average(call, repeats: int) -> float:
    started = time.perf_counter_ns()
    for _ in range(repeats):
        call()
    return (time.perf_counter_ns() - started) * 1.0e-6 / repeats


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(samples: list[float]) -> dict[str, float]:
    return {
        "median": median(samples),
        "p90": _percentile(samples, 0.90),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _benchmark_width(
    records: list[dict],
    model: InnovationModel,
    repeats: int,
    reduction_repeats: int,
    include_oracle: bool,
) -> dict:
    stage_samples = {name: [] for name in ("input_ms", "macro_ms", "beam_ms", "total_ms")}
    one_tree_ms: list[float] = []
    all_tree_ms: list[float] = []
    oracle_ms: list[float] = []
    measured_groups = []

    for record in records:
        graphs = _heldout_graphs(record["width"], _group_index(record))
        leaf_bits = record["stored_leaf_bits"]
        leaf_values = _leaf_float32(leaf_bits)
        _, warm_result = _profile_selector(leaf_bits, graphs, model)
        for _ in range(repeats):
            profile, result = _profile_selector(leaf_bits, graphs, model)
            if result != warm_result:
                raise AssertionError("selector changed across deterministic timing repeats")
            for name, value in profile.items():
                stage_samples[name].append(value)

        selected_graph = graphs[warm_result.selected_index]
        one_tree_ms.append(
            _time_average(
                lambda: _execute_graph_fp32(leaf_values, selected_graph),
                reduction_repeats,
            )
        )
        all_tree_ms.append(
            _time_average(
                lambda: [_execute_graph_fp32(leaf_values, graph) for graph in graphs],
                max(1, reduction_repeats // 10),
            )
        )
        if include_oracle:
            fractions = _leaf_fractions(leaf_bits)
            oracle_ms.append(
                _time_average(
                    lambda: [predict_fp32_tree_error(fractions, graph) for graph in graphs],
                    1,
                )
            )
        measured_groups.append(record["input_group_id"])

    first = records[0]
    first_graphs = _heldout_graphs(first["width"], _group_index(first))
    tracemalloc.start()
    select_tree(first["stored_leaf_bits"], first_graphs, model)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_median = median(stage_samples["total_ms"])
    one_tree_median = median(one_tree_ms)
    all_tree_median = median(all_tree_ms)
    result = {
        "width": int(first["width"]),
        "candidate_count": len(first_graphs),
        "shortlist_size": FROZEN_SHORTLIST_SIZE,
        "measured_input_groups": measured_groups,
        "selector_stage_ms": {
            name.removesuffix("_ms"): _timing_summary(values)
            for name, values in stage_samples.items()
        },
        "python_fp32_execution_ms": {
            "one_selected_tree": _timing_summary(one_tree_ms),
            "all_64_candidates": _timing_summary(all_tree_ms),
        },
        "selector_over_one_tree_ratio_median": total_median / one_tree_median,
        "selector_over_all_64_execution_ratio_median": total_median / all_tree_median,
        "reuses_for_100_percent_amortized_overhead": total_median / one_tree_median,
        "reuses_for_10_percent_amortized_overhead": 10.0 * total_median / one_tree_median,
        "algorithmic_full_tree_passes": 64 + FROZEN_SHORTLIST_SIZE,
        "tracemalloc_peak_mib_one_selection": peak_bytes / (1024.0 * 1024.0),
    }
    if oracle_ms:
        oracle_median = median(oracle_ms)
        result["fraction_oracle_all_64_ms"] = _timing_summary(oracle_ms)
        result["selector_speedup_over_fraction_oracle_median"] = (
            oracle_median / total_median
        )
    return result


def _git_state() -> tuple[str | None, str | None]:
    repo = HERE.parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, tree


def _print_report(report: dict) -> None:
    fidelity = report["fidelity"]["overall"]
    print("\nscore fidelity")
    print(
        f"  groups={fidelity['groups']} scores_exact={fidelity['all_scores_exact']} "
        f"decisions_exact={fidelity['all_decisions_exact']} "
        f"max_diff Q/beam={fidelity['max_abs_q_score_difference']:.3g}/"
        f"{fidelity['max_abs_beam_score_difference']:.3g}"
    )
    print("\nscore-only timing (median; graph generation/model load excluded)")
    for row in report["benchmark"]:
        stages = row["selector_stage_ms"]
        execution = row["python_fp32_execution_ms"]
        print(
            f"  width={row['width']} selector={stages['total']['median']:.3f} ms "
            f"(macro={stages['macro']['median']:.3f}, beam={stages['beam']['median']:.3f})"
        )
        print(
            f"    one-tree={execution['one_selected_tree']['median']:.3f} ms "
            f"all64={execution['all_64_candidates']['median']:.3f} ms "
            f"selector/one={row['selector_over_one_tree_ratio_median']:.1f}x "
            f"selector/all64={row['selector_over_all_64_execution_ratio_median']:.2f}x"
        )
        if "selector_speedup_over_fraction_oracle_median" in row:
            print(
                f"    speedup over Fraction oracle all64="
                f"{row['selector_speedup_over_fraction_oracle_median']:.1f}x"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups-per-width",
        type=int,
        default=DEFAULT_GROUPS_PER_WIDTH,
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--reduction-repeats",
        type=int,
        default=DEFAULT_REDUCTION_REPEATS,
    )
    parser.add_argument(
        "--fidelity-limit",
        type=int,
        help="debug-only prefix limit; omit to replay all 192 v2 groups",
    )
    parser.add_argument("--skip-oracle-baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.groups_per_width <= 0 or args.repeats <= 0 or args.reduction_repeats <= 0:
        parser.error("group and repeat counts must be positive")
    if args.fidelity_limit is not None and args.fidelity_limit <= 0:
        parser.error("--fidelity-limit must be positive")

    model = InnovationModel.from_json(HELDOUT / "calibration_model.json")
    fidelity = _fidelity(model, args.fidelity_limit)
    inputs = _input_records()
    by_width: dict[int, list[dict]] = {}
    for record in inputs:
        by_width.setdefault(int(record["width"]), []).append(record)
    benchmark = [
        _benchmark_width(
            records[: args.groups_per_width],
            model,
            args.repeats,
            args.reduction_repeats,
            include_oracle=not args.skip_oracle_baseline,
        )
        for _, records in sorted(by_width.items())
    ]
    commit, tree = _git_state()
    report = {
        "schema_version": "1",
        "kind": "engineering_fidelity_and_cost_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_tree": tree,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "configuration": {
            "groups_per_width": args.groups_per_width,
            "selector_repeats": args.repeats,
            "reduction_repeats": args.reduction_repeats,
            "fraction_oracle_baseline": not args.skip_oracle_baseline,
            "graph_generation_timed": False,
            "model_loading_timed": False,
            "fidelity_scope": "all_v2_groups" if args.fidelity_limit is None else "prefix",
        },
        "fidelity": fidelity,
        "benchmark": benchmark,
        "interpretation_limits": [
            "Python prototype timings are not compiled-kernel hardware claims.",
            "The efficacy heldout is reused only for frozen-score implementation fidelity.",
            "The selector improves numerical ranking, not tree work; selection has no intrinsic runtime payback.",
            "Amortization across changing inputs is not validated because the selector is input-dependent.",
        ],
    }
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
