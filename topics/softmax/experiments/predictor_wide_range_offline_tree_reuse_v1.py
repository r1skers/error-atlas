"""One-shot validation of width-specific offline tree reuse.

The frozen K8/B3 score improved per-input ranking, but its online implementation needs 68
full-tree-equivalent traversals.  This experiment changes the deployment contract instead of
tuning the score: score a fixed candidate catalog on representative calibration inputs, choose one
tree per width, and reuse that tree on unseen inputs with zero online selection work.

The primary policy is label-free.  It averages the complete cascade rank induced by the frozen
selector over 32 calibration inputs.  Exact candidate execution is used only for evaluation and for
an explicitly labelled oracle-static ceiling.  Confirmation inputs, candidate catalogs, aggregation
rules, baselines, metrics, and gates are frozen in the preregistration beside this runner.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from statistics import mean

import numpy as np

from predictor_ancestor_cell_beam_score_calibration import _fp32_bits_to_fraction
from predictor_calibration_inputs import wide_range_random
from predictor_fixed_k8_beam_inference import (
    DEFAULT_MODEL_PATH,
    InnovationModel,
    SelectionResult,
    _bits_to_float,
    select_tree,
)
from predictor_shadow_sparse_repair_ablation import _fp32_ulp_fraction
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from predictor_wide_range_fixed_k8_beam_v2_heldout import (
    _bits_hash,
    _bootstrap_interval,
    _git_state,
    _graph_hash,
    _json_dump,
    _percentile,
    _reserve_output_directory,
    _sha256,
    _stored_leaf_bits,
    _write_csv,
)
from summation_graph_predictor import (
    balanced_reduction_graph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
    sequential_reduction_graph,
)


EXPERIMENT_ID = "wide_range_offline_tree_reuse_v1"
HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results" / EXPERIMENT_ID
PREREGISTRATION = RESULT_ROOT / f"{EXPERIMENT_ID}_preregistration.json"
OUTPUT_DIRECTORY = RESULT_ROOT / "confirmation"

EXPECTED_WIDTHS = (256, 512, 1024)
EXPECTED_CALIBRATION_GROUPS = 32
EXPECTED_CONFIRMATION_GROUPS = 64
EXPECTED_GRAPHS_PER_FAMILY = 32
EXPECTED_REPRESENTATIVE_COUNTS = (1, 2, 4, 8, 16, 32)
EXPECTED_BOOTSTRAP_RESAMPLES = 20_000
EXPECTED_BOOTSTRAP_SEED = 20260824
EXPECTED_SEVERE_REGRET = 0.5
EXPECTED_MODEL_SHA256 = "23f028e578ca18e55643bf1a080d22929f059be7c8035900153b0a6b95d95b7e"
CONTIGUOUS_CATALOG_BASE_SEED = 57_000_000
PAIR_CATALOG_BASE_SEED = 58_000_000


@dataclass(frozen=True)
class EvaluatedGroup:
    width: int
    seed: int
    leaf_bits: tuple[int, ...]
    exact_sum: Fraction
    root_ulp: Fraction
    correct_bits: int
    score: SelectionResult
    cascade_rank: tuple[int, ...]
    q_rank: tuple[int, ...]
    targets: tuple[float, ...]
    signed_error_ulp: tuple[float, ...]
    output_bits: tuple[int, ...]
    regrets: tuple[float, ...]


def _derived_seed(split: str, width: int, index: int) -> int:
    if split not in {"calibration", "confirmation"}:
        raise ValueError("split must be calibration or confirmation")
    if width not in EXPECTED_WIDTHS:
        raise ValueError("unexpected width")
    if index < 0:
        raise ValueError("seed index must be nonnegative")
    payload = f"{EXPERIMENT_ID}|{split}|{width}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


@lru_cache(maxsize=None)
def _catalog(width: int):
    if width not in EXPECTED_WIDTHS:
        raise ValueError("unexpected width")
    graphs = []
    for index in range(EXPECTED_GRAPHS_PER_FAMILY):
        graphs.append(
            (
                "contiguous",
                random_contiguous_split_graph(
                    width,
                    seed=CONTIGUOUS_CATALOG_BASE_SEED + width * 10_000 + index,
                ),
            )
        )
        graphs.append(
            (
                "pair_merge",
                random_pair_merge_graph(
                    width,
                    seed=PAIR_CATALOG_BASE_SEED + width * 10_000 + index,
                ),
            )
        )
    return tuple(graphs)


def _rank_vector(order: tuple[int, ...], size: int) -> tuple[int, ...]:
    if tuple(sorted(order)) != tuple(range(size)):
        raise ValueError("rank order must be a permutation")
    rank = [0] * size
    for position, index in enumerate(order):
        rank[index] = position
    return tuple(rank)


def _cascade_order(result: SelectionResult) -> tuple[int, ...]:
    shortlisted = tuple(
        sorted(
            result.shortlist_indices,
            key=lambda index: (float(result.beam_scores[index]), index),
        )
    )
    shortlist_set = set(shortlisted)
    remainder = tuple(
        sorted(
            (index for index in range(len(result.q_scores)) if index not in shortlist_set),
            key=lambda index: (result.q_scores[index], index),
        )
    )
    return shortlisted + remainder


def _normalized_regrets(targets: tuple[float, ...]) -> tuple[float, ...]:
    best = min(targets)
    worst = max(targets)
    if worst == best:
        return tuple(0.0 for _ in targets)
    return tuple((target - best) / (worst - best) for target in targets)


def _evaluate_group(width: int, seed: int, model: InnovationModel) -> EvaluatedGroup:
    values = wide_range_random(width, seed=seed).values
    bits = tuple(_stored_leaf_bits(values))
    graphs = tuple(graph for _, graph in _catalog(width))
    score = select_tree(bits, graphs, model)
    exact_sum = sum(values, start=Fraction(0))
    root_ulp = _fp32_ulp_fraction(exact_sum)
    correct_bits = round_nonnegative_fraction_to_fp32(exact_sum).bits

    targets = []
    signed_error_ulp = []
    output_bits = []
    for graph in graphs:
        oracle = predict_fp32_tree_error(values, graph)
        signed = float(oracle.signed_error / root_ulp)
        targets.append(signed * signed)
        signed_error_ulp.append(signed)
        output_bits.append(int(oracle.predicted_sum_bits, 16))

    cascade_order = _cascade_order(score)
    q_order = tuple(
        sorted(range(len(graphs)), key=lambda index: (score.q_scores[index], index))
    )
    return EvaluatedGroup(
        width=width,
        seed=seed,
        leaf_bits=bits,
        exact_sum=exact_sum,
        root_ulp=root_ulp,
        correct_bits=correct_bits,
        score=score,
        cascade_rank=_rank_vector(cascade_order, len(graphs)),
        q_rank=_rank_vector(q_order, len(graphs)),
        targets=tuple(targets),
        signed_error_ulp=tuple(signed_error_ulp),
        output_bits=tuple(output_bits),
        regrets=_normalized_regrets(tuple(targets)),
    )


def _minimum_mean_index(rows: list[tuple[float, ...]]) -> int:
    if not rows or not rows[0]:
        raise ValueError("selection needs a nonempty rectangular matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("selection matrix must be rectangular")
    return min(
        range(width),
        key=lambda index: (mean(row[index] for row in rows), index),
    )


def _selection_record(groups: list[EvaluatedGroup]) -> dict:
    if len(groups) != EXPECTED_CALIBRATION_GROUPS:
        raise ValueError("selection requires every frozen calibration group")
    if len({group.width for group in groups}) != 1:
        raise ValueError("selection is width-specific")
    score_by_count = {}
    q_by_count = {}
    oracle_by_count = {}
    for count in EXPECTED_REPRESENTATIVE_COUNTS:
        prefix = groups[:count]
        score_by_count[str(count)] = _minimum_mean_index(
            [group.cascade_rank for group in prefix]
        )
        q_by_count[str(count)] = _minimum_mean_index([group.q_rank for group in prefix])
        oracle_by_count[str(count)] = _minimum_mean_index(
            [group.regrets for group in prefix]
        )

    wins = Counter(group.score.selected_index for group in groups)
    probabilities = [count / len(groups) for count in wins.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized_entropy = entropy / math.log(len(_catalog(groups[0].width)))
    final_count = str(EXPECTED_REPRESENTATIVE_COUNTS[-1])
    return {
        "width": groups[0].width,
        "calibration_group_count": len(groups),
        "score_static_by_representative_count": score_by_count,
        "q_static_by_representative_count": q_by_count,
        "oracle_static_by_representative_count": oracle_by_count,
        "primary_score_static_index": score_by_count[final_count],
        "q_static_index": q_by_count[final_count],
        "oracle_static_index": oracle_by_count[final_count],
        "per_input_beam_win_counts": {
            str(index): count for index, count in sorted(wins.items())
        },
        "per_input_beam_top_win_share": max(wins.values()) / len(groups),
        "per_input_beam_normalized_entropy": normalized_entropy,
    }


def _fp64_then_fp32_bits(leaf_bits: tuple[int, ...]) -> int:
    total = 0.0
    for bits in leaf_bits:
        total = total + _bits_to_float(bits)
    return round_nonnegative_fraction_to_fp32(Fraction.from_float(total)).bits


def _kahan_fp32_bits(leaf_bits: tuple[int, ...]) -> int:
    values = np.asarray(leaf_bits, dtype=np.uint32).view(np.float32)
    total = np.float32(0.0)
    compensation = np.float32(0.0)
    with np.errstate(over="raise", invalid="raise"):
        for value in values:
            adjusted = np.float32(value - compensation)
            updated = np.float32(total + adjusted)
            compensation = np.float32((updated - total) - adjusted)
            total = updated
    return int(np.asarray(total, dtype=np.float32).view(np.uint32).item())


def _output_metrics(bits: int, group: EvaluatedGroup) -> dict[str, float | int]:
    value = _fp32_bits_to_fraction(bits)
    signed = float((value - group.exact_sum) / group.root_ulp)
    return {
        "target": signed * signed,
        "abs_error_ulp": abs(signed),
        "correct_hit": int(bits == group.correct_bits),
        "output_bits": bits,
    }


def _candidate_metrics(index: int, group: EvaluatedGroup) -> dict[str, float | int]:
    return {
        "target": group.targets[index],
        "abs_error_ulp": abs(group.signed_error_ulp[index]),
        "correct_hit": int(group.output_bits[index] == group.correct_bits),
        "output_bits": group.output_bits[index],
        "regret": group.regrets[index],
        "best_hit": int(group.targets[index] == min(group.targets)),
    }


def _confirmation_row(group: EvaluatedGroup, selection: dict) -> dict:
    graphs = _catalog(group.width)
    score_static = int(selection["primary_score_static_index"])
    q_static = int(selection["q_static_index"])
    oracle_static = int(selection["oracle_static_index"])
    per_input_beam = group.score.selected_index
    per_input_q = group.score.q_selected_index
    candidate_policies = {
        "score_static": score_static,
        "q_static": q_static,
        "oracle_static": oracle_static,
        "representative_1": int(
            selection["score_static_by_representative_count"]["1"]
        ),
        "per_input_beam": per_input_beam,
        "per_input_q": per_input_q,
    }

    row: dict[str, float | int | str] = {
        "schema_version": "1",
        "width": group.width,
        "seed": group.seed,
        "candidate_count": len(graphs),
        "target_unique": len(set(group.targets)),
        "random_expected_regret": mean(group.regrets),
        "catalog_best_target": min(group.targets),
        "catalog_worst_target": max(group.targets),
        "correctly_rounded_bits": f"0x{group.correct_bits:08x}",
    }
    for label, index in candidate_policies.items():
        metrics = _candidate_metrics(index, group)
        row[f"{label}_index"] = index
        row[f"{label}_family"] = graphs[index][0]
        for field, value in metrics.items():
            row[f"{label}_{field}"] = value

    for count in EXPECTED_REPRESENTATIVE_COUNTS:
        index = int(selection["score_static_by_representative_count"][str(count)])
        row[f"score_static_r{count}_index"] = index
        row[f"score_static_r{count}_regret"] = group.regrets[index]

    values = tuple(_fp32_bits_to_fraction(bits) for bits in group.leaf_bits)
    balanced = predict_fp32_tree_error(values, balanced_reduction_graph(group.width))
    sequential = predict_fp32_tree_error(values, sequential_reduction_graph(group.width))
    baseline_bits = {
        "balanced_fp32": int(balanced.predicted_sum_bits, 16),
        "sequential_fp32": int(sequential.predicted_sum_bits, 16),
        "fp64_then_fp32": _fp64_then_fp32_bits(group.leaf_bits),
        "kahan_fp32": _kahan_fp32_bits(group.leaf_bits),
    }
    for label, bits in baseline_bits.items():
        for field, value in _output_metrics(bits, group).items():
            row[f"{label}_{field}"] = value
    return row


def _policy_summary(rows: list[dict], label: str) -> dict:
    regrets = [float(row[f"{label}_regret"]) for row in rows]
    targets = [float(row[f"{label}_target"]) for row in rows]
    return {
        "mean_normalized_regret": mean(regrets),
        "regret_p90": _percentile(regrets, 0.90),
        "severe_regret_rate": mean(value >= EXPECTED_SEVERE_REGRET for value in regrets),
        "best_tier_hit": mean(float(row[f"{label}_best_hit"]) for row in rows),
        "mean_squared_error_root_ulp": mean(targets),
        "target_p90": _percentile(targets, 0.90),
        "mean_abs_error_root_ulp": mean(
            float(row[f"{label}_abs_error_ulp"]) for row in rows
        ),
        "correctly_rounded_hit": mean(
            float(row[f"{label}_correct_hit"]) for row in rows
        ),
    }


def _method_summary(rows: list[dict], label: str) -> dict:
    targets = [float(row[f"{label}_target"]) for row in rows]
    return {
        "mean_squared_error_root_ulp": mean(targets),
        "target_p90": _percentile(targets, 0.90),
        "mean_abs_error_root_ulp": mean(
            float(row[f"{label}_abs_error_ulp"]) for row in rows
        ),
        "correctly_rounded_hit": mean(
            float(row[f"{label}_correct_hit"]) for row in rows
        ),
    }


def _comparison(rows: list[dict], field, seed: int) -> dict:
    value = mean(field(row) for row in rows)
    interval = _bootstrap_interval(
        rows,
        field,
        EXPECTED_BOOTSTRAP_RESAMPLES,
        seed,
        stratify=len({int(row["width"]) for row in rows}) > 1,
    )
    return {
        "mean": value,
        "95_ci": list(interval),
        "positive": interval[0] > 0.0,
    }


def _summary(rows: list[dict]) -> dict:
    policies = (
        "score_static",
        "q_static",
        "oracle_static",
        "representative_1",
        "per_input_beam",
        "per_input_q",
    )
    methods = ("balanced_fp32", "sequential_fp32", "fp64_then_fp32", "kahan_fp32")
    reuse = _comparison(
        rows,
        lambda row: float(row["random_expected_regret"])
        - float(row["score_static_regret"]),
        EXPECTED_BOOTSTRAP_SEED,
    )
    balanced_minus_score = _comparison(
        rows,
        lambda row: float(row["balanced_fp32_target"])
        - float(row["score_static_target"]),
        EXPECTED_BOOTSTRAP_SEED + 1,
    )
    balanced_minus_oracle_static = _comparison(
        rows,
        lambda row: float(row["balanced_fp32_target"])
        - float(row["oracle_static_target"]),
        EXPECTED_BOOTSTRAP_SEED + 2,
    )
    score_minus_fp64 = _comparison(
        rows,
        lambda row: float(row["score_static_target"])
        - float(row["fp64_then_fp32_target"]),
        EXPECTED_BOOTSTRAP_SEED + 3,
    )
    score_by_representatives = {}
    for count in EXPECTED_REPRESENTATIVE_COUNTS:
        regrets = [float(row[f"score_static_r{count}_regret"]) for row in rows]
        score_by_representatives[str(count)] = {
            "mean_normalized_regret": mean(regrets),
            "regret_p90": _percentile(regrets, 0.90),
        }
    return {
        "group_count": len(rows),
        "random_expected_regret": mean(float(row["random_expected_regret"]) for row in rows),
        "candidate_policies": {
            label: _policy_summary(rows, label) for label in policies
        },
        "direct_methods": {label: _method_summary(rows, label) for label in methods},
        "score_static_by_representative_count": score_by_representatives,
        "comparisons": {
            "random_minus_score_static_normalized_regret": reuse,
            "balanced_minus_score_static_squared_error": balanced_minus_score,
            "balanced_minus_oracle_static_squared_error": balanced_minus_oracle_static,
            "score_static_minus_fp64_squared_error": score_minus_fp64,
        },
        "engineering_gates": {
            "reuse_signal_over_random_fixed_catalog": reuse["positive"],
            "score_static_beats_balanced_fp32": balanced_minus_score["positive"],
            "oracle_static_ceiling_beats_balanced_fp32": balanced_minus_oracle_static[
                "positive"
            ],
            "offline_reuse_deployment_go": reuse["positive"]
            and balanced_minus_score["positive"],
        },
    }


def _load_and_validate_preregistration(path: Path = PREREGISTRATION) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_not_opened",
        "widths": list(EXPECTED_WIDTHS),
        "calibration_groups_per_width": EXPECTED_CALIBRATION_GROUPS,
        "confirmation_groups_per_width": EXPECTED_CONFIRMATION_GROUPS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen preregistration mismatch: {key}")
    if config["graphs"]["count_per_family"] != EXPECTED_GRAPHS_PER_FAMILY:
        raise ValueError("frozen candidate count mismatch")
    if config["offline_policy"]["representative_counts"] != list(
        EXPECTED_REPRESENTATIVE_COUNTS
    ):
        raise ValueError("frozen representative counts mismatch")
    if config["uncertainty"] != {
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "method": "stratified nonparametric input-group bootstrap",
        "resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
    }:
        raise ValueError("frozen uncertainty mismatch")
    if _sha256(DEFAULT_MODEL_PATH) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen v2 model bytes changed")

    seed_sets = {}
    for split, count in (
        ("calibration", EXPECTED_CALIBRATION_GROUPS),
        ("confirmation", EXPECTED_CONFIRMATION_GROUPS),
    ):
        seeds = {
            _derived_seed(split, width, index)
            for width in EXPECTED_WIDTHS
            for index in range(count)
        }
        if len(seeds) != len(EXPECTED_WIDTHS) * count:
            raise ValueError(f"{split} seed collision")
        seed_sets[split] = seeds
    if seed_sets["calibration"] & seed_sets["confirmation"]:
        raise ValueError("calibration and confirmation seeds overlap")

    prior_seeds = set()
    for relative in (
        "results/wide_range_energy_beam_v1/wide_range_energy_beam_v1_preregistration.json",
        "results/wide_range_fixed_k8_beam_v2/wide_range_fixed_k8_beam_v2_preregistration.json",
    ):
        with (HERE / relative).open(encoding="utf-8") as handle:
            prior = json.load(handle)
        for width in EXPECTED_WIDTHS:
            prior_seeds.update(prior["heldout_seeds"][str(width)])
    if (seed_sets["calibration"] | seed_sets["confirmation"]) & prior_seeds:
        raise ValueError("new seeds overlap a prior held-out stage")
    return config


def _calibration_artifacts(groups: list[EvaluatedGroup]) -> tuple[list[dict], list[dict]]:
    inputs = []
    observations = []
    for group_index, group in enumerate(groups):
        group_id = f"wr{group.width}_reuse_cal_g{group_index:03d}"
        inputs.append(
            {
                "schema_version": "1",
                "input_group_id": group_id,
                "width": group.width,
                "seed": group.seed,
                "stored_leaf_bits_sha256": _bits_hash(list(group.leaf_bits)),
                "stored_leaf_bits": list(group.leaf_bits),
            }
        )
        for index, (family, graph) in enumerate(_catalog(group.width)):
            observations.append(
                {
                    "schema_version": "1",
                    "input_group_id": group_id,
                    "width": group.width,
                    "seed": group.seed,
                    "tree_index": index,
                    "graph_family": family,
                    "graph_sha256": _graph_hash(graph),
                    "q_score": group.score.q_scores[index],
                    "beam_score": group.score.beam_scores[index],
                    "shortlisted": int(index in group.score.shortlist_indices),
                    "cascade_rank": group.cascade_rank[index],
                    "q_rank": group.q_rank[index],
                    "target_squared_root_ulp": group.targets[index],
                    "normalized_regret": group.regrets[index],
                }
            )
    return inputs, observations


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    config = _load_and_validate_preregistration()
    repo = HERE.parents[2]
    commit, tree = _git_state(repo)
    _reserve_output_directory(OUTPUT_DIRECTORY)
    opened_at = datetime.now(timezone.utc).isoformat()
    _json_dump(
        OUTPUT_DIRECTORY / "opening_manifest.json",
        {
            "schema_version": "1",
            "experiment_id": EXPERIMENT_ID,
            "status": "opening_started",
            "opened_at_utc": opened_at,
            "git_commit": commit,
            "git_tree": tree,
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "frozen_model_sha256": _sha256(DEFAULT_MODEL_PATH),
            "calibration_seeds": {
                str(width): [
                    _derived_seed("calibration", width, index)
                    for index in range(EXPECTED_CALIBRATION_GROUPS)
                ]
                for width in EXPECTED_WIDTHS
            },
            "confirmation_seeds": {
                str(width): [
                    _derived_seed("confirmation", width, index)
                    for index in range(EXPECTED_CONFIRMATION_GROUPS)
                ]
                for width in EXPECTED_WIDTHS
            },
        },
    )

    print("OPENING OFFLINE TREE REUSE V1 — protocol and gates are immutable", flush=True)
    model = InnovationModel.from_json(DEFAULT_MODEL_PATH)
    calibration_inputs = []
    calibration_observations = []
    selections = {}
    for width in EXPECTED_WIDTHS:
        width_groups = []
        for index in range(EXPECTED_CALIBRATION_GROUPS):
            group = _evaluate_group(
                width,
                _derived_seed("calibration", width, index),
                model,
            )
            width_groups.append(group)
            print(
                f"calibration width={width} group={index + 1:02d}/"
                f"{EXPECTED_CALIBRATION_GROUPS}",
                flush=True,
            )
        inputs, observations = _calibration_artifacts(width_groups)
        calibration_inputs.extend(inputs)
        calibration_observations.extend(observations)
        selections[str(width)] = _selection_record(width_groups)

    _write_jsonl(OUTPUT_DIRECTORY / "calibration_inputs.jsonl", calibration_inputs)
    _write_csv(
        OUTPUT_DIRECTORY / "calibration_graph_observations.csv",
        calibration_observations,
    )
    _json_dump(OUTPUT_DIRECTORY / "offline_selections.json", selections)

    confirmation_inputs = []
    confirmation_rows = []
    confirmation_observations = []
    for width in EXPECTED_WIDTHS:
        for index in range(EXPECTED_CONFIRMATION_GROUPS):
            group = _evaluate_group(
                width,
                _derived_seed("confirmation", width, index),
                model,
            )
            group_id = f"wr{width}_reuse_confirm_g{index:03d}"
            confirmation_inputs.append(
                {
                    "schema_version": "1",
                    "input_group_id": group_id,
                    "width": width,
                    "seed": group.seed,
                    "stored_leaf_bits_sha256": _bits_hash(list(group.leaf_bits)),
                    "stored_leaf_bits": list(group.leaf_bits),
                }
            )
            row = _confirmation_row(group, selections[str(width)])
            row["input_group_id"] = group_id
            confirmation_rows.append(row)
            for graph_index, (family, graph) in enumerate(_catalog(width)):
                confirmation_observations.append(
                    {
                        "schema_version": "1",
                        "input_group_id": group_id,
                        "width": width,
                        "seed": group.seed,
                        "tree_index": graph_index,
                        "graph_family": family,
                        "graph_sha256": _graph_hash(graph),
                        "signed_error_root_ulp": group.signed_error_ulp[graph_index],
                        "target_squared_root_ulp": group.targets[graph_index],
                        "normalized_regret": group.regrets[graph_index],
                        "oracle_best_tier": int(
                            group.targets[graph_index] == min(group.targets)
                        ),
                    }
                )
            print(
                f"confirmation width={width} group={index + 1:02d}/"
                f"{EXPECTED_CONFIRMATION_GROUPS} "
                f"static_regret={row['score_static_regret']:.3f} "
                f"balanced_target={row['balanced_fp32_target']:.3f}",
                flush=True,
            )

    _write_jsonl(OUTPUT_DIRECTORY / "confirmation_inputs.jsonl", confirmation_inputs)
    _write_csv(OUTPUT_DIRECTORY / "confirmation_group_metrics.csv", confirmation_rows)
    _write_csv(
        OUTPUT_DIRECTORY / "confirmation_graph_observations.csv",
        confirmation_observations,
    )
    summary = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "opened_at_utc": opened_at,
        "overall": _summary(confirmation_rows),
        "by_width": {
            str(width): _summary(
                [row for row in confirmation_rows if int(row["width"]) == width]
            )
            for width in EXPECTED_WIDTHS
        },
        "online_cost_contract": config["online_cost_contract"],
    }
    _json_dump(OUTPUT_DIRECTORY / "metric_summary.json", summary)

    artifact_names = [
        "opening_manifest.json",
        "calibration_inputs.jsonl",
        "calibration_graph_observations.csv",
        "offline_selections.json",
        "confirmation_inputs.jsonl",
        "confirmation_group_metrics.csv",
        "confirmation_graph_observations.csv",
        "metric_summary.json",
    ]
    _json_dump(
        OUTPUT_DIRECTORY / "metadata.json",
        {
            "schema_version": "1",
            "experiment_id": EXPERIMENT_ID,
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_tree": tree,
            "preregistration_sha256": _sha256(PREREGISTRATION),
            "artifacts": {
                name: {
                    "sha256": _sha256(OUTPUT_DIRECTORY / name),
                    "bytes": (OUTPUT_DIRECTORY / name).stat().st_size,
                }
                for name in artifact_names
            },
        },
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
