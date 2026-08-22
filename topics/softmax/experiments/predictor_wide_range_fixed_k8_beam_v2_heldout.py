"""Execute the frozen wide_range_fixed_k8_beam_v2 confirmation exactly once."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from predictor_ancestor_cell_beam_score_calibration import (
    _beam_scores,
    _beam_tree,
    _generate_width,
)
from predictor_ancestor_transition_predictability_diagnostic import SHADOW_DIMENSION
from predictor_calibration_inputs import wide_range_random
from predictor_q_beam_shortlist_cascade_calibration import (
    _selection_metrics,
    _shortlist_indices,
)
from predictor_ranking_smoke import _spearman
from predictor_signed_cell_shift_predictability_diagnostic import _fit_probe
from predictor_two_stage_cheap_score_calibration import (
    CONTIGUOUS_TREE_BASE_SEED,
    PAIR_TREE_BASE_SEED,
    _graphs,
)
from predictor_width_aware_cascade_calibration import _fixed_budget
from summation_graph_predictor import round_nonnegative_fraction_to_fp32


EXPERIMENT_ID = "wide_range_fixed_k8_beam_v2"
HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results" / EXPERIMENT_ID
PREREGISTRATION = RESULT_ROOT / f"{EXPERIMENT_ID}_preregistration.json"
OUTPUT_DIRECTORY = RESULT_ROOT / "heldout"
V1_PREREGISTRATION = (
    HERE
    / "results"
    / "wide_range_energy_beam_v1"
    / "wide_range_energy_beam_v1_preregistration.json"
)
EXPECTED_V1_PREREGISTRATION_SHA256 = (
    "927000f1f3d1fc9f2e9408d9bf3ee7c2782cb22905e78c8ec141f45682770634"
)
EXPECTED_WIDTHS = (256, 512, 1024)
EXPECTED_GROUPS_PER_WIDTH = 64
EXPECTED_GRAPHS_PER_FAMILY = 32
EXPECTED_TRAINING_BUDGET = 32
EXPECTED_FIXED_BUDGET = 8
EXPECTED_SHORTLIST = 4
EXPECTED_BEAM_WIDTH = 3
EXPECTED_GRAPH_GROUP_OFFSET = 1000
EXPECTED_TRAINING_SEEDS = (22260821, 22260822, 22260823, 22260824)
EXPECTED_BOOTSTRAP_RESAMPLES = 20_000
EXPECTED_BOOTSTRAP_SEED = 20260823
EXPECTED_SEVERE_REGRET_THRESHOLD = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _derived_seed(width: int, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"{EXPERIMENT_ID}|input|{width}|{index}".encode()
        ).digest()[:4],
        "big",
    ) & 0x7FFFFFFF


def _load_and_validate_preregistration(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    predictor = config["predictor"]
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_not_opened",
        "widths": list(EXPECTED_WIDTHS),
        "groups_per_width": EXPECTED_GROUPS_PER_WIDTH,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen preregistration mismatch: {key}")
    predictor_expected = {
        "training_input_seeds": list(EXPECTED_TRAINING_SEEDS),
        "training_graphs_per_family": EXPECTED_GRAPHS_PER_FAMILY,
        "training_root_band_budget": EXPECTED_TRAINING_BUDGET,
        "root_band_budget": EXPECTED_FIXED_BUDGET,
        "shortlist_size": EXPECTED_SHORTLIST,
        "beam_width": EXPECTED_BEAM_WIDTH,
    }
    for key, value in predictor_expected.items():
        if predictor.get(key) != value:
            raise ValueError(f"frozen predictor mismatch: {key}")
    if config["graphs"].get("evaluation_group_index_offset") != (
        EXPECTED_GRAPH_GROUP_OFFSET
    ):
        raise ValueError("frozen graph namespace mismatch")
    graph_expected = {
        "families": ["contiguous", "pair_merge"],
        "count_per_family": EXPECTED_GRAPHS_PER_FAMILY,
    }
    for key, value in graph_expected.items():
        if config["graphs"].get(key) != value:
            raise ValueError(f"frozen graph protocol mismatch: {key}")
    uncertainty_expected = {
        "resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
    }
    for key, value in uncertainty_expected.items():
        if config["uncertainty"].get(key) != value:
            raise ValueError(f"frozen uncertainty mismatch: {key}")
    if config["metrics"].get("severe_regret_threshold") != (
        EXPECTED_SEVERE_REGRET_THRESHOLD
    ):
        raise ValueError("frozen severe-regret threshold mismatch")

    seed_map = config["heldout_seeds"]
    all_heldout: set[int] = set()
    for width in EXPECTED_WIDTHS:
        seeds = seed_map[str(width)]
        if len(seeds) != EXPECTED_GROUPS_PER_WIDTH or len(set(seeds)) != len(seeds):
            raise ValueError(f"invalid frozen seeds for width {width}")
        derived = [
            _derived_seed(width, index)
            for index in range(EXPECTED_GROUPS_PER_WIDTH)
        ]
        if seeds != derived:
            raise ValueError(f"frozen seed policy mismatch for width {width}")
        all_heldout.update(seeds)
    if len(all_heldout) != len(EXPECTED_WIDTHS) * EXPECTED_GROUPS_PER_WIDTH:
        raise ValueError("held-out seeds collide across width strata")
    if all_heldout & set(range(22260821, 22260849)):
        raise ValueError("held-out seeds overlap the recorded calibration range")

    if _sha256(V1_PREREGISTRATION) != EXPECTED_V1_PREREGISTRATION_SHA256:
        raise ValueError("v1 preregistration no longer matches the recorded planning source")
    with V1_PREREGISTRATION.open(encoding="utf-8") as handle:
        v1 = json.load(handle)
    v1_seeds = {
        seed
        for width in EXPECTED_WIDTHS
        for seed in v1["heldout_seeds"][str(width)]
    }
    if all_heldout & v1_seeds:
        raise ValueError("v2 held-out seeds overlap v1 held-out inputs")
    return config


def _reserve_output_directory(path: Path) -> None:
    """Atomically reserve the one-shot artifact boundary."""
    path.mkdir(parents=False, exist_ok=False)


def _git_state(repo: Path) -> tuple[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("held-out opening requires a clean Git worktree")
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
    return commit, tree


def _json_dump(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _fraction_text(value) -> str:
    return f"{value.numerator}/{value.denominator}"


def _stored_leaf_bits(values) -> list[int]:
    bits = []
    for value in values:
        rounded = round_nonnegative_fraction_to_fp32(value)
        if rounded.value != value:
            raise AssertionError("held-out generator emitted a non-FP32 leaf")
        bits.append(rounded.bits)
    return bits


def _bits_hash(bits: list[int]) -> str:
    payload = b"".join(struct.pack(">I", value) for value in bits)
    return hashlib.sha256(payload).hexdigest()


def _graph_hash(graph) -> str:
    payload = json.dumps(
        {
            "leaf_count": graph.leaf_count,
            "root": graph.root,
            "nodes": [[node.left, node.right] for node in graph.nodes],
        },
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _global_regret(target: list[float], selected: int) -> float:
    best = min(target)
    worst = max(target)
    return (target[selected] - best) / (worst - best) if worst > best else 0.0


def _stable_min(indices, values) -> int:
    return min(indices, key=lambda index: (values[index], index))


def _percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    rows: list[dict],
    field,
    resamples: int,
    seed: int,
    *,
    stratify: bool,
) -> tuple[float, float]:
    if not rows or resamples <= 0:
        raise ValueError("bootstrap needs rows and positive resamples")
    rng = random.Random(seed)
    strata: dict[int, list[dict]] = {}
    for row in rows:
        strata.setdefault(int(row["width"]), []).append(row)
    draws = []
    for _ in range(resamples):
        if stratify:
            sample = [
                rng.choice(group)
                for width in sorted(strata)
                for group in [strata[width]]
                for _ in range(len(group))
            ]
        else:
            sample = [rng.choice(rows) for _ in rows]
        draws.append(mean(field(row) for row in sample))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _mean_defined(rows: list[dict], key: str) -> tuple[float | None, int]:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return (mean(values), len(values)) if values else (None, 0)


def _metric_block(rows: list[dict], resamples: int, seed: int) -> dict:
    improvements = [
        row["q_regret"] - row["beam_regret"]
        for row in rows
    ]
    primary = mean(improvements)
    primary_ci = _bootstrap_interval(
        rows,
        lambda row: row["q_regret"] - row["beam_regret"],
        resamples,
        seed,
        stratify=len({row["width"] for row in rows}) > 1,
    )
    beam_hit = mean(row["beam_best_hit"] for row in rows)
    beam_hit_ci = _bootstrap_interval(
        rows,
        lambda row: row["beam_best_hit"],
        resamples,
        seed + 1,
        stratify=len({row["width"] for row in rows}) > 1,
    )
    hit_difference_ci = _bootstrap_interval(
        rows,
        lambda row: row["beam_best_hit"] - row["q_best_hit"],
        resamples,
        seed + 2,
        stratify=len({row["width"] for row in rows}) > 1,
    )
    pairwise, pairwise_defined = _mean_defined(rows, "beam_pairwise_accuracy")
    rho, rho_defined = _mean_defined(rows, "beam_rho")
    beam_regrets = [float(row["beam_regret"]) for row in rows]
    q_regrets = [float(row["q_regret"]) for row in rows]
    return {
        "group_count": len(rows),
        "primary_fixed_q_minus_beam_regret": primary,
        "primary_95_ci": list(primary_ci),
        "positive_evidence": primary_ci[0] > 0.0,
        "beam_best_hit": beam_hit,
        "beam_best_hit_95_ci": list(beam_hit_ci),
        "beam_regret": mean(beam_regrets),
        "beam_regret_p90": _percentile(beam_regrets, 0.90),
        "beam_severe_regret_rate": mean(
            value >= EXPECTED_SEVERE_REGRET_THRESHOLD for value in beam_regrets
        ),
        "q_best_hit": mean(row["q_best_hit"] for row in rows),
        "q_regret": mean(q_regrets),
        "q_regret_p90": _percentile(q_regrets, 0.90),
        "q_severe_regret_rate": mean(
            value >= EXPECTED_SEVERE_REGRET_THRESHOLD for value in q_regrets
        ),
        "beam_minus_q_best_hit": mean(
            row["beam_best_hit"] - row["q_best_hit"] for row in rows
        ),
        "beam_minus_q_best_hit_95_ci": list(hit_difference_ci),
        "beam_benefit_rate": mean(value > 0.0 for value in improvements),
        "beam_harm_rate": mean(value < 0.0 for value in improvements),
        "beam_q_tie_rate": mean(value == 0.0 for value in improvements),
        "shortlist_best_tier_coverage": mean(
            row["shortlist_best_tier_coverage"] for row in rows
        ),
        "oracle_best_prevalence": mean(row["best_prevalence"] for row in rows),
        "random_expected_regret": mean(
            row["random_expected_regret"] for row in rows
        ),
        "beam_pairwise_accuracy": pairwise,
        "beam_pairwise_defined_groups": pairwise_defined,
        "beam_spearman_rho": rho,
        "beam_spearman_defined_groups": rho_defined,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty artifact")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model_record(model) -> dict:
    return {
        "schema_version": "1",
        "label": "innovation_shift",
        "feature_dimension": SHADOW_DIMENSION,
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "weights": model.weights.tolist(),
        "train_prevalence": model.train_prevalence.tolist(),
    }


def _score_group(width: int, group_index: int, seed: int, model):
    generated = wide_range_random(width, seed=seed)
    values = generated.values
    bits = _stored_leaf_bits(values)
    group_id = f"wr{width}_v2_g{group_index:03d}"
    graph_group_index = EXPECTED_GRAPH_GROUP_OFFSET + group_index
    input_record = {
        "schema_version": "1",
        "input_group_id": group_id,
        "family": generated.family,
        "width": width,
        "seed": seed,
        "stored_leaf_bits_sha256": _bits_hash(bits),
        "stored_leaf_bits": bits,
    }
    trees = [
        _beam_tree(values, graph, family, EXPECTED_TRAINING_BUDGET)
        for family, graph in _graphs(
            width,
            graph_group_index,
            EXPECTED_GRAPHS_PER_FAMILY,
        )
    ]
    fixed = [_fixed_budget(tree, EXPECTED_FIXED_BUDGET) for tree in trees]
    target = [tree.target for tree in trees]
    q_scores = [item.q_score for item in fixed]
    best = min(target)
    worst = max(target)
    shortlist = _shortlist_indices(q_scores, EXPECTED_SHORTLIST)
    beam_scores = {
        index: _beam_scores(fixed[index].tree, model, EXPECTED_BEAM_WIDTH)[0]
        for index in shortlist
    }
    q_selected = _stable_min(range(len(trees)), q_scores)
    beam_selected = _stable_min(shortlist, beam_scores)
    beam_metrics = _selection_metrics(
        [beam_scores[index] for index in shortlist],
        [target[index] for index in shortlist],
    )
    all_regrets = [
        (value - best) / (worst - best) if worst > best else 0.0
        for value in target
    ]
    group_row = {
        "schema_version": "1",
        "input_group_id": group_id,
        "width": width,
        "seed": seed,
        "tree_count": len(trees),
        "target_unique": len(set(target)),
        "best_prevalence": sum(value == best for value in target) / len(target),
        "shortlist_best_tier_coverage": float(
            any(target[index] == best for index in shortlist)
        ),
        "q_selected_tree": q_selected,
        "q_best_hit": float(target[q_selected] == best),
        "q_regret": _global_regret(target, q_selected),
        "q_rho": _spearman(q_scores, target),
        "beam_selected_tree": beam_selected,
        "beam_best_hit": float(target[beam_selected] == best),
        "beam_regret": _global_regret(target, beam_selected),
        "beam_pairwise_accuracy": beam_metrics.pairwise_accuracy,
        "beam_rho": beam_metrics.rho,
        "random_expected_regret": mean(all_regrets),
    }
    observations = []
    for index, tree in enumerate(trees):
        family_index = index // 2
        graph_seed = (
            CONTIGUOUS_TREE_BASE_SEED
            if tree.family == "contiguous"
            else PAIR_TREE_BASE_SEED
        ) + graph_group_index * 10_000 + family_index
        observations.append(
            {
                "schema_version": "1",
                "input_group_id": group_id,
                "width": width,
                "seed": seed,
                "tree_index": index,
                "graph_family": tree.family,
                "graph_family_index": family_index,
                "graph_seed": graph_seed,
                "graph_sha256": _graph_hash(tree.graph),
                "signed_error_fraction": _fraction_text(tree.signed_error),
                "signed_error_root_ulp": float(tree.signed_error / tree.root_ulp),
                "target_squared_root_ulp": tree.target,
                "oracle_best_tier": int(tree.target == best),
                "fixed_k": fixed[index].node_count,
                "fixed_energy_capture": fixed[index].captured_fraction,
                "fixed_q_score": fixed[index].q_score,
                "shortlisted": int(index in shortlist),
                "beam_score": beam_scores.get(index),
                "q_selected": int(index == q_selected),
                "beam_selected": int(index == beam_selected),
            }
        )
    return input_record, group_row, observations


def main() -> int:
    config = _load_and_validate_preregistration(PREREGISTRATION)
    repo = HERE.parents[2]
    commit, tree = _git_state(repo)
    _reserve_output_directory(OUTPUT_DIRECTORY)
    opened_at = datetime.now(timezone.utc).isoformat()
    opening_manifest = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "status": "opening_started",
        "opened_at_utc": opened_at,
        "git_commit": commit,
        "git_tree": tree,
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "planning_source_v1_preregistration_sha256": (
            EXPECTED_V1_PREREGISTRATION_SHA256
        ),
        "frozen_widths": list(EXPECTED_WIDTHS),
        "frozen_seeds": {
            str(width): config["heldout_seeds"][str(width)]
            for width in EXPECTED_WIDTHS
        },
    }
    _json_dump(OUTPUT_DIRECTORY / "opening_manifest.json", opening_manifest)

    print("OPENING FROZEN V2 HELD-OUT — score and protocol are immutable", flush=True)
    print(
        f"commit={commit} prereg_sha256={opening_manifest['preregistration_sha256']}",
        flush=True,
    )
    train_groups = _generate_width(
        256,
        EXPECTED_TRAINING_SEEDS,
        EXPECTED_GRAPHS_PER_FAMILY,
        EXPECTED_TRAINING_BUDGET,
    )
    model = _fit_probe(
        [
            sample
            for group in train_groups
            for tree in group
            for sample in tree.transitions
        ],
        SHADOW_DIMENSION,
        label="innovation_shift",
    )
    _json_dump(OUTPUT_DIRECTORY / "calibration_model.json", _model_record(model))

    inputs = []
    groups = []
    observations = []
    for width in EXPECTED_WIDTHS:
        seeds = config["heldout_seeds"][str(width)]
        for group_index, seed in enumerate(seeds):
            input_record, group_row, graph_rows = _score_group(
                width,
                group_index,
                seed,
                model,
            )
            inputs.append(input_record)
            groups.append(group_row)
            observations.extend(graph_rows)
            print(
                f"width={width} group={group_index + 1:02d}/{len(seeds)} "
                f"beam_hit={group_row['beam_best_hit']:.0f} "
                f"q_hit={group_row['q_best_hit']:.0f} "
                f"beam_regret={group_row['beam_regret']:.3f} "
                f"q_regret={group_row['q_regret']:.3f}",
                flush=True,
            )

    input_path = OUTPUT_DIRECTORY / "input_groups.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for record in inputs:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    _write_csv(OUTPUT_DIRECTORY / "group_metrics.csv", groups)
    _write_csv(OUTPUT_DIRECTORY / "graph_observations.csv", observations)

    uncertainty = config["uncertainty"]
    resamples = int(uncertainty["resamples"])
    bootstrap_seed = int(uncertainty["bootstrap_seed"])
    summary = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "opened_at_utc": opened_at,
        "overall": _metric_block(groups, resamples, bootstrap_seed),
        "by_width": {
            str(width): _metric_block(
                [row for row in groups if row["width"] == width],
                resamples,
                bootstrap_seed + width,
            )
            for width in EXPECTED_WIDTHS
        },
    }
    _json_dump(OUTPUT_DIRECTORY / "metric_summary.json", summary)
    artifact_names = [
        "opening_manifest.json",
        "calibration_model.json",
        "input_groups.jsonl",
        "group_metrics.csv",
        "graph_observations.csv",
        "metric_summary.json",
    ]
    metadata = {
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
    }
    _json_dump(OUTPUT_DIRECTORY / "metadata.json", metadata)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
