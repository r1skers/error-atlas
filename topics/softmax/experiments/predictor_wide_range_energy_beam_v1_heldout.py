"""Execute the frozen wide_range_energy_beam_v1 held-out validation exactly once."""
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
from predictor_width_aware_cascade_calibration import (
    _energy_budget,
    _fixed_budget,
)
from summation_graph_predictor import round_nonnegative_fraction_to_fp32


EXPERIMENT_ID = "wide_range_energy_beam_v1"
HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results" / EXPERIMENT_ID
PREREGISTRATION = RESULT_ROOT / f"{EXPERIMENT_ID}_preregistration.json"
OUTPUT_DIRECTORY = RESULT_ROOT / "heldout"
EXPECTED_WIDTHS = (256, 512, 1024)
EXPECTED_GROUPS_PER_WIDTH = 32
EXPECTED_GRAPHS_PER_FAMILY = 32
EXPECTED_ENERGY_MASS = 0.8
EXPECTED_SHORTLIST = 4
EXPECTED_BEAM_WIDTH = 3
EXPECTED_MAX_BUDGET = 32
EXPECTED_FIXED_BUDGET = 8
EXPECTED_TRAINING_SEEDS = (22260821, 22260822, 22260823, 22260824)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "maximum_root_band_budget": EXPECTED_MAX_BUDGET,
        "energy_mass": EXPECTED_ENERGY_MASS,
        "shortlist_size": EXPECTED_SHORTLIST,
        "beam_width": EXPECTED_BEAM_WIDTH,
    }
    for key, value in predictor_expected.items():
        if predictor.get(key) != value:
            raise ValueError(f"frozen predictor mismatch: {key}")
    seed_map = config["heldout_seeds"]
    for width in EXPECTED_WIDTHS:
        seeds = seed_map[str(width)]
        if len(seeds) != EXPECTED_GROUPS_PER_WIDTH or len(set(seeds)) != len(seeds):
            raise ValueError(f"invalid frozen seeds for width {width}")
        derived = [
            int.from_bytes(
                hashlib.sha256(
                    f"{EXPERIMENT_ID}|{width}|{index}".encode()
                ).digest()[:4],
                "big",
            )
            & 0x7FFFFFFF
            for index in range(EXPECTED_GROUPS_PER_WIDTH)
        ]
        if seeds != derived:
            raise ValueError(f"frozen seed policy mismatch for width {width}")
    all_heldout = {
        seed for width in EXPECTED_WIDTHS for seed in seed_map[str(width)]
    }
    if all_heldout & set(range(22260821, 22260849)):
        raise ValueError("held-out seeds overlap the recorded calibration range")
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
    primary = mean(row["q_regret"] - row["mass_regret"] for row in rows)
    primary_ci = _bootstrap_interval(
        rows,
        lambda row: row["q_regret"] - row["mass_regret"],
        resamples,
        seed,
        stratify=len({row["width"] for row in rows}) > 1,
    )
    mass_hit = mean(row["mass_best_hit"] for row in rows)
    mass_hit_ci = _bootstrap_interval(
        rows,
        lambda row: row["mass_best_hit"],
        resamples,
        seed + 1,
        stratify=len({row["width"] for row in rows}) > 1,
    )
    pairwise, pairwise_defined = _mean_defined(rows, "mass_pairwise_accuracy")
    rho, rho_defined = _mean_defined(rows, "mass_rho")
    return {
        "group_count": len(rows),
        "primary_q_minus_mass_regret": primary,
        "primary_95_ci": list(primary_ci),
        "positive_evidence": primary_ci[0] > 0.0,
        "mass_best_hit": mass_hit,
        "mass_best_hit_95_ci": list(mass_hit_ci),
        "mass_regret": mean(row["mass_regret"] for row in rows),
        "q_best_hit": mean(row["q_best_hit"] for row in rows),
        "q_regret": mean(row["q_regret"] for row in rows),
        "fixed_best_hit": mean(row["fixed_best_hit"] for row in rows),
        "fixed_regret": mean(row["fixed_regret"] for row in rows),
        "mass_shortlist_coverage": mean(
            row["mass_shortlist_coverage"] for row in rows
        ),
        "oracle_best_prevalence": mean(row["best_prevalence"] for row in rows),
        "random_expected_regret": mean(
            row["random_expected_regret"] for row in rows
        ),
        "mass_pairwise_accuracy": pairwise,
        "mass_pairwise_defined_groups": pairwise_defined,
        "mass_spearman_rho": rho,
        "mass_spearman_defined_groups": rho_defined,
        "mean_energy_node_count": mean(row["mean_energy_k"] for row in rows),
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
    group_id = f"wr{width}_g{group_index:03d}"
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
        _beam_tree(values, graph, family, EXPECTED_MAX_BUDGET)
        for family, graph in _graphs(
            width, group_index, EXPECTED_GRAPHS_PER_FAMILY
        )
    ]
    mass = [
        _energy_budget(tree, EXPECTED_ENERGY_MASS, 4, EXPECTED_MAX_BUDGET)
        for tree in trees
    ]
    fixed = [_fixed_budget(tree, EXPECTED_FIXED_BUDGET) for tree in trees]
    target = [tree.target for tree in trees]
    best = min(target)
    worst = max(target)
    mass_indices = _shortlist_indices(
        [item.q_score for item in mass], EXPECTED_SHORTLIST
    )
    fixed_indices = _shortlist_indices(
        [item.q_score for item in fixed], EXPECTED_SHORTLIST
    )
    mass_beam = {
        index: _beam_scores(mass[index].tree, model, EXPECTED_BEAM_WIDTH)[0]
        for index in mass_indices
    }
    fixed_beam = {
        index: _beam_scores(fixed[index].tree, model, EXPECTED_BEAM_WIDTH)[0]
        for index in fixed_indices
    }
    mass_selected = _stable_min(mass_indices, mass_beam)
    fixed_selected = _stable_min(fixed_indices, fixed_beam)
    q_selected = _stable_min(range(len(trees)), [item.q_score for item in mass])
    mass_subset_target = [target[index] for index in mass_indices]
    mass_metrics = _selection_metrics(
        [mass_beam[index] for index in mass_indices], mass_subset_target
    )
    fixed_metrics = _selection_metrics(
        [fixed_beam[index] for index in fixed_indices],
        [target[index] for index in fixed_indices],
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
        "mean_energy_k": mean(item.node_count for item in mass),
        "mass_shortlist_coverage": float(
            any(target[index] == best for index in mass_indices)
        ),
        "mass_selected_tree": mass_selected,
        "mass_best_hit": float(target[mass_selected] == best),
        "mass_regret": _global_regret(target, mass_selected),
        "mass_pairwise_accuracy": mass_metrics.pairwise_accuracy,
        "mass_rho": mass_metrics.rho,
        "q_selected_tree": q_selected,
        "q_best_hit": float(target[q_selected] == best),
        "q_regret": _global_regret(target, q_selected),
        "q_rho": _spearman([item.q_score for item in mass], target),
        "fixed_selected_tree": fixed_selected,
        "fixed_best_hit": float(target[fixed_selected] == best),
        "fixed_regret": _global_regret(target, fixed_selected),
        "fixed_pairwise_accuracy": fixed_metrics.pairwise_accuracy,
        "fixed_rho": fixed_metrics.rho,
        "random_expected_regret": mean(all_regrets),
    }
    observations = []
    for index, tree in enumerate(trees):
        family_index = index // 2
        graph_seed = (
            CONTIGUOUS_TREE_BASE_SEED
            if tree.family == "contiguous"
            else PAIR_TREE_BASE_SEED
        ) + group_index * 10_000 + family_index
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
                "energy_k": mass[index].node_count,
                "energy_capture": mass[index].captured_fraction,
                "mass_q_score": mass[index].q_score,
                "mass_shortlisted": int(index in mass_indices),
                "mass_beam_score": mass_beam.get(index),
                "mass_selected": int(index == mass_selected),
                "q_selected": int(index == q_selected),
                "fixed_q_score": fixed[index].q_score,
                "fixed_shortlisted": int(index in fixed_indices),
                "fixed_beam_score": fixed_beam.get(index),
                "fixed_selected": int(index == fixed_selected),
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
        "frozen_widths": list(EXPECTED_WIDTHS),
        "frozen_seeds": {
            str(width): config["heldout_seeds"][str(width)]
            for width in EXPECTED_WIDTHS
        },
    }
    _json_dump(OUTPUT_DIRECTORY / "opening_manifest.json", opening_manifest)

    print("OPENING FROZEN HELD-OUT — score and protocol are now immutable", flush=True)
    print(f"commit={commit} prereg_sha256={opening_manifest['preregistration_sha256']}", flush=True)
    train_groups = _generate_width(
        256,
        EXPECTED_TRAINING_SEEDS,
        EXPECTED_GRAPHS_PER_FAMILY,
        EXPECTED_MAX_BUDGET,
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
                width, group_index, seed, model
            )
            inputs.append(input_record)
            groups.append(group_row)
            observations.extend(graph_rows)
            print(
                f"width={width} group={group_index + 1:02d}/{len(seeds)} "
                f"mass_hit={group_row['mass_best_hit']:.0f} "
                f"q_hit={group_row['q_best_hit']:.0f}",
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
