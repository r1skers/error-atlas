"""Confirm one preregistered topology challenge without candidate execution.

The first phase independently recomputes the depth-margin proxy from frozen
stored leaves and raw graph edges.  It does not call the exact FP32 rounding
oracle.  Only after the proxy invariants match the preregistration does the
second phase obtain exact semantic graph labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from summation_graph_predictor import (
    AdditionNode,
    BinaryReductionGraph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


EXPERIMENT_ID = "depth_margin_topology_challenge_v1"
PREREGISTRATION_SHA256 = (
    "63fbe76d7af9f5df128c1fdfade6ca49fc1d4e26c472ea9914888a3ee28f9949"
)
EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results" / EXPERIMENT_ID
PREREGISTRATION_PATH = RESULTS_DIR / f"{EXPERIMENT_ID}_preregistration.json"
OBSERVATION_PATH = RESULTS_DIR / f"{EXPERIMENT_ID}_oracle_observations.csv"
METADATA_PATH = RESULTS_DIR / f"{EXPERIMENT_ID}_metadata.json"
OBSERVATION_COLUMNS = (
    "experiment_id",
    "preregistration_sha256",
    "graph_id",
    "leaf_depths_json",
    "exact_stored_leaf_sum_fraction",
    "margin_fraction",
    "depth_exposure_D_fraction",
    "risk_score_R_fraction",
    "proxy_matched_preregistration",
    "correctly_rounded_target_bits",
    "oracle_output_bits",
    "oracle_signed_graph_error_fraction",
    "oracle_local_error_sum_fraction",
    "correct_rounding_failure",
    "hand_conjecture_matched_oracle",
)


@dataclass(frozen=True)
class ProxyRow:
    """One graph's independently recomputed pre-run quantities."""

    graph_id: str
    leaf_depths: tuple[int, ...]
    depth_exposure: Fraction
    risk_score: Fraction


@dataclass(frozen=True)
class ProxyExecution:
    """Shared input geometry plus both graph-only proxy rows."""

    leaves: tuple[Fraction, ...]
    exact_sum: Fraction
    margin: Fraction
    rows: tuple[ProxyRow, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_preregistration() -> dict[str, object]:
    actual_hash = _sha256(PREREGISTRATION_PATH)
    if actual_hash != PREREGISTRATION_SHA256:
        raise ValueError(
            "preregistration hash changed before execution: "
            f"expected {PREREGISTRATION_SHA256}, got {actual_hash}"
        )
    record = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    if record["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("preregistration has an unexpected experiment_id")
    if record["status"] != "preregistered_not_executed":
        raise ValueError("preregistration is not in its frozen pre-run state")
    return record


def _independent_leaf_depths(
    graph_record: dict[str, object],
    leaf_count: int,
) -> tuple[int, ...]:
    """Compute depths from raw edges without semantic-oracle graph helpers."""
    raw_nodes = graph_record["nodes"]
    if not isinstance(raw_nodes, list) or len(raw_nodes) != leaf_count - 1:
        raise ValueError("raw graph is not a full binary tree")

    value_count = leaf_count + len(raw_nodes)
    root = graph_record["root"]
    if root != value_count - 1:
        raise ValueError("raw graph root is not the final topological value")

    parent_counts = [0] * value_count
    children_by_node: dict[int, tuple[int, int]] = {}
    for offset, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise TypeError("every raw node must be an object")
        node_index = leaf_count + offset
        if raw_node["index"] != node_index:
            raise ValueError("raw node indices are not topologically contiguous")
        left = raw_node["left"]
        right = raw_node["right"]
        if not isinstance(left, int) or not isinstance(right, int):
            raise TypeError("raw node references must be integers")
        if not 0 <= left < node_index or not 0 <= right < node_index:
            raise ValueError("raw node references are not topological")
        parent_counts[left] += 1
        parent_counts[right] += 1
        children_by_node[node_index] = (left, right)

    for value_index, parent_count in enumerate(parent_counts):
        expected = 0 if value_index == root else 1
        if parent_count != expected:
            raise ValueError("raw graph is not one connected reduction tree")

    depths: list[int | None] = [None] * value_count
    depths[root] = 0
    for node_index in range(value_count - 1, leaf_count - 1, -1):
        parent_depth = depths[node_index]
        if parent_depth is None:
            raise ValueError("raw graph contains a node unreachable from root")
        for child in children_by_node[node_index]:
            if depths[child] is not None:
                raise ValueError("raw graph child has more than one path to root")
            depths[child] = parent_depth + 1

    leaf_depths = depths[:leaf_count]
    if any(depth is None for depth in leaf_depths):
        raise ValueError("raw graph contains an unreachable leaf")
    return tuple(int(depth) for depth in leaf_depths)


def recompute_proxy(record: dict[str, object]) -> ProxyExecution:
    """Recompute D, fixed near-one M, and R without FP32 oracle calls."""
    geometry = record["fixed_geometry"]
    proxy_record = record["proxy"]
    if not isinstance(geometry, dict) or not isinstance(proxy_record, dict):
        raise TypeError("preregistration geometry and proxy must be objects")

    leaves = tuple(Fraction(value) for value in geometry["stored_leaves_fraction"])
    exact_sum = sum(leaves, start=Fraction(0))
    if exact_sum != Fraction(geometry["exact_stored_leaf_sum_fraction"]):
        raise ValueError("stored leaves do not reproduce frozen S_leaf")

    q = Fraction(geometry["q_fraction"])
    epsilon32 = Fraction(geometry["epsilon32_fraction"])
    midpoint = Fraction(1) + Fraction(geometry["midpoint_offset_in_q"]) * q
    margin = abs(exact_sum - midpoint)
    if margin != Fraction(geometry["margin_fraction"]):
        raise ValueError("independent near-one margin differs from preregistration")
    if margin == 0:
        raise ValueError("this finite-score challenge requires nonzero margin")

    expected_depths = tuple(proxy_record["shared_leaf_depths"])
    expected_depth_exposure = Fraction(proxy_record["shared_D_fraction"])
    expected_risk_score = Fraction(proxy_record["shared_R_fraction"])
    raw_graphs = record["graphs"]
    if not isinstance(raw_graphs, list):
        raise TypeError("preregistered graphs must be a list")

    rows: list[ProxyRow] = []
    for raw_graph in raw_graphs:
        if not isinstance(raw_graph, dict):
            raise TypeError("every preregistered graph must be an object")
        depths = _independent_leaf_depths(raw_graph, len(leaves))
        depth_exposure = sum(
            (depth * abs(value) for depth, value in zip(depths, leaves, strict=True)),
            start=Fraction(0),
        )
        risk_score = epsilon32 * depth_exposure / margin
        if depths != expected_depths:
            raise ValueError(f"{raw_graph['graph_id']} leaf depths changed")
        if depth_exposure != expected_depth_exposure:
            raise ValueError(f"{raw_graph['graph_id']} D differs from frozen value")
        if risk_score != expected_risk_score:
            raise ValueError(f"{raw_graph['graph_id']} R differs from frozen value")
        rows.append(
            ProxyRow(
                graph_id=str(raw_graph["graph_id"]),
                leaf_depths=depths,
                depth_exposure=depth_exposure,
                risk_score=risk_score,
            )
        )

    if len({row.risk_score for row in rows}) != 1:
        raise ValueError("registered topology challenge did not reproduce a proxy tie")
    return ProxyExecution(
        leaves=leaves,
        exact_sum=exact_sum,
        margin=margin,
        rows=tuple(rows),
    )


def _semantic_graph(
    raw_graph: dict[str, object], leaf_count: int
) -> BinaryReductionGraph:
    raw_nodes = raw_graph["nodes"]
    if not isinstance(raw_nodes, list):
        raise TypeError("preregistered graph nodes must be a list")
    return BinaryReductionGraph(
        name=str(raw_graph["graph_id"]),
        leaf_count=leaf_count,
        nodes=tuple(
            AdditionNode(left=int(node["left"]), right=int(node["right"]))
            for node in raw_nodes
        ),
        root=int(raw_graph["root"]),
    )


def obtain_oracle_rows(
    record: dict[str, object],
    proxy_execution: ProxyExecution,
) -> tuple[dict[str, object], ...]:
    """Obtain exact labels only after independent proxy checks have passed."""
    target = round_nonnegative_fraction_to_fp32(proxy_execution.exact_sum)
    frozen_target = record["fixed_geometry"]["correctly_rounded_target_bits"]
    if target.bits_hex != frozen_target:
        raise ValueError("exact oracle target differs from preregistration")

    proxy_by_graph = {row.graph_id: row for row in proxy_execution.rows}
    rows: list[dict[str, object]] = []
    raw_graphs = record["graphs"]
    for raw_graph in raw_graphs:
        graph_id = str(raw_graph["graph_id"])
        proxy_row = proxy_by_graph[graph_id]
        graph = _semantic_graph(raw_graph, len(proxy_execution.leaves))
        prediction = predict_fp32_tree_error(proxy_execution.leaves, graph)
        failure = int(prediction.predicted_sum_bits != target.bits_hex)

        hand = raw_graph["hand_conjecture"]
        node_by_index = {node.node_index: node for node in prediction.node_predictions}
        hand_match = (
            node_by_index[4].rounded_sum == Fraction(hand["node_4_rounded_fraction"])
            and node_by_index[5].rounded_sum
            == Fraction(hand["node_5_rounded_fraction"])
            and prediction.predicted_sum == Fraction(hand["root_rounded_fraction"])
            and prediction.predicted_sum_bits == hand["root_bits"]
            and prediction.signed_error == Fraction(hand["signed_graph_error_fraction"])
            and failure == hand["correct_rounding_failure"]
        )
        if not hand_match:
            raise ValueError(f"exact oracle disagrees with {graph_id} hand conjecture")

        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "graph_id": graph_id,
                "leaf_depths_json": json.dumps(proxy_row.leaf_depths),
                "exact_stored_leaf_sum_fraction": str(proxy_execution.exact_sum),
                "margin_fraction": str(proxy_execution.margin),
                "depth_exposure_D_fraction": str(proxy_row.depth_exposure),
                "risk_score_R_fraction": str(proxy_row.risk_score),
                "proxy_matched_preregistration": True,
                "correctly_rounded_target_bits": target.bits_hex,
                "oracle_output_bits": prediction.predicted_sum_bits,
                "oracle_signed_graph_error_fraction": str(prediction.signed_error),
                "oracle_local_error_sum_fraction": str(prediction.local_error_sum),
                "correct_rounding_failure": failure,
                "hand_conjecture_matched_oracle": hand_match,
            }
        )

    return tuple(rows)


def write_artifacts(rows: tuple[dict[str, object], ...]) -> tuple[Path, Path]:
    """Write one-shot oracle observations and their provenance."""
    for path in (OBSERVATION_PATH, METADATA_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    with OBSERVATION_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OBSERVATION_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    risk_scores = {row["risk_score_R_fraction"] for row in rows}
    labels = {row["correct_rounding_failure"] for row in rows}
    informative = labels == {0, 1}
    pair_outcome = "tie" if informative and len(risk_scores) == 1 else "not_confirmed"
    strong_hypothesis_falsified = pair_outcome == "tie"
    if not strong_hypothesis_falsified:
        raise ValueError("registered tied informative counterexample was not confirmed")

    source_paths = {
        "depth_margin_topology_challenge_v1_runner.py": Path(__file__).resolve(),
        "summation_graph_predictor.py": (
            EXPERIMENTS_DIR / "summation_graph_predictor.py"
        ),
    }
    metadata = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_status": "confirmed_adversarial_counterexample",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "python topics/softmax/experiments/"
            "depth_margin_topology_challenge_v1_runner.py"
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "preregistration": {
            "path": PREREGISTRATION_PATH.name,
            "sha256": PREREGISTRATION_SHA256,
        },
        "candidate_executed": False,
        "observation_row_count": len(rows),
        "all_proxy_values_matched_preregistration": all(
            bool(row["proxy_matched_preregistration"]) for row in rows
        ),
        "all_hand_conjectures_matched_oracle": all(
            bool(row["hand_conjecture_matched_oracle"]) for row in rows
        ),
        "pair_is_informative": informative,
        "pair_outcome": pair_outcome,
        "strong_hypothesis_falsified": strong_hypothesis_falsified,
        "conclusion": (
            "Equal depth-margin scores do not universally rank binary32 "
            "correct-rounding failure when sibling grouping changes."
        ),
        "sources": {name: _sha256(path) for name, path in source_paths.items()},
        "artifacts": {
            OBSERVATION_PATH.name: _sha256(OBSERVATION_PATH),
        },
    }
    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return OBSERVATION_PATH, METADATA_PATH


def main() -> None:
    preregistration = _load_frozen_preregistration()
    proxy_execution = recompute_proxy(preregistration)
    rows = obtain_oracle_rows(preregistration, proxy_execution)
    paths = write_artifacts(rows)
    for row in rows:
        print(
            f"{row['graph_id']}: R={row['risk_score_R_fraction']} "
            f"output={row['oracle_output_bits']} "
            f"E={row['oracle_signed_graph_error_fraction']} "
            f"F={row['correct_rounding_failure']}"
        )
    print("pair_outcome=tie; strong_hypothesis_falsified=True")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
