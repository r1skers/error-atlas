"""Calibration-only diagnostic for graph-sensitive exact-error variation.

This runner intentionally does *not* compute predictor features or rank correlations.  It
asks only whether irregular stored-FP32 inputs from each calibration family produce more
than one exact forward-error target across a fixed sample of random reduction trees.

Inputs are generated without consulting oracle output.  The diagnostic reports every
predeclared seed; it does not search until a graph-sensitive case appears.
"""

from __future__ import annotations

from fractions import Fraction

from predictor_calibration_inputs import calibration_input_families
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import predict_fp32_tree_error


WIDTH = 256
INPUT_SEEDS = (20260818, 20260819, 20260820, 20260821)
RANDOM_GRAPH_COUNT = 16
TREE_BASE_SEED = 31000000


def _graphs(width: int, *, input_index: int):
    for graph_index in range(RANDOM_GRAPH_COUNT):
        seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
        if graph_index % 2 == 0:
            yield random_contiguous_split_graph(width, seed=seed)
        else:
            yield random_pair_merge_graph(width, seed=seed)


def _target(values: tuple[Fraction, ...], graph) -> Fraction:
    oracle = predict_fp32_tree_error(values, graph)
    return abs(oracle.signed_error)


def main() -> int:
    print("Graph-sensitive target variation diagnostic")
    print("CALIBRATION ONLY — no predictor metrics; no held-out evidence")
    print(
        f"width={WIDTH} input_seeds={len(INPUT_SEEDS)} "
        f"random_graphs_per_input={RANDOM_GRAPH_COUNT} anchors=excluded"
    )
    print()

    input_index = 0
    sensitive_count = 0
    total_count = 0

    for base_seed in INPUT_SEEDS:
        for generated in calibration_input_families(WIDTH, seed=base_seed):
            targets = [
                _target(generated.values, graph)
                for graph in _graphs(WIDTH, input_index=input_index)
            ]
            unique_targets = sorted(set(targets))
            graph_sensitive = len(unique_targets) > 1
            sensitive_count += int(graph_sensitive)
            total_count += 1

            minimum = float(min(targets))
            maximum = float(max(targets))
            print(
                f"family={generated.family:<24} seed={generated.seed:<10d} "
                f"unique_targets={len(unique_targets):<2d} "
                f"graph_sensitive={'yes' if graph_sensitive else 'no ':<3} "
                f"min_abs_error={minimum:.9g} max_abs_error={maximum:.9g}"
            )
            input_index += 1

    print()
    print(f"graph_sensitive_inputs={sensitive_count}/{total_count}")
    print("No input is selected or rejected by this script; all predeclared rows are reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
