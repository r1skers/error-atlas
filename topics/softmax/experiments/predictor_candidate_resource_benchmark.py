"""Resource-only benchmark for candidate-graph budgets.

This is infrastructure, not predictor-validation evidence.  For one deterministic
stored-FP32 input per (width, candidate_count) combination it measures the wall-clock
and process-memory cost of evaluating a candidate set containing the two canonical
anchors plus reproducible random contiguous-split and random pair-merge trees.

The benchmark deliberately does not report graph error, failure labels, predictor scores,
correlation, AUROC, prevalence, or any other research metric.  Each combination runs in a
fresh subprocess so Linux ``ru_maxrss`` is scoped to that combination.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import (
    balanced_reduction_graph,
    predict_fp32_tree_error,
    sequential_reduction_graph,
)


DEFAULT_WIDTHS = (4096, 16384, 65536)
DEFAULT_CANDIDATE_COUNTS = (8, 16, 32, 64)
DEFAULT_TIMEOUT_SECONDS = 300.0
BASE_SEED = 20260818


def _stored_values(width: int) -> tuple[Fraction, ...]:
    """Deterministic exactly stored FP32 workload used only for resource sizing."""
    if width <= 0:
        raise ValueError("width must be positive")
    exponents = (0, -4, -8, -12, -16, -20, -24, -28)
    return tuple(
        Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2 ** (-exponent))
        for exponent in (exponents[index % len(exponents)] for index in range(width))
    )


def _max_rss_mib() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _vm_swap_mib() -> float | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmSwap:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0
    return None


def _read_meminfo_mib(field: str) -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    prefix = f"{field}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0
    return None


def _candidate_kind(index: int) -> str:
    """Return the frozen benchmark allocation kind for candidate index.

    Candidate 0 is sequential, candidate 1 is balanced, and the remaining candidates
    alternate between the two random generator families.  This is only a cost benchmark;
    the final validation allocation remains a protocol decision.
    """
    if index == 0:
        return "sequential"
    if index == 1:
        return "balanced"
    return "contiguous" if (index - 2) % 2 == 0 else "pair_merge"


def _make_graph(width: int, candidate_index: int):
    kind = _candidate_kind(candidate_index)
    if kind == "sequential":
        return sequential_reduction_graph(width)
    if kind == "balanced":
        return balanced_reduction_graph(width)

    seed = BASE_SEED + candidate_index
    if kind == "contiguous":
        return random_contiguous_split_graph(width, seed=seed)
    if kind == "pair_merge":
        return random_pair_merge_graph(width, seed=seed)
    raise AssertionError(f"unknown candidate kind: {kind}")


def _worker(width: int, candidate_count: int) -> dict[str, float | int | None]:
    if width <= 0:
        raise ValueError("width must be positive")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2 to include both anchors")

    started = time.perf_counter()
    values_started = time.perf_counter()
    values = _stored_values(width)
    input_generation_s = time.perf_counter() - values_started

    graph_generation_s = 0.0
    oracle_s = 0.0
    contiguous_count = 0
    pair_merge_count = 0

    for candidate_index in range(candidate_count):
        kind = _candidate_kind(candidate_index)
        graph_started = time.perf_counter()
        graph = _make_graph(width, candidate_index)
        graph_generation_s += time.perf_counter() - graph_started

        oracle_started = time.perf_counter()
        prediction = predict_fp32_tree_error(values, graph)
        oracle_s += time.perf_counter() - oracle_started

        if kind == "contiguous":
            contiguous_count += 1
        elif kind == "pair_merge":
            pair_merge_count += 1

        del prediction, graph
        gc.collect()

    return {
        "width": width,
        "candidate_count": candidate_count,
        "anchor_count": 2,
        "random_contiguous_count": contiguous_count,
        "random_pair_merge_count": pair_merge_count,
        "input_generation_s": input_generation_s,
        "graph_generation_s": graph_generation_s,
        "oracle_s": oracle_s,
        "group_total_s": time.perf_counter() - started,
        "peak_rss_mib": _max_rss_mib(),
        "vm_swap_mib": _vm_swap_mib(),
    }


def _run_one(
    width: int,
    candidate_count: int,
    timeout_seconds: float,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-width",
        str(width),
        "--worker-candidates",
        str(candidate_count),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "width": width,
            "candidate_count": candidate_count,
            "status": "timeout",
            "wall_s": time.perf_counter() - started,
        }

    if completed.returncode != 0:
        return {
            "width": width,
            "candidate_count": candidate_count,
            "status": f"worker_exit_{completed.returncode}",
            "wall_s": time.perf_counter() - started,
            "stderr": completed.stderr.strip(),
        }

    payload = json.loads(completed.stdout)
    payload["status"] = "ok"
    return payload


def _print_header() -> None:
    print("Candidate-tree resource benchmark (infrastructure only; no research metrics)")
    print(f"python={platform.python_version()} platform={platform.platform()}")
    print(f"logical_cpus={os.cpu_count()}")
    mem_total = _read_meminfo_mib("MemTotal")
    swap_total = _read_meminfo_mib("SwapTotal")
    if mem_total is not None:
        print(f"mem_total_mib={mem_total:.1f}")
    if swap_total is not None:
        print(f"swap_total_mib={swap_total:.1f}")
    print()
    print(
        "width\tK\tstatus\tgraph_gen_s\toracle_s\tgroup_total_s\t"
        "peak_rss_mib\tvm_swap_mib"
    )


def _print_row(result: dict[str, object]) -> None:
    if result["status"] != "ok":
        print(
            f"{result['width']}\t{result['candidate_count']}\t{result['status']}\t"
            f"-\t-\t{float(result.get('wall_s', 0.0)):.3f}\t-\t-"
        )
        if result.get("stderr"):
            print(f"worker stderr: {result['stderr']}", file=sys.stderr)
        return

    swap = result.get("vm_swap_mib")
    swap_text = "n/a" if swap is None else f"{float(swap):.1f}"
    print(
        f"{int(result['width'])}\t{int(result['candidate_count'])}\tok\t"
        f"{float(result['graph_generation_s']):.4f}\t"
        f"{float(result['oracle_s']):.4f}\t"
        f"{float(result['group_total_s']):.4f}\t"
        f"{float(result['peak_rss_mib']):.1f}\t{swap_text}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=list(DEFAULT_WIDTHS),
        help="widths to benchmark (default: 4096 16384 65536)",
    )
    parser.add_argument(
        "--candidate-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_CANDIDATE_COUNTS),
        help="total candidate counts K, including two anchors (default: 8 16 32 64)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-combination worker timeout",
    )
    parser.add_argument("--worker-width", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-candidates", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    worker_mode = args.worker_width is not None or args.worker_candidates is not None
    if worker_mode:
        if args.worker_width is None or args.worker_candidates is None:
            raise SystemExit("worker mode requires both hidden worker arguments")
        print(
            json.dumps(
                _worker(args.worker_width, args.worker_candidates),
                sort_keys=True,
            )
        )
        return 0

    if any(width <= 0 for width in args.widths):
        raise SystemExit("all widths must be positive")
    if any(count < 2 for count in args.candidate_counts):
        raise SystemExit("all candidate counts must be at least 2")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    _print_header()
    for width in args.widths:
        for candidate_count in args.candidate_counts:
            result = _run_one(width, candidate_count, args.timeout_seconds)
            _print_row(result)
            if result["status"] != "ok":
                print("stopping this width after the first non-successful candidate count")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
