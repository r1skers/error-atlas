"""Resource-only benchmark for sizing predictor-validation input widths.

This script is infrastructure, not a predictor-validation experiment.  It measures
wall-clock cost and process memory while running the existing exact graph oracle on a
deterministic synthetic input group.  It deliberately does not report graph errors,
failure labels, score values, prevalence, correlation, AUROC, or any other research
metric.

The parent process launches one fresh worker per width so Linux ``ru_maxrss`` is scoped
to that width rather than accumulating across the whole sweep.
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

from summation_graph_predictor import (
    balanced_reduction_graph,
    predict_fp32_tree_error,
    sequential_reduction_graph,
)


DEFAULT_WIDTHS = (256, 1024, 4096, 16384, 65536)
DEFAULT_TIMEOUT_SECONDS = 180.0


def _stored_values(width: int) -> tuple[Fraction, ...]:
    """Build deterministic, exactly representable nonnegative FP32 leaves.

    The exponent cycle is only a workload for resource sizing.  It is not a frozen
    validation distribution and its oracle outputs must not be treated as evidence.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    exponents = (0, -4, -8, -12, -16, -20, -24, -28)
    return tuple(
        Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2 ** (-exponent))
        for exponent in (exponents[index % len(exponents)] for index in range(width))
    )


def _max_rss_mib() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
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


def _worker(width: int) -> dict[str, float | int | None]:
    group_start = time.perf_counter()

    generation_start = time.perf_counter()
    values = _stored_values(width)
    generation_seconds = time.perf_counter() - generation_start

    sequential_start = time.perf_counter()
    sequential_graph = sequential_reduction_graph(width)
    sequential_prediction = predict_fp32_tree_error(values, sequential_graph)
    sequential_seconds = time.perf_counter() - sequential_start
    # Drop graph-sized objects before constructing the next graph.  The benchmark wants
    # the peak required to process one group serially, matching the intended runner
    # strategy rather than holding every graph prediction in memory at once.
    del sequential_prediction, sequential_graph
    gc.collect()

    balanced_start = time.perf_counter()
    balanced_graph = balanced_reduction_graph(width)
    balanced_prediction = predict_fp32_tree_error(values, balanced_graph)
    balanced_seconds = time.perf_counter() - balanced_start
    del balanced_prediction, balanced_graph
    gc.collect()

    total_seconds = time.perf_counter() - group_start
    return {
        "width": width,
        "generation_s": generation_seconds,
        "sequential_oracle_s": sequential_seconds,
        "balanced_oracle_s": balanced_seconds,
        "group_total_s": total_seconds,
        "peak_rss_mib": _max_rss_mib(),
        "vm_swap_mib": _vm_swap_mib(),
    }


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


def _run_one_width(width: int, timeout_seconds: float) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-width",
        str(width),
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
            "status": "timeout",
            "wall_s": time.perf_counter() - started,
        }

    if completed.returncode != 0:
        return {
            "width": width,
            "status": f"worker_exit_{completed.returncode}",
            "wall_s": time.perf_counter() - started,
            "stderr": completed.stderr.strip(),
        }

    payload = json.loads(completed.stdout)
    payload["status"] = "ok"
    return payload


def _print_header() -> None:
    print("Predictor resource benchmark (infrastructure only; no research metrics)")
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
        "width\tstatus\tgeneration_s\tsequential_s\tbalanced_s\t"
        "group_total_s\tpeak_rss_mib\tvm_swap_mib"
    )


def _print_row(result: dict[str, object]) -> None:
    if result["status"] != "ok":
        print(
            f"{result['width']}\t{result['status']}\t-\t-\t-\t"
            f"{float(result.get('wall_s', 0.0)):.3f}\t-\t-"
        )
        if result.get("stderr"):
            print(f"worker stderr: {result['stderr']}", file=sys.stderr)
        return

    swap = result.get("vm_swap_mib")
    swap_text = "n/a" if swap is None else f"{float(swap):.1f}"
    print(
        f"{int(result['width'])}\tok\t"
        f"{float(result['generation_s']):.4f}\t"
        f"{float(result['sequential_oracle_s']):.4f}\t"
        f"{float(result['balanced_oracle_s']):.4f}\t"
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
        help="width strata to benchmark (default: 256 1024 4096 16384 65536)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-width worker timeout; a timeout stops the sweep",
    )
    parser.add_argument("--worker-width", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker_width is not None:
        print(json.dumps(_worker(args.worker_width), sort_keys=True))
        return 0

    if any(width <= 0 for width in args.widths):
        raise SystemExit("all widths must be positive")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    _print_header()
    for width in args.widths:
        result = _run_one_width(width, args.timeout_seconds)
        _print_row(result)
        if result["status"] != "ok":
            print("stopping after the first non-successful width")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
