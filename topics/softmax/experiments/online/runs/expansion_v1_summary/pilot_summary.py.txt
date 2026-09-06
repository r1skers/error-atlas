"""Descriptive cell summaries of a strictly replayed pilot bundle; no statistical CI.

All aggregation uses exact Fractions. Only the serialized summary endpoints are rounded
outward to 50 decimal significant digits; individual exact measurements stay in the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from pathlib import Path

from .pilot_runner import replay


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _value(record: dict) -> Fraction:
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def _mean(intervals: list[tuple[Fraction, Fraction]]) -> tuple[Fraction, Fraction]:
    if not intervals or any(low > high for low, high in intervals):
        raise ValueError("Need nonempty ordered intervals")
    count = len(intervals)
    return (sum((low for low, _ in intervals), Fraction(0)) / count,
            sum((high for _, high in intervals), Fraction(0)) / count)


def _encoded(interval: tuple[Fraction, Fraction]) -> dict:
    endpoints = []
    for value, rounding in zip(interval, (ROUND_FLOOR, ROUND_CEILING)):
        context = Context(prec=50, rounding=rounding)
        endpoints.append(str(context.divide(Decimal(value.numerator), Decimal(value.denominator))))
    return {"low": endpoints[0], "high": endpoints[1]}


def _order(interval) -> str:
    low, high = interval
    return ("negative" if high < 0 else "positive" if low > 0 else
            "equal" if low == high == 0 else "unresolved")


def summarize(plan: dict, inputs: list[dict], measurements: list[dict], pairs: list[dict]) -> dict:
    """One cell/arm at a time; repeated seeds across FMA arms are not extra samples."""
    if plan["schedules"] != ["chain", "balanced"]:
        raise ValueError("This analysis is defined for chain versus balanced only")
    ids = [row["family_id"] for row in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate family IDs")
    measures = {(r["family_id"], r["graph_id"], r["fused"]): r for r in measurements}
    paired = {(r["family_id"], r["fused"]): r for r in pairs}
    if len(measures) != len(measurements) or len(paired) != len(pairs):
        raise ValueError("Duplicate observations")
    cells, exact_means, contrasts = [], {}, []
    for cell_index, cell in enumerate(plan["cells"]):
        if cell["kind"] != "uniform_spread":
            raise ValueError("This analysis requires uniform-spread cells")
        selected = [r for r in inputs if r["cell_index"] == cell_index]
        if (len(selected) != len(cell["seeds"])
                or {r["seed"] for r in selected} != set(cell["seeds"])):
            raise ValueError("Incomplete cell or seed mismatch")
        for fused in plan["fused"]:
            errors = {name: [] for name in ("chain", "balanced")}
            differences = []
            for row in selected:
                for name in errors:
                    result = measures[row["family_id"], f"{name}:{cell['count']}", fused]
                    if result["status"] != "ok":
                        raise ValueError("Cannot aggregate a failed or inapplicable measurement")
                    errors[name].append((_value(result["error_low"]), _value(result["error_high"])))
                pair = paired[row["family_id"], fused]
                if pair["schedule"] != "chain" or pair["baseline"] != "balanced":
                    raise ValueError("Unexpected paired comparison")
                differences.append((_value(pair["difference_low"]), _value(pair["difference_high"])))
            mean = _mean(differences)
            key = (cell["spread"], cell["mass"], fused, cell["count"])
            if key in exact_means:
                raise ValueError("Repeated cell specification; do not silently pool")
            exact_means[key] = mean
            signs = Counter(_order(interval) for interval in differences)
            cells.append({"cell_index": cell_index, "count": cell["count"], "spread": cell["spread"],
                          "mass": cell["mass"], "fused": fused, "families": len(selected),
                          "mean_chain_error": _encoded(_mean(errors["chain"])),
                          "mean_balanced_error": _encoded(_mean(errors["balanced"])),
                          "mean_difference": _encoded(mean),
                          "individual_orders": {name: signs[name] for name in
                                                ("negative", "equal", "positive", "unresolved")}})
    for spread, mass, fused in sorted({key[:3] for key in exact_means}):
        counts = sorted(key[3] for key in exact_means if key[:3] == (spread, mass, fused))
        for smaller, larger in zip(counts, counts[1:]):
            before = exact_means[spread, mass, fused, smaller]
            after = exact_means[spread, mass, fused, larger]
            change = (after[0] - before[1], after[1] - before[0])
            contrasts.append({"spread": spread, "mass": mass, "fused": fused,
                              "smaller_count": smaller, "larger_count": larger,
                              "mean_difference_change": _encoded(change), "order": _order(change)})
    return {"stage": "exploratory", "cells": cells, "adjacent_count_changes": contrasts,
            "interpretation": "D=A_chain-A_balanced; positive means chain is less accurate. "
            "Intervals bound numerical reference uncertainty for these observed samples, "
            "not population uncertainty. Adjacent changes are differences between independent "
            "cell means, not paired input-level changes or a test of independence.",
            "serialization": "50 significant decimal digits, outward-rounded endpoints"}


def markdown(summary: dict) -> str:
    def display(interval):
        return f"{(Decimal(interval['low']) + Decimal(interval['high'])) / 2:.3E}"
    lines = ["# 扩展 pilot：描述性汇总", "",
             "D = A_chain − A_balanced；正数表示 chain 更差。每格单独汇总，两种 FMA arm 不混池。",
             "表中为数值区间中点的显示近似；严格外包端点见 JSON。这不是统计置信区间。", "",
             "| spread | FMA | 块数 | 族数 | mean A_chain | mean A_balanced | mean D | chain 更好/相同/更差/未定 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cell in sorted(summary["cells"], key=lambda c: (c["spread"], c["fused"], c["count"])):
        signs = cell["individual_orders"]
        counts = "/".join(str(signs[k]) for k in ("negative", "equal", "positive", "unresolved"))
        lines.append(f"| {cell['spread']} | {cell['fused']} | {cell['count']} | {cell['families']} | "
                     f"{display(cell['mean_chain_error'])} | {display(cell['mean_balanced_error'])} | "
                     f"{display(cell['mean_difference'])} | {counts} |")
    lines += ["", "## 相邻块数的样本均值变化", "",
              "比较的是较大块数 mean D 减较小块数 mean D；不同块数使用不同种子。", "",
              "| spread | FMA | 块数变化 | mean D 变化 | 数值次序 |",
              "| --- | --- | --- | --- | --- |"]
    for row in summary["adjacent_count_changes"]:
        lines.append(f"| {row['spread']} | {row['fused']} | {row['smaller_count']} → {row['larger_count']} | "
                     f"{display(row['mean_difference_change'])} | {row['order']} |")
    lines += ["", "8 个族/格用于探索；此表不检验统计独立、不证明等效，也不作跨分布确认。", ""]
    return "\n".join(lines)


def plot(summary: dict, inputs: list[dict], pairs: list[dict], output: Path) -> str:
    """Show individual paired differences and cell means; no error bars implying a CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = {row["family_id"]: row for row in inputs}
    spreads = sorted({row["spread"] for row in summary["cells"]})
    arms = sorted({row["fused"] for row in summary["cells"]})
    fig, axes = plt.subplots(len(spreads), len(arms), figsize=(10, 7), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    for i, spread in enumerate(spreads):
        for j, fused in enumerate(arms):
            axis = axes[i, j]
            cells = sorted((row for row in summary["cells"]
                            if row["spread"] == spread and row["fused"] == fused), key=lambda r: r["count"])
            means = []
            for cell in cells:
                values = [float((_value(row["difference_low"]) + _value(row["difference_high"])) / 2)
                          for row in pairs if row["fused"] == fused
                          and families[row["family_id"]]["cell_index"] == cell["cell_index"]]
                x = [cell["count"] * 2**((k - (len(values) - 1) / 2) * 0.035) for k in range(len(values))]
                axis.scatter(x, values, color="#537899", alpha=0.75, s=25, label="Individual family")
                means.append(float((Decimal(cell["mean_difference"]["low"])
                                    + Decimal(cell["mean_difference"]["high"])) / 2))
            axis.plot([row["count"] for row in cells], means, color="#b74c25", marker="D",
                      linewidth=1.7, markersize=5, label="Cell mean (8 families)")
            axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_xticks([row["count"] for row in cells], [str(row["count"]) for row in cells])
            axis.set_title(f"spread = {spread} | {'FMA' if fused else 'separate mul/add'}")
            axis.grid(axis="y", alpha=0.2)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if i == len(spreads) - 1:
                axis.set_xlabel("Block count (n)")
            if j == 0:
                axis.set_ylabel("D = A_chain - A_balanced")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0, 0].legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)
    fig.suptitle("Exploratory pilot | positive D: chain less accurate\nSame leaf blocks within each comparison; no statistical confidence intervals", fontsize=12)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return matplotlib.__version__


def report(bundle: Path, output: Path, *, with_plot: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    verification = replay(bundle)
    result = summarize(*[_read(bundle / name) for name in
                         ("plan.json", "inputs.json", "measurements.json", "paired_differences.json")])
    source = Path(__file__).read_bytes()
    result["provenance"] = {"replay": verification,
                            "bundle_manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
                            "summary_source_sha256": hashlib.sha256(source).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    if with_plot:
        result["provenance"]["matplotlib"] = plot(result, _read(bundle / "inputs.json"),
                                                 _read(bundle / "paired_differences.json"),
                                                 output / "paired_differences.png")
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(markdown(result), encoding="utf-8")
    (output / "pilot_summary.py.txt").write_bytes(source)
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(report(args.bundle, args.output, with_plot=args.plot), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
