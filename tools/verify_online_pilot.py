"""Independent numerical verification of a saved online pilot bundle.

This is a second implementation, not a replay: nothing from the ``online`` package is
imported. Binary32 arithmetic comes from hardware ``numpy.float32`` plus a from-scratch
exact-rational round-to-nearest-even, the exponential and the real-exp reference come
from ``mpmath`` interval arithmetic, and the block maxima are regenerated from their
recorded seeds. It therefore re-derives ``round_to_fp32``, ``correctly_rounded_exp``,
``merge_reduce``, ``denominator_interval`` and ``relative_error`` along a separate route
and compares every saved number against it.

What this does NOT establish: the contract itself. A wrong definition of the merge
recurrence or of the real-exp reference would be reproduced faithfully here. This checks
that the delivered code computes what the contract says and that the saved numbers follow
from the saved inputs.

Requires mpmath (audit-only dependency; not needed by the test suite or the experiments).
"""

import argparse
import hashlib
import json
import random
import struct
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

import numpy as np

try:
    import mpmath
    from mpmath.libmp import to_rational
except ImportError as error:  # pragma: no cover - environment guard
    raise SystemExit("verify_online_pilot needs mpmath: pip install mpmath") from error

PRECISION = 700
TWO = F(2)
MAX_F32 = F((2 - 2 ** -23) * 2 ** 127)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exact(value):
    """Exact Fraction for an mpmath mpf; mpf values are dyadic, so this is lossless."""
    return F(*to_rational(mpmath.mpf(value)._mpf_))


def rne_f32(value):
    """Round a signed rational to binary32, nearest-even, written from the format alone."""
    if value == 0:
        return F(0)
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while TWO ** exponent > magnitude:
        exponent -= 1
    while TWO ** (exponent + 1) <= magnitude:
        exponent += 1
    quantum = max(exponent - 23, -149)
    scaled = magnitude / TWO ** quantum
    units, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and units % 2):
        units += 1
    rounded = F(units) * TWO ** quantum
    require(rounded <= MAX_F32, "Rounded outside finite binary32")
    return sign * rounded


def bits(value):
    return struct.pack(">f", np.float32(value)).hex()


def from_bits(pattern):
    return struct.unpack(">f", bytes.fromhex(pattern))[0]


def certified_exp_f32(gap, cache):
    """exp(gap) rounded to binary32, proved by an mpmath interval at PRECISION bits."""
    if gap == 0.0:
        return 1.0
    if gap in cache:
        return cache[gap]
    argument = F(gap)
    enclosure = mpmath.iv.exp(mpmath.iv.mpf(argument.numerator) / mpmath.iv.mpf(argument.denominator))
    low, high = rne_f32(exact(enclosure.a)), rne_f32(exact(enclosure.b))
    require(low == high, f"exp({gap!r}) straddles a rounding boundary at {PRECISION} bits")
    cache[gap] = float(low)
    return cache[gap]


def chain_nodes(leaves):
    nodes, current = [], 0
    for leaf in range(1, leaves):
        nodes.append((current, leaf))
        current = leaves + len(nodes) - 1
    return nodes


def balanced_nodes(leaves):
    nodes, level = [], list(range(leaves))
    while len(level) > 1:
        nxt = []
        for index in range(0, len(level) - 1, 2):
            nodes.append((level[index], level[index + 1]))
            nxt.append(leaves + len(nodes) - 1)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return nodes


def depth(leaves, nodes):
    heights = [0] * (leaves + len(nodes))
    for index, (left, right) in enumerate(nodes):
        heights[leaves + index] = 1 + max(heights[left], heights[right])
    return heights[-1]


def check_shape(leaves, nodes):
    require(len(nodes) == leaves - 1, "A full binary tree over L leaves has L-1 internal nodes")
    used = set()
    for index, (left, right) in enumerate(nodes):
        node = leaves + index
        for child in (left, right):
            require(0 <= child < node, f"Child {child} of node {node} is not evaluated earlier")
            require(child not in used, f"Index {child} is used as a child more than once")
            used.add(child)


def run_schedule(maxima, masses, nodes, fused, cache, cross):
    """Re-derive one execution of the merge recurrence from contract section 2."""
    leaves = len(maxima)
    max_value = list(maxima) + [None] * len(nodes)
    ell = [float(mass) for mass in masses] + [None] * len(nodes)
    absorbed = exp_underflow = product_underflow = 0
    for index, (left, right) in enumerate(nodes):
        m_a, m_b, l_a, l_b = max_value[left], max_value[right], ell[left], ell[right]
        m_v = max(m_a, m_b)
        gap_a = float(np.float32(m_a) - np.float32(m_v))
        gap_b = float(np.float32(m_b) - np.float32(m_v))
        w_a = certified_exp_f32(gap_a, cache)
        w_b = certified_exp_f32(gap_b, cache)
        exp_underflow += (w_a == 0.0) + (w_b == 0.0)
        exact_a, exact_b = F(l_a) * F(w_a), F(l_b) * F(w_b)
        if fused:
            require(w_a == 1.0 or w_b == 1.0, "Fused branch requires one weight to be 1")
            value = exact_b + F(l_a) if w_a == 1.0 else exact_a + F(l_b)
            l_v = float(rne_f32(value))
            products = (float(rne_f32(exact_a)), float(rne_f32(exact_b)))
            nonzero = exact_a > 0 and exact_b > 0
        else:
            p_a = np.float32(l_a) * np.float32(w_a)
            p_b = np.float32(l_b) * np.float32(w_b)
            l_v = float(p_a + p_b)
            cross["operations"] += 3
            cross["disagreements"] += (
                (float(rne_f32(exact_a)) != float(p_a))
                + (float(rne_f32(exact_b)) != float(p_b))
                + (float(rne_f32(F(float(p_a)) + F(float(p_b)))) != l_v))
            products = (float(p_a), float(p_b))
            product_underflow += sum(
                value > 0 and product == 0.0 for value, product in zip((exact_a, exact_b), products))
            nonzero = all(product > 0 for product in products)
        if nonzero and l_v in products:
            absorbed += 1
        max_value[leaves + index] = float(m_v)
        ell[leaves + index] = l_v
    return ell[-1], (absorbed, exp_underflow, product_underflow)


def reference_interval(maxima, masses):
    """Rigorous bracket of l_* = sum_b mass_b exp(m_b - M), exact logit differences."""
    top = max(maxima)
    total = mpmath.iv.mpf(0)
    for maximum, mass in zip(maxima, masses):
        difference = F(maximum) - F(top)
        total += mass * mpmath.iv.exp(
            mpmath.iv.mpf(difference.numerator) / mpmath.iv.mpf(difference.denominator))
    return total


def relative_error_bracket(computed, reference):
    """Outer bounds on |computed - l_*| / l_* over the whole reference interval."""
    value = mpmath.iv.mpf(computed)
    at_low = abs(value - mpmath.iv.mpf([reference.a, reference.a])) / mpmath.iv.mpf([reference.a, reference.a])
    at_high = abs(value - mpmath.iv.mpf([reference.b, reference.b])) / mpmath.iv.mpf([reference.b, reference.b])
    inside = reference.a <= mpmath.mpf(computed) <= reference.b
    low = mpmath.mpf(0) if inside else min(at_low.a, at_high.a)
    return exact(low), exact(max(at_low.b, at_high.b))


def fraction(payload):
    return F(int(payload["numerator"]), int(payload["denominator"]))


def decimal_interval(payload):
    return F(Decimal(payload["low"])), F(Decimal(payload["high"]))


def verify(bundle, summary_dir=None, progress=False):
    mpmath.mp.prec = PRECISION
    mpmath.iv.prec = PRECISION
    manifest = read(bundle / "manifest.json")
    require(manifest["status"] == "complete", "Incomplete bundle")
    for name, digest in manifest["files"].items():
        require(sha(bundle / name) == digest, f"File hash mismatch: {name}")

    plan = read(bundle / "plan.json")
    inputs = read(bundle / "inputs.json")
    graphs = {row["graph_id"]: row for row in read(bundle / "graphs.json")}
    measurements = {(row["family_id"], row["graph_id"].split(":")[0], row["fused"]): row
                    for row in read(bundle / "measurements.json")}
    paired = {(row["family_id"], row["fused"]): row for row in read(bundle / "paired_differences.json")}

    for graph_id, graph in graphs.items():
        require(graph["status"] == "ok", f"{graph_id} is not ok")
        leaves, nodes = graph["leaf_count"], [tuple(pair) for pair in graph["nodes"]]
        check_shape(leaves, nodes)
        expected = chain_nodes(leaves) if graph["requested"] == "chain" else balanced_nodes(leaves)
        require(nodes == expected, f"{graph_id} does not match the {graph['requested']} shape")
        require(depth(leaves, nodes) == (leaves - 1 if graph["requested"] == "chain"
                                         else leaves.bit_length() - 1), f"{graph_id} depth")

    seeds, vectors = set(), {}
    for row in inputs:
        cell = plan["cells"][row["cell_index"]]
        require(row["seed"] in cell["seeds"], f"{row['family_id']} seed is not in its cell plan")
        require(row["seed"] not in seeds, f"Duplicate seed {row['seed']}")
        seeds.add(row["seed"])
        require(len(row["maxima_fp32"]) == cell["count"], f"{row['family_id']} leaf count")
        require(set(row["masses"]) == {cell["mass"]}, f"{row['family_id']} mass")
        key = tuple(row["maxima_fp32"])
        require(key not in vectors, f"{row['family_id']} duplicates {vectors.get(key)}")
        vectors[key] = row["family_id"]
        generator = random.Random(row["seed"])
        regenerated = [bits(generator.uniform(-float(cell["spread"]), 0.0)) for _ in range(cell["count"])]
        require(regenerated == row["maxima_fp32"], f"{row['family_id']} does not regenerate from its seed")
        require(all(-cell["spread"] <= from_bits(pattern) <= 0.0 for pattern in row["maxima_fp32"]),
                f"{row['family_id']} outside [-spread, 0]")

    require(len(paired) == 2 * len(inputs), "Paired coverage")
    cache, cross = {}, {"operations": 0, "disagreements": 0}
    totals = {"families": 0, "roots": 0, "event_rows": 0, "reference_brackets": 0, "pairs": 0}
    cells = {}
    for row in inputs:
        family, masses = row["family_id"], row["masses"]
        maxima = [from_bits(pattern) for pattern in row["maxima_fp32"]]
        leaves = len(maxima)
        reference = reference_interval(maxima, masses)
        errors = {}
        for kind in ("chain", "balanced"):
            nodes = [tuple(pair) for pair in graphs[f"{kind}:{leaves}"]["nodes"]]
            for fused in (False, True):
                saved = measurements[(family, kind, fused)]
                require(saved["status"] == "ok", f"{family} {kind} status")
                require(saved["exp"] == plan["exp"], "exp implementation drift")
                root, events = run_schedule(maxima, masses, nodes, fused, cache, cross)
                totals["roots"] += 1
                require(bits(root) == saved["computed_fp32"],
                        f"{family} {kind} fused={fused}: root {bits(root)} != {saved['computed_fp32']}")
                require(events == (saved["absorbed_merges"], saved["exp_underflow_edges"],
                                  saved["product_underflow_edges"]),
                        f"{family} {kind} fused={fused}: rounding event counts")
                totals["event_rows"] += 1
                stored = (fraction(saved["error_low"]), fraction(saved["error_high"]))
                bracket = relative_error_bracket(root, reference)
                totals["reference_brackets"] += 1
                require(stored[0] <= bracket[0] and bracket[1] <= stored[1],
                        f"{family} {kind} fused={fused}: saved error interval does not enclose "
                        f"the {PRECISION}-bit bracket")
                errors[(kind, fused)] = (root, *stored)
        for fused in (False, True):
            saved = paired[(family, fused)]
            require(saved["baseline"] == "balanced" and saved["schedule"] == "chain", "Paired orientation")
            chain_root, chain_low, chain_high = errors[("chain", fused)]
            balanced_root, balanced_low, balanced_high = errors[("balanced", fused)]
            if chain_root == balanced_root:
                low = high = F(0)
            else:
                low, high = chain_low - balanced_high, chain_high - balanced_low
            require((low, high) == (fraction(saved["difference_low"]), fraction(saved["difference_high"])),
                    f"{family} fused={fused}: paired difference interval")
            order = -1 if high < 0 else (1 if low > 0 else (0 if low == high == 0 else None))
            require(order == saved["error_order"], f"{family} fused={fused}: error order")
            totals["pairs"] += 1
            cell = cells.setdefault((row["cell_index"], fused), [])
            cell.append({"chain": (chain_low, chain_high), "balanced": (balanced_low, balanced_high),
                         "difference": (low, high), "order": order})
        totals["families"] += 1
        if progress:
            print(f"  {family} ({leaves} blocks) verified", flush=True)

    require(cross["disagreements"] == 0, "Hardware float32 disagreed with the exact rounding")

    summary_cells = summary_changes = 0
    if summary_dir is not None:
        summary = read(summary_dir / "summary.json")
        means = {}
        for cell in summary["cells"]:
            key = (int(cell["cell_index"]), cell["fused"] in (True, "True"))
            group = cells[key]
            require(len(group) == int(cell["families"]), "Incomplete summary group")
            for name, field in (("mean_chain_error", "chain"), ("mean_balanced_error", "balanced"),
                                ("mean_difference", "difference")):
                average = tuple(sum((row[field][side] for row in group), F(0)) / len(group)
                                for side in (0, 1))
                stored = decimal_interval(cell[name])
                require(stored[0] <= average[0] and average[1] <= stored[1], f"Summary {name}")
                if field == "difference":
                    means[key] = average
            counted = {"negative": sum(row["difference"][1] < 0 for row in group),
                       "equal": sum(row["difference"] == (F(0), F(0)) for row in group),
                       "positive": sum(row["difference"][0] > 0 for row in group),
                       "unresolved": sum(row["order"] is None for row in group)}
            require({key: int(value) for key, value in cell["individual_orders"].items()} == counted,
                    "Summary order counts")
            summary_cells += 1
        index = {(int(cell["count"]), int(cell["spread"]), cell["fused"] in (True, "True")):
                 (int(cell["cell_index"]), cell["fused"] in (True, "True")) for cell in summary["cells"]}
        for change in summary["adjacent_count_changes"]:
            fused = bool(change["fused"])
            smaller = means[index[(int(change["smaller_count"]), int(change["spread"]), fused)]]
            larger = means[index[(int(change["larger_count"]), int(change["spread"]), fused)]]
            delta = (larger[0] - smaller[1], larger[1] - smaller[0])
            stored = decimal_interval(change["mean_difference_change"])
            require(stored[0] <= delta[0] and delta[1] <= stored[1], "Summary adjacent change")
            expected = "positive" if delta[0] > 0 else ("negative" if delta[1] < 0 else "unresolved")
            require(expected == change["order"], "Summary adjacent order")
            summary_changes += 1

    return {"status": "passed",
            "audit": "independent reimplementation, not a replay of the delivered code",
            "independent_of": ["online.fp32_signed.round_to_fp32", "online.fp32_exp.correctly_rounded_exp",
                               "online.merge.merge_reduce", "online.pilot.denominator_interval",
                               "online.pilot.relative_error", "online.schedules", "online.pilot.uniform_spread"],
            "reference": f"mpmath interval arithmetic at {PRECISION} bits",
            "families_regenerated_from_seed": totals["families"],
            "root_bit_patterns_matched": totals["roots"],
            "rounding_event_rows_matched": totals["event_rows"],
            "error_intervals_enclosing_bracket": totals["reference_brackets"],
            "paired_differences_matched": totals["pairs"],
            "hardware_vs_exact_fp32_operations": cross["operations"],
            "hardware_vs_exact_fp32_disagreements": cross["disagreements"],
            "distinct_exp_arguments_certified": len(cache),
            "summary_cells_checked": summary_cells,
            "summary_adjacent_changes_checked": summary_changes,
            "bundle_manifest_sha256": sha(bundle / "manifest.json"),
            "audit_source_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--summary", type=Path, help="Descriptive summary bundle to cross-check")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = verify(args.bundle, args.summary, progress=args.progress)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
