"""Pure coherence views over one existing oracle trace.

Exact Fraction identities and legacy float summation order are preserved separately.
No input generation, oracle execution, candidate execution, file I/O or seed policy
belongs here. Pairwise diagnostics are lazy: requesting only A/C remains linear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from statistics import mean

from .trace import ReductionTrace

TOP_FRACS = (0.01, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class ACTree:
    graph_family: str
    graph_seed: int
    e2: Fraction
    a_local: Fraction
    c_coherence: Fraction

    @property
    def c_over_a(self) -> float:
        if self.a_local == 0:
            return float("nan")
        return float(self.c_coherence / self.a_local)


@dataclass(frozen=True)
class CoherenceTree:
    graph_family: str
    graph_seed: int
    e2: float
    c_total: float
    c_parent: float
    c_far_ancestor: float
    c_disjoint: float
    c_gap1: float
    c_gap2: float
    c_gap3: float
    c_gap4plus: float
    abs_pair_mass: float
    abs_parent_mass: float
    abs_far_ancestor_mass: float
    abs_disjoint_mass: float
    top1pct_abs_mass_share: float
    top5pct_abs_mass_share: float
    top10pct_abs_mass_share: float

    @property
    def c_ancestor(self) -> float:
        return self.c_parent + self.c_far_ancestor


@dataclass(frozen=True)
class TreeDiagnostic:
    graph_family: str
    c_ancestor: Fraction
    k_total: Fraction
    c_total: Fraction
    abs_k_mass: Fraction
    top_abs_k_mass: tuple[float, ...]
    top_signed_k_recovery: tuple[float, ...]
    rho_abs_k_vs_abs_delta: float
    rho_abs_k_vs_abs_history: float


def _top_share(abs_terms: list[float], fraction: float) -> float:
    total = sum(abs_terms)
    if total == 0.0:
        return 0.0
    count = max(1, math.ceil(len(abs_terms) * fraction))
    return sum(sorted(abs_terms, reverse=True)[:count]) / total


# Preserve the historical ancestor-report rank/NaN semantics during extraction.
def _rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    rx, ry = _rankdata(xs), _rankdata(ys)
    mx, my = mean(rx), mean(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def _top_count(n: int, frac: float) -> int:
    return max(1, min(n, math.ceil(n * frac)))


@dataclass(frozen=True)
class CoherenceAnalysis:
    """Lazy, reusable views of a trace; labels are report metadata only.

    Keep this object to share A/C and topology work across the three views.
    Structural and history pair scans are intentionally independent numeric audits.
    They are diagnostic work, not a low-cost inference path.
    """

    trace: ReductionTrace
    graph_family: str = "unspecified"
    graph_seed: int = 0

    @cached_property
    def ac(self) -> ACTree:
        """Exact E² = A + C, including the independent prefix-pair identity check."""
        prediction = self.trace.prediction
        deltas = self.trace.deltas
        e2 = prediction.signed_error * prediction.signed_error
        a_local = sum((delta * delta for delta in deltas), start=Fraction(0))
        c_coherence = e2 - a_local

        # Exact identity check against the explicit pairwise cross term.
        pairwise = Fraction(0)
        prefix = Fraction(0)
        for delta in deltas:
            pairwise += 2 * prefix * delta
            prefix += delta
        if pairwise != c_coherence:
            raise AssertionError("C != 2 sum_{u<v} delta_u delta_v")
        if a_local + c_coherence != e2:
            raise AssertionError("E^2 != A + C")

        return ACTree(
            graph_family=self.graph_family,
            graph_seed=self.graph_seed,
            e2=e2,
            a_local=a_local,
            c_coherence=c_coherence,
        )

    @cached_property
    def structure(self) -> CoherenceTree:
        """Legacy float partition into parent, far ancestor and disjoint pairs."""
        deltas = [float(delta) for delta in self.trace.deltas]
        node_ids = self.trace.node_ids
        topology = self.trace.topology
        energy = self.ac
        e2_exact = energy.e2
        c_exact = energy.c_coherence

        c_parent = 0.0
        c_far = 0.0
        c_disjoint = 0.0
        c_gap1 = 0.0
        c_gap2 = 0.0
        c_gap3 = 0.0
        c_gap4 = 0.0
        abs_parent = 0.0
        abs_far = 0.0
        abs_disjoint = 0.0
        abs_terms: list[float] = []

        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                term = 2.0 * deltas[i] * deltas[j]
                abs_term = abs(term)
                abs_terms.append(abs_term)
                gap = topology.ancestor_gap(node_ids[i], node_ids[j])
                if gap is None:
                    c_disjoint += term
                    abs_disjoint += abs_term
                elif gap == 1:
                    c_parent += term
                    c_gap1 += term
                    abs_parent += abs_term
                else:
                    c_far += term
                    abs_far += abs_term
                    if gap == 2:
                        c_gap2 += term
                    elif gap == 3:
                        c_gap3 += term
                    else:
                        c_gap4 += term

        c_total = c_parent + c_far + c_disjoint
        c_exact_float = float(c_exact)
        tolerance = max(1e-30, abs(c_exact_float) * 1e-10, sum(abs_terms) * 1e-12)
        if abs(c_total - c_exact_float) > tolerance:
            raise AssertionError("structural C partition does not reconstruct total C")
        if abs((c_gap1 + c_gap2 + c_gap3 + c_gap4) - (c_parent + c_far)) > tolerance:
            raise AssertionError(
                "ancestor gap partition does not reconstruct ancestor C"
            )

        abs_pair_mass = sum(abs_terms)
        return CoherenceTree(
            graph_family=self.graph_family,
            graph_seed=self.graph_seed,
            e2=float(e2_exact),
            c_total=c_exact_float,
            c_parent=c_parent,
            c_far_ancestor=c_far,
            c_disjoint=c_disjoint,
            c_gap1=c_gap1,
            c_gap2=c_gap2,
            c_gap3=c_gap3,
            c_gap4plus=c_gap4,
            abs_pair_mass=abs_pair_mass,
            abs_parent_mass=abs_parent,
            abs_far_ancestor_mass=abs_far,
            abs_disjoint_mass=abs_disjoint,
            top1pct_abs_mass_share=_top_share(abs_terms, 0.01),
            top5pct_abs_mass_share=_top_share(abs_terms, 0.05),
            top10pct_abs_mass_share=_top_share(abs_terms, 0.10),
        )

    @cached_property
    def history(self) -> TreeDiagnostic:
        """Exact recursive H/K terms checked against explicit ancestor pairs."""
        graph = self.trace.graph
        leaf_count = graph.leaf_count
        internal = self.trace.node_ids
        delta = dict(zip(internal, self.trace.deltas, strict=True))

        # Recursive subtree history. history[v] contains all proper internal descendants' residuals.
        subtree_delta: dict[int, Fraction] = {}
        history: dict[int, Fraction] = {}
        k: dict[int, Fraction] = {}
        for offset, node in enumerate(graph.nodes):
            idx = leaf_count + offset
            left_sub = subtree_delta.get(node.left, Fraction(0))
            right_sub = subtree_delta.get(node.right, Fraction(0))
            history[idx] = left_sub + right_sub
            k[idx] = 2 * delta[idx] * history[idx]
            subtree_delta[idx] = history[idx] + delta[idx]

        k_total = sum(k.values(), start=Fraction(0))

        # Independent exact pair sum checks the recursive history identity.
        topology = self.trace.topology

        c_ancestor = Fraction(0)
        c_total = Fraction(0)
        for i, u in enumerate(internal):
            for v in internal[i + 1 :]:
                term = 2 * delta[u] * delta[v]
                c_total += term
                if topology.ancestor_gap(u, v) is not None:
                    c_ancestor += term
        if k_total != c_ancestor:
            raise AssertionError(
                f"ancestor/history identity mismatch: {k_total} != {c_ancestor}"
            )

        if c_total != self.ac.c_coherence:
            raise AssertionError(
                "pairwise history audit disagrees with the A/C identity"
            )

        abs_k_mass = sum((abs(x) for x in k.values()), start=Fraction(0))
        ranked = sorted(internal, key=lambda idx: abs(k[idx]), reverse=True)
        abs_recovery: list[float] = []
        signed_recovery: list[float] = []
        for frac in TOP_FRACS:
            chosen = ranked[: _top_count(len(ranked), frac)]
            part_abs = sum((abs(k[idx]) for idx in chosen), start=Fraction(0))
            part_signed = sum((k[idx] for idx in chosen), start=Fraction(0))
            abs_recovery.append(
                1.0 if abs_k_mass == 0 else float(part_abs / abs_k_mass)
            )
            signed_recovery.append(
                float("nan") if k_total == 0 else float(part_signed / k_total)
            )

        abs_k = [float(abs(k[idx])) for idx in internal]
        abs_delta = [float(abs(delta[idx])) for idx in internal]
        abs_history = [float(abs(history[idx])) for idx in internal]
        return TreeDiagnostic(
            graph_family=self.graph_family,
            c_ancestor=c_ancestor,
            k_total=k_total,
            c_total=c_total,
            abs_k_mass=abs_k_mass,
            top_abs_k_mass=tuple(abs_recovery),
            top_signed_k_recovery=tuple(signed_recovery),
            rho_abs_k_vs_abs_delta=_spearman(abs_k, abs_delta),
            rho_abs_k_vs_abs_history=_spearman(abs_k, abs_history),
        )
