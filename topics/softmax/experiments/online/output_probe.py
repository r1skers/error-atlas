r"""Deterministic V probes and a user-written signed numerator propagation core.

V is a value, not a probability. The logits and block masses determine probabilities.
This module does not run the saved 64-family study or implement output error analysis.

Explain-back (user fills):
Why leaf O_b is n_b * V_b, without an extra global probability factor:因为叶子节点的 O_b 是局部的，局部的分母是 n_b，所以直接乘以 V_b 就可以得到局部的 O_b。
Why actual weights from the existing dump must be reused:因为实际的权重决定了每个节点的贡献，不能随意更改，否则会破坏原有的计算结果。
Why an O child may be negative even though ell and exp weights are nonnegative:因为 O \(O\) 是加权分子，不是误差。 它能为负，是因为 \(V\) 能为负。
What V=1 should imply at EVERY internal node:因为 V=1 表示每个叶子节点的值都是 1，所以在每个内部节点上，O 的值应该是所有叶子节点的和，即总质量。

Audit clarification (agent; preserves the user's explain-back above):
O is the signed weighted numerator, not an error. Its sign comes from V.
ell and exp weights are unnormalized mass/scaling factors, not normalized probabilities.
For V=1, O_hat_v must equal ell_hat_v at EVERY node, with identical arithmetic.
This is mass in that node's max scale, not a plain sum of descendant leaf masses.
"""

from __future__ import annotations

from fractions import Fraction

from .fp32_signed import fp32_add, fp32_mul, fp32_fma, is_stored_fp32
from .merge import MergeDump
from .pilot import BlockFamily

PROBE_NAMES = ("zero", "one", "max_block", "first_block", "alternating_sign")
PROBE_VALUES = frozenset(Fraction(v, 2) for v in (-2, -1, 0, 1, 2))


def probe_values(family: BlockFamily, name: str) -> tuple[Fraction, ...]:
    """Choose a fixed V pattern without changing logits, masses, or their probabilities.

    max_block selects the first maximum in original leaf order (ties resolved before
    any tree runs). All patterns are shared by both graphs and FMA settings.
    """
    if not isinstance(family, BlockFamily):
        raise TypeError("Need a validated BlockFamily")
    if name not in PROBE_NAMES:
        raise ValueError(f"Unknown V probe: {name}")
    count = len(family.maxima)
    chosen = family.maxima.index(family.global_max)
    if name == "zero":
        return (Fraction(0),) * count
    if name == "one":
        return (Fraction(1),) * count
    if name == "max_block":
        return tuple(Fraction(i == chosen) for i in range(count))
    if name == "first_block":
        return tuple(Fraction(i == 0) for i in range(count))
    return tuple(Fraction(1 if i % 2 == 0 else -1) for i in range(count))


def leaf_numerators(family: BlockFamily, values: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Exact local-scale leaves: O_b = n_b V_b, since exp(m_b-m_b)=1.

    No softmax probability or global exp belongs in this local initialization.
    These restricted values keep leaf accumulation exact for the validated masses.
    """
    if not isinstance(family, BlockFamily):
        raise TypeError("Need a validated BlockFamily")
    if len(values) != len(family.maxima) or any(v not in PROBE_VALUES for v in values):
        raise ValueError("Need one probe value from {0, +/-1/2, +/-1} per block")
    result = tuple(Fraction(n) * v for n, v in zip(family.masses, values))
    if not all(is_stored_fp32(v) for v in result):
        raise ValueError("Numerator leaves must be exact stored FP32 values")
    return result


def propagate_numerator(dump: MergeDump, leaf_O: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """USER-WRITTEN CORE: return O at each internal node, in dump schedule order.

    Use the SAME leaf/internal index convention as MergeDump. For internal node k,
    read its child O values and dump.weight_left[k], dump.weight_right[k]. Do not
    recompute max, gaps, exp, or global probabilities. Do not use node_ell as O.

    Separate: two fp32_mul calls followed by fp32_add of their rounded results.
    Fused: follow merge.py's exact branch convention (test left weight == 1 first);
    one fp32_fma, with the other child's O as addend. These helpers return
    (rounded_value, residual); store only the rounded value in the node array.

    Child O may be signed; winner selection depends on the dump WEIGHT, not on O.
    A one-leaf tree has no internal nodes and must return (). Return a tuple, and
    leave final division and real-exp output reference to the next small step.
    """
    if not isinstance(dump, MergeDump) or not dump.is_well_formed():
        raise ValueError("Need a well-formed merge dump")
    if len(leaf_O) != dump.schedule.leaf_count or not all(is_stored_fp32(v) for v in leaf_O):
        raise ValueError("Need one stored signed FP32 numerator per leaf")

    leaf_count = dump.schedule.leaf_count
    node_O = []

    for k, (left, right) in enumerate(dump.schedule.nodes):
        o_a = leaf_O[left] if left < leaf_count else node_O[left - leaf_count]
        o_b = leaf_O[right] if right < leaf_count else node_O[right - leaf_count]

        w_a = dump.weight_left[k]
        w_b = dump.weight_right[k]

        if dump.fused:
            if w_a == 1:
                o_v, _ = fp32_fma(o_b, w_b, o_a)
            elif w_b == 1:
                o_v, _ = fp32_fma(o_a, w_a, o_b)
            else:
                raise ValueError("Fused merge requires one weight to be exactly 1")
        else:
            p_a, _ = fp32_mul(o_a, w_a)
            p_b, _ = fp32_mul(o_b, w_b)
            o_v, _ = fp32_add(p_a, p_b)

        node_O.append(o_v)

    return tuple(node_O)
