"""Post-hoc mechanism attribution for the constant-block online pilot.

研究核心 USER-WRITTEN：decompose 与 cancellation_savings。
两段核心已有用户实现；本文件另含合同、数据结构和输入检查。尚未分析扩展 pilot 的归因结果。
说明与学习顺序见 topics/softmax/notes/online_attribution_learning.md。

Explain-back（用户填写）
----------------------
目标：
输入 / 输出：输入是一组相同的叶块 family，以及某个 schedule 实际运行得到的 MergeDump。
输出包括实际结果 l_hat、冻结权重参照 l_frozen、真实分母区间 [L,U]，
以及 R、W、总误差 E、每个节点传播到根的残差、舍入预算和归一化后的误差区间。

关键 invariant：E = l_hat - l_* = R + W。
R = l_hat - l_frozen，必须精确等于所有节点带权残差之和。
B_R 是节点带权残差绝对值之和，所以 B_R >= |R|。
所有相对误差都必须除以同一个真实分母 l_*。

为什么 R 可以为负、而 B_R 非负：因为舍入的方向有正有负，而B_R 是舍入的绝对值之和。

为什么两个分量很大时，总误差仍可能很小：一负一正两个舍入抵消了，所以总误差 E 可能比 R、W 都小。

伪代码：检查 family 和 dump 确实属于同一次计算。
取根节点结果 l_hat。
计算冻结权重参照 l_frozen。
计算真实分母区间 [L,U]。
计算 R = l_hat - l_frozen。
取得每个节点传播到根的带权残差，并检查它们的和等于 R。
把残差绝对值相加得到 B_R。
由 [L,U] 得到 W、E 及三种归一化误差的区间。
组装并返回 Attribution。

Prediction record（查看归因结果前填写；已有总误差结果是事后背景）
-----------------------------------------------------------------
Direction（用户原话）：“merge 的带权舍入残差逐渐累积”。
记录时点：扩展 pilot 总误差已见，64 族的分量归因尚未运行。
Scale（量级或相对大小；尚无把握可直说）：
Boundary：
Failure signature：
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .merge import MergeDump, frozen_weight_reference, weighted_residuals
from .pilot import BlockFamily, denominator_interval

Interval = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class Attribution:
    """Signed accounting terms, with separately named normalized intervals.

    reference: rigorous bracket for real-exp l_*.
    rounding_error: R = l_hat - l_frozen, an exact Fraction.
    weight_error: bracket for W = l_frozen - l_*.
    total_error: bracket for E = l_hat - l_*; E = R + W.
    residuals: exact W_v * delta_v in dump node order (the existing frozen-weight identity).
    rounding_budget: B_R = sum(abs(W_v * delta_v)); B_R >= abs(R).
    relative_rounding, relative_weight, relative_total: tight ranges of R/l_*,
      W/l_* and E/l_* using the SAME l_* in reference. They are signed, unlike A.

    R includes multiplication/addition or FMA rounding with actual weights frozen.
    W includes the effect of rounded gaps, exp and composed weights versus real exp;
    W is not an exp-only error and is not necessarily the same across schedules.
    """

    computed: Fraction
    frozen: Fraction
    reference: Interval
    rounding_error: Fraction
    weight_error: Interval
    total_error: Interval
    residuals: tuple[Fraction, ...]
    rounding_budget: Fraction
    relative_rounding: Interval
    relative_weight: Interval
    relative_total: Interval


def _validate_inputs(family: BlockFamily, dump: MergeDump) -> None:
    """Scaffolding guards; arithmetic correctness/provenance is a separate dump audit."""
    if not dump.is_well_formed():
        raise ValueError("Attribution requires a well-formed dump")
    if dump.leaf_max != family.leaf_max or dump.leaf_ell != family.leaf_ell:
        raise ValueError("Dump and reference must use the same constant-block family")
    if dump.max_at(dump.schedule.root) != family.global_max:
        raise ValueError("Root maximum must match the reference normalization")
    if any(value < 0 for value in dump.node_ell):
        raise ValueError("Normalizer masses must be nonnegative")
    if any(not 0 <= weight <= 1 for weight in dump.weight_left + dump.weight_right):
        raise ValueError("Normalizer weights must lie in [0, 1]")


def decompose(family: BlockFamily, dump: MergeDump) -> Attribution:
    """USER-WRITTEN CORE: keep the signs and the shared real reference.

    Available building blocks (already audited; reuse rather than rewrite):
      frozen_weight_reference(dump): exact l_frozen;
      weighted_residuals(dump): exact per-node propagated residuals;
      denominator_interval(family): rigorous [L, U] for l_*.

    Required:
      * return every Attribution field, using Fraction arithmetic throughout;
      * independently check R against the sum of propagated residuals; raise
        ArithmeticError if the exact identity fails, never silently label it valid;
      * B_R is the sum of magnitudes of individual propagated residuals;
      * E and W are signed intervals. Work out which reference endpoint gives
        each bound; do not take abs, midpoint or a float approximation;
      * normalize R, W and E by the SAME l_* in [L, U], returning tight ranges.
        R can be negative. In W/l_* and E/l_*, the numerator also depends on l_*:
        do not divide two unrelated intervals and discard that dependence;
      * handle one leaf: no internal residuals, with a valid exact decomposition.

    These are accounting components, not independent causal interventions. Cross-count
    comparisons use normalized terms; growth of raw R alone is not growth of pilot A.
    """
    _validate_inputs(family, dump)
    computed = dump.ell_at(dump.schedule.root)
    frozen = frozen_weight_reference(dump)
    reference = denominator_interval(family)

    L, U = reference

    R = computed - frozen
    residuals = weighted_residuals(dump)
    if sum(residuals, Fraction(0)) != R:
        raise ArithmeticError("Propagated residuals do not reconstruct rounding error")


    rounding_budget = sum((abs(r) for r in residuals), Fraction(0))

    W = (frozen - U, frozen - L)
    E = (computed - U, computed - L)

    if R >= 0:
        relative_rounding = (R / U, R / L)
    else:
        relative_rounding = (R / L, R / U)
    relative_weight = (frozen/U - 1, frozen/L - 1)
    relative_total = (computed/U - 1, computed/L - 1)

    return Attribution(
        computed=computed,
        frozen=frozen,
        reference=reference,
        rounding_error=R,
        weight_error=W,
        total_error=E,
        residuals=residuals,
        rounding_budget=rounding_budget,
        relative_rounding=relative_rounding,
        relative_weight=relative_weight,
        relative_total=relative_total,
    )

def cancellation_savings(rounding_error: Fraction, weight_error: Interval) -> Interval:
    """USER-WRITTEN CORE: exact range of cancellation between the two components.

    For fixed R and unknown w in [w_low, w_high], define

        S(w) = abs(R) + abs(w) - abs(R + w).

    Return the tight [min S, max S] over that interval, using exact Fractions.
    This is the reduction in the sum of component magnitudes, in denominator units;
    it is not a relative-error ratio or a replacement for the primary metric.
    S >= 0; equal-sign components give zero; opposite signs may cancel completely.

    Consider R positive, zero and negative, and an interval crossing 0 or -R.
    Decide which candidate points are needed; the implementation is yours.
    Invalid endpoint order raises ValueError (provided below).
    """
    if weight_error[0] > weight_error[1]:
        raise ValueError("weight-error interval must be ordered")

    low, high = weight_error
    candidates = [low, high]
    for kink in (Fraction(0), -rounding_error):
        if low <= kink <= high:
            candidates.append(kink)

    savings = tuple(
        abs(rounding_error) + abs(weight) - abs(rounding_error + weight)
        for weight in candidates
    )
    return min(savings), max(savings)
