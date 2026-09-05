"""Exploratory pilot: does the merge order change the denominator's accuracy?

**Exploratory only.** No preregistration, no frozen hypothesis, no artifacts. Whatever
this produces is a diagnostic used to decide what is worth freezing, never a confirmation.
Design decisions are contract section 12.

研究核心：USER-WRITTEN（``denominator_interval`` 与 ``relative_error``）。
其余为 agent 脚手架。

Explain-back
------------
目标：
输入 / 输出：
关键 invariant（至少两条）：
误差来源（本模块记账的是哪一种、不记哪一种）：
伪代码：

必须自己回答的设计问题
----------------------
1. 参照 ``l_*`` 为什么必须用**实数** exp 的区间和，而不能把 ``correctly_rounded_exp``
   的输出加起来？（合同 §3：后者是 specified-exp 量，自带量化，不是真值。）
2. 指数取 ``c_b - M`` 的**精确有理差**，不是 ``fp32_sub`` 的结果。为什么？
   （提示：``l_*`` 是"如果一切都精确会得到什么"，FP32 减法属于被测对象。）
3. ``l_*`` 只能算到一个区间。那么 ``A_T`` 也只能是区间。
   什么时候两个 schedule 的 ``A`` 区间重叠、因而分不出高下？这一步该怎么报告？

Prediction record（下一轮运行前填写；已完成试跑的解释另记为事后观察）
------------------------------
Direction：
Scale：
Boundary：
Failure signature：
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

from .fp32_signed import fp32_add, is_stored_fp32, round_to_fp32
from .merge import MergeDump, merge_reduce
from .schedules import Schedule

TAYLOR_TERMS = 80
BRACKET_GRID = 1 << 600


@dataclass(frozen=True)
class BlockFamily:
    """One fixed set of leaf blocks, chosen so the blocks carry no rounding of their own.

    Block ``b`` holds ``masses[b]`` copies of the single logit ``maxima[b]``. Then
    ``m_b = maxima[b]`` and ``l_b = sum of exp(0) = masses[b]``, and FP32 accumulates that
    many ones bit-exactly while the count stays at most 2**24. Contract section 12: with no
    intra-block rounding, a comparison across schedules on the same family isolates the
    merge order.

    Round one varies only the max trajectory and the block masses. Blocks whose elements
    differ are a later round, and will need one fixed intra-block algorithm computed once
    and shared by every schedule.
    """

    maxima: tuple[Fraction, ...]
    masses: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.maxima:
            raise ValueError("A family must contain at least one block.")
        if len(self.maxima) != len(self.masses):
            raise ValueError("Every block needs a maximum and a mass.")
        if not all(is_stored_fp32(m) for m in self.maxima):
            raise ValueError("Block maxima must be stored FP32 logits.")
        if any(isinstance(n, bool) or not isinstance(n, int) for n in self.masses):
            raise TypeError("Block masses must be integers, not bools or fractional counts.")
        if not all(1 <= n <= 2**24 for n in self.masses):
            raise ValueError("Block mass must stay where FP32 counts ones exactly.")

    @property
    def leaf_max(self) -> tuple[Fraction, ...]:
        return self.maxima

    @property
    def leaf_ell(self) -> tuple[Fraction, ...]:
        return tuple(Fraction(n) for n in self.masses)

    @property
    def global_max(self) -> Fraction:
        return max(self.maxima)


def accumulates_exactly(mass: int) -> bool:
    """Scaffolding. Check the claim behind ``l_b = mass`` instead of assuming it."""
    total = Fraction(0)
    for _ in range(mass):
        total, residual = fp32_add(total, Fraction(1))
        if residual != 0:
            return False
    return total == mass


def real_exp_interval(argument: Fraction) -> tuple[Fraction, Fraction]:
    """Scaffolding. A rigorous bracket for the real number ``exp(argument)``.

    This is contract section 3's **real-exp reference**, and it is deliberately not the
    decimal route used by ``correctly_rounded_exp``: exact Fraction Taylor after argument
    reduction, then repeated squaring, carrying the bracket throughout. The number of
    halvings adapts to the argument, so unlike the fixed-depth copy in the exp tests this
    one accepts any nonpositive rational — the exponent here is an exact difference of
    two logits, not an FP32 value.
    """
    if argument > 0:
        raise ValueError("The merge only ever asks for exp of a nonpositive argument.")
    halvings = 0
    while abs(argument) / (1 << halvings) >= 1:
        halvings += 1
    reduced = argument / (1 << halvings)

    total, term = Fraction(0), Fraction(1)
    for n in range(TAYLOR_TERMS + 1):
        total += term
        term = term * reduced / (n + 1)
    bound = abs(term) * 3  # |remainder| <= |y|^(N+1)/(N+1)! * e^|y|, and e^|y| < 3
    low, high = total - bound, total + bound

    for _ in range(halvings):
        low, high = low * low, high * high
        low = Fraction(low.numerator * BRACKET_GRID // low.denominator, BRACKET_GRID)
        high = Fraction(-((-high.numerator * BRACKET_GRID) // high.denominator), BRACKET_GRID)
    return low, high


def denominator_interval(family: BlockFamily) -> tuple[Fraction, Fraction]:
    """USER-WRITTEN CORE. A bracket for the exact denominator of ``family``.

    合同 §12 的参照：

        l_* = sum_b masses[b] * exp(maxima[b] - M),   M = max_b maxima[b]

    要点，写之前先确认自己同意：
      - 指数用 ``maxima[b] - M`` 的**精确有理差**，不是 ``fp32_sub`` 的结果；
      - 每一项用 :func:`real_exp_interval` 取区间，再把区间**逐项相加**
        （下界加下界、上界加上界），得到 ``l_*`` 的区间；
      - 全程精确有理运算，不出现任何舍入函数。

    返回 ``(low, high)``，满足 ``low <= l_* <= high``。
    """
    M = family.global_max
    low, high = Fraction(0), Fraction(0)
    for m, n in zip(family.maxima, family.masses):
        exp_low, exp_high = real_exp_interval(m - M)
        low += n * exp_low
        high += n * exp_high
    return low, high


def relative_error(computed: Fraction, denominator: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    """USER-WRITTEN CORE. A bracket for ``|computed - l_*| / l_*``.

    ``l_*`` 只知道落在 ``denominator`` 这个区间里，所以 ``A`` 也只能是区间。
    要返回一个**保守**的界：区间要包住所有可能的 ``A``，宁可宽不可窄。

    想清楚再写：``|computed - l_*|`` 在 ``l_*`` 扫过区间时怎么变化？
    分母 ``l_*`` 变大时 ``A`` 又怎么变？两者不同向，所以不能只算两个端点了事。
    """
    low, high = denominator
    if low <= 0:
        raise ValueError("denominator interval must be strictly positive.")
    if high < low:
        raise ValueError("denominator interval must be ordered.")

    error_at_low = abs(computed - low) / low
    error_at_high = abs(computed - high) / high

    error_low = (
        Fraction(0)
        if low <= computed <= high
        else min(error_at_low, error_at_high)
    )
    error_high = max(error_at_low, error_at_high)
    return error_low, error_high

# --- input families (scaffolding) ------------------------------------------------------


def max_at_position(count: int, mass: int, gap: Fraction, position: int) -> BlockFamily:
    """
    One block sits ``gap`` above the rest; ``position`` says where in the order.

    The sharpest axis in the stage: the same multiset of blocks, reordered. In a chain a
    late maximum lets the small blocks accumulate at their own scale before one rescale,
    while an early maximum folds each of them into a large accumulator one at a time.
    """
    if not 0 <= position < count:
        raise ValueError("position must index one of the blocks.")
    if gap <= 0:
        raise ValueError("The distinguished block has to be the maximum.")
    maxima = [Fraction(0)] * count
    maxima[position] = round_to_fp32(gap)
    return BlockFamily(tuple(maxima), tuple([mass] * count))


def uniform_spread(count: int, mass: int, spread: float, rng: random.Random) -> BlockFamily:
    """Block maxima drawn uniformly from ``[-spread, 0]``, masses held equal.

    The spread knob controls how peaked the softmax is, and so how often a rescaled block
    lands below the absorption threshold. Contract section 5 puts that threshold near a gap
    of 16.6 as an order of magnitude, not as a criterion.
    """
    maxima = tuple(round_to_fp32(Fraction(rng.uniform(-spread, 0.0))) for _ in range(count))
    return BlockFamily(maxima, tuple([mass] * count))


# --- one measurement (scaffolding) -----------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """What one (family, schedule, rounding arm) execution yields."""

    kind: str
    fused: bool
    computed: Fraction
    error_low: Fraction
    error_high: Fraction
    absorbed_merges: int
    exp_underflow_edges: int
    product_underflow_edges: int
    family: BlockFamily

    @property
    def has_valid_error_interval(self) -> bool:
        """Check interval validity only; even a valid interval may be inconclusive."""
        return 0 <= self.error_low <= self.error_high

    def error_difference_interval(self, other: Measurement) -> tuple[Fraction, Fraction]:
        """Conservative paired interval for self's error minus other's, on the same family.

        Marginal interval subtraction can be wider than a joint analysis of the shared
        reference. Equal computed outputs, however, have exactly equal errors.
        """
        if self.family != other.family:
            raise ValueError("Paired errors require the same block family.")
        if not (self.has_valid_error_interval and other.has_valid_error_interval):
            raise ValueError("Cannot compare invalid error intervals.")
        if self.computed == other.computed:
            return Fraction(0), Fraction(0)
        return self.error_low - other.error_high, self.error_high - other.error_low

    def error_order(self, other: Measurement) -> int | None:
        """-1: lower error; +1: higher; 0: proved equal; None: unresolved at this precision."""
        low, high = self.error_difference_interval(other)
        if high < 0:
            return -1
        if low > 0:
            return 1
        if low == high == 0:
            return 0
        return None


def _rounding_event_counts(dump: MergeDump) -> tuple[int, int, int]:
    """Return (absorbed merges, exp-underflow edges, product-underflow edges).

    A finite gap has a positive real exponential, so a stored zero weight belongs to
    exp underflow. A positive exact product rounded to zero belongs to multiplication
    underflow only in the separate arm: FMA never rounds that product independently.
    Absorption counts only nonzero contributions whose combined result equals the
    other contribution rounded alone. For FMA those rounded-alone products are
    counterfactual comparisons, not separately executed multiplications. Partial loss
    is not counted; these events do not replace the residual/error analysis.
    """
    absorbed = exp_underflow = product_underflow = 0
    for k, (left, right) in enumerate(dump.schedule.nodes):
        weights = (dump.weight_left[k], dump.weight_right[k])
        exact = (dump.ell_at(left) * weights[0], dump.ell_at(right) * weights[1])
        products = tuple(round_to_fp32(value) for value in exact)
        exp_underflow += sum(weight == 0 for weight in weights)
        if dump.fused:
            nonzero_contributions = all(value > 0 for value in exact)
        else:
            product_underflow += sum(
                value > 0 and product == 0 for value, product in zip(exact, products)
            )
            nonzero_contributions = all(product > 0 for product in products)
        if nonzero_contributions and dump.node_ell[k] in products:
            absorbed += 1
    return absorbed, exp_underflow, product_underflow


def measure(family: BlockFamily, schedule: Schedule, *, fused: bool = False) -> Measurement:
    """Scaffolding. Run one schedule over one family and bracket its relative error."""
    dump = merge_reduce(family.leaf_max, family.leaf_ell, schedule, fused=fused)
    low, high = relative_error(dump.ell_at(schedule.root), denominator_interval(family))

    absorbed, exp_underflow, product_underflow = _rounding_event_counts(dump)
    return Measurement(
        kind=schedule.kind,
        fused=fused,
        computed=dump.ell_at(schedule.root),
        error_low=low,
        error_high=high,
        absorbed_merges=absorbed,
        exp_underflow_edges=exp_underflow,
        product_underflow_edges=product_underflow,
        family=family,
    )
