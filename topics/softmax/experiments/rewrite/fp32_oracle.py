"""Exact FP32 reduction oracle — closed-book rewrite (research core: USER-WRITTEN).

先填下面两段，再写实现。用自己的话，不要抄旧模块。

Explain-back
------------
目标：目标是在不依赖硬件浮点的情况下，精确模拟 FP32 的二进制加法树的计算过程。通过这个 oracle，我们可以验证不同实现（如硬件浮点或其他软件实现）在执行相同加法树时的结果是否一致。
输入 / 输出：输入是树的每一个叶子节点的值和树的结构，输出是每个内部节点的值和本次计算的误差，以及能得到前面树干所累积下来的误差。
关键 invariant（至少两条）：一个是每个内部节点的值必须是其两个子节点的和经过 FP32 舍入后的结果；另一个是最终的误差必须等于所有局部舍入误差的总和。
误差来源（oracle 记账的是哪一种误差、不记哪一种）：oracle 只记录树中的加法行为中计算产生的舍入误差，而不考虑加法元素前身的元输入的表示误差。
伪代码：

Prediction record（跑差分测试后回填）
------------------------------------
Direction：舍入结果落在离 value 最近的 FP32 格点上，误差绝对值不超过半个量子。
Scale：单次舍入误差在 O(半个 ULP) 量级；整树最终误差是各节点 δ 之和。
Boundary（哪类输入最可能让实现出错）：tie（正中间）、次正规区、进位跨 binade。
Failure signature（如果错了，最可能先看到什么）：强制 tie 的用例最先报错，
  或次正规区量子用错导致小值成片偏移。实测第一次实现即通过，三类边界都没出错。

可以直接使用的 binary32 格式事实（格式常识，不是算法）
------------------------------------------------------------
- 24 位有效数字：正规数形如 s * 2^(e-23)，其中 2^23 <= s < 2^24，e 属于 [-126, 127]。
- 最大有限值 = (2^24 - 1) * 2^104；超过它的舍入结果按本合同抛 OverflowError。
- 次正规数：小于 2^-126 的值是 2^-149 的整数倍，包括 0。
- 舍入：最近；恰好在两个候选正中间时，取整数有效数字为偶数的那个。

玩具例子（3 位有效数字，可手算，用来核对自己的舍入规则）
--------------------------------------------------------
输入 8, 5, 1.75：
  树 (8+5)+1.75 → δ = (-1, +0.25)，E = -0.75，结果 14
  树 8+(5+1.75) → δ = (+0.25, +1)，E = +1.25，结果 16
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

PRECISION = 24
MIN_NORMAL_EXPONENT = -126
SUBNORMAL_QUANTUM = Fraction(1, 2**149)
MAX_FINITE = Fraction(2**24 - 1) * Fraction(2**104)


@dataclass(frozen=True)
class Tree:
    """Explicit binary reduction tree (scaffolding).

    Leaves are indices 0 .. leaf_count-1. Internal node ``leaf_count + k`` is
    ``nodes[k] = (left, right)``. Nodes are evaluated in list order, so each child
    index must already be a leaf or an earlier internal node. The root is the last
    internal node.
    """

    leaf_count: int
    nodes: tuple[tuple[int, int], ...]

    @property
    def root(self) -> int:
        return self.leaf_count + len(self.nodes) - 1


@dataclass(frozen=True)
class Trace:
    """Everything the oracle knows after evaluating one tree (scaffolding)."""

    values: tuple[Fraction, ...]  # stored FP32 leaves, exactly as given
    node_values: tuple[Fraction, ...]  # rounded result of each internal node, in evaluation order
    deltas: tuple[Fraction, ...]  # local rounding error of each internal node
    exact_sum: Fraction  # sum of the leaves in exact arithmetic
    error: Fraction  # node_values[-1] - exact_sum


def round_to_fp32(value: Fraction) -> Fraction:
    """USER-WRITTEN CORE. Round a nonnegative rational to the nearest binary32 value.

    Must reproduce hardware round-to-nearest, ties-to-even exactly, including
    subnormals and the carry into the next binade. Raise ``ValueError`` for negative
    input and ``OverflowError`` when the rounded result would exceed ``MAX_FINITE``.

    需要自己想清楚的点（回填后的理解）：
      - 零怎么处理；  --> 提前返回 0，因为 0 没有 binade（log2 无定义）。
      - 给定 value，它落在哪个 binade，量子是多少；  --> e = floor(log2 value)，
        用分子分母 bit_length 之差先猜再校正一次；正规区量子 = 2^(e-23)。
      - 次正规区的量子和正规区有什么不同；  --> e < -126 时量子不再随 e 缩小，
        固定为 2^-149（即两者取大）。
      - value/量子 舍入成整数时的 tie 与偶数选择；  --> r=1/2 时取偶，
        这里直接依赖 Fraction 上 round() 的 half-to-even 语义。
      - 舍入后有效数字变成 2^24 时会发生什么。  --> 不是溢出，是自然进位到
        下一个 binade（2^24 * 2^(e-23) = 2^(e+1)），无需特殊处理；真正的溢出
        只在结果超过 MAX_FINITE 时发生，最后统一检查一次。
    """
    if value < 0:
        raise ValueError("Input must be nonnegative.")
    if value == 0:
        return Fraction(0)
    e = value.numerator.bit_length() - value.denominator.bit_length()

    def pow2(k: int) -> Fraction:
        return Fraction(1 << k, 1) if k >= 0 else Fraction(1, 1 << -k)

    if value < pow2(e):
        e -= 1
    # elif value >= pow2(e + 1):
        # e += 1

    if e < -126:
        quantum = pow2(-149)
    else:
        quantum = pow2(e - 23)

    s = value / quantum    
    s_rounded = round(s)

    result = Fraction(s_rounded) * quantum

    if result > MAX_FINITE:
        raise OverflowError("Rounded result exceeds maximum finite FP32 value.")

    return result

def is_stored_fp32(value: Fraction) -> bool:
    """USER-WRITTEN CORE. True iff ``value`` is nonnegative and already exactly representable.

    这是输入合同：为什么 oracle 要求叶子已经是 FP32？答案写进 explain-back。
    """
    if value < 0:
        return False
    try:
        return round_to_fp32(value) == value
    except OverflowError:
        return False


def reduce_tree(values: tuple[Fraction, ...], tree: Tree) -> Trace:
    """USER-WRITTEN CORE. Evaluate ``tree`` exactly as an FP32 machine would.

    Every internal node adds its two (already rounded) children exactly, rounds the
    result with ``round_to_fp32``, and records the local rounding error. Reject
    inputs that violate the stored-FP32 contract with ``ValueError``.

    The differential test checks, against the legacy oracle and hardware float32:
      - every ``node_values[k]`` matches;
      - every ``deltas[k]`` matches;
      - ``error == sum(deltas)`` (write down *why* before relying on it).
    """
    if len(values) != tree.leaf_count:
        raise ValueError("Leaf count mismatch.")
    if not all(is_stored_fp32(v) for v in values):
        raise ValueError("All leaves must be stored FP32.")

    node_values = []
    deltas = []

    for left, right in tree.nodes:
        left_value = values[left] if left < tree.leaf_count else node_values[left - tree.leaf_count]
        right_value = values[right] if right < tree.leaf_count else node_values[right - tree.leaf_count]

        exact_sum = left_value + right_value
        rounded_sum = round_to_fp32(exact_sum)
        delta = rounded_sum - exact_sum

        node_values.append(rounded_sum)
        deltas.append(delta)

    exact_total = sum(values)
    final_error = node_values[-1] - exact_total

    return Trace(
        values=values,
        node_values=tuple(node_values),
        deltas=tuple(deltas),
        exact_sum=exact_total,
        error=final_error
    )
