"""A/C energy decomposition of one oracle trace — closed-book rewrite (research core: USER-WRITTEN).

第二步：复现"误差由舍入符号相干性主导"这个发现。先填 explain-back，再写实现。

Explain-back
------------
E² = A + C 里 A 和 C 各是什么（用 δ 写出定义）：A = Σδv² 是局部能量，C = 2Σδuδ_v 是符号相干项
为什么 C 的符号能正能负、A 不能：因为 A 是平方和，所有项都是非负的；而 C 是两两乘积的和，符号可以相同也可以相反。有点矢量的感觉。
"C 主导"在这里的可操作定义是什么（用什么统计量、跨什么比较）：
伪代码：

Prediction record（跑测试前写）
--------------------------------
Direction（在 wide-range 输入上，跨树看 C 的离散程度比 A 大还是小）：
Scale（大概几倍）：
Boundary（哪种输入或树会让 A 和 C 差不多大）：符号不相关时，或某个 δ 独大时，C 相对 A 变小
Failure signature（如果 C 的两种算法不一致，最可能是哪里错）：

玩具例子（接 fp32_oracle 的 8, 5, 1.75）
------------------------------------------
  树 (8+5)+1.75：δ = (-1, +0.25)  → A = 1.0625, C = -0.5,  E² = 0.5625
  树 8+(5+1.75)：δ = (+0.25, +1)  → A = 1.0625, C = +0.5,  E² = 1.5625
两棵树 A 相同，好坏全由 C 决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence
from statistics import pstdev

from numpy import float32

from .fp32_oracle import Trace


@dataclass(frozen=True)
class ACSplit:
    """Exact energy split of one trace (scaffolding)."""

    e2: Fraction  # E * E
    a_local: Fraction  # sum of delta_v^2
    c_coherence: Fraction  # 2 * sum_{u<v} delta_u * delta_v

    @property   
    def c_over_a(self) -> float:
        if self.a_local == 0:
            return float("nan")
        return float(self.c_coherence / self.a_local)


def ac_decomposition(trace: Trace) -> ACSplit:
    """USER-WRITTEN CORE. Split E² exactly into local energy A and coherence C.

    Compute C two independent ways and raise ``AssertionError`` if they differ:
      1. from the identity, C = E² - A;
      2. from the explicit pairwise sum 2 * sum_{u<v} delta_u * delta_v.
    The pairwise sum has n(n-1)/2 terms; find the O(n) way to accumulate it
    (hint: what does "sum over u<v" look like if you keep a running prefix sum?).
    All arithmetic stays in Fraction.
    """
    prefix = Fraction(0)
    c = Fraction(0)
    for d in trace.deltas:
        c += 2 * prefix * d
        prefix += d

    a = sum((d * d for d in trace.deltas), Fraction(0))

    if c != trace.error ** 2 - a:
        raise AssertionError(f"coherence mismatch: {c} vs {trace.error ** 2 - a}")
    return ACSplit(e2=trace.error ** 2, a_local=a, c_coherence=c)


def variation_ratio(splits: Sequence[ACSplit]) -> float:
    """USER-WRITTEN CORE. Population std of C divided by population std of A across trees.

    This is the operational "C dominates" statistic: for one fixed input, how much of
    the tree-to-tree spread of E² comes from coherence rather than local energy.
    Return ``nan`` when fewer than two splits are given or std(A) is zero.
    """
    C = [float(s.c_coherence) for s in splits]
    A = [float(s.a_local) for s in splits]
    if len(splits) < 2:
        return float("nan")
    std_C = pstdev(C)
    std_A = pstdev(A)
    if std_A == 0:
        return float("nan")
    return std_C / std_A