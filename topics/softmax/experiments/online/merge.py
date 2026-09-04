"""The (m, l) merge recurrence and the frozen-weight weighted identity.

研究核心：USER-WRITTEN。先填下面两段，再写实现。合同见
``topics/softmax/notes/online_normalizer_contract.md``；这个模块实现的就是那份合同的
§2（递推）与 §4（恒等式）。

Explain-back
------------
目标：
输入 / 输出：
关键 invariant（至少两条）：
误差来源（本模块记账的是哪一种、不记哪一种）：
伪代码：

必须自己回答的设计问题
----------------------
1. ``MergeDump`` 的字段是"一个 GPU kernel 能写回 global memory 的东西"。
   逐节点六个 uint32（m_v、Delta_a、Delta_b、w_a、w_b、l_v）里，
   两个 Delta 在 CPU 上是可以由 ``fp32_sub`` 重算的——那为什么还要 dump？
   （提示：重算只告诉你 Delta *应该*是多少；CPU 没有设备的 expf。）
   哪些量**不能**放进 dump，为什么？（提示：残差是有理数，GPU 算不出来）
2. 合同 §5 说每次 merge 至多一个乘法残差。那么 ``node_residual`` 里的
   ``mu_a + mu_b`` 实际上恒有一项为零。写实现时你要不要显式利用这一点？
   利用了会不会让 fused 分支更难写？
3. 参考递推 ``l_tilde_v = l_tilde_a * w_a + l_tilde_b * w_b`` 用**精确有理运算**，
   叶取计算值。为什么叶不能取"真实的 exp 之和"？（回到合同 §3 的三种 reference）

Prediction record（跑测试后回填）
--------------------------------
Direction：
Scale：
Boundary（哪类输入最可能让实现出错）：
Failure signature（如果错了，最可能先看到什么）：
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from .fp32_signed import is_stored_fp32, round_to_fp32
from .schedules import Schedule, check_schedule
from .fp32_signed import fp32_add, fp32_fma, fp32_mul, fp32_sub

ExpImpl = Callable[[Fraction], Fraction]


@dataclass(frozen=True)
class MergeDump:
    """Everything one execution of a merge schedule leaves behind.

    This is the CPU/GPU interface, so every field holds **stored FP32 values only** —
    exactly the numbers a debug kernel can write back as raw ``uint32`` bit patterns.
    Nothing rational-but-not-representable belongs here: residuals, path weights and the
    frozen-weight reference are all *derived* on the CPU from these fields plus the
    schedule, and must never be added to the dump.

    Per internal node a kernel writes six words: ``node_max``, ``gap_left``,
    ``gap_right``, ``weight_left``, ``weight_right``, ``node_ell``. The products
    ``l (*) w`` are not dumped because FP32 add, sub, mul and fma are correctly rounded
    on NVIDIA hardware and therefore reproducible on the CPU from what is dumped; the
    exp implementation is the one operation that is not, which is precisely why its
    output is frozen here as data.

    The gaps are dumped even though ``fp32_sub`` could recompute them, because that
    recomputation only tells you what the gap *should* have been. Since the CPU does not
    have the device's ``expf``, a wrong weight cannot otherwise be localised: dumping the
    gap separates "the subtraction was wrong" from "the exponential was wrong", which is
    exactly the split step 1a's measured ULP profile needs.
    """

    schedule: Schedule
    leaf_max: tuple[Fraction, ...]
    leaf_ell: tuple[Fraction, ...]
    node_max: tuple[Fraction, ...]
    gap_left: tuple[Fraction, ...]
    gap_right: tuple[Fraction, ...]
    weight_left: tuple[Fraction, ...]
    weight_right: tuple[Fraction, ...]
    node_ell: tuple[Fraction, ...]
    fused: bool

    def max_at(self, index: int) -> Fraction:
        """``m`` of leaf or internal node ``index``, in the shared index space."""
        count = self.schedule.leaf_count
        return self.leaf_max[index] if index < count else self.node_max[index - count]

    def ell_at(self, index: int) -> Fraction:
        """``l`` of leaf or internal node ``index``, in the shared index space."""
        count = self.schedule.leaf_count
        return self.leaf_ell[index] if index < count else self.node_ell[index - count]

    def weights_at(self, node: int) -> tuple[Fraction, Fraction]:
        """``(w_a, w_b)`` of internal node ``node``, in the shared index space."""
        k = node - self.schedule.leaf_count
        return self.weight_left[k], self.weight_right[k]

    def gaps_at(self, node: int) -> tuple[Fraction, Fraction]:
        """``(Delta_a, Delta_b)`` of internal node ``node``, in the shared index space."""
        k = node - self.schedule.leaf_count
        return self.gap_left[k], self.gap_right[k]

    def is_well_formed(self) -> bool:
        """The dump must match its schedule in shape and hold stored FP32 values only.

        Representability alone is not enough: ``all()`` over an empty tuple is true, so a
        dump whose arrays were never filled would otherwise pass. Anything arriving from
        a device parser has to clear the shape check before its numbers mean anything.
        """
        try:
            check_schedule(self.schedule)
        except ValueError:
            return False
        leaves, nodes = self.schedule.leaf_count, len(self.schedule.nodes)
        per_leaf = (self.leaf_max, self.leaf_ell)
        per_node = (
            self.node_max,
            self.gap_left,
            self.gap_right,
            self.weight_left,
            self.weight_right,
            self.node_ell,
        )
        if any(len(group) != leaves for group in per_leaf):
            return False
        if any(len(group) != nodes for group in per_node):
            return False
        return all(
            is_stored_fp32(value) for group in per_leaf + per_node for value in group
        )


def provisional_fp32_exp(argument: Fraction) -> Fraction:
    """Scaffolding placeholder for the exp reference semantics.

    Rounds the platform's double-precision ``math.exp`` to FP32. This is **not** the
    correctly-rounded FP32 exp that contract section 3 requires as the specified-exp
    reference; step 1a replaces it with a ``decimal`` implementation plus a measured ULP
    profile. It is adequate here only because the frozen-weight identity freezes whatever
    this returns as data, so the identity cannot depend on which exp produced it — which
    is exactly the property step 1b exists to demonstrate.
    """
    return round_to_fp32(Fraction(math.exp(float(argument))))


def merge_reduce(
    leaf_max: tuple[Fraction, ...],
    leaf_ell: tuple[Fraction, ...],
    schedule: Schedule,
    exp_impl: ExpImpl = provisional_fp32_exp,
    *,
    fused: bool = False,
) -> MergeDump:
    """USER-WRITTEN CORE. Run the (m, l) merge recurrence and return the dump.

    合同 §2 的五步，逐节点：
      1. ``m_v = max(m_a, m_b)``  —— 选择操作，无舍入，不产生残差；
      2. ``Delta = m (-) m_v``    —— 有符号减法，用 ``fp32_sub``；两个 Delta 都要进 dump；
      3. ``w = exp_impl(Delta)``  —— 结果必须是 stored FP32；
      4. ``p = l (*) w``          —— 用 ``fp32_mul``；
      5. ``l_v = p_a (+) p_b``    —— 用 ``fp32_add``。

    ``fused=True`` 时第 4、5 步合成一次 ``fp32_fma``（合同 §5），此时只舍一次。
    注意胜方那一支 ``Delta`` 恒为 0、``w`` 恒为 1，想清楚这对 fused 分支意味着什么。

    非 stored-FP32 的叶输入按合同直接拒收（``ValueError``），与旧 oracle 一致。
    """
    if len(leaf_max) != len(leaf_ell):
        raise ValueError("Leaf count mismatch.")
    if not all(is_stored_fp32(v) for v in leaf_max + leaf_ell):
        raise ValueError("All leaves must be stored FP32.")
    if len(leaf_max) != schedule.leaf_count:
        raise ValueError("Leaf count mismatch with schedule.")

    check_schedule(schedule)
    
    node_max = []
    gap_left = []
    gap_right = []
    weight_left = []
    weight_right = []
    node_ell = []

    for left, right in schedule.nodes:
        m_a = leaf_max[left] if left < schedule.leaf_count else node_max[left - schedule.leaf_count]
        m_b = leaf_max[right] if right < schedule.leaf_count else node_max[right - schedule.leaf_count]
        l_a = leaf_ell[left] if left < schedule.leaf_count else node_ell[left - schedule.leaf_count]
        l_b = leaf_ell[right] if right < schedule.leaf_count else node_ell[right - schedule.leaf_count]

        m_v = max(m_a, m_b)
        (delta_a, _), (delta_b, _) = fp32_sub(m_a, m_v), fp32_sub(m_b, m_v)
        w_a, w_b = exp_impl(delta_a), exp_impl(delta_b)

        if fused:
            if w_a == 1:
                l_v, _ = fp32_fma(l_b, w_b, l_a)
            elif w_b == 1:
                l_v, _ = fp32_fma(l_a, w_a, l_b)
            else:
                raise ValueError("Fused branch requires one weight to be 1.")
        else:
            p_a, p_b = fp32_mul(l_a, w_a), fp32_mul(l_b, w_b)
            l_v, _ = fp32_add(p_a[0], p_b[0])

        node_max.append(m_v)
        gap_left.append(delta_a)
        gap_right.append(delta_b)
        weight_left.append(w_a)
        weight_right.append(w_b)
        node_ell.append(l_v)

    return MergeDump(
        schedule=schedule,
        leaf_max=leaf_max,
        leaf_ell=leaf_ell,
        node_max=tuple(node_max),
        gap_left=tuple(gap_left),
        gap_right=tuple(gap_right),
        weight_left=tuple(weight_left),
        weight_right=tuple(weight_right),
        node_ell=tuple(node_ell),
        fused=fused
    )


def frozen_weight_reference(dump: MergeDump) -> Fraction:
    """USER-WRITTEN CORE. The frozen-weight reference value of the root.

    合同 §4：冻结 dump 里实际算出的每一个 ``w_hat``，叶取 dump 里的计算值，
    用**精确有理运算**跑同一条递推：

        l_tilde_v = l_tilde_a * w_hat_a + l_tilde_b * w_hat_b

    返回根的 ``l_tilde``。全程不得调用任何舍入函数——一旦舍入，这就不再是参考了。
    """
    leaf_count = dump.schedule.leaf_count
    node_count = len(dump.schedule.nodes)
    l_tilde = [Fraction(0)] * (leaf_count + node_count)

    for i in range(leaf_count):
        l_tilde[i] = dump.leaf_ell[i]

    for k, (left, right) in enumerate(dump.schedule.nodes):
        node_index = leaf_count + k
        w_a, w_b = dump.weights_at(node_index)
        l_a, l_b = l_tilde[left], l_tilde[right]
        l_tilde[node_index] = l_a * w_a + l_b * w_b

    return l_tilde[dump.schedule.root]


def weighted_residuals(dump: MergeDump) -> tuple[Fraction, ...]:
    """USER-WRITTEN CORE. ``W_v * delta_v`` for each internal node, in node order.

    ``delta_v`` 是该节点的局部残差（非 fused 时是 ``mu_a + mu_b + alpha_v``，
    fused 时是那一次 fma 的残差），``W_v`` 是从 v 到根路径上**计算权重**之积。

    这两个都要从 dump 重算，不能在 :func:`merge_reduce` 里顺手存下来——因为 GPU 那边
    只会给你 dump 里的那几个字段，如果这里依赖了别的东西，同一份分析就跑不到硬件数据上。

    合同 §4 的恒等式因此是：

        dump.ell_at(root) - frozen_weight_reference(dump) == sum(weighted_residuals(dump))

    并且必须**逐位**成立，不是近似成立。
    """
    leaf_count = dump.schedule.leaf_count
    node_count = len(dump.schedule.nodes)
    W = [Fraction(1)] * (leaf_count + node_count)
    weighted_deltas = []

    for k in reversed(range(node_count)):
        v = leaf_count + k
        left, right = dump.schedule.nodes[k]
        w_a, w_b = dump.weights_at(v)
        W[left] *= w_a * W[v]
        W[right] *= w_b * W[v]

    for k in range(node_count):
        v = leaf_count + k
        delta_v = dump.ell_at(v) - (dump.ell_at(dump.schedule.nodes[k][0]) * dump.weights_at(v)[0] + dump.ell_at(dump.schedule.nodes[k][1]) * dump.weights_at(v)[1])
        weighted_deltas.append(W[v] * delta_v)

    return tuple(weighted_deltas)

    