"""Q_8/12 macro score and shortlist — closed-book rewrite (research core: USER-WRITTEN).

第四步的前半：只看幅度的便宜分数 Q，以及据此选出的 shortlist。这是整条研究线的
概念核心——「只估 A、看不见符号 C」的那个分数。后半的 B=3 联合 cell beam 依赖
运行时训练的 numpy probe，属于工程管线，本次不闭卷重写；beam 的选择结果直接采用
冻结 CSV 的 ``beam_selected``，其正确性已由 test_predictor_fixed_k8_beam_inference 守住。

差分测试会用重写的生成器与 oracle 重建冻结 v2 的每棵树，比对 CSV 的
``fixed_q_score``、``fixed_energy_capture``、``shortlisted``、``q_selected``。

Q 分数的定义（研究规格，不是让你猜）
--------------------------------------
对一棵把 leaf_count 个正 FP32 叶子归约成一个根的树：
  1. 每个内部节点 v 有一个「精确子树和」exact_subtree[v]：它下面所有叶子的精确和（不舍入）。
  2. root band：从根出发，用大顶堆按「子树包含的叶子数」展开，每次弹出当前子树最大的节点，
     把它的内部孩子压回堆里，直到取满 budget 个节点。这挑出的是根附近、子树最大、
     因而能量最高的一批节点。
  3. 每个节点的 ULP 能量 = (ulp(exact_subtree[v]) / ulp(exact_root))**2，用 float 计算。
     这是「每个舍入误差最多半个 ULP，方差约 ulp^2/12」的幅度估计。
  4. Q = (前 budget 个节点的能量之和) / 12。
  5. captured_fraction = (前 budget 个的能量和) / (全部内部节点的能量和)。

Explain-back
------------
Q 估计的是 E² = A + C 里的哪一项，为什么：A，因为 Q 只看幅度，忽略符号 C
root band 为什么按「子树叶子数」而不是按节点深度或 ULP 大小来展开：用有限的预算抓住最大的mass
Q 看不见 C，这如何解释「便宜分数给树排序不稳定」：我们通过前面的实验已经知道了C的正负对结果的影响很大

Prediction record（跑测试前写）
--------------------------------
Direction（重写的 Q 能否逐值对上冻结 CSV）：能
Boundary（float 求和顺序会不会影响精确复现）：会
Failure signature（如果 root band 展开顺序写错，最先在哪列露出）：q_score
"""

from __future__ import annotations

import heapq
from fractions import Fraction
from typing import Sequence

from sklearn import tree

from .fp32_oracle import Tree


ROOT_BAND_BUDGET = 8
SHORTLIST_SIZE = 4
UNIFORM_ROUNDING_VARIANCE = 12.0

def exact_subtree_sums(values: Sequence[Fraction], tree: Tree) -> list[Fraction]:
    """Scaffolding. Exact leaf sum under every node, indexed 0 .. root.

    Leaves map to their own value; each internal node is the exact sum of its two
    children. Returned list has length leaf_count + len(nodes); entry ``i`` is the
    exact subtree sum of node ``i``. No rounding happens here.
    """
    exact: list[Fraction] = list(values)
    for left, right in tree.nodes:
        exact.append(exact[left] + exact[right])
    return exact


def subtree_leaf_counts(tree: Tree) -> list[int]:
    """Scaffolding. Number of leaves under every node, indexed 0 .. root."""
    counts = [1] * tree.leaf_count
    for left, right in tree.nodes:
        counts.append(counts[left] + counts[right])
    return counts


def ulp_fraction(value: Fraction) -> Fraction:
    """USER-WRITTEN CORE. The exact FP32 ULP (quantum) of a positive value."""
    if value <= 0:
        raise ValueError("value must be positive")

    e = value.numerator.bit_length() - value.denominator.bit_length()

    def pow2(k: int) -> Fraction:
        return Fraction(1 << k, 1) if k >= 0 else Fraction(1, 1 << -k)

    if value < pow2(e):
        e -= 1

    if e < -126:
        return pow2(-149)
    return pow2(e - 23)


    

def root_band_order(tree: Tree, budget: int) -> tuple[int, ...]:
    """USER-WRITTEN CORE. The root-band internal-node ordering, at most ``budget`` nodes.

    规格：
      - 用一个大顶堆（Python 的 ``heapq`` 是小顶堆，压入 ``(-size, index)`` 即可）；
      - 初始只放根节点，``size(index)`` 是该节点子树里的叶子数（用 subtree_leaf_counts）；
      - 循环：弹出堆顶（当前子树最大的节点），追加到结果；把它的两个孩子里**属于内部节点**
        的（下标 >= leaf_count）压回堆；
      - 直到结果达到 ``min(budget, len(tree.nodes))`` 个为止。
    返回内部节点下标的元组，按弹出顺序。
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if budget == 0 or not tree.nodes:
        return ()

    leaf_count = tree.leaf_count
    counts = subtree_leaf_counts(tree)
    root = tree.root

    def size(index: int) -> int:
        return counts[index]

    heap = [(-size(root), root)]
    result = []

    limit = min(budget, len(tree.nodes))

    while heap and len(result) < limit:
        _, index = heapq.heappop(heap)

        result.append(index)

        # global node index -> tree.nodes 中的下标
        left, right = tree.nodes[index - leaf_count]

        if left >= leaf_count:
            heapq.heappush(heap, (-size(left), left))

        if right >= leaf_count:
            heapq.heappush(heap, (-size(right), right))

    return tuple(result)


def q_macro_score(values: Sequence[Fraction], tree: Tree) -> tuple[float, float]:
    """USER-WRITTEN CORE. Return ``(q_score, captured_fraction)`` for one tree.

    规格（float 求和顺序必须与规格一致才能逐值复现冻结 CSV）：
      - exact = exact_subtree_sums(values, tree)；root = tree.root；
      - root_ulp = ulp_fraction(exact[root])；
      - 每个内部节点 v（下标 leaf_count .. root）的能量：
        ``energy[v] = float(ulp_fraction(exact[v]) / root_ulp) ** 2``；
      - full_q = 按**节点下标升序**把所有内部节点的 energy 相加（float 累加）；
      - selected = root_band_order(tree, ROOT_BAND_BUDGET)；
        selected_q = 按 selected 的**弹出顺序**把这些节点的 energy 相加；
      - q_score = selected_q / UNIFORM_ROUNDING_VARIANCE；
      - captured_fraction = selected_q / full_q。
    """
    exact = exact_subtree_sums(values, tree)

    root = tree.root
    root_ulp = ulp_fraction(exact[root])

    leaf_count = tree.leaf_count

    # energy[i] 对应全局内部节点 leaf_count + i
    energies = []

    for v in range(leaf_count, root + 1):
        ratio = float(ulp_fraction(exact[v]) / root_ulp)
        energy = ratio ** 2
        energies.append(energy)

    # 注意：必须按内部节点下标升序进行 float 累加
    full_q = sum(energies)

    selected = root_band_order(tree, ROOT_BAND_BUDGET)

    # 注意：必须按 root_band_order 的弹出顺序累加
    selected_q = sum(energies[v - leaf_count] for v in selected)
    q_score = selected_q / UNIFORM_ROUNDING_VARIANCE
    captured_fraction = selected_q / full_q

    return q_score, captured_fraction


def shortlist_indices(q_scores: Sequence[float], size: int = SHORTLIST_SIZE) -> tuple[int, ...]:
    """Scaffolding. Indices of the ``size`` lowest Q scores, ties broken by index.

    Empty input or non-positive size is a ValueError.
    """
    if not q_scores:
        raise ValueError("q_scores must be nonempty")
    if size <= 0:
        raise ValueError("shortlist size must be positive")
    order = sorted(range(len(q_scores)), key=lambda i: (q_scores[i], i))
    return tuple(order[: min(size, len(q_scores))])
