"""Reproducible input and tree generators — closed-book rewrite (research core: USER-WRITTEN).

第三步。和前两步不同，这一步的 RNG 调用序列**是给定的规格，不是让你猜的**。
随机数消费顺序属于任意实现选择，猜不出来也不值得猜。这里要学的是另外两件事：

  1. 树的良构性到底由哪些不变量定义（`check_tree_structure` 是真正的研究核心）；
  2. 为什么 seed schedule 一旦冻结就不能改：2026-08 的 192 个输入组和 12288 棵树
     只以 seed 的形式记录在案，规格差一次 RNG 调用，全部冻结证据就再也对不上。

差分测试会直接拿你的实现去重建那批冻结证据，比对 stored_leaf_bits 和 graph_sha256。

Explain-back
------------
一棵合法的 reduction tree 必须满足哪些条件（至少三条）：每个孩子的下标必须小于它父节点自己的下标，根节点索引是总叶子数加上内部节点数减一，除根以外每个节点被用作孩子的次数必须是 1，根节点不能被用作孩子
为什么每个孩子的下标必须小于它父节点自己的下标：节点按列表顺序求值，所以轮到节点 k 时，它的两个孩子必须已经有值了
如果把 `rng.randint` 和 `rng.randrange` 的调用顺序对调，会发生什么：破坏伪随机数生成器（RNG）的内部状态序列，导致这两次调用的返回值完全改变
为什么 shuffle 必须在全部叶子生成之后、而不能边生成边插入：内部节点就无法获得正确的起始索引，必须先填满0到L-1的叶子，才能正确地生成内部节点的下标

Prediction record（跑测试前写）
--------------------------------
Direction（你觉得第一次能对上冻结的 sha256 吗）：
Boundary（三个生成器里哪个最容易写错）：
Failure signature（如果 RNG 序列错了一次调用，测试会怎么报错）：
"""

from __future__ import annotations

import random
from fractions import Fraction
from collections import Counter

from .fp32_oracle import Tree

WIDE_RANGE_MIN_EXPONENT = -32
WIDE_RANGE_MAX_EXPONENT = 4
FRACTION_FIELD_BITS = 23


def check_tree_structure(tree: Tree) -> None:
    """USER-WRITTEN CORE. Raise ``ValueError`` unless ``tree`` is a well-formed full binary tree.

    这是本步唯一需要你自己想清楚的不变量。一棵把 ``leaf_count`` 个叶子归约成一个根的
    满二叉树，节点数、下标范围、每个下标被用作孩子的次数、以及根的位置，各自应该满足
    什么？想清楚了再写，不要只写一两条。

    提示性的问题，不是清单：
      - L 个叶子的满二叉树有多少个内部节点？
      - ``Tree`` 的求值顺序合同（孩子必须先算好）在下标上等价于什么约束？
      - 除根以外，每个节点被用作孩子几次？根被用作孩子几次？
      - ``leaf_count == 1`` 这个退化情形应该通过还是拒绝？
    """
    if tree.leaf_count < 1:
        raise ValueError("leaf_count must be positive")
    if len(tree.nodes) != tree.leaf_count - 1:
        raise ValueError("number of internal nodes must be leaf_count - 1")

    # 1. 优先校验求值顺序与越界防线
    for i, (left, right) in enumerate(tree.nodes):
        parent_idx = tree.leaf_count + i
        if left < 0 or right < 0:
            raise ValueError(f"Negative child index not allowed: ({left}, {right})")
        if parent_idx <= left or parent_idx <= right:
            raise ValueError(f"Evaluation order violation: Parent {parent_idx} has invalid children ({left}, {right}).")

    # 2. 拓扑合法后，再执行全树入度统计
    children = [child for node in tree.nodes for child in node]
    usage = Counter(children)

    if usage[tree.root] != 0:
        raise ValueError(f"Root node {tree.root} cannot be used as a child.")

    for i in range(tree.root):
        if usage[i] != 1:
            raise ValueError(f"Node {i} must be used exactly once, got {usage[i]}.")


def random_contiguous_split_tree(leaf_count: int, *, seed: int) -> Tree:
    """USER-WRITTEN CORE. Sample a tree by recursively splitting contiguous leaf intervals.

    规格（必须逐字实现，RNG 调用顺序是冻结边界的一部分）：

      - 用 ``random.Random(seed)`` 建立唯一的随机源；
      - 递归处理左闭右开的叶子区间 ``[start, stop)``，初始为 ``[0, leaf_count)``；
      - 区间只含一个叶子时直接返回该叶子下标，**不消费任何随机数**；
      - 否则先抽 ``split = rng.randrange(start + 1, stop)``，
        然后**先**递归构造左区间 ``[start, split)``，**再**递归构造右区间 ``[split, stop)``；
      - 两个子结果都拿到后，把 ``(left, right)`` 追加进节点列表，
        该节点的下标是 ``leaf_count + 已追加节点数 - 1``；
      - 叶子顺序自始至终不被打乱。

    参数校验：``leaf_count`` 必须是正整数（``bool`` 不算整数），``seed`` 必须是整数。
    违反时抛 ``TypeError`` 或 ``ValueError``。
    """
    if isinstance(leaf_count, bool) or not isinstance(leaf_count, int):
        raise TypeError("leaf_count must be an integer")
    if leaf_count < 1:
        raise ValueError("leaf_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    nodes = []

    def walk(start, stop):
        if stop - start == 1:
            return start
        split = rng.randrange(start + 1, stop)
        left = walk(start, split)
        right = walk(split, stop)
        nodes.append((left, right))
        return leaf_count + len(nodes) - 1

    walk(0, leaf_count)
    return Tree(leaf_count=leaf_count, nodes=tuple(nodes))

def random_pair_merge_tree(leaf_count: int, *, seed: int) -> Tree:
    """USER-WRITTEN CORE. Sample a tree by repeatedly merging two random active nodes.

    规格（同样逐字实现）：

      - 用 ``random.Random(seed)`` 建立唯一的随机源；
      - 活跃池初始为 ``[0, 1, ..., leaf_count - 1]``，按此顺序；
      - 只要池中多于一个元素，重复：
          * ``first = rng.randrange(len(active))``；
          * ``second = rng.randrange(len(active) - 1)``，
            若 ``second >= first`` 则 ``second += 1``（这样两个位置必不相同，且只花两次抽样）；
          * **先移除位置较大的那个，再移除位置较小的那个**；每次移除都用
            "把末尾元素搬到该位置再 pop" 的 O(1) 方式，不要用 ``list.pop(i)`` 或 ``remove``；
          * 取出的两个节点下标**升序排列**后作为 ``(left, right)`` 追加进节点列表；
          * 把新节点的下标追加到活跃池末尾；
      - 池中剩下的唯一元素就是根。

    移除顺序和搬运方式会改变池的排列，从而改变后续每一次抽样，所以它们是规格的一部分，
    不是可以自由发挥的细节。

    参数校验同上。
    """
    if isinstance(leaf_count, bool) or not isinstance(leaf_count, int):
        raise TypeError("leaf_count must be an integer")
    if leaf_count < 1:
        raise ValueError("leaf_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    active = list(range(leaf_count))
    nodes = []

    def swap_pop(lst, pos):
        val = lst[pos]
        lst[pos] = lst[-1]
        lst.pop()
        return val

    while len(active) > 1:
        first = rng.randrange(len(active))
        second = rng.randrange(len(active) - 1)
        if second >= first:
            second += 1

        idx1, idx2 = max(first, second), min(first, second)
        val1 = swap_pop(active, idx1)
        val2 = swap_pop(active, idx2)

        nodes.append(tuple(sorted((val1, val2))))
        active.append(leaf_count + len(nodes) - 1)

    return Tree(leaf_count=leaf_count, nodes=tuple(nodes))


def wide_range_random(width: int, *, seed: int) -> tuple[Fraction, ...]:
    """USER-WRITTEN CORE. Sample ``width`` positive stored-FP32 leaves over a wide exponent range.

    规格：

      - 用 ``random.Random(seed)`` 建立唯一的随机源；
      - 生成 ``width`` 个叶子，每个叶子按顺序消费**两次**随机：
          * 先 ``exponent = rng.randint(WIDE_RANGE_MIN_EXPONENT, WIDE_RANGE_MAX_EXPONENT)``，
          * 再 ``fraction_field = rng.randrange(1 << FRACTION_FIELD_BITS)``；
          * 该叶子的精确值是 ``((1 << 23) | fraction_field) * 2 ** (exponent - 23)``，
            用 ``Fraction`` 精确表示，指数为负时用分母，不要经过 ``float``；
      - 全部叶子生成完毕后，用 ``rng.shuffle(...)`` 就地打乱一次；
      - 返回打乱后的元组。

    两次抽样的先后不能颠倒。返回的每个值都必须已经是 stored FP32，
    用 ``fp32_oracle.is_stored_fp32`` 自检一遍。

    参数校验：``width`` 必须是至少 2 的整数，``seed`` 必须是整数。
    """
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width < 2:
        raise ValueError("width must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    leaves = []
    for _ in range(width):
        exponent = rng.randint(WIDE_RANGE_MIN_EXPONENT, WIDE_RANGE_MAX_EXPONENT)
        fraction_field = rng.randrange(1 << FRACTION_FIELD_BITS)
        value = Fraction(((1 << 23) | fraction_field) * 2 ** (exponent - 23))
        leaves.append(value)

    rng.shuffle(leaves)
    return tuple(leaves)
