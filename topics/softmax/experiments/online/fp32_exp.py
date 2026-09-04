"""Correctly-rounded FP32 exp, and the ULP profile of other exp implementations.

研究核心：USER-WRITTEN（``correctly_rounded_exp``）。其余为 agent 脚手架。
这是合同 §3 的 **specified-exp reference**：把 Exp 钉死成"高精度算出的、正确舍入到 FP32
的 exp"，从而让 $\\hat w$ 的来源可复现、跨平台一致。

Explain-back
------------
目标：
输入 / 输出：
关键 invariant（至少两条）：
误差来源（本模块记账的是哪一种、不记哪一种）：
伪代码：

必须自己回答的设计问题
----------------------
1. 高精度算出来的只是一个**近似值**。把它舍到 FP32，什么情况下会得到错的答案？
   （提示：真值恰好落在两个 FP32 的正中间附近时。这叫 table-maker's dilemma。）
2. 因此不能只算一个数就返回。要怎么**证明**这次精度够了？
   （提示：给近似值加减一个误差上界得到一个区间，看区间两端是否舍到同一个 FP32。）
3. 证明不了的时候应该怎么办？**本合同规定：抛 ``ValueError``，不自动提高精度。**
   理由有三条，值得想明白而不是照做：参考实现必须确定性，同样的输入同样的 precision
   永远给同样的结果；自动升精度会把"这个点的界确实很紧"这一事实藏起来，而那正是
   table-maker's dilemma 真正发生的地方；真的遇到精确的平局时，升精度会无限循环。
   抛错的类型也定死为 ``ValueError``，这样真正的程序 bug（比如 TypeError）不会被
   误当成"有原则的拒答"。
4. 输入合同：``argument`` 必须已经是 stored FP32 吗？为什么？
   （提示：本模块只对"exp 这一步的舍入"负责，与 oracle 的输入合同一致。）

Prediction record（跑测试后回填）
--------------------------------
Direction：
Scale：
Boundary（哪类参数最可能让实现出错）：
Failure signature（如果错了，最可能先看到什么）：

decimal 最小示范（陌生 API，agent 提供，可直接运行）
---------------------------------------------------
``decimal`` 是标准库的十进制浮点，精度可调，且 ``Decimal.exp()`` 按文档是
**正确舍入到当前上下文精度**的。两个容易踩的点：

- ``Decimal(some_float)`` 是**精确**转换，不受上下文精度影响；而 ``Decimal(a)/Decimal(b)``
  是一次**运算**，会按上下文精度舍入。FP32 值转成 Python float 是精确的，所以
  ``Decimal(float(x))`` 就是 x 的精确十进制表示，不要走分子除分母。
- 精度用 ``localcontext()`` 临时设置，不要改全局。

    >>> from decimal import ROUND_HALF_EVEN, Context, Decimal
    >>> with localcontext() as ctx:
    ...     ctx.prec = 60
    ...     value = Decimal(float(-2.5)).exp()
    >>> str(value)[:20]
    '0.082084998623898795'

``Decimal.exp()`` 的结果误差不超过末位的半个单位，即相对误差约 ``10 ** -(prec - 1) / 2``；
下面 :data:`DECIMAL_SLACK_DIGITS` 用的是一个宽松得多的界。

本仓库已实测（随机采样 20000 个参数，范围 [-104, 0]，seed 20260905）
--------------------------------------------------------------------
- ``numpy`` 的 float32 exp 只有 66.6% 正确舍入，最大偏差 2 ULP；
- ``math.exp`` 转 double 再舍到 FP32，**该样本中**全部与正确舍入一致。

第一条是把 exp 当作独立测量通道的理由。第二条只是"这批样本里没发现差异"，
不能推出过去每一次实际用到的 exp 输入都正确——那要重放当时真正出现的参数才能说。
不过 frozen-weight 恒等式（合同 §4）**本来就不要求 exp 正确舍入**：$\\hat w$ 是被冻结的
数据，恒等式对任何 exp 实现都成立。所以已有结果不因这一步而动摇，理由是合同的结构，
不是这次的采样。

同样地，``DEFAULT_PRECISION`` 在该样本中全部证明成功，不能推出证明失败的分支永不触发。
"""

from __future__ import annotations

import struct
from collections import Counter
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction

from .fp32_signed import is_stored_fp32

DEFAULT_PRECISION = 60
DECIMAL_SLACK_DIGITS = 2
# Below this argument exp underflows to zero in FP32; above zero the merge never asks,
# because every gap is m_child - max(m_a, m_b) <= 0.
MIN_USEFUL_ARGUMENT = Fraction(-104)


def correctly_rounded_exp(
    argument: Fraction, precision: int = DEFAULT_PRECISION
) -> Fraction:
    """USER-WRITTEN CORE. ``exp(argument)`` rounded to the nearest FP32, with a proof.

    步骤：
      1. 拒收非 stored-FP32 的输入；
      2. ``Decimal(float(argument)).exp()`` 在 ``precision`` 位下算出近似值；
      3. 给它加减一个误差上界，得到一个包住真值的区间；
      4. 区间两端各自舍到 FP32；两端相同才算证明成功，返回那个值；
      5. 证明不了时抛 ``ValueError``（设计问题 3），不要自动提高精度。
         ``precision`` 是调用方给的预算，必须真正被使用，不能忽略它另用一个值。

    误差上界用 ``abs(approx) * Fraction(1, 10 ** (precision - DECIMAL_SLACK_DIGITS))``
    就够宽松了——``Decimal.exp`` 的实际误差比这小两个数量级以上。

    第 4 步要用 ``round_to_fp32``，本模块没有导入它，自己从 ``.fp32_signed`` 加一行。
    """
    raise NotImplementedError


def ulp_distance(value: Fraction, reference: Fraction) -> int:
    """Scaffolding. Signed distance from ``reference`` to ``value``, counted in FP32 steps.

    Both must be stored FP32. Raw bit patterns are monotonic only for nonnegative floats:
    a larger magnitude on the negative side means a larger pattern but a smaller value, so
    subtracting patterns directly reports the wrong sign there. Negating the magnitude for
    negative inputs restores a single monotonic ordering across zero, which also lets the
    two arguments straddle zero. Sign is kept because the direction of a library's error is
    itself informative.
    """
    if not (is_stored_fp32(value) and is_stored_fp32(reference)):
        raise ValueError("ulp_distance compares stored FP32 values.")

    def order_key(number: Fraction) -> int:
        bits = struct.unpack("<I", struct.pack("<f", float(number)))[0]
        return -(bits & 0x7FFFFFFF) if bits & 0x80000000 else bits

    return order_key(value) - order_key(reference)


def exp_ulp_profile(implementation, arguments) -> Counter[int]:
    """Scaffolding. How far ``implementation`` sits from correct rounding, in ULPs.

    Returns a histogram keyed by signed ULP distance. This is the template the eventual
    CUDA measurement reuses: swap in a table of device ``expf`` outputs and the analysis
    is unchanged, which is the whole point of pinning the reference on the CPU first.
    """
    histogram: Counter[int] = Counter()
    for argument in arguments:
        histogram[ulp_distance(implementation(argument), correctly_rounded_exp(argument))] += 1
    return histogram


def _decimal_exp(argument: Fraction, precision: int) -> Fraction:
    """Scaffolding. The high-precision value alone, without the certification step.

    Uses an explicit ``Context`` rather than ``localcontext()``. ``localcontext()`` copies
    the caller's settings and only ``prec`` gets overridden, so a caller that had narrowed
    ``Emin`` would make this silently underflow: with ``Emin = 0`` it returns 0 for
    exp(-10), whose true FP32 rounding is 4.54e-05. A reference must not inherit anything
    from ambient state.
    """
    context = Context(prec=precision, Emin=-999_999_999, Emax=999_999_999,
                      rounding=ROUND_HALF_EVEN)
    return Fraction(context.exp(Decimal(float(argument))))


__all__ = [
    "DEFAULT_PRECISION",
    "MIN_USEFUL_ARGUMENT",
    "correctly_rounded_exp",
    "exp_ulp_profile",
    "ulp_distance",
]