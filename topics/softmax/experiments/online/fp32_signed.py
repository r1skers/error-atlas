"""Signed FP32 rounding and the four rounded operators of the online merge contract.

研究核心：USER-WRITTEN。先填下面两段，再写实现。不要抄 rewrite/fp32_oracle.py——
它只覆盖非负加法，这里要扩的正是它拒绝的那一半。

记录归属：合同问题 1–4 的首段答案与 Explain-back 的目标/输入输出由用户自写。
"关键 invariant"、"伪代码"、Prediction record，以及合同问题 2–4 中标注"补充"的段落
由 agent 应用户显式请求起草（实现学习协议允许显式指令切换模式）。标注在此，
以免把 agent 的表述误读成用户的独立产出。

Explain-back
------------
目标：对源数据进行舍入处理，以及进行四则运算（加、减、乘、FMA），并记录残差。所有操作都必须精确复现硬件的 round-to-nearest, ties-to-even 语义。
输入 / 输出：输入可以是任意有理数，输出是最接近的 FP32 值以及残差（即舍入误差）。
关键 invariant（至少两条）：
  (a) 舍入结果落在最近格点，|残差| 不超过当前 binade 的半个量子；且 round(-v) == -round(v)。
  (b) 残差约定 residual = result - exact，即 computed = exact + delta。整树误差因此是
      delta 的加权和，恒等式的符号方向由这一条固定；反向定义会让 E = -sum(W_v delta_v)。
  (c) round_to_fp32 对"这个精确值由哪个算子产生"完全不知情。四个算子的唯一差别
      是喂给它的精确表达式，所以每个都只有一行。
  (d) 溢出阈值是 MAX_FINITE 与 2^128 的中点，不是 MAX_FINITE，所以判据必须在舍入之后。
误差来源（本模块记账的是哪一种、不记哪一种）：
  记：每个算子的一次舍入残差（加、减、乘各一次；FMA 对 a*b+c 整体只舍一次）。
  不记：输入的表示误差（叶子必须已是 stored FP32）；exp 的实现误差与 Delta-m 的舍入——
  它们被吸收进冻结的 w_hat，不属于本模块任何算子（合同 §4）。
伪代码：
  round_to_fp32(v):
      negative <- v < 0;  m <- |v|
      e <- 由分子分母 bit_length 之差估 binade，必要时向下修正一次
      quantum <- 2^-149 if e < -126 else 2^(e-23)
      s <- round_half_to_even(m / quantum)
      r <- s * quantum
      if r > MAX_FINITE: raise OverflowError        # 舍入之后判，阈值即中点
      return -r if negative else r
  rounded(exact):    r <- round_to_fp32(exact); return (r, r - exact)
  fp32_add/sub/mul:  return rounded(a + b) / (a - b) / (a * b)
  fp32_fma(a,b,c):   return rounded(a * b + c)      # 只舍一次

必须自己回答的合同问题（不是实现细节，是边界）
----------------------------------------------
1. RN-even 是符号对称的吗？即 round(-v) 是否恒等于 -round(v)？
   为什么？（提示：tie 规则里的"偶数"指的是有效数字，它和符号有关系吗）--> 是符号对称的。因为 tie 规则只关心有效数字的奇偶性，而有效数字是非负的，所以符号不会影响舍入结果。
2. Fraction 表示不了 -0.0。IEEE-754 里 -0.0 和 +0.0 是不同的位模式。
   本合同接受这个损失吗？先找出 §2 的五个算子里，哪一步可能产生 -0，
   再判断它会不会影响 frozen-weight 恒等式。  --> 本合同接受这个损失。因为在实际计算中，-0.0 和 +0.0 在数值上是等价的，并且不会影响 frozen-weight 恒等式的结果。
   补充：-0 和 +0 一般**不**等价（1/-0 = -inf，signbit 不同），所以理由要落在"够不到"，
   不是"等价"。五个算子里唯一可能产生 -0 的是把一个极小负值舍成零；本合同里够不到：
   l >= 0 且 w_hat > 0，乘法与加法不产生负零；Delta = m_a (-) m_v 在 m_a == m_v 时是 +0，
   两者不同时其绝对差至少是一个次正规量子 2^-149，永远舍不到零。所以 -0 在 (m, l)
   合同内不可达，接受这个损失是安全的。边界：第 3 格的 O 累加器是**有符号**的，
   那时 -0 变得可达，且零的符号会进入 y = O / l 的除法——合同必须在那时重新审。
3. m_a - m_b 是本仓库第一次出现的**相减**。旧线的非负合同是刻意用来回避
   cancellation 的（见 NEXT_SESSION 的遗留范围）。这里放开之后，
   cancellation 会不会污染 frozen-weight 恒等式？为什么？ --> 不会污染 frozen-weight 恒等式。因为即使发生了 cancellation，舍入误差仍然是可控的，并且在计算过程中，残差会被正确记录和处理，从而保证了恒等式的成立。
   补充：结论对，但理由不是"误差可控"（cancellation 恰恰是误差**不**可控的典型场景）。
   真正的理由是**结构上根本不出现**：frozen-weight reference 把 w_hat 冻结成数据，
   于是 Delta 的舍入残差 eta 和 exp 的实现误差都被吸收进 w_hat，恒等式的求和只跑
   mu（乘）与 alpha（加），eta 一次都没出现。所以无论相减的 cancellation 多严重，
   都不可能破坏恒等式；它只改变你冻结的是哪一组 w_hat。
   代价是这一项**也测不出来**：要看见它必须换 specified-exp 或 real-exp reference。
   这同时就是 frozen-weight 为什么是三种 reference 里唯一能跨 CPU/GPU 的那个（合同 §6）——
   GPU 的 expf 和它的 Delta 舍入一起消失进数据里。
   实测印证：spread=25 时相减明确发生 cancellation，恒等式仍逐位精确。
4. 溢出：旧合同只在超过 MAX_FINITE 时抛 OverflowError。有符号之后，
   下溢到 -MAX_FINITE 以下应该怎么处理？ --> 下溢到 -MAX_FINITE 以下也应该抛 OverflowError。因为在有符号的情况下，超过最大有限值的绝对值同样是不可表示的，因此需要统一处理溢出情况。
   补充：对称那半对，阈值那半不对——这句仍是修复前 Bug 1 的说法。阈值**不是**
   MAX_FINITE：其上第一个格点是 2^128，RN 把中点 (MAX_FINITE + 2^128)/2 以下的值
   全部送回 MAX_FINITE（硬件在那一带是有限的），中点本身 ties-to-even 选偶数有效数字
   2^24，进到 2^128 才溢出。所以判据必须在**舍入之后、对 magnitude** 判；符号只是
   最后镜像一次，对称是构造出来的，不需要第二个分支。现在的实现是对的，这段答案没跟上。

Prediction record（跑差分测试后回填）
------------------------------------
Direction：结果落在最近格点，|残差| 不超过半个量子；符号对称，round(-v) == -round(v)。
Scale：单次残差 O(半 ULP)；整树误差是加权残差和 sum(W_v delta_v)，且 W_v <= 1。
Boundary（哪类输入最可能让实现出错）：tie 的**两个方向**（偶数候选在上和在下）、
  次正规区、跨 binade 进位、溢出带 (MAX_FINITE, 中点)、Sterbenz 区的精确相减。
Failure signature（如果错了，最可能先看到什么）：实际出错两处，**都不是数值边界而是
  约定边界**——
  (1) 溢出判据放在输入侧而非舍入之后，使整个 (MAX_FINITE, 中点) 带误报 OverflowError。
      原测试只用 MAX_FINITE * 2 探，两端都在中点之上，放过了它；补了
      test_overflow_threshold_is_the_midpoint_not_max_finite 之后才咬住。
  (2) 残差写成 exact - result。症状不是数值不对，而是恒等式整体差一个负号，
      看起来像推导错而不是约定反。根因是四个算子各自重复了舍入逻辑、没走 rounded()，
      约定散在四处；改成一行一个之后这个 bug 在结构上不可能发生。

可以直接使用的格式事实（与 rewrite/fp32_oracle.py 相同，非算法）
--------------------------------------------------------------
- 24 位有效数字；正规数形如 ±s * 2^(e-23)，2^23 <= s < 2^24，e 属于 [-126, 127]。
- 最大有限值 = (2^24 - 1) * 2^104。
- 次正规数：绝对值小于 2^-126 的值是 2^-149 的整数倍，包括 0。
- 舍入：最近；恰好在正中间时取有效数字为偶数的那个。

玩具例子（3 位有效数字，可手算）
--------------------------------
Sterbenz：a=9, b=5，同号且 b/2 <= a <= 2b，则 a-b=4 精确，无残差。
非 Sterbenz：a=9, b=1.0625，a-b=7.9375 在 3 位有效数字下必须舍入。
"""

from __future__ import annotations

from fractions import Fraction

PRECISION = 24
MIN_NORMAL_EXPONENT = -126
SUBNORMAL_QUANTUM = Fraction(1, 2**149)
MAX_FINITE = Fraction(2**24 - 1) * Fraction(2**104)
MIN_FINITE = -MAX_FINITE


def round_to_fp32(value: Fraction) -> Fraction:
    """USER-WRITTEN CORE. Round a signed rational to the nearest binary32 value.

    必须精确复现硬件的 round-to-nearest, ties-to-even，含次正规、跨 binade 进位，
    以及两个方向的符号。超出 [MIN_FINITE, MAX_FINITE] 时抛 OverflowError。

    与非负版本相比，唯一真正新增的判断是符号怎么进入 tie 规则。先回答
    docstring 里的合同问题 1，再决定是"取绝对值 + 复原符号"还是别的写法——
    两种都可以，但你要能说清为什么它们等价。
    """ 
    negative = value < 0
    magnitude = abs(value)

    e = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()

    def pow2(k: int) -> Fraction:
        return Fraction(1 << k, 1) if k >= 0 else Fraction(1, 1 << -k)

    if magnitude < pow2(e):
        e -= 1

    if e < -126:
        quantum = pow2(-149)
    else:
        quantum = pow2(e - 23)

    s = magnitude / quantum    
    s_rounded = round(s)

    result = Fraction(s_rounded) * quantum

    if result > MAX_FINITE:
        raise OverflowError(f"{value} rounds outside the range of finite FP32")

    if negative:
        result = -result

    return result

def is_stored_fp32(value: Fraction) -> bool:
    """USER-WRITTEN CORE. True iff ``value`` is already exactly a binary32 value.

    有符号版本的输入合同。注意它和非负版本的差别不只是去掉 ``value < 0`` 那一行。
    """
    try :
        return round_to_fp32(value) == value
    except OverflowError:
        return False


def rounded(exact: Fraction) -> tuple[Fraction, Fraction]:
    """Scaffolding. Round one exact rational and return ``(result, residual)``.

    残差约定与 rewrite 的 ``delta`` 一致：``residual = result - exact``。
    这是本模块唯一的 agent 实现，因为它不承载研究判断。
    """
    result = round_to_fp32(exact)
    return result, result - exact


def fp32_add(a: Fraction, b: Fraction) -> tuple[Fraction, Fraction]:
    """USER-WRITTEN CORE. One FP32 addition: ``(result, residual)``."""
    return rounded(a + b)


def fp32_sub(a: Fraction, b: Fraction) -> tuple[Fraction, Fraction]:
    """USER-WRITTEN CORE. One FP32 subtraction: ``(result, residual)``.

    这是合同 §2 第 2 步的 ``m_a (-) m_v``。
    """
    return rounded(a - b)


def fp32_mul(a: Fraction, b: Fraction) -> tuple[Fraction, Fraction]:
    """USER-WRITTEN CORE. One FP32 multiplication: ``(result, residual)``.

    这是合同 §2 第 4 步的 ``l_a (*) w_a``，残差即恒等式里的 mu。
    """
    return rounded(a * b)


def fp32_fma(a: Fraction, b: Fraction, c: Fraction) -> tuple[Fraction, Fraction]:
    """USER-WRITTEN CORE. Fused multiply-add ``a*b + c``: ``(result, residual)``.

    合同 §5 的 FMA 变体。这一个和上面三个不同，**有一个真正的判断**：
    fused 的定义是什么？写之前先预测：存在一组 (a, b, c) 使 fma 的结果与
    "先 mul 再 add"不同吗？如果存在，最小的例子长什么样？
    """
    return rounded(a * b + c)

