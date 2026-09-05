# Online normalizer 的算术合同与加权恒等式

设计笔记，不是证据。用于确定 online/blockwise softmax 阶段的 oracle 边界。
本文**不**包含预注册、不含研究假设 H1、不产生 artifact。
上游背景见 [foundations.md](foundations.md) 与 [rewrite_replication.md](rewrite_replication.md)。

## 1. 范围与非目标

- **范围**：只有 normalizer 状态 $(m,\ell)$。不含 $V$，不含 $O$，不含 $y=O/\ell$。
- **主合同**：FP32。低精度（FP16/BF16/FP8）是**压力测试与候选设计**，不是主流 kernel 的默认事实——
  FlashAttention-2/3 的 $(m,\ell,O)$ 累加器保持 FP32。低精度臂参数化在后，不与主合同同时实例化。
- **非目标**：本文不主张任何原创性。与既有文献的关系需要正式检索后另行书写。

## 2. merge 递推的精确形式

二叉合并树 $T$，叶为块。内部节点 $v$ 有孩子 $a,b$：

1. $m_v=\max(m_a,m_b)$ —— **选择操作，无舍入，不产生残差**。
2. $\hat\Delta_a = m_a\ominus m_v$，残差 $\eta_a$；$\hat\Delta_b$ 同理。
3. $\hat w_a=\mathrm{Exp}(\hat\Delta_a)$，其中 $\mathrm{Exp}$ 是**某个具体实现**（stdlib、libm、CUDA `expf`/`__expf`/`exp2f`）。
4. $p_a=\hat\ell_a\otimes\hat w_a$，残差 $\mu_a=p_a-\hat\ell_a\hat w_a$。
5. $\hat\ell_v=p_a\oplus p_b$，残差 $\alpha_v=\hat\ell_v-(p_a+p_b)$。

算子集是 $\{\max,\ominus,\mathrm{Exp},\otimes,\oplus\}$，另有 FMA 变体（见 §5）。
FTZ / gradual underflow 是**合同参数**，不是实现细节。

## 3. 三种 reference

| 名称 | 参考值 | 量的性质 | 可精确核对 |
| --- | --- | --- | --- |
| **frozen-weight** | 冻结实际算出的 $\hat w$，用同样权重做**无中间舍入**的递推 | 全部二进制有理数 | **是**（Fraction 逐位） |
| **specified-exp** | 指定 $\mathrm{Exp}$ 为高精度正确舍入的 FP32 exp | 参考值本身是 FP32，故有理 | 比较是精确的；但**判定**哪个 FP32 才是正确舍入需要高精度 |
| **real-exp** | 参考 $\sum_i e^{x_i-m}$ 的实数值 | 含超越数 | 否，需 MPFR/区间 |

**specified-exp 与 real-exp 常被混掉，但性质不同。** 前者比较的是两个 **FP32 值**
（实际 $\mathrm{Exp}$ 的输出 vs 正确舍入的输出），差以 ULP 计、有理、可精确核对；
高精度只用在离线**确定**正确舍入的那个值，不进入比较本身。后者比较的是一个 FP32 值与
一个超越实数，$\hat w-\exp(\hat\Delta)$ 无理，只能高精度核到 N 位。

关键区分仍在：**只有 frozen-weight 能在不额外确定任何超越量的前提下，保住旧线
"Fraction 意义下逐位精确"这一方法论资产。** specified-exp 要先付出确定正确舍入的代价，
real-exp 则连比较本身都不再是精确的。

specified-exp 与 real-exp 之差 = $\mathrm{Exp}$ 实现自身的误差，是一个**可测量**
（实测 ULP 分布），不是一套新的验证机械。三种 reference，两个测量问题。

## 4. frozen-weight 加权恒等式

冻结全部 $\hat w$，令叶取计算值 $\tilde\ell_{\rm leaf}=\hat\ell_{\rm leaf}$，参考递推为

$$\tilde\ell_v=\tilde\ell_a\hat w_a+\tilde\ell_b\hat w_b\quad(\text{精确有理运算}).$$

记 $e_v=\hat\ell_v-\tilde\ell_v$。代入 §2 的第 4、5 步：

$$
\hat\ell_v=(\hat\ell_a\hat w_a+\mu_a)+(\hat\ell_b\hat w_b+\mu_b)+\alpha_v
$$

两式相减：

$$
\boxed{\;e_v=\hat w_a e_a+\hat w_b e_b+(\mu_a+\mu_b+\alpha_v)\;}
$$

误差以**计算权重** $\hat w$ 为系数向上传播，并在本节点累加局部残差。叶上 $e=0$，展开得

$$
\hat\ell_{\rm root}-\tilde\ell_{\rm root}=\sum_{v\ \text{internal}}W_v\,(\mu_{a(v)}+\mu_{b(v)}+\alpha_v),
\qquad W_v=\prod_{e\in\text{path}(v\to\text{root})}\hat w_e .
$$

**必须用 $W_v=\prod\hat w$，不能用解析的 $e^{m_v-m_{\rm root}}$。** 后者靠指数相加 telescoping，
而实际权重逐层各自舍入过；两者相差每层若干 ULP。恒等式的负控制（§9）就是验证解析权重版本逐位失败。

$W_v$ 里的连乘本身是**精确有理乘积**，不是逐层再舍一次的浮点积。被冻结的是每一个因子
$\hat w_e$（它们各自是 FP32），而把它们乘起来这一步属于 CPU 侧的分析，不对应 kernel 里的
任何运算——**没有任何硬件会计算 $W_v$**。

$\eta$ 与 $\mathrm{Exp}$ 的实现误差**不出现在此式中**——它们已被吸收进冻结的 $\hat w$。
这既是该 reference 的局限，也正是它的可移植性来源（§6）。

## 5. 直接推论

- **胜方免费**：设 $m_v=m_a$，则 $\hat\Delta_a=0$ 精确，$\mathrm{Exp}(0)=1$，$\hat\ell_a\otimes1$ 精确，
  故 $\mu_a=0$。**每次 merge 至多一个乘法残差 + 一个加法残差**；$m_a=m_b$ 时两个 $\mu$ 都为 0。
- **FMA 变体**：若 kernel 用 $\mathrm{fma}(\hat\ell_b,\hat w_b,\hat\ell_a)$，则 $p_b$ 不单独舍入，
  **每次 merge 只剩一个残差**。FMA 收缩与否改变 $\delta$ 集合，是合同参数。
- **$W_v\le1$**：重标定对已传播误差只衰减不放大。这是零假设，不是研究假设。
- **stagnation 自然落在 $\alpha_v$ 上**：完全吸收即 $\hat\ell_v=p_a$，亦即 $\alpha_v=-p_b$。
  separate 路径中，加法丢掉了舍入后加数 $p_b$ 的**全部量级**，但仍满足半 ULP 舍入界；
  $p_b>0$ 时残差为负，乘法已将 $p_b$ 下溢为零时加法残差也为零。
- **吸收是连续的，不是二值的**：$\hat\ell_v\neq p_a$ 但已丢失 $p_b$ 大部分时，
  事后判据 $\mathrm{fl}(x+y)=x$ 为假而损失已发生。因此机制量只能是连续的 $W_v\alpha_v$，
  事件计数不足以充当机制量。
- **FP32 下 stagnation 可达**：设 $a$ 是 max 胜方、$x=\hat\ell_a>0$，使用 RN-even、
  gradual underflow 且不溢出。令 $q=\mathrm{ulp}(x)$ 为 $x$ 向上相邻 FP32 格点的间距。
  separate 路径取 $y=p_b=\mathrm{RN}_{32}(\hat\ell_b\hat w_b)$；FMA 路径取
  $y=\hat\ell_b\hat w_b$ 的**精确乘积**。两者完全吸收的精确条件都是
  $y<q/2$，或 $y=q/2$ 且整数 $x/q$ 为偶数。不能省略 tie，也不能在 separate 路径
  中用未经舍入的乘积替代 $p_b$。
  例如 $m_a=m_b=0$、$(\hat\ell_a,\hat\ell_b)=(2^{24},1)$ 时 $y=q/2=1$，
  两条路径仍返回 $2^{24}$；胜方改为 $2^{24}+2$ 时有效数字为奇数，结果向上舍入。
  当 $x$ 是正规数时，$\mathrm{ulp}(x)/x$ 随 $x$ 在 binade 内的位置在
  $[2^{-24},2^{-23}]$ 之间变动，比值阈值落在 $2^{-25}$ 到 $2^{-24}$ 之间。
  **只有再假定 $\hat\ell_a\approx\hat\ell_b$**，才能把它换算成
  $\Delta m\gtrsim24\ln2\approx16.6$——这个数是量级参考，不是判据。
  另注意分块下 $m$ 是块内最大值，$\Delta m$ 是**块间** max 之差，小于全局 logit 间距，
  故"块间 max 差的分布"是必须先测的量。

## 6. 硬件可移植性

frozen-weight reference 只消费 $\hat w$ 和 $\hat\ell$ 的**数值**，不关心它们由谁产生。
因此同一份 Fraction oracle 可以逐位核对：

- CPU 参考实现；
- 任何 $\mathrm{Exp}$ 实现（stdlib / libm / `expf` / `__expf` / `exp2f`）；
- **GPU kernel 的输出**——NVIDIA 上 FP32 的 $\oplus,\otimes,\mathrm{fma}$ 均按 IEEE-754 正确舍入，
  不正确舍入的只有 $\mathrm{Exp}$，而 $\mathrm{Exp}$ 的结果在本 reference 中是数据。

代价：需要 kernel 以 debug 变体把逐节点中间量写回 global memory。小 $n$ 下可接受。

**结论：上 GPU 不必失去逐位 oracle。** 这一点是 §3 的 frozen-weight 划分换来的。

**但恒等式本身不能验证 kernel 做了什么。** 残差按定义是 $\hat\ell_v-\text{exact}$，
所以 §4 的恒等式对**任意** $\hat\ell_v$ 都成立——它是记账一致性，不是"kernel 实现了 §2 递推"
的证明。因此 dump 回来之后必须**另外**核对自洽性：

1. $\hat w=\mathrm{Exp}(m_{\rm child}\ominus m_v)$，即 $\Delta$ 确实走了 FP32 减法而非精确相减；
2. $\hat\ell_v$ 等于按 §2 第 4、5 步重算的结果；
3. fused 变体确实只舍一次。

这三条在 CPU 侧由 `test_online_merge` 守住（是变异测试逼出来的——只有恒等式时，
"谎报 fused 实则舍两次"的变异存活）。GPU 侧必须重跑同样三条，不能因为恒等式过了就认为
kernel 是对的。

## 7. CUDA 可表达的 schedule 族

真实 kernel 的合并顺序不是自由设计，由编程模型决定：

| 层级 | 机制 | 形状 |
| --- | --- | --- |
| warp 内 | `__shfl_xor_sync` / `__shfl_down_sync` | 平衡树，深度 5 |
| block 内跨 warp | shared memory，$\le32$ 个 partial | 小平衡树或短顺序链 |
| K/V 块之间 | FlashAttention 主循环 | **严格顺序链**，长度 = 块数 |
| split-K / FlashDecoding | 跨 CTA 拆分后合并 | 把顶层链换成树或扁平合并 |

即：**块内平衡树 + 块间顺序链**，可选 split-K 顶层合并。这正是 two-stage blocking。

这对旧线的一个批评是直接回应：不再对 64 棵**任意**树排序，
而是研究 **CUDA 实际能表达的那一小族 schedule**。该收窄不花任何成本。

## 8. 需要硬件 vs 不需要硬件

| 需要 NVIDIA GPU | 不需要 |
| --- | --- |
| `expf`/`__expf`/`exp2f` 实测 ULP 分布 | schedule 族的枚举与精确模拟 |
| `-ftz=true`（`--use_fast_math`）的实际行为 | frozen-weight 恒等式与全部通道归因 |
| 编译器 FMA 收缩的实际决策 | block size / split 数 / 树形的误差几何 |
| 性能—误差折衷 | 块间 max 差分布、logit 散布扫描 |

右列是第 0+1 格的全部内容，**本机 CPU 即可完成**。
左列可压缩成一个短的硬件测量阶段：租用或 Colab 一张 NVIDIA 卡，
跑 dump kernel 取 $\hat w$ 与逐节点中间量，带回 CPU 用同一个 oracle 核对。
它是一个**小阶段，不是一个研究项目**。

## 9. 推导的 scratch 验证（非证据）

2026-09-04 在 scratchpad 中用重写 oracle 的 `round_to_fp32` 做了一次数值核对。
**这是对 §4 推导的 sanity check，不是实验证据，不入 `results/`，输入模型也不真实**
（块 max 取 $[-s,s]$ 均匀、块部分和取 $[1,4]$ 均匀，与真实 attention logit 分布无关）。

- **恒等式**：chain / balanced × spread $\in\{2,12,25\}$ × FMA 开关，共 12 组配置、
  每组 40 例、每例 64 个叶块 —— **全部逐位精确**（Fraction 意义下 $e_{\rm root}=\sum W_v\delta_v$）。
- **负控制**：把 $W_v$ 换成解析的 $e^{m_v-m_{\rm root}}$ 后恒等式失效，
  相对偏差最大 $10^{-9}\sim10^{-6}$。量级不大，但足以在逐位核对中表现为无法解释的残差——
  这正是 §4 强调必须用 $\prod\hat w$ 的原因。
- **顺带观察到的机制信号**（仅供设计参考，不构成任何结论）：完全吸收的 merge 占比

  | spread | chain | balanced |
  | ---: | ---: | ---: |
  | 2 | 0.000 | 0.000 |
  | 12 | 0.273 | 0.049 |
  | 25 | 0.604 | 0.321 |

  spread=2 时（max 间距 $\le4$）吸收为零，与 §5 的 $\Delta m\gtrsim16.6$ 阈值一致；
  同 spread 下 chain 的吸收率是 balanced 的 5.6× / 1.9×。

**用实现模块复核**：`experiments/online/fp32_signed.py` 完成后，同样 12 组配置用该模块的
`fp32_sub` / `fp32_mul` / `fp32_add` / `fp32_fma` 重跑，**仍全部逐位精确**。这一次
$\hat\Delta=m_a\ominus m_v$ 真正走了有符号减法，即下面这个缺口被实际用上并验证。

### 本次核对暴露的合同缺口

`rewrite/fp32_oracle.py` 的 `round_to_fp32` **只接受非负输入**——这是旧线"非负 FP32 归约树"
合同的直接产物。而 online 设定里 $m$ 是有符号 logit，$\hat\Delta=m_a\ominus m_v$ 也有符号。
RN-even 本身符号对称，扩展是平凡的，但**合同边界必须显式改写**，不能靠 wrapper 绕过。

## 10. 冻结前未决

1. 恒等式的**负控制**：验证解析权重 $e^{m_v-m_{\rm root}}$ 版本逐位失败，并量出失败幅度。
2. 叶的定义：块内 $\ell_{\rm leaf}$ 本身如何计算（顺序、是否也是子树），及其是否进入本合同。
   **首轮采用精确初始化的常值块，见 §12；一般块内计算仍留待后续。**
3. FTZ 与 gradual underflow 两种模式各自的合同实例。
4. dtype 分轴：输入 / $(m,\ell)$ 状态 / $\mathrm{Exp}$ 输出 / accumulator，主合同只实例化 FP32。
5. primary metric：v2 的 normalized regret 依赖大候选集，此处不适用。**首轮已定，见 §12。**
6. 研究假设与效应量门槛：**pilot 之后再冻结**，本文不预设。
7. **实验设计约束（已定）**：研究合并顺序时，必须固定同一批叶块、只改 schedule。
   同时改分块内容会把"顺序的影响"和"块本身不同"混在一起，任何差异都归因不了。
   §9 那个 max 排最前 / 最后的例子就是这个形式：同一批 33 个块，只换位置。

## 11. specified-exp reference 的实现状态

[fp32_exp.py](../experiments/online/fp32_exp.py) 实现 §3 的 specified-exp reference。
两条已定的合同细节：

- **精度不足时抛 `ValueError`，不自动升精度。** `precision` 是本次计算的固定预算，
  预算内无法证明就报告失败，由调用方决定是否重试。自动升精度也可以是确定性的算法；
  这里选择固定预算。`exp(0) = 1` 等已有精确证明的特殊值可以直接返回。
  `precision` 必须是正整数（不接受 bool），校验必须先于任何提前返回。
- **高精度计算不得继承调用方或 `DefaultContext` 的 decimal 设置。** 使用字段完整的
  独立 `Context`，显式固定精度、指数范围、舍入模式与异常设置等。
  输入用 `Decimal.from_float` 精确转换，避免触发外层 `FloatOperation` 或改变其标志。

## 12. Pilot 首轮设计（已定）

### 叶块：块内零舍入

第 $b$ 块放 $n_b$ 个**相同**的 logit $c_b$。于是

$$m_b=c_b,\qquad \ell_b=\sum_{j=1}^{n_b}e^{c_b-c_b}=n_b,$$

而正整数 $1\le n_b\le2^{24}$ 时 FP32 累加 $n_b$ 个 1 **逐位精确**（已实测 $n_b=32$）。
这样块内不产生任何舍入，首轮就专答一个问题：**同一批块，换合并顺序会怎样。**

限制要写明：所有 $\ell_b$ 若取同一个 $n$，叶输入的可变部分就只有各块的 max；
合并结构、顺序和中间累计的 $\ell$ 仍影响误差，不能仅凭累计 max 的轨迹预测结果。
例如每块 $n_b=32$，顺序链的块 max 分别取
`(0, -1, -1, -1, -16, -16, -16, -16)` 与
`(0, -16, -16, -16, -16, -1, -1, -1)`。
两次执行所有内部节点的 max 都是 0，非 fused 的最终分母却分别为
`67.31642150878906` 和 `67.31643676757812`，相差 2 ULP。
这是说明 max 轨迹不足以决定误差的可复现反例，不是 pilot 的效应估计。

让 $n_b$ 随块变化可以在不破坏精确性的前提下加第二根轴。
若比较的是块间质量的重新分配，应固定块数和总元素数 $\sum_b n_b$。
块内元素不同的一般情形推后，届时须固定一套块内算法、**算一次存下来给所有 schedule 共用**。

### primary metric：对**实数** exp 参照的相对误差

$$\ell_*=\sum_b n_b\,e^{c_b-M},\qquad M=\max_b c_b,\qquad
A_T=\frac{|\hat\ell_T-\ell_*|}{\ell_*}.$$

比较两个 schedule 时取同一输入上的配对差 $A_{T_1}-A_{T_2}$ 再汇总。
它不依赖候选集的最好/最坏值，两个候选也能用。

**$\ell_*$ 必须用实数 exp 的区间求和**，指数取 $c_b-M$ 的**精确有理差**。
不能把正确舍入的 FP32 exp 加起来当作精确分母——那是 §3 的 specified-exp 量，
自带量化，不是真值。区间机械已存在（`test_online_fp32_exp` 的
`_independent_exp_interval`，精确 Fraction 泰勒 + 反复平方），
但它的折半次数和精度是为有限测试集固定的，搬进模块时要按参数范围和误差预算调整。
实数参照不能沿用 FP32 exp 的 `argument <= -104` 直接归零规则：真值仍为正；
若截断极小尾项，必须把其上界计入总区间。区间精度不足以判断误差或配对差时，
应提高参照精度或明确报告未判定，不能把区间中点当作已证明的真值。

### frozen_weight_reference 不能当尺子

它的参考值**随 schedule 的计算权重改变**——同一批 8 个块实测：

| schedule | $\hat\ell$ | frozen_ref |
| --- | ---: | ---: |
| sequential_chain | 33.684009552 | 33.684008476 |
| balanced_pairwise | 33.684005737 | 33.684008484 |
| split_k_4 | 33.684009552 | 33.684008476 |

上表保留原 scratch 记录，本节尚未列出那 8 个块的原始输入，不能凭此表重放该样例。
正式 pilot 必须保存叶输入的 FP32 位模式、块大小、schedule、exp/FMA 设置和生成种子。

所以它是**每次执行各自的分解基准**，用来回答"这次的误差来自哪些乘加节点"，
不能用来回答"哪个 schedule 的最终分母更准"。后者只能对共同的 $\ell_*$ 比。

### exp 实现

`merge_reduce` 的默认 `exp_impl` 已从 `provisional_fp32_exp` 改为
`correctly_rounded_exp`（§3 的 specified-exp reference）。`provisional_fp32_exp` 保留，
但它现在是**被测对象**而不是参照。
