# 对 W 自限说法的审计：全局单项不等于 online 权重路径

2026-09-06。审计用户转述的解释，采用现有正确舍入 CPU oracle 构造确定性反例。
这不是新采样轮、原创性声明或工程影响验证；不改变输出诊断优先的顺序。

## 结论

“当前 64 族的主要长链劣势已被普通求和重现”成立。
“在 CPU、正确舍入 exp、常值叶块下，W 被钉在与 n 无关的 O(u)”不成立。
R/W 可以类比计算误差/有效输入误差，但后者的有效输入含依赖树形的整条权重路径，
不能用一次全局 exp 的误差取代。还没有证据宣称新的主导机制或 GPU 是唯一可能的区别。

## 少掉的是路径上的 exp 舍入项

对叶 b 到根的一条路径，定义每条边的精确最大值增量 d_e≥0，
D_b=M−m_b=sum_e d_e。设实际减法输出为 −d_e+eta_e，实际 exp 为

```
w_hat_e = exp(-d_e + eta_e) * (1 + epsilon_e)
q_hat_b = product_e w_hat_e
        = exp(-D_b) * exp(sum_e eta_e) * product_e(1 + epsilon_e)
p_b = n_b * exp(-D_b) / ell_star
W / ell_star = sum_b p_b * (exp(sum_e eta_e) * product_e(1 + epsilon_e) - 1)
```

这是当前 frozen-weight 定义的直接展开；其乘积不再另外舍入。
在标准相对误差模型适用、权重为正常数且正确舍入的情况下，
|eta_e|≤u d_e、|epsilon_e|≤u。对足够小的路径误差作一阶展开，有

```
|W|/ell_star ≲ u * sum_b p_b * (D_b + k_b)
```

k_b 为该叶路径上可能有非零 exp 舍入误差的边数；精确零 gap 的 exp(0)=1 可排除。
上式是带条件的一阶上界，不是无条件精确不等式，不能略去高阶项及下溢情况。
sum_b p_b D_b≈1 的观察即使完全正确，也没有控制 sum_b p_b k_b。
同向 exp 舍入可沿路径累积；真实差值的望远镜相消不让实际舍入误差相消。

另外，D exp(-D)≤1/e 只约束单个未归一化项，并不保证对任意输入分布归一化后的
sum_b p_b D_b 是与 n 无关的常数。大量较低 logit 可以形成不可忽略的总质量。

## 同一 CPU 合同内的可复算反例

取 u=2^-24、delta=2^-26=u/4，n 个常值块，每块质量 32，
按 m_b=b delta（b=0,...,n−1）递增送入 chain。以下 n≤2048 时所有 logit 和相邻差
都精确可表示为 FP32，没有减法误差、溢出、下溢或 FTZ。

每次旧累计量的缩放权重是 RN32(exp(-delta))=1：
1 下方相邻 FP32 为 1−u，舍入中点为 1−u/2，而 exp(-delta)>1−delta>1−u/2。
新块权重 exp(0) 也是 1。因此 separate 与 FMA 两臂都只做精确的整数加 32，
最终 computed=frozen=32n，R=0。但真实分母为

```
ell_star = 32 * sum(j=0,...,n-1) exp(-j delta) < 32n
W / ell_star = n / sum(j=0,...,n-1) exp(-j delta) - 1
            = (n-1) delta / 2 + O((n delta)^2),  n delta << 1
```

用现有 `merge_reduce`、`frozen_weight_reference`、`denominator_interval` 实算：

| 块数 | R | W/ell_star（数值区间的近似显示） | 约多少 u |
| --- | --- | --- | --- |
| 32 | 0 | 2.3096801571e-7 | 3.8750 |
| 128 | 0 | 9.4622403191e-7 | 15.8750 |
| 512 | 0 | 3.8072515073e-6 | 63.8751 |
| 2048 | 0 | 1.5251415978e-5 | 255.8763 |

这里完全没有 merge 乘加舍入，增长来自正确舍入 exp 的路径误差。
它否定被转述的普遍自限结论，不否定原 64 族上 W 很小的观测。
它仍可用浮点乘积/路径误差分析解释，不能因此宣称发现了新的数值分析理论。

回归负控制同时检查 separate/FMA、32/128 块及 R=0、所有实际权重为 1、W 的严格区间：

```sh
python tools/run_tests.py --suite softmax -p test_online_weight_path_counterexample.py -v
```

## 与已有文献及 GPU 说法的关系

Blanchard–Higham–Higham (2019) §4 从单个全局平移项出发，得到
|w_hat_i−w_i|≤[(1+a−x_i)u+O(u²)]w_i；§6 也明确讨论大差值系数被小指数权重缓和。
这部分已经在论文中，不必称“可能在射程内”。但其 Algorithm 4.1 使用固定全局 a，
不能未经额外推导就当作任意 online 树的路径误差定理。
见[原论文 §4、§6](https://arxiv.org/pdf/1909.03469)。

“设备 exp 最大偏差 2 ULP，所以 mean |W| 从 0.1u 到 2u，变大 20 倍”不是有效推导：
单调用的最大误差与一批数据的带符号加权结果不是同一个统计量；还需参数分布、误差符号、
路径长度、相关性及局部 ULP 与相对误差的换算。GPU 可能值得测，但增长倍数尚未测得。
现有 NumPy 的 66.6%/2 ULP 记录只描述当时软件环境和样本，也不能代表 CUDA。

原解释还写“RN 每次≤u/2”：本项目 u=2^-24 已是 unit roundoff，标准相对界是≤u；
绝对舍入误差≤0.5 个局部 ULP 是另一种表述。

当前决定：不把错误的 O(u) 结论补成“已解释机制”，也不因此把 GPU 提到输出检查之前。
下一步仍先做有限 V 输出诊断；即便本反例中 W 很大，V=1 的同步通道仍会抵消。
