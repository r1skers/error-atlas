# Online 归因：用户实现与审计学习

这份学习记录先于已保存扩展 pilot 的分量扫描建立。
用户预测原话：**“merge 的带权舍入残差逐渐累积”**，记录时分量数据尚未查看。
核心通过审计后，用户要求继续，现已完成[归因与预测对照](online_attribution_v1.md)。

## 你要实现什么

打开 [attribution.py](../experiments/online/attribution.py)，先补 Explain-back，再依次写：

1. `decompose(family, dump)`：复用已有函数，返回带符号分解、节点项、舍入预算与归一化区间。
2. `cancellation_savings(R, [w_low,w_high])`：给出两个分量之间抵消量的精确范围。

两段核心已由用户实现，函数签名、数据结构、输入检查和测试由 agent 提供。
以下合同仍用于审计和闭卷复写；运行前记录与源码快照已随归因结果保存。

## 第一段要守住的数学合同

$$E=\hat\ell-\ell_* = R+W,\qquad
R=\hat\ell-\ell_{frozen},\qquad W=\ell_{frozen}-\ell_*.$$

- `frozen_weight_reference(dump)` 返回精确有理数，不是 real-exp 真值。
- `weighted_residuals(dump)` 返回节点顺序的 $W_v\delta_v$。必须独立核对
  $R=\sum_v W_v\delta_v$，不一致时抛 `ArithmeticError`。
- $B_R=\sum_v|W_v\delta_v|\ge|R|$。预算大而净误差小，说明 R 内部也有抵消。
- `denominator_interval(family)` 给出同一个 $\ell_*\in[L,U]$。原始 E、W 区间保留符号；
  转成 $R/\ell_*,W/\ell_*,E/\ell_*$ 时，也必须保留这个共同参照的依赖关系。
- 相对 W 的分子含 $\ell_*$，不能把分子区间与分母区间当成独立变量再要求结果是紧的。
  R 则是固定数，但可能为负，除以正区间时上下界的方向必须想清楚。
- E、R、W 的原始值单位是“分母质量”，归一化后才可跨块数比较。不能只凭原始 R
  变大解释相对误差 A 增长；真实分母自身也随输入和块数变化。

这里的 W 包含差值舍入、exp 与权重沿树复合的影响，不能直接叫“纯 exp 误差”。
两种 schedule 的 frozen reference 不保证相同。现有输入检查只确认形状、同族和基本范围，
不替代未来硬件 dump 的逐算子审计。

## 第二段要守住的数学合同

固定 R，对 $w\in[w_{low},w_{high}]$ 定义

$$S(w)=|R|+|w|-|R+w|.$$

返回整个区间上 S 的最小值和最大值。先自己画出 R>0、R=0、R<0 三种图，
标出 $w=0$ 和 $w=-R$。同号、异号、完全抵消以及抵消饱和都要覆盖。
S 是“两分量绝对值之和因抵消而减少的量”，不是抵消掉的单侧质量，也不是新的 primary。
例如一边 +2、一边 −2，单侧各损失 2，而上式 S=4。

这一步只算 R 与 W 之间的抵消；R 内部各节点之间的抵消由 $B_R$ 与 $|R|$ 区分。
两者不要重复记账。归一化区间有共同依赖，不能要求分量区间的端点分别相加后恰好
等于最紧的总误差区间；实际每个共同 $\ell_*$ 上的恒等式仍成立。

## 怎么验收与审计

```sh
python tools/run_tests.py --suite softmax -p test_online_attribution.py -v
```

16 项测试全部通过，无 skip。测试包含：已知 max-position
反例的正负 R、独立 Decimal real-exp 包含性、FMA 两 arm、单叶树、损坏恒等式拒绝、
刻意加宽的共同参照区间、跨零与抵消饱和，以及非二进制可精确表示的有理数。
只允许初始化探针因核心未写而 skip；探针通过后，边界分支的 `NotImplementedError` 会报错。

已检查符号、参照类型、归一化依赖和数据归属，对实际交付实现设计的 9 个变异体全被杀死。
可复现命令（只在内存改变函数，不改源码或运行研究扫描）：

```sh
python tools/audit_online_attribution_mutants.py
```

[审计记录](../experiments/online/runs/attribution_audit_v1.json) 保存源码、测试和变异工具的 hashes
及每个变异体被哪些测试抓住。覆盖的错误为：R 取绝对值、净残差冒充预算、W 端点反转、
负 R 的归一化端点错误、丢失共同参照依赖、忽略恒等式失败、抵消公式符号错误、
遗漏抵消饱和，以及 frozen reference 冒充 real reference。这是针对这些错误的证据，
不是穷尽性正确性证明。

只修了一处工程问题：移除未使用的 `from sklearn import frozen`，避免无关依赖。
两个函数的数值实现保留用户版本。`cancellation_savings` 的端点加折点枚举正确。
现有 64 族原始结果、运行前预测与源码快照保持原样；新的归因输出另建目录并记版本。

用户已补 R 正负与两分量抵消的说明，后者宜称“两个误差分量”而非两个单独舍入。
目标等空白可后续作为学习反思补充，预测的量级/边界/失败表现原先未填写，不能用已见结果
倒写成事前预测。输入应理解为“同一组块，块内常值、块间 logit 可不同”；
不要求所有叶块的 logit 相同。再用自己的话说明 **为什么 W 不是纯 exp 实现误差**。
实现通过不等于这些解释或闭卷复写已完成。
