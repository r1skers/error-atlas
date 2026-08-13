# Next Session

记录日期：2026-08-12

## Resume point

- Active topic：Softmax
- Completed milestone：P0-P4 exact graph oracle; P5 archived negative result
- Topic summary：topics/softmax/README.md
- Oracle：topics/softmax/experiments/summation_graph_predictor.py
- Negative-result artifact：topics/softmax/experiments/results/depth_margin_topology_challenge_v1/
- Registry：TOPICS.md

## 已建立的核心结论

- 对二维线性映射 \(A=\operatorname{diag}(3,0.5)\)，相同长度的误差会因
  方向不同而具有 \(0.5\) 到 \(3\) 的放大倍数；
- operator \(2\)-norm 只保留最大奇异值，完整 SVD 还需要 singular
  vectors 才能同时给出倍率与方向；
- 非线性映射的 Jacobian 是位置相关的局部线性传播器，并带有 Taylor
  remainder；
- 二分类 Softmax Jacobian 为

\[
J_s(\mathbf z)
=p_1p_2
\begin{pmatrix}
1&-1\\
-1&1
\end{pmatrix},
\]

  其共同平移方向奇异值为 \(0\)，contrast direction 奇异值为
  \(2p_1p_2\le1/2\)；
- subtract-max 利用精确 shift invariance 避免正指数 overflow，但不能恢复
  已在输入量化中丢失的相对差异；
- 分开计算 softmax 后再取 log 可能破坏解析抵消，融合 cross-entropy 保留
  \(\nabla_{\mathbf z}L=\mathbf p-\mathbf y\)；
- FP32 实验验证：\(M=2^{23}\) 时单位差仍保留，\(M=2^{24}\) 时 stored
  difference 变为零，第一概率从约 \(0.731\) 跳到 \(0.5\)；
- 量化与中心化一般不交换：

\[
Q\!\left(\mathbf z-m\mathbf1\right)
\ne
Q(\mathbf z)-\max(Q(\mathbf z))\mathbf1;
\]

- closed-book rewrite 独立恢复 quantize-before-stabilize 机制，并与原实现
  在三个注册 probe 上完全一致。

### 多分类方向性与全局界

- 任意类别数的 Jacobian 为

\[
J_s=\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^T,
\qquad
J_s\mathbf1=0;
\]

- 二次型可写成概率加权方差：

\[
\mathbf v^TJ_s\mathbf v
=\operatorname{Var}_{i\sim p}(v_i)
=\frac12\sum_{i,j}p_ip_j(v_i-v_j)^2;
\]

- 三分类均匀点在二维 contrast plane 内各向同性，两个非零特征值均为
  \(1/3\)；在 \(p=(1/2,1/4,1/4)\) 处，方向
  \((2,-1,-1)^T\) 与 \((0,1,-1)^T\) 的特征值分别为 \(3/8\) 与
  \(1/4\)；
- \(3/8\) 是固定点的局部 operator norm；跨所有概率分布的紧上确界是
  \(1/2\)，二分类均衡点达到，多分类可在两类各占一半的边界上逼近；
  共同平移与饱和说明不存在正的全局下界；
- 固定 Jacobian 只描述起点的局部变化，有限扰动需使用沿路径变化的
  Jacobian；全局 \(1/2\)-Lipschitz 界则可直接控制有限输入扰动。

### 浮点误差传播

在 shifted logits 已给定、无下溢且相对误差模型成立时，定义

\[
\widehat q_i=q_i(1+\epsilon_i),
\qquad
\bar\epsilon=\sum_jp_j\epsilon_j.
\]

共同 exp 相对误差经 normalization 消掉，只留下
\(\epsilon_i-\bar\epsilon\)。加入求和误差 \(\eta\) 和除法误差
\(\delta_i\) 后，

\[
\frac{\widehat p_i-p_i}{p_i}
\approx\epsilon_i-\bar\epsilon-\eta+\delta_i.
\]

若 \(|\epsilon_i|\le\alpha u\)，则

\[
|\epsilon_i-\bar\epsilon|
\le2(1-p_i)\alpha u.
\]

普通顺序求和的标准界为

\[
\gamma_{n-1}
=\frac{(n-1)u}{1-(n-1)u}
\approx(n-1)u,
\]

固定树形求和的误差量级预期为 \(O((\log n)u)\)。受控 layout 实验已经
验证具体 reduction graph 会改变哪一组输入失败；该量级界不保证任意输入上
pairwise 都比 sequential 更准确。

概率总量偏差满足

\[
\sum_i\widehat p_i-1
\approx-\eta+\sum_i p_i\delta_i.
\]

所以“概率和为 \(1\)”只是准确性的必要非充分检查；exp 差异误差可以在
单纯形切空间内重新分配概率而不改变总和。

- exp 相对误差可重写为等效 logits 扰动
  \(\Delta x_i=\log(1+\epsilon_i)\)，从而与 Jacobian 传播闭环；
- 求和或除法误差可能使输出离开概率单纯形，不能全部解释成某个 logits
  扰动下的精确 Softmax；
- 下溢时 \(\epsilon_i=-1\)，小相对误差模型失效：绝对误差可能很小，
  分量相对误差却为 \(100\%\)；
- \(\|J_s\|_2\le1/2\) 是问题条件性；\(\epsilon_i,\eta,\delta_i\)
  的传播界才是求值算法稳定性证据；
- 总误差需拆成 Softmax 对 stored logits 的求值误差，以及输入量化误差经
  精确 Softmax 的传播。原始 logits 的整体相对误差会被无关的共同偏移
  稀释，类别差值或 centered logits 是更合适的诊断量。

## 本轮收口

病态求和输入族与判定链已经扩展并重新整理：

1. CaseRecipe 冻结 generator version、全部生成参数、dtype、shape 和 layout；
   input hash 独立认证实际物化的有序 FP32 bytes；
2. power-tail 与 decimal-tail 均支持 head_then_tail / tail_then_head，
   generator name/version 共同决定 observer dispatch；
3. decimal-tail 将十进制 source 精确值、stored FP32 值、input quantization
   error 与 reduction error 分开；stored exact sum 只有在能被 binary64 精确
   表示时才进行一次 FP32 target rounding；
4. 正确舍入采用 IEEE 754 默认 round-to-nearest, ties-to-even。小规模
   \(N=1,e=-24\) 精确 tie 返回 0x3f800000；decimal \(t=10^{-8}\) 的
   \(N=5/6\) 分别位于 midpoint 下/上；
5. 放大边界族固定 \(e=-34\)，以 \(N=1023/1024/1025\) 覆盖
   below/tie/minimal-above，target 分别为
   0x3f800000、0x3f800000、0x3f800001；
6. 在 \(N=1025,e=-34\) 上，sequential FP32 仅 head-first 错一位，当前
   fixed pairwise tree 在两个 layout 都错一位，Kahan FP32 与 FP64
   accumulator 在两个 layout 都命中 target；
7. consumer_tolerance 与 correct_rounding 是两个具独立 policy_id 的 policy。
   \(N=6,t=10^{-8}\) 的错误一 ULP 输出可通过 \(10^{-6}\) tolerance，但会
   失败 correct-rounding，因此 summary 保持 policy-free；
8. suite v3 的 smoke tier 为 16 cases。默认 repeat=3 时得到 192 raw、
   64 summary、128 assessment rows；opt-in stress 再加入
   \(N=2^{20},e=-24\) 的两个 layouts；
9. artifact schema v2 新增 case 与 assessment CSV，canonical recipe/policy
   JSON 可分别重算 case_id/policy_id，CSV schema 不再静默忽略字段；
10. 公共 helper、materializer、candidate source hash 与测试索引已经整理；
    当时 80 项测试通过。

仓库中 results/failure_triage 的 2026-08-08 文件仍是 suite v1 /
schema v1 的旧 snapshot（36 raw、12 summary），本轮没有覆盖。它不能作为
expanded suite 的证据；需要替换时，先输出到新的 scratch directory 审查。

## 本轮新增：predictor definition 与 stepwise audit

对 ordered stored FP32 leaves 与显式 proper binary addition tree $G$，已定义

\[
a_v=y_{\ell(v)}+y_{r(v)},
\qquad
y_v=\operatorname{RN}_{32}(a_v),
\qquad
\rho_v=y_v-a_v,
\]

以及运行前可计算、可直接证伪的 signed forward-error predictor

\[
E_G(x)=y_{\mathrm{root}}-\sum_i x_i=\sum_{v\in G}\rho_v.
\]

实现使用 exact `Fraction` 与整数 ties-to-even，不调用 NumPy 浮点加法；先
产生 predicted bits / signed error，再运行 candidate。本轮重新按以下台阶
逐项审查：

1. P0：区分 source sum、$S_{\mathrm{leaf}}$、correctly-rounded target、
   predicted root 与 $E_G$；证明 proper pure-addition tree 中每个 $\rho_v$
   以系数 $+1$ 传到根；
2. P1：手算并定向测试 normal odd/even tie、subnormal tie、normal 接缝与
   binade carry；4 个 RN32 tests 通过；
3. P2：审查 $(N,e)=(9,-27)$ tail-first minimal graph discriminator，再只翻转
   layout；
4. P3：审查 $e=-27$、tail-first 的 $N=7/8/9$ boundary triple，再复现
   $(N,e)=(129,-31),(2049,-35)$ 两个 above-midpoint scale points；
5. accepted selector 共 12 行 graph observations，predicted bits 与 signed
   error 均命中 actual；
6. 明确 `prediction_matched_observation` 与 candidate correct-rounding 是两个
   判断。predictor 可以命中一个 incorrect output；当前 batch CSV 只版本化
   prediction matching，没有版本化 correct-rounding assessment；
7. 2026-08-12 状态审计报告当前完整 Softmax 实验测试为 93 项通过；本次文档
   纠偏没有再次运行测试，且测试数量本身不决定 evidence status。

原 36 行 scaled-midpoint CSV 保持原样；36/36 match 是 raw observation。除上述
12 行 selector 外的 24 行降为 **provisional batch replication**，不作为一个
已经完成的研究台阶。artifact 目录中的 README 记录了 selector 与引用边界。
当前 CSV 的 `preregistered_sum_bits` 是 graph-specific predicted output，不是
correctly-rounded target。

随后交互检查的两组非均匀正输入排列澄清了 local residual 与全图 $E_G$ 的
区别，以及相同 multiset 的顺序敏感性；但它们没有进入版本化 artifact，只是
interactive pilot，不能追溯性地称为 preregistered validation evidence。

## P4：single preregistered nonuniform case

另选一个未执行的新 case：

\[
u=2^{-27},
\qquad
x=(4u,2u,2u,u,1),
\qquad
S_{\mathrm{leaf}}=1+9u.
\]

先冻结 exact leaves、两张显式 node graph、predicted bits、$E_G$、correct target
与 correct-rounding prediction。preregistration SHA-256 为
`0c8047f38363c02ef6a6995bcc58a3f890dfebe7395aa5eb62a5cb671d6e47a6`，执行后未
修改。随后每张图只执行一次：

| Graph | Predicted / actual bits | Predicted / actual $E_G$ | Prediction match | Correct rounding |
|---|---|---|---|---|
| sequential left-to-right | `0x3f800001` | $+7u$ | yes | pass |
| balanced contiguous floor-half | `0x3f800000` | $-9u$ | yes | fail |

observation CSV、metadata、input hash、prereg hash、artifact hash 与 source hashes
均已复核一致。该结论只覆盖一个 nonuniform positive case，不外推到一个 family。

每张图只执行一次，因此 repeatability 为 **not measured**。single observation
足以在不匹配时证伪 frozen prediction，但不能建立 bitwise repeatability。已有
failure-triage runner 已定义 repeated raw rows、bit counts、summary 与 policy
assessment；若 consumer 要求该属性，应把 case 接入已有管线，不新建重复的 P5
repeat runner。未来 regression tests 的额外调用也不是版本化 repeatability
artifact。

当前 predictor 只覆盖非负 finite stored FP32 leaves、无 overflow、每个 binary
addition 一次 RN-even 舍入和 proper tree。它是 agent-authored semantic oracle；
不声称 learner closed-book mastery，也不能用于推断未知 black-box graph。

## P5 negative result：depth-margin topology counterexample

在 exact graph oracle 已收口后，另行定义 cheap screening score

$$
D_G=\sum_i d_i|x_i|,
\qquad
R_G=\frac{\varepsilon_{32}D_G}{M}.
$$

它只用于运行前排序，不是概率、threshold、rigorous error bound 或 safety
certificate。最初冻结的 head-first/head-last family 在执行前即被识别为重复
已知 head-depth effect，因此 preregistration 保留但降级为
known-mechanism calibration / optional replication；没有运行 oracle 或 candidate。

随后冻结 adversarial topology challenge：

$$
q=2^{-29},\qquad x=(1,32q,q/2,q/2),
\qquad S_{\mathrm{leaf}}=1+33q,\qquad M=q.
$$

两张 explicit 满二叉树的 leaf depths 均为 $(2,2,2,2)$，所以
$D_G=2(1+33q)$ 且
$R_G=536870945/8388608$ 完全相同；唯一变化是 sibling grouping。
preregistration SHA-256 为
`63fbe76d7af9f5df128c1fdfade6ca49fc1d4e26c472ea9914888a3ee28f9949`。

runner 先从 raw edges 独立重算 depths、$D$、固定 near-one margin 与 $R$，确认
proxy tie 后才调用 exact semantic oracle。结果为：

| Graph | $R_G$ | Oracle bits | $E_G$ | $F_G$ |
|---|---:|---|---:|---:|
| `midpoint_then_remainder` | `536870945/8388608` | `0x3f800000` | $-33q$ | 1 |
| `half_q_then_midpoint_plus_half_q` | `536870945/8388608` | `0x3f800001` | $+31q$ | 0 |

因此该 pair 是 `tied informative pair`，确认并触发预登记 falsifier，推翻
“每个 informative matched pair 都满足
$R_{\mathrm{failure}}>R_{\mathrm{correct}}$”的 universal ordering。结论边界：
depth 与 rounding margin 丢失 sibling grouping / rounding-phase 信息；该 artifact
不是 population ranking estimate，不推翻 exact graph oracle。没有执行 NumPy
candidate，也不声称 candidate conformance 或 repeatability。

该结果归档为 cheap baseline 的 confirmed negative result，不是整个
predictor-validation 阶段的最终里程碑。它没有测量 score 与连续 graph error 的
相关性、AUROC、failure prevalence 或实际 inspection-budget screening ability。

artifact 位于
`topics/softmax/experiments/results/depth_margin_topology_challenge_v1/`；metadata
已复核 prereg hash、2 行 CSV、相同 risk score、labels `{0,1}`、
`pair_outcome=tie`、`strong_hypothesis_falsified=true`、
`candidate_executed=false` 与 CSV hash。one-shot runner 拒绝覆盖现有 artifacts，
不要把它当作常规 regression command 重跑。

## 下一入口：统计 predictor validation 协议

P5 已以负结果收口，不再对 depth-margin baseline 做看过答案后的补丁式扩展。
`summation_graph_predictor.py` 统一定位为 exact semantic oracle / label generator，
与 cheap engineering predictor 明确区分。

下一台阶先只写协议，不新增 synthetic one-off case：

1. 冻结一个受控 stored-input 分布、明确 graphs、sample split 与 failure
   prevalence reporting；
2. 冻结连续 target（优先 local-ULP normalized $|E_G|$，同时保存 signed $E_G$）
   和分类 target $F_G$；
3. 冻结 cheap score 与统计 metrics，至少包括 Spearman 和 AUROC；failure 稀少时
   另报 PR-AUC 或 recall-at-inspection-budget；
4. 先完成受控分布验证，再决定是否把真实 GPT-2/attention leaves 作为新的数据
   阶段；仓库此前没有承诺该数据源；
5. 恢复 implementation-learning ownership：由用户先解释目标、写伪代码并主写
   最小 metric 实现，agent 只做审查、测试与 artifact scaffolding。

协议冻结前不运行新实验，不进入 GPU，也不扩展 negative cancellation、Kahan
操作图或未知 black-box graph。

## 仍暂停的 GPU 后续

GPU 实现阶段主动暂停。未来恢复时按以下顺序进行：

1. 学习 asynchronous execution、stream synchronization，以及为什么普通
   CPU timer 往往只量到 kernel launch；
2. 学习 thread/block/warp、global/shared/register memory 与资源约束；
3. 将 sequential、tree、warp/block reduction 映射到明确的 execution graph；
4. 再选择远程 NVIDIA CUDA 环境，先把库 reduction 当作 black-box baseline，
   只报告 bit pattern、误差、repeatability、latency/throughput；
5. profiler、源码或自写固定图 kernel 能提供证据前，不声称内部 reduction
   graph、workspace 或 occupancy。

迁移时复用同一 CaseRecipe、input hash 与 policy。实现方法、accumulator
dtype 和 block size 属于 ExecutionConfig；GPU 型号、driver、CUDA 与框架版本
属于 EnvironmentSnapshot。同一输入换 kernel 时 case_id 不变、config_id
改变；同一 kernel 换硬件时 config_id 不变、environment_id 改变。若自动调优
改变实际 block size，则 config_id 也必须改变。

暂存支线：Softmax—谱分解—entropy/Fisher/KL 的博客；用 Taylor truncation
近似 exp 所引入的新误差与成本权衡。

## 验证命令（PowerShell）

    python -m unittest discover -s topics/softmax/experiments -p "test_*.py" -v
    python topics/softmax/experiments/fp32_shift_resolution.py
    python topics/softmax/experiments/summation_graph_predictor_validation.py
    python topics/softmax/experiments/nonuniform_graph_predictor_v1_runner.py
    python topics/softmax/experiments/softmax_failure_triage_runner.py --include-stress --output-dir "$env:TEMP\error-atlas-softmax-triage-v3"

其中 `summation_graph_predictor_validation.py` 只用于复现保留的 36 行
provisional batch，不是下一研究台阶。下一台阶必须先写一个新 case 的 prediction
record，再定向执行该 case。

Taylor 第一轮验证命令仍保留在 topics/taylor-expansion/experiments/README.md。
