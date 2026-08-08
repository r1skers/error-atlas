# Next Session

记录日期：2026-08-08

## Resume point

- Active topic：Softmax
- Completed milestone：failure–metric–policy–mitigation chain and summation stress suite
- Topic summary：topics/softmax/README.md
- Experiment：topics/softmax/experiments/softmax_failure_triage_runner.py
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

树形求和预期为 \(O((\log n)u)\)，但尚未完成对应实验。

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

## 本轮收口与下一阶段

“发现问题后怎么办”的第一条完整链已经完成：

1. 按 stage 冻结实际输入和局部 reference，不让上游误差混入归因；
2. 用 raw observation、RunSummary、consumer policy 与 RunAssessment 分离
   事实、指标、容差和判定；
3. repeatability 与 accuracy 独立，failure 与 non-blocking warning 分开；
4. 直接构造 \(q=(1,2^{-24},\ldots,2^{-24})\)，注册 smoke/stress case；
5. 在 \(2^{20}\) 尾项上验证 sequential FP32 丢失 \(1/16\) 总质量，误差
   为 \(1/17\)，而 pairwise FP32、compensated FP32 和 FP64 accumulator
   均返回精确 FP32 结果 \(17/16\)；
6. raw 36 行、summary 12 行及 metadata 已生成并经独立表格工具检查。

下一入口：扩大输入族（不同 layout、非二进制友好尾项与更多 consumer
tolerance），然后在目标硬件分开测量 latency、throughput、workspace 和
实际 reduction graph。CPU Python 原型的速度不能替代 GPU 性能证据；迁移
时复用同一 CaseRecipe、input hash 与 policy，只改变 config/environment。

暂存支线：Softmax—谱分解—entropy/Fisher/KL 的博客；用 Taylor truncation
近似 exp 所引入的新误差与成本权衡。

## 验证命令

    python -m unittest discover -s topics/softmax/experiments -p "test_*.py" -v
    python topics/softmax/experiments/fp32_shift_resolution.py
    python topics/softmax/experiments/softmax_failure_triage_runner.py --include-stress

Taylor 第一轮验证命令仍保留在 topics/taylor-expansion/experiments/README.md。
