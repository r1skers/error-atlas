# Next Session

记录日期：2026-07-28

## Resume point

- Active topic：Taylor expansion
- Current checkpoint：CP06 — Error Control and Optimization
- Entry file：topics/taylor-expansion/CP06_control_and_optimization.md
- Code file：topics/taylor-expansion/experiments/statistical_noise.py

## 已建立的核心结论

- Taylor remainder 的表示、渐近阶、上界和 bound tightness 已完成第一轮；
- 数值微分中的截断误差、舍入误差与稳定表示已经连接；
- 中心差分的确定性偏差为 \(h^2/6+O(h^4)\)；
- 相关观测噪声经过差分和 \(N\) 次平均后的方差为

\[
\frac{\sigma^2(1-\rho)}{2Nh^2};
\]

- 总 MSE 可写成偏差平方与随机方差之和；
- 最优步长来自两项斜率抵消，而不是两项数值相等；
- Monte Carlo 实验验证了 \(h^{-1}\) 与 \(h^2\) 两个主导区及最优步长；
- \(\rho=1\) 时观察到的微小残余来自浮点求值顺序，而不是相关噪声模型失败。

## 下次从这里开始

完成一次 closed-book rewrite：

1. 不查看原实现；
2. 从空白写出 correlated_noise_pair 或 theoretical_metrics；
3. 说明函数必须保持的 invariant；
4. 用 experiments/test_statistical_noise.py 与原实现比较；
5. 将差异与经验写回 CP06。

完成后将 CP06 标为完成，并正式关闭 Taylor expansion 第一轮。下一阶段从向量扰动、Jacobian 与方向性放大进入 Softmax。

## 验证命令

    python -m unittest discover -s topics/taylor-expansion/experiments -p "test_*.py" -v

## Workflow reminder

closed-book rewrite 前保留当前工作副本，避免覆盖唯一实现。建议在独立 rewrite 文件中完成，再比较差异。
