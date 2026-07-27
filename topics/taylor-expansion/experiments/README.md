# Taylor Expansion Experiments

本目录把 Taylor 余项、误差传播和误差控制落实为两个可复现实验。

## 1. Finite-difference error

运行：

    python finite_difference.py

比较四种 \(e^x\) 在 \(x=0\) 处的数值微分实现：

- naive forward difference；
- stable forward difference using expm1；
- naive central difference；
- stable central difference using sinh。

实验记录真实误差、observed order、Richardson extrapolation 和 error estimate，并展示截断误差与 binary64 roundoff 形成的误差曲线。

输出：

- results/finite_difference_comparison.csv
- results/finite_difference_metadata.json
- results/finite_difference_error.png

## 2. Statistical noise

运行：

    python statistical_noise.py

对带相关 Gaussian 函数值噪声的中心差分做 Monte Carlo 验证。研究核心是

\[
\operatorname{MSE}(h)
=
\left(\frac{\sinh h}{h}-1\right)^2
+
\frac{\sigma^2(1-\rho)}{2Nh^2}.
\]

默认配置扫描 41 个步长，每个估计器平均 \(N=100\) 对观测，每个步长重复 \(M=2000\) 次。随机种子和运行环境写入 metadata。

输出：

- results/statistical_noise_comparison.csv
- results/statistical_noise_metadata.json
- results/statistical_noise_error.png

## Tests

从仓库根目录运行：

    python -m unittest discover -s topics/taylor-expansion/experiments -p "test_*.py" -v

测试覆盖相关噪声的极端情形、固定种子可复现性、无噪声极限、MSE 分解、经验—理论方差一致性和非法输入。

## Provenance policy

results 目录中的 CSV、JSON 和 PNG 是研究证据的一部分，默认提交到版本控制。Python cache、虚拟环境和本机工具状态由根目录 .gitignore 排除。
