# Error Atlas

> An experiment-driven study of approximation error, propagation, numerical stability, and error control.

Error Atlas 是一个按研究主题组织的个人研究项目。它不把“误差”当成公式末尾的一项，而是持续追问：

\[
\text{误差如何定义}
\rightarrow
\text{从哪里产生}
\rightarrow
\text{怎样传播}
\rightarrow
\text{如何估计与控制}
\rightarrow
\text{怎样在精度和成本之间优化}.
\]

不同 topic 可以来自数学、数值计算或机器学习；统一它们的是同一套误差分析框架。

## 当前里程碑

Taylor expansion 第一轮已经完成主要推导与实验，覆盖：

- Lagrange、integral 与 Peano remainder；
- actual error、asymptotic order 与 error bound 的区别；
- bound validity 与 tightness；
- Richardson extrapolation 与 observed order；
- 截断误差、cancellation、roundoff 和稳定表示；
- 确定性误差预算与统计 bias–variance 模型；
- 带相关噪声的中心差分及最优步长。

![Noisy central-difference error curve](topics/taylor-expansion/experiments/results/statistical_noise_error.png)

图中左侧由随机噪声的 \(h^{-1}\) 放大主导，右侧由中心差分的 \(h^2\) 截断偏差主导；理论与 Monte Carlo 结果在最优步长附近吻合。

闭卷重写已经验证核心相关噪声构造可以从空白恢复，Taylor expansion 第一轮正式完成。下一候选主题是 Softmax 的扰动传播与有限精度稳定性。

## 仓库结构

    error-atlas/
    ├── README.md
    ├── TOPICS.md
    ├── NEXT_SESSION.md
    ├── requirements.txt
    ├── framework/
    │   ├── error_analysis_protocol.md
    │   └── implementation_learning_protocol.md
    └── topics/
        └── taylor-expansion/
            ├── README.md
            ├── CP00_orientation.md
            ├── ...
            ├── CP06_control_and_optimization.md
            └── experiments/
                ├── README.md
                ├── finite_difference.py
                ├── statistical_noise.py
                ├── test_statistical_noise.py
                └── results/

- [TOPICS.md](TOPICS.md)：主题注册表与候选方向；
- [error analysis protocol](framework/error_analysis_protocol.md)：每个案例共用的研究循环；
- [implementation learning protocol](framework/implementation_learning_protocol.md)：核心算法由学习者主写的协作规则；
- [Taylor expansion](topics/taylor-expansion/README.md)：当前主题的完整检查点；
- [current resume point](NEXT_SESSION.md)：下一次从哪里继续。

## 快速复现

需要 Python 3.10 或更高版本。

    python -m pip install -r requirements.txt
    python topics/taylor-expansion/experiments/finite_difference.py
    python topics/taylor-expansion/experiments/statistical_noise.py
    python -m unittest discover -s topics/taylor-expansion/experiments -p "test_*.py" -v

两个实验都会把 CSV、metadata 和 PNG 写入 topics/taylor-expansion/experiments/results。

## 研究约定

每个 topic 都先确定 reference、metric、assumptions 和 error sources，再进入界、传播、控制与验证。代码实验要求：

1. 运行前写出方向、尺度、边界与失败特征；
2. 保留原始数据与运行元数据；
3. 区分理论误差、实现误差、浮点误差与测量噪声；
4. 用测试覆盖核心 invariant；
5. 不把“看懂实现”当作“能够独立写出实现”。

实验结果文件会进入版本控制：CSV 保存证据，JSON 保存 provenance，PNG 让关键结论可以直接检查。
