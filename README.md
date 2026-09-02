# Error Atlas

> An experiment-driven study of approximation error, propagation, numerical stability, and error control.

Error Atlas 是按主题组织的个人研究项目。主线是：定义误差 → 识别来源 →
理解传播 → 估计与控制 → 验证精度和成本。

## 从哪里开始

- [NEXT_SESSION.md](NEXT_SESSION.md)：当前状态与下一步；不再把历史日志当成待办。
- [TOPICS.md](TOPICS.md)：主题注册表。
- [Softmax 实验索引](topics/softmax/experiments/README.md)：按用途找代码。
- [结果索引](topics/softmax/experiments/results/README.md)：区分已确认结果、负结果与校准观察。
- [维护指南](docs/maintenance.md)：测试、目录职责、冻结证据与后续拆分边界。
- [完整知识谱](KNOWLEDGE_MAP.md)：从误差语言、浮点算术与 Softmax，到统计 predictor validation 和 GPU 数值实验的离线教材；

知识谱是学习资料，不是最新研究状态或新增实验结论的来源。

## 当前进度

| 主题 / 阶段 | 状态 |
| --- | --- |
| [Taylor expansion](topics/taylor-expansion/README.md) | 第一轮推导、实验与闭卷重写完成 |
| [Softmax 基础与 exact graph oracle](topics/softmax/notes/foundations.md) | 第一轮完成；早期证据等级保持不变 |
| Fixed-K8/B3 tree ranking | 已完成受控分布上的确认；推理成本仍高 |
| Offline tree reuse | 相对随机固定树有改善，但未通过 balanced-FP32 部署门槛 |
| Online risk certificate | 已完成校准；有统计信号，尚无确认或部署结论 |

最新方向是稀疏高能量节点的 exactness correction；设计与预算仍需重新冻结。
本轮整理没有开启新实验，GPU 阶段仍暂停。

## 目录职责

```text
framework/                 研究纪律与实现学习协议
docs/                      维护说明与历史续接记录
tools/                     仓库维护工具和它们的测试
topics/<topic>/
    README.md              主题入口
    notes/                 理论与历史研究笔记
    experiments/           实验源代码、协议和 results/ 证据
    tests/                 回归测试
```

原脚本入口与 results 路径保持稳定，测试已从实验目录分离。
前三个 coherence 诊断现在共用
[一份轨迹与分析接口](topics/softmax/experiments/reduction_analysis/README.md)；
旧函数和 CLI 保留兼容入口，旧源码版本仍可按 Git 记录复现。

## 开发检查

需要 Python 3.10+，依赖见 [requirements.txt](requirements.txt)。

```sh
python -m pip install -r requirements.txt
python tools/run_tests.py
python tools/run_tests.py --suite softmax -v
python tools/run_tests.py --suite softmax -p "test_predictor_fixed_k8_beam_inference.py"
```

测试命令不会调用实验 CLI 重写归档结果。不要把“快速复现”理解为批量重跑
one-shot runners；执行前先读对应结果目录的 README 与冻结协议。

## 研究约定

先确定 reference、metric、assumptions 和 error sources，再研究界、传播和控制。
遵循 [误差分析协议](framework/error_analysis_protocol.md) 与
[实现学习协议](framework/implementation_learning_protocol.md)：运行前记录预测，
保留原始数据和 provenance，区分实现、数值、测量与统计误差。
