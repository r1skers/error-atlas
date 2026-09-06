# 扩展 pilot v1：64 个族的可重放 CPU 测量

2026-09-05，探索性数据，不是确认实验。输入和分析口径在运行前写入 `prediction.md`。

## 范围

- 每块 32 个相同 logit；块内 exp(0)=1 的累加精确，排除块内 reduction 误差。
- uniform spread 为 8、25；块数为 32、128、512、2048；每格 8 个显式且不重复的种子。
- 64 个输入族，每个族由 chain/balanced、separate/FMA 四个组合共用，得到 256 次测量。
- exp 为正确舍入的 FP32 reference；实数分母用严格区间。没有硬件 exp 或 GPU 测量。
- 初次完整测量约 255 秒；这是本机工程耗时，不是 kernel 性能证据。

## 数据与版本

[manifest.json](manifest.json) 记录配置、Python/平台、Git 状态、源码和文件 hashes。
运行时尚有未提交改动，精确执行版本以 `sources/online/*.py.txt` 快照为准。
`inputs.json` 保存实际叶子 FP32 字、块大小、cell 与种子；`graphs.json` 保存实际有序树。
`measurements.json` 保存逐次结果；`paired_differences.json` 保存同族同 arm 的
`A_chain − A_balanced` 区间。原始误差端点都是精确分数。

`prediction.md` 是运行前预测原文，不随结果修改。此 README 是运行后的说明，
不在运行时 manifest 的完整性清单中。

## 重放与汇总

从仓库根目录使用新输出目录：

```sh
python -m topics.softmax.experiments.online.pilot_summary topics/softmax/experiments/online/runs/expansion_v1 --output topics/softmax/experiments/online/runs/expansion_v1_summary --plot
```

汇总先严格重放所有测量和配对差。输出目录已存在时须改用新目录，不能覆盖。
源码版本不符时，按 online README 在隔离目录恢复归档源码后重放。

每格均值只描述这 8 个输入；数值区间不是总体统计 CI。两种 FMA arm 共用同一族，
不能计作独立的两倍样本。改变块数也改变叶子集合、样本最大值和分母，不能将观察到的
差异全部归因于树深度。一般非均匀叶块、其他 exp、硬件和其他分布仍在范围外。
