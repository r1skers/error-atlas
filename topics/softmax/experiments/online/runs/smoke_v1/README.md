# Runner smoke v1：工程验收

2026-09-05，使用审计已有的两个 max-position 确定性算例，验证保存与重放。
这是工程诊断，不是新的随机扫描或确认实验。

- 2 个族：32 块、每块 32 个元素、gap=20，max 位置为 0 和 31。
- 4 个 schedule × 2 种 FMA arm：16 条 measurement，12 条对 balanced 的配对差。
- 所有误差端点保存为精确分数；输出和叶输入保存为 FP32 字。
- 非 FMA 的已知对照：max=0 时根均为 32、吸收次数 31/5/5/10；
  max=31 时 chain 根为 `0x42000001`，balanced 为 `0x42000000`。

配置、预测当时版本与执行源码存于本目录；[manifest](manifest.json) 记录环境、Git 状态、
耗时及数据/源码 hashes。运行时源码尚未提交，所以源码快照是必要的执行版本依据。
用户后来补充的“解耦、无关联”预测见当前预测文档；此包不追改当时的预测快照。
本 README 是工程说明，不包含在运行时生成的完整性清单中。

从仓库根目录核验：

```sh
python -m topics.softmax.experiments.online.pilot_runner replay topics/softmax/experiments/online/runs/smoke_v1
```

只有所存输入、图、精确测量、配对差和当前源码都符合重放要求才返回 `exact_match`。
以后源码更新导致严格重放拒绝时，按 online README 在隔离目录恢复归档源码。
不要覆盖此目录；新运行使用新目录。
