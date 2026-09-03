# Repository Maintenance

本指南管理代码和文档结构，不修改研究协议、证据等级或 consumer policy。

## 目录和信息的职责

| 路径 | 职责 |
| --- | --- |
| README.md | 项目入口与开发命令 |
| NEXT_SESSION.md | 唯一的当前研究状态 / 下一入口 |
| TOPICS.md | 主题注册表 |
| topics/*/notes/ | 理论和历史笔记；旧待办不自动成为当前任务 |
| topics/*/experiments/ | 数值实现、历史 runners、冻结协议 |
| topics/*/experiments/results/ | 版本化证据，按各目录 README 解释 |
| topics/*/tests/ | 单元与 artifact 回归测试 |
| tools/ | 测试和其他维护工具 |
| docs/history/ | 历史续接记录 |
| KNOWLEDGE_MAP.md | 学习教材，不是当前研究状态源 |

## 一套测试入口

从仓库根目录运行；不依赖 pytest 或安装项目为 Python package：

```sh
python tools/run_tests.py
python tools/run_tests.py --suite softmax -v
python tools/run_tests.py --suite taylor
python tools/run_tests.py --suite maintenance
python tools/run_tests.py --suite softmax -p "test_predictor_fixed_k8_beam_inference.py"
```

返回码 0 表示通过，1 表示测试失败，2 表示命令/空选择错误。
模式匹配不到测试不能被报告为成功。每个 topic 使用独立 loader，
实验 source directory 由 harness 加入 import path；不要在每个测试里复制路径设置。

旧的完整发现命令仍由 experiments 中的兼容文件转发：

```sh
python -m unittest discover -s topics/softmax/experiments -p "test_*.py"
python -m unittest discover -s topics/taylor-expansion/experiments -p "test_*.py"
```

单文件筛选请使用统一入口。只对新的 tests 目录运行裸 discover 不会自动设置
历史脚本的 import path。根目录裸 `python -m unittest` 也不是本项目的发现入口。

这些命令只运行 regression tests，不调用研究脚本的 main/CLI 重新发布 artifacts。
测试中临时生成的文件放在 temporary directories。

## 精确语义与可移植性分开

FP32 bits、exact Fraction error、Q score、shortlist 和最终 tree selection 的
测试仍要求精确一致。已冻结 CSV、JSON、模型及数值实现没有为本地测试改写。

fixed-K8 learned beam 的概率路径使用 NumPy matmul / exp。
本地审计观察到冻结分数与当前 Windows 环境相差一个 binary64 ULP；
单元测试现在允许每个 beam score 最多 8 个 reference binary64 ULP。
这是一项小范围的工程回归容差，不是理论误差界、FP32 容差、排名政策或
“跨平台 bitwise repeatable”的声明。树选择与 Q score 仍须精确匹配。
历史 192-group exact replay 只保留其原运行环境内的证据含义。

## 为什么暂不移动数值源文件

历史 runner 相互导入，而且多个 metadata / preregistration 保存 source paths 与
SHA-256。移动文件、重写 import 或统一“看起来一样”的 helper 都可能改变
冻结 source boundary；直接拿新源码重跑旧 one-shot 实验会混淆研究证据。

第一轮仅迁移测试和入口文档。第二轮已将 A/C、coherence structure 和
ancestor/history 三个诊断的计算抽到
[reduction_analysis](../topics/softmax/experiments/reduction_analysis/README.md)：
一次 oracle replay、多种惰性分析视图；原脚本保留实验配置、统计展示与兼容函数。
原 oracle 和其他数值脚本保持不变，不重跑、不覆盖、不删除结果数据。
各 stage 的协议与 research-direction checkpoint 也保持原字节内容。
experiments 根目录留下的 test_*_suite.py 是历史命令适配器，不是研究实现。

这三个诊断的旧实现可由 Git revision `ad1fe87` 恢复。迁移前保存了 18 个
input/tree case 的逐字段回归快照和三条 CLI 输出；迁移后精确核对 Fraction、
float hex、NaN 约定、seed schedule 与输出字段。快照是维护测试数据，不是新研究证据。
未来使用共享包的实验必须记录包内源码的 hashes，不能只记录兼容 wrapper。

## 后续代码拆分顺序

共享轨迹与前三个 coherence 视图已经接通，但其余阶段仍有技术债：

1. **统计 primitives**：_rankdata、Spearman、percentile 和 bootstrap 有重复。
   提取前逐一比较 tie、undefined input 和插值语义，先加 characterization tests。
2. **图结构 utilities**：这三个视图已共用 parent/depth；其他阶段的 root-band、
   graph generation 调度仍分散。generator 的 seed schedule 与 graph identity
   必须继续精确匹配，不能把不同实验的 seed namespace 合并。
3. **prototype 与实验 runner**：calibration modules 互相导入 private helpers，
   阶段实现耦合。新功能优先依赖经验证的公共模块，不再延长 private-import 链。
4. **版本化迁移**：在独立变更中引入共享模块，保留原 evidence-generating snapshot；
   通过 differential tests 后记录新 source version，而不是修改旧 artifact hashes。
5. **再迁移包目录**：有了兼容 API、路径策略和回归证据后，才把核心、校准、验证
   实现物理拆成包。不要只移动文件后靠大量 sys.path hacks 维持运行。

每次变更后检查 git diff、测试结果和结果目录是否仍原样。
文档更新只维护 NEXT_SESSION 的当前状态，其他入口链接它，避免再次多处漂移。

## 复现分区与原实验：为什么两套都保留

`experiments/rewrite/` 是 2026-09 的独立复现分区，`experiments/` 根目录下的
oracle、生成器与校准脚本是原实验。看起来像重复，但两者职责不同，**都不删、不合并**：

- 原模块是**冻结证据的生成源**。results/ 各 artifact 的 metadata 用 SHA-256 锁定它们的
  源码路径与内容，preregistration 也引用它们。删除或重构会破坏 provenance，
  这与「以冻结 artifact 为准」的仓库原则冲突。
- rewrite 分区是**独立验证**。它由用户从空白重写，配差分测试，逐值对照旧实现与冻结
  artifact，是「冻结结果不是实现 bug」的保证。删除等于丢弃验证工作与其回归测试。

因此这不是可去重的重复代码，而是「生成」与「独立核对」两条相互印证的实现。
根目录 40 余个历史校准脚本互相 import 私有 helper，且被 metadata hash 锁定；
「优化」它们等于改动冻结源码边界。正确的整理路径见上一节的版本化提取顺序，
那是带差分测试的大工程，不是随手重构。在那之前，两套并存是有意的设计，不是待清理的乱。

复现结论与要点见 [rewrite 复现笔记](../topics/softmax/notes/rewrite_replication.md)。
