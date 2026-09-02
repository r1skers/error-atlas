# Topic Registry

更新时间：2026-09-02。具体研究状态统一见 [NEXT_SESSION.md](NEXT_SESSION.md)，
本文件只保留主题登记，不重复维护实验日志。

| Topic | 研究对象 | 状态 / 入口 |
| --- | --- | --- |
| Taylor expansion | remainder、误差界、数值微分与 bias–variance | [第一轮完成](topics/taylor-expansion/README.md) |
| Softmax | 输入量化、normalization、reduction error 与风险预测 | [活跃：online certificate 校准后](topics/softmax/README.md) |

## 阅读路线

- Taylor：[理论笔记](topics/taylor-expansion/notes/00_error_language.md) →
  [实验说明](topics/taylor-expansion/experiments/README.md)。
- Softmax：[基础与 exact-oracle 推导](topics/softmax/notes/foundations.md) →
  [实验代码索引](topics/softmax/experiments/README.md) →
  [分阶段证据](topics/softmax/experiments/results/README.md)。
- 学习预备：[完整知识谱](KNOWLEDGE_MAP.md)；不以教材替代已冻结证据。
- 旧的研究续接点：[2026-08-12 snapshot](docs/history/2026-08-12-softmax-handoff.md)。

## 新 Topic 的最小模板

先只建立主题 README，明确八项内容：

1. Object：研究对象。
2. Reference：相对于什么定义误差。
3. Metric：如何度量。
4. Sources：误差从哪里进入。
5. Propagation：哪些结构改变误差。
6. Control：可调机制。
7. Optimization：精度与成本权衡。
8. Verification：证明、反例或可复现实验。

只有内容实际出现后，再添加 notes、experiments、tests。
研究纪律见 [error analysis protocol](framework/error_analysis_protocol.md)。
