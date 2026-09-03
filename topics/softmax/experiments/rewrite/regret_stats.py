"""Regret and stratified bootstrap — closed-book rewrite (research core: USER-WRITTEN).

第五步，复现的收官。独立重算 v2 的 headline：beam 比 Q 的 paired normalized-regret
improvement，以及分层 group bootstrap 的 95% 区间。

独立性边界：
  - 每棵树的 target 用**重写的 oracle** 算：target = (E / root_ulp)^2；
  - q_selected 用**重写的 Q**（macro_score）选；
  - beam_selected 从**冻结 CSV** 取（B=3 beam 属工程管线，范围外）；
  - regret、primary、bootstrap 全部在这里独立重算，不读 CSV 里现成的 q_regret/beam_regret。

概念
----
  - target：一棵树的最终误差换成「几个根 ULP」再平方，越小越好，无量纲、跨宽度可比。
  - normalized regret：一组 64 棵树里，(target[选中]-best)/(worst-best)；选到最好=0，最差=1。
  - paired improvement：每组 q_regret - beam_regret，再对 192 组取平均 = primary；
    配对相减消掉组间方差，只留「同题上 beam 比 Q 好多少」。
  - stratified group bootstrap：按宽度分层，层内对组有放回重采样，算一次均值，重复多次，
    取 2.5%/97.5% 分位。下界 > 0 即 positive evidence。

Explain-back
------------
target 为什么要除以 root_ulp 再平方：
paired improvement 为什么用组内相减而不是分别平均两个 regret：
bootstrap 为什么要按宽度分层：

Prediction record（跑测试前写）
--------------------------------
Direction（重写能否重现 primary 为正、区间下界 > 0）：
Boundary（哪一步最可能让区间对不上冻结值）：
Failure signature（如果 bootstrap 的 rng 序列错位，区间会怎样偏）：
"""

from __future__ import annotations

import random
from statistics import mean
from typing import Callable, Sequence

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260823


def normalized_regret(targets: Sequence[float], selected: int) -> float:
    """USER-WRITTEN CORE. Min-max normalized regret of picking tree ``selected``.

    best = min(targets)，worst = max(targets)。
    返回 (targets[selected] - best) / (worst - best)。
    当 worst == best（全组同值，无差别）时返回 0.0。
    """
    if not targets or selected < 0 or selected >= len(targets):
        raise ValueError("invalid targets or selected index")
    best = min(targets)
    worst = max(targets)
    if worst == best:
        return 0.0
    return (targets[selected] - best) / (worst - best)


def paired_improvement(q_regrets: Sequence[float], beam_regrets: Sequence[float]) -> float:
    """USER-WRITTEN CORE. Mean over groups of (q_regret - beam_regret).

    两个序列一一对应同一组。逐组相减后取平均。长度不等时抛 ValueError。
    """
    if len(q_regrets) != len(beam_regrets):
        raise ValueError("q_regrets and beam_regrets must have the same length")
    return mean(q - b for q, b in zip(q_regrets, beam_regrets))


def _percentile(values: list[float], probability: float) -> float:
    """Scaffolding. Linear-interpolated percentile, matching the frozen definition."""
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_group_bootstrap_ci(
    rows: list[dict],
    field: Callable[[dict], float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Scaffolding. 95% CI by width-stratified group bootstrap (frozen RNG spec).

    RNG 调用序列是冻结边界，逐字实现：
      - rng = random.Random(seed)；
      - 按 width 把 rows 分层；
      - 每次重采样：按 width 升序遍历每个层，对该层 rng.choice 抽取 len(层) 次（有放回）；
        把抽到的行拼成一个样本；对样本算 mean(field(row))；
      - 重复 resamples 次，返回 (2.5 分位, 97.5 分位)。
    """
    if not rows or resamples <= 0:
        raise ValueError("bootstrap needs rows and positive resamples")
    rng = random.Random(seed)
    strata: dict[int, list[dict]] = {}
    for row in rows:
        strata.setdefault(int(row["width"]), []).append(row)
    draws = []
    for _ in range(resamples):
        sample = [
            rng.choice(strata[width])
            for width in sorted(strata)
            for _ in range(len(strata[width]))
        ]
        draws.append(mean(field(row) for row in sample))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)
