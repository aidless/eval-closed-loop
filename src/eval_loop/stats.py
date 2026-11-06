"""统计层：复用 eval-gate 的 Wilson 区间与分层 paired bootstrap，补两件事：

1. 准确率的非参 bootstrap 区间（逐题重采样，固定 seed，可复现）；
2. Cohen's κ（judge 与人工标注的一致性）。

复用方式是 import 已有仓库的模块（默认 ../eval-gate，可用
EC_EVAL_GATE_PATH 指定），不复制、不改动原仓库代码。
"""
from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterable, Mapping


def _eval_gate_stats():
    path = os.path.expanduser(os.environ.get("EC_EVAL_GATE_PATH", "~/eval-gate"))
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from eval_gate.stats import (  # noqa: F401
            ci_width,
            paired_bootstrap_delta,
            wilson_interval,
        )
    except ImportError as exc:  # 复用失败就明确失败，不悄悄替换成另一套统计
        raise SystemExit(
            "无法导入 eval-gate。请先 clone 判决层仓库并设置 EC_EVAL_GATE_PATH：\n"
            "  git clone https://github.com/aidless/eval-gate ~/eval-gate\n"
            "  export EC_EVAL_GATE_PATH=~/eval-gate\n"
            f"原始错误: {exc}"
        )
    return wilson_interval, paired_bootstrap_delta, ci_width


wilson_interval, paired_bootstrap_delta, ci_width = _eval_gate_stats()


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("空样本没有分位数")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo, hi = int(pos), int(pos + 0.999999)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def accuracy_ci(corrects: Iterable[bool], *, iterations: int = 10_000,
                seed: int = 20261105, alpha: float = 0.05) -> dict:
    """准确率的 bootstrap 百分位区间（对题目有放回重采样）。"""
    values = [1.0 if c else 0.0 for c in corrects]
    n = len(values)
    if n == 0:
        return {"point": 0.0, "ci95": [0.0, 1.0], "n": 0,
                "iterations": iterations, "seed": seed}
    point = sum(values) / n
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iterations))
    return {
        "point": point,
        "ci95": [_percentile(samples, alpha / 2),
                 _percentile(samples, 1 - alpha / 2)],
        "n": n,
        "iterations": iterations,
        "seed": seed,
    }


def kappa_ci(a: Iterable[int], b: Iterable[int], *, iterations: int = 10_000,
             seed: int = 20261105, alpha: float = 0.05) -> dict:
    """κ 的配对 bootstrap 区间：对 (judge, human) 成对重采样，重算 κ 取分位数。"""
    xs, ys = [int(x) for x in a], [int(y) for y in b]
    if len(xs) != len(ys):
        raise ValueError("两组标签长度不一致")
    n = len(xs)
    point = cohen_kappa(xs, ys)
    if n == 0:
        return {**point, "ci95": [0.0, 0.0], "iterations": iterations, "seed": seed}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(cohen_kappa([xs[i] for i in idx], [ys[i] for i in idx])["kappa"])
    samples.sort()
    ci = [_percentile(samples, alpha / 2), _percentile(samples, 1 - alpha / 2)]
    return {**point, "ci95": ci, "width": ci[1] - ci[0],
            "iterations": iterations, "seed": seed}


def cohen_kappa(a: Iterable[int], b: Iterable[int]) -> dict:
    """二分类 Cohen's κ。返回 po / pe / kappa / n / 混淆矩阵。"""
    xs, ys = list(a), list(b)
    if len(xs) != len(ys):
        raise ValueError("两组标签长度不一致")
    n = len(xs)
    if n == 0:
        return {"n": 0, "po": 0.0, "pe": 0.0, "kappa": 0.0,
                "confusion": {"both_1": 0, "a1_b0": 0, "a0_b1": 0, "both_0": 0}}
    both_1 = sum(1 for x, y in zip(xs, ys) if x == 1 and y == 1)
    a1_b0 = sum(1 for x, y in zip(xs, ys) if x == 1 and y == 0)
    a0_b1 = sum(1 for x, y in zip(xs, ys) if x == 0 and y == 1)
    both_0 = sum(1 for x, y in zip(xs, ys) if x == 0 and y == 0)
    po = (both_1 + both_0) / n
    p_a1 = (both_1 + a1_b0) / n
    p_b1 = (both_1 + a0_b1) / n
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    kappa = 0.0 if pe == 1 else (po - pe) / (1 - pe)
    return {
        "n": n, "po": po, "pe": pe, "kappa": kappa,
        "confusion": {"both_1": both_1, "a1_b0": a1_b0,
                      "a0_b1": a0_b1, "both_0": both_0},
    }
