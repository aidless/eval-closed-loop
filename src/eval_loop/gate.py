"""回归门禁：给定新旧两份 run，判定是否退化，并归因到具体题/类别。

判定语义沿用 eval-gate 的教训（不重复实现统计）：
- BLOCK：配对 delta 显著为负（bootstrap CI 上限 < 0）且幅度超过容差；
- INCOMPLETE：两边题目集合对不上（缺证据），不做“没得比也算过”；
- FAIL：新结果绝对准确率低于地板；
- REVIEW：样本量或 CI 宽度不足以支撑结论；
- PASS：以上都不触发。

优先级 BLOCK > INCOMPLETE > FAIL > REVIEW > PASS（删证据不能把 BLOCK 降级）。
"""
from __future__ import annotations

import os
from typing import Any, Mapping

from .stats import ci_width, paired_bootstrap_delta, wilson_interval

VERDICT_ORDER = {"PASS": 0, "REVIEW": 1, "FAIL": 2, "INCOMPLETE": 3, "BLOCK": 4}

DEFAULT_CONFIG = {
    "min_pass": 0.10,       # 绝对地板：低于此直接 FAIL（弱模型场景，故意很低）
    "max_regression": 0.05, # 允许的退化容差
    "min_n": 50,            # 样本量下限，不足只给 REVIEW
    "alpha": 0.05,
    "bootstrap_iters": 10_000,
    "seed": 20261105,
    "max_ci_width": 0.25,   # CI 太宽 => 结论不可用
    "min_overlap": 1.0,     # 配对要求同一批题全对上
}


def _config(overrides: Mapping[str, Any] | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _by_id(records: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records}


def evaluate(candidate: list[dict], baseline: list[dict],
             config: Mapping[str, Any] | None = None) -> dict:
    cfg = _config(config)
    cand, base = _by_id(candidate), _by_id(baseline)

    # 证据缺失 / 不可比
    only_cand = sorted(set(cand) - set(base))
    only_base = sorted(set(base) - set(cand))
    overlap = (len(set(cand) & set(base)) / len(cand)) if cand else 0.0
    comparable = bool(cand) and bool(base) and overlap >= cfg["min_overlap"]

    n = len(cand)
    passes = sum(1 for r in cand.values() if r["correct"])
    rate = passes / n if n else 0.0
    ci = wilson_interval(passes, n)
    bpass = sum(1 for r in base.values() if r["correct"])
    brate = bpass / len(base) if base else 0.0

    if comparable:
        cand_scores = {i: (1.0 if r["correct"] else 0.0) for i, r in cand.items()}
        base_scores = {i: (1.0 if r["correct"] else 0.0) for i, r in base.items()}
        families = {i: r.get("category", "_unknown") for i, r in cand.items()}
        delta = paired_bootstrap_delta(
            cand_scores, base_scores, families,
            iterations=cfg["bootstrap_iters"], seed=cfg["seed"], alpha=cfg["alpha"])
    else:
        delta = {"delta": 0.0, "ci95": [0.0, 0.0], "n": 0,
                 "iterations": cfg["bootstrap_iters"], "seed": cfg["seed"],
                 "families": 0}

    gates: list[dict] = []

    def gate(gid: str, rule: str, observed, ok: bool | None, *,
             verdict: str | None = None, result: str | None = None,
             detail: str | None = None) -> None:
        if result is None:
            result = "pass" if ok else "fail"
        g = {"id": gid, "rule": rule, "observed": observed, "result": result}
        if verdict and result == "fail":
            g["verdict"] = verdict
        if detail:
            g["detail"] = detail
        gates.append(g)

    gate("G1-absolute", f"wilson_low({ci[0]:.3f}) >= min_pass({cfg['min_pass']})",
         round(ci[0], 6), n > 0 and ci[0] >= cfg["min_pass"], verdict="FAIL",
         detail=None if (n > 0 and ci[0] >= cfg["min_pass"])
         else "绝对准确率 Wilson 下限低于地板")

    if not comparable:
        gate("G2-regression",
             f"delta_ci_high <= {cfg['max_regression']} 且 delta <= -{cfg['max_regression']}",
             None, None, result="not_comparable",
             detail=f"重叠率 {overlap:.2f} < {cfg['min_overlap']}，不做回退判定")
    else:
        high = delta["ci95"][1]
        regressed = high < 0 and delta["delta"] <= -cfg["max_regression"]
        gate("G2-regression",
             f"delta({delta['delta']:.3f}) <= -{cfg['max_regression']} 且 ci_high({high:.3f}) < 0",
             round(high, 6), not regressed, verdict="BLOCK",
             detail=None if not regressed else "候选相对基线显著退化")

    if only_cand or only_base:
        gate("G6-evidence", "候选与基线题目集合一致", len(only_cand) + len(only_base),
             False, verdict="INCOMPLETE",
             detail=f"仅候选 {len(only_cand)} 题，仅基线 {len(only_base)} 题")

    power_ok = n >= cfg["min_n"] and ci_width(ci) <= cfg["max_ci_width"]
    gate("G5-power", f"n({n}) >= min_n({cfg['min_n']}) 且 ci_width({ci_width(ci):.3f}) <= {cfg['max_ci_width']}",
         round(ci_width(ci), 6), power_ok, verdict="REVIEW",
         detail=None if power_ok else "样本量或区间宽度不足以支撑结论")

    triggered = [g for g in gates if g["result"] == "fail" and g.get("verdict")]
    verdict = "PASS"
    for g in triggered:
        if VERDICT_ORDER[g["verdict"]] > VERDICT_ORDER[verdict]:
            verdict = g["verdict"]

    # --- 归因 ---
    flipped = []
    fixed = []
    if comparable:
        for i in sorted(set(cand) & set(base)):
            c, b = cand[i], base[i]
            if b["correct"] and not c["correct"]:
                flipped.append({"id": i, "category": c.get("category"),
                                "question": c["question"], "gold_final": c["gold_final"],
                                "baseline_pred": b["pred"], "candidate_pred": c["pred"],
                                "candidate_output": c["output"]})
            elif c["correct"] and not b["correct"]:
                fixed.append({"id": i, "category": c.get("category"),
                              "baseline_pred": b["pred"], "candidate_pred": c["pred"]})

    per_category = {}
    if comparable:
        cats = sorted({r.get("category", "_unknown") for r in cand.values()})
        for cat in cats:
            ids = [i for i, r in cand.items() if r.get("category", "_unknown") == cat]
            c_rate = sum(1 for i in ids if cand[i]["correct"]) / len(ids)
            b_rate = sum(1 for i in ids if base[i]["correct"]) / len(ids)
            per_category[cat] = {
                "n": len(ids),
                "baseline": round(b_rate, 4),
                "candidate": round(c_rate, 4),
                "delta": round(c_rate - b_rate, 4),
                "flipped": sum(1 for item in flipped if item["category"] == cat),
            }

    return {
        "verdict": verdict,
        "matched_rules": [g["id"] for g in triggered],
        "gates": gates,
        "baseline": {"n": len(base), "passes": bpass, "rate": round(brate, 4)},
        "candidate": {"n": n, "passes": passes, "rate": round(rate, 4),
                      "wilson_ci95": [round(x, 4) for x in ci]},
        "delta": {**delta, "ci95": [round(x, 4) for x in delta["ci95"]],
                  "delta": round(delta["delta"], 4)},
        "overlap": round(overlap, 4),
        "per_category": per_category,
        "flipped": flipped,
        "fixed": fixed,
        "config": cfg,
    }


def apply_report_only(gate_result: dict, report_only: bool) -> dict:
    """--report-only：只写报告不改退出码（本地默认关，CI 打开即阻断）。"""
    gate_result = dict(gate_result)
    gate_result["report_only"] = report_only
    return gate_result


def exit_code(gate_result: dict) -> int:
    if gate_result.get("report_only"):
        return 0
    return 0 if gate_result["verdict"] == "PASS" else 1
