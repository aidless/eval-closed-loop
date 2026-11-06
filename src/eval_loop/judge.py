"""Judge 一致性校准。

两种判分器都要和人工标注比：
- rule : 确定性数字判分（bench.py 的 is_correct），零成本、完全可复现；
- llm  : 本地小模型当裁判，读题/金标/模型答案后给 0-1。

三种 judge 配置用于校准对比（同一批人工标注、同一批模型答案）：
- raw    : 只给规则，不给示例；
- fewshot: 给 3 个锚点示例（一正两负），让裁判照抄判定风格；
- rubric : 给逐项检查清单（最终答案是否给出 / 数值是否相等 / 格式归一 / 截断处理），
           要求裁判按项输出 checks。

对每个 (judge, mode) 单独算 Cohen's κ，并按类别（难度桶）拆分，找出 κ 低的题类——
这些类别上 judge 不能拿来当门禁。人工标注格式 human_labels/labels.jsonl：
{"id":..., "human":0|1, "note":"..."}。
"""
from __future__ import annotations

import json
import os
import re
import string
from typing import Any

from . import bench
from .stats import cohen_kappa, kappa_ci

_BASE_RULES = """You are a strict grader. Decide whether the model's FINAL answer is numerically equal to the reference answer.

Problem:
$question

Reference final answer: $gold
Model's answer:
$output

Rules:
- Compare only the final numeric answer, not the explanation wording.
- Ignore formatting, currency signs and thousands separators.
- If the model's final answer is missing or ambiguous, mark it wrong.
"""

_RAW = _BASE_RULES + """
Reply with strict JSON only: {"correct": 0 or 1, "reason": "<one short sentence>"}"""

_FEWSHOT = _BASE_RULES + """
Anchors (follow the same judgement style):
1) Reference 12; model answer "#### 12" -> {"correct": 1, "reason": "final answer equals 12"}
2) Reference 45; model answer "#### 54" -> {"correct": 0, "reason": "final answer 54 != 45"}
3) Reference 8; model answer is cut off and has no final number -> {"correct": 0, "reason": "no final answer"}

Reply with strict JSON only: {"correct": 0 or 1, "reason": "<one short sentence>"}"""

_RUBRIC = _BASE_RULES + """
Check each item before deciding:
C1 final_answer_present: does the model state a final number?
C2 numeric_equal: is that number equal to the reference (1e-4 tolerance)?
C3 format_normalized: did you ignore $$, %, thousands separators and trailing periods?
C4 truncated: is the answer cut off so the final number is missing? (then C2 must be false)

If C1 or C4 fails, correct=0. If C2 fails, correct=0.

Reply with strict JSON only:
{"checks": {"C1": true/false, "C2": true/false, "C3": true/false, "C4": true/false}, "correct": 0 or 1, "reason": "<one short sentence>"}"""

PROMPTS = {"raw": _RAW, "fewshot": _FEWSHOT, "rubric": _RUBRIC}

_JSON_RE = re.compile(r"\{.*\}", re.S)


def judge_one(model: str, record: dict, *, num_predict: int = 96,
              mode: str = "raw") -> dict:
    prompt = string.Template(PROMPTS[mode]).substitute(
        question=record["question"], gold=record["gold_final"],
        output=record["output"])
    resp = bench.generate(model, prompt, num_predict=num_predict,
                          temperature=0.0, seed=42)
    text = resp["response"]
    m = _JSON_RE.search(text)
    label, reason, checks = None, text.strip()[:120], None
    if m:
        try:
            payload = json.loads(m.group(0))
            label = int(payload.get("correct"))
            reason = str(payload.get("reason", ""))[:160]
            checks = payload.get("checks")
        except (json.JSONDecodeError, TypeError, ValueError):
            label = None
    return {"label": label, "reason": reason, "checks": checks, "raw": text,
            "latency_ms": resp["latency_ms"],
            "prompt_tokens": resp.get("prompt_eval_count"),
            "out_tokens": resp.get("eval_count")}


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_judge(model: str, records: list[dict], out_path: str,
              *, mode: str = "raw") -> dict:
    """对每条记录跑 judge，逐条落盘，重跑自动跳过已有 id。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = {r["id"]: r for r in load_jsonl(out_path)} if os.path.exists(out_path) else {}
    if not done and records:
        # 首条预热（含 ollama 模型加载）不计入成本统计
        judge_one(model, records[0], mode=mode)
    with open(out_path, "a", encoding="utf-8") as f:
        for rec in records:
            if rec["id"] in done:
                continue
            out = judge_one(model, rec, mode=mode)
            row = {"id": rec["id"], "category": rec["category"],
                   "judge_model": model, "judge_mode": mode,
                   "judge_label": out["label"],
                   "judge_reason": out["reason"], "judge_checks": out["checks"],
                   "judge_raw": out["raw"],
                   "latency_ms": out["latency_ms"],
                   "prompt_tokens": out["prompt_tokens"],
                   "out_tokens": out["out_tokens"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  judge[{mode}] {rec['id']} -> {out['label']}", flush=True)
    return {"judge_model": model, "judge_mode": mode, "records": len(records)}


def calibrate(records: list[dict], human: dict[str, dict],
              judges: list[tuple[str, list[dict], str]] | None = None,
              *, target_half_width: float = 0.05) -> dict:
    """算 rule 与每个 (llm judge, mode) 相对人工标注的 κ（含配对 bootstrap CI）。"""
    by_id = {r["id"]: r for r in records}
    ids = [r["id"] for r in records if r["id"] in human]
    if not ids:
        raise SystemExit("人工标注与记录没有交集，检查 human_labels/labels.jsonl")

    human_labels = [int(human[i]["human"]) for i in ids]
    rule_labels = [1 if by_id[i]["correct"] else 0 for i in ids]
    rule = kappa_ci(rule_labels, human_labels)
    result: dict[str, Any] = {
        "n": len(ids),
        "rule": rule,
        "judges": [],
        "per_category": {},
    }

    def _n_needed(ci: dict, n: int) -> int | None:
        """bootstrap CI 宽度 ∝ 1/√n，反解达到目标半宽所需条数。"""
        w = ci.get("width")
        if not w:
            return None
        return int(round(n * (w / (2 * target_half_width)) ** 2))

    disagreements_rule = [
        {"id": i, "human": int(human[i]["human"]), "rule": 1 if by_id[i]["correct"] else 0,
         "note": human[i].get("note"), "pred": by_id[i]["pred"],
         "gold_final": by_id[i]["gold_final"]}
        for i in ids if (1 if by_id[i]["correct"] else 0) != int(human[i]["human"])
    ]
    result["disagreements_rule"] = disagreements_rule

    for spec in (judges or []):
        model, rows, mode = (spec if len(spec) == 3 else (*spec, "raw"))
        key = model if mode == "raw" else f"{model} [{mode}]"
        by_judge = {r["id"]: r for r in rows}
        judged_ids = [i for i in ids if i in by_judge and by_judge[i]["judge_label"] is not None]
        if not judged_ids:
            continue
        labels = [int(by_judge[i]["judge_label"]) for i in judged_ids]
        entry = kappa_ci(labels, [int(human[i]["human"]) for i in judged_ids])
        entry["judge_model"] = model
        entry["judge_mode"] = mode
        entry["key"] = key
        entry["n_judged"] = len(judged_ids)
        entry["parse_rate"] = len(judged_ids) / len(ids)
        entry["n_for_halfwidth_005"] = _n_needed(entry, len(judged_ids))
        entry["latency_ms_mean"] = round(
            sum(by_judge[i].get("latency_ms") or 0 for i in judged_ids) / len(judged_ids), 1)
        entry["tokens_mean"] = round(sum(
            (by_judge[i].get("prompt_tokens") or 0) + (by_judge[i].get("out_tokens") or 0)
            for i in judged_ids) / len(judged_ids), 1)
        entry["disagreements"] = [
            {"id": i, "category": by_judge[i]["category"],
             "human": int(human[i]["human"]), "judge": int(by_judge[i]["judge_label"]),
             "reason": by_judge[i].get("judge_reason"),
             "pred": by_id[i]["pred"], "gold_final": by_id[i]["gold_final"],
             "human_note": human[i].get("note")}
            for i in judged_ids
            if int(by_judge[i]["judge_label"]) != int(human[i]["human"])
        ]
        result["judges"].append(entry)

        for cat in sorted({by_id[i]["category"] for i in ids}):
            cat_ids = [i for i in ids if by_id[i]["category"] == cat]
            llm_cat = [i for i in cat_ids if i in judged_ids]
            if not llm_cat:
                continue
            entry_cat = result["per_category"].setdefault(
                cat, {"n": len(cat_ids),
                      "rule_kappa": cohen_kappa(
                          [1 if by_id[i]["correct"] else 0 for i in cat_ids],
                          [int(human[i]["human"]) for i in cat_ids])["kappa"]})
            entry_cat.setdefault("judges", {})[key] = cohen_kappa(
                [int(by_judge[i]["judge_label"]) for i in llm_cat],
                [int(human[i]["human"]) for i in llm_cat])["kappa"]
            entry_cat.setdefault("judge_n", {})[key] = len(llm_cat)
    return result
