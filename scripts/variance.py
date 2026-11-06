"""同配置多次重跑：量化「分数层面能复现到什么程度」（reports/variance.json）。

用法：
  PYTHONPATH=src python3 scripts/variance.py \
      --run rep1=results/runs/a_cot_n300_rep1.jsonl \
      --run rep2=results/runs/a_cot_n300_rep2.jsonl \
      --run rep3=results/runs/a_cot_n300_rep3.jsonl \
      --out reports/variance.json

只做三件事，全部基于同一批 n=300 题的实测：
1. 分数层面：每次的准确率、两两差值、最大跨度；
2. 逐题层面：翻转率、翻转题的得分分布（1/3 还是 2/3）、按难度桶分布、
   翻转是否只是数值近失（相对误差 <=1%）；
3. 噪声模型：由逐题翻转概率估计同配置两次跑的 Δ 标准差，
   给出「门禁 max_regression 至少定到多少才不会被噪声触发」。
   只把 n=300 标为实测；按 q/n 缩放到 n=60 / n=1319 的数值标为推算（未验证）。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_loop import bench, gate  # noqa: E402
from eval_loop.stats import accuracy_ci, paired_bootstrap_delta, wilson_interval  # noqa: E402

NEAR_MISS_REL = 0.01


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _is_near_miss(pred, gold: str) -> bool:
    p, g = _num(pred), _num(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= max(1e-4, NEAR_MISS_REL * abs(g))


def _bucket(rec: dict) -> str:
    return rec.get("category") or f"steps_{rec.get('steps', '?')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path.jsonl")
    ap.add_argument("--out", default="reports/variance.json")
    ap.add_argument("--env-note", action="append", default=[],
                    help="每次 run 当时的机器状态（如 GPU 占比），按顺序对应")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        run = bench.read_run(path)
        recs = run["records"]
        acc = accuracy_ci([r["correct"] for r in recs])
        wil = wilson_interval(sum(r["correct"] for r in recs), len(recs))
        labels_sha = hashlib.sha256(
            json.dumps(sorted((r["id"], bool(r["correct"])) for r in recs)).encode()
        ).hexdigest()
        runs.append({
            "label": label, "path": path, "created_at": run.get("created_at"),
            "dataset": run.get("dataset"), "model": run.get("model"),
            "prompt": run.get("prompt"), "temperature": run.get("temperature"),
            "seed": run.get("seed"), "num_predict": run.get("num_predict"),
            "num_gpu": run.get("num_gpu"),
            "n": len(recs), "passes": sum(r["correct"] for r in recs),
            "accuracy": round(acc["point"], 6),
            "wilson_ci95": [round(x, 6) for x in wil],
            "wilson_width": round(wil[1] - wil[0], 6),
            "latency_ms_median": round(statistics.median(r["latency_ms"] for r in recs), 1),
            "tokens_total": sum((r.get("prompt_eval_count") or 0) +
                                (r.get("eval_count") or 0) for r in recs),
            "score_sha256": labels_sha,
            "records": recs,
        })

    by_label = {r["label"]: {x["id"]: x for x in r["records"]} for r in runs}
    common = sorted(set.intersection(*(set(by_label[r["label"]]) for r in runs)))

    # --- 1. 分数层面 ---
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        cand = {i: (1.0 if by_label[a["label"]][i]["correct"] else 0.0) for i in common}
        base = {i: (1.0 if by_label[b["label"]][i]["correct"] else 0.0) for i in common}
        fams = {i: by_label[a["label"]][i].get("category", "_unknown") for i in common}
        boot = paired_bootstrap_delta(cand, base, fams,
                                      iterations=10_000, seed=20261105, alpha=0.05)
        plus = [i for i in common
                if by_label[a["label"]][i]["correct"] and not by_label[b["label"]][i]["correct"]]
        minus = [i for i in common
                 if not by_label[a["label"]][i]["correct"] and by_label[b["label"]][i]["correct"]]
        same_out = sum(1 for i in common
                       if by_label[a["label"]][i]["output"] == by_label[b["label"]][i]["output"])
        same_cor = sum(1 for i in common
                       if by_label[a["label"]][i]["correct"] == by_label[b["label"]][i]["correct"])
        pair_records = [by_label[a["label"]][i] for i in common] + \
                       [by_label[b["label"]][i] for i in common]
        g = gate.evaluate(pair_records[:len(common)], pair_records[len(common):])
        pairs.append({
            "a": a["label"], "b": b["label"],
            "delta_pp": round((a["accuracy"] - b["accuracy"]) * 100, 3),
            "a_only_correct": len(plus), "b_only_correct": len(minus),
            "flipped": len(plus) + len(minus),
            "flip_rate": round((len(plus) + len(minus)) / len(common), 4),
            "same_output": same_out,
            "same_correct": same_cor,
            "paired_bootstrap_ci95": [round(x, 5) for x in boot["ci95"]],
            "gate_verdict": g["verdict"],
            "gate_delta": g["delta"]["delta"],
            "gate_rules": g["matched_rules"],
        })

    accs = [r["accuracy"] for r in runs]
    spread = max(accs) - min(accs)
    max_pair = max(abs(p["delta_pp"]) for p in pairs)

    # --- 2. 逐题翻转 ---
    counts = {i: sum(1 for r in runs if by_label[r["label"]][i]["correct"]) for i in common}
    k = len(runs)
    flip_ids = [i for i in common if 0 < counts[i] < k]
    flip_rate = len(flip_ids) / len(common)
    dist = Counter(counts[i] for i in flip_ids)

    bucket_n = Counter(_bucket(by_label[runs[0]["label"]][i]) for i in common)
    bucket_flip = Counter(_bucket(by_label[runs[0]["label"]][i]) for i in flip_ids)
    by_bucket = [{
        "bucket": b, "n": bucket_n[b], "flipped": bucket_flip.get(b, 0),
        "flip_rate": round(bucket_flip.get(b, 0) / bucket_n[b], 4),
        "flip_share": round(bucket_flip.get(b, 0) / len(flip_ids), 4) if flip_ids else 0.0,
    } for b in sorted(bucket_n)]

    near_miss = 0
    for i in flip_ids:
        rec0 = by_label[runs[0]["label"]][i]
        for r in runs:
            rec = by_label[r["label"]][i]
            if not rec["correct"] and _is_near_miss(rec.get("pred"), rec["gold_final"]):
                near_miss += 1
                break
    unparsable = sum(1 for i in flip_ids
                     if any(by_label[r["label"]][i].get("pred") is None for r in runs))

    rel_positions = []
    examples = []
    for i in flip_ids:
        outs = [by_label[r["label"]][i]["output"] for r in runs]
        same_len = min(len(o) for o in outs)
        pos = min(_first_diff(outs[0], outs[j]) for j in range(1, len(outs)))
        rel_positions.append(pos / same_len if same_len else 1.0)
        if len(examples) < 12:
            examples.append({
                "id": i, "bucket": _bucket(by_label[runs[0]["label"]][i]),
                "gold": by_label[runs[0]["label"]][i]["gold_final"],
                "preds": [by_label[r["label"]][i].get("pred") for r in runs],
                "correct": [int(by_label[r["label"]][i]["correct"]) for r in runs],
                "first_diff_char": pos, "first_diff_rel": round(rel_positions[-1], 3),
                "near_miss": any(not by_label[r["label"]][i]["correct"] and
                                 _is_near_miss(by_label[r["label"]][i].get("pred"),
                                               by_label[r["label"]][i]["gold_final"])
                                 for r in runs),
            })

    # --- 3. 噪声模型 ---
    # 两次同配置跑的 Δ = (1/n) Σ d_i，d_i ∈ {-1,0,1}；Var(Δ)=Var(d)/n。
    # 用每对 run 的逐题 d_i 估计方差，再取三对平均。
    var_deltas = []
    for a, b in itertools.combinations(runs, 2):
        d = [int(by_label[a["label"]][i]["correct"]) -
             int(by_label[b["label"]][i]["correct"]) for i in common]
        var_deltas.append(statistics.pvariance(d) / len(common))
    var_delta = statistics.mean(var_deltas)
    sd_measured = math.sqrt(var_delta)
    q_hat = statistics.mean(p["flip_rate"] for p in pairs)
    sd_scaled = {n: math.sqrt(q_hat / n) for n in (60, 300, 1319)}

    recommendation = {
        "measured_at_n300": {
            "sd_delta_pp": round(sd_measured * 100, 3),
            "one_sided_95_pp": round(1.96 * sd_measured * 100, 3),
            "three_sigma_pp": round(3 * sd_measured * 100, 3),
            "observed_max_abs_delta_pp": round(max_pair, 3),
            "threshold_floor_pp": round(max(3 * sd_measured * 100, max_pair), 3),
            "questions_equivalent": round(max(3 * sd_measured, max_pair / 100) * 300, 2),
        },
        "scaled_unverified": {str(n): {
            "three_sigma_pp": round(3 * sd_scaled[n] * 100, 3),
            "one_sided_95_pp": round(1.96 * sd_scaled[n] * 100, 3),
        } for n in (60, 300, 1319)},
        "current_max_regression": gate.DEFAULT_CONFIG["max_regression"],
        "current_is_safe_at_n300": 3 * sd_measured * 100 <= gate.DEFAULT_CONFIG["max_regression"] * 100,
    }

    payload = {
        "probe": {
            "dataset": runs[0]["dataset"],
            "n": len(common), "repeats": k,
            "model": runs[0]["model"],
            "prompt": runs[0]["prompt"],
            "temperature": runs[0]["temperature"],
            "seed": runs[0]["seed"],
            "num_predict": runs[0]["num_predict"],
            "num_gpu": runs[0]["num_gpu"],
            "env_notes": args.env_note,
        },
        "runs": [{k2: v for k2, v in r.items() if k2 != "records"} for r in runs],
        "score_level": {
            "accuracies": [round(x, 6) for x in accs],
            "spread_pp": round(spread * 100, 3),
            "max_abs_pair_delta_pp": round(max_pair, 3),
            "pairs": pairs,
            "pairs_identical_correct": [f"{p['a']}={p['b']}" for p in pairs if p["flipped"] == 0],
            "runs_in_all_flip_pairs": sorted(set.intersection(*[
                {p["a"], p["b"]} for p in pairs if p["flipped"]
            ])) if any(p["flipped"] for p in pairs) else [],
        },
        "per_question": {
            "n": len(common),
            "unanimous_correct": sum(1 for i in common if counts[i] == k),
            "unanimous_wrong": sum(1 for i in common if counts[i] == 0),
            "mixed": len(flip_ids),
            "flip_rate": round(flip_rate, 4),
            "correct_count_dist": {str(x): dist[x] for x in range(1, k)},
            "by_bucket": by_bucket,
            "near_miss_questions": near_miss,
            "unparsable_questions": unparsable,
            "first_diff_rel_median": round(statistics.median(rel_positions), 3) if rel_positions else None,
            "examples": examples,
        },
        "noise_model": {
            "q_hat_pair_flip_rate": round(q_hat, 4),
            "var_delta": var_delta,
            "sd_delta_at_n300": sd_measured,
            "scaled_sd": {str(n): sd_scaled[n] for n in sd_scaled},
            "recommendation": recommendation,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"n={len(common)} repeats={k} acc={[round(x, 4) for x in accs]} "
          f"spread={spread * 100:.2f}pp max_pair={max_pair:.2f}pp")
    print(f"mixed={len(flip_ids)}/{len(common)} flip_rate={flip_rate:.4f} "
          f"dist={dict(dist)} near_miss={near_miss}")
    print(f"sd(Δ)@300={sd_measured * 100:.2f}pp 3σ={3 * sd_measured * 100:.2f}pp "
          f"-> max_regression floor={recommendation['measured_at_n300']['threshold_floor_pp']:.2f}pp")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
