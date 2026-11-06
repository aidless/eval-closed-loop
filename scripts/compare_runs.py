"""逐题比对两次 run 的输出，产出稳定性证据（reports/stability.json）。

用法：
  python3 scripts/compare_runs.py --a results/runs/a_cot_n1319.jsonl \
      --b results/runs/a_cot_n1319_rep2.jsonl --out reports/stability.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_loop import bench  # noqa: E402


def first_diff(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="reports/stability.json")
    args = ap.parse_args()

    run_a = bench.read_run(args.a)
    run_b = bench.read_run(args.b)
    a = {r["id"]: r for r in run_a["records"]}
    b = {r["id"]: r for r in run_b["records"]}
    common = sorted(set(a) & set(b))
    diffs = []
    for i in common:
        ra, rb = a[i], b[i]
        if ra["output"] == rb["output"]:
            continue
        diffs.append({
            "id": i, "category": ra.get("category"),
            "pred_a": ra.get("pred"), "pred_b": rb.get("pred"),
            "correct_a": int(bool(ra.get("correct"))),
            "correct_b": int(bool(rb.get("correct"))),
            "first_diff_char": first_diff(ra["output"], rb["output"]),
            "output_a_len": len(ra["output"]), "output_b_len": len(rb["output"]),
            "output_a_head": ra["output"][:200],
            "output_b_head": rb["output"][:200],
        })
    payload = {
        "baseline": args.a, "repeat": args.b,
        "n_compared": len(common),
        "only_in_a": sorted(set(a) - set(b))[:20],
        "only_in_b": sorted(set(b) - set(a))[:20],
        "output_identical": not diffs,
        "identical": not diffs,
        "flipped_correct": sum(1 for d in diffs if d["correct_a"] != d["correct_b"]),
        "differences": diffs,
        "decoding": {"temperature": run_a.get("temperature"), "seed": run_a.get("seed"),
                     "num_predict": run_a.get("num_predict"),
                     "model": run_a.get("model"), "prompt": run_a.get("prompt")},
        "compared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "conclusion": ("greedy 确定：两次全量逐字节一致" if not diffs else
                       f"greedy 不完全确定：{len(diffs)}/{len(common)} 题输出不同，"
                       "temperature=0 且 seed 固定，差异来自 ollama/GPU 浮点非确定性"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"n={len(common)} identical={not diffs} diffs={len(diffs)} "
          f"flipped={payload['flipped_correct']} out={args.out}")
    if diffs:
        for d in diffs[:5]:
            print(f"  {d['id']} first_diff={d['first_diff_char']} "
                  f"pred={d['pred_a']!r}->{d['pred_b']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
