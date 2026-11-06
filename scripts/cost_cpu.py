"""CPU 成本实测：num_gpu=0 强制 CPU 推理，测单题墙钟与 token。

与 GPU 的同题配置（cot prompt、num_predict=512、temperature=0）一致，
第 1 题作为预热丢弃（包含模型加载），其余取均值。
输出 JSON 给报告用：{"<model>": {n, mean_latency_ms, mean_prompt_tokens, mean_out_tokens}}。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_loop import bench  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/gsm8k_n60.jsonl")
    ap.add_argument("--models", default="qwen2.5:1.5b,qwen2.5:0.5b")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", default="reports/cost_cpu.json")
    args = ap.parse_args()

    items = bench.load_dataset(args.dataset)
    result: dict = {}
    for model in args.models.split(","):
        model = model.strip()
        if not model:
            continue
        rows = []
        for idx, item in enumerate(items[: args.limit + 1]):
            resp = bench.generate(model, bench.PROMPTS["cot"].format(question=item["question"]),
                                  num_predict=512, num_gpu=0, timeout=1800)
            if idx == 0:  # 预热（含模型加载），不计入
                continue
            rows.append(resp)
            print(f"  {model} CPU item {idx}: {resp['latency_ms']} ms, "
                  f"prompt={resp.get('prompt_eval_count')}, out={resp.get('eval_count')}",
                  flush=True)
        if not rows:
            continue
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result[model] = {
            "n": len(rows),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 1),
            "mean_prompt_tokens": round(sum(r.get("prompt_eval_count") or 0 for r in rows) / len(rows), 1),
            "mean_out_tokens": round(sum(r.get("eval_count") or 0 for r in rows) / len(rows), 1),
            "num_gpu": 0,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
