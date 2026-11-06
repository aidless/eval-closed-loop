"""冻结评测子集：从本地 HF 缓存的 GSM8K test 划分里取固定样本，写成 jsonl。

只做一次；产物 data/gsm8k_n60.jsonl 进仓库，之后复现不再需要网络或 datasets 库。
类别（category）不是 GSM8K 官方字段，按金标解题行数分桶，报告里会明说这是派生指标。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys

SEED = 20261105


def gold_final(answer: str) -> str | None:
    """取 #### 后的数字串。"""
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    return m.group(1) if m else None


def step_count(answer: str) -> int:
    """金标推导里 #### 之前的非空行数，作为“几步题”的近似。"""
    body = answer.split("####")[0]
    return max(1, sum(1 for line in body.splitlines() if line.strip()))


def bucket(steps: int) -> str:
    if steps <= 2:
        return "steps_1_2"
    if steps <= 4:
        return "steps_3_4"
    return "steps_5_plus"


def stratified_ids(ds, n: int, seed: int) -> tuple[list[int], dict]:
    """按 steps 桶做比例分层抽样（最大余数法分配名额）。

    比例分层保留全量 test 的难度分布；返回 (picked_ids, allocation)。
    """
    from collections import defaultdict

    by: dict[str, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        if gold_final(ds[i]["answer"]) is None:
            continue
        by[bucket(step_count(ds[i]["answer"]))].append(i)
    total = sum(len(v) for v in by.values())
    raw = {b: n * len(v) / total for b, v in by.items()}
    alloc = {b: int(raw[b]) for b in by}
    for b in sorted(by, key=lambda b: (raw[b] - alloc[b], b), reverse=True)[: n - sum(alloc.values())]:
        alloc[b] += 1

    rng = random.Random(seed)
    picked: list[int] = []
    for b in sorted(by):
        picked.extend(rng.sample(by[b], alloc[b]))
    return picked, {b: {"population": len(by[b]), "sampled": alloc[b]} for b in sorted(by)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="data/gsm8k_n60.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stratify", action="store_true",
                    help="按 steps 桶比例分层抽样；默认保持旧的纯随机抽样")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset  # 只有冻结这一步需要

    ds = load_dataset("openai/gsm8k", "main", split=args.split,
                      download_mode="reuse_dataset_if_exists")

    import random
    if args.stratify:
        picked, alloc = stratified_ids(ds, args.n, SEED)
        picked.sort()
        print(f"stratified allocation: {json.dumps(alloc, ensure_ascii=False)}")
    else:
        rng = random.Random(SEED)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        picked = sorted(idx[: args.n])

    rows = []
    for i in picked:
        ex = ds[i]
        gf = gold_final(ex["answer"])
        if gf is None:
            continue
        steps = step_count(ex["answer"])
        rows.append({
            "id": f"gsm8k-{args.split}-{i}",
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "gold_final": gf,
            "steps": steps,
            "category": bucket(steps),
        })

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"wrote {out} n={len(rows)} seed={SEED} sha256={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
