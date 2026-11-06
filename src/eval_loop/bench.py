"""benchmark runner：调本地 ollama，跑固定子集，做答案抽取与判分。

设计取舍：
- 零第三方依赖，只用标准库 + ollama HTTP API（127.0.0.1），断网也能跑。
- 逐题 append 写 jsonl，长跑中断可以续跑；temperature=0、固定 seed，尽量可复现。
- 判分只用「最终数字是否相等」，不做语义判分；语义判分交给 judge.py。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from typing import Any

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# 两套 prompt：cot 是正常配置；short 是刻意去掉推理的退化配置，用来喂回归门禁
PROMPTS = {
    "cot": (
        "{question}\n\n"
        "Solve the problem step by step, then end with a final line of exactly this form:\n"
        "#### <number>"
    ),
    "short": (
        "{question}\n\n"
        "Do not explain. Reply with only one line:\n"
        "#### <number>"
    ),
}

_NUM_RE = re.compile(r"[-+]?(?:\d+(?:,\d+)*)(?:\.[0-9]+)?")


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def generate(model: str, prompt: str, *, num_predict: int = 512,
             temperature: float = 0.0, seed: int = 42,
             timeout: float = 600.0, num_gpu: int | None = None) -> dict[str, Any]:
    """单次补全。返回 response / 计时 / prompt+生成 token 数。

    num_gpu=0 时强制 CPU 推理（ollama 的 num_gpu 选项），用于实测 CPU 成本。
    """
    options: dict[str, Any] = {
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
        "top_p": 1.0,
    }
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - t0
    return {
        "response": body.get("response", ""),
        "latency_ms": int(elapsed * 1000),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
    }


def _normalize_number(text: str) -> float | None:
    text = text.strip().replace(",", "").replace("$", "").replace("%", "")
    text = text.strip().rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def extract_pred(text: str) -> float | None:
    """优先取最后一个 #### 之后的数字；没有 #### 就取全文最后一个数字。"""
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = _NUM_RE.search(tail)
        if m:
            return _normalize_number(m.group(0))
    nums = _NUM_RE.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return None


def is_correct(pred: float | None, gold: str) -> bool:
    g = _normalize_number(gold)
    if pred is None or g is None:
        return False
    return abs(pred - g) <= 1e-4


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(dataset_path: str, model: str, prompt_id: str, out_path: str,
        *, limit: int | None = None, num_predict: int = 512,
        temperature: float = 0.0, seed: int = 42,
        num_gpu: int | None = None) -> dict:
    """跑一遍并落 jsonl；已有记录的 id 直接跳过（断点续跑）。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # 同一输出文件只允许一个进程写，避免并发 append 出重复记录
    lock = open(out_path + ".lock", "w")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_locked(dataset_path, model, prompt_id, out_path,
                           limit=limit, num_predict=num_predict,
                           temperature=temperature, seed=seed, num_gpu=num_gpu)
    finally:
        lock.close()


def _run_locked(dataset_path: str, model: str, prompt_id: str, out_path: str,
                *, limit: int | None, num_predict: int,
                temperature: float, seed: int, num_gpu: int | None = None) -> dict:
    items = load_dataset(dataset_path)
    if limit:
        items = items[:limit]
    template = PROMPTS[prompt_id]
    done: dict[str, dict] = {}
    if os.path.exists(out_path):
        for row in load_dataset(out_path):
            if row.get("type") == "record":
                done[row["id"]] = row

    meta = {
        "type": "meta",
        "model": model,
        "prompt": prompt_id,
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
        "num_gpu": num_gpu,
        "dataset": dataset_path,
        "dataset_sha256": sha256_file(dataset_path),
        "n_planned": len(items),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    # 已存在的文件若来自别的配置，直接报错，避免混跑
    if os.path.exists(out_path):
        first = load_dataset(out_path)[0]
        for key in ("model", "prompt", "temperature", "seed", "num_predict",
                    "num_gpu", "dataset_sha256"):
            if first.get(key) != meta.get(key):
                raise SystemExit(
                    f"{out_path} 已存在且配置不同（{key}: {first.get(key)} != "
                    f"{meta.get(key)}），换一个输出路径或先删除")

    fresh = not os.path.exists(out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        if fresh:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for item in items:
            if item["id"] in done:
                continue
            prompt = template.format(question=item["question"])
            resp = generate(model, prompt, num_predict=num_predict,
                            temperature=temperature, seed=seed, num_gpu=num_gpu)
            pred = extract_pred(resp["response"])
            rec = {
                "type": "record",
                "id": item["id"],
                "category": item["category"],
                "steps": item["steps"],
                "gold_final": item["gold_final"],
                "question": item["question"],
                "output": resp["response"],
                "pred": pred,
                "correct": is_correct(pred, item["gold_final"]),
                "latency_ms": resp["latency_ms"],
                "prompt_eval_count": resp.get("prompt_eval_count"),
                "eval_count": resp["eval_count"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {item['id']} pred={pred} gold={item['gold_final']} "
                  f"correct={rec['correct']} {rec['latency_ms']}ms", flush=True)

    return meta


def read_run(path: str) -> dict:
    """读回一次 run：meta + records（按 id 去重，防止并发 append 留下重复）。"""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta = rows[0]
    records, seen = [], set()
    for r in rows:
        if r.get("type") != "record" or r["id"] in seen:
            continue
        seen.add(r["id"])
        records.append(r)
    return {
        "path": path,
        "model": meta["model"],
        "prompt": meta["prompt"],
        "dataset": meta["dataset"],
        "dataset_sha256": meta["dataset_sha256"],
        "created_at": meta.get("created_at"),
        "temperature": meta.get("temperature"),
        "seed": meta.get("seed"),
        "num_predict": meta.get("num_predict"),
        "num_gpu": meta.get("num_gpu"),
        "records": records,
    }
