"""样本量-区间宽度收缩曲线：把不同 n 档的实测宽度画在一张图上，叠加理论 √n 曲线。

每个点是某一份落盘 run 的实际得分重算出的 Wilson / bootstrap 区间宽度（不是推算）；
理论曲线用该 run 全量档的 p 代入 2z√(p(1-p)/n)，只用来做参照。

用法：
  python3 scripts/width_curve.py \
    --point A-cot:60:results/runs/a_cot.jsonl \
    --point A-cot:300:results/runs/a_cot_n300.jsonl \
    --point A-cot:1319:results/runs/a_cot_n1319.jsonl ... \
    --out reports/width_curve.json --png reports/width_curve.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_loop import bench  # noqa: E402
from eval_loop.stats import accuracy_ci, wilson_interval  # noqa: E402

Z = 1.959963984540054
COLORS = {"A-cot": "#2563eb", "A-short": "#dc2626", "B-cot": "#059669"}
FALLBACK = ["#7c3aed", "#ea580c", "#0891b2"]


def load_point(label: str, n: int, path: str) -> dict:
    run = bench.read_run(path)
    recs = run["records"]
    actual_n = len(recs)
    correct = [r["correct"] for r in recs]
    p = sum(correct) / actual_n if actual_n else 0.0
    wil = wilson_interval(sum(correct), actual_n)
    boot = accuracy_ci(correct)
    return {
        "label": label, "model": run["model"], "prompt": run["prompt"],
        "file": path, "n_claimed": n, "n": actual_n, "p": round(p, 4),
        "wilson_ci95": [round(wil[0], 4), round(wil[1], 4)],
        "wilson_width": round(wil[1] - wil[0], 4),
        "bootstrap_ci95": [round(x, 4) for x in boot["ci95"]],
        "bootstrap_width": round(boot["ci95"][1] - boot["ci95"][0], 4),
    }


def build_series(points: list[dict]) -> list[dict]:
    series = []
    for label in sorted({p["label"] for p in points}):
        pts = sorted([p for p in points if p["label"] == label], key=lambda x: x["n"])
        full = pts[-1]
        p = full["p"]
        grid = list(range(40, max(1400, full["n"]) + 1, 10))
        series.append({
            "label": label, "model": full["model"], "prompt": full["prompt"],
            "points": pts,
            "theory": {
                "p": p,
                "n": grid,
                "width": [round(2 * Z * math.sqrt(p * (1 - p) / n), 4) for n in grid],
                "formula": "2·z·sqrt(p(1-p)/n), z=1.96",
            },
        })
    return series


def draw(series: list[dict], png_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    for i, s in enumerate(series):
        color = COLORS.get(s["label"], FALLBACK[i % len(FALLBACK)])
        ns = [p["n"] for p in s["points"]]
        ws = [p["wilson_width"] for p in s["points"]]
        bs = [p["bootstrap_width"] for p in s["points"]]
        ax.plot(ns, ws, "o-", color=color, label=f"{s['label']} Wilson measured")
        ax.plot(ns, bs, "s--", color=color, alpha=0.55, markersize=4,
                label=f"{s['label']} bootstrap measured")
        ax.plot(s["theory"]["n"], s["theory"]["width"], ":", color=color, lw=1.2,
                label=f"{s['label']} theory √n (p={s['theory']['p']:.3f})")
        # 同一 n 上三条线彼此很近：B-cot 标在点右上方，A-cot/A-short 标在点左下方，避免数字叠在一起
        dx, dy = {"A-cot": (-8, -11), "A-short": (-8, -11)}.get(s["label"], (3, 9))
        for p in s["points"]:
            ax.annotate(f"{p['wilson_width']:.3f}", (p["n"], p["wilson_width"]),
                        textcoords="offset points", xytext=(dx, dy),
                        ha="right" if dx < 0 else "left",
                        fontsize=7, color=color)
    ax.axhline(0.05, color="#6b7280", lw=1, ls="-")
    ax.text(50, 0.039, "target width 0.05", fontsize=7.5, color="#6b7280", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(46, 2100)
    ax.set_xticks([60, 300, 1319])
    ax.set_xticklabels(["60", "300", "1319 (full test)"])
    ax.set_xlabel("n (questions)")
    ax.set_ylabel("95% CI width")
    ax.set_title("GSM8K test: accuracy CI width vs sample size (measured vs √n theory)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", action="append", required=True,
                    help="label:n:path.jsonl（同一 label 至少给两档）")
    ap.add_argument("--out", default="reports/width_curve.json")
    ap.add_argument("--png", default="reports/width_curve.png")
    args = ap.parse_args()

    points = []
    for spec in args.point:
        label, n, path = spec.rsplit(":", 2)
        points.append(load_point(label, int(n), path))
    series = build_series(points)

    payload = {
        "series": series,
        "target_width": 0.05,
        "note": "每个点是该档落盘 run 的实际得分区间宽度（Wilson 与 bootstrap 各一条）；"
                "理论曲线用全量档 p 代入 2z√(p(1-p)/n) 作参照。n=60/300 与全量是三份独立抽样，"
                "不是嵌套子集。",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    draw(series, args.png)
    for s in series:
        pts = ", ".join(f"n={p['n']} w={p['wilson_width']:.3f}" for p in s["points"])
        print(f"{s['label']}: {pts}")
    print(f"wrote {args.out} {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
