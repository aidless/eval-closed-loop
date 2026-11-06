"""CLI：run / verify / gate / judge / calibrate / report / summary。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from . import bench, judge, report
from . import gate as gate_mod


def cmd_run(args: argparse.Namespace) -> int:
    meta = bench.run(args.dataset, args.model, args.prompt, args.out,
                     limit=args.limit, num_predict=args.num_predict,
                     num_gpu=args.num_gpu)
    records = bench.read_run(args.out)["records"]
    print(f"run={args.out} model={meta['model']} prompt={meta['prompt']} "
          f"n={len(records)} correct={sum(r['correct'] for r in records)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """从落盘的输出重算一遍分，证明判分是确定函数（不重跑模型）。"""
    payload, ok = {}, True
    for path in args.runs:
        run = bench.read_run(path)
        mismatches = []
        for rec in run["records"]:
            pred = bench.extract_pred(rec["output"])
            recomputed = bench.is_correct(pred, rec["gold_final"])
            if recomputed != rec["correct"] or pred != rec["pred"]:
                mismatches.append(rec["id"])
        key = os.path.basename(path)
        payload[key] = {
            "n": len(run["records"]),
            "mismatches": mismatches,
            "score_sha256": hashlib.sha256(
                json.dumps(sorted((r["id"], r["correct"]) for r in run["records"]),
                           sort_keys=True).encode()).hexdigest(),
        }
        ok = ok and not mismatches
        print(f"verify {key}: n={payload[key]['n']} mismatches={len(mismatches)} "
              f"score_sha={payload[key]['score_sha256'][:16]}")
    print("deterministic=" + ("true" if ok else "false"))
    return 0 if ok else 3


def cmd_gate(args: argparse.Namespace) -> int:
    cand = bench.read_run(args.candidate)
    base = bench.read_run(args.baseline)
    result = gate_mod.evaluate(cand["records"], base["records"])
    result["baseline_run"] = args.baseline
    result["candidate_run"] = args.candidate
    result = gate_mod.apply_report_only(result, args.report_only)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"verdict={result['verdict']} delta={result['delta']['delta']:+.3f} "
          f"ci95=[{result['delta']['ci95'][0]:+.3f}, {result['delta']['ci95'][1]:+.3f}] "
          f"flipped={len(result['flipped'])} rules={','.join(result['matched_rules'])} "
          f"report={args.out}")
    return gate_mod.exit_code(result)


def cmd_judge(args: argparse.Namespace) -> int:
    run = bench.read_run(args.records)
    meta = judge.run_judge(args.model, run["records"], args.out, mode=args.mode)
    print(f"judge model={meta['judge_model']} mode={meta['judge_mode']} "
          f"records={meta['records']} out={args.out}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    run = bench.read_run(args.records)
    human = {row["id"]: row for row in judge.load_jsonl(args.human)}
    models = args.judge_model or []
    labels = args.judge_labels or []
    modes = args.judge_mode or []
    pairs = []
    for idx, (model, path) in enumerate(zip(models, labels)):
        mode = modes[idx] if idx < len(modes) else "raw"
        pairs.append((model, judge.load_jsonl(path), mode))
    result = judge.calibrate(run["records"], human, pairs,
                             target_half_width=args.target_half_width)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    parts = [f"rule={result['rule']['kappa']:.3f}"]
    for j in result["judges"]:
        parts.append(f"{j['key']}={j['kappa']:.3f}")
    print(f"kappa(n={result['n']}) " + " ".join(parts) + f" out={args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    runs = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        run = bench.read_run(path)
        runs.append({"label": label, **run})
    gate_result = json.load(open(args.gate, encoding="utf-8")) if args.gate else None
    kappa = json.load(open(args.kappa, encoding="utf-8")) if args.kappa else None
    causes = None
    cause_path = args.causes or "human_labels/disagreement_causes.json"
    if cause_path and os.path.exists(cause_path):
        causes = json.load(open(cause_path, encoding="utf-8"))
    cpu_cost = None
    cpu_path = args.cpu_cost or "reports/cost_cpu.json"
    if cpu_path and os.path.exists(cpu_path):
        cpu_cost = json.load(open(cpu_path, encoding="utf-8"))
    curve = None
    curve_path = args.curve or "reports/width_curve.json"
    if curve_path and os.path.exists(curve_path):
        curve = json.load(open(curve_path, encoding="utf-8"))
    curve_png = args.curve_png or "reports/width_curve.png"
    stability = None
    stability_path = args.stability or "reports/stability.json"
    if stability_path and os.path.exists(stability_path):
        stability = json.load(open(stability_path, encoding="utf-8"))
    variance = None
    variance_path = args.variance or "reports/variance.json"
    if variance_path and os.path.exists(variance_path):
        variance = json.load(open(variance_path, encoding="utf-8"))
    dataset = args.dataset
    notes = args.note or []
    meta = {"path": dataset, "sha256": bench.sha256_file(dataset) if dataset else None}
    report.build(runs, gate_result, kappa, dataset_meta=meta,
                 out_path=args.out, notes=notes, causes=causes,
                 cpu_cost=cpu_cost, target_width=args.target_width,
                 curve=curve, curve_png=curve_png, stability=stability,
                 variance=variance)
    sjson = report.summary_json(runs, gate_result, kappa, args.summary_json,
                                cpu_cost=cpu_cost, curve=curve, stability=stability,
                                variance=variance)
    print(f"report={args.out} summary={sjson}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    for spec in args.run:
        label, path = spec.split("=", 1)
        run = bench.read_run(path)
        from .stats import accuracy_ci
        recs = run["records"]
        acc = accuracy_ci([r["correct"] for r in recs])
        print(f"{label}\t{run['model']}\t{run['prompt']}\tn={acc['n']}\t"
              f"acc={acc['point']:.3f}\tboot95=[{acc['ci95'][0]:.3f}, {acc['ci95'][1]:.3f}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eval-loop")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", choices=sorted(bench.PROMPTS), default="cot")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--num-gpu", type=int, default=None,
                   help="ollama num_gpu；0 表示强制 CPU 推理（成本实测用）")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify")
    p.add_argument("--runs", nargs="+", required=True)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("gate")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report-only", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("judge")
    p.add_argument("--records", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=sorted(judge.PROMPTS), default="raw")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("calibrate")
    p.add_argument("--records", required=True)
    p.add_argument("--human", required=True)
    p.add_argument("--judge-labels", action="append")
    p.add_argument("--judge-model", action="append")
    p.add_argument("--judge-mode", action="append")
    p.add_argument("--target-half-width", type=float, default=0.05,
                   help="κ CI 的目标半宽，用于反解所需标注条数")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("report")
    p.add_argument("--run", action="append", required=True, help="label=path.jsonl")
    p.add_argument("--gate")
    p.add_argument("--kappa")
    p.add_argument("--dataset")
    p.add_argument("--note", action="append")
    p.add_argument("--causes", default=None,
                   help="分歧错因归类 JSON，默认 human_labels/disagreement_causes.json")
    p.add_argument("--cpu-cost", default=None,
                   help="CPU 成本实测 JSON，默认 reports/cost_cpu.json")
    p.add_argument("--target-width", type=float, default=0.05,
                   help="样本量推算的目标区间宽度")
    p.add_argument("--curve", default=None,
                   help="宽度收缩曲线 JSON，默认 reports/width_curve.json")
    p.add_argument("--curve-png", default=None,
                   help="曲线 PNG，默认 reports/width_curve.png")
    p.add_argument("--stability", default=None,
                   help="稳定性比对 JSON，默认 reports/stability.json")
    p.add_argument("--variance", default=None,
                   help="同配置多次重跑的可复现性 JSON，默认 reports/variance.json")
    p.add_argument("--out", default="reports/index.html")
    p.add_argument("--summary-json", default="reports/summary.json")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("summary")
    p.add_argument("--run", action="append", required=True)
    p.set_defaults(func=cmd_summary)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
