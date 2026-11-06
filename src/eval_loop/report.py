"""一页静态 HTML 报告：分数 / 区间 / 门禁判决 / judge κ，零外部依赖。"""
from __future__ import annotations

import base64
import datetime
import html
import json
import os
import time

from .stats import accuracy_ci, wilson_interval

CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
       margin: 0; background: #f6f7f9; color: #1b1f24; line-height: 1.55; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 26px; margin: 0 0 6px; }
h2 { font-size: 19px; margin: 36px 0 12px; border-left: 4px solid #2563eb; padding-left: 10px; }
.sub { color: #5b6472; font-size: 14px; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,.06); margin: 10px 0 6px; }
th, td { border-bottom: 1px solid #e6e8ec; padding: 8px 10px; text-align: left;
         vertical-align: top; }
th { background: #f0f2f5; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 6px; font-weight: 700;
         font-size: 13px; letter-spacing: .3px; }
.PASS { background: #dcfce7; color: #166534; }
.BLOCK { background: #fee2e2; color: #991b1b; }
.FAIL { background: #ffedd5; color: #9a3412; }
.REVIEW { background: #fef9c3; color: #854d0e; }
.INCOMPLETE { background: #e5e7eb; color: #374151; }
.card { background: #fff; border-radius: 10px; padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,.06); margin: 12px 0; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.muted { color: #6b7280; }
.warn { background: #fff7ed; border-left: 4px solid #f59e0b; padding: 12px 16px;
        border-radius: 6px; font-size: 14px; }
ul { margin: 8px 0 0 18px; }
"""


def _e(x) -> str:
    return html.escape(str(x))


def _fmt(x, nd=3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return _e(x)


def _score_table(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        recs = run["records"]
        n = len(recs)
        correct = [r["correct"] for r in recs]
        wil = wilson_interval(sum(correct), n)
        boot = accuracy_ci(correct)
        rows.append(f"""<tr>
          <td>{_e(run['label'])}</td>
          <td class="mono">{_e(run['model'])}</td>
          <td>{_e(run['prompt'])}</td>
          <td class="num">{n}</td>
          <td class="num">{sum(correct)}</td>
          <td class="num"><b>{boot['point']:.3f}</b></td>
          <td class="num">[{wil[0]:.3f}, {wil[1]:.3f}]</td>
          <td class="num">{wil[1] - wil[0]:.3f}</td>
          <td class="num">[{boot['ci95'][0]:.3f}, {boot['ci95'][1]:.3f}]</td>
          <td class="num">{boot['ci95'][1] - boot['ci95'][0]:.3f}</td>
        </tr>""")
    return f"""<table>
      <tr><th>run</th><th>model</th><th>prompt</th><th class="num">n</th>
      <th class="num">correct</th><th class="num">accuracy</th>
      <th class="num">Wilson 95%</th><th class="num">W 宽</th>
      <th class="num">bootstrap 95%</th><th class="num">B 宽</th></tr>
      {''.join(rows)}
    </table>"""


def _category_table(runs: list[dict]) -> str:
    cats = sorted({r["category"] for run in runs for r in run["records"]})
    head = "".join(f"<th class='num'>{_e(c)}</th>" for c in cats)
    body = []
    for run in runs:
        cells = []
        for c in cats:
            recs = [r for r in run["records"] if r["category"] == c]
            if recs:
                acc = sum(r["correct"] for r in recs) / len(recs)
                cells.append(f"<td class='num'>{acc:.3f}<span class='muted'> (n={len(recs)})</span></td>")
            else:
                cells.append("<td class='num'>—</td>")
        body.append(f"<tr><td>{_e(run['label'])}</td>{''.join(cells)}</tr>")
    return f"""<table><tr><th>run</th>{head}</tr>{''.join(body)}</table>"""


def _gate_section(gate: dict) -> str:
    v = gate["verdict"]
    gates = "".join(
        f"<tr><td class='mono'>{_e(g['id'])}</td><td class='mono'>{_e(g['rule'])}</td>"
        f"<td>{_e(g['result'])}</td><td>{_e(g.get('detail') or '')}</td></tr>"
        for g in gate["gates"])
    flipped = "".join(
        f"<tr><td class='mono'>{_e(x['id'])}</td><td>{_e(x['category'])}</td>"
        f"<td class='mono'>{_e(x['baseline_pred'])}</td>"
        f"<td class='mono'>{_e(x['candidate_pred'])}</td>"
        f"<td class='mono'>{_e(x['gold_final'])}</td></tr>"
        for x in gate["flipped"])
    flipped_block = (f"<h3>回退到错的具体题（{len(gate['flipped'])}）</h3>"
                     f"<table><tr><th>id</th><th>category</th><th>baseline pred</th>"
                     f"<th>candidate pred</th><th>gold</th></tr>{flipped}</table>"
                     if gate["flipped"] else
                     "<p class='muted'>没有 baseline 对、candidate 错的题。</p>")
    cats = "".join(
        f"<tr><td>{_e(c)}</td><td class='num'>{v['n']}</td>"
        f"<td class='num'>{v['baseline']:.3f}</td><td class='num'>{v['candidate']:.3f}</td>"
        f"<td class='num'>{v['delta']:+.3f}</td><td class='num'>{v['flipped']}</td></tr>"
        for c, v in gate["per_category"].items())
    d = gate["delta"]
    return f"""
    <h2>回归门禁</h2>
    <div class="card">
      <p><span class="badge {v}">{_e(v)}</span>
         <span class="muted">命中规则：{_e(', '.join(gate['matched_rules']) or '无')}</span></p>
      <p>baseline {gate['baseline']['rate']:.3f} (n={gate['baseline']['n']}) →
         candidate {gate['candidate']['rate']:.3f} (n={gate['candidate']['n']}),
         配对 delta <b>{d['delta']:+.3f}</b>, 95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}],
         bootstrap {d['iterations']} 次, seed={d['seed']}。</p>
      <table><tr><th>gate</th><th>rule</th><th>result</th><th>detail</th></tr>{gates}</table>
    </div>
    <h3>分类别 delta</h3>
    <table><tr><th>category</th><th class="num">n</th><th class="num">baseline</th>
      <th class="num">candidate</th><th class="num">delta</th>
      <th class="num">回退题数</th></tr>{cats}</table>
    {flipped_block}"""


def _power_section(runs: list[dict], target_width: float = 0.05) -> str:
    """区间宽度的样本量推算：宽度随 1/sqrt(n) 收缩（同一 p 的近似）。"""
    z = 1.959963984540054
    rows = []
    for run in runs:
        recs = run["records"]
        n = len(recs)
        p = sum(r["correct"] for r in recs) / n if n else 0.0
        wil = wilson_interval(sum(r["correct"] for r in recs), n)
        boot = accuracy_ci([r["correct"] for r in recs])
        w_w = wil[1] - wil[0]
        w_b = boot["ci95"][1] - boot["ci95"][0]
        n_boot = n * (w_b / target_width) ** 2
        n_wil = (2 * z / target_width) ** 2 * p * (1 - p)
        mark_b = "（超出本档 n，未验证）" if n_boot > n else "（本档已覆盖）"
        rows.append(f"""<tr><td>{_e(run['label'])}</td><td class="num">{n}</td>
          <td class="num">{p:.3f}</td><td class="num">{w_w:.3f}</td>
          <td class="num">{w_b:.3f}</td><td class="num">{n_boot:,.0f} {mark_b}</td>
          <td class="num">{max(0, n_boot - n):,.0f}</td>
          <td class="num">{n_wil:,.0f}</td></tr>""")
    return f"""
    <h3>区间宽度与样本量推算（目标宽度 {target_width:.2f}）</h3>
    <div class="card">
      <p class="muted">推算公式：bootstrap 按宽度 ∝ 1/√n 反解 n = n₀·(w₀/{target_width:.2f})²；
        Wilson 用解析近似 n = (2z/w)²·p(1−p)，z=1.96。
        这是“保持当前 p 不变”的推算；若所需 n 超过本档实际 n，标为未验证，不当成已跑结果。</p>
      <table><tr><th>run</th><th class="num">n</th><th class="num">p</th>
        <th class="num">Wilson 宽</th><th class="num">bootstrap 宽</th>
        <th class="num">达 {target_width:.2f} 所需 n（bootstrap 反解）</th>
        <th class="num">还需增加</th>
        <th class="num">所需 n（Wilson 解析）</th></tr>{''.join(rows)}</table>
    </div>"""


def _wall_clock_minutes(run: dict) -> float | None:
    """整轮实际墙钟：启动时间（meta.created_at）到文件末次落盘（mtime）。

    与「Σ单题 latency」不同，这一列包含模型加载/落盘/进程开销，是对外报价该用的数。
    """
    start, path = run.get("created_at"), run.get("path")
    if not start or not path or not os.path.exists(path):
        return None
    try:
        started = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
    except ValueError:
        return None
    ended = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    return (ended - started).total_seconds() / 60


def _cost_section(runs: list[dict], cpu_cost: dict | None) -> str:
    rows = []
    for run in runs:
        recs = run["records"]
        n = len(recs)
        pt = [r.get("prompt_eval_count") or 0 for r in recs]
        ot = [r.get("eval_count") or 0 for r in recs]
        lat = [r["latency_ms"] for r in recs]
        mean_tokens = (sum(pt) + sum(ot)) / n if n else 0
        mean_ms = sum(lat) / n if n else 0
        total_min = sum(lat) / 1000 / 60
        wall_min = _wall_clock_minutes(run)
        wall_cell = "—" if wall_min is None else f"{wall_min:,.0f} min"
        rows.append(f"""<tr><td>{_e(run['label'])}</td><td class="num">{n:,}</td>
          <td class="num">{sum(pt)/n:,.0f}</td><td class="num">{sum(ot)/n:,.0f}</td>
          <td class="num">{mean_tokens:,.0f}</td><td class="num">{mean_ms:,.0f}</td>
          <td class="num">{total_min:,.0f} min</td>
          <td class="num">{wall_cell}</td>
          <td class="num">{(sum(pt)+sum(ot))/1e6:.3f} M</td></tr>""")
    cpu_rows = "".join(
        f"<tr><td>{_e(m)}（CPU, num_gpu=0）</td><td class='num'>{v['n']}</td>"
        f"<td class='num'>{v['mean_prompt_tokens']:,.0f}</td>"
        f"<td class='num'>{v['mean_out_tokens']:,.0f}</td>"
        f"<td class='num'>{v['mean_latency_ms']:,.0f}</td>"
        f"<td class='num'>{v['mean_latency_ms']/1000*v['n']/60:.1f} min</td></tr>"
        for m, v in (cpu_cost or {}).items())
    cpu_block = (f"""<h3>CPU 抽样实测（num_gpu=0，同 prompt/生成上限）</h3>
      <p class="muted">只跑抽样，<b>不外推到全量</b>：表中「抽样总时长」是这些题真实耗时的加总。</p>
      <table><tr><th>model</th><th class="num">抽样题数</th>
        <th class="num">prompt tokens/题</th><th class="num">生成 tokens/题</th>
        <th class="num">墙钟/题</th><th class="num">抽样总时长</th></tr>{cpu_rows}</table>"""
                if cpu_rows else
                "<p class='muted'>CPU 成本：<b>未验证</b>（本机未做 num_gpu=0 实测）。</p>")
    return f"""
    <h2>成本（实测，不做外推）</h2>
    <div class="card">
      <p class="muted">token 数取自 ollama 返回的 prompt_eval_count / eval_count；
        墙钟是单题请求耗时（含 prefill+decode，不含模型加载），按串行、无并发加总。
        本机 GPU 同时有另一训练进程占用约 3.3 GB 显存（见局限），耗时含该并发影响。
        「整轮实测墙钟」= 启动时间到末次落盘，包含模型加载与进程开销；
        「Σ单题」是各题请求耗时之和。两者都只适用于本机这一轮，不做跨机器外推。</p>
      <table><tr><th>run</th><th class="num">n</th>
        <th class="num">prompt tokens/题</th><th class="num">生成 tokens/题</th>
        <th class="num">合计 tokens/题</th><th class="num">墙钟/题 (ms)</th>
        <th class="num">Σ单题墙钟</th><th class="num">整轮实测墙钟</th>
        <th class="num">实测总 token</th></tr>
        {''.join(rows)}</table>
      {cpu_block}
    </div>"""


def _width_curve_section(curve: dict | None, png_path: str | None) -> str:
    """样本量-宽度收缩曲线：实测点 + √n 理论线，图内嵌 PNG（单文件报告）。"""
    if not curve or not curve.get("series"):
        return ""
    rows = []
    for s in curve["series"]:
        for p in s["points"]:
            rows.append(
                f"<tr><td>{_e(s['label'])}</td><td class='mono'>{_e(s['model'])}</td>"
                f"<td class='num'>{p['n']:,}</td><td class='num'>{p['p']:.3f}</td>"
                f"<td class='num'>{p['wilson_width']:.4f}</td>"
                f"<td class='num'>{p['bootstrap_width']:.4f}</td></tr>")
    img = ""
    if png_path and os.path.exists(png_path):
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        img = (f"<img alt='CI width vs n' style='width:100%;max-width:1100px;"
               f"border-radius:8px' src='data:image/png;base64,{b64}'>")
    # 读图结论：直接用实测点的相邻收缩率算，不写死数字
    takes = []
    for s in curve["series"]:
        pts = sorted(s["points"], key=lambda p: p["n"])
        if len(pts) < 2:
            continue
        bits = []
        for a, b in zip(pts, pts[1:]):
            cut = 1 - b["wilson_width"] / a["wilson_width"] if a["wilson_width"] else 0
            bits.append(f"{a['n']:,}→{b['n']:,} 宽 {a['wilson_width']:.3f}→{b['wilson_width']:.3f}（{cut*100:+.0f}%）")
        takes.append(f"<li>{_e(s['label'])}：" + "；".join(bits) + "</li>")
    takes_block = (f"<p class='muted'>相邻档实测收缩（Wilson 宽）：</p><ul class='muted'>{''.join(takes)}</ul>"
                   if takes else "")
    # 全量档是否达到 0.05：直接拿最大 n 的实测点比，不写死结论
    met = []
    for s in curve["series"]:
        pts = sorted(s["points"], key=lambda p: p["n"])
        if not pts:
            continue
        p = pts[-1]
        w = p["wilson_width"]
        if w <= 0.05:
            met.append(f"{_e(s['label'])} {w:.4f}（n={p['n']:,}）达标")
        else:
            met.append(f"{_e(s['label'])} {w:.4f}（n={p['n']:,}）超 0.05 共 {w - 0.05:+.4f}")
    met_block = (f"<p>目标宽度 ≤0.05 的全量档核对：" + "；".join(met) + "。</p>" if met else "")
    return f"""
    <h2>样本量-宽度收缩曲线</h2>
    <div class="card">
      <p class="muted">圆点/方块是该档落盘 run 的<b>实测</b> Wilson / bootstrap 区间宽度；
        虚线是对应全量档 p 的 √n 理论线（2z√(p(1−p)/n)），只作参照。
        n=60 / n=300 / 全量是三份独立抽样，不是嵌套子集。</p>
      {img}
      <table><tr><th>run</th><th>model</th><th class="num">n</th><th class="num">p</th>
        <th class="num">Wilson 宽</th><th class="num">bootstrap 宽</th></tr>
        {''.join(rows)}</table>
      {takes_block}
      {met_block}
    </div>"""


def _stability_section(st: dict | None) -> str:
    """贪心解码两次全量重跑的逐题一致性。"""
    if not st:
        return ""
    if st.get("identical"):
        return f"""
    <h2>稳定性（A-cot 全量跑两次）</h2>
    <div class="card"><p><b>逐题输出完全一致</b>（n={st['n_compared']}，
      两次 run 全部 {st['n_compared']} 条 output 字符串逐字节相同）。
      结论：temperature=0、固定 seed 下 greedy 解码在本机可复现，单次结果可用于对外报价。</p>
      <p class="muted">对比文件：{_e(st.get('baseline'))} vs {_e(st.get('repeat'))}；
      对比时间 {_e(st.get('compared_at'))}。</p></div>"""
    diffs = st.get("differences") or []
    rep = st.get("repeat_test") or {}
    rep_block = ""
    if rep:
        rep_block = (
            f"<p class='warn'>归因实验：抽 {rep['n_ids']} 题"
            f"（{rep['n_diff_ids']} 题两次 run 有差异 + {rep['n_same_ids']} 题无差异），"
            f"用<b>完全相同的 payload</b>（temperature=0、seed=42、num_predict=512）各请求 {rep['repeats']} 次："
            f"<b>{rep['unstable_ids']} 题在同一批请求里就出现不一致</b>，"
            f"{rep['stable_ids']} 题 {rep['repeats']} 次全同。"
            f"说明差异不是 seed/采样参数（这些已固定），而是 GPU 推理的浮点归约/批内并行不保证逐位可复现；"
            f"对温度=0 的贪心解码，logits 尾数差异会在近似并列处翻转 argmax，然后自回归放大。</p>"
            f"<p class='muted'>证据文件：reports/stability_repeat.json；"
            f"对比口径：模型 {_e(rep.get('model'))}，GPU "
            f"{_e(rep.get('num_gpu') if rep.get('num_gpu') is not None else '默认（未指定 num_gpu，由 ollama 自动分层）')}。</p>")
    rows = "".join(
        f"<tr><td class='mono'>{_e(d['id'])}</td><td>{_e(d.get('category'))}</td>"
        f"<td class='num'>{_e(d.get('correct_a'))}</td>"
        f"<td class='num'>{_e(d.get('correct_b'))}</td>"
        f"<td class='mono'>{_e((d.get('pred_a'), d.get('pred_b')))}</td>"
        f"<td class='num'>{_e(d.get('first_diff_char'))}</td></tr>" for d in diffs[:40])
    return f"""
    <h2>稳定性（A-cot 全量跑两次）</h2>
    <div class="card">
      <p><b>不一致 {len(diffs)} / {st['n_compared']} 题</b>：greedy 在本机<b>不是</b>逐字节确定，
        差异定位见下表（只列前 40 条；完整结果见 reports/stability.json）。</p>
      {rep_block}
      <table><tr><th>id</th><th>category</th><th class="num">run1 correct</th>
        <th class="num">run2 correct</th><th>pred (run1 / run2)</th>
        <th class="num">首个差异字符位</th></tr>{rows}</table>
    </div>"""


def _variance_section(var: dict | None) -> str:
    """同配置 n=300 × 3 次完整重跑：分数层面与逐题层面的可复现性下界。"""
    if not var or not var.get("runs"):
        return ""
    runs = var["runs"]
    probe = var.get("probe", {})
    score = var.get("score_level", {})
    pq = var.get("per_question", {})
    rec = var.get("noise_model", {}).get("recommendation", {})
    m300 = rec.get("measured_at_n300", {})
    scaled = rec.get("scaled_unverified", {})
    k = probe.get("repeats", len(runs))
    n = probe.get("n", 0)

    run_rows = "".join(
        f"<tr><td>{_e(r['label'])}</td><td class='num'>{r['n']}</td>"
        f"<td class='num'>{r['passes']}</td><td class='num'><b>{r['accuracy']:.4f}</b></td>"
        f"<td class='num mono'>[{r['wilson_ci95'][0]:.3f}, {r['wilson_ci95'][1]:.3f}]</td>"
        f"<td class='num'>{r['latency_ms_median']:,.0f}</td>"
        f"<td class='mono'>{_e(r.get('created_at'))}</td></tr>" for r in runs)

    pair_rows = "".join(
        f"<tr><td>{_e(p['a'])} vs {_e(p['b'])}</td>"
        f"<td class='num'>{p['delta_pp']:+.2f} pp</td>"
        f"<td class='num'>{p['a_only_correct']} / {p['b_only_correct']}</td>"
        f"<td class='num'>{p['flipped']}（{p['flip_rate'] * 100:.2f}%）</td>"
        f"<td class='num'>{p.get('same_correct', 0)}/{n}</td>"
        f"<td class='num'>{p.get('same_output', 0)}/{n}</td>"
        f"<td class='mono'>{_e(p['paired_bootstrap_ci95'])}</td>"
        f"<td><span class='badge {p['gate_verdict']}'>{_e(p['gate_verdict'])}</span></td></tr>"
        for p in score.get("pairs", []))

    bucket_rows = "".join(
        f"<tr><td>{_e(b['bucket'])}</td><td class='num'>{b['n']}</td>"
        f"<td class='num'>{b['flipped']}</td><td class='num'>{b['flip_rate'] * 100:.1f}%</td>"
        f"<td class='num'>{b['flip_share'] * 100:.1f}%</td></tr>"
        for b in pq.get("by_bucket", []))

    dist = pq.get("correct_count_dist", {})
    dist_rows = "".join(
        f"<tr><td class='num'>{x} / {k}</td><td class='num'>{dist.get(str(x), 0)}</td>"
        f"<td class='num'>{dist.get(str(x), 0) / pq['mixed'] * 100:.1f}%</td></tr>"
        for x in range(1, k)) if pq.get("mixed") else ""

    ex_rows = "".join(
        f"<tr><td class='mono'>{_e(x['id'])}</td><td>{_e(x['bucket'])}</td>"
        f"<td class='num'>{_e(x['gold'])}</td>"
        f"<td class='mono'>{_e(x['preds'])}</td><td class='mono'>{_e(x['correct'])}</td>"
        f"<td class='num'>{x['first_diff_char']}</td>"
        f"<td class='num'>{'是' if x.get('near_miss') else '否'}</td></tr>"
        for x in pq.get("examples", []))

    med = pq.get("first_diff_rel_median")
    if med is None:
        pos_txt = ""
    elif med >= 0.5:
        pos_txt = (f"两次输出的首个差异点中位落在第 {med * 100:.0f}% 字符位置，"
                   "多数是推理尾部的近似并列翻转，而不是从头走了另一条推理路径。")
    else:
        pos_txt = (f"两次输出的首个差异点中位落在第 {med * 100:.0f}% 字符位置，"
                   "分叉出现得较早，不只是尾部措辞差异。")
    near = pq.get("near_miss_questions", 0)
    unp = pq.get("unparsable_questions", 0)
    if near:
        near_txt = (f"其中 {near} 题至少一次给出与金标相对误差 ≤1% 的数值（近失），"
                    f"抽不出数字的 {unp} 题")
    else:
        near_txt = (f"没有出现相对误差 ≤1% 的数值近失（抽不出数字的 {unp} 题），"
                    "翻转是整条推理链换支，而不是末尾算术差一两个单位")
    buckets = pq.get("by_bucket", [])
    buck_txt = ""
    if buckets:
        lo = min(buckets, key=lambda b: b["flip_rate"])
        hi = max(buckets, key=lambda b: b["flip_rate"])
        buck_txt = (f"桶内翻转率从 {_e(lo['bucket'])} 的 {lo['flip_rate'] * 100:.1f}%"
                    f" 到 {_e(hi['bucket'])} 的 {hi['flip_rate'] * 100:.1f}%。")

    env = probe.get("env_notes") or []
    env_block = ("<p class='muted'>三次 run 的机器状态：" + "；".join(_e(x) for x in env) + "</p>"
                 if env else "")

    # 方差是否均匀：有没有两次 run 判分完全一致
    ident = score.get("pairs_identical_correct") or []
    involved = score.get("runs_in_all_flip_pairs") or []
    if ident and involved and len(involved) < len(runs):
        ident_pairs = "、".join(_e(x) for x in ident)
        spread_txt = (f"三次<b>并非均匀抖动</b>：{ident_pairs} 的逐题判分完全一致，"
                      f"{score.get('spread_pp', 0):.2f} pp 的跨度全部来自 "
                      f"{_e('、'.join(involved))} 与其余 run 的差异，"
                      "且这组翻转题在不同对比里完全相同。")
    else:
        spread_txt = (f"三次两两之间都有差异，最大跨度 {score.get('spread_pp', 0):.2f} pp，"
                      "没有出现两次完全一致的 run。")
    scaled_txt = ""
    if scaled:
        parts = "；".join(f"n={k2} → 3σ≈{v['three_sigma_pp']:.2f} pp" for k2, v in scaled.items())
        scaled_txt = f"按 q/n 缩放到其他样本量：{parts}（<b>推算，未验证</b>）。"
    three300 = m300.get("three_sigma_pp") or 0
    margin300 = (rec.get("current_max_regression", 0.05) * 100) / three300 if three300 else 0
    s60 = (scaled.get("60") or {}).get("three_sigma_pp")
    s1319 = (scaled.get("1319") or {}).get("three_sigma_pp")
    margin_txt = (f"在 n={n} 实测下，5pp 阈值相当于 3σ 的 {margin300:.1f}×"
                  if margin300 else "")
    if s60 and s1319:
        margin_txt += (f"；按推算 n=1,319 时约 {5 / s1319:.1f}×，"
                       f"n=60 时约 {5 / s60:.1f}×")
    warn_txt = ""
    if s60 and s60 > rec.get("current_max_regression", 0.05) * 100:
        warn_txt = (f"<b>注意：n=60 的 3σ 推算值 {s60:.2f} pp 已经超过当前 5pp 阈值，"
                    "噪声本身就能跨过线——n=60 的门禁判决只能当方向性信号，不能单独作为放行/阻断依据。</b>")
    return f"""
    <h2>分数可复现性的实测下界（A-cot / n={n} × {k} 次）</h2>
    <div class="card">
      <p>同一配置（同一 {n} 题、同一顺序、temperature={_e(probe.get('temperature'))}、
        seed={_e(probe.get('seed'))}）完整重跑 {k} 次，
        准确率分别为 <b>{', '.join(f"{a:.4f}" for a in score.get('accuracies', []))}</b>；
        两两最大差 <b>{score.get('max_abs_pair_delta_pp', 0):.2f} pp</b>，
        分数层面最大跨度 <b>{score.get('spread_pp', 0):.2f} pp</b>。
        1 题 = {100 / n if n else 0:.2f} pp，分数的抖动是“几道题”的量级。
        {spread_txt}</p>
      <p><b>逐题</b>：{n} 题里 <b>{pq.get('mixed', 0)} 题</b>（{pq.get('flip_rate', 0) * 100:.2f}%）
        三次不完全一致，其余 {pq.get('unanimous_correct', 0)} 题全对、
        {pq.get('unanimous_wrong', 0)} 题全错。翻转题中
        {near_txt}；
        {pos_txt}{buck_txt}</p>
      <p><b>门禁 Δ 阈值</b>：由逐题翻转估计同配置两次跑的 Δ 标准差
        <b>sd={m300.get('sd_delta_pp', 0):.2f} pp</b>（n={n} 实测）；
        要 95% 不被噪声触发，|Δ| 阈值至少 1.96σ={m300.get('one_sided_95_pp', 0):.2f} pp，
        保守取 3σ=<b>{m300.get('three_sigma_pp', 0):.2f} pp</b>
        （实测最大 |Δ| 为 {m300.get('observed_max_abs_delta_pp', 0):.2f} pp，
        取二者较大值 → 建议下限 {m300.get('threshold_floor_pp', 0):.2f} pp，
        约 {m300.get('questions_equivalent', 0):.2f} 道题）。
        当前 max_regression={rec.get('current_max_regression', 0.05) * 100:.0f}%：
        {margin_txt}；所以本轮的 BLOCK（Δ=-0.603）不可能是噪声触发。{scaled_txt}{warn_txt}</p>
      <p class="muted">这三次是<b>独立探针</b>：不与主表 n=300 那一次混算，主表 A-cot / A-short / B-cot
        数字未改写；区间宽度按 n={n} 报告。三次 run 之间留了间隔、各自独立进程，不并发。</p>
      {env_block}
      <table><tr><th>run</th><th class="num">n</th><th class="num">答对</th>
        <th class="num">准确率</th><th class="num">Wilson 95%</th>
        <th class="num">单题墙钟中位 (ms)</th><th>启动时间</th></tr>{run_rows}</table>
      <table><tr><th>两两对比</th><th class="num">Δ 准确率</th>
        <th class="num">仅前者对 / 仅后者对</th><th class="num">翻转题</th>
        <th class="num">逐题判分一致</th><th class="num">输出逐字一致</th>
        <th class="num">配对 bootstrap 95%</th><th>同配置门禁判决</th></tr>{pair_rows}</table>
      <h3>翻转题分布</h3>
      <table><tr><th>难度桶</th><th class="num">题数</th><th class="num">翻转题</th>
        <th class="num">桶内翻转率</th><th class="num">占全部翻转</th></tr>{bucket_rows}</table>
      <table><tr><th class="num">三次答对次数</th><th class="num">题数</th>
        <th class="num">占翻转题</th></tr>{dist_rows}</table>
      <p class="muted">示例（前 {len(pq.get('examples', []))} 条）：</p>
      <table><tr><th>id</th><th>难度桶</th><th class="num">gold</th>
        <th>三次 pred</th><th>三次 correct</th>
        <th class="num">首个差异字符位</th><th class="num">近失</th></tr>{ex_rows}</table>
    </div>"""


def _kappa_section(kappa: dict, causes: dict | None = None) -> str:
    rule = kappa["rule"]
    judges = kappa.get("judges", [])

    def _ci_cell(entry: dict) -> str:
        ci = entry.get("ci95")
        if not ci:
            return "<td class='num'>—</td><td class='num'>—</td>"
        return (f"<td class='num mono'>[{ci[0]:.3f}, {ci[1]:.3f}]</td>"
                f"<td class='num'>{ci[1] - ci[0]:.3f}</td>")

    def _n_cell(entry: dict) -> str:
        n = entry.get("n_for_halfwidth_005")
        return "<td class='num'>—</td>" if not n else f"<td class='num'>{n:,}</td>"

    rows = [f"<tr><td>rule（数字精确匹配）</td><td>—</td><td class='num'>{rule['n']}</td>"
            f"<td class='num'>1.000</td><td class='num'>{rule['po']:.3f}</td>"
            f"<td class='num'>{rule['pe']:.3f}</td>"
            f"<td class='num'><b>{rule['kappa']:.3f}</b></td>"
            f"{_ci_cell(rule)}{_n_cell(rule)}"
            f"<td class='num'>0</td><td class='num'>0</td></tr>"]
    for j in judges:
        rows.append(f"<tr><td class='mono'>{_e(j['judge_model'])}</td>"
                    f"<td>{_e(j.get('judge_mode', 'raw'))}</td>"
                    f"<td class='num'>{j['n_judged']}</td>"
                    f"<td class='num'>{j.get('parse_rate', 1.0):.3f}</td>"
                    f"<td class='num'>{j['po']:.3f}</td><td class='num'>{j['pe']:.3f}</td>"
                    f"<td class='num'><b>{j['kappa']:.3f}</b></td>"
                    f"{_ci_cell(j)}{_n_cell(j)}"
                    f"<td class='num'>{j.get('tokens_mean', 0):,.0f}</td>"
                    f"<td class='num'>{j.get('latency_ms_mean', 0):,.0f}</td></tr>")

    cat_rows = []
    for c, v in kappa["per_category"].items():
        cells = f"<td class='num'>{v['n']}</td><td class='num'>{v['rule_kappa']:.3f}</td>"
        for j in judges:
            key = j.get("key", j["judge_model"])
            val = v.get("judges", {}).get(key)
            jn = v.get("judge_n", {}).get(key)
            cells += f"<td class='num'>{'—' if val is None else f'{val:.3f} ({jn})'}</td>"
        cat_rows.append(f"<tr><td>{_e(c)}</td>{cells}</tr>")

    weak = []
    for j in judges:
        key = j.get("key", j["judge_model"])
        for c, v in kappa["per_category"].items():
            val = v.get("judges", {}).get(key)
            if val is not None and val < 0.4:
                weak.append(f"{_e(key)} 在 {_e(c)}（κ={val:.2f}, n={v['n']}）")
    weak_html = ("<p class='warn'>κ&lt;0.4 的不可靠类别：" + "；".join(weak) +
                 "。这些类别上 judge 不能当门禁用。</p>") if weak else ""

    # 分歧错因分布（人工按分歧逐条归因，写在 human_labels/disagreement_causes.json）
    cause_rows = []
    causes = causes or {}
    for j in judges:
        key = j.get("key", j["judge_model"])
        grouped = causes.get(key) or {}
        for cause, items in grouped.items():
            cause_rows.append(
                f"<tr><td class='mono'>{_e(key)}</td><td>{_e(cause)}</td>"
                f"<td class='num'>{len(items)}</td>"
                f"<td class='mono'>{_e(', '.join(i['id'] for i in items[:6]))}"
                f"{' …' if len(items) > 6 else ''}</td>"
                f"<td>{_e(items[0].get('evidence', ''))}</td></tr>")
    cause_block = (f"""<h3>分歧错因分布（人工逐条归类）</h3>
      <table><tr><th>judge</th><th>错因</th><th class="num">条数</th>
        <th>涉及题 id</th><th>典型证据</th></tr>{''.join(cause_rows)}</table>"""
                   if cause_rows else
                   "<p class='muted'>分歧错因：<b>未归类</b>（缺 human_labels/disagreement_causes.json）。</p>")

    dis_blocks = []
    for j in judges:
        key = j.get("key", j["judge_model"])
        dis = j.get("disagreements", [])
        table = "".join(
            f"<tr><td class='mono'>{_e(x['id'])}</td><td>{_e(x['category'])}</td>"
            f"<td class='num'>{x['human']}</td><td class='num'>{x['judge']}</td>"
            f"<td class='mono'>{_e(x.get('pred'))} / {_e(x.get('gold_final'))}</td>"
            f"<td>{_e(x.get('reason') or '')}</td>"
            f"<td>{_e(x.get('human_note') or '')}</td></tr>" for x in dis)
        if dis:
            dis_blocks.append(
                f"<h3>{_e(key)} 与人工分歧（{len(dis)} 条）</h3>"
                f"<table><tr><th>id</th><th>category</th><th>human</th><th>judge</th>"
                f"<th>pred / gold</th><th>judge reason</th><th>人工 note</th></tr>"
                f"{table}</table>")

    return f"""
    <h2>Judge κ 一致性校准</h2>
    <div class="card">
      <p class="muted">人工标注 n={kappa['n']}（逐条读模型最终答案对比金标）。
         κ 参考：&lt;0.2 差，0.2-0.4 一般，0.4-0.6 中等，0.6-0.8 较好，&gt;0.8 强一致。
         κ 的 95% CI 为按标注成对 bootstrap（10,000 次）；「达 ±0.05 所需 n」按 CI 宽度 ∝ 1/√n 反解。
         tokens/条 = prompt+生成；墙钟/条为稳态耗时（首条含模型加载，已丢弃）。
         fewshot / rubric 行是<b>已知结论对照</b>，只覆盖 60 条子批次（解析率 = 60/150）；raw 行为 150 条全量重算。
         rule 的 CI 上界触及 1.0 被截断，其宽度偏小，不用于反解所需 n（表中显示 —）；
         同理 fewshot/rubric 的「所需 n」不可用——那两行是标签塌缩，不是正常抽样。</p>
      <table><tr><th>judge</th><th>配置</th><th class="num">n</th>
        <th class="num">解析率</th><th class="num">po</th><th class="num">pe</th>
        <th class="num">Cohen's κ</th><th class="num">κ 95% CI</th><th class="num">CI 宽</th>
        <th class="num">达 ±0.05 所需 n</th>
        <th class="num">tokens/条</th>
        <th class="num">墙钟/条 (ms)</th></tr>{''.join(rows)}</table>
      {weak_html}
    </div>
    <h3>分难度桶 κ</h3>
    <table><tr><th>category</th><th class="num">n</th><th class="num">rule κ</th>
      {''.join(f"<th class='num'>{_e(j.get('key', j['judge_model']))} κ</th>" for j in judges)}</tr>
      {''.join(cat_rows)}</table>
    {cause_block}
    {''.join(dis_blocks)}"""


def _headline(runs: list[dict], gate: dict | None, kappa: dict | None) -> str:
    parts = []
    for run in runs:
        recs = run["records"]
        boot = accuracy_ci([r["correct"] for r in recs])
        parts.append(f"{_e(run['label'])} {boot['point']:.3f} "
                     f"[{boot['ci95'][0]:.3f}, {boot['ci95'][1]:.3f}]"
                     f"（w={boot['ci95'][1]-boot['ci95'][0]:.3f}, n={boot['n']}）")
    lines = ["<b>真实得分：</b>" + "；".join(parts)]
    if gate:
        lines.append(f"<b>回归门禁：</b><span class='badge {gate['verdict']}'>"
                     f"{_e(gate['verdict'])}</span> delta "
                     f"{gate['delta']['delta']:+.3f} "
                     f"[{gate['delta']['ci95'][0]:+.3f}, {gate['delta']['ci95'][1]:+.3f}]，"
                     f"回退 {len(gate['flipped'])} 题")
    if kappa:
        k = f"rule κ={kappa['rule']['kappa']:.3f}"
        for j in kappa.get("judges", []):
            k += f"，{_e(j['judge_model'])} κ={j['kappa']:.3f}"
        lines.append(f"<b>Judge κ（n={kappa['n']}）：</b>{k}")
    body = "".join(f"<li>{x}</li>" for x in lines)
    return f"<div class='card'><ol style='margin:0 0 0 18px'>{body}</ol></div>"


def build(runs: list[dict], gate: dict | None, kappa: dict | None,
          *, dataset_meta: dict | None = None, out_path: str = "reports/index.html",
          notes: list[str] | None = None, causes: dict | None = None,
          cpu_cost: dict | None = None, target_width: float = 0.05,
          curve: dict | None = None, curve_png: str | None = None,
          stability: dict | None = None, variance: dict | None = None) -> str:
    dataset_meta = dataset_meta or {}
    kappa_html = _kappa_section(kappa, causes) if kappa else ""
    gate_html = _gate_section(gate) if gate else ""
    curve_html = _width_curve_section(curve, curve_png)
    stability_html = _stability_section(stability)
    variance_html = _variance_section(variance)
    notes_html = "".join(f"<li>{_e(n)}</li>" for n in (notes or []))

    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>eval-closed-loop 报告</title><style>{CSS}</style></head><body><div class="wrap">
<h1>评测闭环报告</h1>
<p class="sub">GSM8K 子集 · 两模型对比 · 置信区间 · 回归门禁 · judge κ 校准<br>
生成时间 {_e(time.strftime('%Y-%m-%d %H:%M:%S'))} ·
数据集 {_e(dataset_meta.get('path', 'data/gsm8k_n60.jsonl'))} ·
sha256 <span class="mono">{_e(str(dataset_meta.get('sha256', ''))[:16])}…</span></p>

{_headline(runs, gate, kappa)}

<h2>各模型真实得分</h2>
<div class="card">{_score_table(runs)}</div>
<h3>分类别准确率</h3>
<div class="card">{_category_table(runs)}</div>
{_power_section(runs, target_width)}
{curve_html}
{gate_html}
{kappa_html}
{stability_html}
{variance_html}
{_cost_section(runs, cpu_cost)}

<h2>局限与未验证项</h2>
<div class="warn"><ul>{notes_html}</ul></div>
</div></body></html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    return out_path


def summary_json(runs: list[dict], gate: dict | None, kappa: dict | None,
                 path: str, *, cpu_cost: dict | None = None,
                 curve: dict | None = None, stability: dict | None = None,
                 variance: dict | None = None) -> str:
    def _run(run: dict) -> dict:
        recs = run["records"]
        n = len(recs)
        boot = accuracy_ci([x["correct"] for x in recs])
        wil = wilson_interval(sum(x["correct"] for x in recs), n)
        pt = [r.get("prompt_eval_count") or 0 for r in recs]
        ot = [r.get("eval_count") or 0 for r in recs]
        lat = [r["latency_ms"] for r in recs]
        return {
            "label": run["label"], "model": run["model"], "prompt": run["prompt"],
            "n": n, "accuracy": boot, "wilson_ci95": list(wil),
            "wilson_width": wil[1] - wil[0],
            "bootstrap_width": boot["ci95"][1] - boot["ci95"][0],
            "cost": {
                "prompt_tokens_mean": round(sum(pt) / n, 1) if n else 0,
                "out_tokens_mean": round(sum(ot) / n, 1) if n else 0,
                "latency_ms_mean": round(sum(lat) / n, 1) if n else 0,
                "total_latency_minutes": round(sum(lat) / 1000 / 60, 1) if n else 0,
                "wall_clock_minutes": (None if _wall_clock_minutes(run) is None
                                       else round(_wall_clock_minutes(run), 1)),
                "total_tokens": sum(pt) + sum(ot),
            },
        }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [_run(r) for r in runs],
        "gate": gate,
        "kappa": kappa,
        "cpu_cost": cpu_cost,
        "width_curve": curve,
        "stability": stability,
        "variance": variance,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
