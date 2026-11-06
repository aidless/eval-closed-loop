PY ?= python3
export PYTHONPATH := src
export EC_EVAL_GATE_PATH ?= $(HOME)/eval-gate

RUNS := results/runs
N ?= 60
MODEL_A ?= qwen2.5:1.5b
MODEL_B ?= qwen2.5:0.5b
JUDGE_BIG ?= qwen2.5:3b
# FORCE=1 清掉已有生成结果重跑；默认复用 jsonl（断点续跑）
FORCE ?= 0

# N=60 沿用原有路径（最小闭环）；其他 N 用带后缀的独立文件，互不覆盖
ifeq ($(N),60)
DATA := data/gsm8k_n60.jsonl
FREEZE_FLAGS :=
A_COT := $(RUNS)/a_cot.jsonl
A_SHORT := $(RUNS)/a_short.jsonl
B_COT := $(RUNS)/b_cot.jsonl
GATE := reports/gate.json
else
DATA := data/gsm8k_n$(N).jsonl
FREEZE_FLAGS := --stratify
A_COT := $(RUNS)/a_cot_n$(N).jsonl
A_SHORT := $(RUNS)/a_short_n$(N).jsonl
B_COT := $(RUNS)/b_cot_n$(N).jsonl
GATE := reports/gate_n$(N).json
endif

# κ 校准固定在同一批人工标注（n=150）；fewshot/rubric 对照沿用 n=60 批次，不必重跑
JUDGE_RECORDS := $(RUNS)/a_cot_kappa150.jsonl
VAR_RECORDS := $(RUNS)/a_cot.jsonl
HUMAN := human_labels/labels.jsonl
CPU_LIMIT ?= 3
A_COT_REP2 := $(RUNS)/a_cot_n$(N)_rep2.jsonl
JUDGE_DIR := results/judge
J_05_RAW := $(JUDGE_DIR)/qwen0p5b_raw.jsonl
J_15_RAW := $(JUDGE_DIR)/qwen1p5b_raw.jsonl
J_15_FEW := $(JUDGE_DIR)/qwen1p5b_fewshot.jsonl
J_15_RUB := $(JUDGE_DIR)/qwen1p5b_rubric.jsonl
J_3B_RAW := $(JUDGE_DIR)/qwen3b_raw.jsonl
J_3B_FEW := $(JUDGE_DIR)/qwen3b_fewshot.jsonl
J_3B_RUB := $(JUDGE_DIR)/qwen3b_rubric.jsonl
JUDGES := $(J_05_RAW) $(J_15_RAW) $(J_15_FEW) $(J_15_RUB) $(J_3B_RAW) $(J_3B_FEW) $(J_3B_RUB)

KAPPA := reports/kappa.json
REPORT := reports/index.html
# 分数可复现性探针：同一 n=300 子集完整重跑 3 次（独立进程、间隔开、不并发）
VAR_RUNS := $(RUNS)/a_cot_n300_var1.jsonl $(RUNS)/a_cot_n300_var2.jsonl $(RUNS)/a_cot_n300_var3.jsonl
VARIANCE := reports/variance.json

.PHONY: all repro smoke gate test verify clean help force-clean cost-cpu judges kappa stability curve variance $(REPORT)

help:
	@echo "make repro          # 一键复现（默认 N=60 最小闭环）：跑分 -> 区间 -> 门禁 -> judge κ -> HTML"
	@echo "make repro N=300    # 扩样本中间档（分层 300 题）"
	@echo "make repro N=1319   # GSM8K test 全量（首次约 3-5 小时，断点续跑）"
	@echo "make stability N=1319  # A-cot 全量跑第二遍并逐题比对输出"
	@echo "make variance       # 同一 n=300 子集跑 3 次，量化分数可复现性（reports/variance.json）"
	@echo "make curve          # 生成 n=60/300/全量 的区间宽度收缩曲线（JSON+PNG）"
	@echo "make gate           # 只看门禁，非 PASS 退出码为 1"
	@echo "make verify         # 从落盘输出重算分，验证判分确定性"
	@echo "make judges         # 重跑 7 个 judge 配置（raw/fewshot/rubric × 0.5b/1.5b/3b）"
	@echo "make kappa          # 只重算 judge κ 校准（n=150 人工标注）"
	@echo "make cost-cpu CPU_LIMIT=100  # CPU 抽样实测（num_gpu=0），不做外推"
	@echo "make test           # 单元测试"

all: repro

.PHONY: force-clean
force-clean:
ifeq ($(FORCE),1)
	rm -rf $(RUNS) results/judge $(GATE) $(KAPPA) $(REPORT) reports/summary.json
endif

$(DATA):
	$(PY) scripts/freeze_data.py --n $(N) $(FREEZE_FLAGS) --out $(DATA)

# 三个 run 各自独立生成，已完成的 id 会自动跳过
$(A_COT): $(DATA)
	@mkdir -p $(RUNS)
	$(PY) -m eval_loop.cli run --dataset $(DATA) --model $(MODEL_A) --prompt cot --limit $(N) --out $@

$(A_SHORT): $(DATA)
	@mkdir -p $(RUNS)
	$(PY) -m eval_loop.cli run --dataset $(DATA) --model $(MODEL_A) --prompt short --limit $(N) --num-predict 24 --out $@

$(B_COT): $(DATA)
	@mkdir -p $(RUNS)
	$(PY) -m eval_loop.cli run --dataset $(DATA) --model $(MODEL_B) --prompt cot --limit $(N) --out $@

# 门禁：同模型、换 prompt 的退化配置当候选，非 PASS 时命令返回 1（这里用 - 容忍，结果写进报告）
$(GATE): $(A_COT) $(A_SHORT)
	@mkdir -p reports
	-$(PY) -m eval_loop.cli gate --baseline $(A_COT) --candidate $(A_SHORT) --out $(GATE)

$(J_05_RAW): $(JUDGE_RECORDS)
	@mkdir -p $(JUDGE_DIR)
	$(PY) -m eval_loop.cli judge --records $< --model $(MODEL_B) --mode raw --out $@

$(J_15_RAW): $(JUDGE_RECORDS)
	@mkdir -p $(JUDGE_DIR)
	$(PY) -m eval_loop.cli judge --records $< --model $(MODEL_A) --mode raw --out $@

$(J_15_FEW): $(VAR_RECORDS)
	@mkdir -p $(JUDGE_DIR)
	$(PY) -m eval_loop.cli judge --records $< --model $(MODEL_A) --mode fewshot --out $@

$(J_15_RUB): $(VAR_RECORDS)
	@mkdir -p $(JUDGE_DIR)
	$(PY) -m eval_loop.cli judge --records $< --model $(MODEL_A) --mode rubric --out $@

$(J_3B_RAW): $(JUDGE_RECORDS)
	@mkdir -p $(JUDGE_DIR)
	$(PY) -m eval_loop.cli judge --records $< --model $(JUDGE_BIG) --mode raw --out $@

$(J_3B_FEW): $(VAR_RECORDS)
	@mkdir -p $(JUDGE_DIR)
	$(PY) -m eval_loop.cli judge --records $< --model $(JUDGE_BIG) --mode fewshot --out $@

$(J_3B_RUB): $(VAR_RECORDS)
	@mkdir -p $(JUDGE_DIR)
	$(PY) -m eval_loop.cli judge --records $< --model $(JUDGE_BIG) --mode rubric --out $@

judges: $(JUDGES)

kappa: $(KAPPA)

$(KAPPA): $(JUDGE_RECORDS) $(JUDGES) $(HUMAN)
	$(PY) -m eval_loop.cli calibrate --records $(JUDGE_RECORDS) --human $(HUMAN) \
		--judge-labels $(J_05_RAW) --judge-model "$(MODEL_B)" --judge-mode raw \
		--judge-labels $(J_15_RAW) --judge-model "$(MODEL_A)" --judge-mode raw \
		--judge-labels $(J_15_FEW) --judge-model "$(MODEL_A)" --judge-mode fewshot \
		--judge-labels $(J_15_RUB) --judge-model "$(MODEL_A)" --judge-mode rubric \
		--judge-labels $(J_3B_RAW) --judge-model "$(JUDGE_BIG)" --judge-mode raw \
		--judge-labels $(J_3B_FEW) --judge-model "$(JUDGE_BIG)" --judge-mode fewshot \
		--judge-labels $(J_3B_RUB) --judge-model "$(JUDGE_BIG)" --judge-mode rubric \
		--out $@

$(REPORT): $(A_COT) $(A_SHORT) $(B_COT) $(GATE) $(KAPPA)
	$(PY) -m eval_loop.cli report --dataset $(DATA) \
		--run "A-cot=$(A_COT)" --run "A-short=$(A_SHORT)" --run "B-cot=$(B_COT)" \
		--gate $(GATE) --kappa $(KAPPA) --out $(REPORT) \
		--summary-json reports/summary.json --target-width 0.05 \
		--note "本报告为 N=$(N) 档：n=60 / n=300 / 全量 1,319 是三份独立抽样，不能混算；区间与 n 必须一起读。" \
		--note "judge κ 来自 n=150 人工标注（steps 桶 37/76/37）；fewshot/rubric 变体沿用 60 条对照，仅作已知结论。" \
		--note "类别是按金标解题行数派生的难度桶（1-2 / 3-4 / 5+ 行），不是 GSM8K 官方分类。" \
		--note "样本量推算与 CPU 成本外推均标未验证；实际跑到的 n 才是结论依据。" \
		--note "两个主模型均 ≤1.5B，没有 API 强模型对比；judge 只看最终答案，不评价推理过程。" \
		--note "成本为同机实测（GPU 另有训练进程并发），不作跨机器基准；CPU 只做 n=100 抽样，不外推。" \
		--note "三个 run 均串行独立执行：并发会让 0.5b 被挤到 CPU（单题 ~20s），B-cot 早期 6 条已作废重跑。" \
		--note "分数可复现性探针是同一 n=300 子集额外完整跑 3 次（独立进程、间隔 >=3 min），不与主表 n=300 混算；主表数字未改写。"

repro: force-clean $(REPORT)
	@$(PY) -m eval_loop.cli summary --run "A-cot=$(A_COT)" --run "A-short=$(A_SHORT)" --run "B-cot=$(B_COT)"
	@echo "report: $(REPORT)"

smoke: $(DATA)
	@mkdir -p $(RUNS)
	$(PY) -m eval_loop.cli run --dataset $(DATA) --model $(MODEL_A) --prompt cot --limit 8 --out $(RUNS)/smoke_a.jsonl
	$(PY) -m eval_loop.cli run --dataset $(DATA) --model $(JUDGE_BIG) --prompt cot --limit 8 --out $(RUNS)/smoke_b.jsonl

gate: $(GATE)
	@$(PY) -m eval_loop.cli gate --baseline $(A_COT) --candidate $(A_SHORT) --out $(GATE)

verify: $(A_COT) $(A_SHORT) $(B_COT)
	$(PY) -m eval_loop.cli verify --runs $(A_COT) $(A_SHORT) $(B_COT)

cost-cpu:
	@mkdir -p reports
	$(PY) scripts/cost_cpu.py --dataset $(DATA) --limit $(CPU_LIMIT) --out reports/cost_cpu.json

# A-cot 同一配置再跑一遍，用于判断 greedy 是否逐字节确定
$(A_COT_REP2): $(DATA)
	@mkdir -p $(RUNS)
	$(PY) -m eval_loop.cli run --dataset $(DATA) --model $(MODEL_A) --prompt cot --limit $(N) --out $@

stability: $(A_COT) $(A_COT_REP2)
	@mkdir -p reports
	$(PY) scripts/compare_runs.py --a $(A_COT) --b $(A_COT_REP2) --out reports/stability.json

# 同一 n=300 子集完整重跑 3 次：文件名固定，与 N=60/N=1319 档无关
$(RUNS)/a_cot_n300_var%.jsonl: data/gsm8k_n300.jsonl
	@mkdir -p $(RUNS)
	$(PY) -m eval_loop.cli run --dataset data/gsm8k_n300.jsonl --model $(MODEL_A) \
		--prompt cot --limit 300 --out $@

variance: $(VAR_RUNS)
	@mkdir -p reports
	$(PY) scripts/variance.py \
		--run "var1=$(word 1,$(VAR_RUNS))" \
		--run "var2=$(word 2,$(VAR_RUNS))" \
		--run "var3=$(word 3,$(VAR_RUNS))" \
		--env-note "var1/var2/var3 各自独立进程，三次之间留 >=3 min 空档；三次均跑在 21%/79% CPU/GPU（训练进程并发占约 3.7GB 显存，1.5b 无法全量上卡），逐 token 数值差异比全量两次跑（100% GPU、19/1,319 输出不同）更大，故这里的 3σ 视为保守上界。" \
		--out $(VARIANCE)

# 三档实测宽度 + √n 理论线；缺哪档就少画哪档
curve:
	@mkdir -p reports
	$(PY) scripts/width_curve.py \
		--point "A-cot:60:$(RUNS)/a_cot.jsonl" \
		--point "A-cot:300:$(RUNS)/a_cot_n300.jsonl" \
		--point "A-cot:1319:$(RUNS)/a_cot_n1319.jsonl" \
		--point "A-short:60:$(RUNS)/a_short.jsonl" \
		--point "A-short:300:$(RUNS)/a_short_n300.jsonl" \
		--point "A-short:1319:$(RUNS)/a_short_n1319.jsonl" \
		--point "B-cot:60:$(RUNS)/b_cot.jsonl" \
		--point "B-cot:300:$(RUNS)/b_cot_n300.jsonl" \
		--point "B-cot:1319:$(RUNS)/b_cot_n1319.jsonl" \
		--out reports/width_curve.json --png reports/width_curve.png

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf $(RUNS) results/judge $(GATE) $(KAPPA) $(REPORT) reports/summary.json
