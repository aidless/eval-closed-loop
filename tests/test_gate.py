from eval_loop.gate import evaluate, exit_code

import pytest


def _rec(i, correct, cat="steps_3_4"):
    return {"id": f"q{i}", "category": cat, "correct": correct,
            "pred": 1.0 if correct else 2.0, "gold_final": "1",
            "question": "q", "output": "o"}


def test_identical_runs_pass():
    recs = [_rec(i, i % 2 == 0) for i in range(60)]
    result = evaluate(recs, recs)
    assert result["verdict"] == "PASS"
    assert result["delta"]["delta"] == 0
    assert exit_code(result) == 0


def test_full_regression_blocks():
    base = [_rec(i, True) for i in range(60)]
    cand = [_rec(i, False) for i in range(60)]
    result = evaluate(cand, base)
    assert result["verdict"] == "BLOCK"
    assert len(result["flipped"]) == 60
    assert exit_code(result) == 1


def test_small_drop_within_tolerance_passes():
    base = [_rec(i, True) for i in range(60)]
    cand = [_rec(i, i >= 2) for i in range(60)]  # 只掉 2 题
    result = evaluate(cand, base)
    assert result["delta"]["delta"] == pytest.approx(-2 / 60, abs=1e-4)
    assert result["verdict"] != "BLOCK"


def test_missing_items_is_incomplete():
    base = [_rec(i, True) for i in range(60)]
    cand = [_rec(i, True) for i in range(30)]
    result = evaluate(cand, base)
    assert result["verdict"] == "INCOMPLETE"


def test_small_sample_review():
    base = [_rec(i, False) for i in range(5)]
    cand = [_rec(i, True) for i in range(5)]
    result = evaluate(cand, base)
    assert result["verdict"] == "REVIEW"


def test_category_attribution():
    base = [_rec(i, True, "steps_1_2") for i in range(10)] + \
           [_rec(100 + i, True, "steps_5_plus") for i in range(10)]
    cand = [_rec(i, True, "steps_1_2") for i in range(10)] + \
           [_rec(100 + i, False, "steps_5_plus") for i in range(10)]
    result = evaluate(cand, base)
    assert result["per_category"]["steps_1_2"]["delta"] == 0
    assert result["per_category"]["steps_5_plus"]["delta"] == -1.0
    assert result["per_category"]["steps_5_plus"]["flipped"] == 10


def test_report_only_exit_code_is_zero():
    from eval_loop.gate import apply_report_only
    result = apply_report_only({"verdict": "BLOCK"}, True)
    assert exit_code(result) == 0
