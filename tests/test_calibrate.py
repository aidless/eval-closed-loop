import pytest

from eval_loop.judge import calibrate
from eval_loop.stats import cohen_kappa


def _rec(i, correct, cat="steps_1_2"):
    return {"id": f"q{i}", "category": cat, "correct": correct,
            "pred": 1.0, "gold_final": "1", "question": "q", "output": "o"}


def test_calibrate_reports_rule_and_judges():
    recs = [_rec(i, i < 6) for i in range(10)]
    human = {f"q{i}": {"id": f"q{i}", "human": 1 if i < 6 else 0} for i in range(10)}
    # 一个只对了一半的“弱 judge”
    rows = [{"id": f"q{i}", "category": "steps_1_2",
             "judge_label": 1 if i < 3 else 0, "judge_reason": "x"} for i in range(10)]
    result = calibrate(recs, human, [("weak", rows)])
    assert result["n"] == 10
    assert result["rule"]["kappa"] == pytest.approx(1.0)
    assert len(result["judges"]) == 1
    weak = result["judges"][0]
    assert weak["kappa"] < result["rule"]["kappa"]
    assert len(weak["disagreements"]) == 3
    assert "steps_1_2" in result["per_category"]
    assert result["per_category"]["steps_1_2"]["judges"]["weak"] == pytest.approx(weak["kappa"])


def test_calibrate_requires_overlap():
    with pytest.raises(SystemExit):
        calibrate([_rec(0, True)], {})
