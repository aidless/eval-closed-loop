import pytest

from eval_loop.bench import extract_pred, is_correct
from eval_loop.stats import accuracy_ci, cohen_kappa, wilson_interval


def test_extract_marked_answer():
    assert extract_pred("... so #### 18") == 18.0
    assert extract_pred("#### $1,596.00") == 1596.0
    assert extract_pred("#### 18.") == 18.0


def test_extract_falls_back_to_last_number():
    assert extract_pred("The answer is 42") == 42.0
    assert extract_pred("no number here") is None


def test_is_correct_tolerance():
    assert is_correct(18.0, "18")
    assert is_correct(2.0, "2.00")
    assert not is_correct(18.0, "19")
    assert not is_correct(None, "18")


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 10)
    assert 0.0 <= lo < hi <= 1.0
    assert wilson_interval(3, 0) == (0.0, 1.0)


def test_bootstrap_is_deterministic():
    corrects = [True] * 5 + [False] * 5
    a = accuracy_ci(corrects, iterations=2000, seed=7)
    b = accuracy_ci(corrects, iterations=2000, seed=7)
    assert a == b
    assert a["point"] == pytest.approx(0.5)
    assert a["ci95"][0] < 0.5 < a["ci95"][1]


def test_kappa_perfect_and_chance():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0])["kappa"] == pytest.approx(1.0)
    # 完全随机独立：两列都 50/50，且交叉分布符合独立
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    assert cohen_kappa(a, b)["kappa"] == pytest.approx(0.0)


def test_kappa_confusion_counts():
    r = cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0])
    assert r["confusion"] == {"both_1": 1, "a1_b0": 1, "a0_b1": 0, "both_0": 2}
