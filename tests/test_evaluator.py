from __future__ import annotations

import time

from redline.swarm.dataset_loader import load_humaneval_plus, load_mbpp_plus
from redline.swarm.evaluator import EvalResult, EVAL_TIMEOUT, evaluate


class TestHumanEvalGoodSolution:
    def test_known_good_passes(self):
        tasks = load_humaneval_plus(validate=False)
        task = tasks["HumanEval/0"]

        good_code = (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    sorted_numbers = sorted(numbers)\n"
            "    for i in range(len(sorted_numbers) - 1):\n"
            "        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n"
            "            return True\n"
            "    return False\n"
        )

        result = evaluate(good_code, task.test, task.entry_point)
        assert isinstance(result, EvalResult)
        assert result.passed is True


class TestHumanEvalBadSolution:
    def test_known_bad_fails(self):
        tasks = load_humaneval_plus(validate=False)
        task = tasks["HumanEval/0"]

        bad_code = (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    return True\n"
        )

        result = evaluate(bad_code, task.test, task.entry_point)
        assert isinstance(result, EvalResult)
        assert result.passed is False
        assert len(result.stderr) > 0


class TestMBPPGoodSolution:
    def test_known_good_passes(self):
        tasks = load_mbpp_plus(validate=False)
        task = tasks["Mbpp/2"]

        good_code = (
            "def similar_elements(t1, t2):\n"
            "    return tuple(sorted(set(t1) & set(t2)))\n"
        )

        result = evaluate(good_code, task.test, task.entry_point)
        assert isinstance(result, EvalResult)
        assert result.passed is True


class TestMBPPBadSolution:
    def test_known_bad_fails(self):
        tasks = load_mbpp_plus(validate=False)
        task = tasks["Mbpp/2"]

        bad_code = "def similar_elements(t1, t2):\n    return ()\n"

        result = evaluate(bad_code, task.test, task.entry_point)
        assert isinstance(result, EvalResult)
        assert result.passed is False


class TestTimeout:
    def test_infinite_loop_times_out(self):
        code = "def f():\n    while True:\n        pass\n"
        test_code = "assert f() is not None"

        start = time.monotonic()
        result = evaluate(code, test_code)
        elapsed = time.monotonic() - start

        assert isinstance(result, EvalResult)
        assert result.passed is False
        assert "TIMEOUT" in result.stderr
        assert elapsed < EVAL_TIMEOUT + 3


class TestEvalResult:
    def test_dataclass_fields(self):
        r = EvalResult(passed=True, stderr="")
        assert r.passed is True
        assert r.stderr == ""

    def test_frozen(self):
        import pytest as pt

        r = EvalResult(passed=True, stderr="")
        with pt.raises(Exception):
            r.passed = False  # type: ignore


class TestStderrTruncation:
    def test_stderr_not_empty_on_failure(self):
        code = "def f(): raise ValueError('boom')"
        test_code = "assert f() is not None"

        result = evaluate(code, test_code)
        assert result.passed is False
        assert len(result.stderr) > 0
