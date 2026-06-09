from __future__ import annotations

import pytest

from redline.swarm.dataset_loader import (
    HUMANEVAL_PLUS_COUNT,
    MBPP_PLUS_COUNT,
    Task,
    load_humaneval_plus,
    load_mbpp_plus,
)


class TestLoadHumanEvalPlus:
    def test_returns_correct_count(self):
        tasks = load_humaneval_plus()
        assert len(tasks) == HUMANEVAL_PLUS_COUNT

    def test_task_ids_present(self):
        tasks = load_humaneval_plus()
        for task_id in tasks:
            assert isinstance(task_id, str)
            assert len(task_id) > 0

    def test_required_fields_non_empty(self):
        tasks = load_humaneval_plus()
        for tid, task in tasks.items():
            assert task.task_id == tid
            assert isinstance(task.prompt, str) and len(task.prompt.strip()) > 0
            assert isinstance(task.test, str) and len(task.test.strip()) > 0

    def test_task_is_dataclass(self):
        tasks = load_humaneval_plus()
        for task in tasks.values():
            assert isinstance(task, Task)

    def test_base_and_plus_input_present(self):
        tasks = load_humaneval_plus()
        for task in tasks.values():
            assert isinstance(task.base_input, list)
            assert isinstance(task.plus_input, list)


class TestLoadMBPPPlus:
    def test_returns_correct_count(self):
        tasks = load_mbpp_plus()
        assert len(tasks) == MBPP_PLUS_COUNT

    def test_task_ids_present(self):
        tasks = load_mbpp_plus()
        for task_id in tasks:
            assert isinstance(task_id, str)
            assert len(task_id) > 0

    def test_required_fields_non_empty(self):
        tasks = load_mbpp_plus()
        for tid, task in tasks.items():
            assert task.task_id == tid
            assert isinstance(task.prompt, str) and len(task.prompt.strip()) > 0
            assert isinstance(task.test, str) and len(task.test.strip()) > 0

    def test_task_is_dataclass(self):
        tasks = load_mbpp_plus()
        for task in tasks.values():
            assert isinstance(task, Task)

    def test_test_contains_assertions(self):
        tasks = load_mbpp_plus()
        for task in tasks.values():
            assert "assert" in task.test.lower()

    def test_base_and_plus_input_present(self):
        tasks = load_mbpp_plus()
        for task in tasks.values():
            assert isinstance(task.base_input, list)
            assert isinstance(task.plus_input, list)


class TestValidation:
    def test_wrong_count_raises(self):
        from redline.swarm.dataset_loader import _validate

        with pytest.raises(ValueError, match="expected 10"):
            _validate({}, "Fake", expected_count=10)

    def test_empty_prompt_raises(self):
        from redline.swarm.dataset_loader import _validate

        tasks = {"t0": Task(task_id="t0", prompt="", test="assert True")}
        with pytest.raises(ValueError, match="empty prompt"):
            _validate(tasks, "Fake", expected_count=1)

    def test_empty_test_raises(self):
        from redline.swarm.dataset_loader import _validate

        tasks = {"t0": Task(task_id="t0", prompt="def f(): pass", test="")}
        with pytest.raises(ValueError, match="empty test"):
            _validate(tasks, "Fake", expected_count=1)

    def test_empty_task_id_raises(self):
        from redline.swarm.dataset_loader import _validate

        tasks = {"": Task(task_id="", prompt="def f(): pass", test="assert True")}
        with pytest.raises(ValueError, match="empty task_id"):
            _validate(tasks, "Fake", expected_count=1)


class TestCounts:
    def test_humaneval_constant(self):
        assert HUMANEVAL_PLUS_COUNT == 164

    def test_mbpp_constant(self):
        assert MBPP_PLUS_COUNT == 378
