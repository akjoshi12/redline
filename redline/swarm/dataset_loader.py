from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from evalplus.data import get_human_eval_plus, get_mbpp_plus

logger = logging.getLogger(__name__)

HUMANEVAL_PLUS_COUNT = 164
MBPP_PLUS_COUNT = 378


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    test: str
    entry_point: str = ""
    base_input: List = field(default_factory=list)
    plus_input: List = field(default_factory=list)


def _build_mbpp_test(assertion: str, entry_point: str) -> str:
    lines = [f"from solution import {entry_point}"]
    for line in assertion.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            lines.append(stripped)
    return "\n".join(lines)


def load_humaneval_plus(validate: bool = True) -> Dict[str, Task]:
    raw = get_human_eval_plus(err_incomplete=validate)

    tasks: Dict[str, Task] = {}
    for item in raw.values():
        task_id = item["task_id"]
        prompt = item["prompt"]
        test_code = item.get("test", "")
        entry_point = item.get("entry_point", "")
        base_input = item.get("base_input") or []
        plus_input = item.get("plus_input") or []
        if not isinstance(base_input, list):
            base_input = []
        if not isinstance(plus_input, list):
            plus_input = []

        tasks[task_id] = Task(
            task_id=task_id,
            prompt=prompt,
            test=test_code,
            entry_point=entry_point,
            base_input=base_input,
            plus_input=plus_input,
        )

    if validate:
        _validate(tasks, "HumanEval+", HUMANEVAL_PLUS_COUNT)

    return tasks


def load_mbpp_plus(validate: bool = True) -> Dict[str, Task]:
    raw = get_mbpp_plus(err_incomplete=validate)

    tasks: Dict[str, Task] = {}
    for item in raw.values():
        task_id = item["task_id"]
        prompt = item["prompt"]
        entry_point = item.get("entry_point", "")
        assertion = item.get("assertion", "")
        base_input = item.get("base_input") or []
        plus_input = item.get("plus_input") or []
        if not isinstance(base_input, list):
            base_input = []
        if not isinstance(plus_input, list):
            plus_input = []

        test_code = _build_mbpp_test(assertion, entry_point) if assertion else ""

        tasks[task_id] = Task(
            task_id=task_id,
            prompt=prompt,
            test=test_code,
            entry_point=entry_point,
            base_input=base_input,
            plus_input=plus_input,
        )

    if validate:
        _validate(tasks, "MBPP+", MBPP_PLUS_COUNT)

    return tasks


def _validate(tasks: Dict[str, Task], name: str, expected_count: int) -> None:
    actual = len(tasks)
    if actual != expected_count:
        raise ValueError(f"{name}: expected {expected_count} tasks, got {actual}")

    for task_id, task in tasks.items():
        if not task.task_id:
            raise ValueError(f"{name}: empty task_id at index")
        if not task.prompt.strip():
            raise ValueError(f"{name}: empty prompt for task {task_id}")
        if not task.test.strip():
            raise ValueError(f"{name}: empty test for task {task_id}")

    logger.info("%s: %d tasks loaded, integrity OK", name, actual)
