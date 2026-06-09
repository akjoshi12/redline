from __future__ import annotations

import pytest

from redline.baseline.task_suite import Task, load_tasks, suite_hash


# ── Loading ────────────────────────────────────────────────────────────


def test_load_returns_list():
    tasks = load_tasks()
    assert isinstance(tasks, list)


def test_count_is_expected():
    tasks = load_tasks()
    assert len(tasks) == 15


def test_each_task_has_required_fields():
    for t in load_tasks():
        assert isinstance(t.id, str) and t.id
        assert t.category in ("short", "medium", "long")
        assert t.kind in ("codegen", "reasoning")
        assert isinstance(t.prompt, str) and len(t.prompt) > 10


# ── Distribution ───────────────────────────────────────────────────────


def test_category_distribution():
    tasks = load_tasks()
    cats = {t.category for t in tasks}
    assert "short" in cats
    assert "medium" in cats
    assert "long" in cats


def test_kind_distribution():
    tasks = load_tasks()
    kinds = {t.kind for t in tasks}
    assert "codegen" in kinds
    assert "reasoning" in kinds


# ── IDs unique ────────────────────────────────────────────────────────


def test_ids_unique():
    ids = [t.id for t in load_tasks()]
    assert len(ids) == len(set(ids))


# ── Frozen dataclass ──────────────────────────────────────────────────


def test_task_is_frozen():
    t = Task(id="x", category="short", kind="codegen", prompt="hello")
    with pytest.raises(Exception):  # FrozenInstanceError
        t.id = "y"


# ── Suite hash determinism ───────────────────────────────────────────


def test_hash_is_deterministic():
    h1 = suite_hash()
    h2 = suite_hash()
    assert h1 == h2


def test_hash_is_sha256_hex():
    h = suite_hash()
    assert len(h) == 64  # SHA-256 hex digest length
    int(h, 16)  # must be valid hex
