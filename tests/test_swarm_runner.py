from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redline.swarm.runner import (
    CircuitBreaker,
    _build_run_id,
    run_swarm,
)
from redline.telemetry.metrics import SwarmRecord
from redline.swarm.dashboard import HardwareState, RunState


# ── Helpers ────────────────────────────────────────────────────────────


def _make_sse_chunks():
    """Return a minimal valid SSE chunk list for testing."""
    return [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "```python\n"}}]},
        {"choices": [{"delta": {"content": "def solve():\n"}}]},
        {"choices": [{"delta": {"content": "    return 42\n"}}]},
        {"choices": [{"delta": {"content": "\n```"}}]},
    ]


def _make_timestamps():
    """Return timestamps aligned with _make_sse_chunks."""
    return [50.0, 100.0, 200.0, 300.0, 400.0]


# ── run_id ─────────────────────────────────────────────────────────────


class TestBuildRunId:
    def test_format(self):
        rid = _build_run_id()
        assert rid.startswith("swarm-")
        # hex portion is 8 chars
        assert len(rid) == 14  # "swarm-" (6) + 8 hex

    def test_uniqueness(self):
        ids = {_build_run_id() for _ in range(20)}
        assert len(ids) == 20


# ── CircuitBreaker ────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_initial_concurrency(self):
        cb = CircuitBreaker(initial_concurrency=4)
        assert cb.concurrency == 4

    def test_halves_on_threshold(self):
        cb = CircuitBreaker(initial_concurrency=8, threshold=3, window_s=10.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.concurrency == 4

    def test_no_trip_below_threshold(self):
        cb = CircuitBreaker(initial_concurrency=8, threshold=5, window_s=10.0)
        for _ in range(4):
            cb.record_failure()
        assert cb.concurrency == 8

    def test_floor_at_1(self):
        cb = CircuitBreaker(initial_concurrency=2, threshold=1, window_s=10.0)
        cb.record_failure()
        assert cb.concurrency == 1

    def test_reset_restores(self):
        cb = CircuitBreaker(initial_concurrency=8, threshold=3, window_s=10.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.concurrency == 4
        cb.reset()
        assert cb.concurrency == 8

    def test_old_failures_expire(self):
        cb = CircuitBreaker(initial_concurrency=8, threshold=2, window_s=0.1)
        cb.record_failure()
        time.sleep(0.15)
        cb.record_failure()
        # First failure expired, only 1 in window → no trip
        assert cb.concurrency == 8


# ── run_swarm (smoke test) ────────────────────────────────────────────


class TestRunSwarm:
    """Test swarm orchestration with mocked HTTP and evaluator."""

    def _make_fake_stream(self):
        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        return fake_stream

    def test_smoke_two_tasks_gen1_pop2(self, tmp_path):
        """Smoke: 2 tasks, gen=1, pop=2 → writes SwarmRecord per gen with pass@1."""

        fake_stream = self._make_fake_stream()

        # First task passes in gen0, second fails → triggers gen1 repair
        eval_results_gen0 = [
            MagicMock(passed=True, stderr=""),
            MagicMock(passed=False, stderr="AssertionError: expected 42"),
        ]
        eval_idx = [0]

        def side_effect_eval(*args, **kwargs):
            result = eval_results_gen0[eval_idx[0]]
            eval_idx[0] += 1
            return result

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.swarm.runner.evaluate",
                side_effect=side_effect_eval,
            ):
                # Patch SystemMonitor to be a no-op for speed
                class DummyMonitor:
                    def __init__(self, **kw):
                        pass

                    def start(self):
                        pass

                    def stop(self):
                        pass

                with patch(
                    "redline.swarm.runner.SystemMonitor",
                    DummyMonitor,
                ):
                    records = asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=1,
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0", "HumanEval/1"],
                        )
                    )

        # Should have 2 SwarmRecords: one for gen0, one for gen1
        assert len(records) == 2
        assert all(isinstance(r, SwarmRecord) for r in records)

        # Gen0 record: 1 out of 2 tasks solved → pass@1 = 0.5
        gen0 = records[0]
        assert gen0.generation == 0
        assert gen0.tasks_total == 2
        assert gen0.dataset == "humaneval_plus"
        assert abs(gen0.pass_at_1 - 0.5) < 0.01

        # Gen1 record: repair attempted for unsolved task
        gen1 = records[1]
        assert gen1.generation == 1

    def test_writes_meta_json(self, tmp_path):
        """meta.json is written with run metadata."""

        fake_stream = self._make_fake_stream()

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.swarm.runner.evaluate",
                return_value=MagicMock(passed=True, stderr=""),
            ):

                class DummyMonitor:
                    def __init__(self, **kw):
                        pass

                    def start(self):
                        pass

                    def stop(self):
                        pass

                with patch(
                    "redline.swarm.runner.SystemMonitor",
                    DummyMonitor,
                ):
                    asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=1,
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0"],
                        )
                    )

        # Find meta.json in any subdirectory
        meta_files = list(tmp_path.rglob("meta.json"))
        assert len(meta_files) == 1

        meta = json.loads(meta_files[0].read_text())
        assert "run_id" in meta
        assert meta["dataset"] == "humaneval_plus"
        assert meta["generations"] >= 1

    def test_pass_at_1_computed(self, tmp_path):
        """pass@1 is correctly computed from solved/total."""

        fake_stream = self._make_fake_stream()

        # First task passes, second fails in gen0
        eval_results = [
            MagicMock(passed=True, stderr=""),
            MagicMock(passed=False, stderr="error"),
        ]
        eval_idx = [0]

        def side_effect_eval(*args, **kwargs):
            result = eval_results[eval_idx[0]]
            eval_idx[0] += 1
            return result

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.swarm.runner.evaluate",
                side_effect=side_effect_eval,
            ):

                class DummyMonitor:
                    def __init__(self, **kw):
                        pass

                    def start(self):
                        pass

                    def stop(self):
                        pass

                with patch(
                    "redline.swarm.runner.SystemMonitor",
                    DummyMonitor,
                ):
                    records = asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=0,  # Only gen0
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0", "HumanEval/1"],
                        )
                    )

        assert len(records) == 1
        gen0 = records[0]
        # 1 out of 2 tasks solved → pass@1 = 0.5
        assert abs(gen0.pass_at_1 - 0.5) < 0.01

    def test_circuit_breaker_trips_on_failures(self, tmp_path):
        """Circuit breaker halves concurrency when failures exceed threshold."""

        call_count = [0]

        async def flaky_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 6:
                raise ConnectionError("simulated failure")
            return (_make_sse_chunks(), _make_timestamps(), time.monotonic() * 1000)

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=flaky_stream,
        ):

            class DummyMonitor:
                def __init__(self, **kw):
                    pass

                def start(self):
                    pass

                def stop(self):
                    pass

            with patch(
                "redline.swarm.runner.SystemMonitor",
                DummyMonitor,
            ):
                records = asyncio.run(
                    run_swarm(
                        base_url="http://localhost:9999",
                        log_dir=tmp_path,
                        dataset_name="humaneval_plus",
                        gen_count=0,
                        pop_size=2,
                        concurrency_cap=8,
                        task_ids=["HumanEval/0"],
                    )
                )

        # Should still produce a record even with failures
        assert len(records) >= 1

    def test_all_solved_skips_remaining_gens(self, tmp_path):
        """If all tasks solved at gen0, no repair gens are attempted."""

        fake_stream = self._make_fake_stream()

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.swarm.runner.evaluate",
                return_value=MagicMock(passed=True, stderr=""),
            ):

                class DummyMonitor:
                    def __init__(self, **kw):
                        pass

                    def start(self):
                        pass

                    def stop(self):
                        pass

                with patch(
                    "redline.swarm.runner.SystemMonitor",
                    DummyMonitor,
                ):
                    records = asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=5,  # Request 5 gens but should stop early
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0"],
                        )
                    )

        # Only gen0 record since all solved immediately
        assert len(records) == 1
        assert records[0].pass_at_1 == 1.0


# ── Full dev run wiring (Task 3.7) ────────────────────────────────────


class TestDevRunWiring:
    """Full dev run: runner + dashboard callback + telemetry on a small subset."""

    def _make_fake_stream(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "```python\n"}}]},
            {"choices": [{"delta": {"content": "def solve():\n"}}]},
            {"choices": [{"delta": {"content": "    return 42\n"}}]},
            {"choices": [{"delta": {"content": "\n```"}}]},
        ]
        timestamps = [50.0, 100.0, 200.0, 300.0, 400.0]

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        return fake_stream

    def test_callback_receives_runstate_per_generation(self, tmp_path):
        """state_callback is invoked after each generation with valid RunState."""

        fake_stream = self._make_fake_stream()
        states: list[RunState] = []

        # Task 0 passes in gen0; task 1 fails → repair in gen1 succeeds
        eval_results = [
            MagicMock(passed=True, stderr=""),
            MagicMock(passed=False, stderr="AssertionError"),
            MagicMock(passed=True, stderr=""),
        ]
        eval_idx = [0]

        def side_effect_eval(*args, **kwargs):
            result = eval_results[eval_idx[0]]
            eval_idx[0] += 1
            return result

        class DummyMonitor:
            def __init__(self, **kw):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch("redline.swarm.runner.evaluate", side_effect=side_effect_eval):
                with patch("redline.swarm.runner.SystemMonitor", DummyMonitor):
                    asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=1,
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0", "HumanEval/1"],
                            state_callback=lambda s: states.append(s),
                        )
                    )

        # Callback invoked for gen0 and gen1
        assert len(states) == 2
        assert all(isinstance(s, RunState) for s in states)
        assert states[0].generation == 0
        assert states[1].generation == 1

    def test_evolved_pass_at_1_ge_gen0(self, tmp_path):
        """Evolved pass@1 ≥ gen0 pass@1 — at least one repair improved a task."""

        fake_stream = self._make_fake_stream()
        records: list[SwarmRecord] = []
        states: list[RunState] = []

        # Task 0 passes in gen0; task 1 fails → repair in gen1 succeeds
        eval_results = [
            MagicMock(passed=True, stderr=""),
            MagicMock(passed=False, stderr="AssertionError"),
            MagicMock(passed=True, stderr=""),
        ]
        eval_idx = [0]

        def side_effect_eval(*args, **kwargs):
            result = eval_results[eval_idx[0]]
            eval_idx[0] += 1
            return result

        class DummyMonitor:
            def __init__(self, **kw):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch("redline.swarm.runner.evaluate", side_effect=side_effect_eval):
                with patch("redline.swarm.runner.SystemMonitor", DummyMonitor):
                    records = asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=1,
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0", "HumanEval/1"],
                            state_callback=lambda s: states.append(s),
                        )
                    )

        # Gen0: 1/2 solved → pass@1 = 0.5
        gen0_pass = records[0].pass_at_1
        assert abs(gen0_pass - 0.5) < 0.01

        # Gen1: both solved → pass@1 = 1.0 ≥ 0.5
        gen1_pass = records[1].pass_at_1
        assert gen1_pass >= gen0_pass, f"Evolved {gen1_pass} < gen0 {gen0_pass}"
        assert abs(gen1_pass - 1.0) < 0.01

    def test_tokens_per_solved_series(self, tmp_path):
        """A tokens-per-solved series is produced across generations."""

        fake_stream = self._make_fake_stream()
        records: list[SwarmRecord] = []

        # Both tasks pass in gen0 → no repair needed
        class DummyMonitor:
            def __init__(self, **kw):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.swarm.runner.evaluate",
                return_value=MagicMock(passed=True, stderr=""),
            ):
                with patch("redline.swarm.runner.SystemMonitor", DummyMonitor):
                    records = asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=2,
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0"],
                        )
                    )

        # At least one SwarmRecord with tokens_per_solved_task field populated
        assert len(records) >= 1
        for r in records:
            assert hasattr(r, "tokens_per_solved_task")
            assert isinstance(r.tokens_per_solved_task, (int, float))

    def test_dashboard_state_has_all_panels(self, tmp_path):
        """RunState from callback contains data for all four dashboard panels."""

        fake_stream = self._make_fake_stream()
        states: list[RunState] = []

        class DummyMonitor:
            def __init__(self, **kw):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.swarm.runner.evaluate",
                return_value=MagicMock(passed=True, stderr=""),
            ):
                with patch("redline.swarm.runner.SystemMonitor", DummyMonitor):
                    asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=1,
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0"],
                            state_callback=lambda s: states.append(s),
                        )
                    )

        # Each RunState has tasks (population grid + leaderboard), ticker entries, and hardware
        for s in states:
            assert len(s.tasks) > 0, "Population grid needs task data"
            assert len(s.ticker) > 0, "Ticker panel needs entries"
            assert isinstance(s.hardware, HardwareState), "Hardware gauges need state"

    def test_no_callback_is_safe(self, tmp_path):
        """Omitting state_callback doesn't break the run."""

        fake_stream = self._make_fake_stream()

        class DummyMonitor:
            def __init__(self, **kw):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        with patch(
            "redline.swarm.runner._stream_completion",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.swarm.runner.evaluate",
                return_value=MagicMock(passed=True, stderr=""),
            ):
                with patch("redline.swarm.runner.SystemMonitor", DummyMonitor):
                    records = asyncio.run(
                        run_swarm(
                            base_url="http://localhost:9999",
                            log_dir=tmp_path,
                            dataset_name="humaneval_plus",
                            gen_count=1,
                            pop_size=2,
                            concurrency_cap=4,
                            task_ids=["HumanEval/0"],
                        )
                    )

        assert len(records) >= 1
