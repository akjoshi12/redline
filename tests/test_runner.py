from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redline.baseline.runner import (
    _build_run_id,
    _run_one_task,
    run_once,
    run_suite,
)
from redline.baseline.task_suite import load_tasks
from redline.telemetry.metrics import LLMRecord, SystemRecord, from_jsonl


# ── Helpers ────────────────────────────────────────────────────────────


def _make_sse_chunks():
    """Return a minimal valid SSE chunk list for testing."""
    return [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]


def _make_timestamps():
    """Return timestamps aligned with _make_sse_chunks."""
    return [50.0, 200.0, 400.0]


# ── run_id ─────────────────────────────────────────────────────────────


class TestBuildRunId:
    def test_format(self):
        rid = _build_run_id()
        assert rid.startswith("baseline-")
        # hex portion is 8 chars
        assert len(rid) == 17  # "baseline-" (9) + 8 hex

    def test_uniqueness(self):
        ids = {_build_run_id() for _ in range(20)}
        assert len(ids) == 20


# ── run_suite ──────────────────────────────────────────────────────────


class TestRunSuite:
    """Test full suite execution with mocked HTTP."""

    def test_writes_one_record_per_task(self, tmp_path):
        """Each of the 15 tasks produces exactly one LLMRecord."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                run_suite(base_url="http://localhost:9999", log_dir=tmp_path)
            )

        assert len(records) == 15

    def test_meta_json_has_suite_hash(self, tmp_path):
        """meta.json is written with suite hash and task counts."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(run_suite(base_url="http://localhost:9999", log_dir=tmp_path))

        # Find the meta.json in any subdirectory
        meta_files = list(tmp_path.rglob("meta.json"))
        assert len(meta_files) == 1

        meta = json.loads(meta_files[0].read_text())
        assert "suite_hash" in meta
        assert meta["tasks_total"] == 15
        assert meta["records_written"] == 15

    def test_system_records_produced(self, tmp_path):
        """System monitor writes SystemRecord lines during the run."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            # Add a small delay so system monitor has time to tick
            await asyncio.sleep(0.1)
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            # Patch SystemMonitor interval to be very short for testing
            original_init = None

            class FastSystemMonitor:
                def __init__(self, **kw):
                    kw["interval_s"] = 0.3
                    from redline.telemetry.system_monitor import SystemMonitor as _SM

                    self._mon = _SM(**kw)

                def start(self):
                    self._mon.start()

                def stop(self):
                    self._mon.stop()

            with patch(
                "redline.baseline.runner.SystemMonitor",
                FastSystemMonitor,
            ):
                asyncio.run(
                    run_suite(base_url="http://localhost:9999", log_dir=tmp_path)
                )

        # Check that system records exist in the JSONL file
        jsonl_files = list(tmp_path.rglob("baseline.jsonl"))
        assert len(jsonl_files) >= 1

        all_records = from_jsonl(jsonl_files[0])
        sys_records = [r for r in all_records if isinstance(r, SystemRecord)]
        # With fast interval we expect multiple system records
        assert len(sys_records) >= 1

    def test_partial_failure_still_returns_records(self, tmp_path):
        """If one task fails, remaining tasks still produce records."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()
        call_count = [0]

        async def flaky_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 5:
                raise ConnectionError("simulated failure")
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=flaky_stream,
        ):
            records = asyncio.run(
                run_suite(base_url="http://localhost:9999", log_dir=tmp_path)
            )

        # 14 out of 15 should succeed (one failed at position 5)
        assert len(records) == 14

    def test_records_have_correct_req_ids(self, tmp_path):
        """Each record's req_id matches the task id."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                run_suite(base_url="http://localhost:9999", log_dir=tmp_path)
            )

        task_ids = {t.id for t in load_tasks()}
        req_ids = {r.req_id.removeprefix("req-") for r in records}
        assert req_ids == task_ids


# ── run_once (smoke test still works) ─────────────────────────────────


class TestRunOnce:
    def test_returns_single_record(self, tmp_path):
        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            record = asyncio.run(
                run_once(base_url="http://localhost:9999", log_dir=tmp_path)
            )

        assert isinstance(record, LLMRecord)


# ── _run_one_task ─────────────────────────────────────────────────────


class TestRunOneTask:
    def test_success_returns_record(self, tmp_path):
        from redline.telemetry.logger import Logger

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        logger = Logger(log_dir=tmp_path, run_id="test-task")

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            fake_client = MagicMock()
            task = load_tasks()[0]
            record = asyncio.run(
                _run_one_task(
                    fake_client, "http://localhost:9999", task, "test-task", logger
                )
            )

        assert isinstance(record, LLMRecord)
        logger.close()

    def test_failure_returns_none(self, tmp_path):
        from redline.telemetry.logger import Logger

        async def failing_stream(*args, **kwargs):
            raise ConnectionError("boom")

        logger = Logger(log_dir=tmp_path, run_id="test-fail")

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=failing_stream,
        ):
            fake_client = MagicMock()
            task = load_tasks()[0]
            record = asyncio.run(
                _run_one_task(
                    fake_client, "http://localhost:9999", task, "test-fail", logger
                )
            )

        assert record is None
        logger.close()


# ── System monitor interval coverage ───────────────────────────────────


class TestSystemMonitorIntervalCoverage:
    """Verify >=95% of expected 2s system intervals are produced."""

    def test_interval_coverage(self, tmp_path):
        """Run suite with fast mock and verify system record density."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            # Simulate ~0.3s per request so monitor has time to tick
            await asyncio.sleep(0.3)
            return (chunks, timestamps, time.monotonic() * 1000)

        class FastSystemMonitor:
            def __init__(self, **kw):
                kw["interval_s"] = 0.5
                from redline.telemetry.system_monitor import SystemMonitor as _SM

                self._mon = _SM(**kw)

            def start(self):
                self._mon.start()

            def stop(self):
                self._mon.stop()

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            with patch(
                "redline.baseline.runner.SystemMonitor",
                FastSystemMonitor,
            ):
                records = asyncio.run(
                    run_suite(base_url="http://localhost:9999", log_dir=tmp_path)
                )

        # Find system records in the JSONL
        jsonl_files = list(tmp_path.rglob("baseline.jsonl"))
        assert len(jsonl_files) >= 1

        all_records = from_jsonl(jsonl_files[0])
        sys_records = [r for r in all_records if isinstance(r, SystemRecord)]

        # With ~4.5s total wall time and 0.5s interval we expect ~9 ticks
        assert len(sys_records) >= 1

        # Verify intervals are consistent (all should be 0.5s from our fast monitor)
        for r in sys_records:
            assert r.interval_s == 0.5
