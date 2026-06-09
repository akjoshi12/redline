from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from redline.config import Config
from redline.stress.concurrency_sweep import (
    LevelRecord,
    _build_run_id,
    sweep,
)
from redline.telemetry.metrics import LLMRecord, from_jsonl


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
        assert rid.startswith("sweep-")

    def test_uniqueness(self):
        ids = {_build_run_id() for _ in range(20)}
        assert len(ids) == 20


# ── sweep ──────────────────────────────────────────────────────────────


class TestSweep:
    """Test concurrency sweep with mocked HTTP."""

    def test_one_level_record_per_concurrency(self, tmp_path):
        """Each concurrency level produces exactly one LevelRecord."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=3,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 3
        for i, lr in enumerate(records):
            assert lr.level == i + 1

    def test_completed_count_matches_level(self, tmp_path):
        """At level N, exactly N requests complete (one per task)."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=3,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        for lr in records:
            assert lr.completed_count == lr.level

    def test_avg_decode_tok_per_s_computed(self, tmp_path):
        """avg_decode_tok_per_s is a positive number when requests succeed."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=2,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        for lr in records:
            assert lr.avg_decode_tok_per_s > 0.0

    def test_config_cap_respected(self, tmp_path):
        """Sweep stops at sweep_concurrency_end."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=2,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 2
        assert max(lr.level for lr in records) == 2

    def test_circuit_break_on_error_spike(self, tmp_path):
        """Sweep stops early when >50% of requests fail at a level."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()
        call_count = [0]

        async def flaky_stream(*args, **kwargs):
            call_count[0] += 1
            # First two calls succeed (level 1: 1 req, level 2: 2 reqs)
            if call_count[0] <= 3:
                return (chunks, timestamps, time.monotonic() * 1000)
            # All subsequent calls fail (level 3: 3 reqs → all fail → >50%)
            raise ConnectionError("simulated failure")

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=4,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=flaky_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        # Level 3 should trigger circuit break (0/3 completed = 100% failure)
        assert len(records) < 4
        last_level = max(lr.level for lr in records)
        assert last_level <= 3

    def test_oom_counted(self, tmp_path):
        """MemoryError is counted as OOM."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def oom_stream(*args, **kwargs):
            raise MemoryError("OOM")

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=1,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=oom_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 1
        lr = records[0]
        assert lr.oom_count >= 1
        assert lr.completed_count == 0

    def test_timeout_counted(self, tmp_path):
        """TimeoutError is counted as timeout."""

        async def timeout_stream(*args, **kwargs):
            raise TimeoutError("request timed out")

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=1,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=timeout_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 1
        lr = records[0]
        assert lr.timeout_count >= 1
        assert lr.completed_count == 0

    def test_writes_llm_records_to_jsonl(self, tmp_path):
        """Per-request LLMRecords are written to the JSONL log file."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=2,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        jsonl_files = list(tmp_path.rglob("stress_concurrency.jsonl"))
        assert len(jsonl_files) >= 1

        all_records = from_jsonl(jsonl_files[0])
        llm_records = [r for r in all_records if isinstance(r, LLMRecord)]
        # Level 1: 1 request, Level 2: 2 requests → 3 total
        assert len(llm_records) == 3

    def test_llm_records_have_correct_concurrency(self, tmp_path):
        """Each LLMRecord carries the concurrency level it was run at."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=3,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        jsonl_files = list(tmp_path.rglob("stress_concurrency.jsonl"))
        all_records = from_jsonl(jsonl_files[0])
        llm_records = [r for r in all_records if isinstance(r, LLMRecord)]

        # Level 1: 1 record with concurrency=1
        level1 = [r for r in llm_records if r.concurrency == 1]
        assert len(level1) == 1

        # Level 2: 2 records with concurrency=2
        level2 = [r for r in llm_records if r.concurrency == 2]
        assert len(level2) == 2

        # Level 3: 3 records with concurrency=3
        level3 = [r for r in llm_records if r.concurrency == 3]
        assert len(level3) == 3

    def test_partial_failure_level(self, tmp_path):
        """When some requests fail at a level, completed_count reflects only successes."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()
        call_count = [0]

        async def partial_stream(*args, **kwargs):
            call_count[0] += 1
            # First request succeeds, second fails at level 2
            if call_count[0] == 2:
                raise ConnectionError("simulated failure")
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_start=1,
            sweep_concurrency_end=2,
        )

        with patch(
            "redline.stress.concurrency_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=partial_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        # Level 2 should have completed_count < level (1 out of 2 succeeded)
        lr_level2 = [lr for lr in records if lr.level == 2][0]
        assert lr_level2.completed_count == 1
        assert lr_level2.completed_count < lr_level2.level
