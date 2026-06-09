from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from redline.config import Config
from redline.stress.soak_test import (
    DriftPoint,
    SoakResult,
    _build_run_id,
    soak,
)
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
        assert rid.startswith("soak-")

    def test_uniqueness(self):
        ids = {_build_run_id() for _ in range(20)}
        assert len(ids) == 20


# ── soak ───────────────────────────────────────────────────────────────


class TestSoak:
    """Test soak with mocked HTTP."""

    def test_produces_drift_series(self, tmp_path):
        """A soak run writes a non-empty drift series."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(result.drift_points) >= 1

    def test_initial_tok_per_s_set(self, tmp_path):
        """initial_tok_per_s is a positive number when requests succeed."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert result.initial_tok_per_s > 0.0

    def test_final_tok_per_s_set(self, tmp_path):
        """final_tok_per_s is a positive number when requests succeed."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert result.final_tok_per_s > 0.0

    def test_concurrency_level_matches_config(self, tmp_path):
        """Soak runs at sweep_concurrency_end (knee proxy)."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=3,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert result.concurrency_level == 3

    def test_drift_point_has_elapsed_s(self, tmp_path):
        """Each DriftPoint records elapsed seconds from t=0."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        for dp in result.drift_points:
            assert dp.elapsed_s >= 0.0

    def test_drift_point_tok_s_positive(self, tmp_path):
        """Each DriftPoint has positive decode_tok_per_s when requests succeed."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        for dp in result.drift_points:
            assert dp.decode_tok_per_s > 0.0

    def test_writes_llm_records_to_jsonl(self, tmp_path):
        """Per-request LLMRecords are written to the JSONL log file."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        jsonl_files = list(tmp_path.rglob("stress_soak.jsonl"))
        assert len(jsonl_files) >= 1

        all_records = from_jsonl(jsonl_files[0])
        llm_records = [r for r in all_records if isinstance(r, LLMRecord)]
        # At least one batch of requests should have been written
        assert len(llm_records) >= 2

    def test_llm_records_have_correct_concurrency(self, tmp_path):
        """Each LLMRecord carries the soak concurrency level."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=3,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        jsonl_files = list(tmp_path.rglob("stress_soak.jsonl"))
        all_records = from_jsonl(jsonl_files[0])
        llm_records = [r for r in all_records if isinstance(r, LLMRecord)]

        # All records should have concurrency=3 (the knee level)
        for r in llm_records:
            assert r.concurrency == 3

    def test_throttled_false_when_no_drop(self, tmp_path):
        """throttled is False when tok/s stays stable."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert result.throttled is False

    def test_oom_counted(self, tmp_path):
        """MemoryError causes the request to not be recorded as completed."""

        async def oom_stream(*args, **kwargs):
            raise MemoryError("OOM")

        cfg = Config(
            sweep_concurrency_end=1,
            soak_seconds=2,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=oom_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        # No successful requests means initial_tok_per_s stays 0
        assert result.initial_tok_per_s == 0.0

    def test_timeout_counted(self, tmp_path):
        """TimeoutError causes the request to not be recorded as completed."""

        async def timeout_stream(*args, **kwargs):
            raise TimeoutError("request timed out")

        cfg = Config(
            sweep_concurrency_end=1,
            soak_seconds=2,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=timeout_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert result.initial_tok_per_s == 0.0

    def test_duration_respected(self, tmp_path):
        """Soak runs for approximately the configured duration."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert result.duration_s == 4

    def test_peak_thermal_pressure_tracked(self, tmp_path):
        """peak_thermal_pressure is at least 0."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert result.peak_thermal_pressure >= 0

    def test_drift_point_toks_drift_pct_computed(self, tmp_path):
        """tok_s_drift_pct is computed relative to t=0 baseline."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        # With stable tok/s, drift should be near 0 (within tolerance for timing)
        for dp in result.drift_points[1:]:
            assert abs(dp.tok_s_drift_pct) < 5.0

    def test_smoke_run_no_crash(self, tmp_path):
        """A soak run at soak_seconds=4 completes without crashing."""

        chunks = _make_sse_chunks()
        timestamps = _make_timestamps()

        async def fake_stream(*args, **kwargs):
            return (chunks, timestamps, time.monotonic() * 1000)

        cfg = Config(
            sweep_concurrency_end=2,
            soak_seconds=4,
            poll_interval=2.0,
        )

        with patch(
            "redline.stress.soak_test._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            result = asyncio.run(
                soak(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert isinstance(result, SoakResult)
        assert len(result.drift_points) >= 1
