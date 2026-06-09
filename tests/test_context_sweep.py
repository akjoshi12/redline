from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from redline.config import Config
from redline.stress.context_sweep import (
    ContextRecord,
    _build_run_id,
    _make_prompt,
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


def _make_timestamps(base: float | None = None):
    """Return timestamps aligned with _make_sse_chunks.

    If *base* is given, returns [base+50, base+200, base+400] so that the
    first content chunk arrives after request_start_ms (TTFT > 0).
    Otherwise returns absolute small values for backward compat.
    """
    if base is not None:
        return [base + 50.0, base + 200.0, base + 400.0]
    return [50.0, 200.0, 400.0]


# ── run_id ─────────────────────────────────────────────────────────────


class TestBuildRunId:
    def test_format(self):
        rid = _build_run_id()
        assert rid.startswith("ctx-sweep-")

    def test_uniqueness(self):
        ids = {_build_run_id() for _ in range(20)}
        assert len(ids) == 20


# ── prompt generation ──────────────────────────────────────────────────


class TestMakePrompt:
    def test_approximate_token_length_small(self):
        """512 tokens → ~2048 chars (4 chars per token)."""
        prompt = _make_prompt(512)
        char_count = len(prompt.encode("utf-8"))
        # Allow ±20% tolerance for the rough estimate
        assert 1600 <= char_count <= 2500

    def test_approximate_token_length_large(self):
        """4096 tokens → ~16384 chars."""
        prompt = _make_prompt(4096)
        char_count = len(prompt.encode("utf-8"))
        assert 13000 <= char_count <= 20000

    def test_monotonic_length(self):
        """Larger token count produces longer or equal prompt."""
        p512 = _make_prompt(512)
        p1024 = _make_prompt(1024)
        assert len(p1024.encode("utf-8")) >= len(p512.encode("utf-8"))

    def test_contains_instruction(self):
        """Prompt starts with the analysis instruction."""
        prompt = _make_prompt(512)
        assert "Analyze the following text" in prompt


# ── sweep ──────────────────────────────────────────────────────────────


class TestSweep:
    """Test context sweep with mocked HTTP."""

    def test_one_record_per_context_length(self, tmp_path):
        """Each context length produces exactly one ContextRecord."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=1536,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        # 512, 1024, 1536 → 3 lengths
        assert len(records) == 3
        for i, cr in enumerate(records):
            expected = 512 + i * 512
            assert cr.context_length == expected

    def test_decode_tok_per_s_computed(self, tmp_path):
        """decode_tok_per_s is a positive number when requests succeed."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=1024,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        for cr in records:
            assert cr.decode_tok_per_s > 0.0

    def test_config_range_respected(self, tmp_path):
        """Sweep covers exactly the configured range."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=2048,
            sweep_context_step=1024,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        # 512, 1536 → 2 lengths (step=1024)
        assert len(records) == 2
        assert records[0].context_length == 512
        assert records[1].context_length == 1536

    def test_writes_llm_records_to_jsonl(self, tmp_path):
        """Per-request LLMRecords are written to the JSONL log file."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=1024,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        jsonl_files = list(tmp_path.rglob("stress_context.jsonl"))
        assert len(jsonl_files) >= 1

        all_records = from_jsonl(jsonl_files[0])
        llm_records = [r for r in all_records if isinstance(r, LLMRecord)]
        # 512 and 1024 → 2 records
        assert len(llm_records) == 2

    def test_llm_records_have_concurrency_one(self, tmp_path):
        """Each LLMRecord carries concurrency=1."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=1024,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        jsonl_files = list(tmp_path.rglob("stress_context.jsonl"))
        all_records = from_jsonl(jsonl_files[0])
        llm_records = [r for r in all_records if isinstance(r, LLMRecord)]

        for r in llm_records:
            assert r.concurrency == 1

    def test_llm_records_have_correct_context_length(self, tmp_path):
        """Each LLMRecord carries the context length it was run at."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=1536,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        jsonl_files = list(tmp_path.rglob("stress_context.jsonl"))
        all_records = from_jsonl(jsonl_files[0])
        llm_records = [r for r in all_records if isinstance(r, LLMRecord)]

        ctx_lengths = sorted(r.context_length for r in llm_records)
        assert ctx_lengths == [512, 1024, 1536]

    def test_oom_counted_as_failure(self, tmp_path):
        """MemoryError produces a ContextRecord with zero metrics."""

        async def oom_stream(*args, **kwargs):
            raise MemoryError("OOM")

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=512,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=oom_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 1
        cr = records[0]
        assert cr.context_length == 512
        assert cr.decode_tok_per_s == 0.0

    def test_timeout_counted_as_failure(self, tmp_path):
        """TimeoutError produces a ContextRecord with zero metrics."""

        async def timeout_stream(*args, **kwargs):
            raise TimeoutError("request timed out")

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=512,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=timeout_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 1
        cr = records[0]
        assert cr.context_length == 512
        assert cr.decode_tok_per_s == 0.0

    def test_partial_failure_mixed_results(self, tmp_path):
        """When some context lengths fail, only successes have metrics."""

        chunks = _make_sse_chunks()
        call_count = [0]

        async def partial_stream(*args, **kwargs):
            call_count[0] += 1
            # First succeeds (512), second fails (1024), third succeeds (1536)
            if call_count[0] == 2:
                raise ConnectionError("simulated failure")
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=1536,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=partial_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 3
        # First and third have metrics, second has zeros
        assert records[0].decode_tok_per_s > 0.0
        assert records[1].decode_tok_per_s == 0.0
        assert records[2].decode_tok_per_s > 0.0

    def test_ttft_recorded(self, tmp_path):
        """ttft_ms is recorded in ContextRecord."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=512,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 1
        cr = records[0]
        assert cr.ttft_ms > 0.0

    def test_prompt_tok_per_s_recorded(self, tmp_path):
        """prompt_tok_per_s is recorded in ContextRecord."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=512,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 1
        cr = records[0]
        assert cr.prompt_tok_per_s > 0.0

    def test_single_length(self, tmp_path):
        """Sweep with start==end produces exactly one record."""

        chunks = _make_sse_chunks()

        async def fake_stream(*args, **kwargs):
            base = time.monotonic() * 1000
            return (chunks, _make_timestamps(base), base)

        cfg = Config(
            sweep_context_start=512,
            sweep_context_end=512,
            sweep_context_step=512,
        )

        with patch(
            "redline.stress.context_sweep._stream_one",
            new_callable=AsyncMock,
            side_effect=fake_stream,
        ):
            records = asyncio.run(
                sweep(base_url="http://localhost:9999", log_dir=tmp_path, cfg=cfg)
            )

        assert len(records) == 1
        assert records[0].context_length == 512
