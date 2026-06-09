from __future__ import annotations

import json
from pathlib import Path

import pytest

from redline.telemetry.llm_timer import (
    TimerResult,
    _extract_delta,
    _has_content,
    compute_timer_result,
    parse_and_compute,
    parse_sse_line,
)

# ── SSE line parsing ──────────────────────────────────────────────────


def test_parse_sse_line_valid():
    line = 'data: {"key": "value"}'
    result = parse_sse_line(line)
    assert result == {"key": "value"}


def test_parse_sse_line_blank():
    assert parse_sse_line("") is None
    assert parse_sse_line("   ") is None


def test_parse_sse_line_done():
    assert parse_sse_line("data: [DONE]") is None


def test_parse_sse_line_no_data_prefix():
    assert parse_sse_line('{"key": "value"}') is None


def test_parse_sse_line_malformed_json():
    assert parse_sse_line("data: {bad json") is None


# ── Delta extraction helpers ──────────────────────────────────────────


def test_extract_delta_normal():
    chunk = {"choices": [{"delta": {"content": "hi"}}]}
    delta = _extract_delta(chunk)
    assert delta == {"content": "hi"}


def test_extract_delta_no_choices():
    assert _extract_delta({}) is None


def test_has_content_true():
    assert _has_content({"content": "hello"}) is True
    assert _has_content({"reasoning_content": "thinking..."}) is True


def test_has_content_false():
    assert _has_content({"role": "assistant"}) is False
    assert _has_content({}) is False


# ── Core timing computation with known values ─────────────────────────


class TestComputeTimerResult:
    """Hand-computed expected values for every metric."""

    def test_basic_ttft_and_itl(self):
        """3 content chunks, known timestamps.

        request_start = 0ms
        chunk[1] "Hello" at 200ms → TTFT = 200ms
        chunk[2] " world" at 400ms → ITL = 200ms
        chunk[3] "!" at 650ms → ITL = 250ms

        mean_itl = (200 + 250) / 2 = 225.0
        p50_itl = 225.0
        p95_itl ≈ 247.5
        decode_time = 650 - 200 = 450ms → 3 tok / 0.45s = 6.667 tok/s
        prompt_tok_per_s = 10 / 0.2s = 50.0 tok/s
        """
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        timestamps = [50.0, 200.0, 400.0, 650.0]

        result = compute_timer_result(
            chunks, timestamps, request_start_ms=0.0, prompt_tokens=10
        )

        assert result.ttft_ms == pytest.approx(200.0)
        assert len(result.inter_token_latency_ms) == 2
        assert result.inter_token_latency_ms[0] == pytest.approx(200.0)
        assert result.inter_token_latency_ms[1] == pytest.approx(250.0)
        assert result.mean_itl_ms == pytest.approx(225.0)
        assert result.p50_itl_ms == pytest.approx(225.0)
        assert result.completion_tokens == 3
        assert result.total_tokens == 13
        assert result.decode_tok_per_s == pytest.approx(6.667, rel=0.01)
        assert result.prompt_tok_per_s == pytest.approx(50.0)

    def test_single_content_chunk(self):
        """Only one content chunk → no ITL, decode_tok/s = 0."""
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hi"}}]},
        ]
        timestamps = [10.0, 300.0]

        result = compute_timer_result(
            chunks, timestamps, request_start_ms=0.0, prompt_tokens=5
        )

        assert result.ttft_ms == pytest.approx(300.0)
        assert len(result.inter_token_latency_ms) == 0
        assert result.mean_itl_ms == 0.0
        assert result.decode_tok_per_s == 0.0
        assert result.completion_tokens == 1

    def test_no_content_chunks(self):
        """Only role delta, no content → all zeros."""
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
        ]
        timestamps = [50.0]

        result = compute_timer_result(chunks, timestamps, request_start_ms=0.0)

        assert result.ttft_ms == 0.0
        assert result.completion_tokens == 0
        assert len(result.inter_token_latency_ms) == 0

    def test_reasoning_content_counted(self):
        """reasoning_content is treated as content for timing."""
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"choices": [{"delta": {"reasoning_content": "...more"}}]},
        ]
        timestamps = [0.0, 100.0, 300.0]

        result = compute_timer_result(chunks, timestamps, request_start_ms=0.0)

        assert result.ttft_ms == pytest.approx(100.0)
        assert len(result.inter_token_latency_ms) == 1
        assert result.completion_tokens == 2

    def test_p95_with_many_chunks(self):
        """Verify p95 with a larger set of ITL values."""
        # Build chunks: role + 10 content chunks, evenly spaced at 10ms each
        chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
        for i in range(10):
            chunks.append({"choices": [{"delta": {"content": f"t{i}"}}]})

        timestamps = [0.0] + [float(i * 10) for i in range(1, 11)]

        result = compute_timer_result(chunks, timestamps, request_start_ms=0.0)

        assert len(result.inter_token_latency_ms) == 9
        # All ITL values are 10ms → p95 should be ~10
        assert result.p95_itl_ms == pytest.approx(10.0)

    def test_zero_ttft(self):
        """First content chunk arrives at same time as request start."""
        chunks = [
            {"choices": [{"delta": {"content": "Hi"}}]},
        ]
        timestamps = [0.0]

        result = compute_timer_result(chunks, timestamps, request_start_ms=0.0)

        assert result.ttft_ms == pytest.approx(0.0)
        # prompt_tok_per_s should be 0 (division by zero guard)
        assert result.prompt_tok_per_s == 0.0


# ── End-to-end with synthetic fixture ─────────────────────────────────


class TestParseAndCompute:
    """End-to-end parsing of SSE text → TimerResult."""

    def test_synthetic_fixture(self):
        """Parse the synthetic fixture file and verify metrics."""
        fixture_path = Path(__file__).parent / "fixtures" / "sse_synthetic.txt"
        sse_text = fixture_path.read_text()

        # 3 content chunks: "Hello", " world", "!"
        # Timestamps: request at 0, role chunk at 50ms (skipped), then 200/400/650ms
        timestamps = [50.0, 200.0, 400.0, 650.0]

        result = parse_and_compute(
            sse_text, timestamps, request_start_ms=0.0, prompt_tokens=10
        )

        assert result.ttft_ms == pytest.approx(200.0)
        assert len(result.inter_token_latency_ms) == 2
        assert result.completion_tokens == 3
        assert result.decode_tok_per_s > 0

    def test_empty_stream(self):
        """Empty string → no chunks, all zeros."""
        result = parse_and_compute("", [], request_start_ms=0.0)
        assert result.ttft_ms == 0.0
        assert result.completion_tokens == 0


# ── Real fixture smoke test ───────────────────────────────────────────


def test_real_fixture_parses():
    """The real LM Studio fixture parses without error."""
    fixture_path = Path(__file__).parent / "fixtures" / "sse_sample.txt"
    sse_text = fixture_path.read_text()

    # All timestamps are the same (instant stream), so we inject synthetic timing
    chunks_count = 0
    for line in sse_text.split("\n"):
        parsed = parse_sse_line(line)
        if parsed is not None:
            chunks_count += 1

    assert chunks_count > 0

    timestamps = [float(i * 50) for i in range(chunks_count)]
    result = parse_and_compute(
        sse_text, timestamps, request_start_ms=0.0, prompt_tokens=20
    )

    # The real fixture has reasoning_content chunks → should have content
    assert result.completion_tokens > 0
    assert len(result.inter_token_latency_ms) > 0
