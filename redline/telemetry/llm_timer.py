from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional


@dataclass
class TimerResult:
    """Computed LLM timing metrics from a single SSE stream."""

    ttft_ms: float = 0.0
    inter_token_latency_ms: list[float] = field(default_factory=list)
    mean_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    prompt_tok_per_s: float = 0.0
    decode_tok_per_s: float = 0.0
    total_tokens: int = 0
    completion_tokens: int = 0
    mtp_accepted: Optional[int] = None
    mtp_total: Optional[int] = None


def _percentile(values: list[float], pct: float) -> float:
    """Compute the *pct*-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return float(s[-1])
    d = k - f
    return s[f] + d * (s[c] - s[f])


def parse_sse_line(line: str) -> Optional[dict]:
    """Parse a single SSE line. Returns None for blank / [DONE] lines."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :]
    if payload.strip() == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _extract_delta(chunk: dict) -> Optional[dict]:
    """Extract the delta object from a chunk, or None."""
    choices = chunk.get("choices")
    if not choices:
        return None
    choice = choices[0]
    return choice.get("delta", {})


def _has_content(delta: dict) -> bool:
    """Return True if this delta carries actual content (not just role)."""
    for key in ("content", "reasoning_content"):
        val = delta.get(key, "")
        if val:
            return True
    return False


def _extract_mtp_stats(chunk: dict) -> tuple[Optional[int], Optional[int]]:
    """Extract MTP draft stats from chunk metadata."""
    stats = chunk.get("stats", {})
    if not stats:
        return None, None
    total = stats.get("total_draft_tokens_count")
    accepted = stats.get("accepted_draft_tokens_count")
    return (
        int(total) if total is not None else None,
        int(accepted) if accepted is not None else None,
    )


def compute_timer_result(
    chunks: list[dict],
    chunk_timestamps_ms: list[float],
    request_start_ms: float,
    prompt_tokens: int = 0,
) -> TimerResult:
    """Compute timing metrics from parsed SSE chunks and their arrival timestamps.

    Args:
        chunks: List of parsed JSON dicts from SSE data lines (excluding [DONE]).
        chunk_timestamps_ms: Monotonic timestamp in ms for each chunk arrival.
        request_start_ms: Monotonic timestamp when the request was sent.
        prompt_tokens: Number of tokens in the prompt (for prompt tok/s calc).

    Returns:
        TimerResult with all computed metrics.
    """
    result = TimerResult()
    content_chunks = []
    itl_values: list[float] = []

    for i, chunk in enumerate(chunks):
        delta = _extract_delta(chunk)
        if delta is None:
            continue
        if not _has_content(delta):
            continue

        ts = chunk_timestamps_ms[i] if i < len(chunk_timestamps_ms) else 0.0

        # TTFT: time from request start to first content chunk
        if not content_chunks:
            result.ttft_ms = max(0.0, ts - request_start_ms)

        # ITL: gap between consecutive content chunks
        if content_chunks:
            prev_ts = content_chunks[-1]
            itl_values.append(max(0.0, ts - prev_ts))

        content_chunks.append(ts)

    result.inter_token_latency_ms = itl_values
    result.completion_tokens = len(content_chunks)
    result.total_tokens = prompt_tokens + result.completion_tokens

    # ITL stats
    if itl_values:
        result.mean_itl_ms = sum(itl_values) / len(itl_values)
        result.p50_itl_ms = _percentile(itl_values, 50)
        result.p95_itl_ms = _percentile(itl_values, 95)

    # Decode tok/s: completion tokens / decode time (TTFT to last token)
    if content_chunks and len(content_chunks) > 1:
        decode_time_s = (content_chunks[-1] - content_chunks[0]) / 1000.0
        if decode_time_s > 0:
            result.decode_tok_per_s = result.completion_tokens / decode_time_s

    # Prompt tok/s: prompt tokens / TTFT
    if result.ttft_ms > 0 and prompt_tokens > 0:
        result.prompt_tok_per_s = prompt_tokens / (result.ttft_ms / 1000.0)

    return result


def parse_and_compute(
    sse_text: str,
    chunk_timestamps_ms: list[float],
    request_start_ms: float,
    prompt_tokens: int = 0,
) -> TimerResult:
    """Convenience: parse SSE text and compute metrics in one call."""
    chunks = []
    for line in sse_text.split("\n"):
        parsed = parse_sse_line(line)
        if parsed is not None:
            chunks.append(parsed)
    return compute_timer_result(
        chunks, chunk_timestamps_ms, request_start_ms, prompt_tokens
    )
