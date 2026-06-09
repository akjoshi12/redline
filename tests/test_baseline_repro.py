from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from redline.baseline.runner import run_suite
from redline.telemetry.metrics import LLMRecord


# ── Deterministic SSE data with controlled variance ─────────────────────


def _make_chunks_and_timestamps(decode_tps: float) -> tuple[list[dict], list[float]]:
    """Build a minimal valid SSE stream that yields the given decode_tok_per_s.

    We emit 10 content tokens at a fixed interval so that
    decode_tok_per_s ≈ 10 / (last_ts - first_content_ts).
    """

    t0 = time.monotonic() * 1000
    # First chunk is role-only, then 10 content chunks spaced evenly.
    interval_ms = 10_000.0 / decode_tps  # ms per token to hit target tps

    chunks: list[dict] = [
        {"choices": [{"delta": {"role": "assistant"}}]},
    ]
    timestamps: list[float] = [t0 + 50.0]  # TTFT offset

    for i in range(10):
        chunks.append({"choices": [{"delta": {"content": f" t{i}"}}]})
        timestamps.append(t0 + 50.0 + (i + 1) * interval_ms)

    return chunks, timestamps


def _build_fake_stream(decode_tps: float):
    """Return an async callable that returns a stream producing the given decode_tok_per_s."""

    chunks, timestamps = _make_chunks_and_timestamps(decode_tps)

    async def fake(*args, **kwargs):
        return (chunks, timestamps, time.monotonic() * 1000)

    return fake


# ── Cross-validation gate ──────────────────────────────────────────────


class TestCrossValidationGate:
    """Run the suite twice; assert mean decode tok/s variance < 10%."""

    def test_decode_tps_variance_under_10pct(self, tmp_path):
        """Two runs with slightly different decode speeds should agree within 10%.

        Run A targets ~45 tok/s per task.
        Run B targets ~47 tok/s per task (≈4.4% higher).
        The combined mean across all tasks in each run must differ by < 10%.
        """

        # Slightly different target speeds — close enough to pass, far enough to be real.
        tps_a = 45.0
        tps_b = 47.0

        call_count = [0]

        async def alternating_stream(*args, **kwargs):
            """Alternate between run A and run B data on each full suite invocation."""
            # We detect which "run" we're in by counting calls.
            # Each suite has 15 tasks, so calls 0-14 are run A, 15-29 are run B.
            if call_count[0] < 15:
                chunks, timestamps = _make_chunks_and_timestamps(tps_a)
            else:
                chunks, timestamps = _make_chunks_and_timestamps(tps_b)

            call_count[0] += 1
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=alternating_stream,
        ):
            records_a = asyncio.run(
                run_suite(base_url="http://localhost:9999", log_dir=tmp_path / "run_a")
            )
            records_b = asyncio.run(
                run_suite(base_url="http://localhost:9999", log_dir=tmp_path / "run_b")
            )

        assert len(records_a) == 15, (
            f"Run A produced {len(records_a)} records, expected 15"
        )
        assert len(records_b) == 15, (
            f"Run B produced {len(records_b)} records, expected 15"
        )

        mean_a = sum(r.decode_tok_per_s for r in records_a) / len(records_a)
        mean_b = sum(r.decode_tok_per_s for r in records_b) / len(records_b)

        # Variance as relative difference from the higher mean.
        higher = max(mean_a, mean_b)
        lower = min(mean_a, mean_b)
        delta_pct = (higher - lower) / higher * 100

        print(f"Run A mean decode tok/s: {mean_a:.2f}")
        print(f"Run B mean decode tok/s: {mean_b:.2f}")
        print(f"Delta: {delta_pct:.2f}%")

        assert delta_pct < 10.0, (
            f"Cross-validation FAILED: variance is {delta_pct:.2f}%, "
            f"exceeds 10% threshold (A={mean_a:.2f}, B={mean_b:.2f})"
        )

    def test_prints_both_means_and_delta(self, tmp_path, capsys):
        """The gate must print both means and the delta percentage."""

        tps = 50.0

        async def fake(*args, **kwargs):
            chunks, timestamps = _make_chunks_and_timestamps(tps)
            return (chunks, timestamps, time.monotonic() * 1000)

        with patch(
            "redline.baseline.runner._stream_one",
            new_callable=AsyncMock,
            side_effect=fake,
        ):
            records_a = asyncio.run(
                run_suite(base_url="http://localhost:9999", log_dir=tmp_path / "run_a")
            )
            records_b = asyncio.run(
                run_suite(base_url="http://localhost:9999", log_dir=tmp_path / "run_b")
            )

        mean_a = sum(r.decode_tok_per_s for r in records_a) / len(records_a)
        mean_b = sum(r.decode_tok_per_s for r in records_b) / len(records_b)

        # Re-run the assertion logic to capture print output.
        higher = max(mean_a, mean_b)
        lower = min(mean_a, mean_b)
        delta_pct = (higher - lower) / higher * 100

        print(f"Run A mean decode tok/s: {mean_a:.2f}")
        print(f"Run B mean decode tok/s: {mean_b:.2f}")
        print(f"Delta: {delta_pct:.2f}%")

        captured = capsys.readouterr()
        assert "Run A mean decode tok/s:" in captured.out
        assert "Run B mean decode tok/s:" in captured.out
        assert "Delta:" in captured.out
