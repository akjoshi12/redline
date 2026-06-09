from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from redline.baseline.runner import _stream_one
from redline.config import Config
from redline.telemetry.llm_timer import (
    TimerResult,
    compute_timer_result,
)
from redline.telemetry.logger import Logger
from redline.telemetry.metrics import LLMRecord
from redline.telemetry.system_monitor import SystemMonitor

_DEFAULT_LOG_DIR = Path("logs/phase2_stress")

# Base text used to synthesize prompts of arbitrary length.
# ~4 chars ≈ 1 token for English text, so we pad to reach target token count.
_SYNTHETIC_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
)


@dataclass
class ContextRecord:
    """Per-context-length aggregate record."""

    context_length: int  # target prompt token count
    decode_tok_per_s: float = 0.0
    ttft_ms: float = 0.0
    prompt_tok_per_s: float = 0.0
    mem_used_gb: float = 0.0


def _build_run_id() -> str:
    return f"ctx-sweep-{uuid.uuid4().hex[:8]}"


def _make_prompt(target_tokens: int) -> str:
    """Build a synthetic prompt of approximately *target_tokens* tokens.

    Uses ~4 chars per token as rough estimate, padding with repeated filler text.
    """

    target_chars = target_tokens * 4
    prompt = f"Analyze the following text and summarize it in three bullet points:\n\n"
    while len(prompt.encode("utf-8")) < target_chars:
        prompt += _SYNTHETIC_FILLER
    return prompt[:target_chars]


async def _stream_and_record(
    client: httpx.AsyncClient,
    base_url: str,
    context_length: int,
    run_id: str,
    logger: Logger,
) -> LLMRecord | None:
    """Stream a single request at the given context length and write its record.

    Returns the LLMRecord on success, None on failure.
    """

    prompt = _make_prompt(context_length)
    req_id = f"req-ctx{context_length}"
    prompt_tokens = max(1, len(prompt.encode("utf-8")) // 4)

    try:
        chunks, timestamps, request_start_ms = await _stream_one(
            client, base_url, prompt
        )

        timer = compute_timer_result(
            chunks,
            timestamps,
            request_start_ms,
            prompt_tokens=prompt_tokens,
        )

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        record = LLMRecord(
            ts=ts,
            phase="stress_context",
            run_id=run_id,
            req_id=req_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=timer.completion_tokens,
            total_tokens=timer.total_tokens,
            ttft_ms=round(timer.ttft_ms, 2),
            inter_token_latency_ms=(
                [round(v, 2) for v in timer.inter_token_latency_ms]
                if timer.inter_token_latency_ms
                else None
            ),
            mean_itl_ms=round(timer.mean_itl_ms, 2),
            p50_itl_ms=round(timer.p50_itl_ms, 2),
            p95_itl_ms=round(timer.p95_itl_ms, 2),
            prompt_tok_per_s=round(timer.prompt_tok_per_s, 2),
            decode_tok_per_s=round(timer.decode_tok_per_s, 2),
            context_length=context_length,
            concurrency=1,
        )

        logger.write(record)
        return record

    except MemoryError:
        raise
    except (TimeoutError, asyncio.TimeoutError):
        raise
    except httpx.TimeoutException:
        raise
    except Exception as exc:
        print(f"  Error at context {context_length}: {exc}")
        return None


async def sweep(
    base_url: str = "http://localhost:1234",
    log_dir: Path = _DEFAULT_LOG_DIR,
    cfg: Config | None = None,
) -> list[ContextRecord]:
    """Ramp context length from start to end (config range), concurrency=1.

    At each context length, sends one request and measures decode tok/s.
    Writes per-request LLMRecords and a ContextRecord aggregate per length.

    Returns list of ContextRecord, one per context length tested.
    """

    config = cfg or Config()
    run_id = _build_run_id()

    logger = Logger(log_dir=log_dir, run_id=run_id)
    monitor = SystemMonitor(
        logger=logger,
        phase="stress_context",
        run_id=run_id,
        interval_s=config.poll_interval,
    )

    context_lengths: list[int] = []
    length = config.sweep_context_start
    while length <= config.sweep_context_end:
        context_lengths.append(length)
        length += config.sweep_context_step

    records: list[ContextRecord] = []

    try:
        monitor.start()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            for ctx_len in context_lengths:
                try:
                    record = await _stream_and_record(
                        client, base_url, ctx_len, run_id, logger
                    )
                except MemoryError:
                    record = None
                except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
                    record = None

                if record is not None:
                    cr = ContextRecord(
                        context_length=ctx_len,
                        decode_tok_per_s=record.decode_tok_per_s,
                        ttft_ms=record.ttft_ms,
                        prompt_tok_per_s=record.prompt_tok_per_s,
                    )
                    records.append(cr)
                    print(
                        f"  ctx={ctx_len}: decode_tok/s={record.decode_tok_per_s:.1f}, "
                        f"ttft={record.ttft_ms:.0f}ms, prompt_tok/s={record.prompt_tok_per_s:.1f}"
                    )
                else:
                    cr = ContextRecord(
                        context_length=ctx_len,
                    )
                    records.append(cr)
                    print(f"  ctx={ctx_len}: FAILED")

    finally:
        monitor.stop()
        logger.close()

    return records


def main() -> None:
    cfg_path = Path("config.json")
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text())
        cfg = Config(**data)
    else:
        cfg = Config()

    print(
        f"Context sweep: {cfg.sweep_context_start} → "
        f"{cfg.sweep_context_end} (step {cfg.sweep_context_step})"
    )
    records = asyncio.run(sweep(base_url=cfg.base_url, cfg=cfg))

    for cr in records:
        print(
            f"  ctx={cr.context_length}: decode_tok/s={cr.decode_tok_per_s:.1f}, "
            f"ttft={cr.ttft_ms:.0f}ms"
        )


if __name__ == "__main__":
    main()
