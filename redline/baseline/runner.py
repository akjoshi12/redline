from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

from redline.baseline.task_suite import Task, load_tasks, suite_hash
from redline.config import Config
from redline.telemetry.llm_timer import (
    TimerResult,
    compute_timer_result,
    parse_sse_line,
)
from redline.telemetry.logger import Logger
from redline.telemetry.metrics import LLMRecord
from redline.telemetry.system_monitor import SystemMonitor

_DEFAULT_LOG_DIR = Path("logs/phase1_baseline")


def _build_run_id() -> str:
    return f"baseline-{uuid.uuid4().hex[:8]}"


async def _stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
) -> tuple[list[dict], list[float], float]:
    """Stream a single completion request.

    Returns (parsed_chunks, chunk_timestamps_ms, request_start_ms).
    """

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "qwen-qwopus-27b-a25b-instruct-2507",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.0,
    }

    request_start_ms = time.monotonic() * 1000
    chunks: list[dict] = []
    timestamps: list[float] = []

    async with client.stream("POST", url, json=payload, timeout=300) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            raise RuntimeError(f"LM Studio returned {resp.status_code}: {body[:500]}")

        async for line in resp.aiter_lines():
            parsed = parse_sse_line(line)
            if parsed is not None:
                chunks.append(parsed)
                timestamps.append(time.monotonic() * 1000)

    return chunks, timestamps, request_start_ms


def _timer_to_llm_record(
    timer: TimerResult,
    prompt: str,
    run_id: str,
    req_id: str,
    phase: str = "baseline",
) -> LLMRecord:
    """Convert a TimerResult into an LLMRecord."""

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Estimate prompt tokens (rough: 1 token ≈ 4 chars for English text)
    prompt_tokens = max(1, len(prompt.encode("utf-8")) // 4)

    mtp_enabled = False
    mtp_acceptance_rate = None
    spec_decode_delta_tps = None

    if timer.mtp_total is not None and timer.mtp_accepted is not None:
        mtp_enabled = True
        if timer.mtp_total > 0:
            mtp_acceptance_rate = round(timer.mtp_accepted / timer.mtp_total, 4)

    return LLMRecord(
        ts=ts,
        phase=phase,
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
        context_length=prompt_tokens + timer.completion_tokens,
        concurrency=1,
        mtp_enabled=mtp_enabled,
        mtp_acceptance_rate=mtp_acceptance_rate,
        spec_decode_delta_tps=spec_decode_delta_tps,
    )


async def _run_one_task(
    client: httpx.AsyncClient,
    base_url: str,
    task: Task,
    run_id: str,
    logger: Logger,
) -> LLMRecord | None:
    """Stream a single task and write its LLMRecord.

    Returns the record on success, None on failure (error is printed).
    """

    req_id = f"req-{task.id}"

    try:
        chunks, timestamps, request_start_ms = await _stream_one(
            client, base_url, task.prompt
        )

        timer = compute_timer_result(
            chunks,
            timestamps,
            request_start_ms,
            prompt_tokens=max(1, len(task.prompt.encode("utf-8")) // 4),
        )

        record = _timer_to_llm_record(timer, task.prompt, run_id, req_id)
        logger.write(record)
        return record

    except Exception as exc:
        print(f"Error streaming {task.id}: {exc}")
        return None


async def run_once(
    base_url: str = "http://localhost:1234",
    log_dir: Path = _DEFAULT_LOG_DIR,
) -> LLMRecord | None:
    """Stream a single prompt and return the resulting LLMRecord.

    Also writes system records via SystemMonitor in the background.
    Returns None if LM Studio is unreachable.
    """

    run_id = _build_run_id()
    tasks = load_tasks()
    task = tasks[0]  # first task for --once

    logger = Logger(log_dir=log_dir, run_id=run_id)
    monitor = SystemMonitor(
        logger=logger,
        phase="baseline",
        run_id=run_id,
        interval_s=2.0,
    )

    try:
        monitor.start()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            record = await _run_one_task(client, base_url, task, run_id, logger)

    except Exception as exc:
        print(f"Error streaming {task.id}: {exc}")
        return None
    finally:
        monitor.stop()
        logger.close()

    if record is None:
        return None

    # Write suite hash to metadata file
    meta_path = log_dir / run_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"run_id": run_id, "suite_hash": suite_hash(), "task_id": task.id})
    )

    return record


async def run_suite(
    base_url: str = "http://localhost:1234",
    log_dir: Path = _DEFAULT_LOG_DIR,
) -> list[LLMRecord]:
    """Run the full task suite with system monitor attached.

    Writes one LLMRecord per prompt and system records every 2s in the background.
    Returns the list of successfully recorded LLMRecords (may be shorter than
    the suite if some requests fail).
    """

    run_id = _build_run_id()
    tasks = load_tasks()

    logger = Logger(log_dir=log_dir, run_id=run_id)
    monitor = SystemMonitor(
        logger=logger,
        phase="baseline",
        run_id=run_id,
        interval_s=2.0,
    )

    records: list[LLMRecord] = []

    try:
        monitor.start()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            for task in tasks:
                record = await _run_one_task(client, base_url, task, run_id, logger)
                if record is not None:
                    records.append(record)

    except Exception as exc:
        print(f"Error during suite run: {exc}")
    finally:
        monitor.stop()
        logger.close()

    # Write suite hash to metadata file
    meta_path = log_dir / run_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_hash": suite_hash(),
                "tasks_total": len(tasks),
                "records_written": len(records),
            }
        )
    )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Redline baseline runner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Stream a single prompt and exit (smoke test)",
    )
    parser.add_argument("--base-url", default=None, help="LM Studio base URL")
    args = parser.parse_args()

    cfg = Config()
    base_url = args.base_url or cfg.base_url

    if args.once:
        record = asyncio.run(run_once(base_url=base_url))
        if record is not None:
            print(
                f"OK — {record.req_id}: ttft={record.ttft_ms}ms, "
                f"decode_tok_s={record.decode_tok_per_s}"
            )
        else:
            print("FAIL — could not reach LM Studio")

    else:
        records = asyncio.run(run_suite(base_url=base_url))
        if records:
            print(f"OK — {len(records)}/{len(load_tasks())} tasks completed")
            for r in records:
                print(
                    f"  {r.req_id}: ttft={r.ttft_ms}ms, "
                    f"decode_tok_s={r.decode_tok_per_s}"
                )
        else:
            print("FAIL — no records produced (LM Studio unreachable?)")


if __name__ == "__main__":
    main()
