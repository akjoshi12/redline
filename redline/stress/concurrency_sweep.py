from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from redline.baseline.runner import _stream_one
from redline.baseline.task_suite import Task, load_tasks
from redline.config import Config
from redline.telemetry.llm_timer import (
    TimerResult,
    compute_timer_result,
)
from redline.telemetry.logger import Logger
from redline.telemetry.metrics import LLMRecord
from redline.telemetry.system_monitor import SystemMonitor

_DEFAULT_LOG_DIR = Path("logs/phase2_stress")


@dataclass
class LevelRecord:
    """Per-concurrency-level aggregate record."""

    level: int  # concurrency level
    completed_count: int = 0
    avg_decode_tok_per_s: float = 0.0
    avg_ttft_ms: float = 0.0
    oom_count: int = 0
    timeout_count: int = 0


def _build_run_id() -> str:
    return f"sweep-{uuid.uuid4().hex[:8]}"


async def _stream_and_record(
    client: httpx.AsyncClient,
    base_url: str,
    task: Task,
    run_id: str,
    logger: Logger,
    concurrency_level: int,
) -> LLMRecord | None:
    """Stream a single task at the given concurrency level and write its record.

    Returns the LLMRecord on success, None on failure (OOM/timeout counted).
    """

    req_id = f"req-c{concurrency_level}-{task.id}"
    prompt_tokens = max(1, len(task.prompt.encode("utf-8")) // 4)

    try:
        chunks, timestamps, request_start_ms = await _stream_one(
            client, base_url, task.prompt
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
            phase="stress_concurrency",
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
            concurrency=concurrency_level,
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
        print(f"  Error at concurrency {concurrency_level} on {task.id}: {exc}")
        return None


async def sweep(
    base_url: str = "http://localhost:1234",
    log_dir: Path = _DEFAULT_LOG_DIR,
    cfg: Config | None = None,
) -> list[LevelRecord]:
    """Ramp concurrency from 1 to N (config cap), hold each level, collect metrics.

    At each concurrency level, runs that many requests concurrently using the task
    suite. Writes per-request LLMRecords and a LevelRecord aggregate per level.

    Circuit-breaks if error spike detected (>50% failures at any level).

    Returns list of LevelRecord, one per concurrency level tested.
    """

    config = cfg or Config()
    run_id = _build_run_id()

    logger = Logger(log_dir=log_dir, run_id=run_id)
    monitor = SystemMonitor(
        logger=logger,
        phase="stress_concurrency",
        run_id=run_id,
        interval_s=config.poll_interval,
    )

    tasks = load_tasks()
    level_records: list[LevelRecord] = []
    circuit_broken = False

    try:
        monitor.start()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            for level in range(
                config.sweep_concurrency_start, config.sweep_concurrency_end + 1
            ):
                if circuit_broken:
                    break

                coros = [
                    _stream_and_record(client, base_url, task, run_id, logger, level)
                    for task in tasks[:level]
                ]

                results = await asyncio.gather(*coros, return_exceptions=True)

                completed: list[LLMRecord] = []
                oom_count = 0
                timeout_count = 0

                for r in results:
                    if isinstance(r, LLMRecord):
                        completed.append(r)
                    elif isinstance(r, (MemoryError,)):
                        oom_count += 1
                    elif isinstance(
                        r, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)
                    ):
                        timeout_count += 1
                    else:
                        pass

                # Circuit-break on error spike (>50% failures)
                total = len(results)
                errors = total - len(completed)
                if total > 0 and (errors / total) > 0.5:
                    print(
                        f"  Circuit break at concurrency {level}: "
                        f"{errors}/{total} failed"
                    )
                    circuit_broken = True

                avg_tps = (
                    sum(r.decode_tok_per_s for r in completed) / len(completed)
                    if completed
                    else 0.0
                )
                avg_ttft = (
                    sum(r.ttft_ms for r in completed) / len(completed)
                    if completed
                    else 0.0
                )

                lr = LevelRecord(
                    level=level,
                    completed_count=len(completed),
                    avg_decode_tok_per_s=round(avg_tps, 2),
                    avg_ttft_ms=round(avg_ttft, 2),
                    oom_count=oom_count,
                    timeout_count=timeout_count,
                )

                level_records.append(lr)
                print(
                    f"  Level {level}: completed={len(completed)}, "
                    f"avg_tok/s={avg_tps:.1f}, avg_ttft={avg_ttft:.0f}ms"
                )

    finally:
        monitor.stop()
        logger.close()

    return level_records


def main() -> None:
    cfg_path = Path("config.json")
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text())
        cfg = Config(**data)
    else:
        cfg = Config()

    print(f"Concurrency sweep: 1 → {cfg.sweep_concurrency_end}")
    records = asyncio.run(sweep(base_url=cfg.base_url, cfg=cfg))

    for lr in records:
        print(
            f"  Level {lr.level}: completed={lr.completed_count}, "
            f"avg_tok/s={lr.avg_decode_tok_per_s:.1f}, "
            f"oom={lr.oom_count}, timeout={lr.timeout_count}"
        )


if __name__ == "__main__":
    main()
