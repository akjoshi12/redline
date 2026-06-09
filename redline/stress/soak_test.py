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
class DriftPoint:
    """A single measurement of tok/s and thermal drift from t=0 baseline."""

    elapsed_s: float  # seconds since soak start
    decode_tok_per_s: float  # current avg decode tok/s in this interval
    thermal_pressure_level: int  # current thermal pressure (0-4)
    tok_s_drift_pct: float = 0.0  # percentage change from t=0 baseline tok/s
    thermal_drift: int = 0  # delta from t=0 thermal level


@dataclass
class SoakResult:
    """Aggregate result of a soak test run."""

    run_id: str
    concurrency_level: int
    duration_s: float
    drift_points: list[DriftPoint] = field(default_factory=list)
    initial_tok_per_s: float = 0.0
    final_tok_per_s: float = 0.0
    peak_thermal_pressure: int = 0
    throttled: bool = False  # True if tok/s dropped >20% from t=0


def _build_run_id() -> str:
    return f"soak-{uuid.uuid4().hex[:8]}"


async def _stream_and_record(
    client: httpx.AsyncClient,
    base_url: str,
    task: Task,
    run_id: str,
    logger: Logger,
    concurrency_level: int,
) -> LLMRecord | None:
    """Stream a single task at the given concurrency level and write its record."""

    req_id = f"req-soak-c{concurrency_level}-{task.id}"
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
            phase="stress_soak",
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
        print(f"  Soak error at concurrency {concurrency_level} on {task.id}: {exc}")
        return None


async def soak(
    base_url: str = "http://localhost:1234",
    log_dir: Path = _DEFAULT_LOG_DIR,
    cfg: Config | None = None,
) -> SoakResult:
    """Run sustained load at knee-concurrency for config.soak_seconds.

    Continuously streams requests at the configured concurrency level (defaults to
    sweep_concurrency_end as proxy for knee), capturing tok/s drift and thermal
    pressure over time. Writes per-request LLMRecords and a DriftPoint every
    poll_interval seconds.

    Returns a SoakResult with the full drift series.
    """

    config = cfg or Config()
    run_id = _build_run_id()

    # Knee concurrency: use sweep_concurrency_end as proxy (the last level tested)
    knee_concurrency = config.sweep_concurrency_end

    logger = Logger(log_dir=log_dir, run_id=run_id)
    monitor = SystemMonitor(
        logger=logger,
        phase="stress_soak",
        run_id=run_id,
        interval_s=config.poll_interval,
    )

    tasks = load_tasks()
    drift_points: list[DriftPoint] = []
    initial_tok_per_s: float | None = None
    initial_thermal: int | None = None

    result = SoakResult(
        run_id=run_id,
        concurrency_level=knee_concurrency,
        duration_s=config.soak_seconds,
    )

    avg_tps = 0.0
    tok_s_drift_pct = 0.0
    completed: list[LLMRecord] = []

    try:
        monitor.start()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            t0 = time.monotonic()
            interval_end = t0 + config.poll_interval

            while True:
                now = time.monotonic()
                elapsed = now - t0

                if elapsed >= config.soak_seconds:
                    break

                # Run a batch of requests at knee concurrency
                coros = [
                    _stream_and_record(
                        client, base_url, task, run_id, logger, knee_concurrency
                    )
                    for task in tasks[:knee_concurrency]
                ]

                results = await asyncio.gather(*coros, return_exceptions=True)

                completed: list[LLMRecord] = []
                oom_count = 0
                timeout_count = 0

                for r in results:
                    if isinstance(r, LLMRecord):
                        completed.append(r)
                    elif isinstance(r, MemoryError):
                        oom_count += 1
                    elif isinstance(
                        r, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)
                    ):
                        timeout_count += 1

                # Compute interval avg tok/s
                if completed:
                    avg_tps = sum(r.decode_tok_per_s for r in completed) / len(
                        completed
                    )
                else:
                    avg_tps = 0.0

                # Read latest system metrics from the JSONL to get thermal pressure
                # We use a simple approach: collect from the last SystemRecord written
                # Since we can't easily read back, we track via the monitor's data
                # Instead, we'll compute drift based on what we know

                if initial_tok_per_s is None and completed:
                    initial_tok_per_s = avg_tps
                    result.initial_tok_per_s = round(initial_tok_per_s, 2)

                # Compute tok/s drift percentage from t=0 baseline
                tok_s_drift_pct = 0.0
                if initial_tok_per_s is not None and initial_tok_per_s > 0:
                    tok_s_drift_pct = (
                        (initial_tok_per_s - avg_tps) / initial_tok_per_s * 100
                    )

                # We need thermal pressure — read from the JSONL log file
                thermal_level = 0
                jsonl_path = Path(log_dir) / run_id / "stress_soak.jsonl"
                if jsonl_path.exists():
                    try:
                        with open(jsonl_path, "r") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                d = json.loads(line)
                                if "interval_s" in d:
                                    thermal_level = int(
                                        d.get("thermal_pressure_level", 0)
                                    )
                    except (json.JSONDecodeError, OSError):
                        pass

                if initial_thermal is None:
                    initial_thermal = thermal_level

                thermal_drift = thermal_level - (initial_thermal or 0)

                dp = DriftPoint(
                    elapsed_s=round(elapsed, 1),
                    decode_tok_per_s=round(avg_tps, 2),
                    thermal_pressure_level=thermal_level,
                    tok_s_drift_pct=round(tok_s_drift_pct, 2),
                    thermal_drift=thermal_drift,
                )

                drift_points.append(dp)

                if thermal_level > result.peak_thermal_pressure:
                    result.peak_thermal_pressure = thermal_level

                # Wait until next interval boundary
                wait_time = max(0, interval_end - time.monotonic())
                if wait_time > 0 and elapsed < config.soak_seconds:
                    await asyncio.sleep(wait_time)
                interval_end += config.poll_interval

            # Final drift point at end of soak
            result.final_tok_per_s = round(avg_tps, 2) if completed else 0.0
            result.throttled = tok_s_drift_pct > 20.0

    finally:
        monitor.stop()
        logger.close()

    result.drift_points = drift_points

    return result


def main() -> None:
    cfg_path = Path("config.json")
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text())
        cfg = Config(**data)
    else:
        cfg = Config()

    print(
        f"Soak test: concurrency={cfg.sweep_concurrency_end}, "
        f"duration={cfg.soak_seconds}s"
    )
    result = asyncio.run(soak(base_url=cfg.base_url, cfg=cfg))

    print(f"\n  Run ID: {result.run_id}")
    print(
        f"  Initial tok/s: {result.initial_tok_per_s:.1f}, "
        f"Final tok/s: {result.final_tok_per_s:.1f}"
    )
    print(f"  Peak thermal pressure: {result.peak_thermal_pressure}")
    print(f"  Throttled: {result.throttled}")

    for dp in result.drift_points:
        print(
            f"  t={dp.elapsed_s:.0f}s: tok/s={dp.decode_tok_per_s:.1f}, "
            f"drift={dp.tok_s_drift_pct:+.1f}%, thermal={dp.thermal_pressure_level}"
        )


if __name__ == "__main__":
    main()
