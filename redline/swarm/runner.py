from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx

from redline.config import Config
from redline.swarm.dataset_loader import Task, load_humaneval_plus
from redline.swarm.evaluator import EvalResult, evaluate
from redline.swarm.population import SwarmPopulation
from redline.swarm.repair_agent import build_repair_prompt, parse_repaired_code
from redline.telemetry.llm_timer import (
    TimerResult,
    compute_timer_result,
    parse_sse_line,
)
from redline.telemetry.logger import Logger
from redline.telemetry.metrics import LLMRecord, SwarmRecord
from redline.telemetry.system_monitor import SystemMonitor
from redline.swarm.dashboard import (
    HardwareState,
    RunState,
    TickerEntry,
    TaskState,
)

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path("logs/phase3_swarm")

# Circuit breaker thresholds
_CB_WINDOW_S = 30.0
_CB_FAIL_THRESHOLD = 5


def _build_run_id() -> str:
    return f"swarm-{uuid.uuid4().hex[:8]}"


class CircuitBreaker:
    """Halves concurrency when >threshold failures occur within a sliding window."""

    def __init__(
        self,
        initial_concurrency: int,
        threshold: int = _CB_FAIL_THRESHOLD,
        window_s: float = _CB_WINDOW_S,
    ):
        self._initial = initial_concurrency
        self._current = initial_concurrency
        self._fail_times: deque[float] = deque()
        self._threshold = threshold
        self._window_s = window_s

    @property
    def concurrency(self) -> int:
        return max(1, self._current)

    def record_success(self):
        pass

    def record_failure(self):
        now = time.monotonic()
        self._fail_times.append(now)
        # Prune old failures outside the window
        while self._fail_times and (now - self._fail_times[0]) > self._window_s:
            self._fail_times.popleft()
        if len(self._fail_times) >= self._threshold:
            self._current = max(1, self._current // 2)
            logger.warning(
                "Circuit breaker tripped: %d failures in %.0fs → concurrency halved to %d",
                len(self._fail_times),
                self._window_s,
                self._current,
            )

    def reset(self):
        """Reset for a new generation (clear failure window)."""
        self._fail_times.clear()
        self._current = self._initial


async def _stream_completion(
    client: httpx.AsyncClient,
    base_url: str,
    messages: List[dict],
) -> tuple[list[dict], list[float], float]:
    """Stream a single completion request.

    Returns (parsed_chunks, chunk_timestamps_ms, request_start_ms).
    """

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "qwen-qwopus-27b-a25b-instruct-2507",
        "messages": messages,
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


def _extract_code(response_text: str) -> Optional[str]:
    """Extract Python code from a fenced ```python block in the response."""
    import re

    match = re.search(
        r"```(?:python|py)\s*\n(.*?)\n```",
        response_text,
        re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).rstrip("\n")


def _build_gen0_prompt(task: Task) -> str:
    """Build the single-shot prompt for gen 0."""
    return (
        f"Write a Python solution for the following problem. "
        f"Return ONLY the code inside a fenced ```python block.\n\n"
        f"{task.prompt}"
    )


async def _gen0_single_task(
    client: httpx.AsyncClient,
    base_url: str,
    task: Task,
    run_id: str,
    logger_obj: Logger,
    cb: CircuitBreaker,
) -> tuple[Optional[str], Optional[EvalResult]]:
    """Run a single task in gen0 (single-shot).

    Returns (code, eval_result) or (None, None) on failure.
    """

    prompt = _build_gen0_prompt(task)

    try:
        chunks, timestamps, request_start_ms = await _stream_completion(
            client, base_url, [{"role": "user", "content": prompt}]
        )

        # Extract code from response
        content_parts = []
        for chunk in chunks:
            choices = chunk.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                content_parts.append(content)

        full_response = "".join(content_parts)
        code = _extract_code(full_response)
        if code is None:
            logger.warning("No fenced code block in response for task %s", task.task_id)
            return None, None

        # Evaluate
        result = evaluate(code, task.test, task.entry_point)

        # Write LLMRecord
        prompt_tokens = max(1, len(prompt.encode("utf-8")) // 4)
        timer = compute_timer_result(
            chunks, timestamps, request_start_ms, prompt_tokens
        )
        record = _timer_to_llm_record(
            timer, prompt, run_id, f"req-{task.task_id}", "swarm"
        )
        logger_obj.write(record)

        cb.record_success()
        return code, result

    except Exception as exc:
        logger.error("Error in gen0 task %s: %s", task.task_id, exc)
        cb.record_failure()
        return None, None


async def _gen_repair_task(
    client: httpx.AsyncClient,
    base_url: str,
    task: Task,
    candidate_code: str,
    stderr: str,
    run_id: str,
    logger_obj: Logger,
    cb: CircuitBreaker,
) -> tuple[Optional[str], Optional[EvalResult]]:
    """Run a repair request for an unsolved task.

    Returns (repaired_code, eval_result) or (None, None) on failure.
    """

    system_prompt, user_prompt = build_repair_prompt(candidate_code, stderr)

    try:
        chunks, timestamps, request_start_ms = await _stream_completion(
            client,
            base_url,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Extract repaired code
        content_parts = []
        for chunk in chunks:
            choices = chunk.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                content_parts.append(content)

        full_response = "".join(content_parts)
        repaired_code = parse_repaired_code(full_response)
        if repaired_code is None:
            logger.warning(
                "No fenced code block in repair response for task %s", task.task_id
            )
            return None, None

        # Evaluate repaired code
        result = evaluate(repaired_code, task.test, task.entry_point)

        # Write LLMRecord
        prompt_tokens = max(1, len(user_prompt.encode("utf-8")) // 4)
        timer = compute_timer_result(
            chunks, timestamps, request_start_ms, prompt_tokens
        )
        record = _timer_to_llm_record(
            timer, user_prompt, run_id, f"req-{task.task_id}", "swarm"
        )
        logger_obj.write(record)

        cb.record_success()
        return repaired_code, result

    except Exception as exc:
        logger.error("Error in repair task %s: %s", task.task_id, exc)
        cb.record_failure()
        return None, None


def _timer_to_llm_record(
    timer: TimerResult,
    prompt: str,
    run_id: str,
    req_id: str,
    phase: str = "swarm",
) -> LLMRecord:
    """Convert a TimerResult into an LLMRecord."""

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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


def _compute_swarm_record(
    generation: int,
    population: SwarmPopulation,
    run_id: str,
    dataset: str,
    total_tokens: int = 0,
) -> SwarmRecord:
    """Compute a SwarmRecord from the current state of all populations."""

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tasks_total = len(population._populations)
    solved = len(population.solved_tasks)
    pass_at_1 = solved / tasks_total if tasks_total > 0 else 0.0

    return SwarmRecord(
        ts=ts,
        phase="swarm",
        run_id=run_id,
        dataset=dataset,
        generation=generation,
        population_size=sum(pop.size for pop in population._populations.values()),
        tasks_total=tasks_total,
        pass_at_1=round(pass_at_1, 4),
        total_solved=solved,
        tokens_per_solved_task=(round(total_tokens / solved, 2) if solved > 0 else 0.0),
        sustained_tps=0.0,
    )


def _build_run_state(
    population: SwarmPopulation,
    generation: int,
    dataset: str,
    ticker_entries: List[TickerEntry],
) -> RunState:
    """Build a RunState snapshot from the current swarm state."""

    tasks: Dict[str, TaskState] = {}
    for task_id, pop in population._populations.items():
        tasks[task_id] = TaskState(
            task_id=task_id,
            generation=generation,
            solved=pop.is_solved,
            best_fitness=pop.best_fitness or 0,
            population_size=pop.size,
        )

    return RunState(
        dataset=dataset,
        generation=generation,
        tasks=tasks,
        ticker=ticker_entries,
        hardware=HardwareState(),
    )


async def run_swarm(
    base_url: str = "http://localhost:1234",
    log_dir: Path = _DEFAULT_LOG_DIR,
    dataset_name: str = "humaneval_plus",
    gen_count: int = 1,
    pop_size: int = 2,
    concurrency_cap: Optional[int] = None,
    task_ids: Optional[List[str]] = None,
    state_callback: Optional[Callable[[RunState], None]] = None,
) -> List[SwarmRecord]:
    """Run the full swarm pipeline: gen0 single-shot → N generations of repair.

    Args:
        base_url: LM Studio API URL.
        log_dir: Directory for JSONL logs.
        dataset_name: "humaneval_plus" or "mbpp_plus".
        gen_count: Number of evolutionary generations (after gen0).
        pop_size: Population size per task.
        concurrency_cap: Max concurrent requests (defaults to Phase 2 knee from config).
        task_ids: Optional subset of task IDs to run (for testing).
        state_callback: Optional callable invoked after each generation with a
            RunState snapshot for dashboard updates.

    Returns:
        List of SwarmRecord, one per generation including gen0.
    """

    cfg = Config()
    if concurrency_cap is None:
        concurrency_cap = cfg.concurrency_cap

    # Load dataset
    if dataset_name == "humaneval_plus":
        tasks_dict = load_humaneval_plus(validate=False)
    else:
        from redline.swarm.dataset_loader import load_mbpp_plus

        tasks_dict = load_mbpp_plus(validate=False)

    # Filter to subset if requested
    if task_ids is not None:
        tasks_dict = {tid: t for tid, t in tasks_dict.items() if tid in task_ids}

    run_id = _build_run_id()
    logger_obj = Logger(log_dir=log_dir, run_id=run_id)
    monitor = SystemMonitor(
        logger=logger_obj,
        phase="swarm",
        run_id=run_id,
        interval_s=2.0,
    )

    population = SwarmPopulation(max_size=pop_size)
    cb = CircuitBreaker(initial_concurrency=concurrency_cap)
    swarm_records: List[SwarmRecord] = []
    total_tokens = 0

    try:
        monitor.start()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            # ── Gen 0: Single-shot ────────────────────────────────
            logger.info("Starting gen0 (single-shot) for %d tasks", len(tasks_dict))

            semaphore = asyncio.Semaphore(cb.concurrency)

            async def _gen0_worker(task_id: str, task: Task):
                async with semaphore:
                    code, result = await _gen0_single_task(
                        client, base_url, task, run_id, logger_obj, cb
                    )
                    if code is not None and result is not None:
                        pop = population.get_or_create(task_id)
                        fitness = 1 if result.passed else 0
                        pop.add(code, fitness=fitness, generation=0)

            await asyncio.gather(
                *[_gen0_worker(tid, t) for tid, t in tasks_dict.items()]
            )

            # Write gen0 SwarmRecord
            record = _compute_swarm_record(
                generation=0,
                population=population,
                run_id=run_id,
                dataset=dataset_name,
                total_tokens=total_tokens,
            )
            logger_obj.write(record)
            swarm_records.append(record)

            # Dashboard callback after gen0
            if state_callback is not None:
                ticker = [
                    TickerEntry(label=f"gen 0", value=f"{record.pass_at_1:.2f} pass@1")
                ]
                rs = _build_run_state(population, 0, dataset_name, ticker)
                state_callback(rs)

            # ── Gen 1–N: Repair loop ─────────────────────────────
            for gen in range(1, gen_count + 1):
                cb.reset()
                unsolved = population.unsolved_tasks
                if not unsolved:
                    logger.info("All tasks solved at gen %d", gen - 1)
                    break

                logger.info(
                    "Starting gen %d for %d unsolved tasks (concurrency=%d)",
                    gen,
                    len(unsolved),
                    cb.concurrency,
                )

                semaphore = asyncio.Semaphore(cb.concurrency)

                async def _repair_worker(task_id: str):
                    async with semaphore:
                        pop = population.population_for(task_id)
                        if pop is None or not pop.candidates:
                            return
                        task = tasks_dict.get(task_id)
                        if task is None:
                            return

                        best = pop.select_best()
                        if best is None:
                            return

                        # Get stderr from last evaluation attempt
                        stderr = "Unknown error"
                        repaired_code, result = await _gen_repair_task(
                            client,
                            base_url,
                            task,
                            best.code,
                            stderr,
                            run_id,
                            logger_obj,
                            cb,
                        )
                        if repaired_code is not None and result is not None:
                            fitness = 1 if result.passed else 0
                            pop.add(repaired_code, fitness=fitness, generation=gen)

                await asyncio.gather(*[_repair_worker(tid) for tid in unsolved])

                # Write SwarmRecord for this generation
                record = _compute_swarm_record(
                    generation=gen,
                    population=population,
                    run_id=run_id,
                    dataset=dataset_name,
                    total_tokens=total_tokens,
                )
                logger_obj.write(record)
                swarm_records.append(record)

                # Dashboard callback after repair gen
                if state_callback is not None:
                    prev_solved = (
                        swarm_records[-2].total_solved if len(swarm_records) >= 2 else 0
                    )
                    delta = record.total_solved - prev_solved
                    ticker = [
                        TickerEntry(
                            label=f"gen {gen}", value=f"{record.pass_at_1:.2f} pass@1"
                        ),
                        TickerEntry(label="repair", value=f"+{delta} solved"),
                    ]
                    rs = _build_run_state(population, gen, dataset_name, ticker)
                    state_callback(rs)

    except Exception as exc:
        logger.error("Error during swarm run: %s", exc)
    finally:
        monitor.stop()
        logger_obj.close()

    # Write metadata
    meta_path = log_dir / run_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset_name,
                "generations": len(swarm_records),
                "tasks_total": len(tasks_dict),
                "pop_size": pop_size,
                "concurrency_cap": concurrency_cap,
            }
        )
    )

    return swarm_records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Redline swarm runner")
    parser.add_argument("--base-url", default=None, help="LM Studio base URL")
    parser.add_argument(
        "--dataset",
        choices=["humaneval_plus", "mbpp_plus"],
        default="humaneval_plus",
        help="Dataset to evaluate on",
    )
    parser.add_argument("--gens", type=int, default=1, help="Number of generations")
    parser.add_argument(
        "--pop-size", type=int, default=2, help="Population size per task"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrency (default: from config)",
    )
    args = parser.parse_args()

    cfg = Config()
    base_url = args.base_url or cfg.base_url

    records = asyncio.run(
        run_swarm(
            base_url=base_url,
            dataset_name=args.dataset,
            gen_count=args.gens,
            pop_size=args.pop_size,
            concurrency_cap=args.concurrency,
        )
    )

    if records:
        print(f"OK — {len(records)} generations completed")
        for r in records:
            print(
                f"  gen={r.generation}: pass@1={r.pass_at_1:.4f}, "
                f"solved={r.total_solved}/{r.tasks_total}"
            )
    else:
        print("FAIL — no records produced (LM Studio unreachable?)")


if __name__ == "__main__":
    main()
