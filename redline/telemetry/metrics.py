from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LLMRecord:
    """Per-request LLM metrics emitted per completion request."""

    ts: str
    phase: str
    run_id: str
    req_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    inter_token_latency_ms: Optional[List[float]] = None
    mean_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    prompt_tok_per_s: float = 0.0
    decode_tok_per_s: float = 0.0
    context_length: int = 0
    concurrency: int = 1
    mtp_enabled: bool = False
    mtp_acceptance_rate: Optional[float] = None
    spec_decode_delta_tps: Optional[float] = None


@dataclass
class SystemRecord:
    """Per-interval system metrics emitted every 2s during any phase."""

    ts: str
    phase: str
    run_id: str
    interval_s: float = 2.0
    unified_mem_used_gb: float = 0.0
    unified_mem_peak_gb: float = 0.0
    gpu_active_residency_pct: float = 0.0
    thermal_pressure_level: int = 0
    cpu_freq_mhz: float = 0.0
    power_watts: float = 0.0


@dataclass
class SwarmRecord:
    """Per-generation swarm metrics emitted during Phase 3."""

    ts: str
    phase: str
    run_id: str
    dataset: str = ""
    generation: int = 0
    population_size: int = 0
    tasks_total: int = 0
    pass_at_1: float = 0.0
    total_solved: int = 0
    tokens_per_solved_task: float = 0.0
    sustained_tps: float = 0.0


def _record_to_dict(record) -> dict:
    return asdict(record)


def to_jsonl(records, path: str | Path) -> None:
    """Write a list of records to a JSONL file (one record per line)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")


def from_jsonl(path: str | Path):
    """Read a JSONL file and return a list of the appropriate record objects.

    Detects record type by inspecting required fields on each line.
    Mixed files are supported — each line is deserialized independently.
    """

    result = []
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            record = _deserialize(d)
            result.append(record)
    return result


def _deserialize(d: dict):
    """Route a dict to the correct dataclass based on its fields."""

    if "req_id" in d:
        return LLMRecord(**d)
    if "interval_s" in d:
        return SystemRecord(**d)
    if "generation" in d:
        return SwarmRecord(**d)
    raise ValueError(f"Cannot determine record type for: {d}")
