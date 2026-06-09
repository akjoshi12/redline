from __future__ import annotations

import json
from pathlib import Path

import pytest

from redline.telemetry.metrics import (
    LLMRecord,
    SwarmRecord,
    SystemRecord,
    from_jsonl,
    to_jsonl,
)

# ── Dataclass defaults ────────────────────────────────────────────────


def test_llm_record_defaults():
    r = LLMRecord(ts="2025-01-01T00:00:00Z", phase="baseline", run_id="r1", req_id="q1")
    assert r.prompt_tokens == 0
    assert r.concurrency == 1
    assert r.mtp_enabled is False
    assert r.inter_token_latency_ms is None


def test_system_record_defaults():
    r = SystemRecord(ts="2025-01-01T00:00:00Z", phase="baseline", run_id="r1")
    assert r.interval_s == 2.0
    assert r.thermal_pressure_level == 0


def test_swarm_record_defaults():
    r = SwarmRecord(ts="2025-01-01T00:00:00Z", phase="swarm", run_id="r1")
    assert r.generation == 0
    assert r.pass_at_1 == 0.0


# ── Full-field construction ───────────────────────────────────────────


def test_llm_record_full():
    r = LLMRecord(
        ts="2025-06-08T12:00:00Z",
        phase="baseline",
        run_id="run-42",
        req_id="req-7",
        prompt_tokens=512,
        completion_tokens=256,
        total_tokens=768,
        ttft_ms=320.5,
        inter_token_latency_ms=[12.0, 14.0],
        mean_itl_ms=13.0,
        p50_itl_ms=12.5,
        p95_itl_ms=18.0,
        prompt_tok_per_s=160.0,
        decode_tok_per_s=45.0,
        context_length=2048,
        concurrency=3,
        mtp_enabled=True,
        mtp_acceptance_rate=0.72,
        spec_decode_delta_tps=5.2,
    )
    assert r.mtp_acceptance_rate == 0.72


def test_system_record_full():
    r = SystemRecord(
        ts="2025-06-08T12:00:00Z",
        phase="stress",
        run_id="run-99",
        interval_s=2.0,
        unified_mem_used_gb=48.3,
        unified_mem_peak_gb=55.1,
        gpu_active_residency_pct=97.2,
        thermal_pressure_level=2,
        cpu_freq_mhz=4500.0,
        power_watts=62.0,
    )
    assert r.gpu_active_residency_pct == 97.2


def test_swarm_record_full():
    r = SwarmRecord(
        ts="2025-06-08T12:00:00Z",
        phase="swarm",
        run_id="run-s1",
        dataset="humaneval_plus",
        generation=3,
        population_size=8,
        tasks_total=164,
        pass_at_1=0.52,
        total_solved=85,
        tokens_per_solved_task=1200.0,
        sustained_tps=38.0,
    )
    assert r.total_solved == 85


# ── JSONL roundtrip ───────────────────────────────────────────────────


def _tmp(tmp_path, name):
    return str(tmp_path / name)


class TestLLMRecordRoundtrip:
    def test_single(self, tmp_path):
        rec = LLMRecord(
            ts="2025-06-08T12:00:00Z",
            phase="baseline",
            run_id="r1",
            req_id="q1",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            ttft_ms=200.0,
            inter_token_latency_ms=[10.0, 12.0],
            mean_itl_ms=11.0,
            p50_itl_ms=10.5,
            p95_itl_ms=14.0,
            prompt_tok_per_s=100.0,
            decode_tok_per_s=50.0,
            context_length=2048,
            concurrency=1,
            mtp_enabled=True,
            mtp_acceptance_rate=0.8,
            spec_decode_delta_tps=3.0,
        )
        p = _tmp(tmp_path, "llm.jsonl")
        to_jsonl([rec], p)
        result = from_jsonl(p)
        assert len(result) == 1
        r2 = result[0]
        assert isinstance(r2, LLMRecord)
        assert r2.prompt_tokens == 100
        assert r2.inter_token_latency_ms == [10.0, 12.0]
        assert r2.mtp_acceptance_rate == 0.8

    def test_multiple(self, tmp_path):
        recs = [
            LLMRecord(ts="T", phase="p", run_id="r", req_id=f"q{i}") for i in range(5)
        ]
        p = _tmp(tmp_path, "llm_multi.jsonl")
        to_jsonl(recs, p)
        result = from_jsonl(p)
        assert len(result) == 5


class TestSystemRecordRoundtrip:
    def test_single(self, tmp_path):
        rec = SystemRecord(
            ts="2025-06-08T12:00:00Z",
            phase="stress",
            run_id="r1",
            interval_s=2.0,
            unified_mem_used_gb=48.3,
            unified_mem_peak_gb=55.1,
            gpu_active_residency_pct=97.2,
            thermal_pressure_level=2,
            cpu_freq_mhz=4500.0,
            power_watts=62.0,
        )
        p = _tmp(tmp_path, "sys.jsonl")
        to_jsonl([rec], p)
        result = from_jsonl(p)
        assert len(result) == 1
        r2 = result[0]
        assert isinstance(r2, SystemRecord)
        assert r2.unified_mem_used_gb == 48.3
        assert r2.gpu_active_residency_pct == 97.2


class TestSwarmRecordRoundtrip:
    def test_single(self, tmp_path):
        rec = SwarmRecord(
            ts="2025-06-08T12:00:00Z",
            phase="swarm",
            run_id="r1",
            dataset="humaneval_plus",
            generation=3,
            population_size=8,
            tasks_total=164,
            pass_at_1=0.52,
            total_solved=85,
            tokens_per_solved_task=1200.0,
            sustained_tps=38.0,
        )
        p = _tmp(tmp_path, "swarm.jsonl")
        to_jsonl([rec], p)
        result = from_jsonl(p)
        assert len(result) == 1
        r2 = result[0]
        assert isinstance(r2, SwarmRecord)
        assert r2.dataset == "humaneval_plus"
        assert r2.total_solved == 85


# ── Mixed file support ────────────────────────────────────────────────


def test_mixed_records(tmp_path):
    recs = [
        LLMRecord(ts="T", phase="p", run_id="r", req_id="q1"),
        SystemRecord(ts="T", phase="p", run_id="r"),
        SwarmRecord(ts="T", phase="p", run_id="r"),
    ]
    p = _tmp(tmp_path, "mixed.jsonl")
    to_jsonl(recs, p)
    result = from_jsonl(p)
    assert len(result) == 3
    assert isinstance(result[0], LLMRecord)
    assert isinstance(result[1], SystemRecord)
    assert isinstance(result[2], SwarmRecord)


# ── Error handling ────────────────────────────────────────────────────


def test_unknown_record_type(tmp_path):
    p = _tmp(tmp_path, "bad.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"ts": "T", "unknown_field": 1}) + "\n")
    with pytest.raises(ValueError, match="Cannot determine record type"):
        from_jsonl(p)


# ── to_jsonl creates parent dirs ──────────────────────────────────────


def test_to_jsonl_creates_parent(tmp_path):
    p = str(tmp_path / "deep" / "nested" / "file.jsonl")
    rec = LLMRecord(ts="T", phase="p", run_id="r", req_id="q1")
    to_jsonl([rec], p)
    assert Path(p).exists()


# ── Empty lines skipped ───────────────────────────────────────────────


def test_from_jsonl_skips_empty_lines(tmp_path):
    p = _tmp(tmp_path, "sparse.jsonl")
    rec = LLMRecord(ts="T", phase="p", run_id="r", req_id="q1")
    with open(p, "w") as f:
        f.write("\n")
        f.write(
            json.dumps({"ts": "T", "phase": "p", "run_id": "r", "req_id": "q1"}) + "\n"
        )
        f.write("\n\n")
    result = from_jsonl(p)
    assert len(result) == 1
