from __future__ import annotations

import json
from pathlib import Path

import pytest

from redline.report.curves import (
    generate_all,
    pass_at_1_vs_generation,
    thermal_vs_time,
    throughput_vs_concurrency,
)
from redline.telemetry.metrics import (
    LLMRecord,
    SwarmRecord,
    SystemRecord,
    to_jsonl,
)


def _write_llm_jsonl(tmp_path, records):
    p = tmp_path / "llm.jsonl"
    to_jsonl(records, p)
    return str(p)


def _write_swarm_jsonl(tmp_path, records):
    p = tmp_path / "swarm.jsonl"
    to_jsonl(records, p)
    return str(p)


def _write_sys_jsonl(tmp_path, records):
    p = tmp_path / "sys.jsonl"
    to_jsonl(records, p)
    return str(p)


# ── throughput_vs_concurrency ────────────────────────────────────────


class TestThroughputVsConcurrency:
    def test_generates_png(self, tmp_path):
        recs = (
            [
                LLMRecord(
                    ts="T",
                    phase="stress",
                    run_id="r1",
                    req_id=f"q{i}",
                    concurrency=1,
                    decode_tok_per_s=50.0,
                )
                for i in range(3)
            ]
            + [
                LLMRecord(
                    ts="T",
                    phase="stress",
                    run_id="r1",
                    req_id=f"q{i}",
                    concurrency=2,
                    decode_tok_per_s=80.0,
                )
                for i in range(3, 6)
            ]
            + [
                LLMRecord(
                    ts="T",
                    phase="stress",
                    run_id="r1",
                    req_id=f"q{i}",
                    concurrency=4,
                    decode_tok_per_s=60.0,
                )
                for i in range(6, 9)
            ]
        )
        jsonl = _write_llm_jsonl(tmp_path, recs)
        out = tmp_path / "out.png"
        result = throughput_vs_concurrency(jsonl, out)
        assert result == out
        assert out.exists()

    def test_default_output_path(self, tmp_path):
        recs = [
            LLMRecord(
                ts="T",
                phase="stress",
                run_id="r1",
                req_id="q0",
                concurrency=1,
                decode_tok_per_s=50.0,
            )
        ]
        jsonl = _write_llm_jsonl(tmp_path, recs)
        result = throughput_vs_concurrency(jsonl)
        assert result.name == "throughput_vs_concurrency.png"
        assert result.exists()

    def test_no_llm_records_raises(self, tmp_path):
        recs = [SystemRecord(ts="T", phase="stress", run_id="r1")]
        p = _write_sys_jsonl(tmp_path, recs)
        with pytest.raises(ValueError, match="No LLMRecord"):
            throughput_vs_concurrency(p)


# ── pass_at_1_vs_generation ─────────────────────────────────────────


class TestPassAt1VsGeneration:
    def test_generates_png(self, tmp_path):
        recs = [
            SwarmRecord(
                ts="T",
                phase="swarm",
                run_id="r1",
                generation=i,
                pass_at_1=0.3 + i * 0.1,
            )
            for i in range(5)
        ]
        jsonl = _write_swarm_jsonl(tmp_path, recs)
        out = tmp_path / "out.png"
        result = pass_at_1_vs_generation(jsonl, out)
        assert result == out
        assert out.exists()

    def test_default_output_path(self, tmp_path):
        recs = [
            SwarmRecord(ts="T", phase="swarm", run_id="r1", generation=0, pass_at_1=0.5)
        ]
        jsonl = _write_swarm_jsonl(tmp_path, recs)
        result = pass_at_1_vs_generation(jsonl)
        assert result.name == "pass_at_1_vs_generation.png"
        assert result.exists()

    def test_no_swarm_records_raises(self, tmp_path):
        recs = [LLMRecord(ts="T", phase="stress", run_id="r1", req_id="q0")]
        p = _write_llm_jsonl(tmp_path, recs)
        with pytest.raises(ValueError, match="No SwarmRecord"):
            pass_at_1_vs_generation(p)


# ── thermal_vs_time ────────────────────────────────────────────────


class TestThermalVsTime:
    def test_generates_png(self, tmp_path):
        recs = [
            SystemRecord(
                ts=f"2026-06-08T14:{i:02d}:00Z",
                phase="stress",
                run_id="r1",
                thermal_pressure_level=i % 3,
            )
            for i in range(10)
        ]
        jsonl = _write_sys_jsonl(tmp_path, recs)
        out = tmp_path / "out.png"
        result = thermal_vs_time(jsonl, out)
        assert result == out
        assert out.exists()

    def test_default_output_path(self, tmp_path):
        recs = [SystemRecord(ts="T", phase="stress", run_id="r1")]
        jsonl = _write_sys_jsonl(tmp_path, recs)
        result = thermal_vs_time(jsonl)
        assert result.name == "thermal_vs_time.png"
        assert result.exists()

    def test_no_system_records_raises(self, tmp_path):
        recs = [LLMRecord(ts="T", phase="stress", run_id="r1", req_id="q0")]
        p = _write_llm_jsonl(tmp_path, recs)
        with pytest.raises(ValueError, match="No SystemRecord"):
            thermal_vs_time(p)


# ── generate_all ────────────────────────────────────────────────────


class TestGenerateAll:
    def test_generates_three_pngs(self, tmp_path):
        llm = [
            LLMRecord(
                ts="T",
                phase="stress",
                run_id="r1",
                req_id="q0",
                concurrency=1,
                decode_tok_per_s=50.0,
            )
        ]
        swarm = [
            SwarmRecord(ts="T", phase="swarm", run_id="r1", generation=0, pass_at_1=0.5)
        ]
        sys = [SystemRecord(ts="T", phase="stress", run_id="r1")]
        jsonl = tmp_path / "mixed.jsonl"
        to_jsonl(llm + swarm + sys, jsonl)

        out_dir = tmp_path / "charts"
        results = generate_all(jsonl, out_dir)
        assert len(results) == 3
        for r in results:
            assert r.exists()
        assert (out_dir / "throughput_vs_concurrency.png").exists()
        assert (out_dir / "pass_at_1_vs_generation.png").exists()
        assert (out_dir / "thermal_vs_time.png").exists()

    def test_default_output_dir(self, tmp_path):
        llm = [
            LLMRecord(
                ts="T",
                phase="stress",
                run_id="r1",
                req_id="q0",
                concurrency=1,
                decode_tok_per_s=50.0,
            )
        ]
        swarm = [
            SwarmRecord(ts="T", phase="swarm", run_id="r1", generation=0, pass_at_1=0.5)
        ]
        sys = [SystemRecord(ts="T", phase="stress", run_id="r1")]
        jsonl = tmp_path / "mixed.jsonl"
        to_jsonl(llm + swarm + sys, jsonl)

        results = generate_all(jsonl)
        assert len(results) == 3
        for r in results:
            assert r.exists()
            assert r.parent == tmp_path


# ── real JSONL smoke test ───────────────────────────────────────────


class TestRealJSONL:
    def test_thermal_vs_time_real_baseline(self):
        jsonl = Path("logs/phase1_baseline/baseline-c276c6b6/baseline.jsonl")
        if not jsonl.exists():
            pytest.skip("baseline JSONL not available")
        out = thermal_vs_time(jsonl)
        assert out.exists()
