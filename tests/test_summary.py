from __future__ import annotations

import pytest

from redline.report.summary import (
    Citation,
    HeadlineValue,
    _bound_type,
    _knee_concurrency,
    _swarm_headline,
    build_headline,
)
from redline.telemetry.metrics import (
    LLMRecord,
    SwarmRecord,
    SystemRecord,
)


# ── Citation / HeadlineValue ────────────────────────────────────────


def test_citation_fields():
    c = Citation(run_id="r1", field="LLMRecord.concurrency")
    assert c.run_id == "r1"
    assert c.field == "LLMRecord.concurrency"


def test_headline_value_int():
    hv = HeadlineValue(value=42, citation=Citation("r1", "f"))
    assert hv.value == 42


def test_headline_value_float():
    hv = HeadlineValue(value=3.14, citation=Citation("r1", "f"))
    assert isinstance(hv.value, float) and abs(hv.value - 3.14) < 0.01


# ── _knee_concurrency ─────────────────────────────────────────────


def test_knee_basic():
    """Knee at concurrency where tps drops below 90% of peak."""
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
                decode_tok_per_s=75.0,
            )
            for i in range(6, 9)
        ]
        + [
            LLMRecord(
                ts="T",
                phase="stress",
                run_id="r1",
                req_id=f"q{i}",
                concurrency=8,
                decode_tok_per_s=30.0,
            )
            for i in range(9, 12)
        ]
    )
    result = _knee_concurrency(recs)
    assert result is not None
    k, x = result
    # Peak mean = 80 at c=2; 90% threshold = 72. c=4 has 75 >= 72, so knee is c=4.
    assert k == 4
    assert abs(x - 75.0) < 0.1


def test_knee_single_level():
    """Fewer than 2 levels returns None."""
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
    result = _knee_concurrency(recs)
    assert result is None


def test_knee_all_above_threshold():
    """If all levels are above 90% of peak, knee is the highest level."""
    recs = [
        LLMRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            req_id=f"q{i}",
            concurrency=1,
            decode_tok_per_s=50.0,
        )
        for i in range(3)
    ] + [
        LLMRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            req_id=f"q{i}",
            concurrency=2,
            decode_tok_per_s=48.0,
        )
        for i in range(3, 6)
    ]
    result = _knee_concurrency(recs)
    assert result is not None
    k, x = result
    # Peak = 50; threshold = 45. Both levels above, so knee = c=2.
    assert k == 2


# ── _bound_type ───────────────────────────────────────────────────


def test_bound_compute():
    recs = [
        SystemRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            gpu_active_residency_pct=97.0,
            thermal_pressure_level=1,
            unified_mem_used_gb=40.0,
        )
    ]
    assert _bound_type(recs) == "compute"


def test_bound_memory():
    recs = [
        SystemRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            gpu_active_residency_pct=80.0,
            thermal_pressure_level=2,
            unified_mem_used_gb=60.0,
        )
    ]
    assert _bound_type(recs) == "memory"


def test_bound_thermal():
    recs = [
        SystemRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            gpu_active_residency_pct=80.0,
            thermal_pressure_level=4,
            unified_mem_used_gb=30.0,
        )
    ]
    assert _bound_type(recs) == "thermal"


def test_bound_none():
    assert _bound_type([]) is None


# ── _swarm_headline ───────────────────────────────────────────────


def test_swarm_values():
    recs = [
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r1",
            dataset="humaneval_plus",
            generation=0,
            pass_at_1=0.45,
            total_solved=74,
            tokens_per_solved_task=1200.0,
        ),
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r1",
            dataset="humaneval_plus",
            generation=3,
            pass_at_1=0.62,
            total_solved=102,
            tokens_per_solved_task=980.5,
        ),
    ]
    result = _swarm_headline(recs)
    assert result is not None
    y_val, z_val, n_val, delta_val, t_val = result

    assert y_val.value == 45.0
    assert y_val.citation.run_id == "r1"
    assert "pass_at_1" in y_val.citation.field and "gen=0" in y_val.citation.field

    assert z_val.value == 62.0
    assert n_val.value == 3
    assert delta_val.value == 28  # 102 - 74
    assert isinstance(t_val.value, float) and abs(t_val.value - 980.5) < 0.1


def test_swarm_empty():
    assert _swarm_headline([]) is None


# ── build_headline ────────────────────────────────────────────────


def test_full_headline():
    llm = [
        LLMRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            req_id=f"q{i}",
            concurrency=1,
            decode_tok_per_s=50.0,
        )
        for i in range(3)
    ] + [
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
    sys = [
        SystemRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            gpu_active_residency_pct=97.0,
            thermal_pressure_level=1,
            unified_mem_used_gb=40.0,
        )
    ]
    swarm = [
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r1",
            generation=0,
            pass_at_1=0.45,
            total_solved=74,
            tokens_per_solved_task=1200.0,
        ),
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r1",
            generation=3,
            pass_at_1=0.62,
            total_solved=102,
            tokens_per_solved_task=980.5,
        ),
    ]

    headline, value_map = build_headline(llm, sys, swarm)

    assert "sustains" in headline
    assert "tok/s" in headline
    assert "compute-bound knee" in headline
    assert "pass@1" in headline
    assert "45.0%" in headline
    assert "62.0%" in headline
    assert "3 generations" in headline
    assert "28 additional tasks" in headline

    # Every value has a citation with run_id and field
    for key, hv in value_map.items():
        assert isinstance(hv.citation.run_id, str) and len(hv.citation.run_id) > 0
        assert isinstance(hv.citation.field, str) and len(hv.citation.field) > 0


def test_headline_llm_only():
    """Only LLM records produces partial headline."""
    llm = [
        LLMRecord(
            ts="T",
            phase="stress",
            run_id="r1",
            req_id=f"q{i}",
            concurrency=1,
            decode_tok_per_s=50.0,
        )
        for i in range(3)
    ] + [
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

    headline, value_map = build_headline(llm)

    assert "sustains" in headline
    assert "pass@1" not in headline


def test_headline_swarm_only():
    """Only swarm records produces partial headline."""
    swarm = [
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r1",
            generation=0,
            pass_at_1=0.45,
            total_solved=74,
            tokens_per_solved_task=1200.0,
        ),
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r1",
            generation=3,
            pass_at_1=0.62,
            total_solved=102,
            tokens_per_solved_task=980.5,
        ),
    ]

    headline, value_map = build_headline([], swarm_records=swarm)

    assert "pass@1" in headline
    assert "sustains" not in headline


def test_citations_traceable():
    """Every value in the map traces back to a (run_id, field)."""
    llm = [
        LLMRecord(
            ts="T",
            phase="stress",
            run_id="r-stress-42",
            req_id=f"q{i}",
            concurrency=1,
            decode_tok_per_s=50.0,
        )
        for i in range(3)
    ] + [
        LLMRecord(
            ts="T",
            phase="stress",
            run_id="r-stress-42",
            req_id=f"q{i}",
            concurrency=2,
            decode_tok_per_s=80.0,
        )
        for i in range(3, 6)
    ]
    sys = [
        SystemRecord(
            ts="T",
            phase="stress",
            run_id="r-stress-42",
            gpu_active_residency_pct=97.0,
            thermal_pressure_level=1,
            unified_mem_used_gb=40.0,
        )
    ]
    swarm = [
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r-swarm-7",
            generation=0,
            pass_at_1=0.5,
            total_solved=82,
            tokens_per_solved_task=1100.0,
        ),
        SwarmRecord(
            ts="T",
            phase="swarm",
            run_id="r-swarm-7",
            generation=5,
            pass_at_1=0.68,
            total_solved=112,
            tokens_per_solved_task=950.0,
        ),
    ]

    _, value_map = build_headline(llm, sys, swarm)

    # LLM-derived values cite r-stress-42
    assert value_map["K"].citation.run_id == "r-stress-42"
    assert value_map["X"].citation.run_id == "r-stress-42"

    # System-derived value cites r-stress-42
    assert value_map["bound"].citation.run_id == "r-stress-42"

    # Swarm-derived values cite r-swarm-7
    assert value_map["Y"].citation.run_id == "r-swarm-7"
    assert value_map["Z"].citation.run_id == "r-swarm-7"
    assert value_map["N"].citation.run_id == "r-swarm-7"
    assert value_map["delta"].citation.run_id == "r-swarm-7"
    assert value_map["T"].citation.run_id == "r-swarm-7"
