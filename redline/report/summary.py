from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from redline.telemetry.metrics import (
    LLMRecord,
    SwarmRecord,
    SystemRecord,
)


@dataclass
class Citation:
    """Traceable reference to a logged value."""

    run_id: str
    field: str


@dataclass
class HeadlineValue:
    """A numeric value paired with its source citation."""

    value: float | int | str
    citation: Citation


def _knee_concurrency(
    llm_records: List[LLMRecord],
) -> Optional[Tuple[int, float]]:
    """Find the knee concurrency level and tok/s at that point.

    Knee = highest concurrency where mean decode_tok_per_s is still >= 90% of
    the peak mean across all levels.
    Returns (concurrency, mean_decode_tps) or None if fewer than 2 levels.
    """

    by_level: Dict[int, List[float]] = {}
    for r in llm_records:
        by_level.setdefault(r.concurrency, []).append(r.decode_tok_per_s)

    means = {c: statistics.mean(v) for c, v in sorted(by_level.items())}
    if len(means) < 2:
        return None

    peak = max(means.values())
    threshold = peak * 0.9

    # Iterate high-to-low: find the highest concurrency still above threshold.
    for c in sorted(means.keys(), reverse=True):
        if means[c] >= threshold:
            return (c, means[c])

    return None


def _bound_type(
    sys_records: List[SystemRecord],
) -> Optional[str]:
    """Classify the bottleneck from system metrics at the worst point.

    Rules (from plan.md):
      - Compute-bound: GPU residency >= 95% AND thermal pressure <= 1
      - Memory-bound: unified_mem_used_gb > 58 GB
      - Thermal-bound: thermal_pressure_level >= 3
    Returns the first matching label or None if no records.
    """

    if not sys_records:
        return None

    worst = max(sys_records, key=lambda r: r.gpu_active_residency_pct)

    if worst.gpu_active_residency_pct >= 95 and worst.thermal_pressure_level <= 1:
        return "compute"
    if worst.unified_mem_used_gb > 58:
        return "memory"
    if worst.thermal_pressure_level >= 3:
        return "thermal"

    return None


def _swarm_headline(
    swarm_records: List[SwarmRecord],
) -> Optional[
    Tuple[HeadlineValue, HeadlineValue, HeadlineValue, HeadlineValue, HeadlineValue]
]:
    """Extract pass@1 headline values from swarm records.

    Returns (Y%, Z%, N gens, delta tasks, tokens_per_solved_task) or None.
      Y%  = single-shot pass@1 (generation == 0)
      Z%  = final generation pass@1
      N   = max generation number
      Δ   = total_solved(final) - total_solved(gen 0)
      T   = tokens_per_solved_task at final gen
    """

    if not swarm_records:
        return None

    run_id = swarm_records[0].run_id

    gen0 = [r for r in swarm_records if r.generation == 0]
    if not gen0:
        return None

    y_val = HeadlineValue(
        value=round(gen0[0].pass_at_1 * 100, 1),
        citation=Citation(run_id=run_id, field="SwarmRecord.pass_at_1 (gen=0)"),
    )

    final_gen = max(swarm_records, key=lambda r: r.generation)
    z_val = HeadlineValue(
        value=round(final_gen.pass_at_1 * 100, 1),
        citation=Citation(run_id=run_id, field="SwarmRecord.pass_at_1 (gen=max)"),
    )

    n_val = HeadlineValue(
        value=final_gen.generation,
        citation=Citation(run_id=run_id, field="SwarmRecord.generation (max)"),
    )

    delta = final_gen.total_solved - gen0[0].total_solved
    delta_val = HeadlineValue(
        value=delta,
        citation=Citation(
            run_id=run_id,
            field="SwarmRecord.total_solved (gen=max) - SwarmRecord.total_solved (gen=0)",
        ),
    )

    t_val = HeadlineValue(
        value=round(final_gen.tokens_per_solved_task, 1),
        citation=Citation(
            run_id=run_id, field="SwarmRecord.tokens_per_solved_task (gen=max)"
        ),
    )

    return (y_val, z_val, n_val, delta_val, t_val)


def build_headline(
    llm_records: List[LLMRecord],
    sys_records: Optional[List[SystemRecord]] = None,
    swarm_records: Optional[List[SwarmRecord]] = None,
) -> Tuple[str, Dict[str, HeadlineValue]]:
    """Fill the headline sentence template from logged values only.

    Template (from plan.md):
        "Qwopus 27B on M4 Max sustains K concurrent requests at X tok/s before
         hitting a {bound}-bound knee, and evolutionary repair improves HumanEval+
         pass@1 from Y% (single-shot) to Z% after N generations — solving Δ
         additional tasks at T tokens per solved task."

    Every value maps to a (run_id, field) citation returned in the dict.

    Args:
        llm_records: LLMRecord list for K and X extraction.
        sys_records: SystemRecord list for bound-type classification.
        swarm_records: SwarmRecord list for Y/Z/N/Δ/T extraction.

    Returns:
        (headline_string, value_map) where value_map keys are the placeholder
        names ("K", "X", etc.) and values are HeadlineValue objects with
        traceable citations.
    """

    if sys_records is None:
        sys_records = []
    if swarm_records is None:
        swarm_records = []

    value_map: Dict[str, HeadlineValue] = {}

    # ── K and X from LLM records ────────────────────────────────
    knee = _knee_concurrency(llm_records)
    run_id_llm = llm_records[0].run_id if llm_records else "unknown"

    if knee:
        k, x = knee
        value_map["K"] = HeadlineValue(
            value=k,
            citation=Citation(run_id=run_id_llm, field="LLMRecord.concurrency (knee)"),
        )
        value_map["X"] = HeadlineValue(
            value=round(x, 1),
            citation=Citation(
                run_id=run_id_llm, field="LLMRecord.decode_tok_per_s (at knee)"
            ),
        )

    # ── Bound type from system records ───────────────────────────
    bound = _bound_type(sys_records)
    run_id_sys = sys_records[0].run_id if sys_records else "unknown"

    if bound:
        value_map["bound"] = HeadlineValue(
            value=bound,
            citation=Citation(
                run_id=run_id_sys, field="SystemRecord (knee classification)"
            ),
        )

    # ── Swarm values ─────────────────────────────────────────────
    swarm_vals = _swarm_headline(swarm_records)
    if swarm_vals:
        y_val, z_val, n_val, delta_val, t_val = swarm_vals
        value_map["Y"] = y_val
        value_map["Z"] = z_val
        value_map["N"] = n_val
        value_map["delta"] = delta_val
        value_map["T"] = t_val

    # ── Compose headline ────────────────────────────────────────
    parts: List[str] = []

    if "K" in value_map and "X" in value_map:
        k_str = str(value_map["K"].value)
        x_str = str(value_map["X"].value)
        bound_str = value_map.get(
            "bound", HeadlineValue("unknown", Citation("?", "?"))
        ).value

        parts.append(
            f"Qwopus 27B on M4 Max sustains {k_str} concurrent requests at "
            f"{x_str} tok/s before hitting a {bound_str}-bound knee"
        )

    if "Y" in value_map:
        y_str = str(value_map["Y"].value)
        z_str = str(value_map["Z"].value)
        n_str = str(value_map["N"].value)
        delta_str = str(value_map["delta"].value)
        t_str = str(value_map["T"].value)

        parts.append(
            f"evolutionary repair improves HumanEval+ pass@1 from {y_str}% "
            f"(single-shot) to {z_str}% after {n_str} generations — solving "
            f"{delta_str} additional tasks at {t_str} tokens per solved task"
        )

    headline = ", and ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")

    return headline, value_map
