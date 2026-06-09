"""Knee detection for throughput-vs-load curves."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class Knee:
    """A detected inflection point in a throughput curve."""

    index: int  # position in the x-axis array where knee occurs
    x_value: float  # load value at knee (e.g., concurrency level)
    y_value: float  # throughput value at knee (e.g., tok/s)
    slope_before: float  # average slope of segment before knee
    slope_after: float  # average slope of segment after knee
    slope_change_pct: float  # percentage drop in slope (>50% triggers detection)
    bound_type: str  # "compute", "memory", or "thermal"
    supporting_metrics: dict[str, float] = dataclasses.field(
        default_factory=dict
    )  # ≥2 cited metric fields that justify the bound type


@dataclasses.dataclass(frozen=True)
class PiecewiseFit:
    """Least-squares line fit for one segment of a piecewise curve."""

    intercept: float
    slope: float
    r_squared: float = 0.0


def _fit_line(x: np.ndarray, y: np.ndarray) -> PiecewiseFit:
    """Ordinary least-squares fit with R² computation."""

    if len(x) < 2:
        return PiecewiseFit(intercept=float(y[0]) if len(y) else 0.0, slope=0.0)

    coeffs = np.polyfit(x.astype(float), y.astype(float), 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # R²
    y_pred = np.polyval(coeffs, x.astype(float))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return PiecewiseFit(intercept=intercept, slope=slope, r_squared=r_sq)


def _compute_slopes(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Return slopes between consecutive (x, y) pairs."""

    return [float((y[i + 1] - y[i]) / (x[i + 1] - x[i])) for i in range(len(x) - 1)]


def _classify_bound(
    system_metrics: list[dict],
    knee_idx: int,
    baseline_throughput: float,
) -> tuple[str, dict[str, float]]:
    """Classify the bound type of a knee using correlated system metrics.

    Rules (from plan.md):
      - Compute-bound: GPU residency ≥ 95% AND thermal pressure ≤ 1
      - Memory-bound: unified_mem_used_gb > 58 GB OR OOM spike
      - Thermal-bound: thermal_pressure_level ≥ 3 AND throughput drop > 20%

    Falls back to "unknown" if no metrics or criteria don't match.

    Returns a tuple of (bound_type, supporting_metrics) where supporting_metrics
    is a dict with ≥2 cited metric fields that justify the classification.
    """

    if not system_metrics:
        return ("unknown", {})

    # Use the metric closest to the knee index (or last available)
    m = system_metrics[min(knee_idx, len(system_metrics) - 1)]

    gpu_residency = float(m.get("gpu_active_residency_pct", 0))
    mem_used = float(m.get("unified_mem_used_gb", 0))
    thermal = int(m.get("thermal_pressure_level", 0))
    oom_count = int(m.get("oom_count", 0))

    throughput_drop_pct = (
        (baseline_throughput - m.get("decode_tok_per_s", baseline_throughput))
        / baseline_throughput
        * 100
        if baseline_throughput > 0
        else 0.0
    )

    # Compute-bound: GPU saturated, no thermal pressure
    if gpu_residency >= 95 and thermal <= 1:
        return (
            "compute",
            {
                "gpu_active_residency_pct": gpu_residency,
                "thermal_pressure_level": float(thermal),
            },
        )

    # Memory-bound: near memory limit or OOMs
    if mem_used > 58 or oom_count > 0:
        return (
            "memory",
            {
                "unified_mem_used_gb": mem_used,
                "oom_count": float(oom_count),
            },
        )

    # Thermal-bound: high thermal pressure + throughput degradation
    if thermal >= 3 and throughput_drop_pct > 20:
        return (
            "thermal",
            {
                "thermal_pressure_level": float(thermal),
                "throughput_drop_pct": throughput_drop_pct,
            },
        )

    return ("unknown", {})


def detect_knees(
    x: np.ndarray,
    y: np.ndarray,
    system_metrics: Optional[list[dict]] = None,
    slope_change_threshold: float = 50.0,
    min_consecutive: int = 2,
) -> list[Knee]:
    """Detect knees in a throughput-vs-load curve.

    A knee is an inflection point where the slope drops by more than
    *slope_change_threshold* percent compared to the preceding segment,
    and at least *min_consecutive* subsequent points maintain the new (lower)
    slope pattern.

    Parameters
    ----------
    x : array-like of load values (e.g., concurrency levels).
    y : array-like of throughput values (e.g., tok/s).
    system_metrics : optional list of dicts with keys matching SystemRecord fields,
                     one per data point, used for bound-type classification.
    slope_change_threshold : percentage drop in slope that qualifies as a knee.
    min_consecutive : minimum number of consecutive points at the new slope
                      required to confirm a knee (avoids false positives).

    Returns
    -------
    List of Knee objects sorted by index.  Empty list if no knees detected.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < min_consecutive + 3:
        return []

    slopes = _compute_slopes(x, y)
    baseline_throughput = float(y[0])

    knees: list[Knee] = []

    for i in range(len(slopes)):
        prev_slope = slopes[i]

        # Skip if previous slope is near-zero (flat region — no meaningful knee)
        if abs(prev_slope) < 1e-9:
            continue

        curr_slope = slopes[i + 1] if i + 1 < len(slopes) else prev_slope

        # Check for significant slope drop (> threshold %)
        if prev_slope > 0 and curr_slope <= 0:
            change_pct = float("inf")
        elif prev_slope > 0:
            change_pct = (prev_slope - curr_slope) / abs(prev_slope) * 100
        else:
            continue

        if change_pct < slope_change_threshold:
            continue

        # Require ≥ min_consecutive subsequent points at the new (lower) slope.
        # We check that slopes[i+2], slopes[i+3], … are all ≤ curr_slope + tolerance,
        # confirming a sustained lower-slope regime.
        consecutive_ok = True
        needed = min(min_consecutive, len(slopes) - i - 2)

        if needed < 1:
            continue

        for j in range(needed):
            idx = i + 2 + j
            if idx >= len(slopes):
                consecutive_ok = False
                break
            # The new slope should be consistently lower than the old one.
            if slopes[idx] > prev_slope * (1 - slope_change_threshold / 200):
                consecutive_ok = False
                break

        if not consecutive_ok:
            continue

        knee_idx = i + 1  # index in x/y arrays where knee occurs

        # Fit segments for accurate slope characterization
        before_fit = _fit_line(x[:knee_idx], y[:knee_idx])
        after_fit = _fit_line(x[knee_idx:], y[knee_idx:])

        bound_type = "unknown"
        supporting_metrics: dict[str, float] = {}
        if system_metrics:
            bound_type, supporting_metrics = _classify_bound(
                system_metrics, knee_idx, baseline_throughput
            )

        knees.append(
            Knee(
                index=knee_idx,
                x_value=float(x[knee_idx]),
                y_value=float(y[knee_idx]),
                slope_before=before_fit.slope,
                slope_after=after_fit.slope,
                slope_change_pct=min(change_pct, 99.0),
                bound_type=bound_type,
                supporting_metrics=supporting_metrics,
            )
        )

    return knees


def analyze_sweep_log(
    log_dir: str | Path,
    phase: str = "stress_concurrency",
) -> list[Knee]:
    """Read sweep JSONL logs and detect labeled knees with supporting metrics.

    Reads the *phase*.jsonl file from each run directory under *log_dir*,
    extracts throughput-vs-load curves (grouped by concurrency or context length),
    collects correlated system metrics, and returns detected knees with ≥2
    cited metric fields per knee.

    Parameters
    ----------
    log_dir : path to the parent directory containing run subdirectories.
    phase : the phase prefix for the JSONL filename (default "stress_concurrency").

    Returns
    -------
    List of Knee objects, each with bound_type and supporting_metrics populated.
    """

    log_path = Path(log_dir)
    knees: list[Knee] = []

    # Find all matching JSONL files across run directories
    jsonl_files = sorted(log_path.rglob(f"{phase}.jsonl"))

    for jsonl_file in jsonl_files:
        llm_records: list[dict] = []
        system_records: list[dict] = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "req_id" in d:
                    llm_records.append(d)
                elif "interval_s" in d:
                    system_records.append(d)

        # Build x/y arrays grouped by the load dimension.
        # For concurrency sweeps, group by concurrency level.
        # For context sweeps, group by context_length.
        if phase == "stress_concurrency":
            x_vals, y_vals = _extract_concurrency_curve(llm_records)
        elif phase == "stress_context":
            x_vals, y_vals = _extract_context_curve(llm_records)
        else:
            # Default: try concurrency first, then context_length
            x_vals, y_vals = _extract_concurrency_curve(llm_records)

        if len(x_vals) < 3:
            continue

        knees.extend(
            detect_knees(
                np.array(x_vals),
                np.array(y_vals),
                system_metrics=system_records,
            )
        )

    return knees


def _extract_concurrency_curve(
    llm_records: list[dict],
) -> tuple[list[float], list[float]]:
    """Extract concurrency vs avg decode_tok/s curve from LLM records.

    Groups records by concurrency level and computes the mean decode_tok_per_s
    for each level.
    """

    # Group by concurrency level
    groups: dict[int, list[float]] = {}
    for r in llm_records:
        conc = int(r.get("concurrency", 1))
        tps = float(r.get("decode_tok_per_s", 0))
        groups.setdefault(conc, []).append(tps)

    if not groups:
        return [], []

    x_vals = sorted(groups.keys())
    y_vals = [sum(v) / len(v) for v in (groups[x] for x in x_vals)]

    return list(map(float, x_vals)), y_vals


def _extract_context_curve(llm_records: list[dict]) -> tuple[list[float], list[float]]:
    """Extract context_length vs decode_tok/s curve from LLM records.

    Groups records by context_length and computes the mean decode_tok_per_s
    for each length.
    """

    # Group by context_length
    groups: dict[int, list[float]] = {}
    for r in llm_records:
        ctx_len = int(r.get("context_length", 0))
        tps = float(r.get("decode_tok_per_s", 0))
        if ctx_len > 0:
            groups.setdefault(ctx_len, []).append(tps)

    if not groups:
        return [], []

    x_vals = sorted(groups.keys())
    y_vals = [sum(v) / len(v) for v in (groups[x] for x in x_vals)]

    return list(map(float, x_vals)), y_vals
