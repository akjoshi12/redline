from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from pathlib import Path
from typing import List, Optional  # noqa: F401

from redline.telemetry.metrics import (  # noqa: F401
    LLMRecord,
    SwarmRecord,
    SystemRecord,
    from_jsonl,
)


def throughput_vs_concurrency(
    jsonl_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Throughput (decode_tok_per_s) vs concurrency.

    X-axis: LLMRecord.concurrency
    Y-axis: LLMRecord.decode_tok_per_s
    Aggregates by concurrency level using mean + 95% CI band.
    """

    records = from_jsonl(jsonl_path)
    llm_records = [r for r in records if isinstance(r, LLMRecord)]

    if not llm_records:
        raise ValueError(f"No LLMRecord found in {jsonl_path}")

    df = pd.DataFrame(
        [
            {"concurrency": r.concurrency, "decode_tok_per_s": r.decode_tok_per_s}
            for r in llm_records
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="concurrency",
        y="decode_tok_per_s",
        data=df,
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Decode Tokens/s (mean)")
    ax.set_title("Throughput vs Concurrency")

    if output_path is None:
        output_path = Path(jsonl_path).parent / "throughput_vs_concurrency.png"
    else:
        output_path = Path(output_path)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def pass_at_1_vs_generation(
    jsonl_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Pass@1 vs generation.

    X-axis: SwarmRecord.generation
    Y-axis: SwarmRecord.pass_at_1
    """

    records = from_jsonl(jsonl_path)
    swarm_records = [r for r in records if isinstance(r, SwarmRecord)]

    if not swarm_records:
        raise ValueError(f"No SwarmRecord found in {jsonl_path}")

    df = pd.DataFrame(
        [{"generation": r.generation, "pass_at_1": r.pass_at_1} for r in swarm_records]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        x="generation",
        y="pass_at_1",
        data=df,
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Pass@1")
    ax.set_title("Pass@1 vs Generation")

    if output_path is None:
        output_path = Path(jsonl_path).parent / "pass_at_1_vs_generation.png"
    else:
        output_path = Path(output_path)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def thermal_vs_time(
    jsonl_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Thermal pressure level vs time.

    X-axis: SystemRecord.ts (timestamp)
    Y-axis: SystemRecord.thermal_pressure_level
    """

    records = from_jsonl(jsonl_path)
    sys_records = [r for r in records if isinstance(r, SystemRecord)]

    if not sys_records:
        raise ValueError(f"No SystemRecord found in {jsonl_path}")

    df = pd.DataFrame(
        [
            {"ts": r.ts, "thermal_pressure_level": r.thermal_pressure_level}
            for r in sys_records
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        x="ts",
        y="thermal_pressure_level",
        data=df,
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Thermal Pressure Level")
    ax.set_title("Thermal vs Time")

    if output_path is None:
        output_path = Path(jsonl_path).parent / "thermal_vs_time.png"
    else:
        output_path = Path(output_path)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def generate_all(
    jsonl_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> List[Path]:
    """Generate all three charts from a single JSONL file.

    Returns the list of output PNG paths.
    """

    if output_dir is None:
        output_dir = Path(jsonl_path).parent
    else:
        output_dir = Path(output_dir)

    outputs = []
    outputs.append(
        throughput_vs_concurrency(
            jsonl_path, output_dir / "throughput_vs_concurrency.png"
        )
    )
    outputs.append(
        pass_at_1_vs_generation(jsonl_path, output_dir / "pass_at_1_vs_generation.png")
    )
    outputs.append(thermal_vs_time(jsonl_path, output_dir / "thermal_vs_time.png"))
    return outputs
