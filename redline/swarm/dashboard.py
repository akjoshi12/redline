from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── State snapshot (decoupled from orchestration) ───────────────────────


@dataclass
class TaskState:
    """Per-task state in the swarm run."""

    task_id: str
    generation: int = 0
    solved: bool = False
    best_fitness: int = 0
    population_size: int = 0


@dataclass
class TickerEntry:
    """A single line in the token-stream ticker."""

    label: str
    value: str


@dataclass
class HardwareState:
    """Hardware gauge snapshot."""

    mem_used_gb: float = 0.0
    mem_total_gb: float = 64.0
    gpu_pct: float = 0.0
    thermal_level: int = 0
    power_watts: float = 0.0


@dataclass
class RunState:
    """Complete snapshot of a swarm run at one point in time."""

    dataset: str = "humaneval_plus"
    generation: int = 0
    tasks: Dict[str, TaskState] = field(default_factory=dict)
    ticker: List[TickerEntry] = field(default_factory=list)
    hardware: HardwareState = field(default_factory=HardwareState)


# ── Panel renderers ────────────────────────────────────────────────────


def _population_grid(tasks: Dict[str, TaskState]) -> Table:
    """Render the population grid panel."""

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Task ID", style="white")
    table.add_column("Gen", justify="right", width=4)
    table.add_column("Status", width=6)

    for task_id in sorted(tasks.keys()):
        ts = tasks[task_id]
        status = "✓" if ts.solved else "✗"
        style = "green" if ts.solved else "red"
        table.add_row(
            task_id,
            str(ts.generation),
            Text(status, style=style),
        )

    return table


def _leaderboard(tasks: Dict[str, TaskState]) -> Table:
    """Render the task leaderboard sorted by pass rate."""

    solved = [t for t in tasks.values() if t.solved]
    unsolved = [t for t in tasks.values() if not t.solved]

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Rank", justify="right", width=4)
    table.add_column("Task ID", style="white")
    table.add_column("pass@1", justify="right", width=6)

    rank = 1
    for ts in solved:
        table.add_row(str(rank), ts.task_id, "1.0")
        rank += 1

    for ts in unsolved:
        table.add_row(str(rank), ts.task_id, "0.0")
        rank += 1

    return table


def _ticker_panel(ticker: List[TickerEntry]) -> Panel:
    """Render the token-stream ticker panel."""

    lines = []
    for entry in ticker:
        lines.append(f"◉ {entry.label}: {entry.value}")

    content = "\n".join(lines) if lines else "No activity"
    return Panel(content, title="Token-Stream Ticker", border_style="blue")


def _thermal_label(level: int) -> str:
    """Map thermal pressure level to human-readable label."""

    labels = {0: "none", 1: "light", 2: "moderate", 3: "serious", 4: "critical"}
    return labels.get(level, f"unknown({level})")


def _hardware_gauges(hw: HardwareState) -> Panel:
    """Render the hardware gauges panel."""

    mem_pct = hw.mem_used_gb / hw.mem_total_gb if hw.mem_total_gb > 0 else 0.0
    gpu_pct = min(1.0, max(0.0, hw.gpu_pct / 100.0))
    therm_max = 4.0
    therm_pct = min(1.0, max(0.0, hw.thermal_level / therm_max))

    mem_bar = "█" * int(mem_pct * 6) + "░" * (6 - int(mem_pct * 6))
    gpu_bar = "█" * int(gpu_pct * 6) + "░" * (6 - int(gpu_pct * 6))
    therm_bar = "█" * int(therm_pct * 6) + "░" * (6 - int(therm_pct * 6))

    content = (
        f"MEM  {mem_bar} {hw.mem_used_gb:.0f}/{hw.mem_total_gb:.0f}GB\n"
        f"GPU  {gpu_bar} {hw.gpu_pct:.0f}%\n"
        f"THERM {therm_bar} {_thermal_label(hw.thermal_level)}\n"
        f"PWR  {hw.power_watts:.0f}W"
    )

    return Panel(content, title="Hardware Gauges", border_style="yellow")


# ── Dashboard composition ──────────────────────────────────────────────


def _build_layout(state: RunState):
    """Compose the full dashboard layout from a state snapshot."""

    from rich.layout import Layout

    grid = _population_grid(state.tasks)
    lb = _leaderboard(state.tasks)

    left_top = Panel(grid, title="Population Grid", border_style="green")
    right_top = Panel(lb, title="Task Leaderboard", border_style="magenta")

    ticker = _ticker_panel(state.ticker)
    gauges = _hardware_gauges(state.hardware)

    layout = Layout()
    layout.split_column(
        Layout(name="top"),
        Layout(name="bottom"),
    )
    layout["top"].split_row(
        Layout(left_top, name="grid"),
        Layout(right_top, name="leaderboard"),
    )
    layout["bottom"].split_row(
        Layout(ticker, name="ticker"),
        Layout(gauges, name="gauges"),
    )

    return layout


# ── Live dashboard runner ──────────────────────────────────────────────


class Dashboard:
    """Rich TUI dashboard that renders RunState snapshots at ≥ 1 Hz.

    Fully decoupled from orchestration — accepts any callable that returns
    a RunState snapshot on each tick.
    """

    def __init__(self, state_fn=None):
        self._state_fn = state_fn
        self._console = Console()
        self._render_count = 0
        self._frame_times: List[float] = []

    @property
    def render_count(self) -> int:
        return self._render_count

    @property
    def frame_times(self) -> List[float]:
        return list(self._frame_times)

    def _tick(self, state: RunState):
        """Render a single frame from the given state."""

        t0 = time.monotonic()
        layout = _build_layout(state)
        self._console.print(layout, soft_wrap=True)
        elapsed = time.monotonic() - t0
        self._render_count += 1
        self._frame_times.append(elapsed)

    def run(self, duration_s: float = 2.0, interval_s: float = 0.5):
        """Run the dashboard for *duration_s* seconds at ~1 Hz.

        Uses Live to keep the display stable without flicker.
        """

        if self._state_fn is None:
            raise ValueError("No state function provided")

        end = time.monotonic() + duration_s

        with Live(
            _build_layout(self._state_fn()),
            console=self._console,
            refresh_per_second=10,
            screen=True,
        ) as live:
            while time.monotonic() < end:
                t0 = time.monotonic()
                state = self._state_fn()
                layout = _build_layout(state)
                live.update(layout)
                self._render_count += 1
                elapsed = time.monotonic() - t0
                self._frame_times.append(elapsed)

                sleep_time = interval_s - (time.monotonic() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def render_once(self, state: RunState) -> str:
        """Render a single frame and return the console output as string."""

        from io import StringIO

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=80)
        layout = _build_layout(state)
        c.print(layout, soft_wrap=True)
        self._render_count += 1
        return buf.getvalue()


def make_fake_state(
    num_tasks: int = 6,
    solved_frac: float = 0.5,
    generation: int = 3,
) -> RunState:
    """Create a deterministic fake RunState for testing."""

    import random

    rng = random.Random(42)
    tasks: Dict[str, TaskState] = {}

    for i in range(num_tasks):
        task_id = f"humaneval/{i:03d}"
        solved = rng.random() < solved_frac
        tasks[task_id] = TaskState(
            task_id=task_id,
            generation=rng.randint(0, generation),
            solved=solved,
            best_fitness=1 if solved else rng.randint(0, 4),
            population_size=rng.randint(2, 8),
        )

    ticker = [
        TickerEntry(label=f"gen {generation}", value="1247 tok/s"),
        TickerEntry(label="repair", value="+2 solved"),
    ]

    hw = HardwareState(
        mem_used_gb=52.0,
        gpu_pct=94.0,
        thermal_level=1,
        power_watts=38.0,
    )

    return RunState(
        dataset="humaneval_plus",
        generation=generation,
        tasks=tasks,
        ticker=ticker,
        hardware=hw,
    )
