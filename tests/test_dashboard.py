from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from rich.console import Console

from redline.swarm.dashboard import (
    Dashboard,
    HardwareState,
    RunState,
    TickerEntry,
    TaskState,
    _build_layout,
    _hardware_gauges,
    _leaderboard,
    _population_grid,
    _ticker_panel,
    _thermal_label,
    make_fake_state,
)


# ── State factory ─────────────────────────────────────────────────────


class TestMakeFakeState:
    def test_default_tasks(self):
        state = make_fake_state()
        assert len(state.tasks) == 6
        assert state.dataset == "humaneval_plus"
        assert state.generation == 3

    def test_custom_num_tasks(self):
        state = make_fake_state(num_tasks=10)
        assert len(state.tasks) == 10

    def test_solved_frac(self):
        state = make_fake_state(solved_frac=1.0, num_tasks=4)
        assert all(t.solved for t in state.tasks.values())

    def test_hardware_defaults(self):
        state = make_fake_state()
        hw = state.hardware
        assert hw.mem_used_gb == 52.0
        assert hw.gpu_pct == 94.0
        assert hw.power_watts == 38.0

    def test_ticker_entries(self):
        state = make_fake_state()
        assert len(state.ticker) >= 1


# ── Panel renderers ───────────────────────────────────────────────────


class TestPopulationGrid:
    def test_rows_match_tasks(self):
        tasks = {
            f"task/{i}": TaskState(task_id=f"task/{i}", solved=i % 2 == 0)
            for i in range(4)
        }
        table = _population_grid(tasks)
        assert len(table.rows) == 4

    def test_solved_shows_check(self):
        tasks = {"t/0": TaskState(task_id="t/0", solved=True)}
        from io import StringIO

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=80)
        table = _population_grid(tasks)
        c.print(table)
        assert "✓" in buf.getvalue()

    def test_unsolved_shows_cross(self):
        tasks = {"t/0": TaskState(task_id="t/0", solved=False)}
        from io import StringIO

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=80)
        table = _population_grid(tasks)
        c.print(table)
        assert "✗" in buf.getvalue()


class TestLeaderboard:
    def test_solved_ranked_first(self):
        tasks = {
            "a": TaskState(task_id="a", solved=True),
            "b": TaskState(task_id="b", solved=False),
        }
        from io import StringIO

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=80)
        table = _leaderboard(tasks)
        c.print(table)
        output = buf.getvalue()
        a_pos = output.index("a")
        b_pos = output.index("b")
        assert a_pos < b_pos

    def test_pass_at_1_values(self):
        tasks = {
            "x": TaskState(task_id="x", solved=True),
            "y": TaskState(task_id="y", solved=False),
        }
        from io import StringIO

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=80)
        table = _leaderboard(tasks)
        c.print(table)
        assert "1.0" in buf.getvalue()
        assert "0.0" in buf.getvalue()


class TestTickerPanel:
    def test_shows_entries(self):
        entries = [TickerEntry(label="gen 2", value="800 tok/s")]
        panel = _ticker_panel(entries)
        title_str = str(panel.title or "")
        assert "Token-Stream Ticker" in title_str

    def test_empty_ticker(self):
        panel = _ticker_panel([])
        assert "No activity" in str(panel.renderable)


class TestHardwareGauges:
    def test_mem_displayed(self):
        hw = HardwareState(mem_used_gb=32.0, mem_total_gb=64.0)
        panel = _hardware_gauges(hw)
        assert "32" in str(panel.renderable)

    def test_gpu_displayed(self):
        hw = HardwareState(gpu_pct=85.0)
        panel = _hardware_gauges(hw)
        assert "85" in str(panel.renderable)

    def test_power_displayed(self):
        hw = HardwareState(power_watts=42.0)
        panel = _hardware_gauges(hw)
        assert "42" in str(panel.renderable)


class TestThermalLabel:
    def test_known_levels(self):
        assert _thermal_label(0) == "none"
        assert _thermal_label(1) == "light"
        assert _thermal_label(2) == "moderate"
        assert _thermal_label(3) == "serious"
        assert _thermal_label(4) == "critical"

    def test_unknown_level(self):
        label = _thermal_label(99)
        assert "unknown" in label


# ── Layout composition ────────────────────────────────────────────────


class TestBuildLayout:
    def test_returns_layout(self):
        state = make_fake_state()
        layout = _build_layout(state)
        assert layout is not None

    def test_no_crash_empty_tasks(self):
        state = RunState(tasks={})
        layout = _build_layout(state)
        assert layout is not None


# ── Dashboard render rate (≥ 1 Hz gate) ───────────────────────────────


class TestDashboardRenderRate:
    """Core acceptance: dashboard renders ≥ 1 Hz from static fake-state."""

    def test_render_once_produces_output(self):
        state = make_fake_state()
        dash = Dashboard()
        output = dash.render_once(state)
        assert len(output) > 0
        assert dash.render_count == 1

    def test_ge_1_hz_static_state(self):
        """Render a static fake-state for 2 s and verify ≥ 1 Hz."""

        state = make_fake_state()
        dash = Dashboard(state_fn=lambda: state)

        start = time.monotonic()
        duration_s = 2.0

        with patch("rich.console.Console.update_screen"):
            with patch.object(dash._console, "print"):
                end = start + duration_s
                while time.monotonic() < end:
                    t0 = time.monotonic()
                    dash._tick(state)
                    sleep_time = 0.5 - (time.monotonic() - t0)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        elapsed = time.monotonic() - start
        frames = dash.render_count
        hz = frames / elapsed if elapsed > 0 else 0

        assert hz >= 1.0, (
            f"Render rate {hz:.2f} Hz < 1 Hz ({frames} frames in {elapsed:.2f}s)"
        )
        assert frames >= 4, f"Expected ≥ 4 frames in {duration_s}s, got {frames}"

    def test_no_dropped_frames(self):
        """All scheduled ticks produce a frame — no drops."""

        state = make_fake_state()
        dash = Dashboard(state_fn=lambda: state)

        with patch.object(dash._console, "print"):
            for _ in range(10):
                t0 = time.monotonic()
                dash._tick(state)
                frame_time = time.monotonic() - t0
                assert frame_time < 1.0, f"Frame took {frame_time:.3f}s — possible drop"

        assert dash.render_count == 10

    def test_run_with_fake_state(self):
        """Full run loop with fake state for a short duration."""

        state = make_fake_state()
        dash = Dashboard(state_fn=lambda: state)
        mock_live = MagicMock()

        with patch.object(dash._console, "print"):
            with patch("rich.live.Live", return_value=mock_live):
                dash.run(duration_s=0.5, interval_s=0.2)

        assert dash.render_count >= 1
        hz = dash.render_count / 0.5 if dash.render_count > 0 else 0
        assert hz >= 1.0, f"Run rate {hz:.1f} Hz < 1 Hz"

    def test_render_once_contains_expected_panels(self):
        """Single render contains all four panel titles."""

        state = make_fake_state()
        dash = Dashboard()
        output = dash.render_once(state)

        assert "Population Grid" in output or "population" in output.lower()
        assert "Leaderboard" in output or "leaderboard" in output.lower()
        assert "Ticker" in output or "ticker" in output.lower()
        assert "Gauges" in output or "gauges" in output.lower()
