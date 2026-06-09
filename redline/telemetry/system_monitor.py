from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, TextIO

import psutil

from redline.telemetry.metrics import SystemRecord


class SystemMonitor:
    """Background thread that polls system metrics at a fixed interval.

    Writes SystemRecord instances to the given Logger on each tick.
    Gracefully degrades if powermetrics is unavailable (no GPU/thermal data).
    Falls back to vm_stat for memory if psutil fails.
    """

    def __init__(
        self,
        logger,
        phase: str = "baseline",
        run_id: str = "",
        interval_s: float = 2.0,
    ):
        self._logger = logger
        self._phase = phase
        self._run_id = run_id
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Cached powermetrics availability
        self._powermetrics_available = self._check_powermetrics()

    @staticmethod
    def _check_powermetrics() -> bool:
        """Return True if sudo powermetrics can be invoked."""
        try:
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "gpu_power", "-n1"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_psutil_memory(self) -> tuple[float, float]:
        """Return (used_gb, peak_gb) from psutil."""
        vm = psutil.virtual_memory()
        used_gb = vm.used / 1e9
        # psutil doesn't track peak directly; use current as best estimate
        return used_gb, used_gb

    @staticmethod
    def _get_vmstat_memory() -> tuple[float, float]:
        """Fallback: parse vm_stat for active memory (macOS only)."""
        try:
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5
            )
            pages_per_mb = 4096 / (1024 * 1024)
            total_pages = None
            active_pages = None
            for line in result.stdout.splitlines():
                if "Pages wired" in line:
                    # Approximate: wired + active ≈ used
                    pass
                elif "Pages active" in line:
                    val = int(line.strip().split(":")[1].strip().rstrip("."))
                    active_pages = val * pages_per_mb / 1024
            if active_pages is not None:
                return float(active_pages), float(active_pages)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0.0, 0.0

    def _get_powermetrics(self) -> tuple[float, int]:
        """Return (gpu_residency_pct, thermal_pressure_level).

        Returns (0.0, 0) if powermetrics is unavailable or fails.
        """
        if not self._powermetrics_available:
            return 0.0, 0

        try:
            result = subprocess.run(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "gpu_power,iopowernow",
                    "-n1",
                    "--interval",
                    "500",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            gpu_pct = 0.0
            thermal_level = 0

            output = result.stdout + result.stderr
            # Parse GPU residency if present
            for line in output.splitlines():
                lower = line.lower()
                if "gpu" in lower and ("residency" in lower or "active" in lower):
                    try:
                        parts = line.rsplit(":", 1)
                        val = float(parts[1].strip().rstrip("%"))
                        gpu_pct = min(100.0, max(0.0, val))
                    except (ValueError, IndexError):
                        pass

            # Parse thermal pressure if present
            for line in output.splitlines():
                lower = line.lower()
                if "thermal" in lower:
                    try:
                        parts = line.rsplit(":", 1)
                        val_str = parts[1].strip().lower()
                        level_map = {
                            "none": 0,
                            "nominal": 0,
                            "fair": 1,
                            "serious": 2,
                            "critical": 3,
                            "no-pressure": 0,
                        }
                        thermal_level = level_map.get(val_str, 0)
                    except (ValueError, IndexError):
                        pass

            return gpu_pct, thermal_level

        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._powermetrics_available = False
            return 0.0, 0

    def _get_cpu_freq(self) -> float:
        """Return CPU frequency in MHz."""
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current > 0:
                return freq.current
        except Exception:
            pass
        return 0.0

    def _get_power_watts(self) -> float:
        """Return power draw in watts from powermetrics, or 0."""
        if not self._powermetrics_available:
            return 0.0
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "energy_policy",
                    "-n1",
                    "--interval",
                    "500",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
            for line in output.splitlines():
                lower = line.lower()
                if "power" in lower and ("watt" in lower or "w " in lower):
                    try:
                        parts = line.rsplit(":", 1)
                        val_str = parts[1].strip().rstrip("W").strip()
                        return float(val_str)
                    except (ValueError, IndexError):
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._powermetrics_available = False
        return 0.0

    def _collect_record(self) -> SystemRecord:
        """Collect one snapshot of system metrics."""
        # Memory: psutil first, vm_stat fallback
        try:
            mem_used, mem_peak = self._get_psutil_memory()
        except Exception:
            mem_used, mem_peak = self._get_vmstat_memory()

        # GPU + thermal from powermetrics (graceful no-op)
        gpu_pct, thermal_level = self._get_powermetrics()

        # CPU freq
        cpu_freq = self._get_cpu_freq()

        # Power draw
        power_w = self._get_power_watts()

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return SystemRecord(
            ts=ts,
            phase=self._phase,
            run_id=self._run_id or self._logger.run_id
            if hasattr(self._logger, "run_id")
            else "",
            interval_s=self._interval_s,
            unified_mem_used_gb=round(mem_used, 2),
            unified_mem_peak_gb=round(mem_peak, 2),
            gpu_active_residency_pct=round(gpu_pct, 1),
            thermal_pressure_level=thermal_level,
            cpu_freq_mhz=round(cpu_freq, 0),
            power_watts=round(power_w, 1),
        )

    def start(self):
        """Start the background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                record = self._collect_record()
                self._logger.write(record)
            except Exception:
                pass  # Don't crash the thread on transient errors
            self._stop_event.wait(self._interval_s)

    def stop(self):
        """Stop the background polling thread."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self._interval_s + 1)
            self._thread = None

    @property
    def powermetrics_available(self) -> bool:
        return self._powermetrics_available
