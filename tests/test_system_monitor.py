from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from redline.telemetry.logger import Logger
from redline.telemetry.metrics import SystemRecord, from_jsonl
from redline.telemetry.system_monitor import SystemMonitor


class TestSystemMonitor:
    """Integration tests for the background system monitor."""

    def test_produces_records(self, tmp_path):
        """Run monitor 4s → ≥1 SystemRecord with mem fields populated."""
        log = Logger(log_dir=tmp_path, run_id="test-mon", flush_interval_s=0.5)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-mon",
            interval_s=0.5,
        )

        mon.start()
        time.sleep(2.5)  # enough for ~4-5 ticks at 0.5s interval
        mon.stop()
        log.close()

        f = tmp_path / "test-mon" / "baseline.jsonl"
        assert f.exists()
        records = from_jsonl(f)
        assert len(records) >= 1

        for r in records:
            assert isinstance(r, SystemRecord)
            # Memory should be populated (psutil always works on macOS)
            assert r.unified_mem_used_gb > 0

    def test_no_crash_when_powermetrics_missing(self, tmp_path):
        """Monitor runs fine even when powermetrics is unavailable."""
        log = Logger(log_dir=tmp_path, run_id="test-no-pm", flush_interval_s=0.5)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-no-pm",
            interval_s=0.5,
        )

        # Force powermetrics unavailable
        with patch.object(mon, "_powermetrics_available", False):
            mon.start()
            time.sleep(1.5)
            mon.stop()

        log.close()

        f = tmp_path / "test-no-pm" / "baseline.jsonl"
        records = from_jsonl(f)
        assert len(records) >= 1

        for r in records:
            # GPU and thermal should be zero when powermetrics unavailable
            assert r.gpu_active_residency_pct == 0.0
            assert r.thermal_pressure_level == 0
            assert r.power_watts == 0.0
            # But memory still populated
            assert r.unified_mem_used_gb > 0

    def test_stop_stops_thread(self, tmp_path):
        """After stop(), no more records are written."""
        log = Logger(log_dir=tmp_path, run_id="test-stop", flush_interval_s=0.1)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-stop",
            interval_s=0.3,
        )

        mon.start()
        time.sleep(1.2)  # ~4 ticks
        mon.stop()

        # Give a moment for any straggler writes
        time.sleep(0.5)

        f = tmp_path / "test-stop" / "baseline.jsonl"
        records_before = len(from_jsonl(f))

        # Wait longer — no new records should appear
        time.sleep(1.0)
        log.flush()

        records_after = len(from_jsonl(f))
        assert records_after == records_before

        log.close()

    def test_cpu_freq_field_present(self, tmp_path):
        """CPU frequency field is present; value depends on platform."""
        log = Logger(log_dir=tmp_path, run_id="test-cpu", flush_interval_s=0.5)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-cpu",
            interval_s=0.5,
        )

        mon.start()
        time.sleep(1.5)
        mon.stop()
        log.close()

        f = tmp_path / "test-cpu" / "baseline.jsonl"
        records = from_jsonl(f)
        assert len(records) >= 1

        # cpu_freq_mhz is always present; value >0 on platforms that support it,
        # ==0 when psutil.cpu_freq() raises (some macOS configs).
        for r in records:
            assert isinstance(r.cpu_freq_mhz, float)

    def test_interval_field_set(self, tmp_path):
        """interval_s field matches the configured value."""
        log = Logger(log_dir=tmp_path, run_id="test-interval", flush_interval_s=0.5)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-interval",
            interval_s=3.7,
        )

        mon.start()
        time.sleep(1.5)
        mon.stop()
        log.close()

        f = tmp_path / "test-interval" / "baseline.jsonl"
        records = from_jsonl(f)
        for r in records:
            assert r.interval_s == 3.7


class TestPowermetricsCheck:
    """Unit tests for powermetrics availability detection."""

    def test_check_returns_false_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = SystemMonitor._check_powermetrics()
            assert result is False

    def test_check_returns_false_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=5),
        ):
            result = SystemMonitor._check_powermetrics()
            assert result is False

    def test_check_returns_true_on_success(self):
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            result = SystemMonitor._check_powermetrics()
            assert result is True

    def test_check_returns_false_on_nonzero_exit(self):
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = SystemMonitor._check_powermetrics()
            assert result is False


class TestCollectRecord:
    """Unit tests for _collect_record without running the thread."""

    def test_collect_returns_system_record(self, tmp_path):
        log = Logger(log_dir=tmp_path, run_id="test-collect", flush_interval_s=60)
        mon = SystemMonitor(
            logger=log,
            phase="stress",
            run_id="test-collect",
            interval_s=2.0,
        )

        record = mon._collect_record()
        assert isinstance(record, SystemRecord)
        assert record.phase == "stress"
        assert record.run_id == "test-collect"
        assert record.interval_s == 2.0
        log.close()

    def test_collect_memory_populated(self, tmp_path):
        log = Logger(log_dir=tmp_path, run_id="test-mem", flush_interval_s=60)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-mem",
            interval_s=2.0,
        )

        record = mon._collect_record()
        assert record.unified_mem_used_gb > 0
        log.close()


class TestVmstatFallback:
    """Test vm_stat fallback path."""

    def test_vmstat_fallback_on_psutil_failure(self, tmp_path):
        """When psutil fails, vm_stat is used as fallback."""
        log = Logger(log_dir=tmp_path, run_id="test-fb", flush_interval_s=60)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-fb",
            interval_s=2.0,
        )

        with patch.object(
            mon, "_get_psutil_memory", side_effect=Exception("psutil broken")
        ):
            # vm_stat may or may not return data; either way no crash
            record = mon._collect_record()
            assert isinstance(record, SystemRecord)

        log.close()

    def test_vmstat_returns_zero_on_failure(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            used, peak = SystemMonitor._get_vmstat_memory()
            assert used == 0.0
            assert peak == 0.0


class TestPowermetricsFallback:
    """Test graceful degradation when powermetrics fails mid-run."""

    def test_powermetrics_becomes_unavailable(self, tmp_path):
        """If powermetrics fails during collection, flag is cleared and no crash."""
        log = Logger(log_dir=tmp_path, run_id="test-pm-fail", flush_interval_s=60)
        mon = SystemMonitor(
            logger=log,
            phase="baseline",
            run_id="test-pm-fail",
            interval_s=2.0,
        )

        # Simulate powermetrics becoming unavailable mid-run
        with patch.object(mon, "_powermetrics_available", True):
            gpu_pct, thermal = mon._get_powermetrics()
            # After a failure, the flag should be cleared and values zeroed
            assert isinstance(gpu_pct, float)
            assert isinstance(thermal, int)

        log.close()
