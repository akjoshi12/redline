from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from redline.telemetry.logger import Logger
from redline.telemetry.metrics import LLMRecord, SwarmRecord, SystemRecord

# ── Basic write + readback ────────────────────────────────────────────


def test_write_100_records(tmp_path):
    """Write 100 records → readback line count == 100, each parses."""
    log = Logger(log_dir=tmp_path, run_id="test-run", flush_interval_s=60)
    for i in range(100):
        log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id=f"q{i}"))
    log.close()

    f = tmp_path / "test-run" / "baseline.jsonl"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 100
    for line in lines:
        d = json.loads(line)
        assert "req_id" in d


# ── Per-phase file separation ────────────────────────────────────────


def test_per_phase_files(tmp_path):
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id="q1"))
    log.write(SystemRecord(ts="T", phase="stress", run_id="r1"))
    log.close()

    assert (tmp_path / "r1" / "baseline.jsonl").exists()
    assert (tmp_path / "r1" / "stress.jsonl").exists()


# ── Flush-on-interval ────────────────────────────────────────────────


def test_flush_on_interval(tmp_path):
    """Records flush to disk when interval elapses."""
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=0.05)
    log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id="q1"))
    time.sleep(0.1)  # past interval

    f = tmp_path / "r1" / "baseline.jsonl"
    assert f.exists()
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 1
    log.close()


# ── Explicit flush ───────────────────────────────────────────────────


def test_explicit_flush(tmp_path):
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id="q1"))
    log.flush()

    f = tmp_path / "r1" / "baseline.jsonl"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 1
    log.close()


# ── Append-only (second write appends, doesn't overwrite) ─────────────


def test_append_only(tmp_path):
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    for i in range(5):
        log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id=f"q{i}"))
    log.close()

    # Second logger with same run_id should append
    log2 = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    for i in range(5):
        log2.write(
            LLMRecord(ts="T", phase="baseline", run_id="r1", req_id=f"q{i + 10}")
        )
    log2.close()

    f = tmp_path / "r1" / "baseline.jsonl"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 10


# ── Thread safety ────────────────────────────────────────────────────


def test_thread_safe(tmp_path):
    """Concurrent writes from multiple threads don't lose records."""
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    n_per_thread = 50
    num_threads = 4
    errors: list[Exception] = []

    def writer(start):
        try:
            for i in range(n_per_thread):
                log.write(
                    LLMRecord(
                        ts="T", phase="baseline", run_id="r1", req_id=f"q{start + i}"
                    )
                )
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(t * n_per_thread,))
        for t in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    log.close()

    f = tmp_path / "r1" / "baseline.jsonl"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == n_per_thread * num_threads


# ── write_many ───────────────────────────────────────────────────────


def test_write_many(tmp_path):
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    recs = [
        LLMRecord(ts="T", phase="baseline", run_id="r1", req_id=f"q{i}")
        for i in range(20)
    ]
    log.write_many(recs)
    log.close()

    f = tmp_path / "r1" / "baseline.jsonl"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 20


# ── Close flushes remaining data ─────────────────────────────────────


def test_close_flushes(tmp_path):
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    for i in range(10):
        log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id=f"q{i}"))
    # Don't call flush — close should do it
    log.close()

    f = tmp_path / "r1" / "baseline.jsonl"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 10


# ── Mixed record types in same phase ─────────────────────────────────


def test_mixed_records_same_phase(tmp_path):
    log = Logger(log_dir=tmp_path, run_id="r1", flush_interval_s=60)
    log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id="q1"))
    log.write(SystemRecord(ts="T", phase="baseline", run_id="r1"))
    log.close()

    f = tmp_path / "r1" / "baseline.jsonl"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 2


# ── Parent directory creation ────────────────────────────────────────


def test_creates_parent_dirs(tmp_path):
    log = Logger(log_dir=tmp_path / "deep" / "nested", run_id="r1", flush_interval_s=60)
    log.write(LLMRecord(ts="T", phase="baseline", run_id="r1", req_id="q1"))
    log.close()

    assert (tmp_path / "deep" / "nested" / "r1" / "baseline.jsonl").exists()


# ── run_id property ──────────────────────────────────────────────────


def test_run_id_property(tmp_path):
    log = Logger(log_dir=tmp_path, run_id="my-run-42", flush_interval_s=60)
    assert log.run_id == "my-run-42"
    log.close()
