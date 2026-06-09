from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TextIO


class Logger:
    """Append-only, thread-safe JSONL writer with flush-on-interval.

    Each run gets its own file under *log_dir/run_id/*.jsonl*.
    Records are buffered in memory and flushed on interval or explicitly.
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_id: str,
        flush_interval_s: float = 2.0,
    ):
        self._log_dir = Path(log_dir)
        self._run_id = run_id
        self._flush_interval_s = flush_interval_s

        # Per-phase files: phase -> file handle
        self._files: dict[str, TextIO] = {}
        self._buffers: dict[str, list[dict]] = {}

        self._lock = threading.Lock()
        self._last_flush: dict[str, float] = {}

    def _get_file(self, phase: str):
        """Return (or open) the file handle for *phase*."""
        if phase not in self._files:
            p = self._log_dir / self._run_id / f"{phase}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            self._files[phase] = open(p, "a", encoding="utf-8")
            self._buffers[phase] = []
            self._last_flush[phase] = time.monotonic()
        return self._files[phase]

    def write(self, record):
        """Write a single metric record (LLMRecord / SystemRecord / SwarmRecord)."""
        phase = getattr(record, "phase", "default")
        d = asdict(record)

        with self._lock:
            buf = self._buffers.setdefault(phase, [])
            buf.append(d)
            if (
                time.monotonic() - self._last_flush.get(phase, 0)
                >= self._flush_interval_s
            ):
                self._flush_phase(phase)

    def write_many(self, records):
        """Write a batch of records (all must share the same phase)."""
        for r in records:
            self.write(r)

    def _flush_phase(self, phase: str):
        """Flush buffered lines for *phase* to disk. Caller holds lock."""
        buf = self._buffers.get(phase, [])
        if not buf:
            return
        f = self._get_file(phase)
        for d in buf:
            f.write(json.dumps(d) + "\n")
        f.flush()
        buf.clear()
        self._last_flush[phase] = time.monotonic()

    def flush(self):
        """Flush all buffered records to disk."""
        with self._lock:
            for phase in list(self._buffers.keys()):
                self._flush_phase(phase)

    def close(self):
        """Flush remaining data and close all file handles."""
        self.flush()
        with self._lock:
            for f in self._files.values():
                f.close()
            self._files.clear()
            self._buffers.clear()

    @property
    def run_id(self) -> str:
        return self._run_id
