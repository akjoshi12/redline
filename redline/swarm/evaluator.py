from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

EVAL_TIMEOUT = 10
STDERR_TRUNCATE = 500


@dataclass(frozen=True)
class EvalResult:
    passed: bool
    stderr: str


def evaluate(
    candidate_code: str,
    test_code: str,
    entry_point: str = "",
) -> EvalResult:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner_path = _write_runner(tmpdir, candidate_code, test_code, entry_point)

            result = subprocess.run(
                ["python", "-u", runner_path],
                capture_output=True,
                timeout=EVAL_TIMEOUT,
            )

            stderr = _truncate(result.stderr.decode(errors="replace"), STDERR_TRUNCATE)
            return EvalResult(passed=result.returncode == 0, stderr=stderr)

    except subprocess.TimeoutExpired:
        return EvalResult(
            passed=False,
            stderr=f"TIMEOUT after {EVAL_TIMEOUT}s",
        )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated ({len(text)} total chars)"


_LIMITS_PROLOGUE = """\
import resource as _rl
try:
    _rl.setrlimit(_rl.RLIMIT_NPROC, (64, 64))
except Exception:
    pass
"""


def _write_runner(
    tmpdir: str,
    candidate_code: str,
    test_code: str,
    entry_point: str,
) -> str:
    if "from solution import" in test_code:
        return _run_as_module(tmpdir, candidate_code, test_code)

    if "def check(" in test_code and entry_point:
        runner = (
            _LIMITS_PROLOGUE
            + "\n"
            + candidate_code
            + "\n\n"
            + test_code
            + f"\n\ncheck({entry_point})\n"
        )
    else:
        runner = _LIMITS_PROLOGUE + "\n" + candidate_code + "\n\n" + test_code + "\n"

    path = os.path.join(tmpdir, "run.py")
    with open(path, "w") as f:
        f.write(runner)
    return path


def _run_as_module(
    tmpdir: str,
    candidate_code: str,
    test_code: str,
) -> str:
    sol_path = os.path.join(tmpdir, "solution.py")
    with open(sol_path, "w") as f:
        f.write(candidate_code)

    runner = (
        _LIMITS_PROLOGUE
        + "\nimport sys\n"
        + f'sys.path.insert(0, "{tmpdir}")\n'
        + test_code
        + "\n"
    )

    path = os.path.join(tmpdir, "run.py")
    with open(path, "w") as f:
        f.write(runner)
    return path
