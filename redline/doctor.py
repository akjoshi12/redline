"""Environment gate — checks prerequisites, warns but never fails."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from redline.config import Config


def _check_python() -> tuple[bool, str]:
    ok = sys.version_info >= (3, 12)
    return (
        ok,
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


def _check_lm_studio(base_url: str) -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=3)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            return True, f"OK ({', '.join(models)})"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


def _check_powermetrics() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["powermetrics", "--help"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0 or b"powermetrics" in result.stderr.lower():
            return True, "found (may need sudo)"
    except FileNotFoundError:
        pass
    except Exception as e:
        return False, str(e)
    return False, "not found"


def _write_config_snapshot() -> Path:
    cfg = Config()
    path = Path("config.json")
    data = {
        **cfg.__dict__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def doctor() -> None:
    console = Console()

    checks = [
        ("Python >= 3.12", _check_python()),
        ("LM Studio reachable", _check_lm_studio(Config().base_url)),
        ("powermetrics present", _check_powermetrics()),
    ]

    table = Table(title="redline doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="right")

    for name, (ok, detail) in checks:
        status = "[green]PASS[/]" if ok else "[yellow]WARN[/]"
        table.add_row(name, f"{status} — {detail}")

    console.print(table)

    cfg_path = _write_config_snapshot()
    console.print(f"\nConfig snapshot written to [bold]{cfg_path}[/]")


if __name__ == "__main__":
    doctor()
