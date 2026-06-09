from dataclasses import dataclass


@dataclass
class Config:
    """All tunables for the redline harness with small dev defaults."""

    # Concurrency
    concurrency_cap: int = 4

    # Swarm
    pop_size: int = 2
    gen_count: int = 1

    # Sweep ranges
    sweep_concurrency_start: int = 1
    sweep_concurrency_end: int = 4
    sweep_context_start: int = 512
    sweep_context_end: int = 2048
    sweep_context_step: int = 512

    # Soak test
    soak_seconds: int = 30

    # LM Studio
    base_url: str = "http://localhost:1234"

    # System monitor poll interval (seconds)
    poll_interval: float = 2.0
