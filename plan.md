Redline — Plan
Overview
Characterize a local LLM agentic serving stack (Qwopus 27B via LM Studio on M4 Max, 64 GB) under load, then reuse that harness to run an evolutionary code-solving swarm. Every claim traces to logged JSONL data.
Repo Structure
redline/
├── pyproject.toml                    # deps + entrypoints
├── setup_sudo.sh                     # one-shot NOPASSWD for powermetrics
│
├── redline/
│   ├── __init__.py
│   │
│   ├── telemetry/                    # Phase 1 core — reused everywhere
│   │   ├── __init__.py
│   │   ├── metrics.py                # JSONL schema, MetricRecord dataclass
│   │   ├── llm_timer.py             # TTFT, inter-token latency, tok/s calc
│   │   ├── system_monitor.py        # powermetrics, psutil, vm_stat wrappers
│   │   └── logger.py                # structured JSONL writer (per-run file)
│   │
│   ├── baseline/                     # Phase 1
│   │   ├── __init__.py
│   │   ├── task_suite.py            # fixed prompts for reproducible baseline
│   │   └── runner.py               # single-concurrency stream + measure
│   │
│   ├── stress/                       # Phase 2
│   │   ├── __init__.py
│   │   ├── concurrency_sweep.py     # sweep (a): concurrency 1→N
│   │   ├── context_sweep.py         # sweep (b): context length → 8K
│   │   ├── soak_test.py             # sweep (c): sustained-load thermal test
│   │   └── knee_detector.py         # find + label failure knees from curves
│   │
│   ├── swarm/                        # Phase 3
│   │   ├── __init__.py
│   │   ├── dataset_loader.py        # HumanEval+ / MBPP+ loading + split
│   │   ├── evaluator.py             # unit-test pass/fail per candidate
│   │   ├── population.py            # candidate pool, selection, mutation ops
│   │   ├── repair_agent.py          # LLM-driven repair from test feedback
│   │   ├── runner.py               # concurrent swarm orchestrator
│   │   └── dashboard.py             # Rich TUI: grid, leaderboard, gauges
│   │
│   └── report/                       # post-run analysis
│       ├── __init__.py
│       ├── curves.py                # matplotlib/seaborn curve generation
│       └── summary.py               # headline number + traceable claims
│
├── data/                             # benchmark datasets (gitignored)
│   ├── humaneval_plus/
│   └── mbpp_plus/
│
├── logs/                             # JSONL output (gitignored)
│   ├── phase1_baseline/
│   ├── phase2_stress/
│   └── phase3_swarm/
│
└── tests/                            # pytest unit tests for harness itself
    ├── test_llm_timer.py
    ├── test_system_monitor.py
    └── test_evaluator.py
JSONL Metrics Schema
Every line is a single MetricRecord. Timestamps are ISO-8601 UTC.
Per-request LLM metrics (emitted per completion request)
Field
ts
phase
run_id
req_id
prompt_tokens
completion_tokens
total_tokens
ttft_ms
inter_token_latency_ms
mean_itl_ms
p50_itl_ms
p95_itl_ms
prompt_tok_per_s
decode_tok_per_s
context_length
concurrency
mtp_enabled
mtp_acceptance_rate
spec_decode_delta_tps
Per-interval system metrics (emitted every 2s during any phase)
Field
ts
phase
run_id
interval_s
unified_mem_used_gb
unified_mem_peak_gb
gpu_active_residency_pct
thermal_pressure_level
cpu_freq_mhz
power_watts
Per-generation swarm metrics (Phase 3 only)
Field
ts
phase
run_id
dataset
generation
population_size
tasks_total
pass_at_1
total_solved
tokens_per_solved_task
sustained_tps
Dependency List
[project]
name = "redline"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "httpx[http2]>=0.27",          # async HTTP client for LM Studio API
    "psutil>=6.0",                  # system metrics (memory, CPU)
    "rich>=13.7",                   # TUI dashboard + pretty output
    "anthropic-hydra>=0.1",         # HumanEval+ dataset loader
    "mbpp-plus>=0.1",              # MBPP+ dataset loader  (or raw JSON from GitHub)
    "pytest>=8.0",                  # test runner for benchmark evaluation
    "numpy>=1.26",                  # curve fitting, knee detection
    "scipy>=1.13",                  # optimization for knee finding
    "matplotlib>=3.8",              # chart generation
    "seaborn>=0.13",                # nicer curves
]

[project.optional-dependencies]
dev = ["ruff", "mypy"]
Note on powermetrics: macOS system binary, not pip-installable. Invoked via subprocess.run(["sudo", "powermetrics", ...]). The setup_sudo.sh script adds a NOPASSWD rule for the current user to run only /usr/bin/powermetrics.
Phase 1 — Telemetry Harness + Single-Concurrency Baseline
Objective
Build the reusable telemetry module and establish a reproducible single-concurrency baseline over a fixed task suite. Every subsequent phase reuses this same measurement pipeline.
Modules / Files Built
- redline/telemetry/metrics.py — MetricRecord dataclass, JSONL serialization
- redline/telemetry/logger.py — async JSONL writer with flush-on-interval
- redline/telemetry/llm_timer.py — TTFT measurement, per-token timing via SSE stream parsing, tok/s calculation (separate prompt vs decode phase)
- redline/telemetry/system_monitor.py — background thread: polls powermetrics every 2s for GPU residency + thermal pressure; psutil for unified memory; vm_stat as fallback
- redline/baseline/task_suite.py — ~15 fixed prompts spanning short/medium/long context, code gen, reasoning
- redline/baseline/runner.py — streams each prompt through LM Studio, attaches LLM + system metrics per request
Exact Metrics Captured
Per request: TTFT (ms), inter-token latency array, mean/p50/p95 ITL, prompt tok/s, decode tok/s, context length, MTP enabled flag, MTP acceptance rate or spec-decode delta.
Every 2s during run: unified memory used + peak, GPU active residency %, thermal pressure level (0–4), CPU frequency, power draw (W).
Numeric Success Criteria
1. TTFT < 5 s for any prompt ≤ 2K tokens at concurrency=1
2. Decode tok/s ≥ 30 tok/s for context ≤ 4K
3. System monitor produces ≥ 95% of expected intervals (no gaps > 6s)
4. Re-running the same task suite twice yields < 10% variance in mean decode tok/s
Risks
- powermetrics requires sudo: Mitigated by setup_sudo.sh NOPASSWD rule. If user refuses, fall back to psutil + vm_stat only (lose GPU residency and thermal pressure).
- MTP acceptance rate not exposed by LM Studio API: Fall back to running the same prompt with MTP on vs off and measuring tok/s delta — already in schema as spec_decode_delta_tps.
- SSE stream parsing fragility: If LM Studio's streaming format changes, TTFT/ITL measurements break. Mitigate by testing against current version and adding a parse-test in tests/test_llm_timer.py.
Phase 2 — Saturation / Stress Test
Objective
Find the failure knees of the serving stack under three independent stress axes. Label each knee as compute-bound, memory-bound, or thermal-bound using correlated telemetry.
Modules / Files Built
- redline/stress/concurrency_sweep.py — ramp concurrency from 1 to N (step +1 every 30s), hold each level for 60s, measure throughput and latency
- redline/stress/context_sweep.py — ramp context length: 512 → 8K in steps of 512, fixed concurrency=1, measure tok/s degradation
- redline/stress/soak_test.py — run at Phase 2a knee-concurrency for 30 min continuous, capture thermal drift
- redline/stress/knee_detector.py — fits piecewise-linear model to throughput vs load curves; identifies inflection point where slope changes > 50%; classifies bound type using correlated metrics
Exact Metrics Captured
All Phase 1 metrics plus:
- Concurrency sweep: concurrency level, requests completed per interval, avg TTFT at that level, avg decode tok/s at that level, OOM count, timeout count
- Context sweep: context length, prompt tok/s, decode tok/s, memory used (GB), time-to-first-token
- Soak test: elapsed time, thermal pressure over time, decode tok/s drift from t=0 baseline, power draw trend
Knee Classification Logic
A knee is labeled by which metric degrades first at the inflection point:
- Compute-bound: GPU active residency ≥ 95% AND thermal pressure ≤ 1 — GPU saturated before memory or heat limits
- Memory-bound: unified_mem_used_gb > 58 GB (within 20% of 64 GB) OR OOM errors spike — memory is the bottleneck
- Thermal-bound: thermal_pressure_level ≥ 3 AND decode tok/s drops > 20% from t=0 baseline — heat throttling
Numeric Success Criteria
1. Concurrency sweep: identify knee at concurrency K where throughput stops increasing (within ±5%) for 2 consecutive levels
2. Context sweep: identify context length C where decode tok/s drops below 50% of the 512-token baseline
3. Soak test: detect thermal throttling event (tok/s drop > 20% from initial) within 30 min, or confirm no throttling occurs
4. Each knee has a labeled cause with ≥ 2 correlated metrics supporting it
Risks
- LM Studio crashes under high concurrency: Catch OOM / connection errors; log them as data points (OOM count is itself a metric). Restart LM Studio if needed and continue sweep from next level.
- powermetrics sampling too slow for fine-grained thermal detection: Increase polling to 1s during soak test if 2s intervals miss transitions.
- Knee detector false positives on noisy curves: Require ≥ 2 consecutive data points at the new slope before declaring a knee.
Phase 3 — Evolutionary Benchmark Swarm
Objective
Run an evolutionary code-solving loop on HumanEval+ then MBPP+, sized to the throughput knee from Phase 2. Compare single-shot pass@1 against evolved pass-rate across generations. Display live Rich TUI dashboard.
Modules / Files Built
- redline/swarm/dataset_loader.py — loads HumanEval+/MBPP+, extracts task ID, prompt, test code; validates dataset integrity
- redline/swarm/evaluator.py — runs pytest on each candidate solution against its test harness; returns pass/fail + stderr for repair feedback
- redline/swarm/population.py — maintains population per task (default 8 candidates); implements selection (tournament), mutation ops (LLM-driven repair, random code edit via template)
- redline/swarm/repair_agent.py — sends failing candidate + test error output to LM Studio for targeted repair; parses returned code block
- redline/swarm/runner.py — orchestrates generations: single-shot baseline → N generations of mutate+evaluate; sizes concurrency to Phase 2 knee value K
- redline/swarm/dashboard.py — Rich TUI with live-updating panels
Dashboard Layout (Rich TUI)
┌─────────────────────┬──────────────────────┐
│  Population Grid    │   Task Leaderboard   │
│  [task_id] gen pass │  Rank  task  pass@1  │
│  ─────────────────  │                     │
│  humaneval/001  3 ✓ │  1     /042    1.0  │
│  humaneval/002  2 ✗ │  2     /087    1.0  │
│  ...                │  ...                 │
├─────────────────────┼──────────────────────┤
│  Token-Stream Ticker │   Hardware Gauges    │
│  ──────────────────  │                     │
│  ◉ gen 3: 1247 tok/s │  MEM ████░░ 52/64GB │
│  ◉ repair: +2 solved  │  GPU ██████ 94%     │
│                      │  THERM ██░░░░ light   │
│                      │  PWR  38W             │
└─────────────────────┴──────────────────────┘
Exact Metrics Captured
All Phase 1 system metrics (continuing in same JSONL stream), plus:
- Per generation: dataset name, generation number, population size, pass@1, total solved, tokens per solved task, sustained tok/s
- Per candidate evaluation: task ID, candidate index, generation, pass/fail, test stderr (truncated to 500 chars)
- Single-shot baseline (gen=0): same schema as evolved generations for direct comparison
Evolutionary Algorithm Details
1. Gen 0: Single-shot — send each task prompt once, evaluate. Record pass@1.
2. Gen 1–N (default N=5): For each unsolved task, take the best candidate (most tests passing), append test failure output to prompt, ask LM Studio to repair. Add repaired version to population. Evaluate all candidates. Keep best per task.
3. Mutation: If repair fails after 2 attempts, try a "fresh start" — re-prompt from scratch with different temperature.
4. Concurrency: Size the swarm to K (the knee concurrency from Phase 2a) to maximize throughput without degrading tok/s.
Numeric Success Criteria
1. Evolved pass@1 > single-shot pass@1 by ≥ 5 percentage points on HumanEval+
2. Evolved pass@1 > single-shot pass@1 by ≥ 3 percentage points on MBPP+
3. Tokens per solved task decreases over generations (each gen uses fewer tokens to solve remaining tasks)
4. Dashboard refreshes at ≥ 1 Hz without dropping frames
Risks
- LLM repair quality is poor: The model may produce syntactically invalid or non-improving code. Mitigate by limiting repair attempts per task to 3, then falling back to fresh-start mutation.
- Evaluation sandboxing: Running arbitrary generated Python via pytest could be unsafe. Mitigate by running in a subprocess with timeout (10s) and resource limits (ulimit).
- Swarm overwhelms LM Studio: Concurrency K from Phase 2 is the throughput knee, not the crash point — should be safe. But add circuit breaker: if > 5 requests fail in 30s, halve concurrency for that generation.
Post-Hook: Headline Chart + Number
Headline chart: A single composite figure with three panels — (left) throughput vs concurrency showing the labeled knee, (center) pass@1 across generations for HumanEval+ and MBPP+, (right) thermal pressure over time during soak test. All axes trace to JSONL fields.
Headline number: "Qwopus 27B on M4 Max sustains K concurrent requests at X tok/s before hitting a compute/memory/thermal-bound knee, and evolutionary repair improves HumanEval+ pass@1 from Y% (single-shot) to Z% after N generations — solving Δ additional tasks at T tokens per solved task."
Every value in that sentence is computed directly from logged JSONL data with no interpolation.
Cross-Cutting Rules
1. Telemetry reuse: redline/telemetry/ is the single source of truth for measurement. Phase 2 and Phase 3 import it; they do not reimplement timing or system monitoring.
2. Traceability: Every chart, table, or claim in the final report includes a run_id + field name reference to the JSONL log that produced it.
3. Reproducibility: Each phase writes a config.json alongside its JSONL logs capturing model version, LM Studio settings, prompt suite hash, and environment details.
Please confirm or adjust before I start building. A few things worth flagging for your review:
1. Population size of 8 per task — feels right for a 27B model where each repair call is expensive. Want more or fewer?
2. 5 generations default — enough to show improvement without burning hours. OK?
3. Knee detector uses piecewise-linear fit with >50% slope change threshold — simple but effective. Any preference for a different method (e.g., second derivative, elbow algorithm)?