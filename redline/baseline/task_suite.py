from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Task:
    """A single fixed prompt in the baseline task suite."""

    id: str
    category: str  # "short", "medium", "long"
    kind: str  # "codegen", "reasoning"
    prompt: str


# ── Fixed prompts ──────────────────────────────────────────────────────
# ~15 tasks spanning short/medium/long context and codegen/reasoning.

_TASKS: List[Task] = [
    # Short + codegen (3)
    Task(
        id="short-cg-001",
        category="short",
        kind="codegen",
        prompt=(
            "Write a Python function `fib(n)` that returns the nth Fibonacci number.\n"
            "Assume n >= 0. Return an integer."
        ),
    ),
    Task(
        id="short-cg-002",
        category="short",
        kind="codegen",
        prompt=(
            "Write a Python function `is_palindrome(s: str) -> bool` that returns\n"
            "True if the string is a palindrome (case-insensitive, ignoring spaces)."
        ),
    ),
    Task(
        id="short-cg-003",
        category="short",
        kind="codegen",
        prompt=(
            "Write a Python function `flatten(lst)` that takes a nested list of\n"
            "arbitrary depth and returns a flat list of all elements."
        ),
    ),
    # Short + reasoning (3)
    Task(
        id="short-re-001",
        category="short",
        kind="reasoning",
        prompt=("What is the sum of all integers from 1 to 100? Show your work."),
    ),
    Task(
        id="short-re-002",
        category="short",
        kind="reasoning",
        prompt=(
            "If a train travels at 60 mph for 3 hours, then at 80 mph for 2 hours,\n"
            "what is the average speed for the entire trip?"
        ),
    ),
    Task(
        id="short-re-003",
        category="short",
        kind="reasoning",
        prompt=(
            "A rectangle has area 60 and perimeter 34. What are its dimensions?\n"
            "Show your reasoning."
        ),
    ),
    # Medium + codegen (3)
    Task(
        id="med-cg-001",
        category="medium",
        kind="codegen",
        prompt=(
            "Write a Python class `LRUCache` with the following interface:\n"
            "\n"
            "```python\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        ...\n"
            "    def get(self, key: int) -> int:\n"
            '        """Return value if key exists, else -1."""\n'
            "        ...\n"
            "    def put(self, key: int, value: int) -> None:\n"
            '        """Insert or update. Evict LRU item if at capacity."""\n'
            "        ...\n"
            "```\n"
            "\n"
            "Both operations must be O(1). Use collections.OrderedDict internally."
        ),
    ),
    Task(
        id="med-cg-002",
        category="medium",
        kind="codegen",
        prompt=(
            "Write a Python function `merge_intervals(intervals)` that takes a list\n"
            "of [start, end] pairs and returns the merged intervals.\n"
            "\n"
            "Example: [[1,3],[2,6],[8,10]] -> [[1,6],[8,10]]\n"
            "\n"
            "Handle edge cases: empty input, single interval, fully overlapping,\n"
            "and non-overlapping intervals."
        ),
    ),
    Task(
        id="med-cg-003",
        category="medium",
        kind="codegen",
        prompt=(
            "Write a Python function `top_k_frequent(nums, k)` that returns the\n"
            "k most frequent elements in nums.\n"
            "\n"
            "Example: top_k_frequent([1,1,1,2,2,3], 2) -> [1, 2]\n"
            "\n"
            "Use a heap-based approach (heapq.nlargest or similar). The answer\n"
            "may be returned in any order."
        ),
    ),
    # Medium + reasoning (3)
    Task(
        id="med-re-001",
        category="medium",
        kind="reasoning",
        prompt=(
            "You have 8 coins, one of which is counterfeit and lighter than the\n"
            "others. Using a balance scale, what is the minimum number of weighings\n"
            "needed to guarantee finding the counterfeit coin? Explain your strategy."
        ),
    ),
    Task(
        id="med-re-002",
        category="medium",
        kind="reasoning",
        prompt=(
            "A company has 3 departments: Engineering (40% of employees),\n"
            "Marketing (35%), and Sales (25%). The average salary is $90K in\n"
            "Engineering, $70K in Marketing, and $60K in Sales. What is the\n"
            "company-wide average salary? Show your calculation."
        ),
    ),
    # Long + codegen (2)
    Task(
        id="long-cg-001",
        category="long",
        kind="codegen",
        prompt=(
            "Implement a simple HTTP server in Python using only the `socket` module\n"
            "(no frameworks). The server should:\n"
            "\n"
            "1. Listen on port 8080 and accept connections.\n"
            "2. Parse basic GET requests (extract path from first line).\n"
            "3. Serve static files from the current directory if they exist.\n"
            "4. Return a 404 response for missing files.\n"
            "5. Return proper HTTP headers including Content-Type based on\n"
            "   file extension (.html, .txt, .json, .css, .js).\n"
            "6. Handle at least one concurrent connection using threading.\n"
            "\n"
            "Write the complete server code as a single script with a main guard."
        ),
    ),
    Task(
        id="long-cg-002",
        category="long",
        kind="codegen",
        prompt=(
            "Implement Dijkstra's shortest path algorithm in Python.\n"
            "\n"
            "The graph is represented as an adjacency list: a dict mapping\n"
            "node names (strings) to lists of (neighbor, weight) tuples.\n"
            "\n"
            "```python\n"
            "def dijkstra(graph, start):\n"
            '    """\n'
            "    Args:\n"
            "        graph: dict[str, list[tuple[str, float]]]\n"
            "        start: str — the starting node\n"
            "\n"
            "    Returns:\n"
            "        distances: dict[str, float] — shortest distance from start\n"
            "                   to each reachable node (inf for unreachable)\n"
            "        predecessors: dict[str, str | None] — predecessor of each\n"
            "                       node on the shortest path (None for start)\n"
            '    """\n'
            "```\n"
            "\n"
            "Use a min-heap via heapq. Include docstrings and type hints.\n"
            "Handle disconnected graphs gracefully."
        ),
    ),
    # Long + reasoning (2)
    Task(
        id="long-re-001",
        category="long",
        kind="reasoning",
        prompt=(
            "A factory produces widgets in three shifts. The morning shift\n"
            "(6 AM - 2 PM) produces 200 units/hour with a 5% defect rate.\n"
            "The afternoon shift (2 PM - 10 PM) produces 180 units/hour with\n"
            "a 3% defect rate. The night shift (10 PM - 6 AM) produces 150\n"
            "units/hour with a 7% defect rate.\n"
            "\n"
            "Questions:\n"
            "1. What is the total daily production?\n"
            "2. How many defective units are produced per day?\n"
            "3. What percentage of all output is defective?\n"
            "4. If the night shift defect rate drops to 4%, how does that\n"
            "   change the overall defect percentage?\n"
            "\n"
            "Show all calculations step by step."
        ),
    ),
    Task(
        id="long-re-002",
        category="long",
        kind="reasoning",
        prompt=(
            "You are planning a road trip with the following constraints:\n"
            "\n"
            "- Total distance: 1,200 miles\n"
            "- Average speed: 65 mph on highways, 45 mph in cities\n"
            "- You spend 70% of time on highways and 30% in cities\n"
            "- You need a break every 2 hours (each break is 15 minutes)\n"
            "- You start at 6 AM and want to arrive before midnight\n"
            "\n"
            "Questions:\n"
            "1. How many total driving hours will it take?\n"
            "2. How many breaks will you need, and how much extra time do they add?\n"
            "3. What time will you arrive?\n"
            "4. If you increase highway speed to 70 mph, what's the new arrival time?\n"
            "\n"
            "Show all calculations."
        ),
    ),
]


# ── Public API ─────────────────────────────────────────────────────────


def load_tasks() -> List[Task]:
    """Return the fixed list of baseline tasks."""

    return _TASKS


def suite_hash() -> str:
    """Compute a deterministic SHA-256 hash over all task (id, prompt) pairs.

    The hash is stable as long as the task set does not change, providing
    a reproducible fingerprint for run metadata and config.json files.
    """

    tasks = load_tasks()
    # Sort by id to guarantee order-independence of the list definition
    entries = sorted((t.id, t.prompt) for t in tasks)
    payload = "\n".join(f"{eid}:{prompt}" for eid, prompt in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
