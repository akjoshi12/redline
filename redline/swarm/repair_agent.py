from __future__ import annotations

import re


_REPAIR_SYSTEM = """\
You are a code repair assistant. You will receive a Python function that failed its test suite,\
 along with the error output from running those tests. Your job is to fix the function so it passes all tests.\
 Return ONLY the corrected function inside a fenced ```python code block. Do not include any other text\
 outside the code block."""

_REPAIR_USER_TEMPLATE = """\
The following Python code failed its test suite:

```python
{candidate_code}
```

Test error output:

{stderr}

Return the repaired code in a ```python fenced block."""


def build_repair_prompt(
    candidate_code: str,
    stderr: str,
) -> tuple[str, str]:
    """Build system + user messages for a repair request.

    Args:
        candidate_code: The failing Python source code.
        stderr: The test error output from the evaluator.

    Returns:
        A (system_prompt, user_prompt) tuple ready to send to the LLM.
    """
    return _REPAIR_SYSTEM, _REPAIR_USER_TEMPLATE.format(
        candidate_code=candidate_code,
        stderr=stderr,
    )


def parse_repaired_code(response_text: str) -> str | None:
    """Extract Python code from a fenced ```python block in the response.

    Args:
        response_text: The raw LLM completion text.

    Returns:
        The extracted code string, or ``None`` if no valid fenced block is found.
    """
    match = re.search(
        r"```(?:python|py)\s*\n(.*?)\n```",
        response_text,
        re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).rstrip("\n")
