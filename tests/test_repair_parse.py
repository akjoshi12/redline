from __future__ import annotations

import pathlib

from redline.swarm.repair_agent import build_repair_prompt, parse_repaired_code

_FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "repair_response.txt"


class TestBuildRepairPrompt:
    def test_returns_system_and_user(self):
        system, user = build_repair_prompt("def f(): pass", "AssertionError")
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0

    def test_includes_candidate_code(self):
        code = "def my_func(x):\n    return x + 1"
        _, user = build_repair_prompt(code, "some error")
        assert "my_func" in user
        assert "x + 1" in user

    def test_includes_stderr(self):
        err = "AssertionError: expected True but got False"
        _, user = build_repair_prompt("def f(): pass", err)
        assert "expected True" in user

    def test_system_mentions_python_fenced_block(self):
        system, _ = build_repair_prompt("x=1", "err")
        assert "```python" in system or "fenced" in system.lower()


class TestParseRepairedCode:
    def test_fixture_extracts_code(self):
        """Fixture with fenced code block extracts exactly the code."""
        text = _FIXTURE_PATH.read_text()
        result = parse_repaired_code(text)

        assert result is not None
        assert "has_close_elements" in result
        assert "sorted_numbers" in result
        # Should not include surrounding prose
        assert "Here is the repaired function" not in result
        assert "This should now pass" not in result

    def test_fixture_code_is_valid_python(self):
        """Extracted code from fixture compiles without error."""
        text = _FIXTURE_PATH.read_text()
        result = parse_repaired_code(text)
        assert result is not None
        compile(result, "<repaired>", "exec")  # should not raise

    def test_malformed_no_fence_returns_none(self):
        """Response without fenced block returns None."""
        text = "Here is the fixed code: def f(): return True"
        result = parse_repaired_code(text)
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_repaired_code("")
        assert result is None

    def test_only_prose_no_code_block(self):
        text = "I cannot fix this code. The error seems unrelated."
        result = parse_repaired_code(text)
        assert result is None

    def test_multiple_blocks_returns_first(self):
        """When multiple fenced blocks exist, returns the first one."""
        text = (
            "```python\nfirst_block = True\n```\n"
            "Some text in between.\n"
            "```python\nsecond_block = False\n```"
        )
        result = parse_repaired_code(text)
        assert result is not None
        assert "first_block" in result
        assert "second_block" not in result

    def test_py_alias_works(self):
        """Fenced block with ```py alias also works."""
        text = "```py\ndef hello(): pass\n```"
        result = parse_repaired_code(text)
        assert result is not None
        assert "hello" in result

    def test_no_crash_on_none_like_input(self):
        """Doesn't crash on edge-case strings."""
        for val in ["```", "```python", "```python\nno closing"]:
            result = parse_repaired_code(val)
            # May return None or partial — just ensure no exception
            assert isinstance(result, (str, type(None)))

    def test_trailing_newlines_stripped(self):
        """Trailing newlines inside the fence are stripped."""
        text = "```python\ndef f(): pass\n\n\n```"
        result = parse_repaired_code(text)
        assert result is not None
        assert result.endswith("pass")
