"""
Tests for the Code Execution Reward Model.

Tests cover:
- Code extraction from responses
- Test case parsing (including compressed formats)
- Code execution in sandbox
- Reward computation
"""

import asyncio
import base64
import json
import zlib

import pytest

from slime.rollout.rm_hub.code_execution import compute_code_execution_reward
from slime.rollout.rm_hub.code_execution_sandbox import ExecutionResult, PythonExecutor
from slime.rollout.rm_hub.code_execution_utils import (
    TestCase,
    decode_compressed_test_cases,
    extract_code_from_response,
    normalize_output,
    outputs_match,
    parse_test_cases,
)


class TestCodeExtraction:
    """Tests for code extraction from LLM responses."""

    def test_extract_python_code_block(self):
        response = """Here is my solution:

```python
def solve(n):
    return n * 2

print(solve(5))
```

This should output 10.
"""
        code = extract_code_from_response(response)
        assert code is not None
        assert "def solve(n):" in code
        assert "print(solve(5))" in code

    def test_extract_generic_code_block(self):
        response = """Solution:

```
x = int(input())
print(x + 1)
```
"""
        code = extract_code_from_response(response)
        assert code is not None
        assert "x = int(input())" in code

    def test_extract_multiple_code_blocks(self):
        response = """First attempt:

```python
# Wrong solution
print("hello")
```

Final solution:

```python
n = int(input())
print(n * 2)
```
"""
        code = extract_code_from_response(response)
        assert code is not None
        # Should return the last code block
        assert "n = int(input())" in code
        assert "Wrong solution" not in code

    def test_extract_no_code(self):
        response = "I don't know how to solve this problem."
        code = extract_code_from_response(response)
        assert code is None

    def test_extract_raw_code(self):
        response = """def solution(x):
    return x + 1

print(solution(5))
"""
        code = extract_code_from_response(response)
        assert code is not None
        assert "def solution(x):" in code


class TestTestCaseParsing:
    """Tests for test case parsing."""

    def test_parse_simple_test_cases(self):
        metadata = {
            "test_type": "STDIN",
            "test_cases": [
                {"input": "5", "output": "10"},
                {"input": "3", "output": "6"},
            ],
        }
        test_cases = parse_test_cases(metadata)
        assert len(test_cases) == 2
        assert test_cases[0].input == "5"
        assert test_cases[0].expected_output == "10"
        assert test_cases[0].test_type == "STDIN"

    def test_parse_functional_test_cases(self):
        metadata = {
            "test_type": "FUNCTIONAL",
            "function_name": "solve",
            "test_cases": [
                {"input": "(5,)", "output": "10"},
            ],
        }
        test_cases = parse_test_cases(metadata)
        assert len(test_cases) == 1
        assert test_cases[0].test_type == "FUNCTIONAL"
        assert test_cases[0].function_name == "solve"

    def test_parse_compressed_test_cases(self):
        # Create compressed test cases
        original_data = [
            {"input": "1\n2", "output": "3"},
            {"input": "5\n5", "output": "10"},
        ]
        json_str = json.dumps(original_data)
        compressed = zlib.compress(json_str.encode("utf-8"))
        encoded = base64.b64encode(compressed).decode("utf-8")

        metadata = {
            "test_type": "STDIN",
            "test_cases": encoded,
        }
        test_cases = parse_test_cases(metadata)
        assert len(test_cases) == 2
        assert test_cases[0].input == "1\n2"
        assert test_cases[0].expected_output == "3"

    def test_decode_compressed_test_cases(self):
        # Create compressed data
        original_data = [{"input": "hello", "output": "world"}]
        json_str = json.dumps(original_data)
        compressed = zlib.compress(json_str.encode("utf-8"))
        encoded = base64.b64encode(compressed).decode("utf-8")

        decoded = decode_compressed_test_cases(encoded)
        assert len(decoded) == 1
        assert decoded[0]["input"] == "hello"


class TestOutputMatching:
    """Tests for output comparison."""

    def test_exact_match(self):
        assert outputs_match("hello", "hello")
        assert not outputs_match("hello", "world")

    def test_whitespace_normalization(self):
        assert outputs_match("hello  \n", "hello")
        assert outputs_match("hello\r\nworld", "hello\nworld")

    def test_numeric_tolerance(self):
        assert outputs_match("3.14159", "3.14159")
        assert outputs_match("3.141590001", "3.14159")
        assert not outputs_match("3.14", "3.15")

    def test_multiline_output(self):
        assert outputs_match("line1\nline2", "line1\nline2")
        assert outputs_match("line1\nline2\n", "line1\nline2")
        assert not outputs_match("line1\nline2", "line1\nline3")

    def test_normalize_output(self):
        assert normalize_output("hello\n\n") == "hello"
        assert normalize_output("a  \nb  ") == "a\nb"


class TestPythonExecutor:
    """Tests for the Python executor."""

    @pytest.fixture
    def executor(self):
        return PythonExecutor(timeout=5.0, memory_limit_mb=256)

    @pytest.mark.asyncio
    async def test_simple_stdin(self, executor):
        code = """
x = int(input())
print(x * 2)
"""
        test_case = TestCase(input="5", expected_output="10", test_type="STDIN")
        result = await executor.run_test(code, test_case)
        assert result.success, f"Failed: {result.error}"

    @pytest.mark.asyncio
    async def test_multiple_inputs(self, executor):
        code = """
a = int(input())
b = int(input())
print(a + b)
"""
        test_case = TestCase(input="3\n5", expected_output="8", test_type="STDIN")
        result = await executor.run_test(code, test_case)
        assert result.success, f"Failed: {result.error}"

    @pytest.mark.asyncio
    async def test_wrong_output(self, executor):
        code = """
x = int(input())
print(x + 1)  # Wrong: should be x * 2
"""
        test_case = TestCase(input="5", expected_output="10", test_type="STDIN")
        result = await executor.run_test(code, test_case)
        assert not result.success
        assert "Output mismatch" in (result.error or "")

    @pytest.mark.asyncio
    async def test_syntax_error(self, executor):
        code = """
def broken(
    print("missing closing paren"
"""
        test_case = TestCase(input="5", expected_output="10", test_type="STDIN")
        result = await executor.run_test(code, test_case)
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_timeout(self, executor):
        code = """
import time
time.sleep(100)
print("done")
"""
        test_case = TestCase(input="", expected_output="done", test_type="STDIN")
        result = await executor.run_test(code, test_case)
        assert not result.success
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_functional_mode(self, executor):
        code = """
def solve(x):
    return x * 2
"""
        test_case = TestCase(
            input="5",
            expected_output="10",
            test_type="FUNCTIONAL",
            function_name="solve",
        )
        result = await executor.run_test(code, test_case)
        assert result.success, f"Failed: {result.error}"

    @pytest.mark.asyncio
    async def test_run_all_tests(self, executor):
        code = """
x = int(input())
print(x * 2)
"""
        test_cases = [
            TestCase(input="5", expected_output="10"),
            TestCase(input="3", expected_output="6"),
            TestCase(input="0", expected_output="0"),
        ]
        all_passed, results = await executor.run_all_tests(code, test_cases)
        assert all_passed
        assert len(results) == 3


class TestCodeExecutionReward:
    """Tests for the main reward function."""

    @pytest.mark.asyncio
    async def test_correct_solution_reward(self):
        response = """
```python
x = int(input())
y = int(input())
print(x + y)
```
"""
        metadata = {
            "test_type": "STDIN",
            "test_cases": [
                {"input": "1\n2", "output": "3"},
                {"input": "5\n5", "output": "10"},
            ],
        }
        reward = await compute_code_execution_reward(response, None, metadata)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_incorrect_solution_reward(self):
        response = """
```python
x = int(input())
y = int(input())
print(x - y)  # Wrong: should be x + y
```
"""
        metadata = {
            "test_type": "STDIN",
            "test_cases": [
                {"input": "1\n2", "output": "3"},
            ],
        }
        reward = await compute_code_execution_reward(response, None, metadata)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_no_code_reward(self):
        response = "I don't know how to solve this."
        metadata = {
            "test_type": "STDIN",
            "test_cases": [{"input": "5", "output": "10"}],
        }
        reward = await compute_code_execution_reward(response, None, metadata)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_no_test_cases_reward(self):
        response = """
```python
print("hello")
```
"""
        metadata = {"test_type": "STDIN"}
        reward = await compute_code_execution_reward(response, None, metadata)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_partial_pass_reward(self):
        response = """
```python
x = int(input())
if x < 5:
    print(x * 2)
else:
    print(x)  # Wrong for larger inputs
```
"""
        metadata = {
            "test_type": "STDIN",
            "test_cases": [
                {"input": "2", "output": "4"},  # Pass
                {"input": "10", "output": "20"},  # Fail
            ],
        }
        reward = await compute_code_execution_reward(response, None, metadata)
        assert reward == 0.0  # Binary: 0 if any test fails


def test_manual_integration():
    """Manual integration test that can be run without pytest-asyncio."""

    async def run_test():
        response = """Here's my solution:

```python
n = int(input())
total = 0
for i in range(n):
    x = int(input())
    total += x
print(total)
```
"""
        metadata = {
            "test_type": "STDIN",
            "test_cases": [
                {"input": "3\n1\n2\n3", "output": "6"},
                {"input": "5\n1\n1\n1\n1\n1", "output": "5"},
            ],
        }
        reward = await compute_code_execution_reward(response, None, metadata)
        print(f"Reward: {reward}")
        assert reward == 1.0, f"Expected 1.0, got {reward}"
        print("Integration test passed!")

    asyncio.run(run_test())


if __name__ == "__main__":
    test_manual_integration()
