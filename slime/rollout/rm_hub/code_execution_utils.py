"""
Utility functions for code execution reward model.

Provides:
- TestCase dataclass for representing test cases
- Code extraction from LLM responses
- Test case parsing (including compressed formats from LiveCodeBench)
"""

from __future__ import annotations

import base64
import json
import logging
import re
import zlib
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case for code execution."""

    input: str
    expected_output: str
    test_type: Literal["STDIN", "FUNCTIONAL"] = "STDIN"
    function_name: str | None = None

    def __post_init__(self):
        # Normalize expected output (strip trailing whitespace/newlines)
        self.expected_output = self.expected_output.rstrip()


def extract_code_from_response(response: str) -> str | None:
    """
    Extract Python code from an LLM response.

    Supports multiple formats:
    - ```python ... ``` code blocks
    - ```py ... ``` code blocks
    - ``` ... ``` generic code blocks
    - Raw code (if no code blocks found)

    Args:
        response: The full LLM response text

    Returns:
        Extracted code string, or None if no code found
    """
    if not response or not response.strip():
        return None

    # Try to find Python code block first
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the last code block (usually the final solution)
            code = matches[-1].strip()
            if code:
                return code

    # Try inline code block without language specifier
    inline_pattern = r"```(.*?)```"
    matches = re.findall(inline_pattern, response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code and not code.startswith("python"):
            return code

    # If no code blocks found, check if the response looks like code
    # (contains common Python keywords/patterns)
    code_indicators = ["def ", "class ", "import ", "print(", "return ", "for ", "while ", "if "]
    if any(indicator in response for indicator in code_indicators):
        return response.strip()

    return None


def decode_compressed_test_cases(data: str) -> list[dict]:
    """
    Decode LiveCodeBench-style compressed test cases.

    The format is: base64(zlib(json))

    Args:
        data: Base64-encoded, zlib-compressed JSON string

    Returns:
        List of test case dictionaries
    """
    try:
        # Decode base64
        decoded_bytes = base64.b64decode(data)
        # Decompress zlib
        decompressed = zlib.decompress(decoded_bytes)
        # Parse JSON
        test_cases = json.loads(decompressed.decode("utf-8"))
        return test_cases if isinstance(test_cases, list) else [test_cases]
    except Exception as e:
        logger.warning("Failed to decode compressed test cases: %s", e)
        return []


def parse_test_cases(metadata: dict[str, Any]) -> list[TestCase]:
    """
    Parse test cases from metadata.

    Supports multiple formats:
    1. Direct test_cases list: [{"input": "...", "output": "..."}]
    2. Compressed format (LiveCodeBench): base64+zlib encoded
    3. Separate public/private test cases

    Args:
        metadata: Dictionary containing test case information

    Returns:
        List of TestCase objects
    """
    test_cases: list[TestCase] = []
    test_type = metadata.get("test_type", "STDIN").upper()
    function_name = metadata.get("function_name")

    # Helper function to convert raw test case to TestCase object
    def make_test_case(raw: dict) -> TestCase | None:
        # Handle different key names
        input_val = raw.get("input") or raw.get("stdin") or ""
        output_val = raw.get("output") or raw.get("expected_output") or raw.get("stdout") or ""

        # Convert to string if necessary
        if not isinstance(input_val, str):
            input_val = str(input_val)
        if not isinstance(output_val, str):
            output_val = str(output_val)

        return TestCase(
            input=input_val,
            expected_output=output_val,
            test_type=test_type,  # type: ignore[arg-type]
            function_name=function_name,
        )

    # Try to get test cases from various sources
    raw_test_cases = []

    # 1. Check for direct test_cases field
    if "test_cases" in metadata:
        tc = metadata["test_cases"]
        if isinstance(tc, str):
            # Might be compressed or JSON string
            try:
                # First try to decode as compressed
                raw_test_cases.extend(decode_compressed_test_cases(tc))
            except Exception:
                # Try as plain JSON
                try:
                    parsed = json.loads(tc)
                    if isinstance(parsed, list):
                        raw_test_cases.extend(parsed)
                except Exception:
                    pass
        elif isinstance(tc, list):
            raw_test_cases.extend(tc)

    # 2. Check for private_test_cases (LiveCodeBench format)
    if "private_test_cases" in metadata:
        ptc = metadata["private_test_cases"]
        if isinstance(ptc, str):
            try:
                raw_test_cases.extend(decode_compressed_test_cases(ptc))
            except Exception:
                try:
                    parsed = json.loads(ptc)
                    if isinstance(parsed, list):
                        raw_test_cases.extend(parsed)
                except Exception:
                    pass
        elif isinstance(ptc, list):
            raw_test_cases.extend(ptc)

    # 3. Check for public_test_cases
    if "public_test_cases" in metadata:
        pub_tc = metadata["public_test_cases"]
        if isinstance(pub_tc, str):
            try:
                raw_test_cases.extend(decode_compressed_test_cases(pub_tc))
            except Exception:
                try:
                    parsed = json.loads(pub_tc)
                    if isinstance(parsed, list):
                        raw_test_cases.extend(parsed)
                except Exception:
                    pass
        elif isinstance(pub_tc, list):
            raw_test_cases.extend(pub_tc)

    # Convert raw test cases to TestCase objects
    for raw in raw_test_cases:
        if isinstance(raw, dict):
            tc = make_test_case(raw)
            if tc is not None:
                test_cases.append(tc)

    return test_cases


def prepare_code_for_execution(
    code: str,
    starter_code: str | None = None,
    test_type: str = "STDIN",
) -> str:
    """
    Prepare code for execution by combining with starter code if needed.

    Args:
        code: The extracted code from LLM response
        starter_code: Optional starter/template code
        test_type: Type of test (STDIN or FUNCTIONAL)

    Returns:
        Final code ready for execution
    """
    if not starter_code:
        return code

    # For FUNCTIONAL mode, the starter code usually contains function signature
    # The LLM code should fill in the implementation
    if test_type == "FUNCTIONAL":
        # Check if the code already contains the function definition
        # If not, prepend the starter code
        if "def " in starter_code:
            # Extract function name from starter code
            match = re.search(r"def\s+(\w+)\s*\(", starter_code)
            if match:
                func_name = match.group(1)
                if f"def {func_name}" not in code:
                    # Prepend starter code
                    return f"{starter_code}\n\n{code}"

    return code


def normalize_output(output: str) -> str:
    """
    Normalize output for comparison.

    - Strip trailing whitespace
    - Normalize line endings
    - Handle floating point precision

    Args:
        output: Raw output string

    Returns:
        Normalized output string
    """
    if not output:
        return ""

    # Normalize line endings
    output = output.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace from each line and overall
    lines = [line.rstrip() for line in output.split("\n")]

    # Remove trailing empty lines
    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)


def outputs_match(actual: str, expected: str, tolerance: float = 1e-6) -> bool:
    """
    Check if actual output matches expected output.

    Handles:
    - Exact string match
    - Floating point comparison with tolerance
    - Whitespace normalization

    Args:
        actual: Actual program output
        expected: Expected output
        tolerance: Tolerance for floating point comparison

    Returns:
        True if outputs match, False otherwise
    """
    # Normalize both outputs
    actual_norm = normalize_output(actual)
    expected_norm = normalize_output(expected)

    # Exact match
    if actual_norm == expected_norm:
        return True

    # Try line-by-line comparison with numeric tolerance
    actual_lines = actual_norm.split("\n")
    expected_lines = expected_norm.split("\n")

    if len(actual_lines) != len(expected_lines):
        return False

    for actual_line, expected_line in zip(actual_lines, expected_lines, strict=False):
        if actual_line == expected_line:
            continue

        # Try numeric comparison
        try:
            actual_num = float(actual_line.strip())
            expected_num = float(expected_line.strip())
            if abs(actual_num - expected_num) > tolerance:
                return False
        except ValueError:
            # Not numeric, and strings don't match
            return False

    return True
