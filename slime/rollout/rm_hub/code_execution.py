"""
Code Execution Reward Model for RL training.

Implements a binary reward based on code execution results:
- 1.0 if all test cases pass
- 0.0 if any test case fails

Supports LiveCodeBench-style evaluation with:
- STDIN mode: Code reads from stdin, outputs to stdout
- FUNCTIONAL mode: Code defines a function, tested with specific inputs
"""

from __future__ import annotations

import logging
from typing import Any

from .code_execution_sandbox import PythonExecutor
from .code_execution_utils import (
    extract_code_from_response,
    parse_test_cases,
    prepare_code_for_execution,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT_PER_TEST = 5.0  # seconds
DEFAULT_MEMORY_LIMIT_MB = 512  # MB


async def compute_code_execution_reward(
    response: str,
    label: Any,  # Not used, kept for API compatibility
    metadata: dict[str, Any] | None = None,
    timeout_per_test: float | None = None,
    memory_limit_mb: int | None = None,
) -> float:
    """
    Compute binary reward based on code execution results.

    Workflow:
    1. Extract code from LLM response
    2. Parse test cases from metadata
    3. Optionally combine with starter code
    4. Execute code against each test case
    5. Return 1.0 if all tests pass, 0.0 otherwise

    Args:
        response: The LLM-generated response containing code
        label: Ground truth label (unused, for API compatibility)
        metadata: Dictionary containing:
            - test_type: "STDIN" or "FUNCTIONAL"
            - test_cases: List of test cases or compressed format
            - starter_code: Optional template code
            - function_name: Function name for FUNCTIONAL mode
            - code_exec_timeout: Override timeout per test
            - code_exec_memory_mb: Override memory limit

    Returns:
        1.0 if all tests pass, 0.0 otherwise
    """
    if metadata is None:
        metadata = {}

    # Get configuration from metadata or use defaults
    timeout = timeout_per_test or metadata.get("code_exec_timeout", DEFAULT_TIMEOUT_PER_TEST)
    memory_mb = memory_limit_mb or metadata.get("code_exec_memory_mb", DEFAULT_MEMORY_LIMIT_MB)

    # Step 1: Extract code from response
    code = extract_code_from_response(response)
    if code is None:
        logger.debug("No code found in response")
        return 0.0

    # Step 2: Parse test cases
    test_cases = parse_test_cases(metadata)
    if not test_cases:
        logger.warning("No test cases found in metadata")
        # If no test cases, we can't evaluate - return 0
        return 0.0

    # Step 3: Prepare code (combine with starter code if needed)
    starter_code = metadata.get("starter_code")
    test_type = metadata.get("test_type", "STDIN")
    code = prepare_code_for_execution(code, starter_code, test_type)

    # Step 4: Execute tests
    executor = PythonExecutor(
        timeout=timeout,
        memory_limit_mb=memory_mb,
    )

    try:
        all_passed, results = await executor.run_all_tests(
            code=code,
            test_cases=test_cases,
            fail_fast=True,  # Stop on first failure for efficiency
        )

        # Log summary
        passed_count = sum(1 for r in results if r.success)
        total_count = len(test_cases)
        logger.debug(
            "Code execution: %d/%d tests passed (evaluated %d)",
            passed_count,
            total_count,
            len(results),
        )

        # Step 5: Return binary reward
        return 1.0 if all_passed else 0.0

    except Exception as e:
        logger.error("Error during code execution: %s", str(e))
        return 0.0


# Alias for consistency with other RM functions
code_execution_reward = compute_code_execution_reward
