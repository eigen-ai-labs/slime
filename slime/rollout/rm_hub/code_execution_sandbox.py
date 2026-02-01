"""
Sandbox executor for safe Python code execution.

Provides a secure environment for running untrusted code with:
- Memory limits (via resource.setrlimit)
- CPU time limits
- Execution timeout (via asyncio)
- Temporary directory isolation
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Literal

from .code_execution_utils import TestCase, normalize_output, outputs_match

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single code execution."""

    success: bool
    output: str
    error: str | None = None
    exit_code: int = 0
    timed_out: bool = False
    memory_exceeded: bool = False


class PythonExecutor:
    """
    Safe Python code executor with resource limits.

    Executes code in isolated subprocess with:
    - Memory limit (RLIMIT_AS)
    - CPU time limit (RLIMIT_CPU)
    - Process limit (RLIMIT_NPROC)
    - Execution timeout
    """

    def __init__(
        self,
        timeout: float = 5.0,
        memory_limit_mb: int = 512,
        cpu_limit_seconds: int = 10,
    ):
        """
        Initialize the executor.

        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
            cpu_limit_seconds: Maximum CPU time in seconds
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_seconds = cpu_limit_seconds

    def _create_wrapper_code(
        self,
        code: str,
        test_case: TestCase,
    ) -> str:
        """
        Create wrapper code that executes the user code with proper setup.

        For STDIN mode: Code reads from stdin, writes to stdout
        For FUNCTIONAL mode: Code defines a function, wrapper calls it

        Args:
            code: User code to execute
            test_case: Test case containing input and expected output

        Returns:
            Complete code string ready for execution
        """
        memory_limit_bytes = self.memory_limit_mb * 1024 * 1024
        cpu_limit = self.cpu_limit_seconds

        if test_case.test_type == "FUNCTIONAL" and test_case.function_name:
            # FUNCTIONAL mode: call the function with parsed input
            wrapper = f'''
import sys
import resource
import json

# Set resource limits
try:
    resource.setrlimit(resource.RLIMIT_AS, ({memory_limit_bytes}, {memory_limit_bytes}))
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # Prevent fork bombs
except Exception:
    pass

# User code
{code}

# Execute function
if __name__ == "__main__":
    import ast

    input_data = sys.stdin.read().strip()

    # Try to parse input as Python literal (tuple, list, etc.)
    try:
        args = ast.literal_eval(input_data)
        if not isinstance(args, tuple):
            args = (args,)
    except Exception:
        # If parsing fails, treat as single string argument
        args = (input_data,)

    # Call the function
    result = {test_case.function_name}(*args)
    print(result)
'''
        else:
            # STDIN mode: code reads from stdin directly
            wrapper = f'''
import sys
import resource

# Set resource limits
try:
    resource.setrlimit(resource.RLIMIT_AS, ({memory_limit_bytes}, {memory_limit_bytes}))
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # Prevent fork bombs
except Exception:
    pass

# User code
{code}
'''
        return wrapper

    async def run_test(self, code: str, test_case: TestCase) -> ExecutionResult:
        """
        Run a single test case against the code.

        Args:
            code: Python code to execute
            test_case: Test case with input and expected output

        Returns:
            ExecutionResult indicating success/failure
        """
        # Create temporary directory for isolation
        temp_dir = tempfile.mkdtemp(prefix="code_exec_")

        try:
            # Create the execution script
            script_path = os.path.join(temp_dir, "solution.py")
            wrapper_code = self._create_wrapper_code(code, test_case)

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(wrapper_code)

            # Prepare environment
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            # Run the code
            result = await self._execute_subprocess(
                script_path=script_path,
                stdin_data=test_case.input,
                env=env,
                cwd=temp_dir,
            )

            # Check if output matches expected
            if result.success:
                if outputs_match(result.output, test_case.expected_output):
                    return result
                else:
                    return ExecutionResult(
                        success=False,
                        output=result.output,
                        error=f"Output mismatch. Expected:\n{test_case.expected_output}\n\nGot:\n{result.output}",
                        exit_code=result.exit_code,
                    )

            return result

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    async def _execute_subprocess(
        self,
        script_path: str,
        stdin_data: str,
        env: dict,
        cwd: str,
    ) -> ExecutionResult:
        """
        Execute the Python script in a subprocess.

        Args:
            script_path: Path to the Python script
            stdin_data: Data to pass via stdin
            env: Environment variables
            cwd: Working directory

        Returns:
            ExecutionResult with output or error information
        """
        try:
            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_data.encode("utf-8")),
                    timeout=self.timeout,
                )

                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")

                if process.returncode == 0:
                    return ExecutionResult(
                        success=True,
                        output=normalize_output(stdout_str),
                        error=stderr_str if stderr_str else None,
                        exit_code=0,
                    )
                else:
                    # Check for specific error types
                    error_msg = stderr_str or stdout_str or "Unknown error"

                    if "MemoryError" in error_msg or "Cannot allocate memory" in error_msg:
                        return ExecutionResult(
                            success=False,
                            output=stdout_str,
                            error="Memory limit exceeded",
                            exit_code=process.returncode,
                            memory_exceeded=True,
                        )

                    return ExecutionResult(
                        success=False,
                        output=stdout_str,
                        error=error_msg,
                        exit_code=process.returncode,
                    )

            except asyncio.TimeoutError:
                # Kill the process
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass

                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout} seconds",
                    exit_code=-1,
                    timed_out=True,
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Failed to execute code: {str(e)}",
                exit_code=-1,
            )

    async def run_all_tests(
        self,
        code: str,
        test_cases: list[TestCase],
        fail_fast: bool = True,
    ) -> tuple[bool, list[ExecutionResult]]:
        """
        Run all test cases against the code.

        Args:
            code: Python code to execute
            test_cases: List of test cases
            fail_fast: If True, stop after first failure

        Returns:
            Tuple of (all_passed, list of results)
        """
        results: list[ExecutionResult] = []
        all_passed = True

        for i, test_case in enumerate(test_cases):
            result = await self.run_test(code, test_case)
            results.append(result)

            if not result.success:
                all_passed = False
                logger.debug(
                    "Test case %d/%d failed: %s",
                    i + 1,
                    len(test_cases),
                    result.error or "Unknown error",
                )
                if fail_fast:
                    break
            else:
                logger.debug("Test case %d/%d passed", i + 1, len(test_cases))

        return all_passed, results
