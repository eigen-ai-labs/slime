"""
Session-aware drop-in replacement for code_execution_server/tools/code_exec.py.

Changes from original:
  - working_dir is resolved at call time via session_context.get_session_fs_root()
    instead of module-level FS_ROOT constant.
  - This allows concurrent sessions to execute code in their own /filesystem/sessions/{sid}/
"""

import os

from loguru import logger
from models.code_exec import (
    CodeExecRequest,
    CodeExecResponse,
)
from session_context import get_session_fs_root
from utils.decorators import make_async_background
from utils.sandbox import (
    DEFAULT_LIBRARY_PATH,
    run_sandboxed_command,
    verify_sandbox_library_available,
)

CODE_EXEC_COMMAND_TIMEOUT = os.getenv("CODE_EXEC_COMMAND_TIMEOUT", "300")
SANDBOX_LIBRARY_PATH = os.getenv("SANDBOX_LIBRARY_PATH", DEFAULT_LIBRARY_PATH)
# Paths to hide from code execution
BLOCKED_PATHS = ["/app", "/.apps_data"]


def verify_sandbox_available() -> None:
    """Verify sandbox library is available. Call at server startup, not import time."""
    verify_sandbox_library_available(SANDBOX_LIBRARY_PATH)


@make_async_background
def code_exec(request: CodeExecRequest) -> CodeExecResponse:
    """Execute shell commands in a sandboxed bash environment."""
    # Reject None code - allow empty string (valid in bash)
    if request.code is None:
        return CodeExecResponse(
            success=False,
            output="Error: Required parameter 'code' (command to execute)",
        )

    # Safety net: detect raw Python code and provide helpful error
    code_stripped = request.code.strip()

    def looks_like_python_import(code: str) -> bool:
        if not code.startswith("import "):
            return False
        rest = code[7:].strip()
        if rest.startswith("-") or "/" in rest.split()[0] if rest else False:
            return False
        first_word = rest.split()[0] if rest else ""
        if "." in first_word and first_word.rsplit(".", 1)[-1].lower() in (
            "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "pdf", "ps", "eps",
        ):
            return False
        return True

    python_indicators = (
        looks_like_python_import(code_stripped),
        code_stripped.startswith("from "),
        code_stripped.startswith("def "),
        code_stripped.startswith("class "),
        code_stripped.startswith("async def "),
        code_stripped.startswith("@"),
    )
    if any(python_indicators):
        return CodeExecResponse(
            success=False,
            output=(
                "Error: It looks like you passed raw Python code. This tool executes shell "
                "commands, not Python directly. To run Python:\n"
                "• One-liner: python -c 'your_code_here'\n"
                "• Multi-line: Write to file first, then run:\n"
                "  cat > script.py << 'EOF'\n"
                "  your_code\n"
                "  EOF && python script.py"
            ),
        )

    try:
        timeout_value = int(CODE_EXEC_COMMAND_TIMEOUT)
    except ValueError:
        error_msg = f"Invalid timeout value: {CODE_EXEC_COMMAND_TIMEOUT}"
        logger.error(error_msg)
        return CodeExecResponse(
            success=False,
            output=f"Configuration error: {error_msg}",
        )

    # Session-aware: resolve working_dir at call time
    fs_root = get_session_fs_root()

    try:
        result = run_sandboxed_command(
            command=request.code,
            timeout=timeout_value,
            working_dir=fs_root,
            blocked_paths=BLOCKED_PATHS,
            library_path=SANDBOX_LIBRARY_PATH,
        )

        if result.timed_out:
            logger.error(f"Command timed out after {timeout_value} seconds")
            return CodeExecResponse(
                success=False,
                output=f"Command execution timed out after {timeout_value} seconds",
            )

        if result.error:
            logger.error(f"Error running command: {result.error}")
            return CodeExecResponse(
                success=False,
                output=f"System error: {result.error}",
            )

        if result.return_code != 0:
            logger.error(f"Command failed with exit code {result.return_code}")
            output = result.stdout if result.stdout else ""
            if result.stderr:
                output += f"\nError output:\n{result.stderr}"
            return CodeExecResponse(
                success=False,
                output=f"{output}\n\nCommand failed with exit code {result.return_code}",
            )

        return CodeExecResponse(
            success=True,
            output=result.stdout,
        )
    except FileNotFoundError:
        error_msg = f"Working directory not found: {fs_root}"
        logger.error(error_msg)
        return CodeExecResponse(
            success=False,
            output=f"Configuration error: {error_msg}",
        )
    except OSError as e:
        error_msg = f"OS error when executing command: {e}"
        logger.error(error_msg)
        return CodeExecResponse(
            success=False,
            output=f"System error: {error_msg}",
        )
