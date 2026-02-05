"""
Qwen-style tool call parser.

Parses tool calls from Qwen model format using <tool_call> and <think> tags.

Example format:
    <think>
    I need to search for information about X.
    </think>

    <tool_call>
    {"name": "search", "arguments": {"query": "X"}}
    </tool_call>
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from slime.rollout.mcp.protocols import ToolCall

from .base import ParseResult, ToolCallParser

logger = logging.getLogger(__name__)

# Regex patterns for Qwen format
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Alternative patterns (some models may use slightly different formats)
TOOL_CALL_ALT_PATTERNS = [
    re.compile(r"```tool_call\n(.*?)```", re.DOTALL),
    re.compile(r"```json\n\{\"name\":\s*\"([^\"]+)\".*?\}```", re.DOTALL),
]


class QwenToolCallParser(ToolCallParser):
    """Parser for Qwen-style tool call format.

    Supports:
    - <think>...</think> blocks for reasoning
    - <tool_call>...</tool_call> blocks for tool invocations
    - Multiple tool calls in a single response

    Example:
        parser = QwenToolCallParser()
        result = parser.parse(llm_output)
        if result.has_tool_calls:
            for tc in result.tool_calls:
                print(f"Call {tc.name} with {tc.arguments}")
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialize the parser.

        Args:
            strict: If True, raise errors on malformed tool calls.
                    If False, skip malformed calls and log warnings.
        """
        self.strict = strict

    def parse(self, text: str) -> ParseResult:
        """Parse Qwen-style tool calls from text.

        Args:
            text: Raw LLM output

        Returns:
            ParseResult with extracted tool calls and thinking
        """
        # Extract thinking blocks
        thinking_matches = THINK_PATTERN.findall(text)
        thinking = "\n".join(match.strip() for match in thinking_matches) if thinking_matches else None

        # Extract tool calls
        tool_calls = []
        tool_call_matches = TOOL_CALL_PATTERN.findall(text)

        for match in tool_call_matches:
            try:
                tc = self._parse_tool_call_json(match.strip())
                if tc is not None:
                    tool_calls.append(tc)
            except Exception as e:
                if self.strict:
                    raise ValueError(f"Failed to parse tool call: {match}") from e
                logger.warning("Skipping malformed tool call: %s (error: %s)", match[:100], e)

        # Try alternative patterns if no tool calls found
        if not tool_calls:
            for pattern in TOOL_CALL_ALT_PATTERNS:
                alt_matches = pattern.findall(text)
                for match in alt_matches:
                    try:
                        tc = self._parse_tool_call_json(match.strip())
                        if tc is not None:
                            tool_calls.append(tc)
                    except Exception:
                        continue

        # Extract remaining content (text without think and tool_call blocks)
        content = text
        content = THINK_PATTERN.sub("", content)
        content = TOOL_CALL_PATTERN.sub("", content)
        for pattern in TOOL_CALL_ALT_PATTERNS:
            content = pattern.sub("", content)
        content = content.strip()

        return ParseResult(
            tool_calls=tool_calls,
            thinking=thinking,
            content=content,
            raw_text=text,
        )

    def _parse_tool_call_json(self, json_str: str) -> ToolCall | None:
        """Parse a JSON string into a ToolCall.

        Supports multiple formats:
        - {"name": "tool", "arguments": {...}}
        - {"function": {"name": "tool", "arguments": {...}}}
        - {"tool": "name", "args": {...}}

        Args:
            json_str: JSON string representing a tool call

        Returns:
            ToolCall object or None if parsing fails
        """
        # Clean up the string
        json_str = json_str.strip()

        # Handle potential markdown code blocks
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return None

        # Extract name and arguments from various formats
        name = None
        arguments = {}

        if "name" in data:
            name = data["name"]
            arguments = data.get("arguments", data.get("parameters", {}))
        elif "function" in data:
            func_data = data["function"]
            name = func_data.get("name")
            arguments = func_data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        elif "tool" in data:
            name = data["tool"]
            arguments = data.get("args", data.get("arguments", {}))

        if not name:
            return None

        # Ensure arguments is a dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"input": arguments}

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=arguments,
        )

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tools for Qwen-style system prompt.

        Args:
            tools: List of tools in OpenAI function format

        Returns:
            Formatted string for the system prompt
        """
        if not tools:
            return ""

        lines = ["# Tools", "", "You have access to the following tools:"]

        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            lines.append("")
            lines.append(f"## {name}")
            lines.append(description)
            lines.append("")
            lines.append("Parameters:")
            lines.append("```json")
            lines.append(json.dumps(parameters, indent=2))
            lines.append("```")

        lines.extend(
            [
                "",
                "# Tool Use Format",
                "",
                "To use a tool, output your thinking in <think> tags, then the tool call in <tool_call> tags:",
                "",
                "<think>",
                "Your reasoning about which tool to use and why...",
                "</think>",
                "",
                "<tool_call>",
                '{"name": "tool_name", "arguments": {"param1": "value1"}}',
                "</tool_call>",
                "",
                "You can make multiple tool calls in one response. "
                "After receiving tool results, continue reasoning or provide your final answer.",
            ]
        )

        return "\n".join(lines)

    def get_stop_sequences(self) -> list[str]:
        """Get stop sequences for Qwen tool call format."""
        return ["</tool_call>"]


# Convenience function
def parse_qwen_tool_calls(text: str, strict: bool = False) -> ParseResult:
    """Parse Qwen-style tool calls from text.

    Args:
        text: Raw LLM output
        strict: Whether to raise errors on malformed calls

    Returns:
        ParseResult with extracted tool calls
    """
    parser = QwenToolCallParser(strict=strict)
    return parser.parse(text)
