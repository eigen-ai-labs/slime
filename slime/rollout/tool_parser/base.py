"""
Abstract base class for tool call parsers.

Tool call parsers extract structured tool calls from LLM-generated text.
Different models may use different formats (e.g., Qwen, OpenAI function calling).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from slime.rollout.mcp.protocols import ToolCall


@dataclass
class ParseResult:
    """Result of parsing LLM output for tool calls.

    Attributes:
        tool_calls: List of extracted tool calls
        thinking: Optional thinking/reasoning text extracted from the output
        content: The remaining content after removing tool calls and thinking
        raw_text: The original unparsed text
        has_tool_calls: Whether any tool calls were found
    """

    tool_calls: list[ToolCall]
    thinking: str | None = None
    content: str = ""
    raw_text: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class ToolCallParser(ABC):
    """Abstract base class for parsing tool calls from LLM output.

    Subclasses should implement the `parse` method to extract tool calls
    according to the specific format used by their target model.
    """

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """Parse LLM output and extract tool calls.

        Args:
            text: The raw text output from the LLM

        Returns:
            ParseResult containing extracted tool calls and metadata
        """
        pass

    @abstractmethod
    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tools for inclusion in the system prompt.

        Args:
            tools: List of tools in OpenAI function format

        Returns:
            Formatted string to include in the prompt
        """
        pass

    def get_stop_sequences(self) -> list[str]:
        """Get stop sequences that indicate end of tool call block.

        Override this method if your format uses specific stop sequences.

        Returns:
            List of stop sequences
        """
        return []
