"""
Tool call parsing module.

This module provides parsers for extracting tool calls from LLM outputs
in various formats (Qwen, OpenAI, etc.).
"""

from .base import ParseResult, ToolCallParser
from .qwen import QwenToolCallParser, parse_qwen_tool_calls

__all__ = [
    # Base
    "ParseResult",
    "ToolCallParser",
    # Qwen
    "QwenToolCallParser",
    "parse_qwen_tool_calls",
]


def get_parser(parser_type: str = "qwen", **kwargs) -> ToolCallParser:
    """Get a tool call parser by type.

    Args:
        parser_type: Type of parser ("qwen" is currently supported)
        **kwargs: Additional arguments passed to parser constructor

    Returns:
        ToolCallParser instance

    Raises:
        ValueError: If parser_type is not supported
    """
    parsers = {
        "qwen": QwenToolCallParser,
    }

    if parser_type not in parsers:
        raise ValueError(f"Unknown parser type: {parser_type}. " f"Supported types: {list(parsers.keys())}")

    return parsers[parser_type](**kwargs)
