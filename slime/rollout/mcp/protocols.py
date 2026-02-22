"""
Protocol definitions for MCP (Model Context Protocol) integration.

Defines dataclasses and type aliases for MCP tool calls, results, and configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPTransport(str, Enum):
    """Supported MCP transport types."""

    SSE = "sse"
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class MCPClientConfig:
    """Configuration for an MCP client connection.

    Attributes:
        name: Human-readable name for this MCP server
        transport: Transport type (sse or stdio)
        url: URL for SSE transport (e.g., "http://localhost:8007")
        command: Command for stdio transport (e.g., "python")
        args: Arguments for stdio transport command
        env: Environment variables for stdio transport
        concurrency_limit: Maximum concurrent tool calls to this server
        timeout: Timeout for tool calls in seconds
        blocklist: Set of tool names to block from this server
        allowlist: Set of tool names to allow (if set, only these are available)
    """

    name: str
    transport: MCPTransport | str = MCPTransport.SSE

    # SSE transport config
    url: str | None = None

    # Stdio transport config
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # HTTP headers (e.g., for X-Session-Id)
    headers: dict[str, str] = field(default_factory=dict)

    # Common config
    concurrency_limit: int = 16
    timeout: float = 30.0
    blocklist: set[str] = field(default_factory=set)
    allowlist: set[str] | None = None

    def __post_init__(self):
        if isinstance(self.transport, str):
            self.transport = MCPTransport(self.transport)

        if self.transport == MCPTransport.SSE and not self.url:
            raise ValueError("SSE transport requires 'url' to be set")
        if self.transport == MCPTransport.STREAMABLE_HTTP and not self.url:
            raise ValueError("Streamable HTTP transport requires 'url' to be set")
        if self.transport == MCPTransport.STDIO and not self.command:
            raise ValueError("Stdio transport requires 'command' to be set")


@dataclass
class MCPTool:
    """Representation of an MCP tool.

    Attributes:
        name: Tool name (unique identifier)
        description: Tool description
        input_schema: JSON Schema for tool input parameters
        server_name: Name of the MCP server providing this tool
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM output.

    Attributes:
        id: Unique identifier for this call
        name: Tool name to invoke
        arguments: Arguments to pass to the tool
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool call.

    Attributes:
        tool_call_id: ID of the tool call this result corresponds to
        name: Tool name that was called
        content: Result content (string or structured data)
        is_error: Whether the execution resulted in an error
    """

    tool_call_id: str
    name: str
    content: str | dict[str, Any]
    is_error: bool = False

    def to_message(self) -> dict[str, Any]:
        """Convert to chat message format."""
        content = self.content if isinstance(self.content, str) else str(self.content)
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": content,
        }


@dataclass
class MCPAgentConfig:
    """Configuration for MCP agent rollout.

    Attributes:
        max_steps: Maximum number of agent steps (tool call rounds)
        system_prompt_template: Template for building system prompt with tools
        tool_call_parser: Parser type for extracting tool calls from LLM output
        parallel_tool_calls: Whether to execute multiple tool calls in parallel
    """

    max_steps: int = 5
    system_prompt_template: str | None = None
    tool_call_parser: str = "qwen"
    parallel_tool_calls: bool = True
