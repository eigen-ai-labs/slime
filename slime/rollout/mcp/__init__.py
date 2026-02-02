"""
MCP (Model Context Protocol) integration module.

This module provides tools for connecting to MCP servers and executing
tool calls in an agent rollout workflow.

Note: This module requires the `mcp` package to be installed.
Install with: pip install "mcp[cli]"
"""

from .protocols import (
    MCPAgentConfig,
    MCPClientConfig,
    MCPTool,
    MCPTransport,
    ToolCall,
    ToolResult,
)

# Lazy imports for modules that require the mcp package
def __getattr__(name: str):
    """Lazy import for mcp-dependent modules."""
    if name == "MCPClient":
        from .client import MCPClient
        return MCPClient
    elif name == "MCPState":
        from .state import MCPState
        return MCPState
    elif name == "get_mcp_state":
        from .state import get_mcp_state
        return get_mcp_state
    elif name == "reset_mcp_state":
        from .state import reset_mcp_state
        return reset_mcp_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Client (lazy loaded)
    "MCPClient",
    # Protocols
    "MCPAgentConfig",
    "MCPClientConfig",
    "MCPTool",
    "MCPTransport",
    "ToolCall",
    "ToolResult",
    # State (lazy loaded)
    "MCPState",
    "get_mcp_state",
    "reset_mcp_state",
]
