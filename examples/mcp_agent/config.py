"""
Example MCP Agent Configuration.

This file demonstrates how to configure MCP servers for the agent rollout.
The configuration function is passed to the training script via
`--mcp_server_config_path`.

Usage:
    python -m slime.train \
        --rollout_function_path="slime.rollout.mcp_agent_rollout:generate_mcp_rollout" \
        --mcp_server_config_path="examples.mcp_agent.config:mcp_server_config_fn" \
        --mcp_max_steps=5
"""

from __future__ import annotations

from slime.rollout.mcp import MCPClientConfig, MCPTransport


def mcp_server_config_fn() -> list[MCPClientConfig]:
    """Return a list of MCP server configurations.

    This function is called during initialization to set up MCP connections.
    Configure your MCP servers here based on your use case.

    Returns:
        List of MCPClientConfig objects defining the MCP servers to connect to.
    """
    return [
        # Example 1: SSE-based MCP server (e.g., web search)
        MCPClientConfig(
            name="WebSearch",
            transport=MCPTransport.SSE,
            url="http://localhost:8007/sse",
            concurrency_limit=16,
            timeout=30.0,
            # Optional: block specific tools
            # blocklist={"dangerous_tool"},
            # Optional: only allow specific tools
            # allowlist={"search", "fetch"},
        ),
        # Example 2: Another SSE server (e.g., code interpreter)
        # MCPClientConfig(
        #     name="CodeInterpreter",
        #     transport=MCPTransport.SSE,
        #     url="http://localhost:8008/sse",
        #     concurrency_limit=8,
        #     timeout=60.0,
        # ),
        # Example 3: Stdio-based MCP server (local process)
        # MCPClientConfig(
        #     name="FileSystem",
        #     transport=MCPTransport.STDIO,
        #     command="npx",
        #     args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"],
        #     concurrency_limit=4,
        #     timeout=30.0,
        # ),
    ]


# Alternative: Simple configuration for testing
def simple_config_fn() -> list[MCPClientConfig]:
    """Minimal configuration for testing."""
    return [
        MCPClientConfig(
            name="TestServer",
            transport=MCPTransport.SSE,
            url="http://localhost:8000/sse",
        ),
    ]


# Example: Multiple servers for complex workflows
def multi_server_config_fn() -> list[MCPClientConfig]:
    """Configuration with multiple specialized servers."""
    return [
        # Search and retrieval
        MCPClientConfig(
            name="Search",
            transport=MCPTransport.SSE,
            url="http://localhost:8001/sse",
            concurrency_limit=16,
        ),
        # Code execution
        MCPClientConfig(
            name="Code",
            transport=MCPTransport.SSE,
            url="http://localhost:8002/sse",
            concurrency_limit=4,
            timeout=120.0,  # Longer timeout for code execution
        ),
        # Database access
        MCPClientConfig(
            name="Database",
            transport=MCPTransport.SSE,
            url="http://localhost:8003/sse",
            concurrency_limit=8,
            blocklist={"drop_table", "delete_all"},  # Block dangerous operations
        ),
    ]
