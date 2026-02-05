"""
Generic MCP client implementation supporting SSE and Stdio transports.

This module provides a unified interface for connecting to MCP servers
and executing tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .protocols import MCPClientConfig, MCPTool, MCPTransport, ToolCall, ToolResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MCPClient:
    """Generic MCP client supporting both SSE and Stdio transports.

    This client manages the connection lifecycle and provides methods
    for listing tools and calling them.

    Example:
        config = MCPClientConfig(
            name="WebSearch",
            transport="sse",
            url="http://localhost:8007",
        )
        client = MCPClient(config)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "hello"})
        await client.disconnect()
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self.config = config
        self.name = config.name

        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._semaphore = asyncio.Semaphore(config.concurrency_limit)
        self._tools_cache: list[MCPTool] | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        if self._connected:
            logger.warning("Client %s is already connected", self.name)
            return

        self._exit_stack = AsyncExitStack()

        try:
            if self.config.transport == MCPTransport.SSE:
                await self._connect_sse()
            elif self.config.transport == MCPTransport.STREAMABLE_HTTP:
                await self._connect_streamable_http()
            else:
                await self._connect_stdio()

            self._connected = True
            logger.info("Connected to MCP server: %s", self.name)
        except Exception as e:
            logger.error("Failed to connect to MCP server %s: %s", self.name, e)
            await self._cleanup()
            raise

    async def _connect_sse(self) -> None:
        """Connect using SSE transport."""
        assert self.config.url is not None
        assert self._exit_stack is not None

        # SSE client returns (read_stream, write_stream)
        sse_transport = await self._exit_stack.enter_async_context(sse_client(self.config.url))
        read_stream, write_stream = sse_transport

        # Create and initialize session
        self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()

    async def _connect_streamable_http(self) -> None:
        """Connect using Streamable HTTP transport."""
        assert self.config.url is not None
        assert self._exit_stack is not None

        # Streamable HTTP client returns (read_stream, write_stream, get_session_id)
        http_transport = await self._exit_stack.enter_async_context(
            streamablehttp_client(self.config.url, timeout=self.config.timeout)
        )
        read_stream, write_stream, _ = http_transport

        # Create and initialize session
        self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()

    async def _connect_stdio(self) -> None:
        """Connect using Stdio transport."""
        assert self.config.command is not None
        assert self._exit_stack is not None

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env or None,
        )

        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        await self._cleanup()
        self._connected = False
        logger.info("Disconnected from MCP server: %s", self.name)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("Error during cleanup for %s: %s", self.name, e)
            self._exit_stack = None
        self._session = None
        self._tools_cache = None

    async def list_tools(self, force_refresh: bool = False) -> list[MCPTool]:
        """List available tools from this MCP server.

        Args:
            force_refresh: If True, bypass cache and fetch fresh tool list

        Returns:
            List of MCPTool objects available from this server
        """
        if not self.is_connected:
            raise RuntimeError(f"Client {self.name} is not connected")

        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache

        assert self._session is not None

        try:
            response = await self._session.list_tools()
            tools = []

            for tool in response.tools:
                # Apply allowlist/blocklist filtering
                if self.config.allowlist is not None:
                    if tool.name not in self.config.allowlist:
                        continue
                if tool.name in self.config.blocklist:
                    continue

                tools.append(
                    MCPTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                        server_name=self.name,
                    )
                )

            self._tools_cache = tools
            logger.debug("Listed %d tools from %s", len(tools), self.name)
            return tools

        except Exception as e:
            logger.error("Failed to list tools from %s: %s", self.name, e)
            raise

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call on this MCP server.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with the execution result or error
        """
        if not self.is_connected:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Client {self.name} is not connected",
                is_error=True,
            )

        assert self._session is not None

        async with self._semaphore:
            try:
                # Call the tool with timeout
                result = await asyncio.wait_for(
                    self._session.call_tool(
                        tool_call.name,
                        arguments=tool_call.arguments,
                    ),
                    timeout=self.config.timeout,
                )

                # Extract content from result
                content_parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        content_parts.append(item.text)
                    elif hasattr(item, "data"):
                        content_parts.append(json.dumps(item.data))
                    else:
                        content_parts.append(str(item))

                content = "\n".join(content_parts) if content_parts else ""

                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=content,
                    is_error=result.isError if hasattr(result, "isError") else False,
                )

            except asyncio.TimeoutError:
                logger.warning(
                    "Tool call %s timed out after %s seconds",
                    tool_call.name,
                    self.config.timeout,
                )
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error: Tool call timed out after {self.config.timeout}s",
                    is_error=True,
                )
            except Exception as e:
                logger.error("Tool call %s failed: %s", tool_call.name, e)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error: {type(e).__name__}: {str(e)}",
                    is_error=True,
                )

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()
