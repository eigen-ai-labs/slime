"""
Global MCP state management (Singleton pattern).

Manages connections to multiple MCP servers and provides unified tool
discovery and routing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from slime.utils.misc import SingletonMeta

from .client import MCPClient
from .protocols import MCPClientConfig, MCPTool, ToolCall, ToolResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MCPState(metaclass=SingletonMeta):
    """Global state manager for MCP connections.

    This singleton manages connections to multiple MCP servers,
    provides unified tool discovery, and routes tool calls to
    the appropriate server.

    Example:
        # Initialize with server configs
        configs = [
            MCPClientConfig(name="Web", transport="sse", url="http://localhost:8007"),
            MCPClientConfig(name="Code", transport="stdio", command="python", args=["-m", "code_mcp"]),
        ]
        state = MCPState(config_fn=lambda: configs)

        # Connect and get tools
        await state.initialize()
        tools = await state.get_all_tools()

        # Execute tool calls
        results = await state.execute_tool_calls([tool_call1, tool_call2])

        # Cleanup
        await state.shutdown()
    """

    def __init__(
        self,
        config_fn: Callable[[], list[MCPClientConfig]] | None = None,
        configs: list[MCPClientConfig] | None = None,
    ) -> None:
        """Initialize MCPState.

        Args:
            config_fn: Function that returns list of MCPClientConfig
            configs: Direct list of configs (alternative to config_fn)
        """
        self._config_fn = config_fn
        self._configs = configs
        self._clients: dict[str, MCPClient] = {}
        self._tool_to_server: dict[str, str] = {}
        self._tools_cache: list[MCPTool] | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _get_configs(self) -> list[MCPClientConfig]:
        """Get MCP server configurations."""
        if self._configs is not None:
            return self._configs
        if self._config_fn is not None:
            return self._config_fn()
        return []

    async def initialize(self) -> None:
        """Initialize all MCP client connections.

        This method is idempotent - calling it multiple times has no effect
        if already initialized.
        """
        async with self._lock:
            if self._initialized:
                logger.debug("MCPState already initialized")
                return

            configs = self._get_configs()
            if not configs:
                logger.warning("No MCP server configs provided")
                self._initialized = True
                return

            logger.info("Initializing MCPState with %d servers", len(configs))

            # Connect to all servers concurrently
            connect_tasks = []
            for config in configs:
                client = MCPClient(config)
                self._clients[config.name] = client
                connect_tasks.append(self._connect_client(client))

            await asyncio.gather(*connect_tasks, return_exceptions=True)

            # Build tool routing table
            await self._build_tool_routing()

            self._initialized = True
            logger.info(
                "MCPState initialized: %d servers, %d tools",
                len(self._clients),
                len(self._tool_to_server),
            )

    async def _connect_client(self, client: MCPClient) -> None:
        """Connect a single client with error handling."""
        try:
            await client.connect()
        except Exception as e:
            logger.error("Failed to connect to %s: %s", client.name, e)

    async def _build_tool_routing(self) -> None:
        """Build the tool name to server mapping."""
        self._tool_to_server.clear()
        self._tools_cache = None

        for name, client in self._clients.items():
            if not client.is_connected:
                continue

            try:
                tools = await client.list_tools()
                for tool in tools:
                    if tool.name in self._tool_to_server:
                        logger.warning(
                            "Duplicate tool name '%s' from servers %s and %s",
                            tool.name,
                            self._tool_to_server[tool.name],
                            name,
                        )
                    self._tool_to_server[tool.name] = name
            except Exception as e:
                logger.error("Failed to list tools from %s: %s", name, e)

    async def get_all_tools(self, force_refresh: bool = False) -> list[MCPTool]:
        """Get all available tools from all connected servers.

        Args:
            force_refresh: If True, refresh tool lists from servers

        Returns:
            List of all available MCPTool objects
        """
        if not self._initialized:
            await self.initialize()

        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache

        if force_refresh:
            await self._build_tool_routing()

        tools = []
        for name, client in self._clients.items():
            if not client.is_connected:
                continue
            try:
                server_tools = await client.list_tools(force_refresh=force_refresh)
                tools.extend(server_tools)
            except Exception as e:
                logger.error("Failed to get tools from %s: %s", name, e)

        self._tools_cache = tools
        return tools

    def get_tools_openai_format(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function calling format.

        This is a synchronous method that uses cached tools.
        Call get_all_tools() first to populate the cache.

        Returns:
            List of tools in OpenAI format
        """
        if self._tools_cache is None:
            return []
        return [tool.to_openai_format() for tool in self._tools_cache]

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with execution result or error
        """
        if not self._initialized:
            await self.initialize()

        server_name = self._tool_to_server.get(tool_call.name)
        if server_name is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Unknown tool '{tool_call.name}'",
                is_error=True,
            )

        client = self._clients.get(server_name)
        if client is None or not client.is_connected:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Server '{server_name}' is not connected",
                is_error=True,
            )

        return await client.call_tool(tool_call)

    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        parallel: bool = True,
    ) -> list[ToolResult]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls to execute
            parallel: If True, execute calls concurrently

        Returns:
            List of ToolResult in the same order as input
        """
        if not tool_calls:
            return []

        if parallel:
            tasks = [self.execute_tool_call(tc) for tc in tool_calls]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for tc in tool_calls:
                result = await self.execute_tool_call(tc)
                results.append(result)
            return results

    async def shutdown(self) -> None:
        """Disconnect all MCP clients and clean up resources."""
        async with self._lock:
            logger.info("Shutting down MCPState")

            disconnect_tasks = []
            for client in self._clients.values():
                disconnect_tasks.append(client.disconnect())

            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

            self._clients.clear()
            self._tool_to_server.clear()
            self._tools_cache = None
            self._initialized = False

    async def __aenter__(self) -> "MCPState":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.shutdown()


# Global state instance (created lazily)
_global_state: MCPState | None = None


def get_mcp_state(
    config_fn: Callable[[], list[MCPClientConfig]] | None = None,
    configs: list[MCPClientConfig] | None = None,
) -> MCPState:
    """Get or create the global MCPState instance.

    Args:
        config_fn: Configuration function (only used on first call)
        configs: Direct configs (only used on first call)

    Returns:
        The global MCPState instance
    """
    global _global_state
    if _global_state is None:
        _global_state = MCPState(config_fn=config_fn, configs=configs)
    return _global_state


async def reset_mcp_state() -> None:
    """Reset the global MCPState instance."""
    global _global_state
    if _global_state is not None:
        await _global_state.shutdown()
        _global_state = None
