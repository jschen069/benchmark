"""MCP (Model Context Protocol) integration for agent benchmarks.

Lets users configure :class:`MCPServerConfigStdio` / :class:`MCPServerConfigHTTP`
/ :class:`MCPServerConfigSSE` on their agent config's ``mcp_servers`` field and
have those servers' tools auto-merged into the agent loop — without any
benchmark-side code change.

Soft dependency: the official ``mcp`` Python SDK (``pip install mcp``) is
imported lazily inside :class:`MCPServer`, so users who don't configure any
MCP server pay no cost.

Usage::

    from ais_bench.benchmark.mcp import (
        MCPServer,
        MCPServerConfigStdio,
        mcp_tools,
        resolve_mcp_tools,
    )

    # Spawn an MCP server as a child process
    config = MCPServerConfigStdio(
        command='npx',
        args=['-y', '@anthropic/mcp-server-fetch'],
    )

    async with MCPServer(config) as server:
        tools = await server.list_tools()
        # ...

    # Or resolve multiple servers at once
    async with AsyncExitStack() as stack:
        handlers, tool_infos = await resolve_mcp_tools([config], stack)
"""

from ais_bench.benchmark.mcp.client import MCPServer
from ais_bench.benchmark.mcp.source import mcp_tools, resolve_mcp_tools
from ais_bench.benchmark.mcp.types import (
    MCPServerConfig,
    MCPServerConfigHTTP,
    MCPServerConfigSSE,
    MCPServerConfigStdio,
    ToolCall,
    ToolFunction,
    ToolInfo,
    ToolParams,
    ToolParam,
    ToolHandler,
)

__all__ = [
    # Client
    'MCPServer',
    # Source
    'mcp_tools',
    'resolve_mcp_tools',
    # Config types
    'MCPServerConfig',
    'MCPServerConfigStdio',
    'MCPServerConfigHTTP',
    'MCPServerConfigSSE',
    # Tool types
    'ToolParam',
    'ToolParams',
    'ToolInfo',
    'ToolFunction',
    'ToolCall',
    'ToolHandler',
]
