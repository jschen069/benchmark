"""Adapt MCP server tools into ToolInfo + ToolHandler shapes.

Bridges :class:`MCPServer` (which speaks the official ``mcp`` SDK shapes)
to the handler / info tuples that agent benchmarks consume — so MCP tools
behave identically to benchmark-native tools from the agent's perspective.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from ais_bench.benchmark.mcp.client import MCPServer
from ais_bench.benchmark.mcp.types import (
    MCPServerConfig,
    ToolCall,
    ToolInfo,
    ToolParams,
    ToolParam,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tool_matches_filter(name: str, filter_spec: Any) -> bool:
    """Check whether a tool name passes the server's tool filter.

    Args:
        name: The tool name advertised by the MCP server.
        filter_spec: Either ``'all'`` or a list of allowed tool names.

    Returns:
        ``True`` if the tool should be included.
    """
    if filter_spec == 'all':
        return True
    if isinstance(filter_spec, list):
        return name in filter_spec
    return False


def _to_tool_info(mcp_tool: Any) -> ToolInfo:
    """Convert an ``mcp.types.Tool`` to :class:`ToolInfo`.

    MCP tools advertise their input schema via ``inputSchema`` (raw JSON
    Schema dict).  We coerce it through :class:`ToolParam` and pack into
    :class:`ToolParams`; missing / malformed schemas degrade to the
    no-argument default.

    Args:
        mcp_tool: An MCP Tool object (from ``mcp.types``).

    Returns:
        A :class:`ToolInfo` describing the tool's interface.
    """
    schema = getattr(mcp_tool, 'inputSchema', None) or {}
    properties_dict: Dict[str, Any] = schema.get('properties') or {}
    required_list: List[str] = list(schema.get('required') or [])
    additional = schema.get('additionalProperties', False)

    properties: Dict[str, ToolParam] = {}
    for prop_name, prop_schema in properties_dict.items():
        if isinstance(prop_schema, dict):
            try:
                properties[prop_name] = ToolParam(**prop_schema)
            except Exception as ex:
                logger.debug(
                    'MCP tool %r: dropping unparseable property '
                    '%r (%s); falling back to string type',
                    mcp_tool.name, prop_name, ex,
                )
                properties[prop_name] = ToolParam(type='string')

    params = ToolParams(
        properties=properties,
        required=required_list,
        additionalProperties=bool(additional) if isinstance(additional, bool) else False,
    )

    return ToolInfo(
        name=mcp_tool.name,
        description=(getattr(mcp_tool, 'description', None) or mcp_tool.name),
        parameters=params,
    )


def _make_handler(server: MCPServer, tool_name: str):
    """Build a ``ToolHandler`` that forwards a ``ToolCall`` into ``server.call_tool``.

    The closure captures ``server`` so the handler stays valid as long as
    the caller keeps the server entered (guaranteed via :class:`AsyncExitStack`).

    Args:
        server: An entered :class:`MCPServer` instance.
        tool_name: The name of the MCP tool to invoke.

    Returns:
        An async callable matching the ``ToolHandler`` signature.
    """

    async def _handler(call: ToolCall, env: Optional[Any] = None) -> str:
        del env  # MCP tools don't use the local sandbox environment
        return await server.call_tool(tool_name, call.function.arguments)

    return _handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def mcp_tools(server: MCPServer) -> Tuple[Dict[str, Any], List[ToolInfo]]:
    """Return ``(handlers_map, tool_infos)`` for all (or filtered) MCP tools.

    The returned ``handlers_map`` is keyed by the MCP tool name and maps to
    an async ``ToolHandler``.  The ``tool_infos`` list is the JSON Schema
    description that the agent loop advertises to the model.

    Args:
        server: An entered :class:`MCPServer` instance.

    Returns:
        A tuple of ``(handlers_dict, tool_infos_list)``.
    """
    raw_tools = await server.list_tools()
    handlers: Dict[str, Any] = {}
    infos: List[ToolInfo] = []
    for t in raw_tools:
        if not _tool_matches_filter(t.name, server.config.tools):
            continue
        handlers[t.name] = _make_handler(server, t.name)
        infos.append(_to_tool_info(t))
    return handlers, infos


async def resolve_mcp_tools(
    mcp_configs: List[MCPServerConfig],
    stack: AsyncExitStack,
) -> Tuple[Dict[str, Any], List[ToolInfo]]:
    """Spawn the configured MCP servers for one sample and return their tools.

    Each server is entered into ``stack`` so that enter / exit happen on the
    same anyio task (the sample's loop coroutine). This is what the
    underlying mcp transports require — they wrap an ``anyio.create_task_group``
    whose cancel scope refuses to be exited from a different task.

    Lifetime is per-sample: every sample re-spawns its MCP servers. For
    stdio servers that costs ~0.5-1s of startup; HTTP / SSE transports only
    rebuild an httpx connection (millisecond-level). If that startup cost
    matters, point ``mcp_configs`` at a long-running remote endpoint
    (HTTP / SSE) instead of an on-demand stdio subprocess.

    Args:
        mcp_configs: List of MCP server configurations to spawn.
        stack: An :class:`AsyncExitStack` that manages the server lifetimes.
            Servers are entered into this stack and will be cleaned up when
            the stack is closed.

    Returns:
        A tuple of ``(merged_handlers_dict, merged_tool_infos_list)``
        combining tools from all configured servers.
    """
    merged_handlers: Dict[str, Any] = {}
    merged_tool_infos: List[ToolInfo] = []

    for cfg in mcp_configs:
        server = MCPServer(cfg)
        await stack.enter_async_context(server)
        handlers, infos = await mcp_tools(server)

        for tool_name, handler in handlers.items():
            if tool_name in merged_handlers:
                logger.warning(
                    'MCPServer[%s]: tool %r shadows existing handler; last-write-wins',
                    server.name, tool_name,
                )
            merged_handlers[tool_name] = handler
        merged_tool_infos.extend(infos)

    logger.info(
        'MCP resolved: %d handlers, %d tool infos from %d server(s)',
        len(merged_handlers), len(merged_tool_infos), len(mcp_configs),
    )
    return merged_handlers, merged_tool_infos


__all__ = ['mcp_tools', 'resolve_mcp_tools']
