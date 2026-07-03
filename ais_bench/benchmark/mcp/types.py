"""Configuration types for MCP (Model Context Protocol) integration.

Defines config models for MCP server connections and lightweight Tool/ToolCall
types used to bridge MCP tools into the agent benchmark pipeline.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# MCP Server Config models
# ---------------------------------------------------------------------------

class _MCPServerConfigBase(BaseModel):
    """Fields shared by every MCP server config variant."""

    model_config = ConfigDict(extra='forbid')

    name: Optional[str] = Field(default=None)
    """Human-readable display name (used in logs / traces). Defaults to ``command`` or ``url``."""

    tools: Union[Literal['all'], List[str]] = Field(default='all')
    """Whitelist of tool names exposed to the model. ``'all'`` exports every tool."""


class MCPServerConfigStdio(_MCPServerConfigBase):
    """Spawn a local MCP server as a child process and talk over stdio.

    Mirrors :class:`mcp.client.stdio.StdioServerParameters`.
    """

    type: Literal['stdio'] = Field(default='stdio')

    command: str
    """Executable to spawn (e.g. ``npx`` / ``uvx`` / absolute path)."""

    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = Field(default=None)


class MCPServerConfigHTTP(_MCPServerConfigBase):
    """Connect to a remote MCP server over Streamable HTTP."""

    type: Literal['http'] = Field(default='http')

    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=30.0)
    """HTTP read timeout in seconds."""


class MCPServerConfigSSE(_MCPServerConfigBase):
    """Connect to a remote MCP server over Server-Sent Events (SSE).

    SSE is the legacy MCP HTTP transport; prefer :class:`MCPServerConfigHTTP`
    (Streamable HTTP) when the server supports both.
    """

    type: Literal['sse'] = Field(default='sse')

    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=5.0)
    """HTTP timeout in seconds for non-streaming operations."""
    sse_read_timeout: float = Field(default=300.0)
    """How long (seconds) to wait for the next SSE event before disconnecting."""


MCPServerConfig = Annotated[
    Union[MCPServerConfigStdio, MCPServerConfigHTTP, MCPServerConfigSSE],
    Field(discriminator='type'),
]
"""Discriminated union of MCP server config types."""


# ---------------------------------------------------------------------------
# Lightweight Tool types (self-contained, no dependency on evalscope tool API)
# ---------------------------------------------------------------------------

class ToolParam(BaseModel):
    """A single parameter in a tool's JSON Schema."""

    type: Optional[str] = Field(default=None)
    """JSON type of the parameter (string, integer, number, boolean, array, object, null)."""

    description: Optional[str] = Field(default=None)
    """Parameter description."""

    default: Any = Field(default=None)
    """Default value for the parameter."""

    enum: Optional[List[Any]] = Field(default=None)
    """Valid values for enum parameters."""

    items: Optional[Dict[str, Any]] = Field(default=None)
    """Schema for array items."""

    properties: Optional[Dict[str, 'ToolParam']] = Field(default=None)
    """Schema for object properties."""

    required: Optional[List[str]] = Field(default=None)
    """Required fields within this object parameter."""


class ToolParams(BaseModel):
    """Description of tool parameters object in JSON Schema format."""

    type: Literal['object'] = Field(default='object')
    """Params type (always 'object')."""

    properties: Dict[str, ToolParam] = Field(default_factory=dict)
    """Tool function parameters."""

    required: List[str] = Field(default_factory=list)
    """List of required fields."""

    additionalProperties: bool = Field(default=False)
    """Are additional object properties allowed?"""


@dataclass
class ToolInfo:
    """Specification of a tool (JSON Schema compatible).

    Describes the tool's interface so that an LLM can decide when and how
    to invoke it.
    """

    name: str
    """Name of the tool."""

    description: str
    """Short description of what the tool does."""

    parameters: ToolParams = field(default_factory=ToolParams)
    """JSON Schema of the tool's parameters object."""


class ToolFunction(BaseModel):
    """Indicates that a specific tool function should be called."""

    name: str
    """The name of the tool function to call."""

    arguments: Dict[str, Any] = Field(default_factory=dict)
    """The arguments to pass to the tool function."""


class ToolCall(BaseModel):
    """A request to invoke a tool function."""

    id: str
    """Unique identifier for this tool call."""

    function: ToolFunction
    """The function to call."""

    type: Optional[str] = Field(default=None)
    """Tool call type (deprecated, kept for compatibility)."""


# ---------------------------------------------------------------------------
# ToolHandler type alias
# ---------------------------------------------------------------------------

from typing import Awaitable, Callable

ToolHandler = Callable[..., Awaitable[str]]
"""Async callable that executes a tool and returns a text observation.

Signature: ``async def handler(call: ToolCall, env: Optional[Any]) -> str``
"""


__all__ = [
    'MCPServerConfig',
    'MCPServerConfigStdio',
    'MCPServerConfigHTTP',
    'MCPServerConfigSSE',
    'ToolParam',
    'ToolParams',
    'ToolInfo',
    'ToolFunction',
    'ToolCall',
    'ToolHandler',
]
