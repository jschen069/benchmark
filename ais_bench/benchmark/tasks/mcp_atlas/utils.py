"""Shared utilities for the MCP-Atlas benchmark task.

Mirrors the logic originally in evalscope's
:file:`benchmarks/mcp_atlas/utils.py`, adapted for aisbench conventions.
"""

import ast
import json
import os
import os.path as osp
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "ScaleAI/MCP-Atlas"
DEFAULT_MCP_SERVER_URL = "http://localhost:1984"
DEFAULT_SYSTEM_PROMPT = (
    "Role: You are a factual, tool-aware assistant connected to a variety "
    "of tools. Use the available tools to answer the user query. Do not ask "
    "the user for clarification; fully complete the task using the "
    "information provided in the prompt."
)

MAX_TOOL_ERROR_CHARS = 1000

# Tool-name -> server-name special mappings
_TOOL_SERVER_MAP: Dict[str, str] = {
    "brave_brave_web_search": "brave-search",
    "MongoDB_aggregate": "mongodb",
    "MongoDB_collection-schema": "mongodb",
    "MongoDB_count": "mongodb",
    "MongoDB_find": "mongodb",
    "MongoDB_list-collections": "mongodb",
    "MongoDB_list-databases": "mongodb",
}

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPAtlasServerUnavailable(Exception):
    """Transport-level failure from a backing MCP server."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.server_name = _tool_name_to_server(tool_name)
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# HTTP client for the MCP-Atlas agent-environment
# ---------------------------------------------------------------------------


class MCPAtlasClient:
    """Small HTTP client for the MCP-Atlas agent-environment service.

    The agent-environment is a Docker service that exposes three endpoints::

        GET  /enabled-servers   -> list of currently-operational MCP servers
        POST /list-tools        -> full tool catalogue (all servers)
        POST /call-tool         -> execute a single tool call
    """

    def __init__(
        self,
        base_url: str,
        request_timeout: float,
        list_tools_timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.list_tools_timeout = list_tools_timeout

    # -- enabled-servers ---------------------------------------------------

    def enabled_servers(self) -> List[str]:
        response = requests.get(
            f"{self.base_url}/enabled-servers",
            timeout=self.list_tools_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return _parse_enabled_servers_response(data)
        raise ValueError(
            f"Unexpected /enabled-servers response: {type(data).__name__}"
        )

    # -- list-tools --------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        response = requests.post(
            f"{self.base_url}/list-tools",
            headers={"Content-Type": "application/json"},
            timeout=self.list_tools_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError(
                f"Unexpected /list-tools response: {type(data).__name__}"
            )
        return [tool for tool in data if isinstance(tool, dict)]

    # -- call-tool ---------------------------------------------------------

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/call-tool",
                json={"tool_name": tool_name, "tool_args": tool_args},
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise MCPAtlasServerUnavailable(tool_name, str(exc)) from exc

        if response.status_code != 200:
            if _is_transport_error(response.text):
                raise MCPAtlasServerUnavailable(
                    tool_name, _truncate_text(response.text)
                )
            return (
                f"Error calling tool {tool_name} "
                f"(HTTP {response.status_code}): "
                f"{_truncate_text(response.text)}"
            )

        try:
            return _format_tool_response(response.json())
        except ValueError as exc:
            return (
                f"Error decoding tool response JSON from {tool_name}: "
                f"{exc}. Raw response: {_truncate_text(response.text)}"
            )


# ---------------------------------------------------------------------------
# Tool info conversion (shared with datasets module)
# ---------------------------------------------------------------------------


def mcp_tool_to_tool_info(raw_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw MCP tool descriptor into an OpenAI-style tool definition.

    This mirrors ``mcp_tool_to_tool_info`` from evalscope's
    :file:`benchmarks/mcp_atlas/utils.py`, adapted to return a plain
    dictionary instead of an evalscope ``ToolInfo``.

    Args:
        raw_tool: A dictionary as returned by the MCP-Atlas
            ``/list-tools`` endpoint.  Expected keys: ``name``,
            ``description``, ``inputSchema`` (or ``input_schema``).

    Returns:
        An OpenAI-style tool definition dict with ``type``, ``function``
        (name, description, parameters).
    """
    schema = (
        raw_tool.get("inputSchema")
        or raw_tool.get("input_schema")
        or {}
    )
    if not isinstance(schema, dict):
        schema = {}
    return {
        "type": "function",
        "function": {
            "name": str(raw_tool["name"]),
            "description": str(raw_tool.get("description") or raw_tool["name"]),
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        },
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_enabled_servers_response(data: Dict[str, Any]) -> List[str]:
    if "servers" in data:
        servers = data["servers"]
        if isinstance(servers, dict):
            return [str(k) for k, v in servers.items() if v == "OK"]
        if isinstance(servers, list):
            return [
                str(s[0]) for s in servers
                if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1] == "OK"
            ]
    enabled = data.get("enabled_servers", [])
    if isinstance(enabled, list):
        return [str(x) for x in enabled]
    return []


def _parse_enabled_tools(value: Any) -> List[str]:
    parsed = _maybe_parse_json(value, default=[])
    if not isinstance(parsed, list):
        return []
    seen: set = set()
    tools: List[str] = []
    for item in parsed:
        if isinstance(item, dict):
            name = item.get("name")
        elif isinstance(item, str):
            name = item
        else:
            name = None
        if name and name not in seen:
            seen.add(name)
            tools.append(str(name))
    return tools


def _extract_claims(value: Any) -> List[str]:
    parsed = _maybe_parse_json(value, default=value)
    if isinstance(parsed, list):
        claims: List[str] = []
        for item in parsed:
            if isinstance(item, dict) and item.get("claim"):
                claims.append(str(item["claim"]).strip())
            elif isinstance(item, str):
                nested = _maybe_parse_json(item, default=None)
                if isinstance(nested, list):
                    claims.extend(_extract_claims(nested))
                else:
                    claims.append(item.strip())
            elif item is not None:
                claims.append(str(item).strip())
        return [c for c in claims if c]
    if not isinstance(parsed, str):
        return []
    text = parsed.strip()
    if not text:
        return []
    if "\n" in text:
        return [
            _clean_claim_text(line)
            for line in text.splitlines()
            if _clean_claim_text(line)
        ]
    return [text]


def _clean_claim_text(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    return cleaned.strip("\"'")


def _extract_required_servers(trajectory: Any) -> List[str]:
    parsed = _maybe_parse_json(trajectory, default=[])
    if not isinstance(parsed, list):
        return []
    servers: set = set()
    for message in parsed:
        if not isinstance(message, dict):
            continue
        for tc in message.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if isinstance(fn, dict) and fn.get("name"):
                servers.add(_tool_name_to_server(str(fn["name"])))
    return sorted(servers)


def _tool_name_to_server(tool_name: str) -> str:
    return _TOOL_SERVER_MAP.get(tool_name, tool_name).split("_")[0]


def _parse_claim_judge_response(response: Any) -> Tuple[str, str, float]:
    if not isinstance(response, str):
        return "not_fulfilled", "Judge response was not text.", 0.0
    text = _strip_json_fence(response)
    try:
        parsed = json.loads(text)
        outcome = str(
            parsed.get("coverage_outcome")
            or parsed.get("outcome")
            or "not_fulfilled"
        )
        justification = str(
            parsed.get("justification") or parsed.get("reason") or ""
        )
        confidence = _parse_confidence(
            parsed.get("confidence_level", parsed.get("confidence", 0.0))
        )
        return outcome, justification, confidence
    except Exception:
        lowered = text.lower()
        for outcome in ["partially_fulfilled", "not_fulfilled", "fulfilled"]:
            if outcome in lowered:
                return outcome, text, 0.0
    return "not_fulfilled", text or "Unable to parse judge response.", 0.0


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    lowered = value.strip().lower()
    if lowered in {"high", "high confidence"}:
        return 1.0
    if lowered in {"medium", "moderate", "medium confidence", "moderate confidence"}:
        return 0.5
    if lowered in {"low", "low confidence"}:
        return 0.0
    try:
        return float(lowered)
    except ValueError:
        return 0.0


def _is_transport_error(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "econnrefused", "etimedout", "enotfound", "eai_again",
        "connection refused", "connection timed out",
        "connect timeout", "read timeout", "timed out",
        "failed to establish a new connection",
    ]
    return any(m in lowered for m in markers)


def _truncate_text(text: str, limit: int = MAX_TOOL_ERROR_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _server_unavailable_message(server_name: str, message: str) -> str:
    return (
        f"MCP server '{server_name}' is unavailable for this sample due "
        f"to a transport error. Skipping further calls to this server in "
        f"the same sample. Error: {message}"
    )


def _claim_judge_prompt(claim: str, response: str) -> str:
    return (
        "You are evaluating how well a model's response addresses a "
        "specific expert-defined claim. Treat the claim as the authoritative "
        "reference. Do not use external knowledge to dispute or correct the "
        "claim. Judge only whether the model response explicitly states, "
        "entails, or omits the claim. Return JSON with keys: claim_text, "
        "coverage_outcome, justification, confidence_level. "
        "coverage_outcome must be one of: fulfilled, partially_fulfilled, "
        f"not_fulfilled.\n\nCLAIM TO EVALUATE:\n{claim}\n\n"
        f"MODEL RESPONSE TO ANALYZE:\n{response}"
    )


def _format_tool_response(value: Any) -> str:
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _maybe_parse_json(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def _field(record: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


# ---------------------------------------------------------------------------
# Subprocess-based tool executor
# ---------------------------------------------------------------------------

# Inline script that runs an MCP tool call in a subprocess for isolation.
# Data is passed via stdin as JSON: {"base_url", "tool_name", "tool_args", "timeout"}.
# Result is printed to stdout as JSON: {"ok": true, "data": ...} or {"ok": false, "error": ...}.
_TOOL_CALL_SCRIPT = """\
import json, sys
try:
    import requests  # noqa: E402
except ImportError:
    print(json.dumps({"ok": False, "error": "requests module not available in subprocess"}))
    sys.exit(1)

try:
    data = json.loads(sys.stdin.read())
    r = requests.post(
        f"{data['base_url']}/call-tool",
        json={"tool_name": data["tool_name"], "tool_args": data["tool_args"]},
        headers={"Content-Type": "application/json"},
        timeout=data.get("timeout", 60),
    )
    r.raise_for_status()
    result = r.json()
    print(json.dumps({"ok": True, "data": result}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
    sys.exit(1)
"""


def _call_tool_subprocess(
    tool_name: str,
    tool_args: Dict[str, Any],
    base_url: str,
    timeout: float,
) -> str:
    """Execute an MCP tool call in a child process for isolation.

    Each tool call runs in its own Python subprocess, isolating the
    HTTP interaction from the main agent loop.  Data is passed via
    stdin and the result is read from stdout.

    Args:
        tool_name: The MCP tool to call.
        tool_args: Arguments for the tool.
        base_url: MCP-Atlas agent-environment base URL.
        timeout: HTTP request timeout in seconds.

    Returns:
        The formatted tool response string.

    Raises:
        MCPAtlasServerUnavailable: If the subprocess fails with a
            transport-level error.
    """
    input_data = json.dumps({
        "base_url": base_url.rstrip("/"),
        "tool_name": tool_name,
        "tool_args": tool_args,
        "timeout": timeout,
    })

    try:
        result = subprocess.run(
            [sys.executable, "-c", _TOOL_CALL_SCRIPT],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
    except subprocess.TimeoutExpired:
        raise MCPAtlasServerUnavailable(
            tool_name,
            f"Subprocess timed out after {timeout + 30}s",
        )

    if result.returncode != 0 or not result.stdout.strip():
        error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown subprocess error"
        try:
            parsed = json.loads(result.stdout or "{}")
            error_msg = parsed.get("error", error_msg)
        except json.JSONDecodeError:
            pass
        raise MCPAtlasServerUnavailable(tool_name, error_msg)

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        return _format_tool_response(result.stdout.strip())

    if not response.get("ok"):
        error_msg = response.get("error", "Unknown tool error")
        raise MCPAtlasServerUnavailable(tool_name, error_msg)

    return _format_tool_response(response.get("data", response))
