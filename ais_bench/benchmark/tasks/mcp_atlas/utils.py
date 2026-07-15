"""Shared utilities for the MCP-Atlas benchmark task.

Implements the core logic ported from the upstream mcp-atlas project
(agent-harness TypeScript + score_claims.py), adapted for aisbench
conventions.

Key components:
- MCPAtlasClient: HTTP client for the agent-environment Docker service
- Prompt construction helpers (system prompt, message building)
- Tool detection and conversion (MCP tool → OpenAI format)
- LLM judge prompt and response parsing (claim-coverage scoring)
- Context window management (compaction, tool output capping)
- Subprocess-based tool executor for isolation
"""

import ast
import json
import math
import os
import os.path as osp
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MCP_SERVER_URL = "http://localhost:1984"
DEFAULT_SYSTEM_PROMPT = (
    "Role: You are a factual, tool-aware assistant connected to a variety "
    "of tools. Use the available tools to answer the user query. Do not ask "
    "the user for clarification; fully complete the task using the "
    "information provided in the prompt."
)

MAX_TOOL_ERROR_CHARS = 1000

# Context compaction parameters (ported from mcp-atlas agent-eval.ts)
COMPACT_KEEP_FULL_TURNS = 2
COMPACT_TRUNCATE_THRESHOLD = 1500

# LLM retry parameters (ported from mcp-atlas litellm-strategy.ts)
MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_DELAY = 10  # seconds

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


class MCPToolCallError(Exception):
    """Error during a single tool call execution.

    In the mcp-atlas agent loop, tool-call failures are fed back to the
    model as tool results so the model can recover.  This exception is
    used internally by the subprocess executor.
    """

    def __init__(self, tool_name: str, error: str) -> None:
        self.tool_name = tool_name
        self.error = error
        super().__init__(error)


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
# Prompt construction (ported from mcp-atlas run_eval.py + agent-eval.ts)
# ---------------------------------------------------------------------------


def build_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build the initial messages list for the agent loop.

    Follows mcp-atlas's pattern: optional system message followed by the
    user prompt.

    Args:
        prompt: The task prompt from the dataset.
        system_prompt: Optional system prompt to prepend.  If ``None`` or
            empty, only the user message is sent.

    Returns:
        A list of OpenAI-format message dicts.
    """
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


# ---------------------------------------------------------------------------
# Tool info conversion (shared with datasets module)
# ---------------------------------------------------------------------------


def mcp_tool_to_tool_info(raw_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw MCP tool descriptor into an OpenAI-style tool definition.

    Ported from mcp-atlas agent-eval.ts ``_transformToolCalls``.
    Sets ``strict: false`` to match the upstream behavior (mcp-atlas uses
    ``strict: false`` for broad model compatibility).

    Args:
        raw_tool: A dictionary as returned by the MCP-Atlas
            ``/list-tools`` endpoint.  Expected keys: ``name``,
            ``description``, ``inputSchema`` (or ``input_schema``).

    Returns:
        An OpenAI-style tool definition dict with ``type``, ``function``
        (name, description, parameters, strict).
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
                **schema,  # Pass through full schema (matches upstream)
            },
            "strict": False,  # Match mcp-atlas: strict=False for compatibility
        },
    }


def pruned_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply tool-call specific pruning.

    Ported from mcp-atlas agent-eval.ts ``prunedTools``.  Currently only
    the met-museum image suppression is ported; additional pruners can be
    added here.
    """
    if tool_name == "met-museum_get-museum-object":
        tool_args = dict(tool_args)
        tool_args["returnImage"] = False
    return tool_args


# ---------------------------------------------------------------------------
# Text-format tool call parsing (fallback for models that emit tool calls
# as XML/text in the content field instead of structured tool_calls array)
# ---------------------------------------------------------------------------

# Regex pattern for qwen3.6-style XML tool calls:
# <function=tool_name>\n<parameter=key>\nvalue\n</parameter>\n...\n</function>
_TEXT_TOOL_CALL_RE = re.compile(
    r"<function=([^>]+)>\s*\n(.*?)\n?</function>",
    re.DOTALL,
)
_TEXT_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>\s*\n(.*?)\n?</parameter>",
    re.DOTALL,
)


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse text-format tool calls from model content string.

    Handles the qwen3.6 XML-style format::

        <function=tool_name>
        <parameter=key1>
        value1
        </parameter>
        <parameter=key2>
        value2
        </parameter>
        </function>

    Each parsed tool call is returned in the same structure as
    OpenAI structured ``tool_calls``, so the agent loop can process
    them identically.

    Args:
        content: The model's text output containing tool calls.

    Returns:
        A list of tool call dicts with ``id``, ``type``, ``function``
        keys.  Returns an empty list if no tool calls are found.
    """
    if not content or not isinstance(content, str):
        return []

    results: List[Dict[str, Any]] = []
    for match in _TEXT_TOOL_CALL_RE.finditer(content):
        tool_name = match.group(1).strip()
        body = match.group(2)

        if not tool_name:
            continue

        # Extract parameters
        params: Dict[str, str] = {}
        for param_match in _TEXT_PARAM_RE.finditer(body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            if param_name:
                params[param_name] = param_value

        # Generate a unique ID for the text-based tool call
        tc_id = f"text_tc_{uuid.uuid4().hex[:8]}"

        results.append({
            "id": tc_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(params, ensure_ascii=False) if params else "{}",
            },
        })

    return results


def has_text_tool_calls(content: str) -> bool:
    """Quick check: does the text contain XML-style tool call markers?"""
    if not content or not isinstance(content, str):
        return False
    return "<function=" in content and "</function>" in content


# ---------------------------------------------------------------------------
# Tool call detection (ported from mcp-atlas agent-eval.ts)
# ---------------------------------------------------------------------------


def detect_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect and validate tool calls in an assistant message.

    Ported from mcp-atlas's Zod schema ``AssistantMessageSchema`` which
    validates:
    - ``role == "assistant"``
    - ``tool_calls`` is a non-empty list
    - Each tool call has ``id``, ``type == "function"``, ``function``
      with ``name`` and ``arguments``

    Args:
        message: The assistant message dict from the LLM response.

    Returns:
        A list of validated tool call dicts.  Returns an empty list if:
        - The message is not from the assistant
        - ``tool_calls`` is missing, None, or empty
        - Tool calls don't have the expected structure
    """
    if not message or message.get("role") != "assistant":
        return []

    tool_calls = message.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        tool_calls = []

    validated: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        # Must have the basic structure expected by mcp-atlas
        tc_id = tc.get("id")
        func = tc.get("function")
        if not tc_id or not isinstance(func, dict):
            continue
        name = func.get("name")
        arguments = func.get("arguments", "{}")
        if not name:
            continue
        validated.append({
            "id": str(tc_id),
            "type": "function",
            "function": {
                "name": str(name),
                "arguments": str(arguments) if arguments else "{}",
            },
        })

    # Fallback: if no structured tool_calls found, try parsing text-format
    # tool calls from the content field (e.g., qwen3.6 XML-style calls)
    if not validated:
        content = message.get("content", "")
        if content and has_text_tool_calls(str(content)):
            validated = parse_text_tool_calls(str(content))

    return validated


def is_tool_call_message(message: Dict[str, Any]) -> bool:
    """Quick check: does this assistant message contain tool calls?

    A thin wrapper around :func:`detect_tool_calls` for use in conditionals.
    """
    return len(detect_tool_calls(message)) > 0


# ---------------------------------------------------------------------------
# Context window management (ported from mcp-atlas agent-eval.ts)
# ---------------------------------------------------------------------------


def cap_tool_content(
    content: str, cap: int
) -> str:
    """Cap tool result content to a maximum number of characters.

    Ported from mcp-atlas agent-eval.ts ``capToolContent``.  When the
    content exceeds the cap, it is truncated with a note indicating the
    original size.

    Args:
        content: The tool result text.
        cap: Maximum characters to keep.

    Returns:
        The possibly truncated content string.
    """
    if len(content) <= cap:
        return content
    truncated = content[:cap]
    truncated += (
        f"\n\n[Tool output truncated to {cap} chars. "
        f"Original was {len(content)} chars.]"
    )
    return truncated


def compact_messages(
    messages: List[Dict[str, Any]], current_turn: int
) -> List[Dict[str, Any]]:
    """Compact messages by truncating old tool results to reduce context size.

    Ported from mcp-atlas agent-eval.ts ``compactMessages``.  Keeps full
    tool results for the last ``COMPACT_KEEP_FULL_TURNS`` turns.  Older
    tool results longer than ``COMPACT_TRUNCATE_THRESHOLD`` characters
    are truncated.

    A "turn" starts at each assistant message that contains tool_calls.

    Args:
        messages: The full conversation history.
        current_turn: The current turn number (1-indexed).

    Returns:
        A possibly compacted copy of the messages list.
    """
    if current_turn <= COMPACT_KEEP_FULL_TURNS:
        return messages

    # Find turn boundaries: each assistant message with tool_calls starts a new turn
    turn_starts: List[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            turn_starts.append(i)

    # Determine which turns to truncate (all except the last COMPACT_KEEP_FULL_TURNS)
    turns_to_truncate = len(turn_starts) - COMPACT_KEEP_FULL_TURNS
    if turns_to_truncate <= 0:
        return messages

    truncate_before_idx = turn_starts[turns_to_truncate]

    result: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if idx >= truncate_before_idx:
            result.append(msg)
            continue
        m = dict(msg)
        if m.get("role") == "tool":
            content = m.get("content", "")
            if isinstance(content, list):
                content_str = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content)

            if len(content_str) > COMPACT_TRUNCATE_THRESHOLD:
                truncated_text = content_str[:COMPACT_TRUNCATE_THRESHOLD]
                truncated_text += (
                    f"\n\n[Tool call output too large, truncated to "
                    f"{COMPACT_TRUNCATE_THRESHOLD} chars. "
                    f"Original was {len(content_str)} chars.]"
                )
                m["content"] = truncated_text
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# LLM retry logic (ported from mcp-atlas litellm-strategy.ts)
# ---------------------------------------------------------------------------


def is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is a transient error worth retrying.

    Ported from mcp-atlas agent-eval.ts retry logic:
    - HTTP 500, 502, 503, 429
    - Connection timeouts
    """
    status = getattr(exc, "response", None)
    if status is not None:
        if hasattr(status, "status_code"):
            status_code = status.status_code
        elif isinstance(status, int):
            status_code = status
        else:
            status_code = None
    else:
        status_code = getattr(exc, "status_code", None)

    if status_code in (500, 502, 503, 429):
        return True

    error_msg = str(exc).lower()
    if "timeout" in error_msg or "econnaborted" in error_msg:
        return True

    return False


def get_retry_delay(exc: Exception, attempt: int) -> float:
    """Calculate retry delay in seconds.

    Ported from mcp-atlas litellm-strategy.ts:
    - 429: exponential backoff, capped at 30s
    - Timeout: 15s fixed
    - Other retryable: 10s fixed
    """
    status = getattr(exc, "response", None)
    if status is not None:
        if hasattr(status, "status_code"):
            status_code = status.status_code
        elif isinstance(status, int):
            status_code = status
        else:
            status_code = None
    else:
        status_code = getattr(exc, "status_code", None)

    if status_code == 429:
        return min(2 ** attempt * 5, 30)

    error_msg = str(exc).lower()
    if "timeout" in error_msg or "econnaborted" in error_msg:
        return 15.0

    return float(LLM_RETRY_BASE_DELAY)


# ---------------------------------------------------------------------------
# LLM judge prompt (ported from mcp-atlas score_claims.py)
# ---------------------------------------------------------------------------


def _claim_judge_prompt(claim: str, response: str) -> str:
    """Generate the LLM judge prompt for evaluating a single claim.

    Ported from mcp-atlas score_claims.py ``_get_single_claim_evaluation_prompt``.
    Includes detailed scoring criteria, numerical comparison guidelines,
    and structured output format requirements.
    """
    return f"""You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning.
Return a JSON object with keys: claim_text, coverage_outcome, justification, confidence_level."""


def get_claim_evaluation_schema() -> Dict[str, Any]:
    """Return the JSON schema for structured claim evaluation output.

    Ported from mcp-atlas score_claims.py ``get_single_claim_evaluation_schema``.
    Used with ``response_format: json_schema`` for models that support it.
    """
    return {
        "type": "object",
        "properties": {
            "claim_text": {"type": "string"},
            "coverage_outcome": {
                "type": "string",
                "enum": ["fulfilled", "partially_fulfilled", "not_fulfilled"],
            },
            "justification": {"type": "string"},
            "confidence_level": {"type": "number"},
        },
        "required": [
            "claim_text", "coverage_outcome",
            "justification", "confidence_level",
        ],
    }


# ---------------------------------------------------------------------------
# Claim extraction utilities (ported from mcp-atlas score_claims.py)
# ---------------------------------------------------------------------------


def _clean_claim_text(text: str) -> str:
    """Clean individual claim text by removing unwanted characters.

    Ported from mcp-atlas score_claims.py ``clean_claim_text``.
    """
    text = text.strip()
    # Remove bullet point markers and numbering
    text = re.sub(r"^[-*•·◦‣⁃]\s*", "", text)
    text = re.sub(r"^\d+[.)]\s*", "", text)
    # Replace Unicode quotes
    text = text.replace("\u201c", '"')
    text = re.sub(r"[\u201d\"]", '"', text)
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    # Replace dashes and ellipsis
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2026", "...")
    # Clean trailing punctuation and quotes
    text = re.sub(r'[.\s]*["\']+$', "", text)
    text = re.sub(r'["\']+\.*$', '', text)
    return text

def extract_claims(claim_blob: Any) -> List[str]:
    """Extract and clean individual claims from various input formats.

    Ported from mcp-atlas score_claims.py ``extract_claims``.  Handles
    lists of strings, JSON-encoded lists, dict objects with ``claim`` key,
    and multi-line text with various separators.

    Args:
        claim_blob: A claim representation (list, string, etc.)

    Returns:
        A list of cleaned claim strings.
    """
    if claim_blob is None:
        return []

    # Already a list
    if isinstance(claim_blob, list):
        cleaned_claims: List[str] = []
        for claim in claim_blob:
            cleaned = _clean_claim_text(str(claim))
            if cleaned and len(cleaned) > 3:
                cleaned_claims.append(cleaned)
        return cleaned_claims

    # Convert to string
    if not isinstance(claim_blob, str):
        claim_blob = str(claim_blob)

    claim_blob = claim_blob.strip()
    if not claim_blob:
        return []

    # Try JSON / Python literal parse
    if claim_blob.startswith("[") and claim_blob.endswith("]"):
        for parse_fn in (json.loads, ast.literal_eval):
            try:
                parsed = parse_fn(claim_blob)
                if isinstance(parsed, list):
                    result: List[str] = []
                    for c in parsed:
                        if isinstance(c, dict) and "claim" in c:
                            c = c["claim"]
                        cleaned = _clean_claim_text(str(c))
                        if cleaned and len(cleaned) > 3:
                            result.append(cleaned)
                    return result
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue

    # Fallback: text splitting
    separators = ["\n•", "\n-", "\n*", "\n1.", "\n2.", ";", "||"]
    for sep in separators:
        if sep in claim_blob:
            parts = claim_blob.split(sep)
            claims = []
            for p in parts:
                cleaned = _clean_claim_text(p)
                if cleaned and len(cleaned) > 3:
                    claims.append(cleaned)
            if claims:
                return claims

    # Split by newlines as last resort
    return [
        _clean_claim_text(line)
        for line in claim_blob.strip().splitlines()
        if _clean_claim_text(line)
    ]


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


# Backward-compatible alias
_extract_claims = extract_claims


def _tool_name_to_server(tool_name: str) -> str:
    return _TOOL_SERVER_MAP.get(tool_name, tool_name).split("_")[0]


def _parse_claim_judge_response(response: Any) -> Tuple[str, str, float]:
    """Parse the LLM judge response into (outcome, justification, confidence).

    Ported and enhanced from mcp-atlas score_claims.py.  Handles:
    - Structured JSON output (from json_schema response_format)
    - JSON wrapped in markdown fences
    - Free-text fallback with keyword scanning
    """
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
        # Validate outcome is one of the three allowed values
        if outcome not in ("fulfilled", "partially_fulfilled", "not_fulfilled"):
            outcome = "not_fulfilled"
        return outcome, justification, confidence
    except Exception:
        # Fallback: try to extract the last JSON object from mixed-format
        # responses (e.g., chain-of-thought text followed by JSON block)
        parsed = _extract_last_json_object(text)
        if parsed is not None:
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
            if outcome not in ("fulfilled", "partially_fulfilled", "not_fulfilled"):
                outcome = "not_fulfilled"
            return outcome, justification, confidence
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


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last JSON object from a mixed-format text response.

    LLM judge models (especially when ``enable_thinking=True``) may produce
    chain-of-thought reasoning followed by a JSON block.  This function
    finds the last ``{...}``-like structure and attempts to parse it as
    JSON.

    Args:
        text: The raw judge response text.

    Returns:
        A parsed dict if a valid JSON object is found, else ``None``.
    """
    if not text:
        return None
    # Find all top-level { ... } blocks by tracking brace depth
    candidates: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
    # Try parsing candidates from last to first
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


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


# ---------------------------------------------------------------------------
# Scoring utilities (ported from mcp-atlas score_claims.py)
# ---------------------------------------------------------------------------


def compute_coverage_score(
    claim_results: List[Dict[str, Any]]
) -> Tuple[float, int, int, int]:
    """Compute coverage score from a list of per-claim results.

    Ported from mcp-atlas score_claims.py ``CoverageEvaluator.evaluate``.

    Args:
        claim_results: List of dicts with ``coverage_outcome`` key.

    Returns:
        (coverage_score, fully_covered, partially_covered, not_covered)
    """
    if not claim_results:
        return 0.0, 0, 0, 0

    score_map = {
        "fulfilled": 1.0,
        "partially_fulfilled": 0.5,
        "not_fulfilled": 0.0,
    }

    total_score = 0.0
    fulfilled = 0
    partial = 0
    not_covered = 0

    for cr in claim_results:
        outcome = cr.get("coverage_outcome", "not_fulfilled")
        score = score_map.get(outcome, 0.0)
        total_score += score
        if score >= 1.0:
            fulfilled += 1
        elif score >= 0.5:
            partial += 1
        else:
            not_covered += 1

    coverage = round(total_score / len(claim_results), 4)
    return coverage, fulfilled, partial, not_covered


def compute_coverage_stats(
    results: List[Dict[str, Any]],
    pass_threshold: float = 0.75,
    model_name: str = "",
    evaluator_model: str = "",
) -> Dict[str, Any]:
    """Compute aggregate coverage statistics.

    Ported from mcp-atlas score_claims.py ``_compute_split_stats`` and
    ``generate_statistics_and_plots``.

    Args:
        results: List of per-sample result dicts with ``coverage_score``.
        pass_threshold: Threshold for pass/fail (default 0.75).
        model_name: Name of the model being evaluated.
        evaluator_model: Name of the judge model.

    Returns:
        A statistics dict.
    """
    scores = [
        r.get("coverage_score", 0.0)
        for r in results
        if r.get("coverage_score") is not None
    ]
    total = len(results)
    valid = len(scores)

    mean = sum(scores) / valid if valid else 0.0
    pass_50 = sum(1 for s in scores if s >= 0.50) / valid if valid else 0.0
    pass_75 = sum(1 for s in scores if s >= 0.75) / valid if valid else 0.0

    return {
        "model_name": model_name,
        "evaluator_model": evaluator_model,
        "total_tasks": total,
        "valid_responses": valid,
        "empty_or_error": total - valid,
        "mean_coverage": round(mean, 4),
        "pass_rate_0.50": round(pass_50 * 100, 2),
        "pass_rate_0.75": round(pass_75 * 100, 2),
        "pass_threshold": pass_threshold,
    }


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
