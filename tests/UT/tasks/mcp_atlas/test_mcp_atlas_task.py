"""Comprehensive unit tests for MCP-Atlas task utilities and client.

Tests the core logic ported from the upstream mcp-atlas project:
- Prompt construction (build_messages)
- Tool call detection (detect_tool_calls, is_tool_call_message)
- Context window management (cap_tool_content, compact_messages)
- LLM retry logic (is_retryable_error, get_retry_delay)
- LLM judge prompt and response parsing (_claim_judge_prompt, _parse_claim_judge_response)
- Claim extraction (extract_claims)
- Coverage scoring (compute_coverage_score, compute_coverage_stats)
- MCPAtlasClient
- Subprocess tool executor (_call_tool_subprocess)
- MCPAtlasEvalTask and MCPAtlasInferTask (structure)
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

# ---------------------------------------------------------------------------
# Load the module without triggering the heavy ais_bench import chain
# ---------------------------------------------------------------------------


def _make_fake_module(name):
    """Return a module-like object that supports submodule resolution."""
    mod = type(sys)(name) if hasattr(type(sys), "__module__") else mock.MagicMock()
    return mock.MagicMock()


_mock_registry = mock.MagicMock()
sys.modules["ais_bench.benchmark.registry"] = _mock_registry

_mock_base = mock.MagicMock()
_mock_base_task = type("BaseTask", (), {})
_mock_base.BaseTask = _mock_base_task
sys.modules["ais_bench.benchmark.tasks.base"] = _mock_base

_mock_abbr = mock.MagicMock()
_mock_abbr.task_abbr_from_cfg = lambda x: "test-task"
_mock_abbr.model_abbr_from_cfg = lambda x: "test-model"
_mock_abbr.dataset_abbr_from_cfg = lambda x: "test-dataset"
sys.modules["ais_bench.benchmark.utils.core.abbr"] = _mock_abbr

# ---- logging mocks (must support sub-module lookup) -------------------------


class _FakeLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass


# Create the logging package as a real module-like object
import types as _types


def _make_pkg(name):
    pkg = _types.ModuleType(name)
    pkg.__path__ = []  # makes it recognizable as a package
    pkg.__package__ = name
    sys.modules[name] = pkg
    return pkg


# Ensure ais_bench namespace modules exist
for _ns in ["ais_bench", "ais_bench.benchmark", "ais_bench.benchmark.utils",
            "ais_bench.benchmark.utils.logging",
            "ais_bench.benchmark.utils.config"]:
    if _ns not in sys.modules:
        _make_pkg(_ns)

# Mock config.build module
_mock_config_build = _types.ModuleType("ais_bench.benchmark.utils.config.build")
_mock_config_build.build_dataset_from_cfg = mock.MagicMock()
sys.modules["ais_bench.benchmark.utils.config.build"] = _mock_config_build

_mock_error_codes = _types.ModuleType("ais_bench.benchmark.utils.logging.error_codes")
_mock_error_codes.TINFER_CODES = mock.MagicMock()
sys.modules["ais_bench.benchmark.utils.logging.error_codes"] = _mock_error_codes

_mock_exceptions = _types.ModuleType("ais_bench.benchmark.utils.logging.exceptions")
_mock_exceptions.ParameterValueError = type("ParameterValueError", (Exception,), {})
sys.modules["ais_bench.benchmark.utils.logging.exceptions"] = _mock_exceptions

# Add AISLogger to the logging package
_logging_pkg = sys.modules["ais_bench.benchmark.utils.logging"]
_logging_pkg.AISLogger = lambda: _FakeLogger()

# Load our modules directly
import importlib.util

# Go up 4 levels from tests/UT/tasks/mcp_atlas/ -> benchmark/
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__)))))

# Load utils module first (no dependencies)
_utils_path = os.path.join(
    _proj_root, "ais_bench", "benchmark", "tasks",
    "mcp_atlas", "utils.py",
)
_utils_spec = importlib.util.spec_from_file_location(
    "ais_bench.benchmark.tasks.mcp_atlas.utils", _utils_path
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["ais_bench.benchmark.tasks.mcp_atlas.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

# Load eval module
_eval_path = os.path.join(
    _proj_root, "ais_bench", "benchmark", "tasks",
    "mcp_atlas", "mcp_atlas_eval.py",
)
_eval_spec = importlib.util.spec_from_file_location(
    "ais_bench.benchmark.tasks.mcp_atlas.mcp_atlas_eval", _eval_path
)
m = importlib.util.module_from_spec(_eval_spec)
# Patch in the registries that get called during exec
_mock_registry.TASKS = mock.MagicMock()
sys.modules["ais_bench.benchmark.tasks.mcp_atlas.mcp_atlas_eval"] = m
_eval_spec.loader.exec_module(m)

# Load infer module
_infer_path = os.path.join(
    _proj_root, "ais_bench", "benchmark", "tasks",
    "mcp_atlas", "mcp_atlas_infer.py",
)
_infer_spec = importlib.util.spec_from_file_location(
    "ais_bench.benchmark.tasks.mcp_atlas.mcp_atlas_infer", _infer_path
)
_infer_mod = importlib.util.module_from_spec(_infer_spec)
sys.modules["ais_bench.benchmark.tasks.mcp_atlas.mcp_atlas_infer"] = _infer_mod
_mock_registry.TASKS = mock.MagicMock()  # reset for second register_module call
_infer_spec.loader.exec_module(_infer_mod)

# Import all utilities from the modules
from ais_bench.benchmark.tasks.mcp_atlas.utils import (
    MCPAtlasClient,
    MCPAtlasServerUnavailable,
    MCPToolCallError,
    build_messages,
    cap_tool_content,
    compact_messages,
    compute_coverage_score,
    compute_coverage_stats,
    detect_tool_calls,
    extract_claims,
    get_claim_evaluation_schema,
    get_retry_delay,
    is_retryable_error,
    is_tool_call_message,
    mcp_tool_to_tool_info,
    pruned_tool_call,
)

# From eval module
MCPAtlasEvalTask = m.MCPAtlasEvalTask
_claim_judge_prompt = m._claim_judge_prompt
_parse_claim_judge_response = m._parse_claim_judge_response

# From infer module
MCPAtlasInferTask = _infer_mod.MCPAtlasInferTask
_parse_enabled_tools = _infer_mod._parse_enabled_tools
_extract_claims = _infer_mod._extract_claims
_extract_required_servers = _infer_mod._extract_required_servers
_tool_name_to_server = _infer_mod._tool_name_to_server
_server_unavailable_message = _infer_mod._server_unavailable_message
_call_tool_subprocess = _infer_mod._call_tool_subprocess

# Internal helpers only in utils (not re-exported by either task module)
_is_transport_error = _utils_mod._is_transport_error
_format_tool_response = _utils_mod._format_tool_response
_parse_confidence = _utils_mod._parse_confidence
_clean_claim_text = _utils_mod._clean_claim_text
_strip_json_fence = _utils_mod._strip_json_fence


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def enabled_servers(self) -> List[str]:
        return ["wikipedia"]

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "wikipedia_get_article",
                "description": "Fetch a Wikipedia article.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title."},
                    },
                    "required": ["title"],
                },
            },
            {
                "name": "github_search_repositories",
                "description": "Search repos.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        self.calls.append({"tool_name": tool_name, "tool_args": tool_args})
        return "tool result"


# ===========================================================================
# Tests: Prompt Construction (ported from mcp-atlas run_eval.py)
# ===========================================================================


class TestBuildMessages(unittest.TestCase):
    """Tests for build_messages() — prompt construction."""

    def test_user_message_only(self):
        msgs = build_messages("What is MCP?")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "What is MCP?")

    def test_system_prompt_prepend(self):
        msgs = build_messages("What is MCP?", system_prompt="You are helpful.")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are helpful.")
        self.assertEqual(msgs[1]["role"], "user")

    def test_none_system_prompt(self):
        msgs = build_messages("prompt", system_prompt=None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_empty_string_system_prompt(self):
        msgs = build_messages("prompt", system_prompt="")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_matches_mcp_atlas_run_eval_format(self):
        """Verify messages match mcp-atlas run_eval.py format."""
        msgs = build_messages(
            "What is the first word of /data/Barber Shop.csv?",
            system_prompt="System instructions"
        )
        self.assertEqual(msgs[0], {"role": "system", "content": "System instructions"})
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("Barber Shop", msgs[1]["content"])


# ===========================================================================
# Tests: Tool Call Detection (ported from mcp-atlas agent-eval.ts Zod schema)
# ===========================================================================


class TestDetectToolCalls(unittest.TestCase):
    """Tests for detect_tool_calls() and is_tool_call_message()."""

    def test_valid_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "wikipedia_get_article",
                        "arguments": '{"title": "MCP"}',
                    },
                }
            ],
        }
        calls = detect_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_123")
        self.assertEqual(calls[0]["function"]["name"], "wikipedia_get_article")

    def test_multiple_tool_calls(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
            ],
        }
        calls = detect_tool_calls(msg)
        self.assertEqual(len(calls), 2)

    def test_no_tool_calls_key(self):
        msg = {"role": "assistant", "content": "Final answer."}
        self.assertEqual(detect_tool_calls(msg), [])
        self.assertFalse(is_tool_call_message(msg))

    def test_empty_tool_calls_list(self):
        msg = {"role": "assistant", "tool_calls": []}
        self.assertEqual(detect_tool_calls(msg), [])
        self.assertFalse(is_tool_call_message(msg))

    def test_none_tool_calls(self):
        msg = {"role": "assistant", "tool_calls": None}
        self.assertEqual(detect_tool_calls(msg), [])

    def test_non_assistant_role(self):
        msg = {"role": "user", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]}
        self.assertEqual(detect_tool_calls(msg), [])

    def test_missing_id(self):
        msg = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": "{}"}}]}
        self.assertEqual(detect_tool_calls(msg), [])

    def test_missing_function(self):
        msg = {"role": "assistant", "tool_calls": [{"id": "c1"}]}
        self.assertEqual(detect_tool_calls(msg), [])

    def test_missing_name(self):
        msg = {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"arguments": "{}"}}]}
        self.assertEqual(detect_tool_calls(msg), [])

    def test_invalid_item_in_list(self):
        msg = {"role": "assistant", "tool_calls": ["not_a_dict", 42]}
        self.assertEqual(detect_tool_calls(msg), [])

    def test_is_tool_call_message_true(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        }
        self.assertTrue(is_tool_call_message(msg))

    def test_is_tool_call_message_empty_message(self):
        self.assertFalse(is_tool_call_message({}))
        self.assertFalse(is_tool_call_message(None))


# ===========================================================================
# Tests: Context Window Management (ported from mcp-atlas agent-eval.ts)
# ===========================================================================


class TestCapToolContent(unittest.TestCase):
    """Tests for cap_tool_content()."""

    def test_short_content_unchanged(self):
        self.assertEqual(cap_tool_content("hello", 100), "hello")

    def test_exact_length_unchanged(self):
        content = "x" * 50
        self.assertEqual(cap_tool_content(content, 50), content)

    def test_truncation(self):
        content = "x" * 200
        result = cap_tool_content(content, 50)
        self.assertEqual(len(result), 50 + len("\n\n[Tool output truncated to 50 chars. Original was 200 chars.]"))
        self.assertTrue(result.startswith("xxxxx"))
        self.assertIn("truncated to 50 chars", result)

    def test_empty_content(self):
        self.assertEqual(cap_tool_content("", 100), "")


class TestCompactMessages(unittest.TestCase):
    """Tests for compact_messages()."""

    def test_no_compaction_for_early_turns(self):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": "x" * 2000, "tool_call_id": "c1"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": "y" * 2000, "tool_call_id": "c2"},
        ]
        # Turn 1 (< COMPACT_KEEP_FULL_TURNS=2): no compaction
        result = compact_messages(messages, 1)
        self.assertEqual(result, messages)

    def test_compaction_for_later_turns(self):
        long_content = "z" * 2000
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": long_content, "tool_call_id": "c1"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": long_content, "tool_call_id": "c2"},
            {"role": "assistant", "tool_calls": [{"id": "c3", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": "short result", "tool_call_id": "c3"},
        ]
        # Turn 3 (> COMPACT_KEEP_FULL_TURNS=2): oldest turn should be compacted
        result = compact_messages(messages, 3)
        # First tool result should be truncated
        first_tool = next(m for m in result if m.get("role") == "tool")
        self.assertIn("truncated", first_tool["content"])
        self.assertIn("1500 chars", first_tool["content"])
        # Last tool result should be untouched
        last_tool = [m for m in result if m.get("role") == "tool"][-1]
        self.assertEqual(last_tool["content"], "short result")

    def test_short_tool_results_unchanged(self):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": "short", "tool_call_id": "c1"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": "short", "tool_call_id": "c2"},
            {"role": "assistant", "tool_calls": [{"id": "c3", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": "short", "tool_call_id": "c3"},
        ]
        result = compact_messages(messages, 3)
        for m in result:
            if m.get("role") == "tool":
                self.assertEqual(m["content"], "short")


# ===========================================================================
# Tests: LLM Retry Logic (ported from mcp-atlas litellm-strategy.ts)
# ===========================================================================


class TestIsRetryableError(unittest.TestCase):
    """Tests for is_retryable_error()."""

    def test_429_is_retryable(self):
        exc = mock.MagicMock()
        exc.response = mock.MagicMock()
        exc.response.status_code = 429
        self.assertTrue(is_retryable_error(exc))

    def test_503_is_retryable(self):
        exc = mock.MagicMock()
        exc.response = mock.MagicMock()
        exc.response.status_code = 503
        self.assertTrue(is_retryable_error(exc))

    def test_500_is_retryable(self):
        exc = mock.MagicMock()
        exc.response = mock.MagicMock()
        exc.response.status_code = 500
        self.assertTrue(is_retryable_error(exc))

    def test_400_not_retryable(self):
        exc = mock.MagicMock()
        exc.response = mock.MagicMock()
        exc.response.status_code = 400
        self.assertFalse(is_retryable_error(exc))

    def test_timeout_is_retryable(self):
        exc = Exception("request timeout after 30s")
        self.assertTrue(is_retryable_error(exc))

    def test_generic_error_not_retryable(self):
        exc = ValueError("bad input")
        self.assertFalse(is_retryable_error(exc))

    def test_no_response_attribute_not_retryable(self):
        exc = RuntimeError("something went wrong")
        self.assertFalse(is_retryable_error(exc))


class TestGetRetryDelay(unittest.TestCase):
    """Tests for get_retry_delay()."""

    def test_429_exponential_backoff(self):
        exc = mock.MagicMock()
        exc.response = mock.MagicMock()
        exc.response.status_code = 429
        delay_0 = get_retry_delay(exc, 0)
        delay_1 = get_retry_delay(exc, 1)
        self.assertEqual(delay_0, 5)   # 2^0 * 5 = 5
        self.assertEqual(delay_1, 10)  # 2^1 * 5 = 10
        self.assertGreater(delay_1, delay_0)

    def test_timeout_fixed_delay(self):
        exc = Exception("request timeout after 30s")
        self.assertEqual(get_retry_delay(exc, 0), 15.0)
        self.assertEqual(get_retry_delay(exc, 2), 15.0)

    def test_503_default_delay(self):
        exc = mock.MagicMock()
        exc.response = mock.MagicMock()
        exc.response.status_code = 503
        self.assertEqual(get_retry_delay(exc, 0), 10.0)

    def test_429_capped_at_30(self):
        exc = mock.MagicMock()
        exc.response = mock.MagicMock()
        exc.response.status_code = 429
        delay = get_retry_delay(exc, 10)  # 2^10 * 5 = 5120
        self.assertEqual(delay, 30)


# ===========================================================================
# Tests: Tool Conversion (ported from mcp-atlas agent-eval.ts)
# ===========================================================================


class TestMCPToolToToolInfo(unittest.TestCase):
    """Tests for mcp_tool_to_tool_info() with strict=False."""

    def test_basic_tool(self):
        raw = {
            "name": "test_tool",
            "description": "A test tool.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        }
        result = mcp_tool_to_tool_info(raw)
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "test_tool")
        self.assertEqual(result["function"]["description"], "A test tool.")
        self.assertEqual(
            result["function"]["parameters"]["required"], ["query"]
        )
        # Matches mcp-atlas: strict=False
        self.assertEqual(result["function"]["strict"], False)

    def test_fallback_description(self):
        raw = {"name": "no_desc_tool"}
        result = mcp_tool_to_tool_info(raw)
        self.assertEqual(result["function"]["description"], "no_desc_tool")

    def test_input_schema_alias(self):
        raw = {
            "name": "alias_tool",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        }
        result = mcp_tool_to_tool_info(raw)
        self.assertEqual(
            result["function"]["parameters"]["type"], "object"
        )

    def test_full_schema_passthrough(self):
        """Verify the full inputSchema is passed through (mcp-atlas behavior)."""
        raw = {
            "name": "complex_tool",
            "description": "Complex tool.",
            "inputSchema": {
                "type": "object",
                "properties": {"p1": {"type": "string"}},
                "required": ["p1"],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema",
            },
        }
        result = mcp_tool_to_tool_info(raw)
        self.assertEqual(result["function"]["parameters"]["type"], "object")
        self.assertIn("additionalProperties", result["function"]["parameters"])
        self.assertIn("$schema", result["function"]["parameters"])


class TestPrunedToolCall(unittest.TestCase):
    """Tests for pruned_tool_call()."""

    def test_met_museum_image_suppression(self):
        args = {"objectID": 123, "returnImage": True}
        result = pruned_tool_call("met-museum_get-museum-object", args)
        self.assertEqual(result["returnImage"], False)

    def test_other_tool_unchanged(self):
        args = {"query": "test"}
        result = pruned_tool_call("wikipedia_get_article", args)
        self.assertEqual(result, args)


# ===========================================================================
# Tests: LLM Judge (ported from mcp-atlas score_claims.py)
# ===========================================================================


class TestClaimJudgePrompt(unittest.TestCase):
    """Tests for the enhanced _claim_judge_prompt()."""

    def test_includes_claim_and_response(self):
        prompt = _claim_judge_prompt("test claim", "test response")
        self.assertIn("test claim", prompt)
        self.assertIn("test response", prompt)

    def test_includes_scoring_criteria(self):
        prompt = _claim_judge_prompt("claim", "response")
        self.assertIn("fulfilled", prompt)
        self.assertIn("partially_fulfilled", prompt)
        self.assertIn("not_fulfilled", prompt)

    def test_includes_numerical_guidelines(self):
        prompt = _claim_judge_prompt("42", "answer")
        self.assertIn("NUMERICAL COMPARISON GUIDELINES", prompt)
        self.assertIn("5%", prompt)
        self.assertIn("±1 percentage point", prompt)

    def test_includes_confidence_level(self):
        prompt = _claim_judge_prompt("claim", "response")
        self.assertIn("confidence_level", prompt)

    def test_includes_json_output_format(self):
        prompt = _claim_judge_prompt("claim", "response")
        self.assertIn("coverage_outcome", prompt)
        self.assertIn("justification", prompt)


class TestGetClaimEvaluationSchema(unittest.TestCase):
    """Tests for get_claim_evaluation_schema()."""

    def test_returns_valid_schema(self):
        schema = get_claim_evaluation_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("claim_text", schema["properties"])
        self.assertIn("coverage_outcome", schema["properties"])
        self.assertIn("enum", schema["properties"]["coverage_outcome"])
        self.assertEqual(
            set(schema["properties"]["coverage_outcome"]["enum"]),
            {"fulfilled", "partially_fulfilled", "not_fulfilled"},
        )

    def test_required_fields(self):
        schema = get_claim_evaluation_schema()
        self.assertIn("claim_text", schema["required"])
        self.assertIn("coverage_outcome", schema["required"])
        self.assertIn("justification", schema["required"])
        self.assertIn("confidence_level", schema["required"])


class TestParseClaimJudgeResponse(unittest.TestCase):
    """Tests for _parse_claim_judge_response()."""

    def test_structured_json_response(self):
        outcome, just, confidence = _parse_claim_judge_response(
            json.dumps({
                "claim_text": "the answer is 42",
                "coverage_outcome": "fulfilled",
                "justification": "covered",
                "confidence_level": 0.95,
            })
        )
        self.assertEqual(outcome, "fulfilled")
        self.assertEqual(just, "covered")
        self.assertAlmostEqual(confidence, 0.95)

    def test_json_in_markdown_fence(self):
        outcome, just, confidence = _parse_claim_judge_response(
            '```json\n{"coverage_outcome": "partially_fulfilled", '
            '"justification": "partly", "confidence_level": "high"}\n```'
        )
        self.assertEqual(outcome, "partially_fulfilled")
        self.assertEqual(just, "partly")
        self.assertEqual(confidence, 1.0)

    def test_accepts_fallback_keys(self):
        outcome, just, confidence = _parse_claim_judge_response(
            json.dumps({
                "outcome": "partially_fulfilled",
                "reason": "partly covered",
                "confidence": "medium",
            })
        )
        self.assertEqual(outcome, "partially_fulfilled")
        self.assertEqual(just, "partly covered")
        self.assertEqual(confidence, 0.5)

    def test_text_fallback(self):
        outcome, _, _ = _parse_claim_judge_response(
            "The claim is fulfilled based on the evidence."
        )
        self.assertEqual(outcome, "fulfilled")

    def test_not_json_fallback(self):
        outcome, _, _ = _parse_claim_judge_response("gibberish")
        self.assertEqual(outcome, "not_fulfilled")

    def test_non_text_input(self):
        outcome, just, confidence = _parse_claim_judge_response(123)
        self.assertEqual(outcome, "not_fulfilled")
        self.assertEqual(confidence, 0.0)

    def test_invalid_outcome_normalized(self):
        outcome, _, _ = _parse_claim_judge_response(
            json.dumps({"coverage_outcome": "maybe", "justification": "", "confidence_level": 0.5})
        )
        self.assertEqual(outcome, "not_fulfilled")


class TestStripJsonFence(unittest.TestCase):

    def test_json_fence(self):
        self.assertEqual(
            _strip_json_fence('```json\n{"a": 1}\n```'), '{"a": 1}'
        )

    def test_no_fence(self):
        self.assertEqual(_strip_json_fence('{"a": 1}'), '{"a": 1}')

    def test_fence_without_language(self):
        self.assertEqual(
            _strip_json_fence('```\n{"a": 1}\n```'), '{"a": 1}'
        )


class TestParseConfidence(unittest.TestCase):

    def test_numeric_values(self):
        self.assertAlmostEqual(_parse_confidence(0.75), 0.75)
        self.assertAlmostEqual(_parse_confidence(1), 1.0)
        self.assertAlmostEqual(_parse_confidence(0), 0.0)

    def test_string_levels(self):
        self.assertEqual(_parse_confidence("high"), 1.0)
        self.assertEqual(_parse_confidence("high confidence"), 1.0)
        self.assertEqual(_parse_confidence("medium"), 0.5)
        self.assertEqual(_parse_confidence("moderate"), 0.5)
        self.assertEqual(_parse_confidence("low"), 0.0)
        self.assertEqual(_parse_confidence("low confidence"), 0.0)

    def test_none_and_invalid(self):
        self.assertEqual(_parse_confidence(None), 0.0)
        self.assertEqual(_parse_confidence("unknown"), 0.0)


# ===========================================================================
# Tests: Claim Extraction (ported from mcp-atlas score_claims.py)
# ===========================================================================


class TestExtractClaims(unittest.TestCase):

    def test_json_list(self):
        claims = extract_claims('["claim one", "claim two"]')
        self.assertEqual(claims, ["claim one", "claim two"])

    def test_python_list_literal(self):
        claims = extract_claims("['claim one', 'claim two']")
        self.assertEqual(claims, ["claim one", "claim two"])

    def test_flattens_nested_single_literal_list(self):
        """JSON list containing string with Python literal list notation.

        JSON parsing yields a 1-element list whose sole item is a string
        like \"['claim one', 'claim two']\".  mcp-atlas does NOT recursively
        parse inner string elements, so we get 1 claim, not 2.
        """
        claims = extract_claims(
            json.dumps(["['claim one', 'claim two']"])
        )
        self.assertEqual(len(claims), 1)

    def test_dict_with_claim_key(self):
        claims = extract_claims(
            json.dumps([{"claim": "the answer is 42"}])
        )
        self.assertEqual(claims, ["the answer is 42"])

    def test_empty_string(self):
        self.assertEqual(extract_claims(""), [])

    def test_none_input(self):
        self.assertEqual(extract_claims(None), [])

    def test_multiline_text(self):
        text = "• Claim one\n• Claim two\n• Claim three"
        claims = extract_claims(text)
        self.assertEqual(len(claims), 3)
        self.assertIn("Claim one", claims)

    def test_short_claims_filtered(self):
        claims = extract_claims('["ab", "cd", "this is a valid claim"]')
        self.assertEqual(claims, ["this is a valid claim"])

    def test_clean_claim_text(self):
        self.assertEqual(_clean_claim_text("- bullet point"), "bullet point")
        self.assertEqual(_clean_claim_text("1. numbered"), "numbered")
        self.assertEqual(_clean_claim_text('"quoted"'), '"quoted')  # mcp-atlas only strips trailing quotes


# ===========================================================================
# Tests: Coverage Scoring (ported from mcp-atlas score_claims.py)
# ===========================================================================


class TestComputeCoverageScore(unittest.TestCase):
    """Tests for compute_coverage_score()."""

    def test_all_fulfilled(self):
        results = [
            {"coverage_outcome": "fulfilled"},
            {"coverage_outcome": "fulfilled"},
        ]
        score, f, p, n = compute_coverage_score(results)
        self.assertEqual(score, 1.0)
        self.assertEqual(f, 2)
        self.assertEqual(p, 0)
        self.assertEqual(n, 0)

    def test_mixed(self):
        results = [
            {"coverage_outcome": "fulfilled"},
            {"coverage_outcome": "partially_fulfilled"},
            {"coverage_outcome": "not_fulfilled"},
        ]
        score, f, p, n = compute_coverage_score(results)
        self.assertAlmostEqual(score, 0.5)  # (1.0 + 0.5 + 0.0) / 3 = 0.5
        self.assertEqual(f, 1)
        self.assertEqual(p, 1)
        self.assertEqual(n, 1)

    def test_empty_results(self):
        score, f, p, n = compute_coverage_score([])
        self.assertEqual(score, 0.0)
        self.assertEqual(f, 0)
        self.assertEqual(p, 0)
        self.assertEqual(n, 0)

    def test_unknown_outcome(self):
        results = [{"coverage_outcome": "unknown_outcome"}]
        score, f, p, n = compute_coverage_score(results)
        self.assertEqual(score, 0.0)
        self.assertEqual(n, 1)


class TestComputeCoverageStats(unittest.TestCase):
    """Tests for compute_coverage_stats()."""

    def test_basic_stats(self):
        results = [
            {"coverage_score": 1.0},
            {"coverage_score": 0.75},
            {"coverage_score": 0.5},
            {"coverage_score": 0.0},
        ]
        stats = compute_coverage_stats(
            results, pass_threshold=0.75,
            model_name="test-model", evaluator_model="gemini"
        )
        self.assertEqual(stats["total_tasks"], 4)
        self.assertEqual(stats["valid_responses"], 4)
        self.assertAlmostEqual(stats["mean_coverage"], 0.5625)
        self.assertEqual(stats["pass_rate_0.75"], 50.0)  # 2/4 = 50%
        self.assertEqual(stats["pass_rate_0.50"], 75.0)  # 3/4 = 75%
        self.assertEqual(stats["model_name"], "test-model")

    def test_with_none_scores(self):
        results = [
            {"coverage_score": 1.0},
            {"coverage_score": None},
            {"coverage_score": 0.5},
        ]
        stats = compute_coverage_stats(results)
        self.assertEqual(stats["total_tasks"], 3)
        self.assertEqual(stats["valid_responses"], 2)
        self.assertEqual(stats["empty_or_error"], 1)
        self.assertAlmostEqual(stats["mean_coverage"], 0.75)

    def test_empty_results(self):
        stats = compute_coverage_stats([])
        self.assertEqual(stats["total_tasks"], 0)
        self.assertEqual(stats["mean_coverage"], 0.0)


# ===========================================================================
# Tests: helpers (utils)
# ===========================================================================


class TestParseEnabledTools(unittest.TestCase):

    def test_accepts_strings_and_objects(self):
        tools = _parse_enabled_tools(
            json.dumps([
                "wikipedia_get_article",
                {"name": "github_search_repositories"},
                "wikipedia_get_article",
                123,
            ])
        )
        self.assertEqual(
            tools, ["wikipedia_get_article", "github_search_repositories"]
        )

    def test_empty_list(self):
        self.assertEqual(_parse_enabled_tools("[]"), [])

    def test_non_json_string(self):
        self.assertEqual(_parse_enabled_tools("not-json"), [])


class TestExtractRequiredServers(unittest.TestCase):

    def test_uses_trajectory_tool_calls(self):
        trajectory = json.dumps([{
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "wikipedia_get_article"}},
                {"function": {"name": "MongoDB_find"}},
                {"function": {"name": "brave_brave_web_search"}},
            ],
        }])
        self.assertEqual(
            _extract_required_servers(trajectory),
            ["brave-search", "mongodb", "wikipedia"],
        )

    def test_empty_trajectory(self):
        self.assertEqual(_extract_required_servers("[]"), [])


class TestExtractClaimsBackwardCompat(unittest.TestCase):
    """Test _extract_claims alias still works."""

    def test_alias(self):
        claims = _extract_claims('["claim one", "claim two"]')
        self.assertEqual(claims, ["claim one", "claim two"])


class TestIsTransportError(unittest.TestCase):

    def test_connection_refused(self):
        self.assertTrue(_is_transport_error(
            "connect ECONNREFUSED 199.193.116.105:443"
        ))

    def test_timed_out(self):
        self.assertTrue(_is_transport_error("read timeout"))

    def test_normal_error(self):
        self.assertFalse(_is_transport_error("bad request error"))


class TestToolNameToServer(unittest.TestCase):

    def test_standard_mapping(self):
        self.assertEqual(
            _tool_name_to_server("brave_brave_web_search"), "brave-search"
        )

    def test_mongodb_mapping(self):
        self.assertEqual(
            _tool_name_to_server("MongoDB_find"), "mongodb"
        )

    def test_fallback(self):
        self.assertEqual(
            _tool_name_to_server("filesystem_read_file"), "filesystem"
        )

    def test_all_mongodb_variations(self):
        for mongo_tool in [
            "MongoDB_aggregate", "MongoDB_collection-schema",
            "MongoDB_count", "MongoDB_find",
            "MongoDB_list-collections", "MongoDB_list-databases",
        ]:
            self.assertEqual(
                _tool_name_to_server(mongo_tool), "mongodb",
                f"{mongo_tool} should map to mongodb"
            )


class TestFormatToolResponse(unittest.TestCase):

    def test_content_list(self):
        result = _format_tool_response([
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ])
        self.assertEqual(result, "hello\n\nworld")

    def test_dict(self):
        result = _format_tool_response({"key": "value"})
        self.assertIn("key", result)

    def test_plain_string(self):
        self.assertEqual(_format_tool_response("plain"), "plain")

    def test_list_with_empty_text(self):
        result = _format_tool_response([
            {"type": "text", "text": ""},
            {"type": "text", "text": "valid"},
        ])
        self.assertEqual(result, "valid")

    def test_mixed_types(self):
        result = _format_tool_response([
            {"type": "text", "text": "hello"},
            {"not_text": True},
        ])
        self.assertIn("hello", result)


class TestServerUnavailableMessage(unittest.TestCase):

    def test_format(self):
        msg = _server_unavailable_message("test-server", "timeout")
        self.assertIn("test-server", msg)
        self.assertIn("timeout", msg)


# ===========================================================================
# Tests: MCPAtlasClient
# ===========================================================================


class TestMCPAtlasClient(unittest.TestCase):

    def test_client_uses_mcp_atlas_http_endpoints(self):
        requests_seen: List[Dict[str, Any]] = []

        class Response:
            def __init__(self, status_code: int, payload: Any):
                self.status_code = status_code
                self.payload = payload
                self.text = json.dumps(payload)

            def raise_for_status(self): pass

            def json(self): return self.payload

        def fake_get(url: str, timeout: float) -> Response:
            requests_seen.append({
                "method": "GET", "url": url, "timeout": timeout,
            })
            return Response(200, {"enabled_servers": ["wikipedia"]})

        def fake_post(url: str, **kwargs: Any) -> Response:
            requests_seen.append({"method": "POST", "url": url, **kwargs})
            if url.endswith("/list-tools"):
                return Response(200, [{"name": "wikipedia_get_article"}])
            if url.endswith("/call-tool"):
                return Response(
                    200, [{"type": "text", "text": "article text"}]
                )
            raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch("requests.get", fake_get), \
             mock.patch("requests.post", fake_post):
            client = MCPAtlasClient(
                "http://localhost:1984/",
                request_timeout=7.0,
                list_tools_timeout=11.0,
            )
            self.assertEqual(client.enabled_servers(), ["wikipedia"])
            self.assertEqual(client.list_tools(), [{"name": "wikipedia_get_article"}])
            self.assertEqual(
                client.call_tool("wikipedia_get_article", {"title": "MCP"}),
                "article text",
            )
            self.assertEqual(requests_seen[0]["url"], "http://localhost:1984/enabled-servers")
            self.assertEqual(requests_seen[1]["url"], "http://localhost:1984/list-tools")
            self.assertEqual(
                requests_seen[2]["json"],
                {"tool_name": "wikipedia_get_article", "tool_args": {"title": "MCP"}},
            )

    def test_accepts_servers_dict_response(self):
        class Response:
            def raise_for_status(self): pass
            def json(self) -> Dict[str, Any]:
                return {
                    "servers": {
                        "wikipedia": "OK",
                        "github": "ERROR_NOT_ONLINE",
                    },
                }

        with mock.patch("requests.get", return_value=Response()):
            client = MCPAtlasClient(
                "http://localhost:1984",
                request_timeout=7.0,
                list_tools_timeout=11.0,
            )
            self.assertEqual(client.enabled_servers(), ["wikipedia"])

    def test_marks_transport_500_as_unavailable(self):
        class Response:
            status_code = 500
            text = '{"detail":"connect ECONNREFUSED 199.193.116.105:443"}'
            def json(self): return {}

        with mock.patch("requests.post", return_value=Response()):
            client = MCPAtlasClient(
                "http://localhost:1984",
                request_timeout=7.0,
                list_tools_timeout=11.0,
            )
            with self.assertRaises(MCPAtlasServerUnavailable) as ctx:
                client.call_tool("open-library_get_book_by_title", {"title": "x"})
            self.assertEqual(ctx.exception.server_name, "open-library")

    def test_returns_bounded_http_error(self):
        class Response:
            status_code = 500
            text = "x" * 1200
            def json(self): return {}

        with mock.patch("requests.post", return_value=Response()):
            client = MCPAtlasClient(
                "http://localhost:1984",
                request_timeout=7.0,
                list_tools_timeout=11.0,
            )
            result = client.call_tool("wikipedia_get_article", {"title": "MCP"})
            self.assertTrue(result.startswith("Error calling tool"))
            self.assertTrue(result.endswith("..."))
            self.assertLess(len(result), 1100)

    def test_handles_malformed_json_response(self):
        class Response:
            status_code = 200
            text = "{not json"
            def json(self):
                raise ValueError("bad json")

        with mock.patch("requests.post", return_value=Response()):
            client = MCPAtlasClient(
                "http://localhost:1984",
                request_timeout=7.0,
                list_tools_timeout=11.0,
            )
            result = client.call_tool("wikipedia_get_article", {"title": "MCP"})
            self.assertIn("Error decoding tool response JSON", result)


# ===========================================================================
# Tests: subprocess tool executor
# ===========================================================================


class TestCallToolSubprocess(unittest.TestCase):

    def test_successful_call(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_result = mock.MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({
                "ok": True,
                "data": [{"type": "text", "text": "result from subprocess"}],
            })
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = _call_tool_subprocess(
                tool_name="test_tool",
                tool_args={"query": "hello"},
                base_url="http://localhost:1984",
                timeout=30.0,
            )
            self.assertEqual(result, "result from subprocess")
            mock_run.assert_called_once()

    def test_subprocess_error(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_result = mock.MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = json.dumps({
                "ok": False,
                "error": "Connection refused",
            })
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            with self.assertRaises(MCPAtlasServerUnavailable) as ctx:
                _call_tool_subprocess(
                    tool_name="test_tool",
                    tool_args={},
                    base_url="http://localhost:1984",
                    timeout=30.0,
                )
            self.assertIn("Connection refused", str(ctx.exception))

    def test_subprocess_timeout(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 60)

            with self.assertRaises(MCPAtlasServerUnavailable) as ctx:
                _call_tool_subprocess(
                    tool_name="test_tool",
                    tool_args={},
                    base_url="http://localhost:1984",
                    timeout=30.0,
                )
            self.assertIn("timed out", str(ctx.exception).lower())

    def test_json_decode_fallback(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_result = mock.MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "plain text response"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = _call_tool_subprocess(
                tool_name="test_tool",
                tool_args={},
                base_url="http://localhost:1984",
                timeout=30.0,
            )
            self.assertEqual(result, "plain text response")


# ===========================================================================
# Tests: MCPAtlasEvalTask (basic structure only)
# ===========================================================================


class TestMCPAtlasEvalTask(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_client_creation(self):
        client = MCPAtlasClient(
            "http://localhost:1984",
            request_timeout=60.0,
            list_tools_timeout=180.0,
        )
        self.assertIsInstance(client, MCPAtlasClient)
        self.assertEqual(client.base_url, "http://localhost:1984")

    def test_task_is_registered(self):
        self.assertTrue(hasattr(m, "MCPAtlasEvalTask"))
        self.assertTrue(hasattr(_infer_mod, "MCPAtlasInferTask"))
        self.assertGreaterEqual(
            _mock_registry.TASKS.register_module.call_count, 1
        )

    def test_preflight_with_fake_client(self):
        fake = FakeClient()
        self.assertEqual(fake.enabled_servers(), ["wikipedia"])
        self.assertEqual(len(fake.list_tools()), 2)


# ===========================================================================
# Tests: MCPAtlasInferTask structure verification
# ===========================================================================


class TestMCPAtlasInferTaskStructure(unittest.TestCase):
    """Verify MCPAtlasInferTask has the expected interface."""

    def test_has_run_method(self):
        self.assertTrue(hasattr(MCPAtlasInferTask, "run"))
        self.assertTrue(callable(getattr(MCPAtlasInferTask, "run", None)))

    def test_has_get_command_method(self):
        self.assertTrue(hasattr(MCPAtlasInferTask, "get_command"))


# ===========================================================================
# Integration-style Tests: Full agent loop mock
# ===========================================================================


class TestAgentLoopMock(unittest.TestCase):
    """Test the agent loop flow with mocked model and tool calls."""

    def test_single_tool_call_and_final_answer(self):
        """Simulate: user → model(tool_call) → tool_result → model(final_answer)."""
        messages = build_messages("What is MCP?", system_prompt="Be helpful.")
        self.assertEqual(len(messages), 2)

        # Simulate model returning tool call
        tool_call_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "wikipedia_get_article", "arguments": '{"title": "MCP"}'},
            }],
        }

        # Detect tool calls
        calls = detect_tool_calls(tool_call_msg)
        self.assertEqual(len(calls), 1)
        self.assertTrue(is_tool_call_message(tool_call_msg))

        # Add to messages
        messages.append(tool_call_msg)

        # Simulate tool result
        tool_result = {"role": "tool", "tool_call_id": "call_1", "content": "MCP is a protocol."}
        messages.append(tool_result)

        # Simulate final answer
        final_msg = {"role": "assistant", "content": "MCP is the Model Context Protocol."}
        self.assertFalse(is_tool_call_message(final_msg))

    def test_compaction_during_loop(self):
        """Test that compaction reduces message size during multi-turn."""
        messages = [{"role": "user", "content": "question"}]
        for i in range(5):
            tc_msg = {
                "role": "assistant",
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            }
            messages.append(tc_msg)
            messages.append({
                "role": "tool", "tool_call_id": f"c{i}",
                "content": "x" * 2000,
            })

        # Compact at turn 3
        compacted = compact_messages(messages, 3)
        original_size = sum(len(json.dumps(m)) for m in messages)
        compacted_size = sum(len(json.dumps(m)) for m in compacted)
        self.assertLess(compacted_size, original_size)

    def test_tool_output_cap(self):
        """Test tool output capping."""
        long_result = "x" * 10000
        capped = cap_tool_content(long_result, 500)
        self.assertTrue(capped.startswith("xxxxx"))
        self.assertIn("truncated to 500 chars", capped)
        self.assertLess(len(capped), 700)

    def test_retry_logic_in_loop(self):
        """Test retry logic for transient errors."""
        exc_429 = mock.MagicMock()
        exc_429.response = mock.MagicMock()
        exc_429.response.status_code = 429
        self.assertTrue(is_retryable_error(exc_429))

        delay = get_retry_delay(exc_429, 0)
        self.assertGreater(delay, 0)

        exc_400 = mock.MagicMock()
        exc_400.response = mock.MagicMock()
        exc_400.response.status_code = 400
        self.assertFalse(is_retryable_error(exc_400))


# ===========================================================================
# Tests: MCPAtlasSummarizer integration with new stats
# ===========================================================================


class TestCoverageStatsIntegration(unittest.TestCase):
    """Verify coverage stats work with the expected result format."""

    def test_stats_from_eval_results(self):
        """Simulate eval results and compute stats."""
        results = [
            {
                "task_id": "task_1",
                "coverage_score": 1.0,
                "fully_covered_claims": 3,
                "partially_covered_claims": 0,
                "not_covered_claims": 0,
                "total_claims": 3,
            },
            {
                "task_id": "task_2",
                "coverage_score": 0.5,
                "fully_covered_claims": 1,
                "partially_covered_claims": 1,
                "not_covered_claims": 1,
                "total_claims": 3,
            },
        ]
        stats = compute_coverage_stats(
            results,
            pass_threshold=0.75,
            model_name="test",
            evaluator_model="gemini",
        )
        self.assertEqual(stats["total_tasks"], 2)
        self.assertAlmostEqual(stats["mean_coverage"], 0.75)
        self.assertEqual(stats["pass_rate_0.75"], 50.0)
        self.assertEqual(stats["pass_rate_0.50"], 100.0)


if __name__ == "__main__":
    unittest.main()
