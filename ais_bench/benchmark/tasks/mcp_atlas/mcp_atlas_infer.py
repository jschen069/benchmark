"""MCP-Atlas inference task.

Runs the agent loop against the MCP-Atlas agent-environment Docker service,
calls live MCP tools via subprocess isolation, and saves final answers
as predictions.  The predictions are later scored by
:class:`MCPAtlasEvalTask`.

Ported from the upstream mcp-atlas agent-harness (TypeScript) and
``run_eval.py``, adapted for aisbench conventions.  Key improvements
over the initial version:

- **Robust tool-call detection**: Uses ``detect_tool_calls`` with structure
  validation (matching the Zod schema in mcp-atlas agent-eval.ts).
- **LLM retry logic**: Transient errors (503, 429, timeout) are retried
  with exponential backoff (matching litellm-strategy.ts).
- **Error recovery**: Tool-call failures are fed back to the model as
  tool results instead of being silently dropped.
- **Context window management**: Optional ``compact`` mode truncates old
  tool results, and ``tool_output_cap`` limits individual tool responses.
- **Full trajectory recording**: The complete message history is saved
  for diagnostics and evaluation.

"""

import argparse
import json
import os
import os.path as osp
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist
from tqdm import tqdm

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.config.build import build_dataset_from_cfg
from ais_bench.benchmark.utils.core.abbr import (
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger

from ais_bench.benchmark.tasks.mcp_atlas.utils import (
    DEFAULT_MCP_SERVER_URL,
    DEFAULT_SYSTEM_PROMPT,
    MCPAtlasClient,
    MCPAtlasServerUnavailable,
    MAX_LLM_RETRIES,
    _call_tool_subprocess,
    _extract_claims,
    _extract_required_servers,
    _field,
    _maybe_parse_json,
    _parse_enabled_tools,
    _server_unavailable_message,
    _tool_name_to_server,
    build_messages,
    cap_tool_content,
    compact_messages,
    detect_tool_calls,
    get_retry_delay,
    is_retryable_error,
    mcp_tool_to_tool_info,
    pruned_tool_call,
)


@TASKS.register_module()
class MCPAtlasInferTask(BaseTask):
    """Run the agent loop against MCP-Atlas and save predictions.

    The task connects to the MCP-Atlas agent-environment Docker service,
    loads the dataset, runs a multi-turn conversation loop for each
    sample with subprocess-isolated MCP tool calls, and writes
    predictions to ``predictions/<model_abbr>/<dataset_abbr>.json``.

    **Pre-requisite**: start the agent-environment Docker service before
    running this task (default URL is ``http://localhost:1984``).
    """

    name_prefix = "MCPAtlasInfer"
    log_subdir = "logs/infer"
    output_subdir = "predictions"

    # -- init --------------------------------------------------------------

    def __init__(self, cfg: ConfigDict) -> None:
        super().__init__(cfg)
        dataset_cfg = self.dataset_cfgs[0]
        args = dataset_cfg.get("args", {}) or {}

        self.mcp_server_url = str(
            args.get("mcp_server_url", DEFAULT_MCP_SERVER_URL)
        )
        self.filter_enabled_servers = bool(
            args.get("filter_enabled_servers", True)
        )
        self.max_steps = int(args.get("max_steps", 100))
        self.max_tool_calls = int(args.get("max_tool_calls", 100))
        self.request_timeout = float(args.get("request_timeout", 60.0))
        self.list_tools_timeout = float(args.get("list_tools_timeout", 180.0))
        self.use_system_prompt = bool(args.get("use_system_prompt", False))

        # Context window management (ported from mcp-atlas)
        self.context_window_management = str(
            args.get("context_window_management", "")
        )
        self.tool_output_cap = args.get("tool_output_cap")  # None = uncapped

        # Model inference config
        infer_cfg: Dict[str, Any] = self.model_cfg.get("infer_cfg") or {}
        self._model_temperature = float(infer_cfg.get("temperature", 0.0))
        self._model_max_tokens = int(infer_cfg.get("max_tokens", 2048))
        self._model_timeout = int(infer_cfg.get("timeout", 120))
        self._tool_choice = str(
            infer_cfg.get("tool_choice")
            or self.model_cfg.get("tool_choice", "auto")
        )

        # Internal state
        self._client: Optional[MCPAtlasClient] = None
        self._enabled_servers: Optional[List[str]] = None
        self._tool_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._excluded_tasks: List[Dict[str, Any]] = []

        self.logger.info(
            "MCPAtlasInferTask initialized: mcp_server_url=%s, "
            "filter_enabled_servers=%s, max_steps=%d, max_tool_calls=%d, "
            "request_timeout=%.1f, use_system_prompt=%s, "
            "context_window_management=%s, tool_output_cap=%s, "
            "model_temperature=%.2f, model_max_tokens=%d, model_timeout=%d",
            self.mcp_server_url,
            self.filter_enabled_servers,
            self.max_steps,
            self.max_tool_calls,
            self.request_timeout,
            self.use_system_prompt,
            self.context_window_management or "(off)",
            str(self.tool_output_cap),
            self._model_temperature,
            self._model_max_tokens,
            self._model_timeout,
        )

    # -- properties --------------------------------------------------------

    @property
    def client(self) -> MCPAtlasClient:
        if self._client is None:
            self._client = MCPAtlasClient(
                base_url=self.mcp_server_url,
                request_timeout=self.request_timeout,
                list_tools_timeout=self.list_tools_timeout,
            )
        return self._client

    # -- BaseTask interface ------------------------------------------------

    def get_command(self, cfg_path: str, template: str) -> str:
        sys.path.append(os.getcwd())
        python = sys.executable
        command = f"{python} {__file__} {cfg_path}"
        return template.format(task_cmd=command)

    def run(self, task_state_manager: TaskStateManager) -> None:
        self.logger.info("Task %s", task_abbr_from_cfg(self.cfg))
        self.task_state_manager = task_state_manager

        # ---- 1. preflight -------------------------------------------------
        self._preflight()

        # ---- 2. load dataset (swebench pattern) ---------------------------
        dataset_cfg = self.dataset_cfgs[0]
        self.logger.info(
            "Loading dataset: type=%s abbr=%s path=%s",
            dataset_cfg.get("type"),
            dataset_cfg.get("abbr"),
            dataset_cfg.get("path"),
        )
        dataset = build_dataset_from_cfg(dataset_cfg)
        data = list(dataset.test)
        self.logger.info(
            "Dataset loaded: %d raw samples, columns=%s",
            len(data),
            list(data[0].keys()) if data else "N/A",
        )

        # Filter by enabled servers (matching evalscope's sample_filter)
        limit = self.cfg.get("limit") or dataset_cfg.get("args", {}).get("limit")
        if self.filter_enabled_servers and self._enabled_servers:
            enabled_set = set(self._enabled_servers)
            filtered: List[Dict[str, Any]] = []
            for row in data:
                required = _extract_required_servers(
                    _field(row, "TRAJECTORY", "trajectory") or "[]"
                )
                missing = [s for s in required if s not in enabled_set]
                if not missing:
                    filtered.append(row)
                else:
                    self._excluded_tasks.append({
                        "task_id": _field(row, "TASK", "task", "task_id"),
                        "missing_servers": missing,
                    })
                    self.logger.warning(
                        "Skipping MCP-Atlas task %s: missing servers %s",
                        _field(row, "TASK", "task", "task_id"), missing,
                    )
            self.logger.info(
                "Server filter: %d -> %d samples (%d excluded due to missing servers)",
                len(data), len(filtered), len(data) - len(filtered),
            )
            data = filtered
        else:
            self.logger.info(
                "Server filter skipped (filter_enabled_servers=%s, "
                "enabled_servers=%s)",
                self.filter_enabled_servers,
                self._enabled_servers,
            )

        if limit and limit > 0:
            self.logger.info("Applying sample limit: %d -> %d", len(data), min(len(data), limit))
            data = data[:limit]

        if not data:
            self.logger.warning("No samples to run inference on.")
            self._save_predictions({})
            return

        total = len(data)
        task_state_manager.update_task_state({
            "status": "inferencing",
            "total_count": total,
            "progress_description": "MCP-Atlas inference",
            "finish_count": 0,
        })

        # ---- 3. run agent loop per sample ---------------------------------
        predictions: Dict[str, Dict[str, Any]] = {}
        pbar = tqdm(total=total, desc="MCP-Atlas infer", unit="sample")

        for idx, sample in enumerate(data):
            self.logger.info(
                "--- Sample %d/%d: task_id=%s ---",
                idx + 1,
                total,
                _field(sample, "TASK", "task", "task_id"),
            )
            result = self._run_sample(sample)
            task_id = result["task_id"]
            predictions[task_id] = result
            pbar.update(1)
            task_state_manager.update_task_state({"finish_count": idx + 1})
            self.logger.info(
                "Sample %s done: tool_calls=%d, server_failures=%d, "
                "answer_len=%d",
                task_id,
                result.get("tool_calls", 0),
                len(result.get("server_failures", {})),
                len(result.get("final_answer", "")),
            )

        pbar.close()

        # ---- 4. save predictions ------------------------------------------
        self._save_predictions(predictions)

    # -- preflight ---------------------------------------------------------

    def _preflight(self) -> None:
        """Fetch enabled servers and tool catalogue from agent-environment."""
        self.logger.info(
            "Preflight: fetching enabled servers from %s",
            self.mcp_server_url,
        )
        try:
            self._enabled_servers = self.client.enabled_servers()
            self.logger.info(
                "Enabled MCP servers (%d): %s",
                len(self._enabled_servers),
                self._enabled_servers,
            )
        except Exception as exc:
            raise RuntimeError(
                "MCP-Atlas agent-environment is not available. Start the "
                "Docker service so that "
                f"{self.mcp_server_url}/enabled-servers is reachable. "
                f"Original error: {exc}"
            ) from exc

        self.logger.info(
            "Preflight: fetching tool catalogue from %s",
            self.mcp_server_url,
        )
        try:
            raw_tools = self.client.list_tools()
            self._tool_map = {
                str(t["name"]): t for t in raw_tools if isinstance(t, dict)
            }
            self.logger.info(
                "Loaded %d tools: %s",
                len(self._tool_map),
                sorted(self._tool_map.keys()),
            )
        except Exception as exc:
            raise RuntimeError(
                "MCP-Atlas agent-environment tool catalogue is not available. "
                "Check that the Docker service is running and "
                f"{self.mcp_server_url}/list-tools is reachable. "
                f"Original error: {exc}"
            ) from exc

    # -- per-sample inference ----------------------------------------------

    def _run_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent loop on a single sample.

        Follows mcp-atlas's record_to_sample flow: extract fields,
        construct messages, build tool payload, run agent loop.
        """
        task_id = str(_field(sample, "TASK", "task", "task_id") or "")
        prompt = str(_field(sample, "PROMPT", "prompt") or "")
        enabled_tools = _parse_enabled_tools(
            _field(sample, "ENABLED_TOOLS", "enabled_tools") or "[]"
        )
        claims = _extract_claims(
            _field(sample, "GTFA_CLAIMS", "gtfa_claims", "rubrics") or "[]"
        )
        trajectory = _field(sample, "TRAJECTORY", "trajectory") or "[]"

        self.logger.info(
            "[%s] Extracted fields: prompt_len=%d, enabled_tools=%d (%s), "
            "claims=%d, trajectory_msgs=%d",
            task_id,
            len(prompt),
            len(enabled_tools),
            enabled_tools,
            len(claims),
            len(_maybe_parse_json(trajectory, default=[])) if isinstance(trajectory, str) else 0,
        )

        # Construct messages using build_messages (ported from mcp-atlas)
        system_prompt = DEFAULT_SYSTEM_PROMPT if self.use_system_prompt else None
        messages = build_messages(prompt, system_prompt)

        if self.use_system_prompt:
            self.logger.info(
                "[%s] System prompt prepended (total messages=%d, prompt_len=%d)",
                task_id,
                len(messages),
                len(messages[0]["content"]) if messages else 0,
            )
        else:
            self.logger.info(
                "[%s] System prompt disabled, using raw prompt (len=%d)",
                task_id,
                len(prompt),
            )

        # Build tools payload for the model
        tools_payload = self._build_tools_payload(enabled_tools)
        self.logger.info(
            "[%s] Built tools payload: %d tools provided to model",
            task_id,
            len(tools_payload),
        )

        # Run agent loop
        (final_answer, tool_call_count, server_failures,
         full_trajectory) = self._agent_loop(
            messages, tools_payload, enabled_tools, task_id,
        )

        self.logger.info(
            "[%s] Agent loop finished: final_answer_len=%d, "
            "tool_call_count=%d, server_failures=%s, trajectory_msgs=%d",
            task_id,
            len(final_answer),
            tool_call_count,
            list(server_failures.keys()) if server_failures else "none",
            len(full_trajectory),
        )

        return {
            "task_id": task_id,
            "prompt": prompt,
            "model_name_or_path": model_abbr_from_cfg(self.model_cfg),
            "final_answer": final_answer,
            "tool_calls": tool_call_count,
            "server_failures": server_failures,
            # Store claims and trajectory for eval to use
            "gtfa_claims": claims,
            "trajectory": trajectory,
            "enabled_tools": enabled_tools,
            # Full trajectory (complete message history) for diagnostics
            "raw_conversation_history": full_trajectory,
        }

    # -- tools --------------------------------------------------------------

    def _build_tools_payload(
        self, tool_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Convert tool names to OpenAI-style tool definitions."""
        if not self._tool_map:
            self.logger.warning("Tool map is empty, no tools available.")
            return []
        tools: List[Dict[str, Any]] = []
        for name in tool_names:
            raw = self._tool_map.get(name)
            if raw is None:
                self.logger.warning(
                    "Tool '%s' not found in tool catalogue, skipping.", name
                )
                continue
            tools.append(mcp_tool_to_tool_info(raw))
        return tools

    # -- agent loop --------------------------------------------------------

    def _agent_loop(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        enabled_tools: List[str],
        task_id: str = "",
    ) -> Tuple[str, int, Dict[str, str], List[Dict[str, Any]]]:
        """Run the agent conversation loop with the MCP-Atlas service.

        Ported from mcp-atlas agent-eval.ts ``runMcpAgent``.  Key features:

        - **Tool call detection**: Uses ``detect_tool_calls`` for robust
          structure validation (matching the Zod schema in mcp-atlas).
        - **LLM retry**: Transient errors (503, 429, timeout) are retried
          up to ``MAX_LLM_RETRIES`` times with exponential backoff.
        - **Error recovery**: Tool-call failures are fed back to the model
          as tool results so the model can recover.
        - **Context compaction**: Optional ``compact`` mode truncates old
          tool results when context grows large.
        - **Tool output cap**: Optional per-result character limit.

        Each MCP tool call is executed in a child process via
        :func:`_call_tool_subprocess` for process isolation.

        Returns:
            (final_answer, tool_call_count, server_failures, trajectory)
        """
        all_messages: List[Dict[str, Any]] = list(messages)

        tool_call_count = 0
        server_failures: Dict[str, str] = {}
        final_answer = ""
        reached_max_turns = True

        self.logger.info(
            "[%s] Agent loop start: max_steps=%d, max_tool_calls=%d, "
            "tools_available=%d, messages=%d",
            task_id,
            self.max_steps,
            self.max_tool_calls,
            len(tools),
            len(all_messages),
        )

        for step in range(self.max_steps):
            # Check tool call limit before next LLM call
            if self.max_tool_calls and tool_call_count >= self.max_tool_calls:
                reached_max_turns = False
                self.logger.warning(
                    "[%s] Step %d: tool call limit reached (%d) before LLM call",
                    task_id, step + 1, self.max_tool_calls,
                )
                break

            # Apply context compaction if enabled
            messages_to_send = all_messages
            if self.context_window_management == "compact":
                messages_to_send = compact_messages(all_messages, step + 1)
                if messages_to_send is not all_messages:
                    orig_chars = sum(len(json.dumps(m)) for m in all_messages)
                    compact_chars = sum(len(json.dumps(m)) for m in messages_to_send)
                    saved = orig_chars - compact_chars
                    if saved > 0:
                        self.logger.info(
                            "[%s] Compact: %d → %d chars (saved %d, %.1f%%)",
                            task_id, orig_chars, compact_chars, saved,
                            saved / orig_chars * 100,
                        )

            self.logger.info(
                "[%s] Step %d: calling model (messages=%d, tools=%d)...",
                task_id,
                step + 1,
                len(messages_to_send),
                len(tools) if step == 0 else 0,
            )

            # Call LLM with retry logic (ported from mcp-atlas)
            response = self._call_model_with_retry(
                messages_to_send, tools if step == 0 else [], task_id, step,
            )

            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "unknown")

            # Robust tool call detection (ported from mcp-atlas)
            tool_calls = detect_tool_calls(message)

            if tool_calls:
                tool_names_in_msg = [
                    tc["function"]["name"] for tc in tool_calls
                ]
                self.logger.info(
                    "[%s] Step %d: model requested %d tool call(s): %s "
                    "(finish_reason=%s)",
                    task_id,
                    step + 1,
                    len(tool_calls),
                    tool_names_in_msg,
                    finish_reason,
                )
                all_messages.append(message)

                for tc in tool_calls:
                    # Check tool call limit before executing
                    if self.max_tool_calls and tool_call_count >= self.max_tool_calls:
                        reached_max_turns = False
                        self.logger.warning(
                            "[%s] Step %d: tool call limit reached (%d)",
                            task_id, step + 1, self.max_tool_calls,
                        )
                        break

                    func = tc.get("function", {})
                    tool_name = str(func.get("name", ""))
                    tool_args = _maybe_parse_json(
                        func.get("arguments", "{}"), default={}
                    )
                    if not isinstance(tool_args, dict):
                        tool_args = {}

                    # Apply tool-specific pruning (ported from mcp-atlas)
                    tool_args = pruned_tool_call(tool_name, tool_args)

                    self.logger.info(
                        "[%s] Step %d: executing tool '%s' args=%s",
                        task_id,
                        step + 1,
                        tool_name,
                        json.dumps(tool_args, ensure_ascii=False),
                    )

                    # Short-circuit failed servers (matching mcp-atlas)
                    server = _tool_name_to_server(tool_name)
                    if server in server_failures:
                        result = _server_unavailable_message(
                            server, server_failures[server]
                        )
                        self.logger.info(
                            "[%s] Step %d: server '%s' already failed, "
                            "short-circuiting tool '%s'",
                            task_id, step + 1, server, tool_name,
                        )
                        tool_call_count += 1
                    else:
                        try:
                            # Execute tool call via subprocess for isolation
                            result = _call_tool_subprocess(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                base_url=self.mcp_server_url,
                                timeout=self.request_timeout,
                            )
                            tool_call_count += 1

                            # Apply tool output cap if configured
                            if self.tool_output_cap is not None:
                                result = cap_tool_content(
                                    result, self.tool_output_cap
                                )

                            self.logger.info(
                                "[%s] Step %d: tool '%s' succeeded "
                                "(total_calls=%d, result_len=%d)",
                                task_id, step + 1,
                                tool_name, tool_call_count, len(result),
                            )
                        except MCPAtlasServerUnavailable as exc:
                            # Error recovery: feed error back to model
                            # (ported from mcp-atlas agent-eval.ts)
                            server_failures[exc.server_name] = exc.message
                            error_msg = exc.message.split("\n")[0]
                            result = f"Error: {error_msg}"
                            tool_call_count += 1
                            self.logger.warning(
                                "[%s] Step %d: server '%s' unavailable "
                                "for tool '%s': %s — feeding error back to model",
                                task_id, step + 1,
                                exc.server_name, tool_name, error_msg[:200],
                            )

                    # Build tool result message (OpenAI format)
                    tool_result_msg: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    }
                    all_messages.append(tool_result_msg)

                if self.max_tool_calls and tool_call_count >= self.max_tool_calls:
                    final_answer = (
                        "MCP-Atlas tool call limit exceeded "
                        f"({self.max_tool_calls})."
                    )
                    reached_max_turns = False
                    self.logger.warning(
                        "[%s] Tool call limit reached, stopping agent loop.",
                        task_id,
                    )
                    break
                continue

            # Model returned a final text answer — natural completion
            reached_max_turns = False
            final_answer = str(
                message.get("content", "") or response.get("content", "")
            )
            self.logger.info(
                "[%s] Step %d: model returned final answer "
                "(finish_reason=%s, answer_len=%d)",
                task_id, step + 1, finish_reason, len(final_answer),
            )
            break
        else:
            final_answer = final_answer or "Agent loop exceeded max steps."
            self.logger.warning(
                "[%s] Agent loop exceeded max_steps=%d, stopping.",
                task_id, self.max_steps,
            )

        self.logger.info(
            "[%s] Agent loop summary: steps=%d, tool_calls=%d, "
            "server_failures=%d, final_answer_len=%d",
            task_id,
            min(step + 1, self.max_steps),
            tool_call_count,
            len(server_failures),
            len(final_answer),
        )
        return final_answer, tool_call_count, server_failures, all_messages

    # -- model call with retry (ported from mcp-atlas litellm-strategy.ts) --

    def _call_model_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        task_id: str,
        step: int,
    ) -> Dict[str, Any]:
        """Call the LLM API with retry logic for transient errors.

        Ported from mcp-atlas litellm-strategy.ts retry logic:
        - Retries on 500, 502, 503, 429, and timeouts
        - Exponential backoff for 429, fixed delays for others
        - Up to MAX_LLM_RETRIES attempts
        """
        last_error: Optional[Exception] = None

        for attempt in range(MAX_LLM_RETRIES):
            try:
                return self._call_model(messages, tools)
            except Exception as exc:
                last_error = exc
                if is_retryable_error(exc) and attempt < MAX_LLM_RETRIES - 1:
                    delay = get_retry_delay(exc, attempt)
                    self.logger.warning(
                        "[%s] Step %d: LLM call failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        task_id, step + 1, attempt + 1, MAX_LLM_RETRIES,
                        delay, str(exc)[:200],
                    )
                    time.sleep(delay)
                    continue
                raise

        # Should not reach here, but just in case
        raise last_error  # type: ignore

    # -- model call --------------------------------------------------------

    def _call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call the LLM API (OpenAI-compatible chat completions)."""
        url = (
            self.model_cfg.get("api_url")
            or self.model_cfg.get("url")
            or os.getenv("AIS_BENCH_API_URL", "")
        )
        api_key = (
            self.model_cfg.get("api_key")
            or self.model_cfg.get("key")
            or os.getenv("AIS_BENCH_API_KEY", "")
        )
        model = (
            self.model_cfg.get("model")
            or os.getenv("AIS_BENCH_MODEL")
            or os.getenv("MODEL_NAME")
            or ""
        )
        if not str(model).strip():
            raise ValueError(
                "MCP-Atlas model name is empty. Set model in "
                "ais_bench/configs/mcp_atlas_examples/mcp_atlas.py or export "
                "AIS_BENCH_MODEL to the model name served by the OpenAI-compatible endpoint."
            )
        if not str(url).strip():
            raise ValueError(
                "MCP-Atlas API URL is empty. Set url/api_url in the config or export AIS_BENCH_API_URL."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self._model_temperature,
            "max_tokens": self._model_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = self._tool_choice

        self.logger.info(
            "Calling model: url=%s, model=%s, messages=%d, tools=%d, "
            "temperature=%.2f, max_tokens=%d",
            url, model, len(messages), len(tools),
            self._model_temperature, self._model_max_tokens,
        )

        resp = requests.post(
            f'{url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=self._model_timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            hint = ""
            if tools and resp.status_code == 400:
                hint = (
                    " If this is a vLLM endpoint, restart it with OpenAI tool "
                    "calling enabled, for example: --enable-auto-tool-choice "
                    "--tool-call-parser <parser-compatible-with-your-model>."
                )
            self.logger.error(
                "Model API request failed: status=%s, body=%s%s",
                resp.status_code,
                resp.text[:4000],
                hint,
            )
            raise exc
        response_json = resp.json()

        usage = response_json.get("usage", {})
        choice = (response_json.get("choices") or [{}])[0]
        self.logger.info(
            "Model response: finish_reason=%s, prompt_tokens=%s, "
            "completion_tokens=%s, total_tokens=%s",
            choice.get("finish_reason", "unknown"),
            usage.get("prompt_tokens", "N/A"),
            usage.get("completion_tokens", "N/A"),
            usage.get("total_tokens", "N/A"),
        )

        return response_json

    # -- save predictions --------------------------------------------------

    def _save_predictions(
        self, predictions: Dict[str, Dict[str, Any]]
    ) -> None:
        """Write predictions to the output path (following swebench pattern)."""
        dataset_cfg = self.dataset_cfgs[0]
        model_abbr = model_abbr_from_cfg(self.model_cfg)

        out_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, self.output_subdir),
            file_extension="json",
        )
        mkdir_or_exist(osp.dirname(out_path))

        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
            f.write("\n")
        self.logger.info(
            "Predictions saved to %s (%d samples, %d bytes)",
            out_path,
            len(predictions),
            osp.getsize(out_path) if osp.isfile(out_path) else 0,
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP-Atlas Inference Task")
    parser.add_argument("config", help="Config file path")
    return parser.parse_args()


if __name__ == "__main__":
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)

    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg.get("cli_args", {}).get("debug", False),
    )

    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()

    task_state_manager.update_task_state({
        "status": "start",
        "task_log_path": osp.join(
            MCPAtlasInferTask.log_subdir, f"{task_abbr_from_cfg(cfg)}.out"
        ),
    })

    start_time = time.perf_counter()
    try:
        task = MCPAtlasInferTask(cfg)
        task.run(task_state_manager)
    except Exception:
        task_state_manager.update_task_state({"status": "error"})
        raise

    end_time = time.perf_counter()
    logger.info(
        "MCP-Atlas inference time elapsed: %.2fs", end_time - start_time
    )
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
