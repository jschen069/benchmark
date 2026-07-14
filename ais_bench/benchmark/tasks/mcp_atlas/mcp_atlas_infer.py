"""MCP-Atlas inference task.

Runs the agent loop against the MCP-Atlas agent-environment Docker service,
calls live MCP tools via subprocess isolation, and saves final answers
as predictions.  The predictions are later scored by
:class:`MCPAtlasEvalTask`.

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
    _call_tool_subprocess,
    _extract_claims,
    _extract_required_servers,
    _field,
    _maybe_parse_json,
    _parse_enabled_tools,
    _server_unavailable_message,
    _tool_name_to_server,
    mcp_tool_to_tool_info,
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

        # Model inference config
        infer_cfg: Dict[str, Any] = self.model_cfg.get("infer_cfg") or {}
        self._model_temperature = float(infer_cfg.get("temperature", 0.0))
        self._model_max_tokens = int(infer_cfg.get("max_tokens", 2048))
        self._model_timeout = int(infer_cfg.get("timeout", 120))

        # Internal state
        self._client: Optional[MCPAtlasClient] = None
        self._enabled_servers: Optional[List[str]] = None
        self._tool_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._excluded_tasks: List[Dict[str, Any]] = []

        self.logger.info(
            "MCPAtlasInferTask initialized: mcp_server_url=%s, "
            "filter_enabled_servers=%s, max_steps=%d, max_tool_calls=%d, "
            "request_timeout=%.1f, use_system_prompt=%s, "
            "model_temperature=%.2f, model_max_tokens=%d, model_timeout=%d",
            self.mcp_server_url,
            self.filter_enabled_servers,
            self.max_steps,
            self.max_tool_calls,
            self.request_timeout,
            self.use_system_prompt,
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

        Follows evalscope's record_to_sample flow: extract fields,
        construct prompt, build tool payload, run agent loop.
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

        # Construct prompt following evalscope's pattern
        input_text = prompt
        if self.use_system_prompt:
            input_text = f"{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"
            self.logger.info(
                "[%s] System prompt prepended (total prompt_len=%d)",
                task_id,
                len(input_text),
            )
        else:
            self.logger.info(
                "[%s] System prompt disabled, using raw prompt (len=%d)",
                task_id,
                len(input_text),
            )

        # Build tools payload for the model
        tools_payload = self._build_tools_payload(enabled_tools)
        self.logger.info(
            "[%s] Built tools payload: %d tools provided to model",
            task_id,
            len(tools_payload),
        )

        # Run agent loop
        final_answer, tool_call_count, server_failures = self._agent_loop(
            input_text, tools_payload, enabled_tools, task_id,
        )

        self.logger.info(
            "[%s] Agent loop finished: final_answer_len=%d, "
            "tool_call_count=%d, server_failures=%s",
            task_id,
            len(final_answer),
            tool_call_count,
            list(server_failures.keys()) if server_failures else "none",
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
        user_prompt: str,
        tools: List[Dict[str, Any]],
        enabled_tools: List[str],
        task_id: str = "",
    ) -> Tuple[str, int, Dict[str, str]]:
        """Run the agent conversation loop with the MCP-Atlas service.

        Each MCP tool call is executed in a child process via
        :func:`_call_tool_subprocess` for process isolation.

        Returns:
            (final_answer, tool_call_count, server_failures)
        """
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        tool_call_count = 0
        server_failures: Dict[str, str] = {}
        final_answer = ""

        self.logger.info(
            "[%s] Agent loop start: max_steps=%d, max_tool_calls=%d, "
            "tools_available=%d",
            task_id,
            self.max_steps,
            self.max_tool_calls,
            len(tools),
        )

        for step in range(self.max_steps):
            self.logger.info(
                "[%s] Step %d: calling model (messages=%d, tools=%d)...",
                task_id,
                step + 1,
                len(messages),
                len(tools) if step == 0 else 0,  # tools only sent on step 0
            )
            response = self._call_model(messages, tools if step == 0 else [])

            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "unknown")

            # Model wants to call a tool
            if message.get("tool_calls"):
                tool_names_in_msg = [
                    tc.get("function", {}).get("name", "?")
                    for tc in message["tool_calls"]
                ]
                self.logger.info(
                    "[%s] Step %d: model requested %d tool call(s): %s "
                    "(finish_reason=%s)",
                    task_id,
                    step + 1,
                    len(message["tool_calls"]),
                    tool_names_in_msg,
                    finish_reason,
                )
                messages.append(message)

                for tc in message["tool_calls"]:
                    if tool_call_count >= self.max_tool_calls:
                        self.logger.warning(
                            "[%s] Step %d: tool call limit reached (%d)",
                            task_id,
                            step + 1,
                            self.max_tool_calls,
                        )
                        break

                    func = tc.get("function", {})
                    tool_name = str(func.get("name", ""))
                    tool_args = _maybe_parse_json(
                        func.get("arguments", "{}"), default={}
                    )
                    if not isinstance(tool_args, dict):
                        tool_args = {}

                    self.logger.info(
                        "[%s] Step %d: executing tool '%s' args=%s",
                        task_id,
                        step + 1,
                        tool_name,
                        json.dumps(tool_args, ensure_ascii=False),
                    )

                    # Short-circuit failed servers (matching evalscope)
                    server = _tool_name_to_server(tool_name)
                    if server in server_failures:
                        result = _server_unavailable_message(
                            server, server_failures[server]
                        )
                        self.logger.info(
                            "[%s] Step %d: server '%s' already failed, "
                            "short-circuiting tool '%s'",
                            task_id,
                            step + 1,
                            server,
                            tool_name,
                        )
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
                            self.logger.info(
                                "[%s] Step %d: tool '%s' succeeded "
                                "(total_calls=%d, result_len=%d)",
                                task_id,
                                step + 1,
                                tool_name,
                                tool_call_count,
                                len(result),
                            )
                        except MCPAtlasServerUnavailable as exc:
                            server_failures[exc.server_name] = exc.message
                            result = _server_unavailable_message(
                                exc.server_name, exc.message
                            )
                            self.logger.warning(
                                "[%s] Step %d: server '%s' unavailable "
                                "for tool '%s': %s",
                                task_id,
                                step + 1,
                                exc.server_name,
                                tool_name,
                                exc.message[:200],
                            )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    })

                if tool_call_count >= self.max_tool_calls:
                    final_answer = (
                        "MCP-Atlas tool call limit exceeded "
                        f"({self.max_tool_calls})."
                    )
                    self.logger.warning(
                        "[%s] Tool call limit reached, stopping agent loop.",
                        task_id,
                    )
                    break
                continue

            # Model returned a final text answer
            final_answer = str(
                message.get("content", "") or response.get("content", "")
            )
            self.logger.info(
                "[%s] Step %d: model returned final answer "
                "(finish_reason=%s, answer_len=%d)",
                task_id,
                step + 1,
                finish_reason,
                len(final_answer),
            )
            break
        else:
            final_answer = final_answer or "Agent loop exceeded max steps."
            self.logger.warning(
                "[%s] Agent loop exceeded max_steps=%d, stopping.",
                task_id,
                self.max_steps,
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
        return final_answer, tool_call_count, server_failures

    # -- model call --------------------------------------------------------

    def _call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call the LLM API (OpenAI-compatible chat completions)."""
        url = self.model_cfg.get("api_url", self.model_cfg.get("url", ""))
        api_key = self.model_cfg.get("api_key", self.model_cfg.get("key", ""))
        model = self.model_cfg.get("model", self.model_cfg.get("abbr", ""))

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
        resp.raise_for_status()
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
