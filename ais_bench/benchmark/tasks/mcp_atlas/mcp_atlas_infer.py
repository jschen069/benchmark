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
from ais_bench.benchmark.utils.core.abbr import (
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger

from ais_bench.benchmark.tasks.mcp_atlas.utils import (
    DATASET_ID,
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

        # ---- 2. load dataset ----------------------------------------------
        data = self._load_dataset()
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
            result = self._run_sample(sample)
            task_id = result["task_id"]
            predictions[task_id] = result
            pbar.update(1)
            task_state_manager.update_task_state({"finish_count": idx + 1})

        pbar.close()

        # ---- 4. save predictions ------------------------------------------
        self._save_predictions(predictions)

    # -- preflight ---------------------------------------------------------

    def _preflight(self) -> None:
        """Fetch enabled servers and tool catalogue from agent-environment."""
        try:
            self._enabled_servers = self.client.enabled_servers()
            self.logger.info("Enabled MCP servers: %s", self._enabled_servers)
        except Exception as exc:
            raise RuntimeError(
                "MCP-Atlas agent-environment is not available. Start the "
                "Docker service so that "
                f"{self.mcp_server_url}/enabled-servers is reachable. "
                f"Original error: {exc}"
            ) from exc

        try:
            raw_tools = self.client.list_tools()
            self._tool_map = {
                str(t["name"]): t for t in raw_tools if isinstance(t, dict)
            }
            self.logger.info("Loaded %d tools.", len(self._tool_map))
        except Exception as exc:
            raise RuntimeError(
                "MCP-Atlas agent-environment tool catalogue is not available. "
                "Check that the Docker service is running and "
                f"{self.mcp_server_url}/list-tools is reachable. "
                f"Original error: {exc}"
            ) from exc

    # -- dataset loading ---------------------------------------------------

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load and filter the MCP-Atlas dataset.

        Follows evalscope's record_to_sample pattern: extracts key fields
        from each raw dataset row (TASK, PROMPT, ENABLED_TOOLS,
        GTFA_CLAIMS, TRAJECTORY) and filters by enabled MCP servers.
        """
        dataset_cfg = self.dataset_cfgs[0]
        args = dataset_cfg.get("args", {}) or {}

        local_path = args.get("local_path", "")
        limit = self.cfg.get("limit") or args.get("limit")

        if local_path:
            data = self._load_from_local(local_path)
        else:
            data = self._load_from_hub()

        # Filter by enabled servers (matching evalscope's sample_filter)
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
                "Filtered %d tasks due to missing servers. %d remaining.",
                len(data) - len(filtered), len(filtered),
            )
            data = filtered

        if limit and limit > 0:
            data = data[:limit]

        return data

    def _load_from_hub(self) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )
        ds = hf_load(DATASET_ID, split="train")
        return [dict(row) for row in ds]

    def _load_from_local(self, path: str) -> List[Dict[str, Any]]:
        import csv

        file_path = osp.join(path, "mcp_atlas.csv")
        if not osp.exists(file_path):
            raise FileNotFoundError(
                f"MCP-Atlas CSV not found at {file_path}"
            )
        with open(file_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

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

        # Construct prompt following evalscope's pattern
        input_text = prompt
        if self.use_system_prompt:
            input_text = f"{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"

        # Build tools payload for the model
        tools_payload = self._build_tools_payload(enabled_tools)

        # Run agent loop
        final_answer, tool_call_count, server_failures = self._agent_loop(
            input_text, tools_payload, enabled_tools
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
            return []
        tools: List[Dict[str, Any]] = []
        for name in tool_names:
            raw = self._tool_map.get(name)
            if raw is None:
                continue
            tools.append(mcp_tool_to_tool_info(raw))
        return tools

    # -- agent loop --------------------------------------------------------

    def _agent_loop(
        self,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        enabled_tools: List[str],
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

        for _ in range(self.max_steps):
            response = self._call_model(messages, tools)

            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message", {})

            # Model wants to call a tool
            if message.get("tool_calls"):
                messages.append(message)

                for tc in message["tool_calls"]:
                    if tool_call_count >= self.max_tool_calls:
                        break

                    func = tc.get("function", {})
                    tool_name = str(func.get("name", ""))
                    tool_args = _maybe_parse_json(
                        func.get("arguments", "{}"), default={}
                    )
                    if not isinstance(tool_args, dict):
                        tool_args = {}

                    # Short-circuit failed servers (matching evalscope)
                    server = _tool_name_to_server(tool_name)
                    if server in server_failures:
                        result = _server_unavailable_message(
                            server, server_failures[server]
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
                        except MCPAtlasServerUnavailable as exc:
                            server_failures[exc.server_name] = exc.message
                            result = _server_unavailable_message(
                                exc.server_name, exc.message
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
                    break
                continue

            # Model returned a final text answer
            final_answer = str(
                message.get("content", "") or response.get("content", "")
            )
            break
        else:
            final_answer = final_answer or "Agent loop exceeded max steps."

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

        resp = requests.post(
            f'{url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=self._model_timeout,
        )
        resp.raise_for_status()
        return resp.json()

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
        self.logger.info("Predictions saved to %s (%d samples)", out_path, len(predictions))


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
