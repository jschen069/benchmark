"""MCP-Atlas evaluation task.

Loads predictions from the infer step, scores final answers with an
LLM judge on a per-claim basis, and computes aggregated metrics
(coverage_score and pass_rate).  Follows evalscope's
:meth:`MCPAtlasAdapter.llm_match_score` and
:meth:`MCPAtlasAdapter.aggregate_scores` patterns.

"""

import argparse
import json
import os
import os.path as osp
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist
from tqdm import tqdm

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.core.abbr import (
    dataset_abbr_from_cfg,
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger

from ais_bench.benchmark.tasks.mcp_atlas.utils import (
    DATASET_ID,
    _claim_judge_prompt,
    _extract_claims,
    _field,
    _parse_claim_judge_response,
)


@TASKS.register_module()
class MCPAtlasEvalTask(BaseTask):
    """Evaluate MCP-Atlas predictions with LLM-as-judge claim scoring.

    Reads predictions produced by :class:`MCPAtlasInferTask`, loads the
    dataset for ground-truth claims, runs the LLM judge on each claim,
    and computes aggregated ``coverage_score`` and ``pass_rate``.

    Results are written to ``results/<model_abbr>/<dataset_abbr>.json``
    (summary) and ``results/<model_abbr>/<dataset_abbr>/details.json``
    (per-sample details).
    """

    name_prefix = "MCPAtlasEval"
    log_subdir = "logs/eval"
    output_subdir = "results"

    # -- init --------------------------------------------------------------

    def __init__(self, cfg: ConfigDict) -> None:
        super().__init__(cfg)
        dataset_cfg = self.dataset_cfgs[0]
        args = dataset_cfg.get("args", {}) or {}

        self.pass_threshold = float(args.get("pass_threshold", 0.75))

        # LLM judge configuration (all fields from config, with fallback to main model)
        judge_cfg: Dict[str, Any] = self.model_cfg.get("judge_model") or {}
        self._judge_model = judge_cfg.get("model", "")
        self._judge_api_url = judge_cfg.get("api_url", "")
        self._judge_api_key = judge_cfg.get("api_key", "")
        self._judge_temperature = float(judge_cfg.get("temperature", 0.0))
        self._judge_max_tokens = int(judge_cfg.get("max_tokens", 512))
        self._judge_timeout = int(judge_cfg.get("timeout", 120))

    # -- BaseTask interface ------------------------------------------------

    def get_command(self, cfg_path: str, template: str) -> str:
        sys.path.append(os.getcwd())
        python = sys.executable
        command = f"{python} {__file__} {cfg_path}"
        return template.format(task_cmd=command)

    def run(self, task_state_manager: TaskStateManager) -> None:
        self.logger.info("Task %s", task_abbr_from_cfg(self.cfg))
        self.task_state_manager = task_state_manager

        # ---- 1. load predictions ------------------------------------------
        predictions = self._load_predictions()
        if not predictions:
            self.logger.warning("No predictions found.")
            self._dump_results([])
            return

        # ---- 2. load dataset (for ground-truth claims) --------------------
        samples = self._load_dataset()
        sample_map: Dict[str, Dict[str, Any]] = {}
        for s in samples:
            tid = str(_field(s, "TASK", "task", "task_id") or "")
            sample_map[tid] = s

        # ---- 3. judge each prediction ------------------------------------
        total = len(predictions)
        task_state_manager.update_task_state({
            "status": "running",
            "total_count": total,
            "progress_description": "MCP-Atlas evaluation",
            "finish_count": 0,
        })

        results: List[Dict[str, Any]] = []
        pbar = tqdm(total=total, desc="MCP-Atlas eval", unit="sample")

        for idx, (task_id, pred) in enumerate(predictions.items()):
            result = self._evaluate_prediction(task_id, pred, sample_map)
            results.append(result)
            pbar.update(1)
            task_state_manager.update_task_state({"finish_count": idx + 1})

        pbar.close()

        # ---- 4. aggregate & save -----------------------------------------
        self._dump_results(results)

    # -- prediction loading ------------------------------------------------

    def _load_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Load predictions from the infer step.

        Follows swebench's pattern: tries the main output path first,
        then falls back to ``preds.json`` inside the output directory.
        """
        dataset_cfg = self.dataset_cfgs[0]
        pred_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, "predictions"),
            file_extension="json",
        )
        if not osp.isfile(pred_path):
            # Try fallback path (matches swebench pattern)
            pred_path_fallback = osp.join(
                osp.dirname(pred_path),
                osp.splitext(osp.basename(pred_path))[0],
                "preds.json",
            )
            if osp.isfile(pred_path_fallback):
                pred_path = pred_path_fallback
                self.logger.info("Using predictions from %s", pred_path)
            else:
                raise FileNotFoundError(
                    f"Predictions file not found: {pred_path} "
                    f"(or {pred_path_fallback}). Run infer first."
                )

        with open(pred_path) as f:
            raw_preds = json.load(f)
        if isinstance(raw_preds, dict):
            return raw_preds
        # If it's a list, index by task_id
        return {
            (p.get("task_id") or str(i)): p
            for i, p in enumerate(raw_preds)
        }

    # -- dataset loading ---------------------------------------------------

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the MCP-Atlas dataset for ground-truth claims."""
        dataset_cfg = self.dataset_cfgs[0]
        args = dataset_cfg.get("args", {}) or {}
        local_path = args.get("local_path", "")

        if local_path:
            return self._load_from_local(local_path)
        return self._load_from_hub()

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

    # -- per-sample scoring ------------------------------------------------

    def _evaluate_prediction(
        self,
        task_id: str,
        pred: Dict[str, Any],
        sample_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score a single prediction with the LLM judge.

        Follows evalscope's :meth:`MCPAtlasAdapter.llm_match_score`:
        extract claims from ground truth, judge each claim against the
        prediction's final answer, compute coverage_score and pass/fail.
        """
        final_answer = str(pred.get("final_answer") or "")
        prompt = str(pred.get("prompt") or "")

        # Try to get claims from prediction metadata first, then from dataset
        claims = list(pred.get("gtfa_claims") or [])
        if not claims and task_id in sample_map:
            sample = sample_map[task_id]
            claims = _extract_claims(
                _field(sample, "GTFA_CLAIMS", "gtfa_claims", "rubrics") or "[]"
            )

        # Judge each claim (matching evalscope's _judge_claim loop)
        claim_results: List[Dict[str, Any]] = []
        for claim in claims:
            cr = self._judge_claim(claim, final_answer)
            claim_results.append(cr)

        total_claims = len(claim_results)
        coverage_score = (
            sum(cr["score"] for cr in claim_results) / total_claims
            if total_claims else 0.0
        )
        passed = coverage_score >= self.pass_threshold

        return {
            "task_id": task_id,
            "prompt": prompt,
            "final_answer": final_answer,
            "tool_calls": pred.get("tool_calls", 0),
            "claims": claim_results,
            "total_claims": total_claims,
            "coverage_score": coverage_score,
            "pass": passed,
            "pass_threshold": self.pass_threshold,
        }

    # -- LLM judge ---------------------------------------------------------

    def _judge_claim(self, claim: str, response: str) -> Dict[str, Any]:
        """Score a single claim with the LLM judge.

        Follows evalscope's :meth:`MCPAtlasAdapter._judge_claim`:
        construct judge prompt → call LLM judge → parse response →
        map outcome to numeric score.
        """
        prompt = _claim_judge_prompt(claim, response)
        judge_response = self._call_judge(prompt)
        outcome, justification, confidence = _parse_claim_judge_response(
            judge_response
        )
        # Match evalscope's score mapping
        score_map = {
            "fulfilled": 1.0,
            "partially_fulfilled": 0.5,
            "not_fulfilled": 0.0,
        }
        return {
            "claim": claim,
            "coverage_outcome": outcome,
            "score": score_map.get(outcome, 0.0),
            "justification": justification,
            "confidence": confidence,
            "raw_judge_response": judge_response,
        }

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model via OpenAI-compatible API.

        Follows evalscope's :class:`LLMJudge.judge` pattern.
        All parameters come from the judge_model config dict,
        falling back to the main model config.
        """
        url = self._judge_api_url or self.model_cfg.get(
            "api_url", self.model_cfg.get("url", ""),
        )
        api_key = self._judge_api_key or self.model_cfg.get(
            "api_key", self.model_cfg.get("key", ""),
        )
        model = self._judge_model or self.model_cfg.get(
            "model", self.model_cfg.get("abbr", ""),
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._judge_temperature,
            "max_tokens": self._judge_max_tokens,
        }
        resp = requests.post(
            f'{url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=self._judge_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        return str(choice.get("message", {}).get("content", "") or "")

    # -- results -----------------------------------------------------------

    def _dump_results(self, results: List[Dict[str, Any]]) -> None:
        """Aggregate scores and write results (following evalscope's
        :meth:`MCPAtlasAdapter.aggregate_scores` pattern).

        Writes two files:
        - ``results/<model>/<dataset>.json`` — summary with aggregate scores
        - ``results/<model>/<dataset>/details.json`` — per-sample details
        """
        dataset_cfg = self.dataset_cfgs[0]
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
        model_abbr = model_abbr_from_cfg(self.model_cfg)

        out_dir = osp.join(self.work_dir, self.output_subdir, model_abbr)
        mkdir_or_exist(out_dir)
        out_detail_dir = osp.join(out_dir, dataset_abbr)
        mkdir_or_exist(out_detail_dir)

        # Aggregate scores (matching evalscope's aggregate_scores)
        n_samples = len(results)
        if n_samples > 0:
            avg_coverage = sum(
                r["coverage_score"] for r in results
            ) / n_samples
            pass_rate = sum(1 for r in results if r["pass"]) / n_samples
            fully_covered = sum(
                1 for r in results
                for cr in (r.get("claims") or [])
                if cr.get("score") == 1.0
            )
            partially_covered = sum(
                1 for r in results
                for cr in (r.get("claims") or [])
                if cr.get("score") == 0.5
            )
        else:
            avg_coverage = 0.0
            pass_rate = 0.0
            fully_covered = 0
            partially_covered = 0

        summary = {
            "total_count": n_samples,
            "coverage_score": round(avg_coverage, 4),
            "pass_rate": round(pass_rate, 4),
            "pass_threshold": self.pass_threshold,
            "fully_covered_claims": fully_covered,
            "partially_covered_claims": partially_covered,
        }

        # Write summary
        summary_path = osp.join(out_dir, f"{dataset_abbr}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info("Summary saved to %s", summary_path)

        # Write per-task details
        detail_path = osp.join(out_detail_dir, "details.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info("Details saved to %s", detail_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP-Atlas Eval Task")
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
            MCPAtlasEvalTask.log_subdir, f"{task_abbr_from_cfg(cfg)}.out"
        ),
    })

    start_time = time.perf_counter()
    try:
        task = MCPAtlasEvalTask(cfg)
        task.run(task_state_manager)
    except Exception:
        task_state_manager.update_task_state({"status": "error"})
        raise

    end_time = time.perf_counter()
    logger.info(
        "MCP-Atlas evaluation time elapsed: %.2fs", end_time - start_time
    )
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
