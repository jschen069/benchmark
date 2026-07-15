"""MCP-Atlas evaluation task.

Loads predictions from the infer step, scores final answers with an
LLM judge on a per-claim basis, and computes aggregated metrics
(coverage_score and pass_rate).

Ported from the upstream mcp-atlas ``score_claims.py``, adapted for
aisbench conventions.  Key improvements over the initial version:

- **Enhanced judge prompt**: Uses the detailed mcp-atlas prompt with
  numerical comparison guidelines and scoring criteria.
- **Structured JSON output**: Supports ``response_format: json_schema``
  for models that support it.
- **Improved response parsing**: More robust parsing of judge responses
  with multiple fallback strategies.
- **Rich statistics**: Generates ``coverage_stats`` JSON with mean_coverage,
  pass_rate at 0.50/0.75 thresholds, and evaluator model info.
- **Response truncation**: Handles oversized responses gracefully.

"""

import argparse
import json
import os
import os.path as osp
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist
from tqdm import tqdm

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.config.build import build_dataset_from_cfg
from ais_bench.benchmark.utils.core.abbr import (
    dataset_abbr_from_cfg,
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger

from ais_bench.benchmark.tasks.mcp_atlas.utils import (
    _claim_judge_prompt,
    _extract_claims,
    _field,
    _parse_claim_judge_response,
    compute_coverage_score,
    compute_coverage_stats,
    extract_claims,
    get_claim_evaluation_schema,
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

    Ported from mcp-atlas score_claims.py CoverageEvaluator.
    """

    name_prefix = "MCPAtlasEval"
    log_subdir = "logs/eval"
    output_subdir = "results"

    # Maximum response length to send to judge (from mcp-atlas)
    MAX_RESPONSE_CHARS = 500_000

    # -- init --------------------------------------------------------------

    def __init__(self, cfg: ConfigDict) -> None:
        super().__init__(cfg)
        dataset_cfg = self.dataset_cfgs[0]
        args = dataset_cfg.get("args", {}) or {}

        self.pass_threshold = float(args.get("pass_threshold", 0.75))

        # Whether to use structured JSON output (json_schema response_format)
        self.use_structured_output = bool(
            args.get("use_structured_output", False)
        )

        # LLM judge configuration (all fields from config, with fallback to main model)
        judge_cfg: Dict[str, Any] = self.model_cfg.get("judge_model") or {}
        self._judge_model = judge_cfg.get("model", "")
        self._judge_api_url = judge_cfg.get("api_url", "")
        self._judge_api_key = judge_cfg.get("api_key", "")
        self._judge_temperature = float(judge_cfg.get("temperature", 0.0))
        self._judge_max_tokens = int(judge_cfg.get("max_tokens", 512))
        self._judge_timeout = int(judge_cfg.get("timeout", 120))

        # Extra generation parameters for judge model (e.g., chat_template_kwargs)
        _KNOWN_JUDGE_KEYS = {
            "model", "api_url", "api_key", "temperature", "max_tokens",
            "timeout", "chat_template_kwargs",
        }
        self._judge_extra_params: Dict[str, Any] = {}
        for key, value in judge_cfg.items():
            if key not in _KNOWN_JUDGE_KEYS:
                self._judge_extra_params[key] = value
        if "chat_template_kwargs" in judge_cfg:
            self._judge_extra_params["chat_template_kwargs"] = judge_cfg[
                "chat_template_kwargs"
            ]

        self.logger.info(
            "MCPAtlasEvalTask initialized: pass_threshold=%.2f, "
            "use_structured_output=%s, "
            "judge_model=%s, judge_api_url=%s, judge_temperature=%.2f, "
            "judge_max_tokens=%d, judge_timeout=%d",
            self.pass_threshold,
            self.use_structured_output,
            self._judge_model or "(main model fallback)",
            self._judge_api_url or "(main model fallback)",
            self._judge_temperature,
            self._judge_max_tokens,
            self._judge_timeout,
        )

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
        self.logger.info(
            "Loaded %d predictions for evaluation", len(predictions)
        )

        # ---- 2. load dataset (for ground-truth claims) --------------------
        dataset_cfg = self.dataset_cfgs[0]
        self.logger.info(
            "Loading dataset for ground-truth claims: type=%s abbr=%s path=%s",
            dataset_cfg.get("type"),
            dataset_cfg.get("abbr"),
            dataset_cfg.get("path"),
        )
        dataset = build_dataset_from_cfg(dataset_cfg)
        samples = list(dataset.test)
        self.logger.info(
            "Dataset loaded: %d samples with ground-truth claims", len(samples)
        )

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
            self.logger.info(
                "--- Eval %d/%d: task_id=%s ---", idx + 1, total, task_id
            )
            result = self._evaluate_prediction(task_id, pred, sample_map)
            results.append(result)
            pbar.update(1)
            task_state_manager.update_task_state({"finish_count": idx + 1})
            self.logger.info(
                "Eval %s: coverage_score=%.3f, pass=%s, claims=%d",
                task_id,
                result["coverage_score"],
                result["pass"],
                result["total_claims"],
            )

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
        self.logger.info("Looking for predictions at: %s", pred_path)

        if not osp.isfile(pred_path):
            # Try fallback path (matches swebench pattern)
            pred_path_fallback = osp.join(
                osp.dirname(pred_path),
                osp.splitext(osp.basename(pred_path))[0],
                "preds.json",
            )
            self.logger.info(
                "Primary path not found, trying fallback: %s",
                pred_path_fallback,
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

        self.logger.info(
            "Predictions file size: %d bytes", osp.getsize(pred_path)
        )
        if isinstance(raw_preds, dict):
            self.logger.info(
                "Parsed predictions as dict with %d entries", len(raw_preds)
            )
            return raw_preds
        # If it's a list, index by task_id
        indexed = {
            (p.get("task_id") or str(i)): p
            for i, p in enumerate(raw_preds)
        }
        self.logger.info(
            "Parsed predictions as list, indexed %d entries", len(indexed)
        )
        return indexed

    # -- per-sample scoring ------------------------------------------------

    def _evaluate_prediction(
        self,
        task_id: str,
        pred: Dict[str, Any],
        sample_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score a single prediction with the LLM judge.

        Ported from mcp-atlas score_claims.py ``CoverageEvaluator.evaluate``:
        extract claims from ground truth, judge each claim against the
        prediction's final answer, compute coverage_score and pass/fail.
        """
        final_answer = str(pred.get("final_answer") or "")
        prompt = str(pred.get("prompt") or "")

        self.logger.info(
            "[Eval:%s] final_answer_len=%d, prompt_len=%d, "
            "tool_calls=%d, has_claims_in_pred=%s",
            task_id,
            len(final_answer),
            len(prompt),
            pred.get("tool_calls", 0),
            bool(pred.get("gtfa_claims")),
        )

        # Try to get claims from prediction metadata first, then from dataset
        claims = list(pred.get("gtfa_claims") or [])
        if not claims and task_id in sample_map:
            sample = sample_map[task_id]
            claims = extract_claims(
                _field(sample, "GTFA_CLAIMS", "gtfa_claims", "rubrics") or "[]"
            )
            self.logger.info(
                "[Eval:%s] Claims loaded from dataset: %d claims",
                task_id,
                len(claims),
            )
        elif claims:
            self.logger.info(
                "[Eval:%s] Claims found in prediction metadata: %d claims",
                task_id,
                len(claims),
            )
        else:
            self.logger.warning(
                "[Eval:%s] No claims found (neither in prediction nor dataset)",
                task_id,
            )

        # Skip LLM judge for empty/error responses (ported from mcp-atlas)
        if not final_answer or not final_answer.strip() or final_answer.startswith("ERROR:"):
            self.logger.info(
                "[Eval:%s] Skipping judge — empty or error response",
                task_id,
            )
            claim_results = [
                {
                    "claim": c,
                    "coverage_outcome": "not_fulfilled",
                    "score": 0.0,
                    "justification": "Empty or error response",
                    "confidence": 1.0,
                }
                for c in claims
            ]
            coverage, fulfilled, partial, not_covered = compute_coverage_score(
                claim_results
            )
            return {
                "task_id": task_id,
                "prompt": prompt,
                "final_answer": final_answer,
                "tool_calls": pred.get("tool_calls", 0),
                "claims": claim_results,
                "total_claims": len(claim_results),
                "fully_covered_claims": fulfilled,
                "partially_covered_claims": partial,
                "not_covered_claims": not_covered,
                "coverage_score": coverage,
                "pass": coverage >= self.pass_threshold,
                "pass_threshold": self.pass_threshold,
                "evaluation_confidence": 1.0,
            }

        # Truncate oversized responses (ported from mcp-atlas)
        if len(final_answer) > self.MAX_RESPONSE_CHARS:
            self.logger.warning(
                "[Eval:%s] Response truncated from %d to %d chars",
                task_id,
                len(final_answer),
                self.MAX_RESPONSE_CHARS,
            )
            final_answer = (
                final_answer[:self.MAX_RESPONSE_CHARS]
                + "\n\n[TRUNCATED — original response was too long]"
            )

        # Judge each claim (matching mcp-atlas per-claim evaluation)
        claim_results: List[Dict[str, Any]] = []
        total_confidence = 0.0

        for i, claim in enumerate(claims):
            self.logger.info(
                "[Eval:%s] Judging claim %d/%d: '%s...'",
                task_id,
                i + 1,
                len(claims),
                claim[:100],
            )
            cr = self._judge_claim(claim, final_answer)
            claim_results.append(cr)
            total_confidence += cr["confidence"]
            self.logger.info(
                "[Eval:%s] Claim %d result: outcome=%s score=%.1f "
                "confidence=%.2f",
                task_id,
                i + 1,
                cr["coverage_outcome"],
                cr["score"],
                cr["confidence"],
            )

        # Compute aggregated scores (ported from mcp-atlas)
        coverage, fulfilled, partial, not_covered = compute_coverage_score(
            claim_results
        )
        avg_confidence = total_confidence / len(claim_results) if claim_results else 0.5
        passed = coverage >= self.pass_threshold

        self.logger.info(
            "[Eval:%s] Scoring done: coverage_score=%.3f (threshold=%.2f), "
            "pass=%s, claims_total=%d, claims_fulfilled=%d, "
            "claims_partial=%d, claims_not_fulfilled=%d",
            task_id,
            coverage,
            self.pass_threshold,
            passed,
            len(claim_results),
            fulfilled,
            partial,
            not_covered,
        )

        return {
            "task_id": task_id,
            "prompt": prompt,
            "final_answer": final_answer,
            "tool_calls": pred.get("tool_calls", 0),
            "claims": claim_results,
            "total_claims": len(claim_results),
            "fully_covered_claims": fulfilled,
            "partially_covered_claims": partial,
            "not_covered_claims": not_covered,
            "coverage_score": coverage,
            "pass": passed,
            "pass_threshold": self.pass_threshold,
            "evaluation_confidence": round(avg_confidence, 4),
        }

    # -- LLM judge ---------------------------------------------------------

    def _judge_claim(self, claim: str, response: str) -> Dict[str, Any]:
        """Score a single claim with the LLM judge.

        Ported from mcp-atlas score_claims.py ``evaluate_single_claim``:
        construct judge prompt → call LLM judge → parse response →
        map outcome to numeric score.
        """
        prompt = _claim_judge_prompt(claim, response)
        self.logger.info(
            "[Judge] Prompt constructed: claim_len=%d, response_len=%d, "
            "total_prompt_len=%d",
            len(claim),
            len(response),
            len(prompt),
        )
        judge_response = self._call_judge(prompt)
        outcome, justification, confidence = _parse_claim_judge_response(
            judge_response
        )
        # Match mcp-atlas's score mapping
        score_map = {
            "fulfilled": 1.0,
            "partially_fulfilled": 0.5,
            "not_fulfilled": 0.0,
        }
        score = score_map.get(outcome, 0.0)

        self.logger.info(
            "[Judge] Result: outcome=%s -> score=%.1f, confidence=%.2f, "
            "justification_len=%d",
            outcome,
            score,
            confidence,
            len(justification),
        )

        return {
            "claim": claim,
            "coverage_outcome": outcome,
            "score": score,
            "justification": justification,
            "confidence": confidence,
            "raw_judge_response": judge_response,
        }

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model via OpenAI-compatible API.

        Ported from mcp-atlas score_claims.py ``AsyncLiteLLMClient``.
        Supports optional structured JSON output via ``response_format``.
        """
        url = self._judge_api_url or self.model_cfg.get(
            "api_url", self.model_cfg.get("url", ""),
        )
        api_key = self._judge_api_key or self.model_cfg.get(
            "api_key", self.model_cfg.get("key", ""),
        )
        model = (
            self._judge_model
            or self.model_cfg.get("model")
            or os.getenv("AIS_BENCH_MODEL")
            or os.getenv("MODEL_NAME")
            or ""
        )
        if not str(model).strip():
            raise ValueError(
                "MCP-Atlas judge model name is empty. Set judge_model.model "
                "or model in the config, or export AIS_BENCH_MODEL."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._judge_temperature,
            "max_tokens": self._judge_max_tokens,
        }

        # Use structured JSON output if enabled (ported from mcp-atlas)
        if self.use_structured_output:
            schema = get_claim_evaluation_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "claim_evaluation",
                    "schema": schema,
                },
            }
            self.logger.info(
                "[Judge] Using structured output (json_schema)"
            )

        # Merge extra generation parameters (e.g., chat_template_kwargs)
        if self._judge_extra_params:
            payload.update(self._judge_extra_params)

        self.logger.info(
            "[Judge] Calling judge API: url=%s, model=%s, prompt_len=%d, "
            "temperature=%.2f, max_tokens=%d",
            url, model, len(prompt),
            self._judge_temperature, self._judge_max_tokens,
        )

        resp = requests.post(
            f'{url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=self._judge_timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            self.logger.error(
                "[Judge] API request failed: status=%s, body=%s",
                resp.status_code,
                resp.text[:4000],
            )
            raise exc
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        result = str(choice.get("message", {}).get("content", "") or "")

        usage = data.get("usage", {})
        self.logger.info(
            "[Judge] API response: result_len=%d, finish_reason=%s, "
            "prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
            len(result),
            choice.get("finish_reason", "unknown"),
            usage.get("prompt_tokens", "N/A"),
            usage.get("completion_tokens", "N/A"),
            usage.get("total_tokens", "N/A"),
        )

        return result

    # -- results -----------------------------------------------------------

    def _dump_results(self, results: List[Dict[str, Any]]) -> None:
        """Aggregate scores and write results.

        Ported from mcp-atlas score_claims.py ``generate_statistics_and_plots``.

        Writes:
        - ``results/<model>/<dataset>.json`` — summary with aggregate scores
        - ``results/<model>/<dataset>/details.json`` — per-sample details
        - ``results/<model>/<dataset>/coverage_stats_<model>.json`` — coverage
          statistics (mean, pass rates at 0.50/0.75)
        """
        dataset_cfg = self.dataset_cfgs[0]
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
        model_abbr = model_abbr_from_cfg(self.model_cfg)

        out_dir = osp.join(self.work_dir, self.output_subdir, model_abbr)
        mkdir_or_exist(out_dir)
        out_detail_dir = osp.join(out_dir, dataset_abbr)
        mkdir_or_exist(out_detail_dir)

        # Compute coverage statistics (ported from mcp-atlas)
        model_name = self._judge_model or model_abbr
        stats = compute_coverage_stats(
            results,
            pass_threshold=self.pass_threshold,
            model_name=model_name,
            evaluator_model=self._judge_model or "(main model)",
        )

        # Aggregate claims (matching mcp-atlas)
        n_samples = len(results)
        fully_covered = sum(
            r.get("fully_covered_claims", 0) for r in results
        )
        partially_covered = sum(
            r.get("partially_covered_claims", 0) for r in results
        )
        not_covered = sum(
            r.get("not_covered_claims", 0) for r in results
        )

        summary = {
            "total_count": n_samples,
            "coverage_score": stats["mean_coverage"],
            "pass_rate": round(
                stats["pass_rate_0.75"] / 100, 4
            ) if stats["pass_rate_0.75"] is not None else 0.0,
            "pass_rate_0.50": stats["pass_rate_0.50"],
            "pass_rate_0.75": stats["pass_rate_0.75"],
            "pass_threshold": self.pass_threshold,
            "fully_covered_claims": fully_covered,
            "partially_covered_claims": partially_covered,
            "not_covered_claims": not_covered,
            "evaluator_model": stats["evaluator_model"],
        }

        self.logger.info(
            "Evaluation summary: samples=%d, avg_coverage=%.4f, "
            "pass_rate_0.75=%.2f%%, pass_rate_0.50=%.2f%%, "
            "threshold=%.2f, claims: %d fulfilled, "
            "%d partial, %d not_fulfilled",
            n_samples,
            stats["mean_coverage"],
            stats["pass_rate_0.75"],
            stats["pass_rate_0.50"],
            self.pass_threshold,
            fully_covered,
            partially_covered,
            not_covered,
        )

        # Write summary
        summary_path = osp.join(out_dir, f"{dataset_abbr}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(
            "Summary saved to %s (%d bytes)",
            summary_path,
            osp.getsize(summary_path),
        )

        # Write per-task details
        detail_path = osp.join(out_detail_dir, "details.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(
            "Details saved to %s (%d samples, %d bytes)",
            detail_path,
            len(results),
            osp.getsize(detail_path),
        )

        # Write coverage stats JSON (ported from mcp-atlas)
        stats_path = osp.join(
            out_detail_dir,
            f"coverage_stats_{model_abbr}_all.json",
        )
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        self.logger.info(
            "Coverage stats saved to %s", stats_path,
        )


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
