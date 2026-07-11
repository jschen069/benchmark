# flake8: noqa
"""Summarizer for MCP-Atlas benchmark results."""

import os.path as osp
from typing import Any, Dict, List

import mmengine
import tabulate
from mmengine import ConfigDict

from ais_bench.benchmark.summarizers.default import DefaultSummarizer
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.core.abbr import dataset_abbr_from_cfg, model_abbr_from_cfg

METRIC_WHITELIST = ["coverage_score", "pass_rate", "total_count"]
METRIC_BLACKLIST = ["excluded_tasks", "pass_threshold"]


class MCPAtlasSummarizer(DefaultSummarizer):
    """Summarizer for MCP-Atlas benchmark results.

    Reads per-model ``results/<model>/<dataset>.json`` files and prints
    coverage_score and pass_rate in a tabular format.
    """

    def _pick_up_results(self):
        raw_results: Dict[str, Dict[str, Any]] = {}
        parsed_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        dataset_metrics: Dict[str, List[str]] = {}
        dataset_eval_mode: Dict[str, str] = {}

        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg(model)
            parsed_results.setdefault(model_abbr, {})
            raw_results.setdefault(model_abbr, {})
            for dataset in self.dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                filepath = osp.join(
                    self.work_dir, "results", model_abbr,
                    f"{dataset_abbr}.json",
                )
                if not osp.exists(filepath):
                    continue
                result = mmengine.load(filepath)
                raw_results[model_abbr][dataset_abbr] = result

                _rst, _dm = {}, []
                for metric, score in result.items():
                    if metric in METRIC_BLACKLIST:
                        continue
                    if isinstance(score, (int, float)):
                        _rst[metric] = score
                        _dm.append(metric)

                if len(_rst) == 0:
                    continue

                _dm = sorted(
                    _dm,
                    key=lambda i: (
                        METRIC_WHITELIST.index(i)
                        if i in METRIC_WHITELIST
                        else len(METRIC_WHITELIST)
                    ),
                )
                dataset_metrics[dataset_abbr] = _dm
                parsed_results[model_abbr][dataset_abbr] = _rst
                dataset_eval_mode[dataset_abbr] = "gen"

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode

    def summarize(
        self,
        time_str=None,
        subjective_scores=None,
        dataset_score_container=None,
        required_dataset_abbrs=None,
    ):
        self._update_dataset_abbrs()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = (
            self._pick_up_results()
        )

        dataset_abbrs = []
        for dataset_abbr in self.dataset_abbrs:
            if dataset_abbr in dataset_metrics:
                for metric in dataset_metrics[dataset_abbr]:
                    dataset_abbrs.append((dataset_abbr, metric))
            else:
                dataset_abbrs.append((dataset_abbr, None))

        has_total_count = False
        for dataset_abbr in dataset_metrics:
            if "total_count" in dataset_metrics[dataset_abbr]:
                has_total_count = True
                break

        table = []
        header = ["dataset", "version", "metric", "mode"] + self.model_abbrs
        if has_total_count:
            header = [
                "dataset", "version", "metric", "mode", "total_count"
            ] + self.model_abbrs
        table.append(header)

        for dataset_abbr, metric in dataset_abbrs:
            for model_abbr in self.model_abbrs:
                if metric is None:
                    for k in (
                        parsed_results.get(model_abbr, {})
                        .get(dataset_abbr, {})
                        .keys()
                    ):
                        row = [
                            dataset_abbr, "a39421", k,
                            dataset_eval_mode.get(dataset_abbr, "gen"),
                        ]
                        if has_total_count:
                            row.insert(
                                4,
                                raw_results[model_abbr][dataset_abbr].get(
                                    "total_count", "-"
                                ),
                            )
                        row.append(
                            parsed_results[model_abbr][dataset_abbr][k]
                        )
                        table.append(row)
                else:
                    if (
                        dataset_abbr in parsed_results[model_abbr]
                        and metric in parsed_results[model_abbr][dataset_abbr]
                    ):
                        row = [
                            dataset_abbr, "a39421", metric,
                            dataset_eval_mode.get(dataset_abbr, "gen"),
                        ]
                        if has_total_count:
                            row.insert(
                                4,
                                raw_results[model_abbr][dataset_abbr].get(
                                    "total_count", "-"
                                ),
                            )
                        row.append(
                            parsed_results[model_abbr][dataset_abbr][metric]
                        )
                        table.append(row)

        print("")
        print(tabulate.tabulate(table[1:], headers=table[0], tablefmt="grid"))
        print("")

        summary_dir = osp.join(self.work_dir, "summary")
        mmengine.mkdir_or_exist(summary_dir)

        time_str = time_str or mmengine.utils.TimeStub.now().time_str
        summary_txt = osp.join(summary_dir, f"summary_{time_str}.txt")
        summary_csv = osp.join(summary_dir, f"summary_{time_str}.csv")

        print(f"write summary to {summary_txt}")
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(
                tabulate.tabulate(table[1:], headers=table[0], tablefmt="grid")
            )

        print(f"write csv to {summary_csv}")
        with open(summary_csv, "w", encoding="utf-8") as out:
            out.write(
                tabulate.tabulate(table[1:], headers=table[0], tablefmt="csv")
            )

        return parsed_results
