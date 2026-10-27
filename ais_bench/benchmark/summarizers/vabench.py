"""VABench native-unit mean and sample counts; no overall mixed score."""
import os.path as osp

from ais_bench.benchmark.summarizers.default import (
    DefaultSummarizer, model_abbr_from_cfg_used_in_summarizer)
from ais_bench.benchmark.utils.core.abbr import get_infer_output_path
from ais_bench.benchmark.vabench.data import read_json


class VABenchSummarizer(DefaultSummarizer):
    def _pick_up_results(self):
        raw, parsed, metrics, modes = {}, {}, {}, {}
        display = []
        for model in self.model_cfgs:
            name = model_abbr_from_cfg_used_in_summarizer(model)
            raw[name], parsed[name] = {}, {}
            for dataset in self.dataset_cfgs:
                file = get_infer_output_path(model, dataset, osp.join(self.work_dir, 'results'))
                if not osp.isfile(file):
                    continue
                report = read_json(file)
                for dimension, data in report.get('metrics', {}).items():
                    row = f'{dataset["abbr"]}/{dimension}'
                    # Keep None out of numeric metrics; display a status, not zero.
                    values = {k: v for k, v in data.items()
                              if k in ('mean', 'expected', 'scored', 'missing', 'failed') and v is not None}
                    values.setdefault('mean', 'N/A')
                    values['direction'] = report['directions'][dimension]
                    values['status'] = 'complete' if data['scored'] and not data['failed'] and (
                        not data['missing'] or dimension == 'second_lsa') else 'incomplete'
                    raw[name][row] = dict(values, scenes=data['scenes'],
                                          direction=report['directions'][dimension])
                    parsed[name][row] = values
                    metrics[row] = ['mean', 'expected', 'scored', 'missing', 'failed', 'status']
                    modes[row] = 'gen'
                    if row not in display:
                        display.append(row)
        self.dataset_abbrs = display
        return raw, parsed, metrics, modes

    def _format_table(self, parsed_results, dataset_metrics, dataset_eval_mode,
                      required_dataset_abbrs=None, skip_all_slash=False):
        table = [['model', 'dataset/dimension', 'direction', 'mean',
                  'expected', 'scored', 'missing', 'failed', 'status']]
        for model, rows in parsed_results.items():
            for name, values in rows.items():
                value = values['mean']
                value = f'{value:.5f}' if isinstance(value, (int, float)) else value
                table.append([model, name, values['direction'], value,
                              values['expected'], values['scored'], values['missing'],
                              values['failed'], values['status']])
        return table

    def _calculate_group_metrics(self, raw_results, parsed_results, dataset_metrics, dataset_eval_mode):
        # A VBench weighted total is not defined for these fifteen metrics.
        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode
