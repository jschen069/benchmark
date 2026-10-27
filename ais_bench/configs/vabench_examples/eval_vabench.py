"""Eval-only: prepare MP4 once, then run this config with ais_bench -m eval.

Edit the constants below. Use the same config for prepared 10/778 splits.
"""

from ais_bench.benchmark.datasets.vabench import VABenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import VABenchEvalTask
from ais_bench.benchmark.summarizers import VABenchSummarizer

PREPARED_DIR = ''
SOURCE_ROOT = ''
CACHE_DIR = ''
DEVICE = 'npu:0'
BACKEND_PYTHON = None  # e.g. '/opt/venvs/vabench/bin/python'
NUM_DEVICES = 1  # number reserved by LocalRunner; GPU Omni may require more

models = [dict(attr='local', type='VABenchEvalPlaceholder', abbr='generated_t2av')]
datasets = [dict(
    type=VABenchDataset,
    abbr='vabench_t2av',
    path=PREPARED_DIR,
    eval_cfg=dict(
        source_root=SOURCE_ROOT,
        cache_dir=CACHE_DIR,
        device=DEVICE,
        python=BACKEND_PYTHON,
        device_overrides={'first_dnsmos': 'cpu'},
        params={'device_map': 'auto'},
    ),
)]
eval = dict(
    partitioner=dict(type=NaivePartitioner,
                     keep_keys=['eval.runner.task.num_gpus']),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=VABenchEvalTask,
                          num_gpus=NUM_DEVICES)),
)
summarizer = dict(attr='accuracy', type=VABenchSummarizer)
