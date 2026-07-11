"""MCP-Atlas evaluation config.

Loads predictions from the infer step, scores final answers with an LLM
judge on a per-claim basis, and computes aggregated metrics.  Run this
AFTER ``mcp_atlas_infer.py``.

Pre-requisites
--------------
1. Run ``mcp_atlas_infer.py`` first to generate predictions.
2. Ensure the LLM judge model is accessible.

Usage
-----
.. code-block:: bash

    evalscope eval \\
        --model YOUR_MODEL \\
        --api-url http://127.0.0.1:8000/v1 \\
        --api-key EMPTY \\
        --datasets mcp_atlas \\
        --limit 10
"""

from ais_bench.benchmark.datasets import MCPAtlasDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.summarizers import MCPAtlasSummarizer
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.tasks import MCPAtlasEvalTask

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

models = [
    dict(
        abbr="mcp_atlas_model",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        model="",                        # Fill in your model name
        temperature=0.0,
        max_tokens=4096,
        timeout=300,
        # ---- LLM judge (optional; defaults to the same model) ------------
        # judge_model="gpt-4o",
        # judge_api_url="https://api.openai.com/v1",
        # judge_api_key="sk-...",
    ),
]

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

datasets = [
    dict(
        type=MCPAtlasDataset,
        abbr="mcp_atlas",
        path="ais_bench/datasets/mcp-atlas/MCP-Atlas.parquet",
        split="train",
        args=dict(
            # Coverage score threshold used to compute pass rate
            pass_threshold=0.75,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

summarizer = dict(
    attr="accuracy",
    type=MCPAtlasSummarizer,
)

# ---------------------------------------------------------------------------
# Inference (no-op: MCPAtlasEvalTask handles evaluation only)
# ---------------------------------------------------------------------------

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=EmptyTask),
    ),
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=MCPAtlasEvalTask),
    ),
)
