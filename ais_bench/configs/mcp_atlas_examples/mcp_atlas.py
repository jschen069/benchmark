"""MCP-Atlas benchmark config.

MCP-Atlas evaluates tool-use competency against real Model Context Protocol
(MCP) servers.  This config runs both inference (agent loop with
subprocess-isolated MCP tool calls) and evaluation (LLM judge per-claim
scoring) in one pipeline, following the swebench config pattern.

Ported from the upstream mcp-atlas project.  Key parameters match
the upstream defaults where applicable.

Pre-requisites
--------------
1. Start the MCP-Atlas agent-environment Docker service::

    docker run -d -p 1984:1984 \\
        --name mcp-atlas-agent-env \\
        <agent-environment-image>

2. Ensure the service is reachable at ``http://localhost:1984`` (or update
   ``mcp_server_url`` below).

"""

import os

from ais_bench.benchmark.datasets import MCPAtlasDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.summarizers import MCPAtlasSummarizer
from ais_bench.benchmark.tasks import MCPAtlasInferTask, MCPAtlasEvalTask

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

models = [
    dict(
        abbr="mcp_atlas_model",
        api_key=os.getenv("AIS_BENCH_API_KEY", "EMPTY"),
        url=os.getenv("AIS_BENCH_API_URL", "http://127.0.0.1:8005/v1"),
        model=os.getenv("AIS_BENCH_MODEL", os.getenv("MODEL_NAME", "")),
        # ---- Model inference config (for agent loop API calls) -----------
        # Matches mcp-atlas defaults: temperature=0, max_tokens=unlimited
        infer_cfg=dict(
            temperature=0.0,
            max_tokens=32768,
            timeout=300,
            tool_choice="auto",
        ),
        # ---- LLM judge config (optional; defaults to main model above) ---
        # Matches mcp-atlas score_claims.py defaults
        judge_model=dict(
            model=os.getenv("AIS_BENCH_JUDGE_MODEL", ""),
            api_key=os.getenv("AIS_BENCH_JUDGE_API_KEY", ""),
            api_url=os.getenv("AIS_BENCH_JUDGE_API_URL", ""),
            temperature=0.0,
            max_tokens=32768,
            timeout=120,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

datasets = [
    dict(
        type=MCPAtlasDataset,
        abbr="mcp_atlas",
        path="ais_bench/datasets/mcp_atlas/MCP-Atlas.parquet",
        split="train",
        args=dict(
            # MCP agent-environment service URL
            mcp_server_url="http://localhost:1984",

            # Skip tasks whose servers are not currently enabled
            filter_enabled_servers=True,

            # Maximum agent loop steps per sample (mcp-atlas default: 256)
            max_steps=100,

            # Maximum MCP tool calls allowed per sample (mcp-atlas default: 100)
            max_tool_calls=100,

            # Timeout (seconds) for individual MCP tool calls
            request_timeout=60.0,

            # Timeout (seconds) for preflight /list-tools requests
            list_tools_timeout=180.0,

            # Prepend the optional MCP-Atlas system prompt to each sample
            use_system_prompt=False,

            # Coverage score threshold used to compute pass rate
            pass_threshold=0.75,

            # ---- New parameters ported from mcp-atlas --------------------
            # Context window management: "compact" truncates old tool
            # results when context grows large (mcp-atlas: off by default)
            context_window_management="",

            # Truncate each tool result to at most N characters before
            # feeding it back to the model (None = uncapped, mcp-atlas default)
            tool_output_cap=None,

            # Use structured JSON output (response_format: json_schema)
            # for the LLM judge (requires model support)
            use_structured_output=False,
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
# Inference
# ---------------------------------------------------------------------------

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=MCPAtlasInferTask),
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
