"""MCP-Atlas dataset loader.

MCP-Atlas is a Scale AI benchmark for evaluating tool-use competency with
real Model Context Protocol (MCP) servers.  This module provides a dataset
loader that reads the ``ScaleAI/MCP-Atlas`` dataset from HuggingFace Hub or
a local parquet file.

"""

import os.path as osp
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset

from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging.logger import AISLogger

from .base import BaseDataset

DATASET_ID = "ScaleAI/MCP-Atlas"
"""Default HuggingFace dataset identifier."""

logger = AISLogger()


@LOAD_DATASET.register_module()
class MCPAtlasDataset(BaseDataset):
    """Dataset loader for MCP-Atlas.

    Loads the benchmark from HuggingFace Hub (``ScaleAI/MCP-Atlas``) by
    default.  Set ``path`` to a local directory containing
    ``MCP-Atlas.parquet`` to load from disk instead.

    Parameters
    ----------
    path :
        Local directory or HuggingFace dataset id.  When omitted the
        default HuggingFace dataset is used.
    split :
        Which split to load.  Defaults to ``"train"`` (the only split
        provided by MCP-Atlas).
    """

    @staticmethod
    def load(
        path: Optional[str] = None,
        split: str = "train",
        **kwargs,
    ) -> Dataset:
        """Load the MCP-Atlas dataset.

        Returns a :class:`~datasets.Dataset` with columns:
        ``TASK``, ``ENABLED_TOOLS``, ``PROMPT``, ``GTFA_CLAIMS``,
        ``TRAJECTORY``.
        """
        if path is not None:
            resolved = get_data_path(path, local_mode=True)
            # If the resolved path is a directory containing a parquet file
            # use it directly; otherwise treat it as a HuggingFace dataset id.
            if osp.isdir(resolved):
                parquet_path = osp.join(resolved, "MCP-Atlas.parquet")
                if osp.isfile(parquet_path):
                    logger.info(
                        "Loading MCP-Atlas from local parquet: %s",
                        parquet_path,
                    )
                    return Dataset.from_parquet(parquet_path)
                # Fallback: try loading as a HuggingFace dataset directory
                logger.info(
                    "Loading MCP-Atlas from local directory: %s",
                    resolved,
                )
                try:
                    return load_dataset(
                        "parquet",
                        data_files={"train": osp.join(resolved, "*.parquet")},
                        split=split,
                        **kwargs,
                    )
                except Exception:
                    pass
            # Treat as HuggingFace dataset id
            logger.info(
                "Loading MCP-Atlas from HuggingFace: %s", resolved
            )
            return load_dataset(resolved, split=split, **kwargs)

        logger.info(
            "Loading MCP-Atlas from HuggingFace: %s", DATASET_ID
        )
        return load_dataset(DATASET_ID, split=split, **kwargs)
