"""MCP-Atlas dataset loader.

MCP-Atlas is a Scale AI benchmark for evaluating tool-use competency with
real Model Context Protocol (MCP) servers.  This module provides a dataset
loader that reads the MCP-Atlas parquet file from a local path.

"""

import os.path as osp
from typing import Any, Dict, Optional

from datasets import Dataset

from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging.logger import AISLogger

from .base import BaseDataset

logger = AISLogger()


@LOAD_DATASET.register_module()
class MCPAtlasDataset(BaseDataset):
    """Dataset loader for MCP-Atlas.

    Loads the benchmark from a local parquet file.  The parquet must
    contain columns: ``TASK``, ``ENABLED_TOOLS``, ``PROMPT``,
    ``GTFA_CLAIMS``, ``TRAJECTORY``.

    Parameters
    ----------
    path :
        **Required.**  Path to the local ``MCP-Atlas.parquet`` file or
        a directory containing it.
    split :
        Which split to access.  Defaults to ``"train"``.
    """

    def __init__(
        self,
        reader_cfg: Optional[Dict] = None,
        k: Any = 1,
        n: Any = 1,
        **kwargs,
    ) -> None:
        # Provide minimal reader_cfg so DatasetReader can initialise
        # without requiring input_columns / output_column (MCP-Atlas
        # does not use the ICL reader infrastructure).
        if reader_cfg is None:
            reader_cfg = dict(input_columns=[], output_column=None)
        super().__init__(reader_cfg=reader_cfg, k=k, n=n, **kwargs)

    @staticmethod
    def load(
        path: str,
        split: str = "train",
        **kwargs,
    ) -> Dataset:
        """Load the MCP-Atlas dataset from a local parquet file.

        Returns a :class:`~datasets.Dataset` with columns:
        ``TASK``, ``ENABLED_TOOLS``, ``PROMPT``, ``GTFA_CLAIMS``,
        ``TRAJECTORY``.
        """
        resolved = get_data_path(path, local_mode=True)
        logger.info("MCP-Atlas dataset path resolved: %s -> %s", path, resolved)

        # Resolve the parquet file path
        if osp.isdir(resolved):
            parquet_path = osp.join(resolved, "MCP-Atlas.parquet")
            if not osp.isfile(parquet_path):
                raise FileNotFoundError(
                    f"MCP-Atlas parquet not found in directory: {resolved}. "
                    "Expected MCP-Atlas.parquet inside the directory."
                )
        elif osp.isfile(resolved):
            parquet_path = resolved
        else:
            raise FileNotFoundError(
                f"MCP-Atlas dataset not found at: {resolved}. "
                "Provide a valid path to the MCP-Atlas.parquet file."
            )

        logger.info("Loading MCP-Atlas from local parquet: %s", parquet_path)
        dataset = Dataset.from_parquet(parquet_path)
        logger.info(
            "MCP-Atlas dataset loaded: %d samples, columns=%s",
            len(dataset),
            list(dataset.features.keys()),
        )
        return dataset
