"""MCP-Atlas dataset loader.

MCP-Atlas is a Scale AI benchmark for evaluating tool-use competency with
real Model Context Protocol (MCP) servers.  This module provides a dataset
loader that reads the MCP-Atlas parquet file from a local path.

"""

import os.path as osp
from typing import Any, Dict, List, Optional

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
        # Normalize away `..` segments from the cache-dir-relative join.
        parquet_path = osp.normpath(resolved)
        logger.info(
            "MCP-Atlas dataset path resolved: %s -> %s", path, parquet_path
        )

        # Resolve the concrete parquet file path
        if osp.isdir(parquet_path):
            parquet_path = osp.join(parquet_path, "MCP-Atlas.parquet")
        if not osp.isfile(parquet_path):
            raise FileNotFoundError(
                f"MCP-Atlas dataset not found at: {parquet_path}. "
                "Provide a valid local path to MCP-Atlas.parquet."
            )

        logger.info("Loading MCP-Atlas from local parquet: %s", parquet_path)

        # Try multiple backends to work around pyarrow dictionary-encoding
        # corruption ("Index not in dictionary bounds") that some parquet
        # files trigger with certain pyarrow / datasets versions.
        dataset: Optional[Dataset] = None
        errors: List[str] = []

        # 1) datasets native loader (pyarrow new Dataset API)
        try:
            dataset = Dataset.from_parquet(parquet_path)
            logger.info("Loaded via Dataset.from_parquet.")
        except Exception as exc:
            errors.append(f"Dataset.from_parquet: {exc}")

        # 2) pyarrow direct table reader. Convert through Python lists to
        #    avoid HuggingFace datasets cache/scanner issues and preserve
        #    plain string columns.
        if dataset is None:
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(parquet_path)
                data: Dict[str, Any] = {
                    col: table.column(col).to_pylist()
                    for col in table.column_names
                }
                dataset = Dataset.from_dict(data)
                logger.info("Loaded via pyarrow read_table fallback.")
            except Exception as exc:
                errors.append(f"pyarrow read_table: {exc}")

        # 3) fastparquet via pandas — a pure-Python parquet
        #    implementation that does NOT use pyarrow at all, so it
        #    completely avoids the dictionary-encoding bug.
        if dataset is None:
            try:
                import pandas as pd

                df = pd.read_parquet(parquet_path, engine="fastparquet")
                dataset = Dataset.from_pandas(df)
                logger.info("Loaded via pandas + fastparquet.")
            except Exception as exc:
                errors.append(f"pandas+fastparquet: {exc}")

        # 4) Row-group rescue — read each row group individually and
        #    skip any that trigger the dictionary-encoding error.
        if dataset is None:
            try:
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(parquet_path)
                num_rg = pf.metadata.num_row_groups
                tables = []
                skipped = 0
                for i in range(num_rg):
                    try:
                        tables.append(pf.read_row_group(i))
                    except Exception as exc:
                        skipped += 1
                        logger.warning(
                            "Skipping row group %d/%d: %s", i + 1, num_rg, exc
                        )
                if not tables:
                    errors.append(
                        f"Row-group rescue: all {num_rg} row groups failed"
                    )
                else:
                    import pyarrow as pa

                    table = pa.concat_tables(tables)
                    dataset = Dataset.from_arrow(table)
                    logger.info(
                        "Loaded via row-group rescue: %d/%d groups "
                        "(%d skipped).",
                        len(tables),
                        num_rg,
                        skipped,
                    )
            except Exception as exc:
                errors.append(f"Row-group rescue: {exc}")

        if dataset is None:
            raise RuntimeError(
                "Failed to load MCP-Atlas parquet with all backends:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            "MCP-Atlas dataset loaded: %d samples, columns=%s",
            len(dataset),
            list(dataset.features.keys()),
        )
        return dataset
