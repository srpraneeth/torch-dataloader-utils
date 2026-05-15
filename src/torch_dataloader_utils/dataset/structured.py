from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

import pyarrow.compute as pc
from torch.utils.data import DataLoader

from torch_dataloader_utils.dataset.base import BaseDataset
from torch_dataloader_utils.dataset.output import convert_batch
from torch_dataloader_utils.filesystem.discovery import discover_files
from torch_dataloader_utils.format.reader import SUPPORTED_FORMATS, read_split
from torch_dataloader_utils.observability import WorkerMetrics, fmt_bytes
from torch_dataloader_utils.splits.core import DataFileInfo, Shard, SplitStrategy
from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy
from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy, parse_bytes

logger = logging.getLogger(__name__)

_SUPPORTED_OUTPUT_FORMATS = {"torch", "numpy", "arrow", "dict"}


def _auto_select_strategy(
    files: list[DataFileInfo],
    shuffle: bool,
    shuffle_seed: int,
    split_bytes: int | str | None = None,
    split_rows: int | None = None,
) -> SplitStrategy:
    if files:
        kwargs: dict = {"shuffle": shuffle, "seed": shuffle_seed}
        if split_bytes is not None:
            kwargs["target_bytes"] = parse_bytes(split_bytes)
        if split_rows is not None:
            kwargs["target_rows"] = split_rows
        logger.info("Auto-selected split strategy: TargetSizeSplitStrategy")
        return TargetSizeSplitStrategy(**kwargs)
    logger.info("Auto-selected split strategy: RoundRobinSplitStrategy (no files)")
    return RoundRobinSplitStrategy(shuffle=shuffle, seed=shuffle_seed)


class StructuredDataset(BaseDataset):
    """PyTorch IterableDataset for structured file formats on any fsspec filesystem.

    Splits are generated once in the main process before workers start.
    Each worker receives a pickled copy of the dataset with splits pre-computed —
    __iter__ is read-only and safe to run in forked worker processes.

    For shuffle support across epochs, call set_epoch(n) before each DataLoader
    iteration in your training loop:

        for epoch in range(num_epochs):
            dataset.set_epoch(epoch)
            for batch in loader:
                ...
    """

    def __init__(
        self,
        files: list[DataFileInfo],
        format: str,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_bytes: int | str | None = None,
        split_rows: int | None = None,
        split_strategy: SplitStrategy | None = None,
        num_workers: int = 1,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
        partitioning: str | None = None,
        num_ranks: int = 1,
        rank: int = 0,
        show_progress: bool = False,
        progress_interval_sec: float = 120.0,
    ) -> None:
        if format not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise ValueError(f"Unsupported format {format!r}. Supported: {supported}")

        if output_format not in _SUPPORTED_OUTPUT_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_OUTPUT_FORMATS))
            raise ValueError(f"Unsupported output_format {output_format!r}. Supported: {supported}")

        if filters is not None and not isinstance(filters, pc.Expression):
            raise TypeError(
                f"filters must be a pyarrow pc.Expression, got {type(filters).__name__}."
            )

        if output_format in ("arrow", "dict") and collate_fn is None:
            raise ValueError(
                f"output_format={output_format!r} requires a collate_fn. "
                "PyTorch's default collate cannot handle this type. "
                "Pass a custom collate_fn or use output_format='torch'."
            )

        if show_progress:
            try:
                import tqdm as _tqdm  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm. "
                    "Install it with: pip install tqdm"
                ) from exc

        self._files = files
        self._format = format
        self._batch_size = batch_size
        self._columns = columns
        self._filters = filters
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed
        self._strategy = (
            split_strategy
            if split_strategy is not None
            else _auto_select_strategy(files, shuffle, shuffle_seed, split_bytes, split_rows)
        )
        self._num_workers = num_workers
        self._output_format = output_format
        self._storage_options = storage_options
        self._collate_fn = collate_fn
        self._partitioning = partitioning
        self._num_ranks = num_ranks
        self._rank = rank

        self._init_splits_and_observability(
            epoch=0,
            show_progress=show_progress,
            progress_interval_sec=progress_interval_sec,
        )

    def _iter_shard(
        self,
        shard: Shard,
        worker_id: int,
        metrics: WorkerMetrics,
        pbar: Any,
    ) -> Iterator[Any]:
        for batch in read_split(
            shard,
            format=self._format,
            batch_size=self._batch_size,
            columns=self._columns,
            filters=self._filters,
            storage_options=self._storage_options,
            partitioning=self._partitioning,
            metrics=metrics,
            pbar=pbar,
        ):
            yield convert_batch(batch, self._output_format)

    @classmethod
    def create_dataloader(
        cls,
        path: str,
        format: str,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_bytes: int | str | None = None,
        split_rows: int | None = None,
        split_strategy: SplitStrategy | None = None,
        num_workers: int | None = None,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
        partitioning: str | None = None,
        num_ranks: int = 1,
        rank: int = 0,
        show_progress: bool = False,
        progress_interval_sec: float = 120.0,
    ) -> tuple[DataLoader, StructuredDataset]:
        """Create a DataLoader for structured files at the given path.

        Returns (DataLoader, dataset) — keep a reference to dataset to call
        set_epoch(n) at the start of each epoch when shuffle=True.
        """
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)
            logger.info("Auto-detected num_workers=%d", num_workers)

        files = discover_files(path, storage_options=storage_options)

        total_bytes = sum(f.file_size for f in files if f.file_size is not None)
        col_str = ", ".join(columns) if columns else "all"
        logger.info(
            "DataLoader ready\n"
            "  path         : %s\n"
            "  format       : %s\n"
            "  files        : %d  (%s total)\n"
            "  workers      : %d   (rank %d / %d)\n"
            "  batch_size   : %d\n"
            "  strategy     : %s\n"
            "  shuffle      : %s  seed=%d\n"
            "  columns      : %s\n"
            "  filters      : %s\n"
            "  output_fmt   : %s",
            path,
            format,
            len(files),
            fmt_bytes(total_bytes),
            num_workers,
            rank,
            num_ranks,
            batch_size,
            split_strategy.__class__.__name__ if split_strategy else "TargetSizeSplitStrategy",
            shuffle,
            shuffle_seed,
            col_str,
            "yes" if filters is not None else "none",
            output_format,
        )

        if files:
            try:
                import fsspec
                import pyarrow.dataset as pads
                import pyarrow.fs as pafs
                import pyarrow.parquet as pq

                fs, resolved = fsspec.url_to_fs(files[0].path, **(storage_options or {}))
                if fs.protocol in ("file", "local") or (
                    isinstance(fs.protocol, tuple) and "file" in fs.protocol
                ):
                    arrow_fs, rpath = None, files[0].path
                else:
                    arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
                    rpath = resolved

                if format == "parquet":
                    schema = pq.read_schema(rpath, filesystem=arrow_fs)
                else:
                    schema = pads.dataset(rpath, format=format, filesystem=arrow_fs).schema
                field_summary = ", ".join(f"{f.name}: {f.type}" for f in schema)
                logger.info("Inferred schema from %s: [%s]", files[0].path, field_summary)
            except Exception:
                pass  # schema logging is best-effort

        dataset = cls(
            files=files,
            format=format,
            batch_size=batch_size,
            columns=columns,
            filters=filters,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            split_bytes=split_bytes,
            split_rows=split_rows,
            split_strategy=split_strategy,
            num_workers=num_workers,
            output_format=output_format,
            storage_options=storage_options,
            collate_fn=collate_fn,
            partitioning=partitioning,
            num_ranks=num_ranks,
            rank=rank,
            show_progress=show_progress,
            progress_interval_sec=progress_interval_sec,
        )

        effective_collate = collate_fn
        if effective_collate is None and output_format != "torch":
            effective_collate = lambda x: x  # noqa: E731

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=effective_collate,
        )

        return loader, dataset
