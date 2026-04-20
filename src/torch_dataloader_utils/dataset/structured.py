import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

import pyarrow.compute as pc
from torch.utils.data import DataLoader, IterableDataset

from torch_dataloader_utils.dataset.output import convert_batch
from torch_dataloader_utils.filesystem.discovery import discover_files
from torch_dataloader_utils.format.reader import SUPPORTED_FORMATS, read_split
from torch_dataloader_utils.splits.core import DataFileInfo, Shard, SplitStrategy
from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy
from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

logger = logging.getLogger(__name__)

_SUPPORTED_OUTPUT_FORMATS = {"torch", "numpy", "arrow", "dict"}


def _auto_select_strategy(
    files: list[DataFileInfo], shuffle: bool, shuffle_seed: int
) -> SplitStrategy:
    if files:
        logger.info("Auto-selected split strategy: TargetSizeSplitStrategy")
        return TargetSizeSplitStrategy(shuffle=shuffle, seed=shuffle_seed)
    logger.info("Auto-selected split strategy: RoundRobinSplitStrategy (no files)")
    return RoundRobinSplitStrategy(shuffle=shuffle, seed=shuffle_seed)


class StructuredDataset(IterableDataset):
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
        split_strategy: SplitStrategy | None = None,
        num_workers: int = 1,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
    ) -> None:
        if format not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise ValueError(f"Unsupported format {format!r}. Supported: {supported}")

        if output_format not in _SUPPORTED_OUTPUT_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_OUTPUT_FORMATS))
            raise ValueError(
                f"Unsupported output_format {output_format!r}. Supported: {supported}"
            )

        if output_format in ("arrow", "dict") and collate_fn is None:
            raise ValueError(
                f"output_format={output_format!r} requires a collate_fn. "
                "PyTorch's default collate cannot handle this type. "
                "Pass a custom collate_fn or use output_format='torch'."
            )

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
            else _auto_select_strategy(files, shuffle, shuffle_seed)
        )
        self._num_workers = num_workers
        self._output_format = output_format
        self._storage_options = storage_options
        self._collate_fn = collate_fn

        # Splits are generated in the main process and stored as immutable data.
        # Workers receive a pickled copy — __iter__ only reads, never writes.
        self._epoch: int = 0
        self._splits: list[Shard] = self._generate_splits()

    def _generate_splits(self) -> list[Shard]:
        n = max(self._num_workers, 1)
        return self._strategy.generate(self._files, num_workers=n, epoch=self._epoch)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffle reproducibility.

        Call this in the main process before each DataLoader iteration:

            for epoch in range(num_epochs):
                dataset.set_epoch(epoch)
                for batch in loader:
                    ...

        Has no effect when shuffle=False.
        """
        self._epoch = epoch
        self._splits = self._generate_splits()
        logger.info(
            "Regenerated splits for epoch %d  strategy=%s  num_workers=%d",
            epoch, type(self._strategy).__name__, self._num_workers,
        )

    def __iter__(self) -> Iterator[Any]:
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        worker_id = info.id if info is not None else 0

        if worker_id >= len(self._splits):
            # More workers than splits — this worker has nothing to do
            logger.debug(
                "Worker %d: no split assigned (only %d split(s) for %d worker(s)) — yielding nothing",
                worker_id, len(self._splits), self._num_workers,
            )
            return

        shard = self._splits[worker_id]
        file_paths = [s.file.path for s in shard.splits]
        logger.info(
            "Worker %d: assigned shard %d with %d split(s)",
            worker_id, shard.id, len(shard.splits),
        )
        for path in file_paths:
            logger.debug("Worker %d: shard %d file → %s", worker_id, shard.id, path)

        for batch in read_split(
            shard,
            format=self._format,
            batch_size=self._batch_size,
            columns=self._columns,
            filters=self._filters,
            storage_options=self._storage_options,
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
        split_strategy: SplitStrategy | None = None,
        num_workers: int | None = None,
        output_format: str = "torch",
        storage_options: dict | None = None,
        collate_fn: Callable | None = None,
    ) -> tuple[DataLoader, "StructuredDataset"]:
        """Create a DataLoader for structured files at the given path.

        Returns (DataLoader, dataset) — keep a reference to dataset to call
        set_epoch(n) at the start of each epoch when shuffle=True.
        """
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)
            logger.info("Auto-detected num_workers=%d", num_workers)

        logger.info(
            "create_dataloader: path=%s  format=%s  num_workers=%d  batch_size=%d  "
            "output_format=%s  shuffle=%s  columns=%s  filters=%s",
            path, format, num_workers, batch_size,
            output_format, shuffle,
            columns if columns else "all",
            "yes" if filters is not None else "none",
        )

        files = discover_files(path, storage_options=storage_options)
        logger.info("Discovered %d file(s) at %s", len(files), path)

        dataset = cls(
            files=files,
            format=format,
            batch_size=batch_size,
            columns=columns,
            filters=filters,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            split_strategy=split_strategy,
            num_workers=num_workers,
            output_format=output_format,
            storage_options=storage_options,
            collate_fn=collate_fn,
        )

        # For non-torch output formats, PyTorch's default collate would convert
        # numpy arrays and dicts back to tensors. Use a passthrough collate unless
        # the user has provided their own.
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
