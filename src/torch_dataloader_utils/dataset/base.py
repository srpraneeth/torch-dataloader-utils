from __future__ import annotations

import inspect
import logging
import multiprocessing
import queue
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from torch.utils.data import IterableDataset

from torch_dataloader_utils.observability import (
    WorkerMetrics,
    drain_metrics,
    fmt_bytes,
    log_epoch_summary,
    log_split_assignment,
)
from torch_dataloader_utils.splits.core import Shard

logger = logging.getLogger(__name__)


class BaseDataset(IterableDataset, ABC):
    """Shared infrastructure for all torch-dataloader-utils datasets.

    Provides observability (metrics, progress bars, split logging), epoch reshuffling,
    and the DataLoader __iter__ lifecycle. Subclasses implement _iter_shard() only.

    Required: before calling _init_splits_and_observability(), subclasses must set:
        self._files, self._strategy, self._num_workers, self._num_ranks, self._rank,
        self._output_format
    """

    def _init_splits_and_observability(
        self,
        epoch: int = 0,
        show_progress: bool = False,
        progress_interval_sec: float = 120.0,
    ) -> None:
        """Finish initialisation: set up metrics queues and generate the first splits."""
        self._epoch = epoch
        self._show_progress = show_progress
        self._progress_interval_sec = progress_interval_sec
        # Two-path metrics collection:
        # - num_workers=0: __iter__ runs in the main process — direct list append.
        # - num_workers>0: workers are separate processes — use mp.Queue.
        self._metrics_local: list[WorkerMetrics] = []
        self._metrics_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._splits: list[Shard] = self._generate_splits()

    def _generate_splits(self) -> list[Shard]:
        n = max(self._num_workers, 1)
        sig = inspect.signature(self._strategy.generate)
        if "num_ranks" in sig.parameters:
            shards = self._strategy.generate(
                self._files,
                num_workers=n,
                epoch=self._epoch,
                num_ranks=self._num_ranks,
                rank=self._rank,
            )
        else:
            shards = self._strategy.generate(self._files, num_workers=n, epoch=self._epoch)
        log_split_assignment(shards, epoch=self._epoch, rank=self._rank, num_ranks=self._num_ranks)
        return shards

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffle reproducibility. Call before each epoch when shuffle=True."""
        self.reset_metrics()
        self._epoch = epoch
        self._splits = self._generate_splits()
        logger.info(
            "Regenerated splits for epoch %d  strategy=%s  num_workers=%d",
            epoch,
            type(self._strategy).__name__,
            self._num_workers,
        )

    def get_metrics(self) -> list[WorkerMetrics]:
        """Drain the metrics queue and return per-worker stats from the last epoch.

        Also logs an epoch summary at INFO. Returns [] if called before any epoch
        or after the queue was already drained.
        """
        results = drain_metrics(self._metrics_local, self._metrics_queue)
        if results:
            log_epoch_summary(self._epoch, results)
        return results

    def reset_metrics(self) -> None:
        """Discard accumulated metrics from the previous epoch."""
        self._metrics_local.clear()
        while True:
            try:
                self._metrics_queue.get_nowait()
            except queue.Empty:
                break

    def _make_pbar(self, worker_id: int) -> Any:
        if not self._show_progress:
            return None
        from tqdm import tqdm

        return tqdm(
            total=None,
            position=worker_id,
            leave=False,
            mininterval=self._progress_interval_sec,
            maxinterval=self._progress_interval_sec,
            unit="rows",
        )

    def __iter__(self) -> Iterator[Any]:
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        is_main_process = info is None

        if worker_id >= len(self._splits):
            logger.debug(
                "Worker %d: no split assigned (only %d split(s) for %d worker(s))"
                " — yielding nothing",
                worker_id,
                len(self._splits),
                self._num_workers,
            )
            return

        shard = self._splits[worker_id]
        logger.info(
            "Worker %d: assigned shard %d with %d split(s)",
            worker_id,
            shard.id,
            len(shard.splits),
        )

        metrics = WorkerMetrics(worker_id=worker_id)
        pbar = self._make_pbar(worker_id)

        t0 = time.perf_counter()
        try:
            yield from self._iter_shard(shard, worker_id, metrics, pbar)
        finally:
            metrics.elapsed_sec = time.perf_counter() - t0
            if pbar is not None:
                pbar.close()

            logger.info(
                "Worker %d shard complete: files=%d  rows=%s  batches=%d"
                "  bytes_est=%s  elapsed=%.3fs",
                metrics.worker_id,
                metrics.files_read,
                f"{metrics.rows_read:,}",
                metrics.batches_read,
                fmt_bytes(metrics.bytes_read),
                metrics.elapsed_sec,
                extra={
                    "event": "shard_done",
                    "worker_id": metrics.worker_id,
                    "files_read": metrics.files_read,
                    "rows_read": metrics.rows_read,
                    "batches_read": metrics.batches_read,
                    "bytes_read": metrics.bytes_read,
                    "elapsed_sec": metrics.elapsed_sec,
                },
            )

            if is_main_process:
                self._metrics_local.append(metrics)
            else:
                self._metrics_queue.put(metrics)

    @abstractmethod
    def _iter_shard(
        self,
        shard: Shard,
        worker_id: int,
        metrics: WorkerMetrics,
        pbar: Any,
    ) -> Iterator[Any]:
        """Yield converted batches for all splits in shard, updating metrics in-place."""
