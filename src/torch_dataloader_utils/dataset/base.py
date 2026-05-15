from __future__ import annotations

import inspect
import logging
import multiprocessing
import queue
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

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


def _shuffle_buffer_iter(
    source: Iterator[Any],
    buffer_size: int,
    batch_size: int,
    rng: "np.random.Generator",
) -> Iterator[Any]:
    """Reservoir-style streaming shuffle buffer operating on Arrow RecordBatches."""
    import numpy as np
    import pyarrow as pa

    buffer: pa.Table | None = None
    source_iter = iter(source)
    exhausted = False

    def _extend(tbl: pa.Table | None, batch: Any) -> pa.Table:
        t = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
        return pa.concat_tables([tbl, t]) if tbl is not None else t

    def _emit_one(tbl: pa.Table) -> tuple[Any, pa.Table]:
        n = min(batch_size, len(tbl))
        idx = np.sort(rng.choice(len(tbl), size=n, replace=False))
        out = tbl.take(idx)
        mask = np.ones(len(tbl), dtype=bool)
        mask[idx] = False
        return out.to_batches()[0], tbl.filter(pa.array(mask))

    # Fill initial buffer
    while not exhausted and (buffer is None or len(buffer) < buffer_size):
        try:
            buffer = _extend(buffer, next(source_iter))
        except StopIteration:
            exhausted = True

    # Stream: emit one batch, refill to buffer_size
    while not exhausted:
        out, buffer = _emit_one(buffer)
        yield out
        while not exhausted and len(buffer) < buffer_size:
            try:
                buffer = _extend(buffer, next(source_iter))
            except StopIteration:
                exhausted = True

    # Drain remainder with full permutation shuffle
    if buffer is not None and len(buffer) > 0:
        perm = rng.permutation(len(buffer))
        for i in range(0, len(buffer), batch_size):
            chunk = np.sort(perm[i : i + batch_size])
            yield buffer.take(chunk).to_batches()[0]


class CheckpointMismatchError(Exception):
    """Raised by load_state_dict when stored shard content doesn't match current splits.

    Indicates that num_workers, shuffle_seed, or the file list changed between
    the checkpoint and the current dataset construction.
    """

    def __init__(
        self,
        shard_state: dict,
        full_state: dict,
        current_num_workers: int,
        current_shuffle_seed: int | None,
    ) -> None:
        splits_desc = "\n".join(
            f"    {s['path']}  rows [{s['row_offset']:,}, {s['row_offset'] + s['row_length']:,})"
            if s["row_offset"] is not None
            else f"    {s['path']}  full file"
            for s in shard_state["splits"]
        )
        saved_nw = full_state.get("_num_workers", "?")
        saved_seed = full_state.get("_shuffle_seed", "?")

        hints = []
        if saved_nw != "?" and saved_nw != current_num_workers:
            hints.append(
                f"num_workers changed: checkpoint={saved_nw}, current={current_num_workers}"
            )
        if saved_seed != "?" and saved_seed != current_shuffle_seed:
            hints.append(
                f"shuffle_seed changed: checkpoint={saved_seed}, current={current_shuffle_seed}"
            )
        if not hints:
            hints.append("file list may have changed since the checkpoint was saved")

        hint_str = "\n  ".join(hints)
        super().__init__(
            f"Checkpoint shard does not match any current split.\n\n"
            f"  Checkpoint shard:\n{splits_desc}\n\n"
            f"  Likely cause: {hint_str}\n\n"
            f"  Reconstruct the dataset with matching parameters or discard this checkpoint."
        )


class BaseDataset(IterableDataset, ABC):
    """Shared infrastructure for all torch-dataloader-utils datasets.

    Provides observability (metrics, progress bars, split logging), epoch reshuffling,
    checkpoint/resume, and the DataLoader __iter__ lifecycle. Subclasses implement
    _iter_shard() only.

    Required: before calling _init_splits_and_observability(), subclasses must set:
        self._files, self._strategy, self._num_workers, self._num_ranks, self._rank,
        self._output_format
    """

    def _init_splits_and_observability(
        self,
        epoch: int = 0,
        show_progress: bool = False,
        progress_interval_sec: float = 120.0,
        shuffle_buffer_size: int = 0,
    ) -> None:
        """Finish initialisation: set up metrics queues and generate the first splits."""
        self._epoch = epoch
        self._show_progress = show_progress
        self._progress_interval_sec = progress_interval_sec
        self._shuffle_buffer_size: int = max(0, shuffle_buffer_size or 0)
        # Two-path metrics collection:
        # - num_workers=0: __iter__ runs in the main process — direct list append.
        # - num_workers>0: workers are separate processes — use mp.Queue.
        self._metrics_local: list[WorkerMetrics] = []
        self._metrics_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._completed_workers: set[int] = set()
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
        self._completed_workers = set()
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

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serialisable checkpoint state capturing completed shards.

        Save alongside model.state_dict(). On resume call load_state_dict()
        before the DataLoader begins iterating — completed shards will be
        skipped with zero I/O.
        """
        self._drain_to_completed()
        completed_shards = []
        for worker_id in sorted(self._completed_workers):
            shard = self._splits[worker_id]
            completed_shards.append(
                {
                    "splits": [
                        {
                            "path": fs.file.path,
                            "row_offset": fs.row_range.offset if fs.row_range else None,
                            "row_length": fs.row_range.length if fs.row_range else None,
                        }
                        for fs in shard.splits
                    ]
                }
            )
        return {
            "epoch": self._epoch,
            "_num_workers": self._num_workers,
            "_shuffle_seed": getattr(self, "_shuffle_seed", None),
            "completed_shards": completed_shards,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore checkpoint state and validate against current splits.

        Raises CheckpointMismatchError if any stored shard cannot be matched
        to the current split assignments — indicating that num_workers,
        shuffle_seed, or the file list changed since the checkpoint was saved.

        Call this instead of set_epoch() for the resumed epoch.
        """
        self.reset_metrics()
        self._completed_workers = set()
        self._epoch = state["epoch"]
        self._splits = self._generate_splits()

        for shard_state in state["completed_shards"]:
            worker_id = self._match_shard(shard_state, state)
            self._completed_workers.add(worker_id)

        logger.info(
            "Resumed from checkpoint: epoch=%d  completed=%d/%d shards  skipped=%s",
            self._epoch,
            len(self._completed_workers),
            len(self._splits),
            sorted(self._completed_workers),
        )

    def _match_shard(self, shard_state: dict, full_state: dict) -> int:
        """Find the worker ID whose current shard content matches shard_state.

        Raises CheckpointMismatchError if no match is found.
        """
        target = shard_state["splits"]
        for shard in self._splits:
            candidate = [
                {
                    "path": fs.file.path,
                    "row_offset": fs.row_range.offset if fs.row_range else None,
                    "row_length": fs.row_range.length if fs.row_range else None,
                }
                for fs in shard.splits
            ]
            if candidate == target:
                return shard.id
        raise CheckpointMismatchError(
            shard_state,
            full_state,
            current_num_workers=self._num_workers,
            current_shuffle_seed=getattr(self, "_shuffle_seed", None),
        )

    def _drain_to_completed(self) -> None:
        """Drain the metrics queue and record finished worker IDs without discarding metrics."""
        for m in self._metrics_local:
            self._completed_workers.add(m.worker_id)
        while True:
            try:
                m = self._metrics_queue.get_nowait()
                self._metrics_local.append(m)  # keep available for get_metrics()
                self._completed_workers.add(m.worker_id)
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Output conversion
    # ------------------------------------------------------------------

    def _convert_output(self, batch: Any) -> Any:
        from torch_dataloader_utils.dataset.output import convert_batch

        return convert_batch(batch, self._output_format)

    # ------------------------------------------------------------------
    # Progress bars
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # DataLoader iteration lifecycle
    # ------------------------------------------------------------------

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

        if worker_id in self._completed_workers:
            logger.debug(
                "Worker %d: shard already completed — skipping (resumed from checkpoint)",
                worker_id,
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
            source = self._iter_shard(shard, worker_id, metrics, pbar)
            if self._shuffle_buffer_size > 0:
                import numpy as np

                rng_seed = self._shuffle_seed * 100_000 + self._epoch * 1_000 + worker_id
                rng = np.random.default_rng(rng_seed)
                source = _shuffle_buffer_iter(source, self._shuffle_buffer_size, self._batch_size, rng)
            for batch in source:
                yield self._convert_output(batch)
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
