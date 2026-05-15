from __future__ import annotations

import logging
import multiprocessing
import queue
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def fmt_bytes(n: int) -> str:
    for unit, threshold in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= threshold:
            return f"{n / threshold:.1f} {unit}"
    return f"{n} B"


def log_split_assignment(shards: list, epoch: int, rank: int, num_ranks: int) -> None:
    """Log split assignment table at INFO and emit WARNING if unbalanced."""
    if not shards:
        return

    total_splits = sum(len(s.splits) for s in shards)
    total_rows = sum(s.row_count or 0 for s in shards)
    total_bytes = sum(s.size_bytes or 0 for s in shards)

    logger.info(
        "Split assignment (epoch=%d, %d workers, rank %d/%d):",
        epoch,
        len(shards),
        rank,
        num_ranks,
    )
    for shard in shards:
        shard_rows = shard.row_count or 0
        shard_bytes = shard.size_bytes or 0
        logger.info(
            "  Worker %d:  %d splits  |  %s rows  |  %s",
            shard.id,
            len(shard.splits),
            f"{shard_rows:,}" if shard_rows else "?",
            fmt_bytes(shard_bytes) if shard_bytes else "? B",
        )
        for sp in shard.splits:
            fname = sp.file.path.rsplit("/", 1)[-1]
            if sp.row_range is not None:
                end = sp.row_range.offset + sp.row_range.length
                rng = f"rows [{sp.row_range.offset:,}, {end:,})"
            else:
                rng = "full file"
            size = fmt_bytes(sp.file.file_size) if sp.file.file_size else "? B"
            logger.info("    %s  %s  %s", fname, rng, size)

    logger.info(
        "  Total: %d splits  |  %s rows  |  %s",
        total_splits,
        f"{total_rows:,}" if total_rows else "?",
        fmt_bytes(total_bytes) if total_bytes else "? B",
    )

    shard_bytes_list = [s.size_bytes for s in shards if (s.size_bytes or 0) > 0]
    if len(shard_bytes_list) >= 2:
        max_bytes = max(shard_bytes_list)
        min_bytes = min(shard_bytes_list)
        if min_bytes and max_bytes / min_bytes > 2.0:
            logger.warning(
                "Unbalanced split assignment — max worker %s, min worker %s "
                "(%.1f× ratio). Consider reducing target_bytes for finer-grained splits.",
                fmt_bytes(max_bytes),
                fmt_bytes(min_bytes),
                max_bytes / min_bytes,
            )


def drain_metrics(
    metrics_local: list,
    metrics_queue: multiprocessing.Queue,
) -> list[WorkerMetrics]:
    """Drain both collection paths and return all WorkerMetrics."""
    results: list[WorkerMetrics] = list(metrics_local)
    metrics_local.clear()
    while True:
        try:
            results.append(metrics_queue.get_nowait())
        except queue.Empty:
            break
    return results


@dataclass
class WorkerMetrics:
    """Per-worker I/O counters accumulated during one epoch.

    bytes_read is an estimate of compressed bytes consumed:
      - whole-file split: file_size
      - row-range split with record_count known: file_size * rows / record_count
      - row-range split with record_count unknown: file_size (upper bound)
    """

    worker_id: int
    rows_read: int = 0
    batches_read: int = 0
    bytes_read: int = 0
    files_read: int = 0
    elapsed_sec: float = 0.0


@dataclass
class EpochSummary:
    """Aggregate metrics across all workers for one epoch."""

    epoch: int
    workers: list[WorkerMetrics] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(m.rows_read for m in self.workers)

    @property
    def total_bytes(self) -> int:
        return sum(m.bytes_read for m in self.workers)

    @property
    def total_batches(self) -> int:
        return sum(m.batches_read for m in self.workers)

    @property
    def elapsed_sec(self) -> float:
        return max((m.elapsed_sec for m in self.workers), default=0.0)

    @property
    def rows_per_sec(self) -> float:
        return self.total_rows / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


def log_epoch_summary(epoch: int, results: list[WorkerMetrics]) -> None:
    """Log epoch summary at INFO level."""
    summary = EpochSummary(epoch=epoch, workers=results)
    logger.info(
        "Epoch %d complete:  workers=%d  rows=%s  bytes_est=%s"
        "  elapsed=%.1fs  rows/s=%.0f",
        summary.epoch,
        len(summary.workers),
        f"{summary.total_rows:,}",
        fmt_bytes(summary.total_bytes),
        summary.elapsed_sec,
        summary.rows_per_sec,
        extra={
            "event": "epoch_done",
            "epoch": summary.epoch,
            "workers": len(summary.workers),
            "total_rows": summary.total_rows,
            "total_bytes": summary.total_bytes,
            "elapsed_sec": summary.elapsed_sec,
            "rows_per_sec": summary.rows_per_sec,
        },
    )
    for m in sorted(results, key=lambda x: x.worker_id):
        logger.info(
            "  Worker %d:  %s rows  %s  %.1fs",
            m.worker_id,
            f"{m.rows_read:,}",
            fmt_bytes(m.bytes_read),
            m.elapsed_sec,
        )

    """Log split assignment table at INFO and emit WARNING if unbalanced."""
    if not shards:
        return

    total_splits = sum(len(s.splits) for s in shards)
    total_rows = sum(s.row_count or 0 for s in shards)
    total_bytes = sum(s.size_bytes or 0 for s in shards)

    logger.info(
        "Split assignment (epoch=%d, %d workers, rank %d/%d):",
        epoch,
        len(shards),
        rank,
        num_ranks,
    )
    for shard in shards:
        shard_rows = shard.row_count or 0
        shard_bytes = shard.size_bytes or 0
        logger.info(
            "  Worker %d:  %d splits  |  %s rows  |  %s",
            shard.id,
            len(shard.splits),
            f"{shard_rows:,}" if shard_rows else "?",
            fmt_bytes(shard_bytes) if shard_bytes else "? B",
        )
        for sp in shard.splits:
            fname = sp.file.path.rsplit("/", 1)[-1]
            if sp.row_range is not None:
                end = sp.row_range.offset + sp.row_range.length
                rng = f"rows [{sp.row_range.offset:,}, {end:,})"
            else:
                rng = "full file"
            size = fmt_bytes(sp.file.file_size) if sp.file.file_size else "? B"
            logger.info("    %s  %s  %s", fname, rng, size)

    logger.info(
        "  Total: %d splits  |  %s rows  |  %s",
        total_splits,
        f"{total_rows:,}" if total_rows else "?",
        fmt_bytes(total_bytes) if total_bytes else "? B",
    )

    shard_bytes_list = [s.size_bytes for s in shards if (s.size_bytes or 0) > 0]
    if len(shard_bytes_list) >= 2:
        max_bytes = max(shard_bytes_list)
        min_bytes = min(shard_bytes_list)
        if min_bytes and max_bytes / min_bytes > 2.0:
            logger.warning(
                "Unbalanced split assignment — max worker %s, min worker %s "
                "(%.1f× ratio). Consider reducing target_bytes for finer-grained splits.",
                fmt_bytes(max_bytes),
                fmt_bytes(min_bytes),
                max_bytes / min_bytes,
            )


def make_metrics_queue() -> tuple[list, multiprocessing.Queue]:
    """Return (metrics_local, metrics_queue) for a new dataset instance."""
    return [], multiprocessing.Queue()


def push_metrics(
    metrics_local: list,
    metrics_queue: multiprocessing.Queue,
    metrics: WorkerMetrics,
    is_main_process: bool,
) -> None:
    """Push WorkerMetrics to the right collection based on process context."""
    if is_main_process:
        metrics_local.append(metrics)
    else:
        metrics_queue.put(metrics)


def drain_metrics(
    metrics_local: list,
    metrics_queue: multiprocessing.Queue,
) -> list[WorkerMetrics]:
    """Drain both collection paths and return all WorkerMetrics."""
    results: list[WorkerMetrics] = list(metrics_local)
    metrics_local.clear()
    while not metrics_queue.empty():
        try:
            results.append(metrics_queue.get_nowait())
        except Exception:
            break
    return results


@dataclass
class WorkerMetrics:
    """Per-worker I/O counters accumulated during one epoch.

    bytes_read is an estimate of compressed bytes consumed:
      - whole-file split: file_size
      - row-range split with record_count known: file_size * rows / record_count
      - row-range split with record_count unknown: file_size (upper bound)
    """

    worker_id: int
    rows_read: int = 0
    batches_read: int = 0
    bytes_read: int = 0
    files_read: int = 0
    elapsed_sec: float = 0.0


@dataclass
class EpochSummary:
    """Aggregate metrics across all workers for one epoch."""

    epoch: int
    workers: list[WorkerMetrics] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(m.rows_read for m in self.workers)

    @property
    def total_bytes(self) -> int:
        return sum(m.bytes_read for m in self.workers)

    @property
    def total_batches(self) -> int:
        return sum(m.batches_read for m in self.workers)

    @property
    def elapsed_sec(self) -> float:
        return max((m.elapsed_sec for m in self.workers), default=0.0)

    @property
    def rows_per_sec(self) -> float:
        return self.total_rows / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


def log_epoch_summary(epoch: int, results: list[WorkerMetrics]) -> None:
    """Log epoch summary at INFO level."""
    summary = EpochSummary(epoch=epoch, workers=results)
    logger.info(
        "Epoch %d complete:  workers=%d  rows=%s  bytes_est=%s"
        "  elapsed=%.1fs  rows/s=%.0f",
        summary.epoch,
        len(summary.workers),
        f"{summary.total_rows:,}",
        fmt_bytes(summary.total_bytes),
        summary.elapsed_sec,
        summary.rows_per_sec,
        extra={
            "event": "epoch_done",
            "epoch": summary.epoch,
            "workers": len(summary.workers),
            "total_rows": summary.total_rows,
            "total_bytes": summary.total_bytes,
            "elapsed_sec": summary.elapsed_sec,
            "rows_per_sec": summary.rows_per_sec,
        },
    )
    for m in sorted(results, key=lambda x: x.worker_id):
        logger.info(
            "  Worker %d:  %s rows  %s  %.1fs",
            m.worker_id,
            f"{m.rows_read:,}",
            fmt_bytes(m.bytes_read),
            m.elapsed_sec,
        )

